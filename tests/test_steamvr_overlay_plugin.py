import importlib.util
import math
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakePluginBase:
    def __init__(self, *_args, **_kwargs):
        self._test_settings = {}
        self._test_enabled = True
        if hasattr(self, "__plugin_init__"):
            self.__plugin_init__()

    def get_plugin_setting(self, name, default=None):
        return self._test_settings.get(name, default)

    def set_plugin_setting(self, name, value):
        self._test_settings[name] = value

    def is_enabled(self, default=False):
        return self._test_enabled

    def init_plugin_settings(self, settings, settings_groups=None):
        self._test_settings_groups = settings_groups
        for name, value in settings.items():
            if name in self._test_settings:
                continue
            if isinstance(value, dict) and "value" in value:
                self._test_settings[name] = value["value"]
            else:
                self._test_settings[name] = value


def _load_plugin_module():
    fake_plugins = types.SimpleNamespace(Base=_FakePluginBase)
    fake_downloader = types.SimpleNamespace(
        download_extract=lambda *_args, **_kwargs: False,
        extract_zip=lambda *_args, **_kwargs: None,
    )
    previous_plugins = sys.modules.get("Plugins")
    previous_downloader = sys.modules.get("downloader")
    sys.modules["Plugins"] = fake_plugins
    sys.modules["downloader"] = fake_downloader
    try:
        spec = importlib.util.spec_from_file_location(
            "steamvr_overlay_plugin_test_module",
            PROJECT_ROOT / "Plugins" / "steamvr_overlay_plugin.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_plugins is None:
            sys.modules.pop("Plugins", None)
        else:
            sys.modules["Plugins"] = previous_plugins
        if previous_downloader is None:
            sys.modules.pop("downloader", None)
        else:
            sys.modules["downloader"] = previous_downloader


PLUGIN_MODULE = _load_plugin_module()


def test_settings_groups_use_a_uniform_shape_supported_by_the_ui():
    plugin = PLUGIN_MODULE.SteamVROverlayPlugin()
    plugin._test_enabled = False
    plugin.init()

    for group in plugin._test_settings_groups.values():
        assert isinstance(group, list)
        if not group:
            continue
        if isinstance(group[0], str):
            assert all(isinstance(item, str) for item in group)
        else:
            assert all(
                isinstance(column, list)
                and all(isinstance(item, str) for item in column)
                for column in group
            )


class _FakeMatrix34:
    def __init__(self):
        self.rows = [[0.0] * 4 for _ in range(3)]

    def __getitem__(self, index):
        return self.rows[index]


def test_openvr_transform_places_overlay_in_front_of_anchor():
    fake_openvr = types.SimpleNamespace(HmdMatrix34_t=_FakeMatrix34)
    matrix = PLUGIN_MODULE.make_openvr_transform(
        fake_openvr,
        x=0.2,
        y=-0.35,
        distance=1.2,
        pitch=0,
        yaw=0,
        roll=0,
    )

    assert matrix.rows[0][:3] == [1.0, 0.0, 0.0]
    assert matrix.rows[1][:3] == [0.0, 1.0, 0.0]
    assert matrix.rows[2][:3] == [0.0, 0.0, 1.0]
    assert math.isclose(matrix.rows[0][3], 0.2)
    assert math.isclose(matrix.rows[1][3], -0.35)
    assert math.isclose(matrix.rows[2][3], -1.2)


def test_overlay_renderer_produces_fixed_rgba_texture():
    image = PLUGIN_MODULE.render_overlay_image(
        "Whispering Tiger\n日本語 / Deutsch / Español",
        width=640,
        height=256,
        font_size=34,
        background_opacity=0.65,
    )

    assert image.mode == "RGBA"
    assert image.size == (640, 256)
    alpha_minimum, alpha_maximum = image.getchannel("A").getextrema()
    assert alpha_minimum == 0
    assert alpha_maximum == 255
    assert image.getpixel((320, 128))[3] >= round(0.65 * 255)


def test_source_translation_and_intermediate_text_are_selected_correctly():
    plugin = PLUGIN_MODULE.SteamVROverlayPlugin()
    plugin._test_settings.update(
        {
            "display_mode": "both",
            "translation_separator": "\\n",
            "history_entries": 2,
            "max_characters": 1600,
            "show_intermediate": True,
        }
    )

    plugin.stt(
        "Hallo Welt",
        {"text": "Hello world", "txt_translation": "Hallo Welt"},
    )
    plugin.stt_intermediate(
        "Wie geht es dir",
        {"text": "How are you", "txt_translation": "Wie geht es dir"},
    )

    with plugin._state_lock:
        display = plugin._current_display_text_locked()
    assert display == (
        "Hello world\nHallo Welt\n\nHow are you\nWie geht es dir"
    )

    plugin._test_settings["display_mode"] = "translation"
    with plugin._state_lock:
        display = plugin._current_display_text_locked()
    assert display == "Hallo Welt\n\nWie geht es dir"


def test_history_limit_keeps_newest_final_entries():
    plugin = PLUGIN_MODULE.SteamVROverlayPlugin()
    plugin._test_settings.update(
        {
            "display_mode": "source",
            "history_entries": 2,
            "max_characters": 1600,
        }
    )

    plugin.stt("one", {"text": "one"})
    plugin.stt("two", {"text": "two"})
    plugin.stt("three", {"text": "three"})

    with plugin._state_lock:
        assert list(plugin._history) == [("two", ""), ("three", "")]
        assert plugin._current_display_text_locked() == "two\n\nthree"


def test_fade_alpha_honors_display_and_fade_durations():
    configuration = {
        "overlay_opacity": 0.8,
        "display_duration": 10.0,
        "fade_duration": 2.0,
    }
    fade = PLUGIN_MODULE.SteamVROverlayPlugin._fade_alpha

    assert math.isclose(fade(5.0, configuration), 0.8)
    assert math.isclose(fade(11.0, configuration), 0.4)
    assert math.isclose(fade(12.5, configuration), 0.0)


def test_overlay_connect_and_disconnect_own_openvr_resources():
    calls = []

    class FakeSystem:
        @staticmethod
        def isTrackedDeviceConnected(device_index):
            calls.append(("connected", device_index))
            return True

    class FakeOverlay:
        @staticmethod
        def createOverlay(key, name):
            calls.append(("create", key, name))
            return 73

        @staticmethod
        def setOverlayWidthInMeters(handle, width):
            calls.append(("width", handle, width))

        @staticmethod
        def setOverlayTransformTrackedDeviceRelative(handle, device, matrix):
            calls.append(("transform", handle, device, matrix.rows))

        @staticmethod
        def setOverlayRaw(handle, _buffer, width, height, depth):
            calls.append(("raw", handle, width, height, depth))

        @staticmethod
        def setOverlayAlpha(handle, alpha):
            calls.append(("alpha", handle, alpha))

        @staticmethod
        def showOverlay(handle):
            calls.append(("show", handle))

        @staticmethod
        def hideOverlay(handle):
            calls.append(("hide", handle))

        @staticmethod
        def destroyOverlay(handle):
            calls.append(("destroy", handle))

    fake_system = FakeSystem()
    fake_overlay = FakeOverlay()
    fake_openvr = types.SimpleNamespace(
        HmdMatrix34_t=_FakeMatrix34,
        VRApplication_Background=3,
        k_unTrackedDeviceIndex_Hmd=0,
        k_unTrackedDeviceIndexInvalid=0xFFFFFFFF,
        TrackedControllerRole_LeftHand=1,
        TrackedControllerRole_RightHand=2,
        init=lambda application: (
            calls.append(("init", application)) or fake_system
        ),
        VROverlay=lambda: fake_overlay,
        shutdown=lambda: calls.append(("shutdown",)),
    )

    plugin = PLUGIN_MODULE.SteamVROverlayPlugin()
    plugin._openvr = fake_openvr
    plugin._connect_overlay(fake_openvr)

    assert plugin._overlay_handle == 73
    assert ("init", 3) in calls
    assert any(call[:2] == ("raw", 73) and call[2:] == (1, 1, 4) for call in calls)
    assert ("show", 73) in calls

    plugin._disconnect_overlay(fake_openvr)
    assert ("hide", 73) in calls
    assert ("destroy", 73) in calls
    assert ("shutdown",) in calls
    assert plugin._overlay_handle is None


def test_pinned_openvr_wheel_metadata_is_complete():
    assert PLUGIN_MODULE.OPENVR_VERSION == "2.12.1401"
    assert PLUGIN_MODULE.OPENVR_WHEEL_URLS[0].endswith("-py3-none-any.whl")
    assert len(PLUGIN_MODULE.OPENVR_WHEEL_SHA256) == 64
    int(PLUGIN_MODULE.OPENVR_WHEEL_SHA256, 16)
