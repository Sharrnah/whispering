import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakePluginBase:
    def __init__(self, *_args, **_kwargs):
        self._test_settings = {}

    def get_plugin_setting(self, name, default=None):
        return self._test_settings.get(name, default)


def _load_plugin_module():
    previous_plugins = sys.modules.get("Plugins")
    sys.modules["Plugins"] = types.SimpleNamespace(Base=_FakePluginBase)
    try:
        spec = importlib.util.spec_from_file_location(
            "subtitles_display_plugin_test_module",
            PROJECT_ROOT / "Plugins" / "subtitles_display_plugin.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_plugins is None:
            sys.modules.pop("Plugins", None)
        else:
            sys.modules["Plugins"] = previous_plugins


PLUGIN_MODULE = _load_plugin_module()


class SubtitleDisplayRouteTests(unittest.TestCase):
    def test_additional_route_uses_its_translated_pipeline_text(self):
        plugin = PLUGIN_MODULE.SubtitleDisplayPlugin()
        plugin._test_settings["transcription_display_source_transcript"] = True
        result = {
            "text": "Hello world",
            "txt_translation": "Bonjour le monde",
            "audio_source_id": "game-chat",
        }

        self.assertEqual(
            plugin._select_display_text("Bonjour le monde", result),
            "Bonjour le monde",
        )

    def test_main_microphone_preserves_the_plugin_source_preference(self):
        plugin = PLUGIN_MODULE.SubtitleDisplayPlugin()
        result = {
            "text": "Hello world",
            "txt_translation": "Bonjour le monde",
            "audio_source_id": "main",
        }

        plugin._test_settings["transcription_display_source_transcript"] = True
        self.assertEqual(
            plugin._select_display_text("Bonjour le monde", result),
            "Hello world",
        )

        plugin._test_settings["transcription_display_source_transcript"] = False
        self.assertEqual(
            plugin._select_display_text("Bonjour le monde", result),
            "Bonjour le monde",
        )


if __name__ == "__main__":
    unittest.main()
