"""Native additional audio capture routes sharing the primary STT service."""

import copy
import re
import threading

import audio_processing_recording
import audio_tools
import VRC_OSCLib
from Models.STS import SmartTurn, VAD


SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = audio_tools.pyaudio.paInt16
MAX_ADDITIONAL_ROUTES = 8
ROUTE_EXCLUDED_PLUGINS = {"SecondaryProfilePlugin"}

_ROUTE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_UNSET = object()
_DENOISE_MODES = {"", "noise_reduce", "deepfilter"}
_audio_enhancer_cache = {}
_audio_enhancer_cache_lock = threading.RLock()


def _send_route_typing_indicator(osc_ip, osc_port, _send_websocket=True):
    """Publish only VRChat typing activity for an additional audio route."""
    if osc_ip and osc_ip != "0":
        VRC_OSCLib.Bool(
            True, "/chatbox/typing", IP=osc_ip, PORT=osc_port
        )


class _SynchronizedAudioEnhancer:
    """Serialize access to one shared denoiser across capture routes."""

    def __init__(self, enhancer):
        self._enhancer = enhancer
        self._lock = threading.RLock()

    def enhance_audio(self, *args, **kwargs):
        with self._lock:
            return self._enhancer.enhance_audio(*args, **kwargs)


def get_audio_enhancer(mode, post_filter=False):
    """Load one process-wide denoiser per type and return a locked facade."""
    normalized_mode = str(mode or "").strip().casefold()
    if normalized_mode not in _DENOISE_MODES:
        raise ValueError(f"Unsupported noise filter: {mode!r}.")
    if not normalized_mode:
        return None

    with _audio_enhancer_cache_lock:
        enhancer = _audio_enhancer_cache.get(normalized_mode)
        if enhancer is not None:
            return enhancer

        if normalized_mode == "deepfilter":
            from Models.STS import DeepFilterNet
            loaded = DeepFilterNet.DeepFilterNet(post_filter=bool(post_filter))
        else:
            from Models.STS import Noisereduce
            loaded = Noisereduce.Noisereduce()
        enhancer = _SynchronizedAudioEnhancer(loaded)
        _audio_enhancer_cache[normalized_mode] = enhancer
        return enhancer


def select_plugins(all_plugins, configured_names):
    """Return an allowlisted set while preserving legacy None == all."""
    if configured_names is None:
        return list(all_plugins or ())
    requested = {str(name) for name in configured_names}
    return [
        plugin for plugin in (all_plugins or ())
        if plugin.__class__.__name__ in requested
    ]


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
        raise ValueError(f"Expected a boolean value, got {value!r}.")
    return bool(value)


def _bounded_float(value, default, minimum, maximum, field_name):
    try:
        result = float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if result < minimum or result > maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}."
        )
    return result


def _bounded_int(value, default, minimum, maximum, field_name):
    try:
        result = int(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a whole number.") from exc
    if result < minimum or result > maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}."
        )
    return result


def _base_default(base_settings, name, fallback):
    value = base_settings.GetOption(name)
    return fallback if value is None else value


def _route_default(route, base_settings, name, fallback):
    if name in route and route[name] is not None:
        return route[name]
    return _base_default(base_settings, name, fallback)


def _denoise_mode(value):
    mode = str(value or "").strip().casefold()
    if mode not in _DENOISE_MODES:
        raise ValueError(
            "denoise_audio must be disabled, noise_reduce, or deepfilter."
        )
    return mode


def _plugin_names(value, field_name):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of plugin names.")
    result = []
    seen = set()
    for item in value:
        name = str(item or "").strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def normalize_main_audio_plugins(value):
    if value is None:
        return None
    return _plugin_names(value, "main_audio_plugins")


def normalize_route(route, index, base_settings):
    if not isinstance(route, dict):
        raise ValueError(f"Audio route {index + 1} must be an object.")

    route_id = str(route.get("id") or f"audio-route-{index + 1}").strip()
    if not _ROUTE_ID_PATTERN.fullmatch(route_id):
        raise ValueError(
            "Audio route IDs must start with a letter or number and contain "
            "only letters, numbers, dots, underscores, or dashes."
        )
    name = str(route.get("name") or f"Audio source {index + 1}").strip()
    if not name:
        raise ValueError(f"Audio route {route_id!r} needs a name.")
    if len(name) > 100:
        raise ValueError(f"Audio route {route_id!r} has a name longer than 100 characters.")

    audio_api = str(route.get("audio_api") or base_settings.GetOption("audio_api") or "").strip()
    if not audio_api:
        raise ValueError(f"Audio route {name!r} needs an audio API.")

    process_id = _bounded_int(
        route.get("audio_input_process_id"), 0, 0, 2 ** 31 - 1,
        "audio_input_process_id"
    )
    device_index = route.get("device_index", -1)
    if device_index not in (None, ""):
        device_index = _bounded_int(
            device_index, -1, -1, 2 ** 31 - 1, "device_index"
        )

    normalized = {
        "id": route_id,
        "name": name,
        "enabled": _as_bool(route.get("enabled"), True),
        "audio_api": audio_api,
        "audio_input_device": str(route.get("audio_input_device") or "Default").strip(),
        "audio_input_process": str(route.get("audio_input_process") or "").strip(),
        "audio_input_process_id": process_id,
        "device_index": device_index,
        "stt_enabled": _as_bool(route.get("stt_enabled"), True),
        "current_language": route.get(
            "current_language", base_settings.GetOption("current_language")
        ),
        "whisper_task": str(
            route.get("whisper_task")
            or base_settings.GetOption("whisper_task")
            or "transcribe"
        ),
        "energy": _bounded_int(
            route.get("energy"), _base_default(base_settings, "energy", 300),
            0, 32767, "energy"
        ),
        "vad_confidence_threshold": _bounded_float(
            route.get("vad_confidence_threshold"),
            _base_default(base_settings, "vad_confidence_threshold", 0.4),
            0.0, 1.0, "vad_confidence_threshold"
        ),
        "phrase_time_limit": _bounded_float(
            route.get("phrase_time_limit"),
            _base_default(base_settings, "phrase_time_limit", 0),
            0.0, 600.0, "phrase_time_limit"
        ),
        "pause": _bounded_float(
            route.get("pause"), _base_default(base_settings, "pause", 1.0),
            0.0, 30.0, "pause"
        ),
        "realtime": _as_bool(route.get("realtime"), False),
        "realtime_frequency_time": _bounded_float(
            route.get("realtime_frequency_time"),
            _base_default(base_settings, "realtime_frequency_time", 1.0),
            0.0, 60.0, "realtime_frequency_time"
        ),
        "silence_cutting_enabled": _as_bool(
            _route_default(
                route, base_settings, "silence_cutting_enabled", True
            )
        ),
        "denoise_audio": _denoise_mode(
            _route_default(route, base_settings, "denoise_audio", "")
        ),
        "denoise_strength": _bounded_float(
            route.get("denoise_strength"),
            _base_default(base_settings, "denoise_strength", 1.0),
            0.0, 1.0, "denoise_strength"
        ),
        "vad_smart_turn_enabled": _as_bool(
            _route_default(
                route, base_settings, "vad_smart_turn_enabled", False
            )
        ),
        "vad_smart_turn_min_length": _bounded_float(
            route.get("vad_smart_turn_min_length"),
            _base_default(base_settings, "vad_smart_turn_min_length", 2.0),
            0.0, 30.0, "vad_smart_turn_min_length"
        ),
        "vad_smart_turn_probability_threshold": _bounded_float(
            route.get("vad_smart_turn_probability_threshold"),
            _base_default(
                base_settings, "vad_smart_turn_probability_threshold", 0.5
            ),
            0.0, 1.0, "vad_smart_turn_probability_threshold"
        ),
        "vad_smart_turn_pause_length": _bounded_float(
            route.get("vad_smart_turn_pause_length"),
            _base_default(
                base_settings, "vad_smart_turn_pause_length", 0.5
            ),
            0.0, 30.0, "vad_smart_turn_pause_length"
        ),
        "txt_translate": _as_bool(route.get("txt_translate"), False),
        "src_lang": str(
            route.get("src_lang") or base_settings.GetOption("src_lang") or "auto"
        ),
        "trg_lang": str(
            route.get("trg_lang") or base_settings.GetOption("trg_lang") or "eng_Latn"
        ),
        "txt_romaji": _as_bool(route.get("txt_romaji"), False),
        "websocket_enabled": _as_bool(route.get("websocket_enabled"), True),
        "osc_enabled": _as_bool(route.get("osc_enabled"), False),
        "osc_typing_indicator": _as_bool(
            route.get("osc_typing_indicator"), False
        ),
        "osc_chat_notification": _as_bool(
            route.get("osc_chat_notification"), False
        ),
        "osc_chat_prefix": str(
            _route_default(route, base_settings, "osc_chat_prefix", "") or ""
        ),
        "plugins": _plugin_names(route.get("plugins"), "plugins"),
    }
    return normalized


def normalize_routes(routes, base_settings):
    if routes is None:
        routes = []
    if not isinstance(routes, (list, tuple)):
        raise ValueError("additional_audio_routes must be a list.")
    if len(routes) > MAX_ADDITIONAL_ROUTES:
        raise ValueError(
            f"At most {MAX_ADDITIONAL_ROUTES} additional audio routes are supported."
        )

    normalized = []
    route_ids = set()
    for index, route in enumerate(routes):
        item = normalize_route(route, index, base_settings)
        if item["id"] in route_ids:
            raise ValueError(f"Audio route ID {item['id']!r} is used more than once.")
        route_ids.add(item["id"])
        normalized.append(item)
    return normalized


class FrozenRouteSettings:
    def __init__(self, base_settings, overrides):
        self._base_settings = base_settings
        self._overrides = overrides

    def GetOption(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return self._base_settings.GetOption(name)

    def get_option(self, name):
        return self.GetOption(name)

    def get_all_settings(self):
        values = copy.deepcopy(self._base_settings.get_all_settings())
        values.update(copy.deepcopy(self._overrides))
        return values


class RouteSettings:
    """Read-through settings with only route-safe overrides."""

    _DIRECT_OVERRIDES = (
        "stt_enabled",
        "current_language",
        "whisper_task",
        "energy",
        "vad_confidence_threshold",
        "phrase_time_limit",
        "pause",
        "realtime",
        "realtime_frequency_time",
        "silence_cutting_enabled",
        "denoise_audio",
        "denoise_strength",
        "vad_smart_turn_enabled",
        "vad_smart_turn_min_length",
        "vad_smart_turn_probability_threshold",
        "vad_smart_turn_pause_length",
        "txt_translate",
        "src_lang",
        "trg_lang",
        "txt_romaji",
        "osc_typing_indicator",
        "osc_chat_notification",
        "osc_chat_prefix",
    )

    def __init__(self, base_settings, route):
        self._base_settings = base_settings
        self._lock = threading.RLock()
        self._route = copy.deepcopy(route)

    def update(self, route):
        with self._lock:
            self._route = copy.deepcopy(route)

    def _overrides(self):
        route = self._route
        values = {key: route[key] for key in self._DIRECT_OVERRIDES}
        values.update({
            # A capture route can publish transcripts and use selected plugins,
            # but it must never speak, pass audio through, or save into the
            # microphone route's files implicitly.
            "tts_answer": False,
            "mic_passthrough_routing": False,
            "push_to_talk_key": "",
            "audio_processor_caller": "",
            "speaker_diarization": False,
            "txt_second_translation_enabled": False,
            "transcription_auto_save_file": "",
            "transcription_save_audio_dir": "",
            "osc_auto_processing_enabled": route["osc_enabled"],
            "websocket_final_messages": route["websocket_enabled"],
        })
        if not route["osc_enabled"]:
            values["osc_ip"] = "0"
        if not route["websocket_enabled"]:
            values["websocket_ip"] = "0"
        return values

    def GetOption(self, name):
        with self._lock:
            overrides = self._overrides()
            if name in overrides:
                return overrides[name]
        return self._base_settings.GetOption(name)

    def get_option(self, name):
        return self.GetOption(name)

    def snapshot(self):
        with self._lock:
            overrides = copy.deepcopy(self._overrides())
        return FrozenRouteSettings(self._base_settings, overrides)


class AudioRoute:
    def __init__(
            self, config, base_settings, all_plugins, audio_queue,
            audio_enhancer_resolver=None):
        self.config = copy.deepcopy(config)
        self.settings = RouteSettings(base_settings, self.config)
        self.all_plugins = list(all_plugins or ())
        self.plugins = select_plugins(self.all_plugins, self.config["plugins"])
        self.audio_queue = audio_queue
        self.audio_enhancer_resolver = (
            audio_enhancer_resolver or get_audio_enhancer
        )
        self.vad_model = None
        self.turn_model = None
        self.processor = None
        self.controller = None

    def _report_unavailable_plugins(self):
        loaded = {plugin.__class__.__name__: plugin for plugin in self.plugins}
        all_loaded_names = {
            plugin.__class__.__name__ for plugin in self.all_plugins
        }
        for plugin_name in self.config["plugins"]:
            if plugin_name not in all_loaded_names:
                print(
                    f"Audio route {self.config['name']!r}: plugin {plugin_name!r} "
                    "is not loaded and will not receive this source."
                )
                continue
            plugin = loaded.get(plugin_name)
            if plugin is not None and hasattr(plugin, "is_enabled") \
                    and not plugin.is_enabled(False):
                print(
                    f"Audio route {self.config['name']!r}: plugin {plugin_name!r} "
                    "is globally disabled and will not receive this source."
                )

    def start(self):
        if not self.config["enabled"]:
            return self.config

        self._report_unavailable_plugins()

        try:
            vad_thread_num = int(float(self.settings.GetOption("vad_thread_num") or 1))
        except (TypeError, ValueError):
            vad_thread_num = 1
        self.vad_model = VAD.VAD(vad_thread_num)
        if not self.vad_model.is_loaded():
            raise RuntimeError("The voice activity detector could not be loaded.")

        frames_per_buffer = int(self.settings.GetOption("vad_frames_per_buffer") or 512)
        if frames_per_buffer not in (256, 512):
            frames_per_buffer = 512
        self.vad_model.set_vad_frames_per_buffer(frames_per_buffer)

        if self.settings.GetOption("vad_smart_turn_enabled"):
            self.turn_model = SmartTurn.SmartTurn(
                min_audio_length=self.settings.GetOption("vad_smart_turn_min_length") or 2
            )

        audio_enhancer = self.audio_enhancer_resolver(
            self.settings.GetOption("denoise_audio"),
            post_filter=self.settings.GetOption("denoise_audio_post_filter"),
        )
        typing_indicator_function = None
        if self.settings.GetOption("osc_auto_processing_enabled") \
                and self.settings.GetOption("osc_typing_indicator"):
            typing_indicator_function = _send_route_typing_indicator
        self.processor = audio_processing_recording.AudioProcessor(
            default_sample_rate=SAMPLE_RATE,
            start_rec_on_volume_threshold=False,
            push_to_talk_key=None,
            keyboard_rec_force_stop=False,
            vad_model=self.vad_model,
            turn_model=self.turn_model,
            plugins=self.plugins,
            audio_enhancer=audio_enhancer,
            osc_ip=self.settings.GetOption("osc_ip"),
            osc_port=self.settings.GetOption("osc_port"),
            chunk=frames_per_buffer,
            channels=CHANNELS,
            sample_format=FORMAT,
            audio_queue=self.audio_queue,
            settings=self.settings,
            typing_indicator_function=typing_indicator_function,
            source_id=self.config["id"],
            source_name=self.config["name"],
            enable_mic_passthrough=False,
            verbose=bool(self.settings.GetOption("verbose")),
        )
        self.controller = audio_tools.AudioInputStreamController(
            sample_format=FORMAT,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            chunk=frames_per_buffer,
            py_audio=audio_tools.main_app_py_audio,
            audio_processor=self.processor,
        )
        selected = self.controller.start(self.config)
        for key in (
            "audio_api", "audio_input_device", "audio_input_process",
            "audio_input_process_id", "device_index"
        ):
            self.config[key] = selected[key]
        self.settings.update(self.config)
        print(
            f"Started audio route {self.config['name']!r} from "
            f"{self.config['audio_input_device']!r}."
        )
        return copy.deepcopy(self.config)

    def close(self):
        close_error = None
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception as exc:
                close_error = exc
            finally:
                self.controller = None
        if self.processor is not None:
            try:
                self.processor.close()
            except Exception as exc:
                if close_error is None:
                    close_error = exc
            self.processor = None
        self.turn_model = None
        self.vad_model = None
        if close_error is not None:
            raise close_error


class AudioRouteManager:
    def __init__(
            self, base_settings, all_plugins, audio_queue,
            audio_enhancer_resolver=None):
        self.base_settings = base_settings
        self.all_plugins = all_plugins
        self.audio_queue = audio_queue
        self.audio_enhancer_resolver = (
            audio_enhancer_resolver or get_audio_enhancer
        )
        self._lock = threading.RLock()
        self._routes = {}
        self._configs = []
        self._main_audio_plugins = None
        self._main_processor = None

    def set_main_processor(self, processor):
        with self._lock:
            self._main_processor = processor
            processor.plugins = self.main_plugins()

    def main_plugins(self):
        return select_plugins(self.all_plugins, self._main_audio_plugins)

    def _config_index(self, route_id):
        for index, config in enumerate(self._configs):
            if config["id"] == route_id:
                return index
        return None

    def _start_single_route(self, config):
        route = AudioRoute(
            copy.deepcopy(config), self.base_settings, self.all_plugins,
            self.audio_queue, self.audio_enhancer_resolver
        )
        try:
            route.start()
        except BaseException:
            try:
                route.close()
            except Exception as close_error:
                print(f"Could not clean up failed audio route: {close_error}")
            raise
        return route, copy.deepcopy(route.config)

    def _discard_queued_route_audio(self, route_id):
        discard_source = getattr(self.audio_queue, "discard_source", None)
        if callable(discard_source):
            discard_source(route_id)

    def upsert_route(self, route_config):
        """Add or replace one route without interrupting unrelated streams."""
        if not isinstance(route_config, dict):
            raise ValueError("The audio route must be an object.")

        with self._lock:
            requested_id = str(route_config.get("id") or "").strip()
            if not requested_id:
                raise ValueError("The audio route needs a stable ID.")
            existing_index = self._config_index(requested_id)
            normalize_index = (
                existing_index if existing_index is not None else len(self._configs)
            )
            normalized = normalize_route(
                copy.deepcopy(route_config), normalize_index, self.base_settings
            )
            existing_index = self._config_index(normalized["id"])
            if existing_index is None and len(self._configs) >= MAX_ADDITIONAL_ROUTES:
                raise ValueError(
                    f"At most {MAX_ADDITIONAL_ROUTES} additional audio routes are supported."
                )

            previous_config = (
                copy.deepcopy(self._configs[existing_index])
                if existing_index is not None else None
            )
            previous_runtime = self._routes.get(normalized["id"])
            if (
                previous_config == normalized
                and (not normalized["enabled"] or previous_runtime is not None)
            ):
                return self.configuration()

            if previous_config is not None:
                self._discard_queued_route_audio(normalized["id"])

            if previous_runtime is not None:
                self._routes.pop(normalized["id"], None)
                try:
                    previous_runtime.close()
                except Exception as close_error:
                    print(
                        f"Could not close audio route {normalized['name']!r} "
                        f"before updating it: {close_error}"
                    )

            candidate = None
            selected_config = copy.deepcopy(normalized)
            try:
                if normalized["enabled"]:
                    candidate, selected_config = self._start_single_route(normalized)
            except BaseException as update_error:
                if previous_runtime is not None and previous_config["enabled"]:
                    try:
                        restored, restored_config = self._start_single_route(
                            previous_config
                        )
                        self._routes[previous_config["id"]] = restored
                        self._configs[existing_index] = restored_config
                    except BaseException as restore_error:
                        raise RuntimeError(
                            "Could not update the audio route, and its previous "
                            f"stream could not be restored: {restore_error}"
                        ) from update_error
                    raise RuntimeError(
                        "Could not update the audio route. Its previous stream "
                        f"was restored: {update_error}"
                    ) from update_error
                raise RuntimeError(
                    f"Could not start the audio route: {update_error}"
                ) from update_error

            if existing_index is None:
                self._configs.append(selected_config)
            else:
                self._configs[existing_index] = selected_config
            if candidate is not None:
                self._routes[selected_config["id"]] = candidate
            return self.configuration()

    def set_route_enabled(self, route_id, enabled):
        with self._lock:
            index = self._config_index(str(route_id or "").strip())
            if index is None:
                raise ValueError(f"Unknown audio route: {route_id!r}.")
            updated = copy.deepcopy(self._configs[index])
            updated["enabled"] = _as_bool(enabled)
            return self.upsert_route(updated)

    def delete_route(self, route_id):
        with self._lock:
            normalized_id = str(route_id or "").strip()
            index = self._config_index(normalized_id)
            if index is None:
                raise ValueError(f"Unknown audio route: {route_id!r}.")
            runtime = self._routes.pop(normalized_id, None)
            self._discard_queued_route_audio(normalized_id)
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as close_error:
                    print(
                        f"Could not completely close audio route "
                        f"{runtime.config.get('name')!r}: {close_error}"
                    )
            del self._configs[index]
            return self.configuration()

    def update_plugin_routing(self, route_plugins, main_audio_plugins):
        """Apply plugin allowlists live without reopening capture devices."""
        if not isinstance(route_plugins, dict):
            raise ValueError("route_plugins must be an object keyed by route ID.")
        normalized_plugins = {
            str(route_id): _plugin_names(names, "plugins")
            for route_id, names in route_plugins.items()
        }
        normalized_main = normalize_main_audio_plugins(main_audio_plugins)

        with self._lock:
            configured_ids = {config["id"] for config in self._configs}
            unknown_ids = set(normalized_plugins) - configured_ids
            if unknown_ids:
                unknown = sorted(unknown_ids)[0]
                raise ValueError(f"Unknown audio route: {unknown!r}.")

            for index, config in enumerate(self._configs):
                if config["id"] not in normalized_plugins:
                    continue
                updated = copy.deepcopy(config)
                updated["plugins"] = normalized_plugins[config["id"]]
                self._configs[index] = updated

                runtime = self._routes.get(config["id"])
                if runtime is None:
                    continue
                runtime.config = copy.deepcopy(updated)
                runtime.settings.update(updated)
                runtime.plugins = select_plugins(
                    runtime.all_plugins, updated["plugins"]
                )
                if runtime.processor is not None:
                    runtime.processor.plugins = runtime.plugins

            self._main_audio_plugins = normalized_main
            if self._main_processor is not None:
                self._main_processor.plugins = self.main_plugins()
            return self.configuration()

    def enable_plugins_for_main(self, plugin_names):
        """Add newly enabled plugins to an explicit microphone allowlist."""
        requested = _plugin_names(plugin_names, "plugins")
        with self._lock:
            # None is the legacy all-plugin mode, so enabling a plugin already
            # makes it active without converting the profile to an allowlist.
            if self._main_audio_plugins is None:
                return False, None

            known_plugins = {
                plugin.__class__.__name__ for plugin in (self.all_plugins or ())
            }
            updated = list(self._main_audio_plugins)
            changed = False
            for plugin_name in requested:
                if (
                    plugin_name in known_plugins
                    and plugin_name not in ROUTE_EXCLUDED_PLUGINS
                    and plugin_name not in updated
                ):
                    updated.append(plugin_name)
                    changed = True

            if changed:
                self._main_audio_plugins = updated
                if self._main_processor is not None:
                    self._main_processor.plugins = self.main_plugins()
            return changed, copy.deepcopy(self._main_audio_plugins)

    def _close_routes(self, routes):
        for route in routes.values():
            try:
                route.close()
            except Exception as exc:
                print(f"Could not close audio route {route.config.get('name')!r}: {exc}")

    def _start_candidates(self, configs):
        candidates = {}
        started_configs = []
        try:
            for config in configs:
                config = copy.deepcopy(config)
                if config["enabled"]:
                    route = AudioRoute(
                        config, self.base_settings, self.all_plugins,
                        self.audio_queue, self.audio_enhancer_resolver
                    )
                    try:
                        route.start()
                    except BaseException:
                        try:
                            route.close()
                        except Exception as close_error:
                            print(f"Could not clean up failed audio route: {close_error}")
                        raise
                    candidates[config["id"]] = route
                    config = copy.deepcopy(route.config)
                started_configs.append(config)
        except BaseException:
            self._close_routes(candidates)
            raise
        return candidates, started_configs

    def start_from_settings(self):
        routes = normalize_routes(
            self.base_settings.GetOption("additional_audio_routes"),
            self.base_settings,
        )
        main_plugins = normalize_main_audio_plugins(
            self.base_settings.GetOption("main_audio_plugins")
        )
        with self._lock:
            self._main_audio_plugins = main_plugins
            self._configs = copy.deepcopy(routes)
            for config in routes:
                if not config["enabled"]:
                    continue
                route = None
                try:
                    route = AudioRoute(
                        config, self.base_settings, self.all_plugins,
                        self.audio_queue, self.audio_enhancer_resolver
                    )
                    route.start()
                    self._routes[config["id"]] = route
                    config.update(route.config)
                except Exception as exc:
                    if route is not None:
                        try:
                            route.close()
                        except Exception as close_error:
                            print(f"Could not clean up failed audio route: {close_error}")
                    print(
                        f"Could not start audio route {config['name']!r}: {exc}. "
                        "It remains configured and can be retried from the UI."
                    )
            self._configs = copy.deepcopy(routes)
            setter = getattr(self.base_settings, "SetOption", None)
            if callable(setter):
                setter("additional_audio_routes", copy.deepcopy(self._configs))
                setter("main_audio_plugins", copy.deepcopy(self._main_audio_plugins))
            return self.configuration()

    def apply(self, routes, main_audio_plugins=_UNSET):
        normalized_routes = normalize_routes(routes, self.base_settings)
        normalized_main_plugins = (
            self._main_audio_plugins
            if main_audio_plugins is _UNSET
            else normalize_main_audio_plugins(main_audio_plugins)
        )

        with self._lock:
            previous_routes = self._routes
            previous_configs = copy.deepcopy(self._configs)
            self._routes = {}
            self._close_routes(previous_routes)

            try:
                candidates, started_configs = self._start_candidates(normalized_routes)
            except BaseException as apply_error:
                try:
                    restored_routes, restored_configs = self._start_candidates(previous_configs)
                    self._routes = restored_routes
                    self._configs = restored_configs
                except BaseException as restore_error:
                    self._routes = {}
                    self._configs = previous_configs
                    raise RuntimeError(
                        "Could not apply the audio routes, and one or more previous "
                        f"routes could not be restored: {restore_error}"
                    ) from apply_error
                raise RuntimeError(
                    f"Could not apply the audio routes. Previous routes were restored: {apply_error}"
                ) from apply_error

            self._routes = candidates
            self._configs = started_configs
            self._main_audio_plugins = normalized_main_plugins
            if self._main_processor is not None:
                self._main_processor.plugins = self.main_plugins()
            return self.configuration()

    def configuration(self):
        return {
            "routes": copy.deepcopy(self._configs),
            "main_audio_plugins": copy.deepcopy(self._main_audio_plugins),
        }

    def close(self):
        with self._lock:
            routes = self._routes
            self._routes = {}
            self._close_routes(routes)


_manager = None
_manager_lock = threading.RLock()


def set_audio_route_manager(manager):
    global _manager
    with _manager_lock:
        _manager = manager


def get_audio_route_manager():
    with _manager_lock:
        return _manager


def apply_audio_routes(routes, main_audio_plugins=_UNSET):
    manager = get_audio_route_manager()
    if manager is None:
        raise RuntimeError(
            "Additional audio routes are not available until backend startup is complete."
        )
    return manager.apply(routes, main_audio_plugins)


def enable_plugins_for_main(plugin_names):
    manager = get_audio_route_manager()
    if manager is None:
        return False, None
    return manager.enable_plugins_for_main(plugin_names)


def upsert_audio_route(route_config):
    manager = get_audio_route_manager()
    if manager is None:
        raise RuntimeError(
            "Additional audio routes are not available until backend startup is complete."
        )
    return manager.upsert_route(route_config)


def set_audio_route_enabled(route_id, enabled):
    manager = get_audio_route_manager()
    if manager is None:
        raise RuntimeError(
            "Additional audio routes are not available until backend startup is complete."
        )
    return manager.set_route_enabled(route_id, enabled)


def delete_audio_route(route_id):
    manager = get_audio_route_manager()
    if manager is None:
        raise RuntimeError(
            "Additional audio routes are not available until backend startup is complete."
        )
    return manager.delete_route(route_id)


def update_audio_route_plugin_routing(route_plugins, main_audio_plugins):
    manager = get_audio_route_manager()
    if manager is None:
        raise RuntimeError(
            "Additional audio routes are not available until backend startup is complete."
        )
    return manager.update_plugin_routing(route_plugins, main_audio_plugins)


def close_audio_routes():
    manager = get_audio_route_manager()
    if manager is not None:
        manager.close()
