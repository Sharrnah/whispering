import copy
import unittest
from unittest import mock

import audio_routes
import audio_processing_recording
import audioprocessor


class _Settings:
    def __init__(self, **overrides):
        self.values = {
            "audio_api": "WASAPI",
            "current_language": "en",
            "whisper_task": "transcribe",
            "energy": 300,
            "vad_confidence_threshold": 0.4,
            "phrase_time_limit": 0,
            "pause": 1.0,
            "realtime_frequency_time": 1.0,
            "silence_cutting_enabled": True,
            "denoise_audio": "",
            "denoise_strength": 1.0,
            "denoise_audio_post_filter": False,
            "vad_smart_turn_enabled": False,
            "vad_smart_turn_min_length": 2.0,
            "vad_smart_turn_probability_threshold": 0.5,
            "vad_smart_turn_pause_length": 0.5,
            "max_sentence_repetition": -1,
            "verbose": False,
            "transcription_auto_save_continuous_text": False,
            "src_lang": "auto",
            "trg_lang": "deu_Latn",
            "model": "large-v3",
            "stt_type": "faster_whisper",
            "whisper_precision": "float16",
            "ai_device": "cuda",
            "osc_ip": "127.0.0.1",
            "osc_chat_prefix": "[main] ",
            "websocket_ip": "127.0.0.1",
            "plugins": {},
            "additional_audio_routes": [],
            "main_audio_plugins": None,
        }
        self.values.update(overrides)

    def GetOption(self, name):
        return self.values.get(name)

    def get_all_settings(self):
        return self.values

    def SetOption(self, name, value):
        self.values[name] = value
        return value


class _PluginA:
    pass


class _PluginB:
    pass


def _route(**overrides):
    route = {
        "id": "game",
        "name": "Game audio",
        "enabled": True,
        "audio_api": "WASAPI",
        "audio_input_device": "game.exe (PID 42)",
        "audio_input_process": "game.exe",
        "audio_input_process_id": 42,
        "plugins": ["_PluginB"],
    }
    route.update(overrides)
    return route


class AudioRouteSettingsTests(unittest.TestCase):
    def test_transcript_only_plugins_do_not_schedule_realtime_audio_work(self):
        class _RealtimePlugin:
            def realtime_sts(self, *_args, **_kwargs):
                pass

        class _DisabledRealtimePlugin(_RealtimePlugin):
            def is_enabled(self, _default):
                return False

        self.assertFalse(
            audio_processing_recording.has_realtime_sts_plugin([_PluginA()])
        )
        self.assertTrue(
            audio_processing_recording.has_realtime_sts_plugin([_RealtimePlugin()])
        )
        self.assertFalse(
            audio_processing_recording.has_realtime_sts_plugin([_DisabledRealtimePlugin()])
        )

    def test_route_can_override_language_without_overriding_model(self):
        base = _Settings()
        normalized = audio_routes.normalize_route(
            _route(current_language="ja", whisper_task="translate"), 0, base
        )
        settings = audio_routes.RouteSettings(base, normalized)

        self.assertEqual(settings.GetOption("current_language"), "ja")
        self.assertEqual(settings.GetOption("whisper_task"), "translate")
        self.assertEqual(settings.GetOption("model"), "large-v3")
        self.assertEqual(settings.GetOption("stt_type"), "faster_whisper")
        self.assertEqual(settings.GetOption("whisper_precision"), "float16")

    def test_explicit_auto_language_does_not_inherit_the_main_language(self):
        base = _Settings(current_language="de")
        normalized = audio_routes.normalize_route(
            _route(current_language=""), 0, base
        )

        self.assertEqual(normalized["current_language"], "")
        self.assertEqual(
            audio_routes.RouteSettings(base, normalized).GetOption(
                "current_language"
            ),
            "",
        )

    def test_route_outputs_are_safe_by_default(self):
        base = _Settings()
        normalized = audio_routes.normalize_route(_route(), 0, base)
        settings = audio_routes.RouteSettings(base, normalized)

        self.assertFalse(settings.GetOption("tts_answer"))
        self.assertFalse(settings.GetOption("mic_passthrough_routing"))
        self.assertEqual(settings.GetOption("osc_ip"), "0")
        self.assertEqual(settings.GetOption("websocket_ip"), "127.0.0.1")
        self.assertFalse(settings.GetOption("osc_typing_indicator"))
        self.assertFalse(settings.GetOption("osc_chat_notification"))
        self.assertEqual(settings.GetOption("osc_chat_prefix"), "[main] ")

    def test_route_can_override_or_clear_the_main_osc_prefix(self):
        base = _Settings(osc_chat_prefix="[main] ")

        inherited = audio_routes.normalize_route(_route(), 0, base)
        custom = audio_routes.normalize_route(
            _route(osc_chat_prefix="[game] "), 0, base
        )
        cleared = audio_routes.normalize_route(
            _route(osc_chat_prefix=""), 0, base
        )

        self.assertEqual(inherited["osc_chat_prefix"], "[main] ")
        self.assertEqual(
            audio_routes.RouteSettings(base, custom).GetOption("osc_chat_prefix"),
            "[game] ",
        )
        self.assertEqual(
            audio_routes.RouteSettings(base, cleared).GetOption("osc_chat_prefix"),
            "",
        )

    def test_snapshot_is_not_changed_by_a_later_route_update(self):
        base = _Settings()
        first = audio_routes.normalize_route(_route(current_language="en"), 0, base)
        settings = audio_routes.RouteSettings(base, first)
        snapshot = settings.snapshot()

        second = copy.deepcopy(first)
        second["current_language"] = "fr"
        settings.update(second)

        self.assertEqual(snapshot.GetOption("current_language"), "en")
        self.assertEqual(settings.GetOption("current_language"), "fr")

    def test_route_overrides_audio_processing_without_changing_shared_model(self):
        base = _Settings(
            realtime_frequency_time=1.0,
            silence_cutting_enabled=True,
            denoise_audio="",
            vad_smart_turn_enabled=False,
        )
        normalized = audio_routes.normalize_route(
            _route(
                realtime_frequency_time=0.4,
                silence_cutting_enabled=False,
                denoise_audio="noise_reduce",
                denoise_strength=0.65,
                vad_smart_turn_enabled=True,
                vad_smart_turn_min_length=1.5,
                vad_smart_turn_probability_threshold=0.7,
                vad_smart_turn_pause_length=0.25,
            ),
            0,
            base,
        )
        settings = audio_routes.RouteSettings(base, normalized)

        self.assertEqual(settings.GetOption("realtime_frequency_time"), 0.4)
        self.assertFalse(settings.GetOption("silence_cutting_enabled"))
        self.assertEqual(settings.GetOption("denoise_audio"), "noise_reduce")
        self.assertEqual(settings.GetOption("denoise_strength"), 0.65)
        self.assertTrue(settings.GetOption("vad_smart_turn_enabled"))
        self.assertEqual(settings.GetOption("vad_smart_turn_min_length"), 1.5)
        self.assertEqual(
            settings.GetOption("vad_smart_turn_probability_threshold"), 0.7
        )
        self.assertEqual(settings.GetOption("vad_smart_turn_pause_length"), 0.25)
        self.assertEqual(settings.GetOption("model"), "large-v3")

    def test_plugin_none_means_legacy_all_and_empty_means_none(self):
        plugins = [_PluginA(), _PluginB()]

        self.assertEqual(audio_routes.select_plugins(plugins, None), plugins)
        self.assertEqual(audio_routes.select_plugins(plugins, []), [])
        self.assertEqual(
            audio_routes.select_plugins(plugins, ["_PluginB"]),
            [plugins[1]],
        )

    def test_route_translation_reaches_the_plugin_dispatch_message(self):
        base = _Settings()
        normalized = audio_routes.normalize_route(
            _route(txt_translate=True, src_lang="auto", trg_lang="fr"),
            0,
            base,
        )
        route_settings = audio_routes.RouteSettings(base, normalized).snapshot()
        result = {"text": "Hello world", "language": "en"}

        with mock.patch.object(
            audioprocessor.texttranslate,
            "TranslateLanguage",
            return_value=("Bonjour le monde", "en", "fr"),
        ), mock.patch.object(
            audioprocessor.Utilities, "add_transcription"
        ), mock.patch.object(
            audioprocessor, "send_message"
        ) as send_message:
            audioprocessor.whisper_result_handling(
                result,
                audio_timestamp=1,
                final_audio=True,
                settings=route_settings,
                plugins=[_PluginB()],
            )

        self.assertEqual(result["txt_translation"], "Bonjour le monde")
        self.assertEqual(send_message.call_args.args[0], "Bonjour le monde")
        self.assertIs(send_message.call_args.args[1], result)

    def test_route_osc_output_uses_its_notification_and_the_process_timer(self):
        base = _Settings(
            osc_ip="127.0.0.1",
            osc_address="/chatbox/input",
            osc_port=9000,
            txt_second_translation_enabled=False,
            txt_second_translation_wrap="\\n",
            osc_min_time_between_messages=0,
            initial_prompt="",
            osc_chat_prefix="",
            osc_type_transfer_split="\\n",
            osc_type_transfer="translation",
            osc_send_type="full",
            osc_chat_prioritize_latest=False,
            osc_convert_ascii=False,
        )
        normalized = audio_routes.normalize_route(
            _route(
                osc_enabled=True,
                osc_typing_indicator=False,
                osc_chat_notification=True,
                osc_chat_prefix="[game] ",
                websocket_enabled=False,
            ),
            0,
            base,
        )
        snapshot = audio_routes.RouteSettings(base, normalized).snapshot()

        with mock.patch.object(
            audioprocessor.VRC_OSCLib, "set_min_time_between_messages"
        ), mock.patch.object(
            audioprocessor.VRC_OSCLib, "Chat"
        ) as osc_chat, mock.patch.object(
            audioprocessor.main_settings, "SetOption"
        ) as set_main_option:
            audioprocessor.send_message(
                "Hello",
                {"text": "Hello", "language": "en"},
                final_audio=True,
                settings=snapshot,
                plugins=None,
            )

        osc_chat.assert_called_once()
        self.assertEqual(osc_chat.call_args.args[0], "[game] Hello")
        self.assertTrue(osc_chat.call_args.args[2])
        self.assertFalse(snapshot.GetOption("osc_typing_indicator"))
        self.assertTrue(snapshot.GetOption("osc_chat_notification"))
        set_main_option.assert_called_once_with("plugin_timer_stopped", True)

    def test_main_osc_notification_keeps_the_legacy_combined_setting(self):
        base = _Settings(osc_typing_indicator=True)

        self.assertTrue(audioprocessor._osc_chat_notification_enabled(base))


class AudioRouteManagerTests(unittest.TestCase):
    def test_startup_normalizes_incomplete_route_for_profile_and_ui(self):
        base = _Settings(
            additional_audio_routes=[{
                "id": "game",
                "name": "Game audio",
                "enabled": False,
                "audio_api": "WASAPI",
            }]
        )
        manager = audio_routes.AudioRouteManager(base, [], object())

        manager.start_from_settings()

        saved_route = base.values["additional_audio_routes"][0]
        self.assertTrue(saved_route["stt_enabled"])
        self.assertTrue(saved_route["websocket_enabled"])
        self.assertEqual(saved_route["plugins"], [])
        self.assertEqual(saved_route["realtime_frequency_time"], 1.0)
        self.assertTrue(saved_route["silence_cutting_enabled"])
        self.assertEqual(saved_route["denoise_audio"], "")
        self.assertEqual(saved_route["denoise_strength"], 1.0)
        self.assertFalse(saved_route["vad_smart_turn_enabled"])
        self.assertEqual(saved_route["vad_smart_turn_min_length"], 2.0)
        self.assertFalse(saved_route["osc_typing_indicator"])
        self.assertFalse(saved_route["osc_chat_notification"])
        self.assertEqual(saved_route["osc_chat_prefix"], "[main] ")

    def test_enabling_plugin_adds_it_to_explicit_main_route_live(self):
        base = _Settings(main_audio_plugins=[])
        plugins = [_PluginA(), _PluginB()]
        processor = mock.Mock()
        manager = audio_routes.AudioRouteManager(base, plugins, object())
        manager.start_from_settings()
        manager.set_main_processor(processor)

        changed, selected = manager.enable_plugins_for_main(["_PluginB"])

        self.assertTrue(changed)
        self.assertEqual(selected, ["_PluginB"])
        self.assertEqual(manager.configuration()["main_audio_plugins"], ["_PluginB"])
        self.assertEqual(processor.plugins, [plugins[1]])

    def test_enabling_plugin_keeps_legacy_all_plugin_mode(self):
        plugins = [_PluginA(), _PluginB()]
        manager = audio_routes.AudioRouteManager(_Settings(), plugins, object())
        manager.start_from_settings()

        changed, selected = manager.enable_plugins_for_main(["_PluginB"])

        self.assertFalse(changed)
        self.assertIsNone(selected)
        self.assertEqual(manager.main_plugins(), plugins)

    def test_secondary_profile_compatibility_plugin_is_not_routed(self):
        class SecondaryProfilePlugin:
            pass

        manager = audio_routes.AudioRouteManager(
            _Settings(main_audio_plugins=[]), [SecondaryProfilePlugin()], object()
        )
        manager.start_from_settings()

        changed, selected = manager.enable_plugins_for_main(
            ["SecondaryProfilePlugin"]
        )

        self.assertFalse(changed)
        self.assertEqual(selected, [])

    def test_failed_apply_restores_previous_routes(self):
        base = _Settings()
        created = []

        class FakeRoute:
            def __init__(self, config, *_args):
                self.config = copy.deepcopy(config)
                self.closed = False
                created.append(self)

            def start(self):
                if self.config["audio_input_process"] == "missing.exe":
                    raise OSError("application is not running")
                return self.config

            def close(self):
                self.closed = True

        manager = audio_routes.AudioRouteManager(base, [], object())
        with mock.patch.object(audio_routes, "AudioRoute", FakeRoute):
            manager.apply([_route()], [])
            with self.assertRaisesRegex(RuntimeError, "Previous routes were restored"):
                manager.apply(
                    [_route(audio_input_process="missing.exe")],
                    ["_PluginA"],
                )

        configuration = manager.configuration()
        self.assertEqual(configuration["routes"][0]["audio_input_process"], "game.exe")
        self.assertEqual(configuration["main_audio_plugins"], [])
        self.assertTrue(created[0].closed)
        self.assertGreaterEqual(len(created), 3)

    def test_route_toggle_does_not_restart_an_unrelated_stream(self):
        base = _Settings()
        audio_queue = mock.Mock()

        class FakeRoute:
            def __init__(self, config, *_args):
                self.config = copy.deepcopy(config)
                self.closed = False

            def start(self):
                return self.config

            def close(self):
                self.closed = True

        manager = audio_routes.AudioRouteManager(base, [], audio_queue)
        with mock.patch.object(audio_routes, "AudioRoute", FakeRoute):
            manager.apply([
                _route(enabled=False),
                _route(
                    id="discord",
                    name="Discord",
                    audio_input_process="discord.exe",
                    audio_input_process_id=84,
                ),
            ], [])
            discord_runtime = manager._routes["discord"]

            manager.set_route_enabled("game", True)
            game_runtime = manager._routes["game"]

            self.assertIs(manager._routes["discord"], discord_runtime)
            self.assertFalse(discord_runtime.closed)
            self.assertTrue(manager.configuration()["routes"][0]["enabled"])

            manager.set_route_enabled("game", False)

        self.assertTrue(game_runtime.closed)
        self.assertIs(manager._routes["discord"], discord_runtime)
        self.assertFalse(discord_runtime.closed)
        self.assertFalse(manager.configuration()["routes"][0]["enabled"])
        self.assertEqual(
            audio_queue.discard_source.call_args_list,
            [mock.call("game"), mock.call("game")],
        )

    def test_failed_route_edit_restores_only_that_route(self):
        base = _Settings()

        class FakeRoute:
            def __init__(self, config, *_args):
                self.config = copy.deepcopy(config)
                self.closed = False

            def start(self):
                if self.config["audio_input_process"] == "missing.exe":
                    raise OSError("application is not running")
                return self.config

            def close(self):
                self.closed = True

        manager = audio_routes.AudioRouteManager(base, [], object())
        with mock.patch.object(audio_routes, "AudioRoute", FakeRoute):
            manager.apply([
                _route(),
                _route(
                    id="discord",
                    name="Discord",
                    audio_input_process="discord.exe",
                    audio_input_process_id=84,
                ),
            ], [])
            original_game = manager._routes["game"]
            original_discord = manager._routes["discord"]

            with self.assertRaisesRegex(RuntimeError, "previous stream was restored"):
                manager.upsert_route(_route(audio_input_process="missing.exe"))

        self.assertTrue(original_game.closed)
        self.assertIs(manager._routes["discord"], original_discord)
        self.assertFalse(original_discord.closed)
        self.assertEqual(
            manager.configuration()["routes"][0]["audio_input_process"],
            "game.exe",
        )

    def test_plugin_routing_updates_live_processors_without_restarting(self):
        plugins = [_PluginA(), _PluginB()]
        manager = audio_routes.AudioRouteManager(_Settings(), plugins, object())
        route_settings = mock.Mock()
        route_processor = mock.Mock()
        runtime = mock.Mock()
        runtime.config = audio_routes.normalize_route(_route(), 0, manager.base_settings)
        runtime.settings = route_settings
        runtime.all_plugins = plugins
        runtime.processor = route_processor
        runtime._report_unavailable_plugins = mock.Mock()
        main_processor = mock.Mock()
        manager._configs = [copy.deepcopy(runtime.config)]
        manager._routes = {"game": runtime}
        manager._main_audio_plugins = []
        manager._main_processor = main_processor

        selected = manager.update_plugin_routing(
            {"game": ["_PluginA"]}, ["_PluginB"]
        )

        self.assertEqual(selected["routes"][0]["plugins"], ["_PluginA"])
        self.assertEqual(runtime.plugins, [plugins[0]])
        self.assertEqual(route_processor.plugins, [plugins[0]])
        route_settings.update.assert_called_once()
        self.assertEqual(main_processor.plugins, [plugins[1]])


class AudioRouteLifecycleTests(unittest.TestCase):
    def test_route_opens_application_capture_with_source_and_plugin_identity(self):
        base = _Settings(vad_frames_per_buffer=512, vad_thread_num=1)
        plugins = [_PluginA(), _PluginB()]
        normalized = audio_routes.normalize_route(
            _route(osc_enabled=True, osc_typing_indicator=True), 0, base
        )

        vad = mock.Mock()
        vad.is_loaded.return_value = True
        processor = mock.Mock()
        audio_enhancer = object()
        enhancer_resolver = mock.Mock(return_value=audio_enhancer)
        controller = mock.Mock()
        controller.start.return_value = {
            "audio_api": "WASAPI",
            "audio_input_device": "game.exe (PID 42)",
            "audio_input_process": "game.exe",
            "audio_input_process_id": 42,
            "device_index": -1,
        }

        with mock.patch.object(audio_routes.VAD, "VAD", return_value=vad), \
                mock.patch.object(
                    audio_routes.audio_processing_recording,
                    "AudioProcessor",
                    return_value=processor,
                ) as processor_type, \
                mock.patch.object(
                    audio_routes.audio_tools,
                    "AudioInputStreamController",
                    return_value=controller,
                ):
            route = audio_routes.AudioRoute(
                normalized, base, plugins, object(), enhancer_resolver
            )
            route.start()

        processor_options = processor_type.call_args.kwargs
        self.assertEqual(processor_options["source_id"], "game")
        self.assertEqual(processor_options["source_name"], "Game audio")
        self.assertFalse(processor_options["enable_mic_passthrough"])
        self.assertEqual(processor_options["plugins"], [plugins[1]])
        self.assertIs(processor_options["audio_enhancer"], audio_enhancer)
        typing_indicator = processor_options["typing_indicator_function"]
        self.assertTrue(callable(typing_indicator))
        with mock.patch.object(audio_routes.VRC_OSCLib, "Bool") as osc_bool:
            typing_indicator("127.0.0.1", 9000, True)
        osc_bool.assert_called_once_with(
            True, "/chatbox/typing", IP="127.0.0.1", PORT=9000
        )
        enhancer_resolver.assert_called_once_with("", post_filter=False)
        controller.start.assert_called_once()
        opened = controller.start.call_args.args[0]
        self.assertEqual(opened["audio_input_process"], "game.exe")
        self.assertEqual(opened["audio_input_process_id"], 42)


if __name__ == "__main__":
    unittest.main()
