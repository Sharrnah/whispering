import importlib.util
import json
import queue
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import audioprocessor
import audio_processing_recording
import audio_routes
from Models.STT.vibevoice_selection import is_streaming_selection
import settings
import streaming_display
from streaming_text import utf16_length


def event(text, revision, final=False, source="main"):
    return dict(text=text, type="transcript", language="", streaming=True, stream_id=source + "-session",
                stream_revision=revision, text_delta=text, final=final, audio_source_id=source)


class VibeVoiceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.displays = streaming_display.StableDisplayScheduler(clock=lambda: self.now, threaded=False)
        patcher = mock.patch.object(streaming_display, "displays", self.displays)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = settings.SettingsManager()
        self.settings.translate_settings.update(
            stt_type="vibevoice_asr", model="VibeVoice-ASR-Streaming-1.5B", realtime=False, txt_translate=False,
            osc_auto_processing_enabled=True, osc_chat_limit=12, osc_ip="127.0.0.1",
            websocket_ip="127.0.0.1", websocket_final_messages=True, tts_answer=False,
        )
        audioprocessor.last_audio_timestamps.clear()

    def test_model_selection_routes_to_the_correct_runtime_and_keeps_legacy_profiles(self):
        for kind, name, streaming in (
            ("vibevoice_asr", "VibeVoice-ASR-HF", False),
            ("vibevoice_asr", "VibeVoice-ASR-Streaming-1.5B", True),
            ("vibevoice_asr", "VibeVoice-ASR-Streaming-7B", True),
            ("vibevoice_asr", "custom-streaming", True),
            ("vibevoice_asr_streaming", "custom", True),
        ):
            with self.subTest(kind=kind, model=name), \
                    mock.patch.object(audioprocessor, "main_settings", self.settings), \
                    mock.patch.object(audioprocessor.vibevoice_asr_streaming, "VibeVoiceStreamingASR") as native, \
                    mock.patch.object(audioprocessor.vibevoice_asr, "TransformerVibeVoiceASR") as original:
                self.settings.translate_settings.update(stt_type=kind, model=name)
                self.assertEqual(is_streaming_selection(kind, name), streaming)
                result = audioprocessor.load_whisper(name, "cpu")
                self.assertIs(result, native.return_value if streaming else original.return_value)
                if streaming:
                    native.return_value.load_model.assert_called_once_with(name)
                    original.assert_not_called()
                else:
                    native.assert_not_called()

    def test_additional_routes_stream_independently_and_respect_output_switches(self):
        class CaptionPlugin:
            supports_stable_streaming = True
            def __init__(self):
                self.stt_caption = mock.Mock(return_value=None)
                self.stt = mock.Mock()
                self.stt_intermediate = mock.Mock()
            def is_enabled(self, *args):
                return True
            def streaming_caption_text(self, text, result):
                return text

        plugin = CaptionPlugin()
        routes = {}
        recorded = queue.Queue()
        pcm = np.full(512, 1000, dtype="<i2").tobytes()
        for source, output in (("game", True), ("music", True), ("private", False)):
            config = audio_routes.normalize_route(dict(
                id=source, name=source.title(), enabled=True, websocket_enabled=output,
                osc_enabled=source == "game", plugins=["CaptionPlugin"] if output else [],
                whisper_task="translate", realtime=False,
            ), 0, self.settings)
            route = audio_routes.AudioRoute(config, self.settings, [plugin], recorded)
            self.assertEqual(route.settings.GetOption("whisper_task"), "transcribe")
            recorder = audio_processing_recording.AudioProcessor(
                default_sample_rate=16000, recorded_sample_rate=16000, input_channel_num=1,
                plugins=route.plugins, settings=route.settings, audio_queue=recorded,
                source_id=source, source_name=config["name"], enable_mic_passthrough=False,
            )
            self.addCleanup(recorder.close)
            recorder._queue_streaming_snapshot(pcm, False)
            recorder._queue_audio(b"edited-final", True)
            routes[source] = recorder
        self.assertEqual(recorded.qsize(), 3)
        model = mock.Mock()
        def recognize(audio, *, source_id, stream_id, final, **kwargs):
            return iter([{**event(source_id + " words", 2 if final else 1, final, source_id), "stream_id": stream_id}])
        model.process_snapshot.side_effect = recognize
        def delivered(*args, **kwargs):
            receipt = kwargs["receipt"]
            receipt.sent, receipt.sent_at = True, self.now
            receipt.done.set()
        with mock.patch.object(audioprocessor.websocket, "BroadcastMessage") as broadcast, \
                mock.patch.object(audioprocessor.VRC_OSCLib, "Chat", side_effect=delivered) as osc, \
                mock.patch.object(audioprocessor.Utilities, "add_transcription"), \
                mock.patch.object(audioprocessor.main_settings, "SetOption"):
            while not recorded.empty():
                audioprocessor.process_vibevoice_snapshot(model, recorded.get_nowait())
            self.displays.tick()
            captions = [json.loads(c.args[0]) for c in broadcast.call_args_list if json.loads(c.args[0])["type"] == "streaming_caption"]
            self.assertEqual({c["audio_source_id"] for c in captions}, {"game", "music"})
            self.assertTrue(all(c["audio_source_name"] == c["audio_source_id"].title() for c in captions))
            self.assertEqual({c.args[1]["audio_source_id"] for c in plugin.stt_caption.call_args_list}, {"game", "music"})
            self.assertEqual([c.args[0] for c in osc.call_args_list], ["game words"])
            routes["game"]._queue_streaming_snapshot(pcm, True)
            audioprocessor.process_vibevoice_snapshot(model, recorded.get_nowait())
            self.displays.tick()  # Deliver the final block, starting its reading deadline.
            self.now = 10
            self.displays.tick()
            self.assertNotIn(("websocket", "game"), self.displays.lanes)
            self.assertIn(("websocket", "music"), self.displays.lanes)
            self.assertIn(("plugin", id(plugin), "music"), self.displays.lanes)
            self.assertTrue(plugin.stt_caption.call_args.args[1]["display_done"])
            self.assertEqual(plugin.stt_caption.call_args.args[1]["audio_source_id"], "game")

    def test_source_display_and_chat_limit_override_or_inherit_main_settings(self):
        self.settings.translate_settings.update(streaming_display_mode="rolling", osc_chat_limit=144)
        config = audio_routes.normalize_route(dict(id="game", streaming_display_mode="blocks", osc_chat_limit=72), 0, self.settings)
        route = audio_routes.RouteSettings(self.settings, config)
        snapshot = route.snapshot()
        self.assertEqual(streaming_display.display_options(snapshot)["mode"], "blocks")
        self.assertEqual(snapshot.GetOption("osc_chat_limit"), 72)
        self.assertEqual(self.settings.GetOption("osc_chat_limit"), 144)
        config.update(streaming_display_mode="", osc_chat_limit=None)
        route.update(config)
        self.assertEqual(streaming_display.display_options(route)["mode"], "rolling")
        self.assertEqual(route.GetOption("osc_chat_limit"), 144)
        self.settings.translate_settings.update(streaming_display_mode="blocks", osc_chat_limit=120)
        self.assertEqual(streaming_display.display_options(route)["mode"], "blocks")
        self.assertEqual(route.GetOption("osc_chat_limit"), 120)
        self.assertEqual(snapshot.GetOption("osc_chat_limit"), 72)
        with self.assertRaisesRegex(ValueError, "display mode"):
            audio_routes.normalize_route(dict(id="game", streaming_display_mode="invalid"), 0, self.settings)

    def test_streams_through_outputs_and_saves_only_one_final(self):
        self.settings.translate_settings["special_settings"] = {"stt_vibevoice_streaming": {"osc_mode": "rolling"}}
        plugin = mock.Mock()
        plugin.is_enabled.return_value = True
        model = mock.Mock()
        model.process_snapshot.return_value = iter([event("old words", 1), event("old words new words", 2),
                                                     event("old words new words", 3, True)])
        pcm = np.zeros(100, dtype="<i2").tobytes()
        with mock.patch.object(audioprocessor.VRC_OSCLib, "Chat") as osc, \
                mock.patch.object(audioprocessor.websocket, "BroadcastMessage") as websocket, \
                mock.patch.object(audioprocessor.Utilities, "add_transcription") as save, \
                mock.patch.object(audioprocessor.main_settings, "SetOption"):
            audioprocessor.process_vibevoice_snapshot(model, dict(
                data=b"unused", streaming_pcm=pcm, stream_id="main-session", final=True, time=10,
                settings=self.settings, plugins=[plugin], source_id="main",
            ))
        self.assertEqual(plugin.stt_intermediate.call_count, 2)
        self.assertEqual(plugin.stt.call_count, 1)
        self.assertEqual(save.call_count, 1)
        self.assertEqual([call.args[0] for call in osc.call_args_list], ["old words", "new words", "new words"])
        self.assertTrue(all(call.kwargs["replaceable"] for call in osc.call_args_list))
        messages = [json.loads(call.args[0]) for call in websocket.call_args_list]
        self.assertEqual([x["stream_revision"] for x in messages if x.get("streaming")], [1, 2, 3])
        np.testing.assert_array_equal(model.process_snapshot.call_args.args[0], np.zeros(100, dtype=np.float32))

    def test_live_default_shows_newest_view_and_saves_complete_history(self):
        text = "One two three four five six seven eight nine ten"
        def delivered(*args, **kwargs):
            receipt = kwargs["receipt"]
            receipt.sent, receipt.sent_at = True, self.now
            receipt.done.set()
        with mock.patch.object(audioprocessor.VRC_OSCLib, "Chat", side_effect=delivered) as osc, \
                mock.patch.object(audioprocessor.websocket, "BroadcastMessage") as broadcast, \
                mock.patch.object(audioprocessor.Utilities, "add_transcription") as save:
            audioprocessor.whisper_result_handling(event(text, 1, True), 10, True, self.settings, [])
            save.assert_called_once()
            self.assertEqual(osc.call_count, 0)
            for step in range(150):
                self.now = step / 4
                self.displays.tick()
        pages = [call.args[0] for call in osc.call_args_list]
        self.assertEqual(pages, ["nine ten"])
        self.assertEqual(save.call_args.args[2], text)
        self.assertTrue(all(utf16_length(page) <= 12 for page in pages))
        self.assertTrue(all(call.kwargs["replaceable"] for call in osc.call_args_list))
        messages = [json.loads(call.args[0]) for call in broadcast.call_args_list]
        self.assertEqual(sum(m["type"] == "transcript" for m in messages), 1)
        captions = [m for m in messages if m["type"] == "streaming_caption"]
        self.assertTrue(captions[-1]["display_done"])
        self.assertFalse(self.displays.lanes)

    def test_native_stream_uses_live_translation_when_sync_enabled(self):
        self.settings.translate_settings.update(txt_translate=True, txt_translate_realtime_sync=True)
        with mock.patch.object(audioprocessor.texttranslate, "TranslateLanguage", return_value=("Neue Wörter", "en", "de")) as translate, \
                mock.patch.object(audioprocessor, "send_message") as send:
            audioprocessor.whisper_result_handling(event("New words", 1), 10, False, self.settings, [])
        translate.assert_called_once()
        self.assertEqual(send.call_args.args[0], "Neue Wörter")

    def test_empty_final_clears_draft_without_saving_blank_history(self):
        with mock.patch.object(audioprocessor, "send_message") as send, \
                mock.patch.object(audioprocessor.Utilities, "add_transcription") as save:
            audioprocessor.whisper_result_handling(event("", 1, True), 10, True, self.settings, [])
        send.assert_called_once()
        save.assert_not_called()

    def test_disabled_websocket_finals_still_close_the_live_preview(self):
        self.settings.translate_settings.update(websocket_final_messages=False, osc_ip="0")
        with mock.patch.object(audioprocessor.websocket, "BroadcastMessage") as broadcast:
            audioprocessor.send_message("complete text", event("complete text", 2, True), True, self.settings, [])
        payload = json.loads(broadcast.call_args.args[0])
        self.assertEqual(payload["type"], "processing_data")
        self.assertEqual(payload["data"], "")
        self.assertTrue(payload["final"])
        self.assertEqual(payload["stream_revision"], 2)

    def test_capture_preserves_raw_pcm_and_marks_utterance_identity(self):
        recorder = object.__new__(audio_processing_recording.AudioProcessor)
        recorder.settings = self.settings
        recorder.plugins = []
        recorder.audio_queue = queue.Queue()
        recorder.source_id, recorder.source_name = "mic", "Microphone"
        recorder.start_time = 42.0
        recorder.default_sample_rate = 16000
        pcm = np.arange(50, dtype="<i2").tobytes()
        recorder._queue_streaming_snapshot(pcm, False)
        recorder._queue_audio(b"silence-cut-final", True)
        recorder._queue_streaming_snapshot(pcm, True)
        self.assertEqual(recorder.audio_queue.qsize(), 2)
        draft, final = recorder.audio_queue.get(), recorder.audio_queue.get()
        self.assertEqual(draft["stream_id"], final["stream_id"])
        self.assertEqual(final["streaming_pcm"], pcm)
        self.assertTrue(final["final"])
        recorder.start_time = 43.0
        recorder._queue_streaming_snapshot(pcm, False)
        self.assertNotEqual(draft["stream_id"], recorder.audio_queue.get()["stream_id"])

    def test_capture_callback_keeps_one_stream_while_pause_timestamp_changes(self):
        self.settings.translate_settings.update(
            energy=100, pause=0.5, vad_confidence_threshold=0.5,
            normalize_enabled=False, silence_cutting_enabled=False,
            denoise_audio="", speaker_diarization=False, vad_on_full_clip=True,
        )
        recorded = queue.Queue()
        recorder = audio_processing_recording.AudioProcessor(
            default_sample_rate=16000, recorded_sample_rate=16000, input_channel_num=1,
            plugins=[], settings=self.settings, audio_queue=recorded, enable_mic_passthrough=False,
        )
        speech = np.full(512, 1000, dtype=np.int16).tobytes()
        silence = bytes(len(speech))
        try:
            # Silence must close the utterance even without a VAD model. Each
            # speech callback changes start_time, but keeps the same stream ID.
            with mock.patch.object(audio_processing_recording.time, "time", return_value=100.0) as clock:
                recorder.start_time = 100.0
                recorder.intermediate_time_start = 99.0
                recorder.callback(speech, 512, None, None)
                clock.return_value = 100.3
                recorder.callback(speech, 512, None, None)
                clock.return_value = 101.0
                recorder.callback(silence, 512, None, None)
                items = [recorded.get_nowait() for _ in range(recorded.qsize())]
                self.assertEqual([item["final"] for item in items], [False, False, True])
                self.assertEqual(len({item["stream_id"] for item in items}), 1)
                self.assertEqual(items[-1]["streaming_pcm"], speech * 2)
                self.assertFalse(recorder.frames)
                self.assertFalse(recorder.start_rec_on_volume_threshold)
                clock.return_value = 101.3
                recorder.callback(speech, 512, None, None)
                # Switching capture devices also flushes a pending utterance.
                recorder.reset_for_audio_input_switch()
                next_items = [recorded.get_nowait() for _ in range(recorded.qsize())]
                self.assertTrue(next_items[-1]["final"])
                self.assertNotEqual(next_items[-1]["stream_id"], items[-1]["stream_id"])
                self.assertFalse(recorder.frames)
        finally:
            recorder.close()


class CaptionPluginTests(unittest.TestCase):
    @staticmethod
    def plugin_module(name):
        # Reuse the existing isolated loader; no desktop or SteamVR is opened.
        path = Path(__file__).parent / ("test_" + name + ".py")
        spec = importlib.util.spec_from_file_location("streaming_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PLUGIN_MODULE

    def test_subtitle_stream_updates_one_draft_even_without_extra_line(self):
        plugin = self.plugin_module("subtitles_display_plugin").SubtitleDisplayPlugin()
        plugin.transcriptions = []
        plugin.transcription_times = {}
        plugin.update_intermediate_label_text = mock.Mock()
        plugin._test_settings["extra_intermediate_line"] = False
        for result in [event("hello", 1), event("hello world", 2), event("hello world", 3, True), event("hello", 1)]:
            plugin.update_label(result["text"], result, is_final=result["final"])
        self.assertEqual(plugin.transcriptions, ["hello world"])
        self.assertFalse(plugin._stream_intermediates)

    def test_steamvr_does_not_clear_another_sources_draft(self):
        plugin = self.plugin_module("steamvr_overlay_plugin").SteamVROverlayPlugin()
        plugin.stt_intermediate("mic", event("mic", 1))
        plugin.stt_intermediate("game", event("game", 1, source="game"))
        plugin.stt("mic final", event("mic final", 2, True))
        plugin.stt_intermediate("stale", event("stale", 1))
        self.assertEqual(list(plugin._history), [("mic final", "")])
        self.assertEqual(plugin._stream_intermediates, {"game": ("game", "")})
        self.assertIn("game", plugin._current_display_text_locked())
        plugin.stt("", event("", 2, True, source="game"))
        self.assertFalse(plugin._stream_intermediates)

    def test_stable_subtitles_replace_one_caption_and_do_not_replay_final_history(self):
        plugin = self.plugin_module("subtitles_display_plugin").SubtitleDisplayPlugin()
        plugin.is_enabled = mock.Mock(return_value=True)
        plugin.transcriptions = []
        plugin.transcription_times = {}
        plugin.update_intermediate_label_text = mock.Mock()
        result = {**event("full source text", 2, True), "display_mode": "blocks"}
        plugin.stt("full source text", result)
        self.assertEqual(plugin.transcriptions, [])
        plugin.stt_caption("First line\nSecond line", {**result, "display_done": False})
        self.assertEqual(plugin._stable_captions["main"], "First line\nSecond line")
        plugin.stt_caption("Next phrase\n", {**result, "display_done": False})
        self.assertEqual(list(plugin._stable_captions.values()), ["Next phrase\n"])
        plugin.stt_caption("", {**result, "display_done": True})
        self.assertFalse(plugin._stable_captions)

    def test_stable_steamvr_keeps_other_sources_and_clears_after_reading(self):
        plugin = self.plugin_module("steamvr_overlay_plugin").SteamVROverlayPlugin()
        result = {**event("full source text", 2, True), "display_mode": "blocks"}
        plugin.stt("full source text", result)
        self.assertFalse(plugin._history)
        plugin.stt_caption("Microphone line\n", {**result, "display_done": False})
        plugin.stt_caption("Game line\n", {**result, "audio_source_id": "game", "display_done": False})
        plugin.stt_caption("", {**result, "display_done": True})
        self.assertNotIn("Microphone", plugin._current_display_text_locked())
        self.assertIn("Game line", plugin._current_display_text_locked())
        plugin.clear_overlay()
        self.assertFalse(plugin._stable_captions)


if __name__ == "__main__":
    unittest.main()
