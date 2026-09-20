import base64
import io
import json
import tempfile
import threading
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import Plugins
import settings

from Models import audio_cpp_runtime
from Models.STT import audio_cpp as audio_cpp_stt
from Models.TTS import audio_cpp as audio_cpp_tts


class _Response:
    def __init__(self, *, payload=None, content=b"", lines=()):
        self._payload = payload
        self.content = content
        self._lines = list(lines)
        self.closed = False

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        del decode_unicode
        yield from self._lines

    def close(self):
        self.closed = True


class _Process:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        del timeout
        return 0

    def kill(self):
        self.terminated = True


def _wav_bytes(samples, sample_rate=44100):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


class AudioCppRuntimeTests(unittest.TestCase):
    def test_native_backends_keep_the_selected_adapter_index(self):
        self.assertEqual(
            audio_cpp_runtime.normalize_backend_device("vulkan", 3),
            ("vulkan", 3),
        )
        self.assertEqual(
            audio_cpp_runtime.normalize_backend_device("rocm:2", 0),
            ("hip", 2),
        )
        with self.assertRaisesRegex(ValueError, "DirectML"):
            audio_cpp_runtime.normalize_backend_device("direct-ml:0", 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            audio_cpp_runtime.normalize_backend_device("cuda", -1)

    def test_hip_is_hidden_from_selectors_but_remains_runtime_compatible(self):
        values = settings.SETTINGS.get_available_setting_values()
        self.assertNotIn("hip", values["ai_device"])
        self.assertNotIn("hip", values["tts_ai_device"])
        self.assertEqual(
            audio_cpp_runtime.normalize_backend_device("hip", 2),
            ("hip", 2),
        )

    def test_plugin_activity_probe_uses_plugin_specific_state(self):
        inactive = types.SimpleNamespace(
            on_plugin_tts_after_audio_call=lambda data: data,
            is_plugin_tts_after_audio_active=lambda: False,
        )
        active = types.SimpleNamespace(
            on_plugin_tts_after_audio_call=lambda data: data,
            is_plugin_tts_after_audio_active=lambda: True,
        )
        with mock.patch.object(Plugins, "plugins", [inactive]):
            self.assertFalse(
                Plugins.plugin_custom_event_has_active_handler(
                    "plugin_tts_after_audio"
                )
            )
        with mock.patch.object(Plugins, "plugins", [inactive, active]):
            self.assertTrue(
                Plugins.plugin_custom_event_has_active_handler(
                    "plugin_tts_after_audio"
                )
            )

    def test_server_config_contains_backend_device_threads_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.write_bytes(b"gguf")
            server_path = Path(directory) / "audiocpp_server.exe"
            server_path.touch()
            process = _Process()
            health = types.SimpleNamespace(ok=True)

            with mock.patch.object(
                audio_cpp_runtime, "ensure_runtime", return_value=server_path
            ):
                with mock.patch.object(
                    audio_cpp_runtime, "_free_loopback_port", return_value=19091
                ):
                    with mock.patch.object(
                        audio_cpp_runtime, "_write_json_atomic"
                    ) as write_config:
                        with mock.patch.object(
                            audio_cpp_runtime.processmanager,
                            "run_process",
                            return_value=process,
                        ):
                            with mock.patch.object(
                                audio_cpp_runtime.requests, "get", return_value=health
                            ):
                                server = audio_cpp_runtime.AudioCppServer("unit-stt")
                                server.configure(
                                    backend="vulkan",
                                    device_index=2,
                                    family="qwen3_asr",
                                    model_path=model_path,
                                    task="asr",
                                    mode="offline",
                                    threads=6,
                                    load_options={"adapter": "unit-test"},
                                    session_options={"qwen3_asr.weight_type": "q8_0"},
                                    default_request_options={"max_tokens": 512},
                                )

            config = write_config.call_args.args[1]
            self.assertEqual(config["backend"], "vulkan")
            self.assertEqual(config["device"], 2)
            self.assertEqual(config["threads"], 6)
            self.assertEqual(config["models"][0]["family"], "qwen3_asr")
            self.assertEqual(config["models"][0]["path"], str(model_path.resolve()))
            self.assertEqual(config["models"][0]["load_options"], {"adapter": "unit-test"})
            self.assertEqual(
                config["models"][0]["session_options"],
                {"qwen3_asr.weight_type": "q8_0"},
            )
            server.stop()
            self.assertTrue(process.terminated)


class AudioCppASRTests(unittest.TestCase):
    def test_all_requested_asr_models_are_exposed_without_forced_aligner(self):
        self.assertEqual(
            tuple(audio_cpp_stt.STT_MODELS),
            (
                "Qwen3-ASR-0.6B-GGUF",
                "Qwen3-ASR-1.7B-GGUF",
                "Nemotron-3.5-ASR-Streaming-0.6B-GGUF",
                "VibeVoice-ASR-GGUF",
                "Voxtral-Mini-4B-Realtime-2602-GGUF",
                "Audio8-ASR-0.1B-GGUF",
                "Kroko-ASR-English-64L-GGUF",
                "Canary-180M-Flash-GGUF",
                "Cohere-Transcribe-GGUF",
                "Moonshine-Streaming-Tiny-GGUF",
                "Moonshine-Streaming-Small-GGUF",
                "Moonshine-Streaming-Medium-GGUF",
                "Niagara-19M-Batch-English-GGUF",
                "Niagara-38M-Batch-English-GGUF",
                "MOSS-Transcribe-Diarize-GGUF",
                "VibeVoice-ASR-Streaming-7B-GGUF",
            ),
        )
        self.assertNotIn("qwen3_forced_aligner", audio_cpp_stt.MODEL_ALIASES)
        self.assertEqual(
            audio_cpp_stt.precision_options("Voxtral-Mini-4B-Realtime-2602-GGUF"),
            ("q4_k", "q8_0", "bf16"),
        )
        self.assertEqual(
            audio_cpp_stt.precision_options("Audio8-ASR-0.1B-GGUF"),
            ("q8_0", "f16"),
        )

    def test_every_asr_default_is_routed_as_request_session_or_mode(self):
        for settings_key, defaults in audio_cpp_stt.STT_SETTINGS_DEFAULTS.items():
            surfaced = {
                *audio_cpp_stt.REQUEST_KEYS.get(settings_key, ()),
                *audio_cpp_stt.SESSION_KEYS.get(settings_key, ()),
                "mode",
            }
            self.assertEqual(
                set(defaults) - surfaced,
                set(),
                f"unrouted audio.cpp ASR setting(s) for {settings_key}",
            )

    def test_asr_session_does_not_reinterpret_preconverted_gguf_weights(self):
        options = audio_cpp_stt._session_options(
            "Qwen3-ASR-0.6B-GGUF",
            {
                "weight_type": "bf16",
                "audio_encoder_weight_type": "f16",
                "thinker_weight_type": "q8_0",
                "audio_encoder_graph_arena_mb": 192,
            },
        )
        self.assertEqual(options, {"qwen3_asr.audio_encoder_graph_arena_mb": 192})

    def test_model_and_precision_compatibility_aliases(self):
        self.assertEqual(
            audio_cpp_stt.normalize_model("Qwen3-ASR-0.6B-hf"),
            "Qwen3-ASR-0.6B-GGUF",
        )
        self.assertEqual(audio_cpp_stt.normalize_precision("8bit"), "q8_0")
        self.assertEqual(audio_cpp_stt.normalize_precision("4bit"), "q8_0")
        self.assertEqual(audio_cpp_stt.normalize_precision("float16"), "f16")
        self.assertEqual(audio_cpp_stt.normalize_precision("int16"), "f16")

    def test_existing_cpu_thread_setting_reaches_audio_cpp(self):
        adapter = audio_cpp_stt.AudioCppASR(
            compute_type="q8_0",
            device="cpu",
            cpu_threads=7,
        )
        adapter.server = mock.Mock(process=None)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.write_bytes(b"gguf")
            with mock.patch.object(audio_cpp_stt, "download_model", return_value=True):
                with mock.patch.object(
                    audio_cpp_stt, "get_model_path", return_value=model_path
                ):
                    adapter.load_model("Qwen3-ASR-0.6B-GGUF")

        self.assertEqual(adapter.server.configure.call_args.kwargs["threads"], 7)

    def test_transcription_uses_generic_task_route_and_existing_context(self):
        response = _Response(payload={"text": "  hello from Vulkan  "})
        fake_server = mock.Mock(model_id="whispering-tiger-stt")
        captured = {}

        def request(*args, **kwargs):
            captured["args"] = args
            captured["json"] = kwargs["json"]
            audio_path = Path(kwargs["json"]["request"]["audio"])
            captured["audio_exists_during_request"] = audio_path.is_file()
            captured["audio"] = audio_path.read_bytes()
            return response

        fake_server.request.side_effect = request
        adapter = audio_cpp_stt.AudioCppASR(device="vulkan", device_index=1)
        adapter.server = fake_server

        with mock.patch.object(adapter, "load_model"):
            result = adapter.transcribe(
                np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                language="en",
                prompt="Whispering Tiger vocabulary",
            )

        self.assertEqual(captured["args"], ("POST", "/v1/tasks/run"))
        request_payload = captured["json"]["request"]
        self.assertEqual(request_payload["language"], "English")
        self.assertEqual(
            request_payload["text"], "Whispering Tiger vocabulary"
        )
        self.assertTrue(captured["audio_exists_during_request"])
        with wave.open(io.BytesIO(captured["audio"]), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
        self.assertEqual(
            result,
            {"text": "hello from Vulkan", "type": "transcribe", "language": "en"},
        )
        self.assertTrue(response.closed)

    def test_structured_segments_turns_and_words_are_preserved(self):
        response = _Response(
            payload={
                "text": "Speaker one.",
                "language": "en",
                "segments": [{"text": "Speaker one.", "start_sample": 1600, "end_sample": 4800}],
                "speaker_turns": [{"speaker": 0, "start_sample": 1600, "end_sample": 4800}],
                "words": [{"word": "Speaker", "start_sample": 1600, "end_sample": 2400}],
            }
        )
        adapter = audio_cpp_stt.AudioCppASR()
        adapter.server = mock.Mock(model_id="unit", request=mock.Mock(return_value=response))

        with mock.patch.object(adapter, "load_model"):
            result = adapter.transcribe(
                np.zeros(1600, dtype=np.float32),
                model="VibeVoice-ASR-GGUF",
                return_timestamps=True,
            )

        self.assertEqual(result["segments"][0]["start"], 0.1)
        self.assertEqual(result["speaker_turns"][0]["end"], 0.3)
        self.assertEqual(result["words"][0]["end"], 0.15)
        request_options = adapter.server.request.call_args.kwargs["json"]["request"]["options"]
        self.assertTrue(request_options["return_timestamps"])

    def test_audio8_is_a_private_local_model_with_a_clear_expected_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(audio_cpp_stt, "MODEL_CACHE_PATH", Path(directory)):
                expected = (
                    Path(directory)
                    / "Audio8-ASR-0.1B-GGUF"
                    / "q8_0"
                    / "audio8-asr-0.1b-q8_0.gguf"
                )
                self.assertEqual(
                    audio_cpp_stt.get_model_path("Audio8-ASR-0.1B-GGUF", "q8_0"),
                    expected,
                )
                self.assertFalse(audio_cpp_stt.is_managed_model("Audio8-ASR-0.1B-GGUF"))
                with self.assertRaisesRegex(FileNotFoundError, "privately self-host"):
                    audio_cpp_stt.download_model("Audio8-ASR-0.1B-GGUF", "q8_0")


class AudioCppTTSTests(unittest.TestCase):
    @staticmethod
    def _adapter():
        adapter = object.__new__(audio_cpp_tts.AudioCppTTS)
        adapter.sample_rate = audio_cpp_tts.SAMPLE_RATE
        adapter.server = mock.Mock(model_id="whispering-tiger-tts")
        adapter.loaded_configuration = None
        adapter.compute_device = "cpu"
        adapter.device_index = 0
        adapter.precision = "orig"
        adapter.last_generation = {
            "audio": None,
            "sample_rate": audio_cpp_tts.SAMPLE_RATE,
            "text": "",
        }
        adapter.audio_streamer = None
        adapter.stop_event = threading.Event()
        adapter.generation_lock = threading.RLock()
        adapter.response_lock = threading.RLock()
        adapter.active_response = None
        return adapter

    def test_existing_voice_and_rate_settings_map_to_supertonic(self):
        adapter = self._adapter()
        values = {"tts_voice": "F3", "tts_prosody_rate": "fast"}
        with mock.patch.object(
            audio_cpp_tts.settings,
            "GetOption",
            side_effect=lambda name: values.get(name),
        ):
            with mock.patch.object(adapter, "_language", return_value="de"):
                payload, voice_path, source_path, emotion_path = adapter._request_payload("Guten Tag")

        self.assertEqual(payload["voice"], "F3")
        self.assertEqual(payload["language"], "de")
        self.assertEqual(payload["options"]["speaking_rate"], 1.22)
        self.assertEqual(payload["num_inference_steps"], 8)
        self.assertIsNone(voice_path)
        self.assertIsNone(source_path)
        self.assertIsNone(emotion_path)

    def test_all_requested_tts_models_and_published_precisions_are_exposed(self):
        self.assertEqual(
            tuple(audio_cpp_tts.TTS_MODELS),
            (
                "Supertonic-3-GGUF",
                "Confucius4-TTS-GGUF",
                "DotTTS-SOAR-GGUF",
                "DotTTS-MeanFlow-GGUF",
                "DotTTS-Edit-GGUF",
                "IndexTTS2-GGUF",
                "IndexTTS2.5-GGUF",
                "MagpieTTS-Multilingual-357M-GGUF",
                "OmniVoice-GGUF",
                "VoxCPM1-0.5B-GGUF",
                "VoxCPM2-GGUF",
                "Breeze-TTS-2-GGUF",
                "Chatterbox-Turbo-GGUF",
                "CosyVoice3-GGUF",
                "Kokoro-82M-GGUF",
                "Audio8-TTS-Preview-0.6B-GGUF",
                "sanoTTS-heart-nano-GGUF",
                "sanoTTS-heart-GGUF",
                "sanoTTS-amy-GGUF",
                "sanoTTS-hfc-GGUF",
                "sanoTTS-kristin-GGUF",
                "sanoTTS-vi-GGUF",
                "sanoTTS-id-GGUF",
                "sanoTTS-cs-GGUF",
                "sanoTTS-de-GGUF",
                "sanoTTS-es-GGUF",
                "sanoTTS-fr-GGUF",
                "sanoTTS-it-GGUF",
                "sanoTTS-pt-GGUF",
                "sanoTTS-ro-GGUF",
                "sanoTTS-ru-GGUF",
                "sanoTTS-tr-GGUF",
                "sanoTTS-ne-GGUF",
                "sanoTTS-hi-GGUF",
            ),
        )
        self.assertEqual(
            audio_cpp_tts.precision_options("IndexTTS2.5-GGUF"),
            ("q8_0", "f16", "orig"),
        )
        self.assertEqual(
            audio_cpp_tts.precision_options("VoxCPM2-GGUF"),
            ("q8_0", "bf16", "orig"),
        )
        self.assertTrue(audio_cpp_tts.TTS_SETTINGS_DEFAULTS["voxcpm2"]["mem_saver"])
        self.assertFalse(audio_cpp_tts.TTS_SETTINGS_DEFAULTS["voxcpm1"]["mem_saver"])

    def test_every_tts_default_is_routed_or_consumed_by_shared_input_controls(self):
        shared_inputs = {
            "language",
            "reference_text",
            "voice_instruction",
            "source_audio",
            "emotion_audio",
            "emotion_happy",
            "emotion_angry",
            "emotion_sad",
            "emotion_afraid",
            "emotion_disgusted",
            "emotion_melancholic",
            "emotion_surprised",
            "emotion_calm",
        }
        for settings_key, defaults in audio_cpp_tts.TTS_SETTINGS_DEFAULTS.items():
            surfaced = {
                *audio_cpp_tts.REQUEST_KEYS.get(settings_key, ()),
                *audio_cpp_tts.SESSION_KEYS.get(settings_key, ()),
                *shared_inputs,
            }
            self.assertEqual(
                set(defaults) - surfaced,
                set(),
                f"unrouted audio.cpp TTS setting(s) for {settings_key}",
            )

    def test_stale_generic_precision_has_a_safe_original_dtype_fallback(self):
        self.assertEqual(audio_cpp_tts.normalize_precision("bfloat16"), "orig")
        self.assertEqual(audio_cpp_tts.normalize_precision("8bit"), "orig")

    def test_tts_session_does_not_reinterpret_preconverted_gguf_weights(self):
        options = audio_cpp_tts._session_options(
            "Supertonic-3-GGUF",
            {"weight_type": "bf16", "style_cache_slots": 6},
        )
        self.assertEqual(options, {"supertonic.style_cache_slots": 6})

    def test_wav_decode_and_export_use_normalized_float32_and_pcm16(self):
        source = np.array([-32767, 0, 32767], dtype=np.int16)
        wave_tensor, rate = audio_cpp_tts._decode_pcm16_wav(_wav_bytes(source))
        self.assertEqual(rate, 44100)
        self.assertEqual(wave_tensor.dtype, torch.float32)
        np.testing.assert_allclose(wave_tensor.numpy()[0], [-1.0, 0.0, 1.0])

        adapter = self._adapter()
        exported = adapter.return_wav_file_binary(wave_tensor, rate)
        with wave.open(io.BytesIO(exported), "rb") as wav_file:
            self.assertEqual(wav_file.getcomptype(), "NONE")
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 44100)

    def test_index_tts_emotion_reference_uses_generic_task_audio_input(self):
        adapter = self._adapter()
        output = _wav_bytes(np.array([-32767, 0, 32767], dtype=np.int16), 22050)
        response = _Response(
            payload={"audio": base64.b64encode(output).decode("ascii"), "sample_rate": 22050}
        )
        captured = {}

        with tempfile.TemporaryDirectory() as directory:
            voice = Path(directory) / "voice.wav"
            emotion = Path(directory) / "emotion.wav"
            voice.write_bytes(_wav_bytes(np.zeros(8, dtype=np.int16), 22050))
            emotion.write_bytes(_wav_bytes(np.ones(8, dtype=np.int16), 22050))
            values = {
                "tts_model": ["Voice cloning and emotion", "IndexTTS2.5-GGUF"],
                "tts_voice": "auto",
                "special_settings": {
                    "tts_audio_cpp": {"index_tts2": {"emotion_audio": str(emotion)}}
                },
            }

            def request(*args, **kwargs):
                captured["args"] = args
                captured["payload"] = kwargs["json"]
                task = kwargs["json"]["request"]
                captured["voice_exists"] = Path(task["voice_ref"]).is_file()
                captured["emotion_exists"] = Path(task["audio"]).is_file()
                return response

            adapter.server.request.side_effect = request
            with mock.patch.object(
                audio_cpp_tts.settings,
                "GetOption",
                side_effect=lambda name: values.get(name),
            ):
                with mock.patch.object(adapter, "load"):
                    with mock.patch.object(adapter, "_language", return_value="en"):
                        with mock.patch.object(
                            adapter, "_finish_audio", side_effect=lambda audio, **kwargs: audio
                        ):
                            audio, rate = adapter.tts("Emotion test", ref_audio=str(voice))

        self.assertEqual(captured["args"], ("POST", "/v1/tasks/run"))
        self.assertTrue(captured["voice_exists"])
        self.assertTrue(captured["emotion_exists"])
        self.assertEqual(captured["payload"]["request"]["language"], "en")
        self.assertEqual(rate, 22050)
        np.testing.assert_allclose(audio.numpy()[0], [-1.0, 0.0, 1.0])
        self.assertTrue(response.closed)

    def test_omnivoice_clone_requires_a_reference_transcript(self):
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            voice = Path(directory) / "voice.wav"
            voice.write_bytes(_wav_bytes(np.zeros(8, dtype=np.int16), 24000))
            values = {
                "tts_model": ["Cloning and voice design", "OmniVoice-GGUF"],
                "tts_voice": "auto",
                "special_settings": {},
            }
            with mock.patch.object(
                audio_cpp_tts.settings,
                "GetOption",
                side_effect=lambda name: values.get(name),
            ):
                with self.assertRaisesRegex(ValueError, "requires a transcript"):
                    adapter._request_payload("Clone me", ref_audio=str(voice))

    def test_streamed_playback_and_returned_audio_are_byte_identical(self):
        adapter = self._adapter()
        first = np.array([-32767, 0], dtype="<i2").tobytes()
        second = np.array([16384, 32767], dtype="<i2").tobytes()
        events = []
        for payload in (first, second):
            events.append(
                "data: "
                + json.dumps(
                    {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(payload).decode("ascii"),
                    }
                )
            )
        events.extend(
            [
                'data: {"type":"speech.audio.done"}',
                "data: [DONE]",
            ]
        )
        response = _Response(lines=events)
        adapter.server.request.return_value = response
        streamer = mock.Mock()
        adapter.audio_streamer = streamer

        with mock.patch.object(adapter, "load"):
            with mock.patch.object(adapter, "init_audio_stream_playback"):
                with mock.patch.object(
                    adapter,
                    "_request_payload",
                    return_value=({"model": "unit", "options": {}}, None, None, None),
                ):
                    with mock.patch.object(
                        adapter,
                        "_finish_audio",
                        side_effect=lambda audio, **kwargs: audio,
                    ):
                        audio, rate = adapter.tts_streaming("stream me")

        played = b"".join(call.args[0] for call in streamer.add_audio_chunk.call_args_list)
        self.assertEqual(played, adapter.return_pcm_audio(audio))
        self.assertEqual(rate, 44100)
        self.assertTrue(response.closed)

    def test_streamer_is_recreated_when_switching_from_24k_to_supertonic(self):
        adapter = self._adapter()
        old_streamer = mock.Mock()
        old_streamer.source_sample_rate = audio_cpp_tts.TTS_MODELS[
            "OmniVoice-GGUF"
        ]["sample_rate"]
        old_streamer.device_index = 7
        adapter.audio_streamer = old_streamer
        adapter.sample_rate = audio_cpp_tts.TTS_MODELS["Supertonic-3-GGUF"][
            "sample_rate"
        ]
        values = {
            "device_out_index": -1,
            "device_default_out_index": 7,
            "tts_streamed_min_play_time": 0.1,
            "tts_streamed_chunk_size": 512,
        }
        replacement = mock.Mock()

        with mock.patch.object(
            audio_cpp_tts.settings,
            "GetOption",
            side_effect=lambda name: values.get(name),
        ):
            with mock.patch.object(
                audio_cpp_tts.audio_tools,
                "AudioStreamer",
                return_value=replacement,
            ) as create_streamer:
                adapter.init_audio_stream_playback()

        old_streamer.stop.assert_called_once_with()
        create_streamer.assert_called_once_with(
            7,
            source_sample_rate=44100,
            start_playback_timeout=1.0,
            min_buffer_play_time=0.1,
            playback_channels=2,
            buffer_size=512,
            input_channels=1,
            dtype="float32",
            tag="tts",
        )
        self.assertIs(adapter.audio_streamer, replacement)

    def test_streamer_is_reused_when_rate_and_device_still_match(self):
        adapter = self._adapter()
        existing_streamer = mock.Mock()
        existing_streamer.source_sample_rate = 44100
        existing_streamer.device_index = 7
        adapter.audio_streamer = existing_streamer
        values = {
            "device_out_index": 7,
            "tts_streamed_min_play_time": 0.1,
            "tts_streamed_chunk_size": 512,
        }

        with mock.patch.object(
            audio_cpp_tts.settings,
            "GetOption",
            side_effect=lambda name: values.get(name),
        ):
            with mock.patch.object(
                audio_cpp_tts.audio_tools, "AudioStreamer"
            ) as create_streamer:
                adapter.init_audio_stream_playback()

        existing_streamer.stop.assert_not_called()
        create_streamer.assert_not_called()
        self.assertIs(adapter.audio_streamer, existing_streamer)

    def test_streaming_buffers_before_a_whole_audio_plugin(self):
        adapter = self._adapter()
        first = np.array([-32767, 0], dtype="<i2").tobytes()
        second = np.array([16384, 32767], dtype="<i2").tobytes()
        response = _Response(
            lines=[
                "data: "
                + json.dumps(
                    {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(payload).decode("ascii"),
                    }
                )
                for payload in (first, second)
            ]
            + ["data: [DONE]"],
        )
        adapter.server.request.return_value = response
        streamer = mock.Mock()
        adapter.audio_streamer = streamer

        class HalfVolumePlugin:
            @staticmethod
            def is_plugin_tts_after_audio_active():
                return True

            @staticmethod
            def on_plugin_tts_after_audio_call(data):
                data["audio"] = data["audio"] * 0.5
                return data

        with mock.patch.object(adapter, "load"):
            with mock.patch.object(adapter, "init_audio_stream_playback") as init_playback:
                with mock.patch.object(
                    adapter,
                    "_request_payload",
                    return_value=({"model": "unit", "options": {}}, None, None, None),
                ):
                    with mock.patch.object(Plugins, "plugins", [HalfVolumePlugin()]):
                        with mock.patch.object(
                            audio_cpp_tts.settings,
                            "GetOption",
                            side_effect=lambda name: {
                                "tts_normalize": False,
                                "tts_volume": 1.0,
                            }.get(name),
                        ):
                            with mock.patch.object(
                                adapter, "_finish_audio", wraps=adapter._finish_audio
                            ) as finish_audio:
                                audio, rate = adapter.tts_streaming("convert me")

        finish_audio.assert_called_once()
        buffered = finish_audio.call_args.args[0]
        np.testing.assert_allclose(
            buffered.numpy()[0],
            [-1.0, 0.0, 16384.0 / 32767.0, 1.0],
        )
        self.assertEqual(
            finish_audio.call_args.kwargs,
            {"normalize": True, "call_plugin": True},
        )
        expected = buffered * 0.5
        init_playback.assert_called_once_with()
        streamer.add_audio_chunk.assert_called_once_with(
            adapter.return_pcm_audio(expected)
        )
        torch.testing.assert_close(audio, expected)
        self.assertEqual(rate, 44100)
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
