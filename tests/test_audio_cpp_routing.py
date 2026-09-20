import unittest
from unittest import mock

import numpy as np

import audioprocessor


class _Settings:
    def __init__(self, **overrides):
        self.values = {
            "whisper_task": "transcribe",
            "current_language": "en",
            "target_language": "en",
            "condition_on_previous_text": False,
            "logprob_threshold": None,
            "no_speech_threshold": None,
            "beam_size": 1,
            "realtime_whisper_beam_size": 1,
            "word_timestamps": False,
            "faster_without_timestamps": False,
            "length_penalty": 1.0,
            "beam_search_patience": 1.0,
            "temperature_fallback": False,
            "realtime_temperature_fallback": False,
            "initial_prompt": "audio.cpp vocabulary",
            "prompt_reset_on_temperature": 0.5,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "stt_type": "audio_cpp",
            "model": "Qwen3-ASR-0.6B-GGUF",
            "whisper_precision": "q8_0",
            "realtime": True,
            "realtime_whisper_model": "Qwen3-ASR-1.7B-GGUF",
            "realtime_whisper_precision": "f16",
            "ai_device": "vulkan",
            "ai_device_index": 2,
        }
        self.values.update(overrides)

    def GetOption(self, name):
        return self.values.get(name)


class AudioCppRoutingTests(unittest.TestCase):
    def test_tts_receives_language_in_both_threaded_playback_modes(self):
        import settings

        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                configured = settings.SettingsManager()
                configured.translate_settings.update(
                    tts_type="audio_cpp", tts_answer=True, tts_queue_enabled=False,
                    tts_streamed_playback=streaming, osc_ip="0", websocket_ip="0",
                    current_language="auto", whisper_task="transcribe",
                )
                adapter = mock.Mock()
                adapter.tts.return_value = ("audio", 24000)
                adapter.tts_streaming.return_value = ("audio", 24000)
                calls = []

                def thread(*, target, args=(), kwargs=None, **ignored):
                    # Execute after send_message has returned, as a real thread may.
                    calls.append(lambda: target(*args, **(kwargs or {})))
                    return mock.Mock()

                with mock.patch.object(audioprocessor.tts, "init", return_value=True), \
                        mock.patch.object(audioprocessor.tts, "tts", adapter), \
                        mock.patch.object(audioprocessor.threading, "Thread", side_effect=thread):
                    audioprocessor.send_message("Hallo", {"text": "Hallo", "language": "de"},
                                                True, configured, None)
                    audioprocessor.send_message("Bonjour", {"text": "Bonjour", "language": "fr"},
                                                True, configured, None)
                    for call in reversed(calls):
                        call()
                method = adapter.tts_streaming if streaming else adapter.tts
                self.assertEqual(method.call_args_list,
                                 [mock.call("Bonjour", language="fr"), mock.call("Hallo", language="de")])

    def test_loader_reuses_existing_cpu_thread_setting_and_gpu_index(self):
        values = {
            "whisper_cpu_threads": 5,
            "whisper_num_workers": 1,
            "stt_type": "audio_cpp",
            "whisper_precision": "q8_0",
            "ai_device": "vulkan",
            "ai_device_index": 3,
        }
        with mock.patch.object(
            audioprocessor.main_settings,
            "GetOption",
            side_effect=lambda name: values.get(name),
        ):
            with mock.patch.object(
                audioprocessor.audio_cpp_stt, "AudioCppASR", return_value="adapter"
            ) as adapter_class:
                result = audioprocessor.load_whisper("Qwen3-ASR-0.6B-GGUF", "vulkan")

        self.assertEqual(result, "adapter")
        adapter_class.assert_called_once_with(
            compute_type="q8_0",
            device="vulkan",
            device_index=3,
            cpu_threads=5,
            role="stt",
        )

    def test_nonfinal_realtime_audio_uses_the_separate_audio_cpp_adapter(self):
        primary = mock.Mock()
        realtime = mock.Mock()
        realtime.transcribe.return_value = {
            "text": "native result",
            "type": "transcribe",
            "language": "en",
        }
        settings = _Settings()

        with mock.patch.object(
            audioprocessor,
            "convert_audio",
            return_value=np.zeros(1600, dtype=np.float32),
        ):
            with mock.patch.object(audioprocessor, "whisper_result_thread"):
                audioprocessor.whisper_ai_thread(
                    b"audio",
                    1,
                    primary,
                    realtime,
                    "",
                    False,
                    settings,
                    [],
                )

        primary.transcribe.assert_not_called()
        realtime.set_compute_type.assert_called_once_with("f16")
        realtime.set_compute_device.assert_called_once_with("vulkan", 2)
        options = realtime.transcribe.call_args.kwargs
        self.assertEqual(options["model"], "Qwen3-ASR-1.7B-GGUF")
        self.assertEqual(options["language"], "en")
        self.assertEqual(options["prompt"], "audio.cpp vocabulary")


if __name__ == "__main__":
    unittest.main()
