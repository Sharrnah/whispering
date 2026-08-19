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
            "beam_size": 4,
            "realtime_whisper_beam_size": 1,
            "word_timestamps": False,
            "faster_without_timestamps": False,
            "length_penalty": 1.0,
            "beam_search_patience": 1.0,
            "temperature_fallback": False,
            "realtime_temperature_fallback": False,
            "initial_prompt": "Qwen vocabulary",
            "prompt_reset_on_temperature": 0.5,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "stt_type": "qwen3_asr",
            "model": "Qwen3-ASR-0.6B-hf",
            "whisper_precision": "bfloat16",
            "realtime": False,
            "realtime_whisper_model": "Qwen3-ASR-1.7B-hf",
            "realtime_whisper_precision": "float16",
            "ai_device": "cuda",
        }
        self.values.update(overrides)

    def GetOption(self, name):
        return self.values.get(name)


def _model():
    model = mock.Mock()
    model.transcribe.return_value = {
        "text": "Qwen result",
        "type": "transcribe",
        "language": "en",
    }
    return model


class Qwen3ASRRoutingTests(unittest.TestCase):
    def _run(self, settings, primary, realtime=None, final_audio=True):
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
                    final_audio,
                    settings,
                    [],
                )

    def test_normal_qwen_uses_normal_beam_setting(self):
        primary = _model()

        self._run(
            _Settings(
                beam_size=4,
                length_penalty=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            ),
            primary,
        )

        options = primary.transcribe.call_args.kwargs
        self.assertEqual(options["beam_size"], 4)
        self.assertEqual(options["length_penalty"], 0.9)
        self.assertEqual(options["repetition_penalty"], 1.15)
        self.assertEqual(options["no_repeat_ngram_size"], 3)
        self.assertNotIn("temperature", options)
        self.assertNotIn("condition_on_previous_text", options)
        self.assertNotIn("patience", options)
        self.assertEqual(
            options["model"],
            "Qwen3-ASR-0.6B-hf",
        )

    def test_realtime_qwen_on_primary_uses_realtime_beam_setting(self):
        primary = _model()

        self._run(
            _Settings(realtime=True, realtime_whisper_beam_size=2),
            primary,
            final_audio=False,
        )

        self.assertEqual(primary.transcribe.call_args.kwargs["beam_size"], 2)

    def test_separate_realtime_qwen_uses_realtime_model_and_beam(self):
        primary = _model()
        realtime = _model()

        self._run(
            _Settings(realtime=True, realtime_whisper_beam_size=3),
            primary,
            realtime=realtime,
            final_audio=False,
        )

        primary.transcribe.assert_not_called()
        self.assertEqual(realtime.transcribe.call_args.kwargs["beam_size"], 3)
        self.assertEqual(
            realtime.transcribe.call_args.kwargs["model"],
            "Qwen3-ASR-1.7B-hf",
        )
        realtime.set_compute_type.assert_called_once_with("float16")


if __name__ == "__main__":
    unittest.main()
