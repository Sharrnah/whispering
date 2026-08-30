import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from Models.STT import tansformer_whisper


class _FakeProcessor:
    def __init__(self):
        self.feature_extractor = WhisperFeatureExtractor()
        self.call_kwargs = None

    def __call__(self, audio_sample, **kwargs):
        self.call_kwargs = kwargs
        return self.feature_extractor(audio_sample, **kwargs)

    def batch_decode(self, predicted_ids, skip_special_tokens):
        self.predicted_ids = predicted_ids
        self.skip_special_tokens = skip_special_tokens
        return ["complete transcription"]


class _FakeModel:
    def __init__(self, detected_language="de", multilingual=True):
        language_ids = {"en": 50259, "de": 50261, "ja": 50266}
        self.detected_language = detected_language
        self.generation_config = SimpleNamespace(
            is_multilingual=multilingual,
            lang_to_id=(
                {f"<|{code}|>": token_id for code, token_id in language_ids.items()}
                if multilingual
                else None
            ),
        )
        self.detect_language_calls = []
        self.generate_kwargs = None

    def detect_language(self, **kwargs):
        self.detect_language_calls.append(kwargs)
        token_id = self.generation_config.lang_to_id[f"<|{self.detected_language}|>"]
        return torch.tensor([token_id], dtype=torch.long)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


class TransformerWhisperTests(unittest.TestCase):
    def setUp(self):
        # Avoid the singleton constructor because it downloads the model manifest.
        self.adapter = object.__new__(tansformer_whisper.TransformerWhisper)
        self.adapter.compute_type = "float32"
        self.adapter.compute_device = torch.device("cpu")
        self.adapter.compute_device_str = "cpu"
        self.adapter.processor = _FakeProcessor()
        self.adapter.model = _FakeModel()
        self.adapter.load_model = mock.Mock()

    def test_long_audio_is_not_truncated_and_returns_detected_language(self):
        audio = np.zeros(31 * 16_000, dtype=np.float32)

        result = self.adapter.transcribe(
            audio,
            model="tiny",
            task="transcribe",
            language=None,
            beam_size=2,
        )

        self.assertFalse(self.adapter.processor.call_kwargs["truncation"])
        generated_features = self.adapter.model.generate_kwargs["input_features"]
        self.assertEqual(generated_features.shape[-1], 3100)
        self.assertEqual(self.adapter.model.generate_kwargs["attention_mask"].sum().item(), 3100)
        self.assertTrue(self.adapter.model.generate_kwargs["return_timestamps"])
        self.assertEqual(self.adapter.model.generate_kwargs["language"], "de")
        self.assertEqual(len(self.adapter.model.detect_language_calls), 1)
        self.assertEqual(
            result,
            {"text": "complete transcription", "type": "transcribe", "language": "de"},
        )

    def test_explicit_language_skips_detection(self):
        result = self.adapter.transcribe(
            np.zeros(16_000, dtype=np.float32),
            model="tiny",
            task="translate",
            language="ja",
        )

        self.assertEqual(self.adapter.model.detect_language_calls, [])
        self.assertEqual(self.adapter.model.generate_kwargs["language"], "ja")
        self.assertEqual(result["language"], "ja")

    def test_english_only_model_reports_english_without_forcing_language(self):
        self.adapter.model = _FakeModel(multilingual=False)

        result = self.adapter.transcribe(
            np.zeros(16_000, dtype=np.float32),
            model="tiny.en",
            task="transcribe",
            language="auto",
        )

        self.assertEqual(self.adapter.model.detect_language_calls, [])
        self.assertIsNone(self.adapter.model.generate_kwargs["language"])
        self.assertEqual(result["language"], "en")


if __name__ == "__main__":
    unittest.main()
