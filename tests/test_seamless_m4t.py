import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch

import Models


class _RecordingProcessor:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"input_features": torch.zeros((1, 1), dtype=torch.float32)}

    @staticmethod
    def decode(_tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "transcript"


class _RecordingModel:
    def __init__(self):
        self.config = SimpleNamespace()
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(sequences=torch.tensor([[1]], dtype=torch.long))


def _load_seamless_module():
    module_path = Path(__file__).parents[1] / "Models" / "Multi" / "seamless_m4t.py"
    spec = importlib.util.spec_from_file_location("seamless_m4t_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    isolated_imports = {
        "audio_tools": ModuleType("audio_tools"),
        "downloader": ModuleType("downloader"),
        "settings": ModuleType("settings"),
        "Models.languageClassification": ModuleType("Models.languageClassification"),
    }
    with mock.patch.dict(sys.modules, isolated_imports), mock.patch.object(
        Models,
        "languageClassification",
        isolated_imports["Models.languageClassification"],
        create=True,
    ):
        spec.loader.exec_module(module)
    return module


class SeamlessM4TTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seamless_module = _load_seamless_module()

    def make_adapter(self):
        adapter = object.__new__(self.seamless_module.SeamlessM4T)
        adapter.processor = _RecordingProcessor()
        adapter.model = _RecordingModel()
        adapter.precision = torch.float32
        adapter.device = torch.device("cpu")
        return adapter

    def test_transcribe_passes_audio_using_current_transformers_keyword(self):
        adapter = self.make_adapter()
        audio = np.zeros(160, dtype=np.float32)

        result = adapter.transcribe(
            audio,
            source_lang="eng",
            target_lang="deu",
            beam_size=1,
        )

        self.assertIs(adapter.processor.kwargs["audio"], audio)
        self.assertNotIn("audios", adapter.processor.kwargs)
        self.assertEqual(adapter.processor.kwargs["sampling_rate"], 16000)
        self.assertEqual(result["text"], "transcript")

    def test_transcribe_passes_generation_controls_without_mutating_model_config(self):
        adapter = self.make_adapter()

        adapter.transcribe(
            np.zeros(160, dtype=np.float32),
            source_lang="eng",
            target_lang="deu",
            beam_size=3,
            repetition_penalty=1.1,
            length_penalty=1.2,
            no_repeat_ngram_size=2,
        )

        self.assertEqual(adapter.model.generate_kwargs["text_num_beams"], 3)
        self.assertEqual(adapter.model.generate_kwargs["text_repetition_penalty"], 1.1)
        self.assertEqual(adapter.model.generate_kwargs["text_length_penalty"], 1.2)
        self.assertEqual(adapter.model.generate_kwargs["text_no_repeat_ngram_size"], 2)
        self.assertFalse(hasattr(adapter.model.config, "repetition_penalty"))
        self.assertFalse(hasattr(adapter.model.config, "length_penalty"))
        self.assertFalse(hasattr(adapter.model.config, "no_repeat_ngram_size"))


if __name__ == "__main__":
    unittest.main()
