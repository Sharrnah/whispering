import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from Models.TextTranslation import texttranslate_hunyuan as hunyuan


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_options = kwargs
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, tokens, **kwargs):
        self.decoded_tokens = tokens
        self.decode_options = kwargs
        return " Hallo Welt. "


class _FakeModel:
    device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generation_options = kwargs
        return torch.tensor([[1, 2, 10, 11]], dtype=torch.long)


class HunyuanTranslationTests(unittest.TestCase):
    def setUp(self):
        self.original_model = hunyuan.model
        self.original_tokenizer = hunyuan.tokenizer
        self.original_configuration = hunyuan.loaded_configuration
        self.original_device = hunyuan.torch_device
        hunyuan.model = None
        hunyuan.tokenizer = None
        hunyuan.loaded_configuration = None
        hunyuan.torch_device = torch.device("cpu")

    def tearDown(self):
        hunyuan.model = self.original_model
        hunyuan.tokenizer = self.original_tokenizer
        hunyuan.loaded_configuration = self.original_configuration
        hunyuan.torch_device = self.original_device

    def test_unloaded_model_has_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            hunyuan.translate_language("Hello", "en", "de")

    def test_translation_uses_the_loaded_tokenizer_instance(self):
        hunyuan.model = _FakeModel()
        hunyuan.tokenizer = _FakeTokenizer()

        result = hunyuan.translate_language("Hello world.", "en", "de")

        self.assertEqual(result, ("Hallo Welt.", "en", "de"))
        self.assertEqual(hunyuan.tokenizer.messages[0]["role"], "user")
        self.assertTrue(hunyuan.tokenizer.template_options["return_dict"])
        self.assertIn("attention_mask", hunyuan.model.generation_options)
        self.assertTrue(torch.equal(hunyuan.tokenizer.decoded_tokens, torch.tensor([10, 11])))

    def test_loader_instantiates_local_model_and_tokenizer(self):
        fake_model = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "small").mkdir()
            with mock.patch.object(
                hunyuan,
                "cache_path",
                Path(temp_dir),
            ), mock.patch.object(
                hunyuan.downloader,
                "download_model",
                return_value=True,
            ), mock.patch.object(
                hunyuan.AutoModelForCausalLM,
                "from_pretrained",
                return_value=fake_model,
            ) as load_model, mock.patch.object(
                hunyuan.AutoTokenizer,
                "from_pretrained",
                return_value=fake_tokenizer,
            ) as load_tokenizer:
                hunyuan.load_model("small", compute_type="float32")

        self.assertIs(hunyuan.model, fake_model)
        self.assertIs(hunyuan.tokenizer, fake_tokenizer)
        self.assertTrue(fake_model.eval_called)
        self.assertTrue(load_model.call_args.kwargs["local_files_only"])
        self.assertEqual(load_model.call_args.kwargs["attn_implementation"], "sdpa")
        self.assertTrue(load_tokenizer.call_args.kwargs["local_files_only"])
        self.assertNotIn("token", load_tokenizer.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
