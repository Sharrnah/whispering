import string
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from Models.STT import qwen3_asr


class _FakeBatch(dict):
    def to(self, device, dtype):
        self.to_args = (device, dtype)
        return self


class _FakeProcessor:
    def __init__(self):
        self.messages = None
        self.template_options = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_options = kwargs
        return _FakeBatch(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    def decode(self, generated_ids, return_format):
        self.generated_ids = generated_ids
        self.return_format = return_format
        return [{"language": None, "transcription": "hello Qwen"}]


class _FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


class Qwen3ASRTests(unittest.TestCase):
    def setUp(self):
        self.adapter = qwen3_asr.Qwen3ASR(compute_type="float32", device="cpu")
        self.adapter.model = _FakeModel()
        self.adapter.processor = _FakeProcessor()

    def tearDown(self):
        self.adapter.model = None
        self.adapter.processor = None
        self.adapter.loaded_configuration = None

    def test_manifests_use_hosted_zip_archives_and_sha256(self):
        expected_files = {
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        for model_name, entry in qwen3_asr.MODEL_LINKS.items():
            with self.subTest(model=model_name):
                self.assertEqual(set(entry["file_checksums"]), expected_files)
                self.assertEqual(entry["path"], model_name)
                self.assertNotIn("file_urls", entry)
                self.assertNotIn("revision", entry)
                self.assertEqual(len(entry["checksum"]), 64)
                self.assertTrue(all(character in string.hexdigits for character in entry["checksum"]))
                for checksum in entry["file_checksums"].values():
                    self.assertEqual(len(checksum), 64)
                    self.assertTrue(all(character in string.hexdigits for character in checksum))
                self.assertEqual(len(entry["urls"]), 3)
                for url in entry["urls"]:
                    self.assertNotIn("huggingface.co", url)
                    self.assertTrue(url.endswith(f"/{model_name}.zip"))

    def test_pending_archive_checksum_never_falls_back_to_huggingface(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(qwen3_asr, "MODEL_CACHE_PATH", Path(temp_dir)):
                with mock.patch.object(qwen3_asr.downloader, "download_model") as download:
                    with self.assertRaisesRegex(RuntimeError, "not currently available"):
                        qwen3_asr.download_model("Qwen3-ASR-0.6B-hf")
                    download.assert_not_called()

    def test_configured_archive_uses_standard_zip_downloader(self):
        entry = qwen3_asr.MODEL_LINKS["Qwen3-ASR-0.6B-hf"]
        with mock.patch.dict(entry, {"checksum": "1" * 64}):
            with mock.patch.object(qwen3_asr, "needs_download", return_value=True):
                with mock.patch.object(
                    qwen3_asr.downloader, "download_model", return_value=True
                ) as download:
                    self.assertTrue(
                        qwen3_asr.download_model(
                            "Qwen3-ASR-0.6B-hf", force_non_ui_dl=True
                        )
                    )

        download.assert_called_once()
        download_settings = download.call_args.args[0]
        self.assertEqual(download_settings["extract_format"], "zip")
        self.assertTrue(download_settings["force_non_ui_dl"])
        self.assertIs(download_settings["model_link_dict"], qwen3_asr.MODEL_LINKS)

    def test_primary_and_realtime_adapters_are_independent(self):
        other = qwen3_asr.Qwen3ASR(compute_type="float32", device="cpu")
        self.assertIsNot(self.adapter, other)

    def test_language_selector_contains_all_52_qwen_variants(self):
        expected_standard_languages = {
            "Arabic",
            "Cantonese",
            "Chinese",
            "Czech",
            "Danish",
            "Dutch",
            "English",
            "Filipino",
            "Finnish",
            "French",
            "German",
            "Greek",
            "Hindi",
            "Hungarian",
            "Indonesian",
            "Italian",
            "Japanese",
            "Korean",
            "Macedonian",
            "Malay",
            "Persian",
            "Polish",
            "Portuguese",
            "Romanian",
            "Russian",
            "Spanish",
            "Swedish",
            "Thai",
            "Turkish",
            "Vietnamese",
        }
        expected_dialects = {
            "Anhui",
            "Dongbei",
            "Fujian",
            "Gansu",
            "Guizhou",
            "Hebei",
            "Henan",
            "Hubei",
            "Hunan",
            "Jiangxi",
            "Ningxia",
            "Shandong",
            "Shaanxi",
            "Shanxi",
            "Sichuan",
            "Tianjin",
            "Yunnan",
            "Zhejiang",
            "Cantonese (HK)",
            "Cantonese (Guangdong)",
            "Wu",
            "Minnan",
        }

        languages = qwen3_asr.get_languages()

        self.assertEqual(len(qwen3_asr.STANDARD_LANGUAGES), 30)
        self.assertEqual(len(qwen3_asr.DIALECTS), 22)
        self.assertEqual(len(qwen3_asr.LANGUAGES), 52)
        self.assertEqual(len(languages), 53)  # Auto plus all supported variants.
        self.assertEqual(languages[0], {"code": "", "name": "Auto"})
        self.assertEqual(
            set(qwen3_asr.STANDARD_LANGUAGES.values()),
            expected_standard_languages,
        )
        self.assertEqual(set(qwen3_asr.DIALECTS.values()), expected_dialects)
        self.assertEqual(len({entry["code"] for entry in languages}), len(languages))

    def test_dialect_names_and_official_aliases_resolve_to_ui_codes(self):
        self.assertEqual(
            self.adapter._resolve_language("zh-anhui"),
            ("zh-anhui", "Anhui"),
        )
        self.assertEqual(
            self.adapter._resolve_language("Cantonese (Hong Kong accent)"),
            ("yue-hk", "Cantonese (HK)"),
        )
        self.assertEqual(
            self.adapter._language_code_from_name("Wu language"),
            "zh-wu",
        )

    def test_transcription_preserves_context_and_forces_language(self):
        with mock.patch.object(self.adapter, "load_model"):
            result = self.adapter.transcribe(
                np.zeros(1600, dtype=np.float32),
                language="en",
                prompt="Vocabulary: Whispering Tiger, Qwen.",
                beam_size=2,
            )

        messages = self.adapter.processor.messages
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"][0]["text"], "Vocabulary: Whispering Tiger, Qwen.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["content"][0]["text"], "language English<asr_text>")
        self.assertTrue(self.adapter.processor.template_options["continue_final_message"])
        self.assertEqual(self.adapter.model.generate_kwargs["num_beams"], 2)
        self.assertEqual(self.adapter.model.generate_kwargs["max_new_tokens"], 512)
        self.assertFalse(self.adapter.model.generate_kwargs["do_sample"])
        self.assertTrue(self.adapter.model.generate_kwargs["use_cache"])
        self.assertEqual(result, {"text": "hello Qwen", "type": "transcribe", "language": "en"})

    def test_auto_language_uses_generation_prompt(self):
        self.adapter.processor.decode = mock.Mock(
            return_value=[{"language": "German", "transcription": "Guten Tag"}]
        )
        with mock.patch.object(self.adapter, "load_model"):
            result = self.adapter.transcribe(np.zeros(1600, dtype=np.float32), language=None)

        self.assertEqual(len(self.adapter.processor.messages), 1)
        self.assertTrue(self.adapter.processor.template_options["add_generation_prompt"])
        self.assertEqual(self.adapter.model.generate_kwargs["num_beams"], 1)
        self.assertEqual(result["language"], "de")

    def test_compatible_transformers_decoding_controls_are_forwarded(self):
        with mock.patch.object(self.adapter, "load_model"):
            self.adapter.transcribe(
                np.zeros(1600, dtype=np.float32),
                beam_size=3,
                length_penalty=0.8,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
            )

        generation = self.adapter.model.generate_kwargs
        self.assertEqual(generation["length_penalty"], 0.8)
        self.assertEqual(generation["repetition_penalty"], 1.1)
        self.assertEqual(generation["no_repeat_ngram_size"], 4)

    def test_neutral_or_inapplicable_controls_keep_qwen_generation_defaults(self):
        with mock.patch.object(self.adapter, "load_model"):
            self.adapter.transcribe(
                np.zeros(1600, dtype=np.float32),
                beam_size=1,
                length_penalty=0.5,
                repetition_penalty=0,
                no_repeat_ngram_size=0,
            )

        generation = self.adapter.model.generate_kwargs
        self.assertNotIn("length_penalty", generation)
        self.assertNotIn("repetition_penalty", generation)
        self.assertNotIn("no_repeat_ngram_size", generation)
        self.assertNotIn("temperature", generation)

    def test_detected_dialect_uses_its_ui_code(self):
        self.adapter.processor.decode = mock.Mock(
            return_value=[{"language": "Minnan language", "transcription": "li ho"}]
        )
        with mock.patch.object(self.adapter, "load_model"):
            result = self.adapter.transcribe(np.zeros(1600, dtype=np.float32))

        self.assertEqual(result["language"], "zh-minnan")

    def test_translation_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "transcription"):
            self.adapter.transcribe(np.zeros(10, dtype=np.float32), task="translate")


if __name__ == "__main__":
    unittest.main()
