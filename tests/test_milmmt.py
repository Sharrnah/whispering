import string
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from Models.TextTranslation import texttranslate_milmmt as milmmt
from Models.TextTranslation import texttranslate as translation_router


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.tokenizer_options = kwargs
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, generated_tokens, **kwargs):
        self.generated_tokens = generated_tokens
        self.decode_options = kwargs
        return " Hello world. "


class _FakeModel:
    device = torch.device("cpu")

    def generate(self, **kwargs):
        self.generation_options = kwargs
        return torch.tensor([[1, 2, 10, 11]], dtype=torch.long)


class MiLMMTTests(unittest.TestCase):
    def setUp(self):
        self.original_model = milmmt.model
        self.original_tokenizer = milmmt.tokenizer
        self.original_loaded_configuration = milmmt.loaded_configuration
        self.original_router_configuration = translation_router._active_translator_configuration
        milmmt.model = _FakeModel()
        milmmt.tokenizer = _FakeTokenizer()
        milmmt.loaded_configuration = (milmmt.DEFAULT_MODEL, "bfloat16", "cpu")
        translation_router._active_translator_configuration = None

    def tearDown(self):
        milmmt.model = self.original_model
        milmmt.tokenizer = self.original_tokenizer
        milmmt.loaded_configuration = self.original_loaded_configuration
        translation_router._active_translator_configuration = self.original_router_configuration

    def test_language_list_matches_the_46_official_languages(self):
        expected_names = {
            "Arabic",
            "Azerbaijani",
            "Bulgarian",
            "Bengali",
            "Catalan",
            "Czech",
            "Danish",
            "German",
            "Greek",
            "English",
            "Spanish",
            "Persian",
            "Finnish",
            "French",
            "Hebrew",
            "Hindi",
            "Croatian",
            "Hungarian",
            "Indonesian",
            "Italian",
            "Japanese",
            "Kazakh",
            "Khmer",
            "Korean",
            "Lao",
            "Malay",
            "Burmese",
            "Norwegian",
            "Dutch",
            "Polish",
            "Portuguese",
            "Romanian",
            "Russian",
            "Slovak",
            "Slovenian",
            "Swedish",
            "Tamil",
            "Thai",
            "Tagalog",
            "Turkish",
            "Urdu",
            "Uzbek",
            "Vietnamese",
            "Cantonese",
            "Chinese (Simplified)",
            "Chinese (Traditional)",
        }

        languages = milmmt.get_installed_language_names()

        self.assertEqual(len(languages), 46)
        self.assertEqual({entry["name"] for entry in languages}, expected_names)
        self.assertEqual(len({entry["code"] for entry in languages}), 46)

    def test_manifests_use_only_hosted_zip_archives(self):
        expected_weight_files = {
            "MiLMMT-46-1B-v1.0": {"model.safetensors"},
            "MiLMMT-46-4B-v1.0": {
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            },
            "MiLMMT-46-12B-v1.0": {
                "model-00001-of-00005.safetensors",
                "model-00002-of-00005.safetensors",
                "model-00003-of-00005.safetensors",
                "model-00004-of-00005.safetensors",
                "model-00005-of-00005.safetensors",
                "model.safetensors.index.json",
            },
        }
        common_files = {
            "README.md",
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        }

        for model_name, entry in milmmt.MODEL_LINKS.items():
            with self.subTest(model=model_name):
                self.assertEqual(
                    set(entry["file_checksums"]),
                    common_files | expected_weight_files[model_name],
                )
                self.assertEqual(entry["path"], model_name)
                self.assertEqual(len(entry["checksum"]), 64)
                self.assertTrue(
                    all(character in string.hexdigits for character in entry["checksum"])
                )
                for checksum in entry["file_checksums"].values():
                    self.assertEqual(len(checksum), 64)
                    self.assertTrue(
                        all(character in string.hexdigits for character in checksum)
                    )
                self.assertEqual(len(entry["urls"]), 3)
                for url in entry["urls"]:
                    self.assertNotIn("huggingface.co", url)
                    self.assertTrue(url.endswith(f"/{model_name}.zip"))

    def test_pending_archive_never_falls_back_to_huggingface(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(milmmt, "MODEL_CACHE_PATH", Path(temp_dir)):
                with mock.patch.object(milmmt.downloader, "download_model") as download:
                    with self.assertRaisesRegex(RuntimeError, "application-hosted"):
                        milmmt.download_model("MiLMMT-46-1B-v1.0")
                    download.assert_not_called()

    def test_configured_archive_uses_standard_zip_downloader(self):
        entry = milmmt.MODEL_LINKS["MiLMMT-46-1B-v1.0"]
        with mock.patch.dict(entry, {"checksum": "1" * 64}):
            with mock.patch.object(milmmt, "needs_download", return_value=True):
                with mock.patch.object(
                    milmmt.downloader,
                    "download_model",
                    return_value=True,
                ) as download:
                    self.assertTrue(
                        milmmt.download_model(
                            "MiLMMT-46-1B-v1.0",
                            force_non_ui_dl=True,
                        )
                    )

        download_settings = download.call_args.args[0]
        self.assertEqual(download_settings["extract_format"], "zip")
        self.assertTrue(download_settings["force_non_ui_dl"])
        self.assertIs(download_settings["model_link_dict"], milmmt.MODEL_LINKS)

    def test_official_prompt_and_generated_token_slicing(self):
        translation, source_code, target_code = milmmt.translate_language(
            "Hallo Welt.",
            "de",
            "en",
        )

        self.assertEqual(
            milmmt.tokenizer.prompt,
            "Translate this from German to English:\n"
            "German: Hallo Welt.\n"
            "English:",
        )
        self.assertFalse(milmmt.tokenizer.tokenizer_options["add_special_tokens"])
        self.assertTrue(torch.equal(milmmt.tokenizer.generated_tokens, torch.tensor([10, 11])))
        self.assertTrue(milmmt.tokenizer.decode_options["skip_special_tokens"])
        self.assertFalse(milmmt.model.generation_options["do_sample"])
        self.assertEqual(milmmt.model.generation_options["max_new_tokens"], 64)
        self.assertEqual(translation, "Hello world.")
        self.assertEqual(source_code, "de")
        self.assertEqual(target_code, "en")

    def test_auto_source_language_uses_existing_lid_model(self):
        with mock.patch.object(
            milmmt.languageClassification,
            "classify",
            return_value=("deu_Latn", 0.99),
        ) as classify:
            translation, source_code, target_code = milmmt.translate_language(
                "Hallo Welt.",
                "auto",
                "fra_Latn",
            )

        classify.assert_called_once_with("Hallo Welt.")
        self.assertEqual(translation, "Hello world.")
        self.assertEqual(source_code, "de")
        self.assertEqual(target_code, "fr")
        self.assertIn("from German to French", milmmt.tokenizer.prompt)

    def test_script_variants_resolve_to_distinct_chinese_prompts(self):
        self.assertEqual(
            milmmt._resolve_language("zho_Hans"),
            ("zh", "Chinese (Simplified)"),
        )
        self.assertEqual(
            milmmt._resolve_language("zho_Hant"),
            ("zh-Hant", "Chinese (Traditional)"),
        )
        self.assertEqual(
            milmmt._resolve_language("yue_Hant"),
            ("yue", "Cantonese"),
        )

    def test_legacy_size_aliases_resolve_to_v1_models(self):
        self.assertEqual(
            milmmt.get_model_path("small").name,
            "MiLMMT-46-1B-v1.0",
        )
        self.assertEqual(
            milmmt.get_model_path("medium").name,
            "MiLMMT-46-4B-v1.0",
        )
        self.assertEqual(
            milmmt.get_model_path("large").name,
            "MiLMMT-46-12B-v1.0",
        )

    def test_unsupported_4bit_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported MiLMMT precision"):
            milmmt._dtype_settings("4bit")

    def test_legacy_float16_uses_a_stable_dtype(self):
        with mock.patch.object(
            milmmt,
            "torch_device",
            torch.device("cuda"),
        ), mock.patch.object(
            milmmt.torch.cuda,
            "is_bf16_supported",
            return_value=True,
        ):
            self.assertEqual(milmmt._effective_compute_type("float16"), "bfloat16")

        with mock.patch.object(milmmt, "torch_device", torch.device("cpu")):
            self.assertEqual(milmmt._effective_compute_type("float16"), "float32")

    def test_empty_generation_is_an_error_instead_of_a_blank_translation(self):
        milmmt.tokenizer.decode = mock.Mock(return_value="   ")

        with self.assertRaisesRegex(RuntimeError, "empty translation"):
            milmmt.translate_language("Hallo Welt.", "de", "en")

    def test_text_translation_router_selects_milmmt(self):
        selected_settings = {
            "txt_translator_device": "cuda",
            "txt_translator_size": "MiLMMT-46-1B-v1.0",
            "txt_translator_precision": "bfloat16",
        }
        with mock.patch.object(
            translation_router,
            "get_current_translator",
            return_value="milmmt",
        ):
            with mock.patch.object(
                translation_router.settings,
                "GetOption",
                side_effect=selected_settings.get,
            ):
                with mock.patch.object(milmmt, "set_device") as set_device:
                    with mock.patch.object(milmmt, "load_model") as load_model:
                        translation_router.InstallLanguages()

                        self.assertEqual(
                            translation_router.GetInstalledLanguageNames(),
                            milmmt.get_installed_language_names(),
                        )

                        with mock.patch.object(
                            milmmt,
                            "translate_language",
                            return_value=("Hallo!", "en", "de"),
                        ) as translate:
                            result = translation_router.TranslateLanguage("Hello!", "en", "de")

            set_device.assert_called_once_with("cuda")
            load_model.assert_called_once_with(
                "MiLMMT-46-1B-v1.0",
                compute_type="bfloat16",
            )

            translate.assert_called_once_with("Hello!", "en", "de", False)
            self.assertEqual(result, ("Hallo!", "en", "de"))

    def test_router_loads_a_translator_selected_after_startup(self):
        selected = {
            "translator": "milmmt",
            "txt_translator_device": "cuda",
            "txt_translator_size": "MiLMMT-46-1B-v1.0",
            "txt_translator_precision": "bfloat16",
        }
        loaded = {"milmmt": False, "hunyuan_mt": False}

        def load_milmmt(*args, **kwargs):
            loaded["milmmt"] = True

        def release_milmmt():
            loaded["milmmt"] = False

        def load_hunyuan(*args, **kwargs):
            loaded["hunyuan_mt"] = True

        def release_hunyuan():
            loaded["hunyuan_mt"] = False

        hunyuan = translation_router.texttranslate_hunyuan
        with (
            mock.patch.object(
                translation_router,
                "get_current_translator",
                side_effect=lambda: selected["translator"],
            ),
            mock.patch.object(
                translation_router.settings,
                "GetOption",
                side_effect=selected.get,
            ),
            mock.patch.object(
                milmmt,
                "is_model_loaded",
                side_effect=lambda: loaded["milmmt"],
            ),
            mock.patch.object(
                milmmt,
                "set_device",
            ),
            mock.patch.object(
                milmmt,
                "load_model",
                side_effect=load_milmmt,
            ) as milmmt_load,
            mock.patch.object(
                milmmt,
                "release_model",
                side_effect=release_milmmt,
            ) as milmmt_release,
            mock.patch.object(
                milmmt,
                "translate_language",
                return_value=("Hello", "de", "en"),
            ),
            mock.patch.object(
                hunyuan,
                "is_model_loaded",
                side_effect=lambda: loaded["hunyuan_mt"],
            ),
            mock.patch.object(
                hunyuan,
                "set_device",
            ),
            mock.patch.object(
                hunyuan,
                "load_model",
                side_effect=load_hunyuan,
            ) as hunyuan_load,
            mock.patch.object(
                hunyuan,
                "release_model",
                side_effect=release_hunyuan,
            ),
            mock.patch.object(
                hunyuan,
                "translate_language",
                return_value=("Bonjour", "de", "fr"),
            ),
        ):
            first = translation_router.TranslateLanguage("Hallo", "de", "en")

            selected.update(
                {
                    "translator": "hunyuan_mt",
                    "txt_translator_size": "small",
                    "txt_translator_precision": "float16",
                }
            )
            second = translation_router.TranslateLanguage("Hallo", "de", "fr")

        self.assertEqual(first, ("Hello", "de", "en"))
        self.assertEqual(second, ("Bonjour", "de", "fr"))
        milmmt_load.assert_called_once()
        milmmt_release.assert_called_once()
        hunyuan_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
