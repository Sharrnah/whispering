import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILERO_SOURCE = PROJECT_ROOT / "Models" / "TTS" / "silero.py"
ALLOWED_DOWNLOAD_HOSTS = {
    "eu2.contabostorage.com",
    "usc1.contabostorage.com",
    "s3.libs.space",
}


def _fake_module(name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _load_silero_module():
    fake_downloader = _fake_module(
        "downloader",
        download_extract=mock.Mock(),
        sha256_checksum=mock.Mock(),
    )
    fake_modules = {
        "Plugins": _fake_module(
            "Plugins", plugin_custom_event_call=mock.Mock(return_value=None)
        ),
        "audio_tools": _fake_module("audio_tools"),
        "downloader": fake_downloader,
        "settings": _fake_module("settings", GetOption=mock.Mock(return_value="cpu")),
    }
    spec = importlib.util.spec_from_file_location("silero_tts_test_module", SILERO_SOURCE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class SileroMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.silero = _load_silero_module()

    def setUp(self):
        self.silero.downloader.download_extract.reset_mock(
            return_value=True, side_effect=True
        )
        self.silero.downloader.sha256_checksum.reset_mock(
            return_value=True, side_effect=True
        )

    def _adapter(self, language="en", model_id="v3_en"):
        adapter = object.__new__(self.silero.Silero)
        adapter.lang = language
        adapter.model_id = model_id
        adapter.model = None
        adapter._package_importer = None
        adapter._verified_model_files = set()
        return adapter

    def test_manifest_contains_only_application_mirrors(self):
        for model_info in self.silero.SILERO_TTS_MODELS.values():
            self.assertEqual(len(model_info["sha256"]), 64)
            self.assertTrue(model_info["urls"])
            self.assertTrue(
                all(
                    urlsplit(url).hostname in ALLOWED_DOWNLOAD_HOSTS
                    for url in model_info["urls"]
                )
            )

        source = SILERO_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("models.silero.ai", source)
        self.assertNotIn("download_url_to_file", source)
        self.assertNotIn("torch.hub.load", source)

    def test_selector_exposes_only_models_with_mirror_entries(self):
        adapter = self._adapter()
        models_by_language = adapter.list_models()

        exposed_models = {
            model_id
            for model_ids in models_by_language.values()
            for model_id in model_ids
        }
        self.assertEqual(exposed_models, set(self.silero.SILERO_TTS_MODELS))
        self.assertEqual(adapter.list_languages(), list(models_by_language))

    def test_verified_cached_package_loads_without_network_or_torch_hub(self):
        adapter = self._adapter()
        expected_hash = self.silero.SILERO_TTS_MODELS[adapter.model_id]["sha256"]

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            self.silero, "cache_path", Path(directory)
        ):
            model_path = (
                Path(directory)
                / self.silero.SILERO_MODEL_CACHE_RELATIVE_PATH
                / "v3_en.pt"
            )
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"verified model")
            self.silero.downloader.sha256_checksum.return_value = expected_hash

            loaded_model = object()
            importer = mock.Mock()
            importer.load_pickle.return_value = loaded_model
            with mock.patch.object(
                self.silero, "PackageImporter", return_value=importer
            ) as package_importer, mock.patch.object(
                self.silero.torch.hub, "load"
            ) as torch_hub_load:
                self.assertTrue(adapter._load_model())

            self.silero.downloader.download_extract.assert_not_called()
            torch_hub_load.assert_not_called()
            package_importer.assert_called_once_with(str(model_path))
            importer.load_pickle.assert_called_once_with("tts_models", "model")
            self.assertIs(adapter.model, loaded_model)

    def test_missing_package_downloads_from_mirrors_before_local_load(self):
        adapter = self._adapter()
        model_info = self.silero.SILERO_TTS_MODELS[adapter.model_id]

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            self.silero, "cache_path", Path(directory)
        ):
            def create_downloaded_model(urls, extract_dir, checksum, **kwargs):
                self.assertEqual(urls, model_info["urls"])
                self.assertEqual(checksum, model_info["sha256"])
                self.assertTrue(kwargs["force_non_ui_dl"])
                self.assertEqual(kwargs["extract_format"], "none")
                target = Path(extract_dir) / os.path.basename(urls[0])
                target.write_bytes(b"downloaded model")
                return True

            self.silero.downloader.download_extract.side_effect = create_downloaded_model
            self.silero.downloader.sha256_checksum.return_value = model_info["sha256"]
            importer = mock.Mock()
            importer.load_pickle.return_value = object()

            with mock.patch.object(
                self.silero, "PackageImporter", return_value=importer
            ) as package_importer:
                self.assertTrue(adapter._load_model())

            self.silero.downloader.download_extract.assert_called_once()
            package_importer.assert_called_once()

    def test_corrupt_cached_package_is_replaced_from_mirrors(self):
        adapter = self._adapter()
        model_info = self.silero.SILERO_TTS_MODELS[adapter.model_id]

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            self.silero, "cache_path", Path(directory)
        ):
            model_path = (
                Path(directory)
                / self.silero.SILERO_MODEL_CACHE_RELATIVE_PATH
                / "v3_en.pt"
            )
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"corrupt model")

            def replace_model(urls, extract_dir, checksum, **kwargs):
                del checksum, kwargs
                (Path(extract_dir) / os.path.basename(urls[0])).write_bytes(
                    b"verified replacement"
                )
                return True

            self.silero.downloader.sha256_checksum.side_effect = [
                "0" * 64,
                model_info["sha256"],
            ]
            self.silero.downloader.download_extract.side_effect = replace_model
            importer = mock.Mock()
            importer.load_pickle.return_value = object()

            with mock.patch.object(
                self.silero, "PackageImporter", return_value=importer
            ):
                self.assertTrue(adapter._load_model())

            self.assertEqual(model_path.read_bytes(), b"verified replacement")
            self.silero.downloader.download_extract.assert_called_once()

    def test_unmirrored_or_wrong_language_model_fails_before_network(self):
        for adapter in (
            self._adapter(model_id="aidar_v2"),
            self._adapter(language="de", model_id="v3_en"),
        ):
            with self.subTest(language=adapter.lang, model=adapter.model_id):
                with mock.patch.object(self.silero, "PackageImporter") as importer:
                    self.assertFalse(adapter._load_model())
                importer.assert_not_called()

        self.silero.downloader.download_extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
