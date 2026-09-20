import unittest
from unittest import mock

import settings
from Models import audio_cpp_catalog as catalog, audio_cpp_runtime as runtime
from Models.STT import audio_cpp as stt
from Models.TTS import audio_cpp as tts


class CatalogTests(unittest.TestCase):
    def test_new_models_are_accepted_by_profile_validation(self):
        manager = settings.SettingsManager()
        manager.translate_settings["stt_type"] = "audio_cpp"
        self.assertEqual(manager.get_available_models(), list(stt.STT_MODELS))

    def test_published_files_have_immutable_urls_and_matching_hashes(self):
        for models in (catalog.STT_MODELS, catalog.TTS_MODELS):
            for name, definition in models.items():
                self.assertIn(definition["default_precision"], definition["variants"], name)
                for variant in definition["variants"].values():
                    for entry in (variant, *variant.get("additional_files", ())):
                        self.assertRegex(entry["urls"][0], r"/resolve/[a-f0-9]{40}/")
                        self.assertRegex(entry["checksum"], r"^[a-f0-9]{64}$")
                        self.assertEqual(entry["file_checksums"][entry["filename"]], entry["checksum"])
                        self.assertGreater(entry["size"], 0)

    def test_new_fixed_language_asr_models_validate_before_request(self):
        self.assertEqual(stt._language_for_model("Moonshine-Streaming-Tiny-GGUF", None), ("en", "en"))
        self.assertEqual(stt._language_for_model("Canary-180M-Flash-GGUF", "de-DE"), ("de", "de"))
        with self.assertRaises(ValueError):
            stt._language_for_model("Niagara-19M-Batch-English-GGUF", "de")
        self.assertEqual(stt.normalize_precision("float32", "Niagara-19M-Batch-English-GGUF"), "f32")
        self.assertEqual(stt.normalize_precision("q4_0", "Cohere-Transcribe-GGUF"), "q4_0")

    def test_cpu_packages_are_portable_and_version_pinned(self):
        for key, packages in runtime.RUNTIME_PACKAGES.items():
            for package in packages:
                self.assertIn("v" + runtime.AUDIO_CPP_VERSION, package["filename"])
                self.assertRegex(package["checksum"], r"^[0-9a-f]{64}$")
                if key[2] == "cpu":
                    self.assertIn("cpu-portable", package["filename"])

    def test_bundled_espeak_reaches_native_models_without_global_mutation(self):
        with mock.patch.object(runtime, "espeak_paths", return_value=("library", "data")), \
                mock.patch.object(runtime, "_normalized_system", return_value="windows"), \
                mock.patch.dict(runtime.os.environ, {}, clear=True):
            self.assertEqual(runtime._server_environment(None, "kokoro_tts"),
                             {"AUDIOCPP_ESPEAK_LIBRARY": "library", "AUDIOCPP_ESPEAK_DATA": "data"})
            self.assertNotIn("AUDIOCPP_ESPEAK_LIBRARY", runtime.os.environ)
        with mock.patch.object(tts, "espeak_paths", return_value=("library", "data")):
            self.assertEqual(tts._session_options("sanoTTS-de-GGUF", {}),
                             {"sanotts.espeak_library_path": "library", "sanotts.espeak_data_path": "data"})


if __name__ == "__main__":
    unittest.main()
