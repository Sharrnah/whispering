import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.TTS import zonos2_tts


class Zonos2DownloadTests(unittest.TestCase):
    def test_hosted_manifests_are_offline_and_complete(self):
        expected_files = {
            "zonos2-bf16": {"zonos2-bf16.safetensors"},
            "zonos2-fp8-mixed": {"zonos2-fp8-mixed.safetensors"},
            "dac_44khz": {
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            },
            "speaker_encoder": {
                "config.json",
                "configuration_ecapa_tdnn.py",
                "feature_extraction_ecapa_tdnn.py",
                "model.safetensors",
                "modeling_ecapa_tdnn.py",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "tokenizer_ecapa_tdnn.py",
            },
            "emotion_directions": {
                "angry.npy",
                "arousal.npy",
                "calibration.json",
                "happy.npy",
                "manifest.json",
                "sad.npy",
                "surprised.npy",
                "valence.npy",
            },
        }
        placeholder_entries = {
            "zonos2-bf16",
            "zonos2-fp8-mixed",
            "speaker_encoder",
            "emotion_directions",
        }

        self.assertEqual(set(zonos2_tts.TTS_MODEL_LINKS), set(expected_files))
        for name, entry in zonos2_tts.TTS_MODEL_LINKS.items():
            self.assertEqual(set(entry["file_checksums"]), expected_files[name])
            self.assertEqual(len(entry["urls"]), 3)
            self.assertTrue(all("huggingface" not in url.lower() for url in entry["urls"]))
            self.assertTrue(
                all(
                    len(checksum) == 64 and checksum != "0" * 64
                    for checksum in entry["file_checksums"].values()
                )
            )
            if name in placeholder_entries:
                self.assertEqual(entry["checksum"], "0" * 64)
            else:
                self.assertNotEqual(entry["checksum"], "0" * 64)

    def test_placeholder_archive_hash_prevents_network_request(self):
        adapter = object.__new__(zonos2_tts.Zonos2TTS)
        adapter.download_state = {"is_downloading": False}
        download = mock.Mock()
        fake_downloader = types.SimpleNamespace(
            model_needs_download=mock.Mock(return_value=True),
            download_model=download,
        )

        with mock.patch.dict(sys.modules, {"downloader": fake_downloader}):
            with self.assertRaisesRegex(
                RuntimeError,
                "zonos2-bf16.zip is not currently available",
            ):
                adapter.download_model("zonos2-bf16")

        download.assert_not_called()

    def test_verified_local_files_do_not_require_archive_hash(self):
        adapter = object.__new__(zonos2_tts.Zonos2TTS)
        adapter.download_state = {"is_downloading": False}
        download = mock.Mock()
        fake_downloader = types.SimpleNamespace(
            model_needs_download=mock.Mock(return_value=False),
            download_model=download,
        )

        with mock.patch.dict(sys.modules, {"downloader": fake_downloader}):
            self.assertTrue(adapter.download_model("zonos2-bf16"))

        download.assert_not_called()

    def test_load_installs_selected_and_shared_components(self):
        adapter = object.__new__(zonos2_tts.Zonos2TTS)
        adapter.bundle = None
        adapter.compute_device = None
        adapter.last_speaker_audio = "stale"
        adapter.last_speaker_embedding = object()
        adapter._ensure_special_settings = mock.Mock()
        adapter.set_compute_device = mock.Mock()
        adapter._get_model_name = mock.Mock(return_value="zonos2-fp8-mixed")
        adapter.download_model = mock.Mock(return_value=True)
        adapter.download_voices = mock.Mock(return_value=True)
        adapter.special_settings = {"attention": "auto"}
        loaded_bundle = types.SimpleNamespace(attention_backend="sdpa")

        with mock.patch.object(zonos2_tts.settings, "GetOption", return_value="cpu"):
            with mock.patch.object(
                zonos2_tts,
                "load_bundle",
                return_value=loaded_bundle,
            ) as load_bundle:
                adapter.load()

        self.assertEqual(
            adapter.download_model.call_args_list,
            [
                mock.call("zonos2-fp8-mixed"),
                mock.call("dac_44khz"),
                mock.call("speaker_encoder"),
                mock.call("emotion_directions"),
            ],
        )
        adapter.download_voices.assert_called_once_with()
        load_bundle.assert_called_once_with(
            zonos2_tts.MODEL_CACHE_PATH,
            None,
            model_name="zonos2-fp8-mixed",
            attention="auto",
        )

    def test_model_directories_match_runtime_layout(self):
        expected = {
            "zonos2-bf16": "bf16",
            "zonos2-fp8-mixed": "fp8",
            "dac_44khz": "dac_44khz",
            "speaker_encoder": "speaker_encoder",
            "emotion_directions": "emotion_directions",
        }
        for name, relative_path in expected.items():
            actual = zonos2_tts.Zonos2TTS._model_directory(name)
            self.assertEqual(actual, zonos2_tts.MODEL_CACHE_PATH / relative_path)
            self.assertEqual(os.fspath(actual.name), relative_path)


if __name__ == "__main__":
    unittest.main()
