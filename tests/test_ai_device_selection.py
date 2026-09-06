import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from Models.ai_device import (
    cuda_device_context,
    get_device,
    normalize_device_index,
    resolve_device,
    resolve_torch_device,
    split_ctranslate2_device,
)
from Models.TTS.tts_config import get_tts_device, get_tts_precision
from settings import SettingsManager


class AIDeviceSelectionTests(unittest.TestCase):
    def test_cuda_device_uses_selected_index(self):
        self.assertEqual(
            resolve_device("cuda", 2, cuda_available=True, cuda_device_count=4),
            "cuda:2",
        )

    def test_explicit_cuda_device_remains_compatible(self):
        self.assertEqual(
            resolve_device("cuda:3", 1, cuda_available=True, cuda_device_count=4),
            "cuda:3",
        )

    def test_unavailable_cuda_falls_back_to_cpu(self):
        self.assertEqual(
            resolve_device("cuda", 1, cuda_available=False, cuda_device_count=0),
            "cpu",
        )

    def test_out_of_range_cuda_index_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "detected 2 CUDA device"):
            resolve_device("cuda", 2, cuda_available=True, cuda_device_count=2)

    def test_ctranslate2_receives_separate_device_index(self):
        self.assertEqual(split_ctranslate2_device("cuda:2"), ("cuda", 2))
        self.assertEqual(split_ctranslate2_device("cpu"), ("cpu", 0))

    def test_invalid_negative_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_device_index(-1)

    def test_selected_cuda_device_becomes_the_context_default(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.device") as cuda_device:
                with cuda_device_context("cuda:2"):
                    pass
        cuda_device.assert_called_once_with("cuda:2")

    def test_explicit_settings_source_keeps_route_specific_device(self):
        class RouteSettings:
            values = {"ai_device": "cuda", "ai_device_index": 2}

            def GetOption(self, name):
                return self.values[name]

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.device_count", return_value=4):
                self.assertEqual(
                    get_device("ai_device", "ai_device_index", RouteSettings()),
                    "cuda:2",
                )

    def test_vulkan_device_is_not_forwarded_to_pytorch(self):
        self.assertEqual(
            resolve_torch_device(
                "vulkan",
                1,
                cuda_available=True,
                cuda_device_count=1,
            ),
            "cuda:0",
        )

    def test_native_device_falls_back_to_cpu_without_torch_accelerator(self):
        self.assertEqual(
            resolve_torch_device(
                "vulkan",
                1,
                cuda_available=False,
                cuda_device_count=0,
            ),
            "cpu",
        )

    def test_rocm_uses_pytorch_cuda_compatible_device_api(self):
        self.assertEqual(
            resolve_torch_device(
                "rocm",
                1,
                cuda_available=True,
                cuda_device_count=2,
            ),
            "cuda:1",
        )

    def test_metal_maps_to_pytorch_mps(self):
        self.assertEqual(
            resolve_torch_device("metal", 0, mps_available=True),
            "mps",
        )

    def test_shared_tts_helper_shields_torch_plugin_from_audio_cpp_backend(self):
        values = {"tts_ai_device": "vulkan", "tts_ai_device_index": 1}
        with mock.patch(
            "Models.ai_device.settings.GetOption",
            side_effect=values.get,
        ), mock.patch(
            "Models.ai_device._cuda_capabilities",
            return_value=(True, 1),
        ):
            self.assertEqual(get_tts_device(), "cuda:0")


class TTSPrecisionMigrationTests(unittest.TestCase):
    def test_legacy_adapter_precision_migrates_in_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "tts_type": "qwen3_tts",
                        "special_settings": {
                            "tts_qwen3_tts": {"precision": "bfloat16"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            manager = SettingsManager()
            manager.load_yaml(profile_path)
            self.assertEqual(manager.get_option("tts_precision"), "bfloat16")

    def test_new_top_level_precision_wins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "tts_type": "qwen3_tts",
                        "tts_precision": "float32",
                        "special_settings": {
                            "tts_qwen3_tts": {"precision": "bfloat16"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            manager = SettingsManager()
            manager.load_yaml(profile_path)
            self.assertEqual(manager.get_option("tts_precision"), "float32")

    def test_auto_sentinel_keeps_legacy_concrete_precision(self):
        with mock.patch("Models.TTS.tts_config.settings.GetOption", return_value="auto"):
            self.assertEqual(get_tts_precision("float16"), "float16")
            self.assertEqual(get_tts_precision("auto"), "auto")


if __name__ == "__main__":
    unittest.main()
