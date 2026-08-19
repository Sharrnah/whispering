import unittest
from unittest import mock

import torch

from Models import transformers_attention as attention


class TransformersAttentionTests(unittest.TestCase):
    def test_compatible_cuda_half_precision_prefers_flash_attention_2(self):
        with mock.patch.object(
            attention,
            "is_flash_attn_2_available",
            return_value=True,
        ), mock.patch.object(
            attention.torch.cuda,
            "get_device_capability",
            return_value=(8, 6),
        ):
            result = attention.get_preferred_attention_implementation(
                torch.device("cuda"),
                torch.float16,
            )

        self.assertEqual(result, attention.FLASH_ATTENTION_2)

    def test_cpu_float32_missing_package_and_turing_use_sdpa(self):
        self.assertEqual(
            attention.get_preferred_attention_implementation("cpu", torch.float16),
            attention.SDPA,
        )
        self.assertEqual(
            attention.get_preferred_attention_implementation("cuda", torch.float32),
            attention.SDPA,
        )

        with mock.patch.object(
            attention,
            "is_flash_attn_2_available",
            return_value=False,
        ):
            self.assertEqual(
                attention.get_preferred_attention_implementation(
                    "cuda",
                    torch.float16,
                ),
                attention.SDPA,
            )

        with mock.patch.object(
            attention,
            "is_flash_attn_2_available",
            return_value=True,
        ), mock.patch.object(
            attention.torch.cuda,
            "get_device_capability",
            return_value=(7, 5),
        ):
            self.assertEqual(
                attention.get_preferred_attention_implementation(
                    "cuda",
                    torch.float16,
                ),
                attention.SDPA,
            )

    def test_bfloat16_requires_cuda_bfloat16_support(self):
        with mock.patch.object(
            attention,
            "is_flash_attn_2_available",
            return_value=True,
        ), mock.patch.object(
            attention.torch.cuda,
            "get_device_capability",
            return_value=(8, 0),
        ), mock.patch.object(
            attention.torch.cuda,
            "is_bf16_supported",
            return_value=False,
        ):
            result = attention.get_preferred_attention_implementation(
                "cuda",
                torch.bfloat16,
            )

        self.assertEqual(result, attention.SDPA)

    def test_runtime_flash_failure_retries_once_with_sdpa(self):
        attempts = []
        fallback_model = object()

        def loader(implementation):
            attempts.append(implementation)
            if implementation == attention.FLASH_ATTENTION_2:
                raise RuntimeError("incompatible flash kernel")
            return fallback_model

        with mock.patch.object(attention, "_clear_accelerator_cache") as clear_cache:
            model, implementation = attention.load_with_attention_fallback(
                loader,
                attention.FLASH_ATTENTION_2,
                "test model",
            )

        self.assertIs(model, fallback_model)
        self.assertEqual(implementation, attention.SDPA)
        self.assertEqual(attempts, [attention.FLASH_ATTENTION_2, attention.SDPA])
        clear_cache.assert_called_once_with()

    def test_sdpa_load_failures_are_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "broken model"):
            attention.load_with_attention_fallback(
                lambda implementation: (_ for _ in ()).throw(RuntimeError("broken model")),
                attention.SDPA,
                "test model",
            )


if __name__ == "__main__":
    unittest.main()
