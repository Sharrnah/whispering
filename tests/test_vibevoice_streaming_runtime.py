import tempfile
import unittest

import torch

from Models.STT.vibevoice_streaming_runtime.configuration_vibevoice import VibeVoiceASRConfig
from Models.STT.vibevoice_streaming_runtime.modeling_vibevoice import VibeVoiceASRForConditionalGeneration


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_local_checkpoint_roundtrip_and_cached_decode_for_both_weight_tying_modes(self):
        encoder = dict(encoder_depths="1-1", encoder_n_filters=4, encoder_ratios=[2], vae_dim=4)
        for tied in (True, False):
            with self.subTest(tied=tied):
                cfg = VibeVoiceASRConfig(
                    acoustic_tokenizer_config=dict(encoder), semantic_tokenizer_config=dict(encoder),
                    decoder_config=dict(model_type="qwen2", hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                        vocab_size=32, tie_word_embeddings=tied),
                )
                cfg.decoder_config._attn_implementation = "sdpa"
                model = VibeVoiceASRForConditionalGeneration(cfg).eval()
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    loaded = VibeVoiceASRForConditionalGeneration.from_pretrained(
                        directory, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True,
                    ).eval()
                embeddings = loaded.get_input_embeddings()
                self.assertEqual(loaded.lm_head.weight.data_ptr() == embeddings.weight.data_ptr(), tied)
                with torch.inference_mode():
                    output = loaded(inputs_embeds=embeddings(torch.tensor([[1, 2, 3]])))
                    self.assertEqual(output.past_key_values.get_seq_length(), 3)
                    output = loaded(inputs_embeds=embeddings(torch.tensor([[4]])), past_key_values=output.past_key_values)
                    self.assertEqual(output.past_key_values.get_seq_length(), 4)
                    self.assertEqual(tuple(output.logits.shape), (1, 1, 32))
                    self.assertTrue(torch.isfinite(output.logits).all())
                    self.assertTrue(torch.isfinite(loaded.encode_speech(torch.zeros(1, 32))).all())

    def test_nonlocal_checkpoint_is_rejected_before_transformers_loading(self):
        with self.assertRaisesRegex(ValueError, "verified local"):
            VibeVoiceASRForConditionalGeneration.from_pretrained("microsoft/VibeVoice-ASR-Streaming-7B")


if __name__ == "__main__":
    unittest.main()
