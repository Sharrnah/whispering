# Adapted from microsoft/VibeVoice@1541f590c7099820f10ea012f48d2399282df69f.
# Copyright (c) Microsoft Corporation. MIT License; see LICENSE.
"""Inference-only bridge to the application's installed Transformers runtime."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .configuration_vibevoice import VibeVoiceASRConfig
from .modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
)


class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        return self.fc2(self.norm(self.fc1(features)))


class VibeVoiceASRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = Qwen2Model(config.decoder_config)
        config.acoustic_tokenizer_config._attn_implementation = "eager"
        config.semantic_tokenizer_config._attn_implementation = "eager"
        # Instantiate our own classes directly. Registering the upstream model
        # names globally would collide with other VibeVoice integrations.
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(config.semantic_tokenizer_config)
        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, config.decoder_config.hidden_size)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, config.decoder_config.hidden_size)


class VibeVoiceASRForConditionalGeneration(PreTrainedModel):
    config_class = VibeVoiceASRConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _no_split_modules = ["Qwen2DecoderLayer", "TokenizerEncoder"]

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceASRModel(config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if not Path(pretrained_model_name_or_path).is_dir():
            raise ValueError("VibeVoice streaming requires a verified local checkpoint directory.")
        kwargs["local_files_only"] = True
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_input_embeddings(self, embeddings):
        self.model.language_model.set_input_embeddings(embeddings)

    def forward(self, inputs_embeds, past_key_values=None, **kwargs):
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds, past_key_values=past_key_values,
            use_cache=True, return_dict=True,
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(outputs.last_hidden_state[:, -1:]),
            past_key_values=outputs.past_key_values,
        )

    @torch.inference_mode()
    def encode_speech(self, speech_tensors):
        # The trained acoustic encoder samples its fixed-variance latents.
        # Preserve this even though text decoding itself is greedy.
        audio = speech_tensors.to(dtype=self.dtype).unsqueeze(1)
        acoustic = self.model.acoustic_tokenizer.encode(audio)
        acoustic = acoustic.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
        semantic = self.model.semantic_tokenizer.encode(audio).mean
        return self.model.acoustic_connector(acoustic) + self.model.semantic_connector(semantic)

    @torch.inference_mode()
    def init_streaming_state(self, tokenizer, context_info: str = None):
        """Streaming model: build the initial KV cache and cached embeddings."""
        device = next(self.parameters()).device
        embed_tokens = self.get_input_embeddings()

        keys_str = "speaker, content"
        if context_info:
            prompt_text = (
                "You are a helpful assistant that transcribes audio input into text output. "
                f"Please transcribe the following audios streamingly with these keys: {keys_str} "
                f"and extra info: {context_info}\n"
            )
        else:
            prompt_text = (
                "You are a helpful assistant that transcribes audio input into text output. "
                f"Please transcribe the following audios streamingly with these keys: {keys_str}\n"
            )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_embeds = embed_tokens(prompt_tensor)

        outputs = self(
            inputs_embeds=prompt_embeds,
            use_cache=True,
            return_dict=True,
        )

        return {
            "past_key_values": outputs.past_key_values,
            "sp_start_embed": embed_tokens(torch.tensor([[tokenizer.speech_start_id]], device=device)),
            "sp_end_embed": embed_tokens(torch.tensor([[tokenizer.speech_end_id]], device=device)),
            "text_chunk_end_id": tokenizer.text_chunk_end_id,
            "eos_id": tokenizer.eos_token_id,
            "embed_tokens": embed_tokens,
        }


    @torch.inference_mode()
    def streaming_generate_step(
        self,
        audio_features: torch.FloatTensor,
        streaming_state: dict,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        on_first_token: Optional[Callable[[int], None]] = None,
    ) -> Tuple[str, dict]:
        """Streaming model: advance one audio chunk against the running KV cache."""
        device = next(self.parameters()).device
        past_key_values = streaming_state["past_key_values"]
        sp_start_embed = streaming_state["sp_start_embed"]
        sp_end_embed = streaming_state["sp_end_embed"]
        text_chunk_end_id = streaming_state["text_chunk_end_id"]
        eos_id = streaming_state["eos_id"]
        embed_tokens = streaming_state["embed_tokens"]

        audio_embeds = torch.cat([sp_start_embed, audio_features, sp_end_embed], dim=1)

        outputs = self(
            inputs_embeds=audio_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]

        chunk_tokens = []
        for _ in range(max_new_tokens):
            if temperature <= 0:
                next_token_id = torch.argmax(next_logits[:, -1, :], dim=-1).item()
            else:
                probs = torch.nn.functional.softmax(next_logits[:, -1, :] / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

            if next_token_id == text_chunk_end_id or next_token_id == eos_id:
                break

            chunk_tokens.append(next_token_id)
            if len(chunk_tokens) == 1 and on_first_token is not None:
                on_first_token(next_token_id)

            next_embed = embed_tokens(torch.tensor([[next_token_id]], device=device))
            outputs = self(
                inputs_embeds=next_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits
        else:
            raise RuntimeError(
                f"VibeVoice did not finish an audio chunk within {max_new_tokens} text tokens. "
                "The stream was reset; increase stt_vibevoice_streaming.max_new_tokens if needed."
            )

        tce_embed = embed_tokens(torch.tensor([[text_chunk_end_id]], device=device))
        outputs = self(
            inputs_embeds=tce_embed,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        streaming_state["past_key_values"] = past_key_values

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        for _st in ['<|text_chunk_end|>', '<|object_ref_start|>', '<|object_ref_end|>',
                    '<|box_start|>', '<|speech_start|>', '<|speech_end|>', '<|speech_pad|>']:
            chunk_text = chunk_text.replace(_st, '')

        return chunk_text, streaming_state

