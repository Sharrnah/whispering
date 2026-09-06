"""Text-to-speech families served by the native audio.cpp GGUF runtime."""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import audio_tools
import settings
from Models.Singleton import SingletonMeta
from Models.audio_cpp_runtime import AudioCppServer, normalize_backend_device
from Models.TTS.tts_config import get_tts_precision


DEFAULT_MODEL = "Supertonic-3-GGUF"
SAMPLE_RATE = 44_100
MODEL_CACHE_PATH = Path.cwd() / ".cache" / "audio.cpp" / "models"
VOICES_PATH = Path.cwd() / ".cache" / "chatterbox-tts-cache" / "voices"
GGUF_REVISION = "78f9d27aa214792b77256affe774eea57e35b9ae"
GGUF_REPOSITORY_ROOT = (
    "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/"
    f"{GGUF_REVISION}"
)
SPECIAL_SETTINGS_NAME = "tts_audio_cpp"

SUPERTONIC_LANGUAGES = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
    "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
}
CONFUCIUS_LANGUAGES = {
    "zh", "en", "ja", "ko", "de", "fr", "es", "id", "it", "th", "pt",
    "ru", "ms", "vi",
}
MAGPIE_LANGUAGES = {
    "ar-ae", "ar-msa", "ar-sa", "de", "en", "es", "fr", "hi", "it", "ko",
    "pt-br", "vi", "zh",
}
BUILT_IN_VOICES = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")
MAGPIE_VOICES = ("Aria", "Jason", "John", "Leo", "Sofia")
RATE_MULTIPLIERS = {
    "x-slow": 0.65,
    "slow": 0.82,
    "medium": 1.05,
    "fast": 1.22,
    "x-fast": 1.4,
}


def _variant(path: str, sha256: str, size: int) -> dict:
    filename = Path(path).name
    return {
        "urls": [f"{GGUF_REPOSITORY_ROOT}/{path}"],
        "checksum": sha256,
        "file_checksums": {filename: sha256},
        "filename": filename,
        "size": size,
    }


TTS_MODELS = {
    "Supertonic-3-GGUF": {
        "family": "supertonic", "group": "Preset voices",
        "description": "Fast multilingual preset voices; no voice cloning",
        "sample_rate": 44_100, "task": "tts", "streaming": True,
        "settings_key": "supertonic", "default_precision": "orig",
        "variants": {
            # The published q8_0 has no useful size/quality distinction yet.
            "orig": _variant(
                "Supertonic-3-GGUF/supertonic-3-orig.gguf",
                "af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662", 454_072_836),
            "f16": _variant(
                "Supertonic-3-GGUF/supertonic-3-f16.gguf",
                "b312b57797d40ac5c09d915893dbdbaf6405b7dc043f544776c5c95712dff88c", 312_784_196),
        },
    },
    "Confucius4-TTS-GGUF": {
        "family": "confucius4_tts", "group": "Voice cloning",
        "description": "Experimental multilingual cloning; reference WAV required",
        "sample_rate": 22_050, "task": "clon", "streaming": True,
        "requires_reference": True, "settings_key": "confucius4_tts",
        "default_precision": "orig",
        "variants": {
            "orig": _variant(
                "Confucius4-TTS-GGUF/confucius4-tts-orig.gguf",
                "eec4ab3fae3cda1e8ec2b96599442def30dbc980ea817f42b94e073158536244", 8_192_757_760),
        },
    },
    "DotTTS-SOAR-GGUF": {
        "family": "dots_tts", "group": "Voice cloning",
        "description": "DotTTS SOAR multilingual synthesis and cloning",
        "sample_rate": 24_000, "task": "tts", "streaming": True,
        "settings_key": "dots_tts", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant(
                "DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf",
                "0633a6b4b705accee3858ffb129f403dc5aa64e634a80f713f4bde6f002fc2f0", 2_962_749_632),
            "bf16": _variant(
                "DotTTS-SOAR-GGUF/dots-tts-soar-bf16.gguf",
                "a77cc9e0881d50064990dba19b01e7faec8610b5ba2fdf563a86699006fb7a51", 4_788_730_464),
        },
    },
    "DotTTS-MeanFlow-GGUF": {
        "family": "dots_tts", "group": "Voice cloning",
        "description": "DotTTS MeanFlow multilingual synthesis and cloning",
        "sample_rate": 24_000, "task": "tts", "streaming": True,
        "settings_key": "dots_tts", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant(
                "DotTTS-MF-GGUF/dots-tts-mf-q8_0.gguf",
                "062c5e040e0ec9a2d31dc1209d70233ed2bb132e9ead5cea84af420644079053", 2_965_375_840),
            "bf16": _variant(
                "DotTTS-MF-GGUF/dots-tts-mf-bf16.gguf",
                "aedfa0415229fbb5d0b59ab4fe0327513720bacfc3f7705cca3c2d6315f911af", 4_791_356_672),
        },
    },
    "DotTTS-Edit-GGUF": {
        "family": "dots_tts", "group": "Speech editing",
        "description": "Edit speech from a source WAV using text instructions",
        "sample_rate": 24_000, "task": "tts", "streaming": False,
        "settings_key": "dots_tts_edit", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant(
                "DotTTS-Edit-GGUF/dots-tts-edit-q8_0.gguf",
                "5f2df7c42849feaaa4b3f0808b72804132bee6e6314563125973b15fc1e3a122", 2_962_805_408),
            "bf16": _variant(
                "DotTTS-Edit-GGUF/dots-tts-edit-bf16.gguf",
                "3f4711edcb28a1688462eaba75711bf073514fbd5336a1141b88430d4c6a884f", 4_788_786_240),
        },
    },
    "IndexTTS2-GGUF": {
        "family": "index_tts2", "group": "Voice cloning and emotion",
        "description": "IndexTTS 2 cloning; English and Chinese",
        "sample_rate": 22_050, "task": "clon", "streaming": False,
        "requires_reference": True, "settings_key": "index_tts2",
        "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("IndexTTS2-GGUF/index-tts2-q8_0.gguf", "f9be73ac5b6cbdc5f603f04dcb6c9fe37b3a301a365a1ea8d580d3ef7ba22e8e", 3_633_888_608),
            "f16": _variant("IndexTTS2-GGUF/index-tts2-f16.gguf", "0f7b95d3d32e18bf9352912a7b21a0bd4a8406853a0aaa2c9180f77389c21523", 4_646_898_304),
            "orig": _variant("IndexTTS2-GGUF/index-tts2-orig.gguf", "c96cb948339b7169b7240f43dbd42ad6354ae3d0a22465e292ec8e090c338e16", 8_084_552_000),
        },
    },
    "IndexTTS2.5-GGUF": {
        "family": "index_tts2", "group": "Voice cloning and emotion",
        "description": "IndexTTS 2.5 multilingual cloning and emotion control",
        "sample_rate": 22_050, "task": "clon", "streaming": False,
        "requires_reference": True, "settings_key": "index_tts2",
        "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf", "5e827b2072042e4a1b21ccf24a5cb4f71cb1011403067a0a9b039311d8b38628", 3_502_955_328),
            "f16": _variant("IndexTTS2.5-GGUF/index-tts2_5-f16.gguf", "87bed9b82fc8f22119a1a1042332091016c28e37f29b0e93343ccdbfa76ef66a", 4_547_355_072),
            "orig": _variant("IndexTTS2.5-GGUF/index-tts2_5-orig.gguf", "2c736df55a306155df4e607f49fe963cf2192fc2193ad449a4bdffb15e52b93a", 7_885_093_440),
        },
    },
    "MagpieTTS-Multilingual-357M-GGUF": {
        "family": "magpie_tts", "group": "Preset voices",
        "description": "Small multilingual model with five built-in voices",
        "sample_rate": 22_050, "task": "tts", "streaming": False,
        "settings_key": "magpie_tts", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("MagpieTTS-Multilingual-357M-GGUF/magpie-tts-multilingual-357m-q8_0.gguf", "c762503a80f9af75db33379763b923c5c8a00ca79374e1e0d4051e6d68151377", 1_562_142_912),
            "orig": _variant("MagpieTTS-Multilingual-357M-GGUF/magpie-tts-multilingual-357m-orig.gguf", "3ed26e41e19ec0b47249e31859fe2f29b2c9a5b0cd6f518c0c27782de575096f", 1_912_137_280),
        },
    },
    "OmniVoice-GGUF": {
        "family": "omnivoice", "group": "Cloning and voice design",
        "description": "600+ languages; optional clone reference or voice instruction",
        "sample_rate": 24_000, "task": "tts", "streaming": True,
        "settings_key": "omnivoice", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("OmniVoice-GGUF/omnivoice-q8_0.gguf", "2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b", 1_350_288_416),
            "f16": _variant("OmniVoice-GGUF/omnivoice-f16.gguf", "7421b6c387351d2cdce7eb8b931fa279739d628ef25acd4ff2062a70e6e1ab63", 1_639_548_768),
            "bf16": _variant("OmniVoice-GGUF/omnivoice-bf16.gguf", "7301a41a0a5a2cb04430c1831a705e3906bfca51562c5cdb2f30a90d505a819c", 1_639_548_640),
        },
    },
    "VoxCPM1-0.5B-GGUF": {
        "family": "voxcpm1", "group": "Cloning and text to speech",
        "description": "Compact 0.5B model; optional short voice reference",
        "sample_rate": 16_000, "task": "tts", "streaming": True,
        "settings_key": "voxcpm1", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf", "01210319c5ce617613c9d1c38e34649f7479e98a60d51b719f7df82970658241", 847_888_032),
        },
    },
    "VoxCPM2-GGUF": {
        "family": "voxcpm2", "group": "Cloning and voice design",
        "description": "2B voice cloning, voice design, and streaming",
        "sample_rate": 48_000, "task": "tts", "streaming": True,
        "settings_key": "voxcpm2", "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("VoxCPM2-GGUF/voxcpm2-q8_0.gguf", "c8e01ab4416011e12a28f24ede298a1aa5ce64b43f8e8aaad53b1e2fe7c96432", 2_955_000_480),
            "bf16": _variant("VoxCPM2-GGUF/voxcpm2-bf16.gguf", "0c0cd2515c7a1145c0ddba06c871724656c36cb53ee3b0118116b34cb4153699", 4_772_288_288),
            "orig": _variant("VoxCPM2-GGUF/voxcpm2-orig.gguf", "13181e01e15d555c9eae7779626eb395c69334f15e948c9ea265ddf37ab50b16", 4_960_716_192),
        },
    },
}

MODEL_VARIANTS = {name: model["variants"] for name, model in TTS_MODELS.items()}
MODEL_ALIASES = {
    "Supertonic-3": DEFAULT_MODEL, "Supertonic 3": DEFAULT_MODEL,
    "confucius4_tts": "Confucius4-TTS-GGUF", "dots_tts": "DotTTS-SOAR-GGUF",
    "dots_tts_soar": "DotTTS-SOAR-GGUF", "dots_tts_meanflow": "DotTTS-MeanFlow-GGUF",
    "dots_tts_edit": "DotTTS-Edit-GGUF", "index_tts2": "IndexTTS2-GGUF",
    "index_tts2_5": "IndexTTS2.5-GGUF", "magpie_tts": "MagpieTTS-Multilingual-357M-GGUF",
    "omnivoice": "OmniVoice-GGUF", "OmniVoice": "OmniVoice-GGUF",
    "voxcpm": "VoxCPM2-GGUF", "voxcpm1": "VoxCPM1-0.5B-GGUF", "voxcpm2": "VoxCPM2-GGUF",
}
DOWNLOAD_STATES = {
    model: {precision: {"is_downloading": False} for precision in definition["variants"]}
    for model, definition in TTS_MODELS.items()
}


TTS_SETTINGS_DEFAULTS = {
    "supertonic": {
        "language": "auto", "num_inference_steps": 8, "speaking_rate": 1.05,
        "seed": 1234, "text_chunk_size": 300, "text_chunk_mode": "default",
        "weight_type": "native", "style_cache_slots": 4,
    },
    "confucius4_tts": {
        "language": "auto", "temperature": 0.8, "top_p": 0.8, "top_k": 30,
        "num_beams": 3, "repetition_penalty": 10.0, "max_tokens": 1520,
        "num_inference_steps": 25, "guidance_scale": 0.7, "text_chunk_size": 80,
        "text_chunk_mode": "default", "cross_fade_duration_sec": 0.3,
        "edge_fade_duration_sec": 0.1, "edge_pad_duration_sec": 0.1,
        "seed": 1234, "mem_saver": False, "graph_arena_mb": 512,
        "weight_context_mb": 1024, "weight_type": "native",
        "conv_weight_type": "native", "reference_cache_slots": 1,
    },
    "dots_tts": {
        "language": "none", "reference_text": "", "reference_duration_sec": 0.0,
        "template_name": "tts", "instruction": "", "use_xvector": "auto",
        "num_inference_steps": 10, "guidance_scale": 1.2, "speaker_scale": 1.5,
        "sampler_mode": "euler", "max_tokens": 500, "text_chunk_size": 320,
        "text_chunk_mode": "tag_aware", "vocoder_merge_steps": 4, "seed": 42,
        "weight_type": "native", "speaker_encoder_weight_type": "native",
        "codec_weight_type": "native", "patch_encoder_weight_type": "native",
        "llm_weight_type": "native", "flow_weight_type": "native",
        "codec_conv_weight_type": "native", "reference_cache_slots": 4, "mem_saver": False,
    },
    "dots_tts_edit": {
        "language": "none", "source_audio": "", "source_text": "", "target_text": "",
        "instruction": "", "use_xvector": "auto", "num_inference_steps": 10,
        "guidance_scale": 1.2, "speaker_scale": 1.5, "sampler_mode": "euler",
        "max_tokens": 500, "text_chunk_size": 320, "text_chunk_mode": "tag_aware",
        "vocoder_merge_steps": 4, "seed": 42, "weight_type": "native",
        "speaker_encoder_weight_type": "native", "codec_weight_type": "native",
        "patch_encoder_weight_type": "native", "llm_weight_type": "native",
        "flow_weight_type": "native", "codec_conv_weight_type": "native",
        "reference_cache_slots": 4, "mem_saver": False,
    },
    "index_tts2": {
        "language": "auto", "emotion_audio": "", "emotion_text": "", "emotion_alpha": 1.0,
        "emotion_happy": 0.0, "emotion_angry": 0.0, "emotion_sad": 0.0,
        "emotion_afraid": 0.0, "emotion_disgusted": 0.0, "emotion_melancholic": 0.0,
        "emotion_surprised": 0.0, "emotion_calm": 0.0, "use_emotion_text": False,
        "use_random_emotion": False, "interval_silence_ms": 200, "duration_factor": 1.0,
        "text_chunk_size": 0, "text_chunk_mode": "default", "max_tokens": 1500,
        "temperature": 0.8, "top_p": 0.8, "top_k": 30, "repetition_penalty": 10.0,
        "do_sample": True, "length_penalty": 0.0, "num_beams": 3, "mem_saver": False,
        "seed": -1,
        "weight_type": "native", "conv_weight_type": "native", "speaker_cache_slots": 1,
        "emotion_cache_slots": 1, "emotion_text_cache_slots": 1, "gpt_graph_arena_mb": 0,
        "s2mel_graph_arena_mb": 0, "reference_graph_arena_mb": 0,
        "emotion_text_prefill_graph_arena_mb": 0, "emotion_text_decode_graph_arena_mb": 0,
        "emotion_text_max_tokens": 256, "weight_context_mb": 32,
    },
    "magpie_tts": {
        "language": "auto", "voice_id": "Aria", "temperature": 0.6, "top_k": 80,
        "guidance_scale": 2.5, "max_tokens": 500, "text_chunk_size": 300,
        "text_chunk_mode": "default", "seed": 0, "graph_arena_mb": 1024,
        "weight_context_mb": 2048, "weight_type": "native", "conv_weight_type": "native",
    },
    "omnivoice": {
        "language": "auto", "reference_text": "", "voice_instruction": "",
        "text_chunk_size": 160, "text_chunk_mode": "tag_aware", "num_inference_steps": 32,
        "guidance_scale": 2.0, "speed": 1.0, "duration": 0.0, "t_shift": 0.1,
        "denoise": True, "preprocess_prompt": True, "postprocess_output": True,
        "layer_penalty_factor": 5.0, "position_temperature": 5.0,
        "class_temperature": 0.0, "audio_chunk_duration": 15.0,
        "audio_chunk_threshold": 30.0, "seed": -1, "mem_saver": False, "perf_mode": "off",
        "generator_weight_type": "native", "audio_tokenizer_weight_type": "native",
        "audio_tokenizer_graph_arena_mb": 0, "generator_prefill_graph_arena_mb": 0,
        "generator_decode_graph_arena_mb": 0, "audio_tokenizer_weight_context_mb": 0,
        "generator_weight_context_mb": 0,
    },
    "voxcpm1": {
        "reference_text": "", "text_chunk_size": 2048, "text_chunk_mode": "tag_aware",
        "seed": 1234, "max_tokens": 4096, "min_tokens": 2, "num_inference_steps": 10,
        "guidance_scale": 2.0, "retry_badcase": True, "retry_badcase_max_times": 3,
        "retry_badcase_ratio_threshold": 6.0, "cfm_noise_file": "",
        "mem_saver": False, "prompt_cache_slots": 1,
        "weight_type": "native", "audiovae_weight_type": "native",
        "weight_context_mb": 0, "text_embedding_graph_context_mb": 0,
        "lm_step_graph_context_mb": 0, "projection_graph_context_mb": 0,
        "local_encoder_graph_context_mb": 0, "dit_graph_context_mb": 0,
        "audiovae_weight_context_mb": 0, "audiovae_graph_context_mb": 0,
        "audiovae_encoder_graph_context_mb": 0, "audiovae_latent_capacity": 0,
        "audiovae_encoder_sample_capacity": 0,
    },
    "voxcpm2": {
        "reference_text": "", "voice_instruction": "", "text_chunk_size": 2048,
        "text_chunk_mode": "tag_aware", "seed": 1234, "max_tokens": 4096, "min_tokens": 2,
        "num_inference_steps": 10, "guidance_scale": 2.0, "retry_badcase": True,
        "retry_badcase_max_times": 3, "retry_badcase_ratio_threshold": 6.0,
        "cfm_noise_file": "", "mem_saver": True,
        "prompt_cache_slots": 1, "weight_type": "native", "audiovae_weight_type": "native",
        "weight_context_mb": 0, "text_embedding_graph_context_mb": 0,
        "lm_step_graph_context_mb": 0, "projection_graph_context_mb": 0,
        "local_encoder_graph_context_mb": 0, "dit_graph_context_mb": 0,
        "audiovae_weight_context_mb": 0, "audiovae_graph_context_mb": 0,
        "audiovae_encoder_graph_context_mb": 0, "audiovae_latent_capacity": 0,
        "audiovae_encoder_sample_capacity": 0,
    },
}

REQUEST_KEYS = {
    "supertonic": ("num_inference_steps", "speaking_rate", "seed", "text_chunk_size", "text_chunk_mode"),
    "confucius4_tts": ("temperature", "top_p", "top_k", "num_beams", "repetition_penalty", "max_tokens", "num_inference_steps", "guidance_scale", "text_chunk_size", "text_chunk_mode", "cross_fade_duration_sec", "edge_fade_duration_sec", "edge_pad_duration_sec", "seed"),
    "dots_tts": ("reference_duration_sec", "template_name", "instruction", "use_xvector", "num_inference_steps", "guidance_scale", "speaker_scale", "sampler_mode", "max_tokens", "text_chunk_size", "text_chunk_mode", "vocoder_merge_steps", "seed"),
    "dots_tts_edit": ("source_text", "target_text", "instruction", "use_xvector", "num_inference_steps", "guidance_scale", "speaker_scale", "sampler_mode", "max_tokens", "text_chunk_size", "text_chunk_mode", "vocoder_merge_steps", "seed"),
    "index_tts2": ("emotion_text", "emotion_alpha", "use_emotion_text", "use_random_emotion", "interval_silence_ms", "duration_factor", "text_chunk_size", "text_chunk_mode", "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "do_sample", "length_penalty", "num_beams", "seed"),
    "magpie_tts": ("voice_id", "temperature", "top_k", "guidance_scale", "max_tokens", "text_chunk_size", "text_chunk_mode", "seed"),
    "omnivoice": ("text_chunk_size", "text_chunk_mode", "num_inference_steps", "guidance_scale", "speed", "duration", "t_shift", "denoise", "preprocess_prompt", "postprocess_output", "layer_penalty_factor", "position_temperature", "class_temperature", "audio_chunk_duration", "audio_chunk_threshold", "seed"),
    "voxcpm1": ("text_chunk_size", "text_chunk_mode", "seed", "max_tokens", "min_tokens", "num_inference_steps", "guidance_scale", "retry_badcase", "retry_badcase_max_times", "retry_badcase_ratio_threshold", "cfm_noise_file"),
    "voxcpm2": ("text_chunk_size", "text_chunk_mode", "seed", "max_tokens", "min_tokens", "num_inference_steps", "guidance_scale", "retry_badcase", "retry_badcase_max_times", "retry_badcase_ratio_threshold", "cfm_noise_file"),
}

SESSION_KEYS = {
    "supertonic": ("weight_type", "style_cache_slots"),
    "confucius4_tts": ("mem_saver", "graph_arena_mb", "weight_context_mb", "weight_type", "conv_weight_type", "reference_cache_slots"),
    "dots_tts": ("weight_type", "speaker_encoder_weight_type", "codec_weight_type", "patch_encoder_weight_type", "llm_weight_type", "flow_weight_type", "codec_conv_weight_type", "reference_cache_slots", "mem_saver"),
    "dots_tts_edit": ("weight_type", "speaker_encoder_weight_type", "codec_weight_type", "patch_encoder_weight_type", "llm_weight_type", "flow_weight_type", "codec_conv_weight_type", "reference_cache_slots", "mem_saver"),
    "index_tts2": ("mem_saver", "weight_type", "conv_weight_type", "speaker_cache_slots", "emotion_cache_slots", "emotion_text_cache_slots", "gpt_graph_arena_mb", "s2mel_graph_arena_mb", "reference_graph_arena_mb", "emotion_text_prefill_graph_arena_mb", "emotion_text_decode_graph_arena_mb", "emotion_text_max_tokens", "weight_context_mb"),
    "magpie_tts": ("graph_arena_mb", "weight_context_mb", "weight_type", "conv_weight_type"),
    "omnivoice": ("mem_saver", "perf_mode", "generator_weight_type", "audio_tokenizer_weight_type", "audio_tokenizer_graph_arena_mb", "generator_prefill_graph_arena_mb", "generator_decode_graph_arena_mb", "audio_tokenizer_weight_context_mb", "generator_weight_context_mb"),
    "voxcpm1": ("mem_saver", "prompt_cache_slots", "weight_type", "audiovae_weight_type", "weight_context_mb", "text_embedding_graph_context_mb", "lm_step_graph_context_mb", "projection_graph_context_mb", "local_encoder_graph_context_mb", "dit_graph_context_mb", "audiovae_weight_context_mb", "audiovae_graph_context_mb", "audiovae_encoder_graph_context_mb", "audiovae_latent_capacity", "audiovae_encoder_sample_capacity"),
    "voxcpm2": ("mem_saver", "prompt_cache_slots", "weight_type", "audiovae_weight_type", "weight_context_mb", "text_embedding_graph_context_mb", "lm_step_graph_context_mb", "projection_graph_context_mb", "local_encoder_graph_context_mb", "dit_graph_context_mb", "audiovae_weight_context_mb", "audiovae_graph_context_mb", "audiovae_encoder_graph_context_mb", "audiovae_latent_capacity", "audiovae_encoder_sample_capacity"),
}


def normalize_model(model: str | None) -> str:
    selected = str(model or DEFAULT_MODEL).split("(", 1)[0].strip()
    if selected in TTS_MODELS:
        return selected
    alias = MODEL_ALIASES.get(selected) or MODEL_ALIASES.get(selected.casefold())
    if alias:
        return alias
    raise ValueError(f"Unknown audio.cpp TTS model: {model}")


def precision_options(model: str = DEFAULT_MODEL) -> tuple[str, ...]:
    return tuple(TTS_MODELS[normalize_model(model)]["variants"])


def normalize_precision(precision, model: str = DEFAULT_MODEL) -> str:
    model = normalize_model(model)
    variants = TTS_MODELS[model]["variants"]
    selected = str(precision or "auto").strip().lower()
    selected = {"original": "orig", "native": "orig", "float32": "orig", "fp16": "f16", "float16": "f16", "bfloat16": "bf16", "q8": "q8_0", "int8": "q8_0", "8bit": "q8_0"}.get(selected, selected)
    if selected in {"auto", "default", "4bit", "q4_k"}:
        return TTS_MODELS[model]["default_precision"]
    if selected in variants:
        return selected
    return TTS_MODELS[model]["default_precision"]


def _model_precision_args(model, precision):
    if precision is None:
        try:
            normalized = normalize_model(model)
        except ValueError:
            return DEFAULT_MODEL, normalize_precision(model, DEFAULT_MODEL)
        return normalized, normalize_precision(None, normalized)
    normalized = normalize_model(model)
    return normalized, normalize_precision(precision, normalized)


def get_model_path(model=DEFAULT_MODEL, precision=None) -> Path:
    model, precision = _model_precision_args(model, precision)
    entry = TTS_MODELS[model]["variants"][precision]
    return MODEL_CACHE_PATH / model / precision / entry["filename"]


def needs_download(model=DEFAULT_MODEL, precision=None) -> bool:
    model, precision = _model_precision_args(model, precision)
    entry = TTS_MODELS[model]["variants"][precision]
    import downloader
    return downloader.model_needs_download(MODEL_CACHE_PATH / model / precision, entry["file_checksums"])


def download_model(model=DEFAULT_MODEL, precision=None, force_non_ui_dl: bool = False) -> bool:
    model, precision = _model_precision_args(model, precision)
    entry = TTS_MODELS[model]["variants"][precision]
    if not needs_download(model, precision):
        return True
    import downloader
    return downloader.download_model({
        "model_path": MODEL_CACHE_PATH,
        "model_link_dict": {"model": {**entry, "path": str(Path(model) / precision)}},
        "model_name": "model", "title": f"Text to Speech (audio.cpp) - {model} {precision}",
        "alt_fallback": False, "force_non_ui_dl": force_non_ui_dl, "extract_format": "none",
    }, DOWNLOAD_STATES[model][precision])


def _decode_pcm16_wav(payload: bytes) -> tuple[torch.Tensor, int]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            channels, sample_width, sample_rate = wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise RuntimeError("audio.cpp returned an invalid WAV response.") from exc
    if sample_width != 2:
        raise RuntimeError(f"audio.cpp returned {sample_width * 8}-bit WAV; expected PCM16.")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio.copy()).reshape(1, -1), int(sample_rate)


def _decode_task_audio_response(response) -> tuple[torch.Tensor, int]:
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("audio.cpp returned an invalid generic task response.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("audio"), str):
        raise RuntimeError("audio.cpp returned no audio from the generic task route.")
    try:
        wav_payload = base64.b64decode(payload["audio"], validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise RuntimeError("audio.cpp returned invalid base64 audio.") from exc
    return _decode_pcm16_wav(wav_payload)


def _dict_value(value):
    return value if isinstance(value, dict) else {}


def _special_settings_for(model: str) -> dict:
    key = TTS_MODELS[model]["settings_key"]
    defaults = TTS_SETTINGS_DEFAULTS[key]
    root = _dict_value(_dict_value(settings.GetOption("special_settings")).get(SPECIAL_SETTINGS_NAME))
    configured = _dict_value(root.get(key))
    if key == "supertonic":
        configured = {**{k: v for k, v in root.items() if k in defaults}, **configured}
    return {**defaults, **configured}


def _session_options(model: str, configured: dict) -> dict:
    definition, result = TTS_MODELS[model], {}
    family, key = definition["family"], definition["settings_key"]
    for name in SESSION_KEYS.get(key, ()):
        # The profile precision selects a pre-converted GGUF package. Asking
        # audio.cpp to reinterpret those tensors as another storage type is a
        # separate low-level conversion knob, not a compute precision choice.
        # It can also be invalid for tensors that the selected package already
        # stores in a model-specific format (Supertonic BF16, for example).
        # Keep accepting old profiles, but always let the GGUF declare its
        # native tensor types.
        if name == "weight_type" or name.endswith("_weight_type"):
            continue
        value = configured.get(name)
        if value in {None, ""} or (name.endswith(("_mb", "_capacity")) and value == 0):
            continue
        result[f"{family}.{name}"] = value
    if family == "omnivoice" and result.get("omnivoice.mem_saver"):
        result["omnivoice.perf_mode"] = "off"
    return result


@contextmanager
def _wav_reference(path: str | os.PathLike | None):
    if not path:
        yield None
        return
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"audio.cpp voice/source audio does not exist: {source}")
    if source.suffix.lower() == ".wav":
        yield str(source)
        return
    temporary = None
    try:
        from pydub import AudioSegment
        handle = tempfile.NamedTemporaryFile(prefix="whispering-tiger-audiocpp-", suffix=".wav", delete=False)
        temporary = Path(handle.name)
        handle.close()
        AudioSegment.from_file(str(source)).export(str(temporary), format="wav")
        yield str(temporary.resolve())
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


class AudioCppTTS(metaclass=SingletonMeta):
    sample_rate = SAMPLE_RATE

    def __init__(self):
        VOICES_PATH.mkdir(parents=True, exist_ok=True)
        self.server = AudioCppServer("tts")
        self.loaded_configuration = None
        self.compute_device = "cpu"
        self.device_index = 0
        self.precision = "orig"
        self.current_model = DEFAULT_MODEL
        self.last_generation = {"audio": None, "sample_rate": SAMPLE_RATE, "text": ""}
        self.audio_streamer = None
        self.stop_event = threading.Event()
        self.generation_lock = threading.RLock()
        self.response_lock = threading.RLock()
        self.active_response = None

    def list_models(self):
        grouped = {}
        for model, definition in TTS_MODELS.items():
            grouped.setdefault(definition["group"], []).append(model)
        return grouped

    def list_models_indexed(self):
        return tuple(
            {"language": group, "models": models}
            for group, models in self.list_models().items()
        )

    @staticmethod
    def _voice_files():
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        return [
            {"name": path.stem, "audio_filename": str(path.resolve())}
            for path in sorted(VOICES_PATH.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]

    def list_voices(self):
        model = self._selected_model()
        family = TTS_MODELS[model]["family"]
        if family == "supertonic":
            return tuple({"name": voice, "value": voice} for voice in BUILT_IN_VOICES)
        if family == "magpie_tts":
            return tuple({"name": voice, "value": voice} for voice in MAGPIE_VOICES)
        voices = [{"name": "Auto / no reference", "value": "auto"}]
        voices.extend({"name": voice["name"], "value": voice["name"]} for voice in self._voice_files())
        voices.append({"name": "open_voice_dir", "value": "open_dir:" + str(VOICES_PATH.resolve())})
        return tuple(voices)

    def _selected_model(self):
        selected = settings.GetOption("tts_model")
        if isinstance(selected, (list, tuple)) and len(selected) >= 2:
            try:
                return normalize_model(selected[1])
            except ValueError:
                pass
        if isinstance(selected, str) and selected:
            try:
                return normalize_model(selected)
            except ValueError:
                pass
        return DEFAULT_MODEL

    def _selected_device(self):
        return normalize_backend_device(
            settings.GetOption("tts_ai_device"),
            settings.GetOption("tts_ai_device_index"),
        )

    def load(self, streaming: bool = False):
        model = self._selected_model()
        definition = TTS_MODELS[model]
        precision = normalize_precision(get_tts_precision("auto"), model)
        backend, device_index = self._selected_device()
        mode = "streaming" if streaming and definition["streaming"] else "offline"
        configured = _special_settings_for(model)
        session_options = _session_options(model, configured)
        desired = (
            model, precision, backend, device_index, mode,
            json.dumps(session_options, sort_keys=True),
        )
        if desired == self.loaded_configuration and self.server.process is not None:
            if self.server.process.poll() is None:
                return True

        if not download_model(model, precision):
            raise RuntimeError(f"Could not download audio.cpp TTS model {model} ({precision}).")
        self.server.configure(
            backend=backend,
            device_index=device_index,
            family=definition["family"],
            model_path=get_model_path(model, precision),
            task=definition["task"],
            mode=mode,
            session_options=session_options,
        )
        self.compute_device = backend
        self.device_index = device_index
        self.precision = precision
        self.current_model = model
        self.sample_rate = int(definition["sample_rate"])
        self.loaded_configuration = desired
        print(
            f"audio.cpp {model} loaded on {backend}:{device_index} "
            f"with {precision} GGUF weights ({mode})."
        )
        return True

    def load_model(self):
        return self.load()

    @staticmethod
    def _wave_tensor(audio):
        if isinstance(audio, torch.Tensor):
            tensor = audio.detach().float().cpu()
        else:
            tensor = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(1, -1)
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        return torch.nan_to_num(tensor).clamp(-1.0, 1.0).contiguous()

    def _finish_audio(self, audio, *, normalize: bool, call_plugin: bool):
        wave_tensor = self._wave_tensor(audio)
        if normalize and settings.GetOption("tts_normalize") and wave_tensor.numel():
            wave_tensor, _ = audio_tools.normalize_audio_lufs(
                wave_tensor, self.sample_rate, -24.0, -16.0, 1.3, verbose=True,
            )
            wave_tensor = self._wave_tensor(wave_tensor)
        try:
            volume = float(settings.GetOption("tts_volume") or 1.0)
        except (TypeError, ValueError):
            volume = 1.0
        if volume != 1.0:
            wave_tensor = self._wave_tensor(audio_tools.change_volume(wave_tensor, volume))
        if call_plugin:
            import Plugins
            result = Plugins.plugin_custom_event_call(
                "plugin_tts_after_audio",
                {"audio": wave_tensor, "sample_rate": self.sample_rate},
            )
            if isinstance(result, dict) and result.get("audio") is not None:
                wave_tensor = self._wave_tensor(result["audio"])
        return wave_tensor

    @staticmethod
    def _detect_language(text: str, supported: set[str], fallback: str) -> str:
        try:
            from Models import languageClassification
            classification = languageClassification.classify(text, to_code="iso1")
            detected = classification[0] if isinstance(classification, tuple) else classification
            detected = str(detected or "").strip().lower()
            if detected in supported:
                return detected
            if detected == "pt" and "pt-br" in supported:
                return "pt-BR"
            if detected == "ar" and "ar-msa" in supported:
                return "ar-MSA"
        except Exception as exc:
            print(f"audio.cpp TTS language detection failed ({exc}); using {fallback}.")
        return fallback

    def _language(self, text: str, model: str, configured: dict) -> str:
        selected = str(configured.get("language", "auto") or "auto").strip()
        if selected.casefold() not in {"", "auto", "none"}:
            return selected
        family = TTS_MODELS[model]["family"]
        if family == "supertonic":
            return self._detect_language(text, SUPERTONIC_LANGUAGES, "en")
        if family == "confucius4_tts":
            return self._detect_language(text, CONFUCIUS_LANGUAGES, "en")
        if family == "magpie_tts":
            return self._detect_language(text, MAGPIE_LANGUAGES, "en")
        if family == "index_tts2":
            supported = {"zh", "en", "ja", "es", "ar"} if model == "IndexTTS2.5-GGUF" else {"zh", "en"}
            return self._detect_language(text, supported, "en")
        return ""

    @staticmethod
    def _selected_preset(choices, fallback):
        selected = str(settings.GetOption("tts_voice") or fallback).strip()
        for choice in choices:
            if selected.casefold() == choice.casefold():
                return choice
        return fallback

    def _resolve_voice(self, model: str, ref_audio=None):
        if ref_audio:
            path = Path(ref_audio).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"audio.cpp voice reference does not exist: {path}")
            return str(path)
        selected_name = str(settings.GetOption("tts_voice") or "").strip()
        for voice in self._voice_files():
            if voice["name"] == selected_name:
                return voice["audio_filename"]
        if TTS_MODELS[model].get("requires_reference"):
            voices = self._voice_files()
            if voices:
                return voices[0]["audio_filename"]
            raise FileNotFoundError(
                f"{model} needs a voice-cloning reference. Add a WAV, MP3, FLAC, "
                f"or OGG file to {VOICES_PATH.resolve()}."
            )
        return None

    @staticmethod
    def _reference_text(configured: dict, voice_path: str | None) -> str:
        explicit = str(configured.get("reference_text", "") or "").strip()
        if explicit:
            return explicit
        if voice_path:
            sidecar = Path(voice_path).with_suffix(".txt")
            if sidecar.is_file():
                try:
                    return sidecar.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeDecodeError):
                    pass
        return ""

    def _request_payload(self, text: str, ref_audio=None, *, streaming=False):
        model = self._selected_model()
        definition = TTS_MODELS[model]
        key = definition["settings_key"]
        configured = _special_settings_for(model)
        clean_text = text
        if definition["family"] == "voxcpm2":
            instruction = str(configured.get("voice_instruction", "") or "").strip()
            if instruction:
                clean_text = f"({instruction}){clean_text}"

        options = {}
        for name in REQUEST_KEYS.get(key, ()):
            value = configured.get(name)
            if value is None or value == "":
                continue
            if name == "reference_duration_sec" and float(value or 0) <= 0:
                continue
            if name == "seed" and int(value or -1) < 0:
                continue
            if name == "text_chunk_size" and int(value or 0) <= 0:
                continue
            if name in {"duration", "retry_badcase_ratio_threshold"} and float(value or 0) <= 0:
                continue
            options[name] = value
        if streaming and definition["family"] in {"voxcpm1", "voxcpm2"}:
            options["retry_badcase"] = False
        if key == "dots_tts_edit":
            options["template_name"] = "edit"
        if key == "index_tts2":
            emotion = [
                configured.get("emotion_happy", 0.0), configured.get("emotion_angry", 0.0),
                configured.get("emotion_sad", 0.0), configured.get("emotion_afraid", 0.0),
                configured.get("emotion_disgusted", 0.0), configured.get("emotion_melancholic", 0.0),
                configured.get("emotion_surprised", 0.0), configured.get("emotion_calm", 0.0),
            ]
            if any(float(value or 0) != 0 for value in emotion):
                options["emotion_vector"] = ",".join(str(float(value or 0)) for value in emotion)

        language = self._language(clean_text, model, configured)
        payload = {"model": self.server.model_id, "input": clean_text, "options": options}
        if language:
            payload["language"] = language
        voice_path = None
        source_path = None
        emotion_path = None
        family = definition["family"]
        if family == "supertonic":
            payload["voice"] = self._selected_preset(BUILT_IN_VOICES, "M1")
            options["speaking_rate"] = RATE_MULTIPLIERS.get(
                str(settings.GetOption("tts_prosody_rate") or "").strip().lower(),
                float(configured.get("speaking_rate", 1.05)),
            )
            # Preserve the convenient top-level field used by the OpenAI route.
            payload["num_inference_steps"] = options.pop("num_inference_steps", 8)
        elif family == "magpie_tts":
            options["voice_id"] = self._selected_preset(
                MAGPIE_VOICES, str(configured.get("voice_id", "Aria"))
            )
        elif key == "dots_tts_edit":
            source_path = str(configured.get("source_audio", "") or "").strip()
            if not source_path:
                raise FileNotFoundError(
                    "DotTTS Edit requires Source audio in its audio.cpp advanced settings."
                )
        else:
            voice_path = self._resolve_voice(model, ref_audio)
            reference_text = self._reference_text(configured, voice_path)
            if reference_text:
                payload["reference_text"] = reference_text
            instruction = str(configured.get("voice_instruction", "") or "").strip()
            if family == "omnivoice" and instruction:
                payload["instructions"] = instruction
            if family == "omnivoice" and voice_path and not reference_text:
                raise ValueError(
                    "OmniVoice voice cloning requires a transcript for the selected "
                    "reference. Add a same-name UTF-8 .txt file beside the voice sample "
                    "or enter Reference text in the audio.cpp advanced settings."
                )
            if key == "index_tts2":
                emotion_path = str(configured.get("emotion_audio", "") or "").strip()
        return payload, voice_path, source_path, emotion_path

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        del remove_silence, silence_after_segments, normalize
        clean_text = str(text or "").strip()
        if not clean_text:
            return torch.zeros((1, 0), dtype=torch.float32), self.sample_rate
        with self.generation_lock:
            self.stop_event.clear()
            self.load(streaming=False)
            payload, voice_path, source_path, emotion_path = self._request_payload(clean_text, ref_audio)
            with (
                _wav_reference(voice_path) as wav_voice,
                _wav_reference(source_path) as wav_source,
                _wav_reference(emotion_path) as wav_emotion,
            ):
                if wav_voice:
                    payload["voice_ref"] = wav_voice
                if wav_source:
                    payload["options"]["source_audio"] = wav_source
                if wav_emotion:
                    task_request = {
                        "text": payload["input"],
                        "audio": wav_emotion,
                        "voice_ref": wav_voice,
                        "options": payload["options"],
                    }
                    if payload.get("language"):
                        task_request["language"] = payload["language"]
                    if payload.get("reference_text"):
                        task_request["reference_text"] = payload["reference_text"]
                    response = self.server.request(
                        "POST",
                        "/v1/tasks/run",
                        json={"model": self.server.model_id, "request": task_request},
                    )
                else:
                    response = self.server.request("POST", "/v1/audio/speech", json=payload)
                try:
                    if wav_emotion:
                        audio, sample_rate = _decode_task_audio_response(response)
                    else:
                        audio, sample_rate = _decode_pcm16_wav(response.content)
                finally:
                    response.close()
            self.sample_rate = sample_rate
            audio = self._finish_audio(audio, normalize=True, call_plugin=True)
            self.last_generation = {"audio": audio, "sample_rate": sample_rate, "text": clean_text}
            return audio, sample_rate

    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")
        # AudioStreamer retains the source rate it was constructed with.  The
        # same AudioCppTTS instance can switch between models whose outputs range
        # from 16 to 48 kHz, so reusing (for example) an OmniVoice 24 kHz player
        # for Supertonic's 44.1 kHz PCM makes speech markedly slower and deeper.
        # The SSE stream contains raw PCM without a per-chunk WAV header, making
        # the selected model's rate the required source-of-truth here.
        if self.audio_streamer is not None:
            current_rate = getattr(self.audio_streamer, "source_sample_rate", None)
            current_device = getattr(self.audio_streamer, "device_index", None)
            if current_rate != self.sample_rate or current_device != audio_device:
                self.audio_streamer.stop()
                self.audio_streamer = None
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(
                audio_device, source_sample_rate=self.sample_rate, start_playback_timeout=1.0,
                min_buffer_play_time=float(settings.GetOption("tts_streamed_min_play_time")),
                playback_channels=2, buffer_size=settings.GetOption("tts_streamed_chunk_size"),
                input_channels=1, dtype="float32", tag="tts",
            )

    def tts_streaming(self, text, ref_audio=None):
        clean_text = str(text or "").strip()
        if not clean_text:
            return torch.zeros((1, 0), dtype=torch.float32), self.sample_rate
        model = self._selected_model()
        if not TTS_MODELS[model]["streaming"]:
            audio, sample_rate = self.tts(clean_text, ref_audio)
            if audio.numel() and not self.stop_event.is_set():
                self.init_audio_stream_playback()
                if self.audio_streamer is not None:
                    self.audio_streamer.add_audio_chunk(self.return_pcm_audio(audio))
            return audio, sample_rate

        with self.generation_lock:
            import Plugins
            self.stop_event.clear()
            self.load(streaming=True)
            has_postprocessor = Plugins.plugin_custom_event_has_active_handler("plugin_tts_after_audio")
            if not has_postprocessor:
                self.init_audio_stream_playback()
            payload, voice_path, source_path, _ = self._request_payload(clean_text, ref_audio, streaming=True)
            payload.update({"response_format": "pcm", "stream_format": "sse"})
            with _wav_reference(voice_path) as wav_voice, _wav_reference(source_path) as wav_source:
                if wav_voice:
                    payload["voice_ref"] = wav_voice
                if wav_source:
                    payload["options"]["source_audio"] = wav_source
                response = self.server.request(
                    "POST", "/v1/audio/speech", json=payload,
                    headers={"Accept": "text/event-stream"}, stream=True,
                )
                with self.response_lock:
                    self.active_response = response
                chunks = []
                try:
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if self.stop_event.is_set():
                            break
                        line = str(raw_line or "").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        event = json.loads(data)
                        event_type = event.get("type")
                        if event_type == "error":
                            error = event.get("error")
                            if isinstance(error, dict):
                                error = error.get("message") or error
                            raise RuntimeError(f"audio.cpp streaming failed: {error}")
                        if event_type != "speech.audio.delta":
                            continue
                        pcm = base64.b64decode(event.get("audio") or "", validate=True)
                        if len(pcm) % 2:
                            raise RuntimeError("audio.cpp returned an unaligned PCM16 stream chunk.")
                        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
                        chunk = torch.from_numpy(samples).reshape(1, -1)
                        if not has_postprocessor:
                            chunk = self._finish_audio(chunk, normalize=False, call_plugin=False)
                        if not chunk.numel():
                            continue
                        chunks.append(chunk)
                        if not has_postprocessor and self.audio_streamer is not None:
                            self.audio_streamer.add_audio_chunk(self.return_pcm_audio(chunk))
                except Exception:
                    if not self.stop_event.is_set():
                        raise
                finally:
                    response.close()
                    with self.response_lock:
                        if self.active_response is response:
                            self.active_response = None
            audio = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, 0))
            if has_postprocessor and audio.numel() and not self.stop_event.is_set():
                audio = self._finish_audio(audio, normalize=True, call_plugin=True)
                if audio.numel():
                    self.init_audio_stream_playback()
                    if self.audio_streamer is not None:
                        self.audio_streamer.add_audio_chunk(self.return_pcm_audio(audio))
            self.last_generation = {"audio": audio, "sample_rate": self.sample_rate, "text": clean_text}
            return audio, self.sample_rate

    def stop(self):
        self.stop_event.set()
        with self.response_lock:
            response = self.active_response
        if response is not None:
            response.close()
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def release_model(self):
        self.stop()
        self.server.stop()
        self.loaded_configuration = None

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def play_audio(self, audio, device=None):
        if device is None:
            device = settings.GetOption("device_default_out_index")
        secondary = None
        if settings.GetOption("tts_use_secondary_playback"):
            secondary = settings.GetOption("tts_secondary_playback_device")
            if secondary == -1:
                secondary = settings.GetOption("device_default_out_index")
        audio_tools.play_audio(
            self._wave_tensor(audio), device, source_sample_rate=self.sample_rate,
            audio_device_channel_num=1, target_channels=1, input_channels=1,
            dtype="float32", tensor_sample_with=4, tensor_channels=1,
            secondary_device=secondary,
            stop_play=not settings.GetOption("tts_allow_overlapping_audio"), tag="tts",
        )

    def return_wav_file_binary(self, audio, sample_rate=None):
        sample_rate = int(sample_rate or self.sample_rate)
        samples = self._wave_tensor(audio).squeeze(0).numpy()
        pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return output.getvalue()

    def return_pcm_audio(self, audio):
        return self._wave_tensor(audio).squeeze(0).numpy().astype("<f4", copy=False).tobytes()
