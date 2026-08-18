import gc
import io
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from transformers import StoppingCriteria, StoppingCriteriaList

import audio_tools
import settings
from Models.Singleton import SingletonMeta
from Models.transformers_attention import (
    FLASH_ATTENTION_2,
    SDPA,
    get_preferred_attention_implementation,
    load_with_attention_fallback,
)


SAMPLE_RATE = 24000
ARCHIVE_CHECKSUM_PLACEHOLDER = "0" * 64
MODEL_CACHE_PATH = Path.cwd() / ".cache" / "qwen3-tts"
TOKENIZER_MODEL = "Qwen3-TTS-Tokenizer-12Hz"
DEFAULT_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"

# Qwen3-TTS can reuse the voice sample pack already shared by Chatterbox and
# IndexTTS. A matching .txt sidecar can contain the exact reference transcript
# for higher-fidelity ICL cloning.
VOICE_CACHE_PATH = Path.cwd() / ".cache" / "chatterbox-tts-cache"
VOICES_PATH = VOICE_CACHE_PATH / "voices"

SUPPORTED_LANGUAGES = {
    "auto": "Auto",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

PRESET_VOICES = {
    "Vivian": "Vivian — bright young female (Chinese)",
    "Serena": "Serena — warm gentle young female (Chinese)",
    "Uncle_Fu": "Uncle Fu — mellow seasoned male (Chinese)",
    "Dylan": "Dylan — youthful Beijing male (Chinese)",
    "Eric": "Eric — lively Chengdu male (Chinese)",
    "Ryan": "Ryan — dynamic rhythmic male (English)",
    "Aiden": "Aiden — sunny American male (English)",
    "Ono_Anna": "Ono Anna — playful female (Japanese)",
    "Sohee": "Sohee — warm female (Korean)",
}

MODEL_LIST = {
    "Voice cloning": [
        "Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen3-TTS-12Hz-1.7B-Base",
    ],
    "Built-in voices and control": [
        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ],
    "Voice design": ["Qwen3-TTS-12Hz-1.7B-VoiceDesign"],
}


def _hosted_urls(archive_name):
    return [
        f"https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/qwen3-tts/{archive_name}",
        f"https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/qwen3-tts/{archive_name}",
        f"https://s3.libs.space:9000/ai-models/qwen3-tts/{archive_name}",
    ]


_COMMON_MODEL_HASHES = {
    "generation_config.json": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "preprocessor_config.json": "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119",
    "tokenizer_config.json": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def _model_manifest(archive_name, config_hash, model_hash, revision):
    return {
        "urls": _hosted_urls(archive_name),
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "config.json": config_hash,
            **_COMMON_MODEL_HASHES,
            "model.safetensors": model_hash,
        },
        "path": archive_name.removesuffix(".zip"),
        "source_revision": revision,
    }


# Source hashes are the SHA-256 values from immutable official Qwen revisions.
# Archive hashes intentionally remain zero until the maintainer creates and
# uploads the application-hosted ZIPs. Runtime downloads never contact HF.
TTS_MODEL_LINKS = {
    # Qwen/Qwen3-TTS-Tokenizer-12Hz @ 7dd38ad4e9bad454aae9cd937d0cd577604fe229
    TOKENIZER_MODEL: {
        "urls": _hosted_urls("qwen3-tts-tokenizer-12hz.zip"),
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
            "configuration.json": "6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd",
            "model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
            "preprocessor_config.json": "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb",
        },
        "path": "qwen3-tts-tokenizer-12hz",
        "source_revision": "7dd38ad4e9bad454aae9cd937d0cd577604fe229",
    },
    # Qwen/Qwen3-TTS-12Hz-0.6B-Base @ 5d83992436eae1d760afd27aff78a71d676296fc
    "Qwen3-TTS-12Hz-0.6B-Base": _model_manifest(
        "qwen3-tts-12hz-0.6b-base.zip",
        "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
        "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",
        "5d83992436eae1d760afd27aff78a71d676296fc",
    ),
    # Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice @ 85e237c12c027371202489a0ec509ded67b5e4b5
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": _model_manifest(
        "qwen3-tts-12hz-0.6b-customvoice.zip",
        "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
        "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
        "85e237c12c027371202489a0ec509ded67b5e4b5",
    ),
    # Qwen/Qwen3-TTS-12Hz-1.7B-Base @ fd4b254389122332181a7c3db7f27e918eec64e3
    "Qwen3-TTS-12Hz-1.7B-Base": _model_manifest(
        "qwen3-tts-12hz-1.7b-base.zip",
        "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
        "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c",
        "fd4b254389122332181a7c3db7f27e918eec64e3",
    ),
    # Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice @ 0c0e3051f131929182e2c023b9537f8b1c68adfe
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": _model_manifest(
        "qwen3-tts-12hz-1.7b-customvoice.zip",
        "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
        "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137",
        "0c0e3051f131929182e2c023b9537f8b1c68adfe",
    ),
    # Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign @ 5ecdb67327fd37bb2e042aab12ff7391903235d3
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": _model_manifest(
        "qwen3-tts-12hz-1.7b-voicedesign.zip",
        "aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0",
        "391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522",
        "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    ),
}

VOICE_MODEL_LINKS = {
    "voices": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/chatterbox-tts/voices.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/chatterbox-tts/voices.zip",
            "https://s3.libs.space:9000/ai-models/chatterbox-tts/voices.zip",
        ],
        "checksum": "1219fc592b50118807d54e3049e6b019d248e2e1a6be2324e398b3edd6df19a9",
        "file_checksums": {
            "Announcer_Ahri.wav": "2a3fd17d45b3c5633dd64e2d80a6e3fc924fa829da405e3b591a5bacdc88f9fc",
            "Attenborough.wav": "358540c89932baf1103960d99949b78ea7466f95b2225fdcd8f8bb8b976f09ee",
            "Jane.wav": "d1d2235af1a4408c641a765427978486f5cca9b369fc6456d8086449f1f92fe3",
            "Justin.wav": "a83c37f408b53efaeb9189f166c6669d1a0dc6cf779e85913fa9cbbbbe0d5aaf",
            "Xiaochen.wav": "7f0b735e188a06dc9f104eeb3fd71a3ef580d1f2133c95630c92a244dd253732",
            "default_voice.wav": "3ebc531cdaba358a327099c1c4f0448026719957bcf4d8e9868767f227e02f4e",
            "en_0.wav": "f006e2e9c76523bde4f5bbe67a7be9a600786d7432cbcc9486bc9501053298b7",
            "en_1.wav": "b0e22048e72414fcc1e6b6342e47a774d748a195ed34e4a5b3fcf416707f2b71",
            "fallback_audio.wav": "eaa7796d2c44424c645a0b384d82f09aac48fab2c9977de6f53b6a4f9d0e0da1",
            "female_shadowheart.wav": "8abb726ad6aaa5203e62de4c92ac2aab3d3fa1fdb509c9b76d254722178ab70a",
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede",
            "tiktok_adam.wav": "2ed130b6dd069ee4c306f6cb8fedb94db75567aefa084085c6a069bd2c34662d",
            "tiktok_jessie.wav": "5a26de921ea3e7c1ce1bfd2344fb107781def9366b56e2f583c7500a1052dbbd",
        },
        "path": "voices",
    }
}


class _StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        del input_ids, scores, kwargs
        return self.event.is_set()


class _QwenCodecAudioStreamer:
    """Decode completed Qwen codec frames while the talker is still running."""

    def __init__(
        self,
        speech_tokenizer,
        on_audio_chunk,
        eos_token_id,
        frames_per_chunk=2,
        context_frames=25,
    ):
        self.speech_tokenizer = speech_tokenizer
        self.on_audio_chunk = on_audio_chunk
        self.eos_token_id = int(eos_token_id)
        self.frames_per_chunk = max(1, int(frames_per_chunk))
        self.context_frames = max(0, int(context_frames))
        self.samples_per_frame = int(speech_tokenizer.get_decode_upsample_rate())
        self.prefix_codes = None
        self.generated_frames = []
        self.output_cursor = 0
        self.ended = False

    def set_prefix_codes(self, prefix_codes):
        if prefix_codes is None:
            self.prefix_codes = None
            return
        if not isinstance(prefix_codes, torch.Tensor):
            prefix_codes = torch.as_tensor(prefix_codes, dtype=torch.long)
        prefix_codes = prefix_codes.detach()
        if prefix_codes.ndim == 3 and prefix_codes.shape[0] == 1:
            prefix_codes = prefix_codes[0]
        if prefix_codes.ndim != 2:
            raise ValueError(
                "Qwen3-TTS streaming reference codes must have shape (frames, codebooks)."
            )
        self.prefix_codes = prefix_codes

    def put(self, codec_ids):
        if self.ended:
            return
        if not isinstance(codec_ids, torch.Tensor):
            codec_ids = torch.as_tensor(codec_ids, dtype=torch.long)
        codec_ids = codec_ids.detach()
        if codec_ids.ndim == 1:
            codec_ids = codec_ids.unsqueeze(0)
        if codec_ids.ndim != 2 or codec_ids.shape[0] != 1:
            raise ValueError("Qwen3-TTS live audio streaming currently requires a batch size of one.")

        frame = codec_ids[0]
        if int(frame[0].item()) == self.eos_token_id:
            return
        self.generated_frames.append(frame)
        while len(self.generated_frames) - self.output_cursor >= self.frames_per_chunk:
            self._emit(self.frames_per_chunk)

    def _emit(self, frame_count):
        if frame_count <= 0:
            return
        generated = torch.stack(self.generated_frames, dim=0)
        prefix = self.prefix_codes
        if prefix is not None:
            prefix = prefix.to(device=generated.device, dtype=generated.dtype)
            all_codes = torch.cat((prefix, generated), dim=0)
            prefix_length = int(prefix.shape[0])
        else:
            all_codes = generated
            prefix_length = 0

        output_start = prefix_length + self.output_cursor
        context = min(self.context_frames, output_start)
        decode_start = output_start - context
        decode_end = output_start + frame_count
        wavs, sample_rate = self.speech_tokenizer.decode(
            [{"audio_codes": all_codes[decode_start:decode_end]}]
        )
        decoded = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        crop_start = context * self.samples_per_frame
        crop_end = crop_start + frame_count * self.samples_per_frame
        chunk = decoded[crop_start:crop_end]
        if chunk.size == 0:
            raise RuntimeError("Qwen3-TTS incremental codec decode returned no new audio.")
        self.output_cursor += frame_count
        self.on_audio_chunk(chunk, int(sample_rate))

    def end(self):
        if self.ended:
            return
        remaining = len(self.generated_frames) - self.output_cursor
        if remaining > 0:
            self._emit(remaining)
        self.ended = True


class _QwenBatchCodecAudioStreamer:
    """Decode independent Qwen generation lanes while preserving lane identity."""

    def __init__(
        self,
        speech_tokenizer,
        on_audio_chunk,
        on_lane_end,
        eos_token_id,
        pad_token_id,
        batch_size,
        frames_per_chunk=2,
        context_frames=25,
    ):
        self.speech_tokenizer = speech_tokenizer
        self.on_audio_chunk = on_audio_chunk
        self.on_lane_end = on_lane_end
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.batch_size = max(1, int(batch_size))
        self.frames_per_chunk = max(1, int(frames_per_chunk))
        self.context_frames = max(0, int(context_frames))
        self.samples_per_frame = int(speech_tokenizer.get_decode_upsample_rate())
        self.prefix_codes = [None] * self.batch_size
        self.generated_frames = [[] for _ in range(self.batch_size)]
        self.output_cursors = [0] * self.batch_size
        self.lane_ended = [False] * self.batch_size
        self.lane_notified = [False] * self.batch_size
        self.ended = False

    @staticmethod
    def _normalize_prefix(prefix_codes):
        if prefix_codes is None:
            return None
        if not isinstance(prefix_codes, torch.Tensor):
            prefix_codes = torch.as_tensor(prefix_codes, dtype=torch.long)
        prefix_codes = prefix_codes.detach()
        if prefix_codes.ndim == 3 and prefix_codes.shape[0] == 1:
            prefix_codes = prefix_codes[0]
        if prefix_codes.ndim != 2:
            raise ValueError(
                "Qwen3-TTS batch streaming reference codes must have shape "
                "(frames, codebooks)."
            )
        return prefix_codes

    def set_prefix_codes(self, prefix_codes):
        if prefix_codes is None:
            self.prefix_codes = [None] * self.batch_size
            return
        if isinstance(prefix_codes, torch.Tensor) and prefix_codes.ndim == 3:
            values = [prefix_codes[index] for index in range(prefix_codes.shape[0])]
        elif isinstance(prefix_codes, (list, tuple)):
            values = list(prefix_codes)
        else:
            values = [prefix_codes]
        if len(values) == 1 and self.batch_size > 1:
            values *= self.batch_size
        if len(values) != self.batch_size:
            raise ValueError(
                "Qwen3-TTS batch streaming needs one reference-code prefix per lane."
            )
        self.prefix_codes = [self._normalize_prefix(value) for value in values]

    def put(self, codec_ids):
        if self.ended:
            return
        if not isinstance(codec_ids, torch.Tensor):
            codec_ids = torch.as_tensor(codec_ids, dtype=torch.long)
        codec_ids = codec_ids.detach()
        if codec_ids.ndim == 1:
            codec_ids = codec_ids.unsqueeze(0)
        if codec_ids.ndim != 2 or codec_ids.shape[0] != self.batch_size:
            raise ValueError(
                f"Qwen3-TTS batch streamer expected {self.batch_size} lanes, "
                f"received shape {tuple(codec_ids.shape)}."
            )

        # One synchronization handles all lanes. Finished Hugging Face batch
        # rows receive EOS and subsequently PAD while longer rows continue.
        first_tokens = codec_ids[:, 0].detach().cpu().tolist()
        for lane_index, token_id in enumerate(first_tokens):
            if self.lane_ended[lane_index]:
                continue
            if int(token_id) in {self.eos_token_id, self.pad_token_id}:
                self.lane_ended[lane_index] = True
                continue
            self.generated_frames[lane_index].append(codec_ids[lane_index])

        self._emit_ready()
        self._notify_completed_lanes()

    def _emit_ready(self, final=False):
        while True:
            decode_inputs = []
            metadata = []
            for lane_index in range(self.batch_size):
                cursor = self.output_cursors[lane_index]
                pending = len(self.generated_frames[lane_index]) - cursor
                if pending <= 0:
                    continue
                if pending >= self.frames_per_chunk:
                    frame_count = self.frames_per_chunk
                elif final or self.lane_ended[lane_index]:
                    frame_count = pending
                else:
                    continue

                generated = torch.stack(self.generated_frames[lane_index], dim=0)
                prefix = self.prefix_codes[lane_index]
                if prefix is not None:
                    prefix = prefix.to(device=generated.device, dtype=generated.dtype)
                    all_codes = torch.cat((prefix, generated), dim=0)
                    prefix_length = int(prefix.shape[0])
                else:
                    all_codes = generated
                    prefix_length = 0

                output_start = prefix_length + cursor
                context = min(self.context_frames, output_start)
                decode_start = output_start - context
                decode_end = output_start + frame_count
                decode_inputs.append(
                    {"audio_codes": all_codes[decode_start:decode_end]}
                )
                metadata.append((lane_index, frame_count, context))

            if not decode_inputs:
                return

            wavs, sample_rate = self.speech_tokenizer.decode(decode_inputs)
            if len(wavs) != len(metadata):
                raise RuntimeError(
                    "Qwen3-TTS batch codec decoder returned a different number "
                    "of waveforms than requested lanes."
                )
            for wav, (lane_index, frame_count, context) in zip(wavs, metadata):
                decoded = np.asarray(wav, dtype=np.float32).reshape(-1)
                crop_start = context * self.samples_per_frame
                crop_end = crop_start + frame_count * self.samples_per_frame
                chunk = decoded[crop_start:crop_end]
                if chunk.size == 0:
                    raise RuntimeError(
                        f"Qwen3-TTS batch codec decoder returned no audio for lane {lane_index}."
                    )
                self.output_cursors[lane_index] += frame_count
                self.on_audio_chunk(lane_index, chunk, int(sample_rate))

            self._notify_completed_lanes()

    def _notify_completed_lanes(self):
        for lane_index in range(self.batch_size):
            if (
                self.lane_ended[lane_index]
                and not self.lane_notified[lane_index]
                and self.output_cursors[lane_index]
                >= len(self.generated_frames[lane_index])
            ):
                self.lane_notified[lane_index] = True
                self.on_lane_end(lane_index)

    def end(self):
        if self.ended:
            return
        self.lane_ended = [True] * self.batch_size
        self._emit_ready(final=True)
        self._notify_completed_lanes()
        self.ended = True


class Qwen3TTS(metaclass=SingletonMeta):
    sample_rate = SAMPLE_RATE
    special_settings_defaults = {
        "precision": "auto",
        "attention": "auto",
        "language": "auto",
        "voice_instruction": "",
        "reference_text": "",
        "clone_mode": "auto",
        "model_text_mode": "auto",
        "apply_prosody_to_instruction": True,
        "seed": -1,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "subtalker_do_sample": True,
        "subtalker_temperature": 0.9,
        "subtalker_top_p": 1.0,
        "subtalker_top_k": 50,
        "max_new_tokens": 2048,
        "streaming_mode": "codec",
        "streaming_codec_frames": 2,
        "streaming_decoder_context_frames": 25,
        "streaming_buffer_mode": "adaptive",
        "streaming_buffer_safety_ms": 500,
        "streaming_lookahead_batch_size": 3,
        "streaming_lookahead_characters": 80,
        "streaming_lookahead_codec_frames": 6,
        "streaming_lookahead_pause_ms": 0,
        "streaming_segment_characters": 180,
        "pause_between_segments_ms": 120,
    }

    def __init__(self):
        self.model = None
        self.loaded_configuration = None
        self.compute_device_str = "cpu"
        self.compute_device = torch.device("cpu")
        self.special_settings = dict(self.special_settings_defaults)
        self.last_generation = {"audio": None, "sample_rate": None, "text": ""}
        self.audio_streamer = None
        self.download_state = {"is_downloading": False}
        self.voice_download_state = {"is_downloading": False}
        self.generation_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.voice_prompt_cache = {}
        self.streaming_rtf_history = {}
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        os.makedirs(VOICES_PATH, exist_ok=True)
        self.set_compute_device(settings.GetOption("tts_ai_device"))

    def list_models(self):
        return MODEL_LIST

    def list_models_indexed(self):
        return tuple(
            {"language": group, "models": models}
            for group, models in self.list_models().items()
        )

    def set_special_setting(self, special_settings):
        if isinstance(special_settings, dict):
            self.special_settings = {**self.special_settings_defaults, **special_settings}

    def _ensure_special_settings(self):
        all_settings = settings.GetOption("special_settings")
        if not isinstance(all_settings, dict):
            all_settings = {}
        configured = all_settings.get("tts_qwen3_tts")
        if isinstance(configured, dict):
            self.special_settings = {**self.special_settings_defaults, **configured}
        else:
            self.special_settings = dict(self.special_settings_defaults)
            all_settings["tts_qwen3_tts"] = dict(self.special_settings)
            settings.SetOption("special_settings", all_settings)

    def set_compute_device(self, requested):
        device = str(requested or "").strip().lower()
        if device in {"", "auto", "cuda"}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("direct-ml"):
            raise ValueError("Qwen3-TTS supports CUDA and CPU, but not DirectML.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"Unsupported Qwen3-TTS device: {requested}")
        self.compute_device_str = device
        self.compute_device = torch.device(device)

    @staticmethod
    def _strip_model_group(value):
        return re.sub(r"\s+\([^()]+\)\s*$", "", str(value or "")).strip()

    @staticmethod
    def _model_group(model_name):
        for group, model_names in MODEL_LIST.items():
            if model_name in model_names:
                return group
        return ""

    def _get_model_name(self):
        selected = settings.GetOption("tts_model")
        if isinstance(selected, (list, tuple)) and selected:
            candidate = selected[1] if len(selected) > 1 else selected[0]
        else:
            candidate = selected
        candidate = self._strip_model_group(candidate)
        model_name = candidate if candidate in MODEL_LIST_FLAT else DEFAULT_MODEL
        canonical_selection = [self._model_group(model_name), model_name]
        if not isinstance(selected, list) or selected != canonical_selection:
            # Older UI builds sent the display string instead of the canonical
            # two-element value. Normalize it before settings are broadcast so
            # current Go clients can deserialize and restore the selection.
            settings.SetOption("tts_model", canonical_selection)
        return model_name

    @staticmethod
    def _model_mode(model_name):
        if model_name.endswith("-Base"):
            return "base"
        if model_name.endswith("-CustomVoice"):
            return "custom_voice"
        if model_name.endswith("-VoiceDesign"):
            return "voice_design"
        raise ValueError(f"Unsupported Qwen3-TTS checkpoint: {model_name}")

    @staticmethod
    def _model_directory(model_name):
        return MODEL_CACHE_PATH / TTS_MODEL_LINKS[model_name]["path"]

    def download_model(self, model_name, force_non_ui_dl=False):
        import downloader

        if model_name not in TTS_MODEL_LINKS:
            raise ValueError(f"Unknown Qwen3-TTS model: {model_name}")
        entry = TTS_MODEL_LINKS[model_name]
        model_directory = self._model_directory(model_name)
        if not downloader.model_needs_download(model_directory, entry["file_checksums"]):
            return True
        if entry["checksum"] == ARCHIVE_CHECKSUM_PLACEHOLDER:
            archive_name = Path(entry["urls"][0]).name
            raise RuntimeError(
                f"The application-hosted Qwen3-TTS archive {archive_name} is not ready. "
                "Create the ZIP with the manifest files at its root, upload it, and "
                "replace the zero SHA-256 checksum in TTS_MODEL_LINKS."
            )
        return downloader.download_model(
            {
                "model_path": MODEL_CACHE_PATH,
                "model_link_dict": TTS_MODEL_LINKS,
                "model_name": model_name,
                "title": f"Text to Speech ({model_name})",
                "alt_fallback": False,
                "force_non_ui_dl": force_non_ui_dl,
                "extract_format": "zip",
            },
            self.download_state,
        )

    def download_voices(self, force_non_ui_dl=False):
        import downloader

        entry = VOICE_MODEL_LINKS["voices"]
        if not downloader.model_needs_download(VOICES_PATH, entry["file_checksums"]):
            return True
        return downloader.download_model(
            {
                "model_path": VOICE_CACHE_PATH,
                "model_link_dict": VOICE_MODEL_LINKS,
                "model_name": "voices",
                "title": "Voice samples (Qwen3-TTS / IndexTTS / Chatterbox)",
                "alt_fallback": False,
                "force_non_ui_dl": force_non_ui_dl,
                "extract_format": "zip",
            },
            self.voice_download_state,
        )

    def update_voices(self):
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        return [
            {"name": path.stem, "audio_filename": str(path.resolve())}
            for path in sorted(VOICES_PATH.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]

    def list_voices(self):
        mode = self._model_mode(self._get_model_name())
        if mode == "custom_voice":
            return [
                {"name": label, "value": speaker}
                for speaker, label in PRESET_VOICES.items()
            ]
        if mode == "voice_design":
            return [{"name": "Voice generated from the instruction below", "value": "voice_design"}]
        voices = [
            {"name": voice["name"], "value": voice["name"]}
            for voice in self.update_voices()
        ]
        voices.append({"name": "open_voice_dir", "value": "open_dir:" + str(VOICES_PATH.resolve())})
        return voices

    def get_voice_by_name(self, voice_name):
        for voice in self.update_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def _resolve_voice(self, ref_audio):
        if ref_audio:
            path = Path(ref_audio).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Qwen3-TTS voice reference does not exist: {path}")
            return path
        selected = self.get_voice_by_name(settings.GetOption("tts_voice"))
        if selected is None:
            selected = self.get_voice_by_name("default_voice")
        if selected is None:
            voices = self.update_voices()
            selected = voices[0] if voices else None
        if selected is None:
            raise FileNotFoundError(
                f"Qwen3-TTS Base needs a voice reference. Add a WAV/MP3/FLAC/OGG file to {VOICES_PATH}."
            )
        return Path(selected["audio_filename"])

    def _effective_precision(self):
        precision = str(self.special_settings.get("precision", "auto")).lower()
        if precision not in {"auto", "float32", "float16", "bfloat16"}:
            precision = "auto"
        if self.compute_device.type != "cuda":
            return "float32"
        if precision == "auto":
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            return "float32"
        if precision == "float16":
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
        return precision

    def _attention_implementation(self, dtype):
        requested = str(self.special_settings.get("attention", "auto")).lower()
        automatic = get_preferred_attention_implementation(self.compute_device, dtype)
        if requested == "eager":
            return "eager"
        if requested == SDPA:
            return SDPA
        if requested == FLASH_ATTENTION_2:
            if automatic == FLASH_ATTENTION_2:
                return FLASH_ATTENTION_2
            print("Qwen3-TTS FlashAttention 2 is unavailable for this device/precision; using SDPA.")
            return SDPA
        return automatic

    @staticmethod
    def _runtime_class():
        runtime_parent = Path(__file__).resolve().parent
        runtime_parent_string = str(runtime_parent)
        if runtime_parent_string not in sys.path:
            sys.path.insert(0, runtime_parent_string)
        from qwen3_tts_runtime import Qwen3TTSModel

        return Qwen3TTSModel

    def load(self):
        with self.generation_lock:
            self._ensure_special_settings()
            self.set_compute_device(settings.GetOption("tts_ai_device"))
            model_name = self._get_model_name()
            requested_precision = str(
                self.special_settings.get("precision", "auto")
            ).lower()
            if requested_precision not in {
                "auto",
                "float32",
                "float16",
                "bfloat16",
            }:
                requested_precision = "auto"
            precision = self._effective_precision()
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[precision]
            attention = self._attention_implementation(dtype)
            configuration = (model_name, self.compute_device_str, precision, attention)
            if self.model is not None and self.loaded_configuration == configuration:
                return
            if self.model is not None:
                self.release_model()

            if requested_precision == "float16":
                print(
                    "Qwen3-TTS float16 is numerically unstable in the talker "
                    f"and sub-talker; using {precision}."
                )
            elif requested_precision == "bfloat16" and precision == "float32":
                print(
                    "Qwen3-TTS bfloat16 is unavailable on this device; using float32."
                )

            if not self.download_model(TOKENIZER_MODEL):
                raise RuntimeError("Qwen3-TTS tokenizer installation failed.")
            if not self.download_model(model_name):
                raise RuntimeError(f"{model_name} installation failed.")
            if self._model_mode(model_name) == "base" and not self.download_voices():
                print("Qwen3-TTS voice sample pack could not be installed; custom references remain usable.")

            model_directory = self._model_directory(model_name).resolve()
            tokenizer_directory = self._model_directory(TOKENIZER_MODEL).resolve()
            print(
                f"Loading {model_name} on {self.compute_device_str} with {precision} "
                f"precision and {attention} attention..."
            )
            runtime_class = self._runtime_class()

            def loader(attention_implementation):
                return runtime_class.from_pretrained(
                    str(model_directory),
                    speech_tokenizer_path=str(tokenizer_directory),
                    device_map=self.compute_device_str,
                    dtype=dtype,
                    attn_implementation=attention_implementation,
                    local_files_only=True,
                )

            self.model, actual_attention = load_with_attention_fallback(
                loader, attention, model_name
            )
            self.model.model.eval()
            self.loaded_configuration = (
                model_name,
                self.compute_device_str,
                precision,
                actual_attention,
            )
            self.voice_prompt_cache.clear()
            print(f"{model_name} loaded.")

    def release_model(self):
        self.model = None
        self.loaded_configuration = None
        self.voice_prompt_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop(self):
        self.stop_event.set()
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    @staticmethod
    def _float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _language(self):
        language = str(self.special_settings.get("language", "auto")).lower()
        language = language.replace("_", "-").split("-")[0]
        if language not in SUPPORTED_LANGUAGES:
            print(f"Qwen3-TTS does not support language '{language}'; using automatic language selection.")
            language = "auto"
        return SUPPORTED_LANGUAGES[language]

    def _reference_text(self, voice_path):
        configured = str(self.special_settings.get("reference_text", "") or "").strip()
        if configured:
            return configured
        sidecar = voice_path.with_suffix(".txt")
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8-sig").strip()
            if text:
                return text
        return None

    def _clone_prompt(self, ref_audio):
        voice_path = self._resolve_voice(ref_audio)
        reference_text = self._reference_text(voice_path)
        clone_mode = str(self.special_settings.get("clone_mode", "auto")).lower()
        if clone_mode not in {"auto", "icl", "x_vector"}:
            clone_mode = "auto"
        if clone_mode == "icl" and not reference_text:
            raise ValueError(
                "Qwen3-TTS ICL cloning needs the exact reference transcript. Enter it in "
                "Qwen3-TTS settings or add a same-name .txt file beside the voice sample."
            )
        x_vector_only = clone_mode == "x_vector" or (clone_mode == "auto" and not reference_text)
        stat = voice_path.stat()
        cache_key = (
            str(voice_path),
            stat.st_mtime_ns,
            stat.st_size,
            reference_text,
            x_vector_only,
        )
        prompt = self.voice_prompt_cache.get(cache_key)
        if prompt is None:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(voice_path),
                ref_text=reference_text,
                x_vector_only_mode=x_vector_only,
            )
            self.voice_prompt_cache = {cache_key: prompt}
        return prompt

    @staticmethod
    def _reference_code_prefixes(clone_prompt, batch_size):
        prefixes = []
        if isinstance(clone_prompt, (list, tuple)):
            prefixes = [getattr(item, "ref_code", None) for item in clone_prompt]
        elif isinstance(clone_prompt, dict):
            prefixes = clone_prompt.get("ref_code")
            if prefixes is None:
                prefixes = []
            if isinstance(prefixes, torch.Tensor):
                prefixes = [prefixes]
        if not prefixes:
            prefixes = [None]
        if len(prefixes) == 1 and batch_size > 1:
            prefixes = list(prefixes) * batch_size
        if len(prefixes) != batch_size:
            raise ValueError(
                "Qwen3-TTS voice-clone prompt count does not match the lookahead batch."
            )
        return list(prefixes)

    def _instruction(self, mode):
        instruction = str(self.special_settings.get("voice_instruction", "") or "").strip()
        if bool(self.special_settings.get("apply_prosody_to_instruction", True)):
            rate = str(settings.GetOption("tts_prosody_rate") or "").strip()
            pitch = str(settings.GetOption("tts_prosody_pitch") or "").strip()
            rate_text = {
                "x-fast": "Speak extremely quickly.",
                "fast": "Speak quickly.",
                "medium": "Speak at a normal pace.",
                "slow": "Speak slowly.",
                "x-slow": "Speak extremely slowly.",
            }.get(rate)
            pitch_text = {
                "x-low": "Use a very low pitch.",
                "low": "Use a low pitch.",
                "medium": "Use a natural pitch.",
                "high": "Use a high pitch.",
                "x-high": "Use a very high pitch.",
            }.get(pitch)
            instruction = " ".join(filter(None, (instruction, rate_text, pitch_text)))
        if mode == "voice_design" and not instruction:
            instruction = "A clear, natural, expressive voice."
        return instruction

    def _non_streaming_mode(self, mode):
        selected = str(self.special_settings.get("model_text_mode", "auto")).lower()
        if selected == "full_text":
            return True
        if selected == "streaming_simulation":
            return False
        # Preserve the official wrapper defaults.
        return mode != "base"

    def _generation_kwargs(self):
        seed = self._int(self.special_settings.get("seed"), -1)
        if seed < 0:
            seed = int(torch.randint(1, 2**31 - 1, (1,)).item())
        torch.manual_seed(seed)
        if self.compute_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return {
            "do_sample": bool(self.special_settings.get("do_sample", True)),
            "temperature": max(0.01, min(2.0, self._float(self.special_settings.get("temperature"), 0.9))),
            "top_p": max(0.01, min(1.0, self._float(self.special_settings.get("top_p"), 1.0))),
            "top_k": max(0, min(500, self._int(self.special_settings.get("top_k"), 50))),
            "repetition_penalty": max(0.1, min(5.0, self._float(self.special_settings.get("repetition_penalty"), 1.05))),
            "subtalker_dosample": bool(self.special_settings.get("subtalker_do_sample", True)),
            "subtalker_temperature": max(0.01, min(2.0, self._float(self.special_settings.get("subtalker_temperature"), 0.9))),
            "subtalker_top_p": max(0.01, min(1.0, self._float(self.special_settings.get("subtalker_top_p"), 1.0))),
            "subtalker_top_k": max(0, min(500, self._int(self.special_settings.get("subtalker_top_k"), 50))),
            "max_new_tokens": max(32, min(16384, self._int(self.special_settings.get("max_new_tokens"), 2048))),
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([_StopOnEvent(self.stop_event)]),
        }

    def _default_speaker(self):
        language = str(self.special_settings.get("language", "auto")).lower()
        return {
            "zh": "Vivian",
            "ja": "Ono_Anna",
            "ko": "Sohee",
            "en": "Ryan",
        }.get(language, "Ryan")

    def _speaker(self):
        selected = str(settings.GetOption("tts_voice") or "")
        for speaker in PRESET_VOICES:
            if selected.lower() == speaker.lower():
                return speaker
        return self._default_speaker()

    def _generate_one(
        self,
        text,
        ref_audio=None,
        codec_streamer=None,
    ):
        model_name = self._get_model_name()
        mode = self._model_mode(model_name)
        language = self._language()
        kwargs = self._generation_kwargs()
        if codec_streamer is not None:
            kwargs["codec_streamer"] = codec_streamer
            # The incremental decoder already produced every sample. Avoid a
            # second full-utterance codec decode after streaming has ended.
            kwargs["decode_audio"] = False
        # Codec/audio streaming is independent of Qwen's text-conditioning
        # mode, so continue honoring the explicit/official model_text_mode.
        non_streaming_mode = self._non_streaming_mode(mode)
        if mode == "base":
            clone_prompt = self._clone_prompt(ref_audio)
            if codec_streamer is not None:
                codec_streamer.set_prefix_codes(
                    self._reference_code_prefixes(clone_prompt, 1)[0]
                )
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=clone_prompt,
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )
        elif mode == "custom_voice":
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=self._speaker(),
                instruct=self._instruction(mode),
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )
        else:
            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=self._instruction(mode),
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )
        if codec_streamer is not None and not wavs:
            self.sample_rate = int(sample_rate)
            return torch.zeros((1, 0), dtype=torch.float32)
        if not wavs or np.asarray(wavs[0]).size == 0:
            raise RuntimeError(f"{model_name} returned no audio.")
        self.sample_rate = int(sample_rate)
        return self._wave_tensor(wavs[0])

    def _generate_batch(self, texts, ref_audio=None, codec_streamer=None):
        texts = [str(text).strip() for text in texts if str(text).strip()]
        if not texts:
            return []
        model_name = self._get_model_name()
        mode = self._model_mode(model_name)
        language = self._language()
        languages = [language] * len(texts)
        kwargs = self._generation_kwargs()
        if codec_streamer is not None:
            kwargs["codec_streamer"] = codec_streamer
            kwargs["decode_audio"] = False
        non_streaming_mode = self._non_streaming_mode(mode)

        if mode == "base":
            clone_prompt = self._clone_prompt(ref_audio)
            if codec_streamer is not None:
                codec_streamer.set_prefix_codes(
                    self._reference_code_prefixes(clone_prompt, len(texts))
                )
            wavs, sample_rate = self.model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=clone_prompt,
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )
        elif mode == "custom_voice":
            wavs, sample_rate = self.model.generate_custom_voice(
                text=texts,
                language=languages,
                speaker=[self._speaker()] * len(texts),
                instruct=[self._instruction(mode)] * len(texts),
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )
        else:
            wavs, sample_rate = self.model.generate_voice_design(
                text=texts,
                language=languages,
                instruct=[self._instruction(mode)] * len(texts),
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )

        self.sample_rate = int(sample_rate)
        if codec_streamer is not None and not wavs:
            return []
        if not wavs or len(wavs) != len(texts):
            raise RuntimeError(
                f"{model_name} returned {len(wavs) if wavs else 0} waveforms "
                f"for a batch of {len(texts)} texts."
            )
        return [self._wave_tensor(wav) for wav in wavs]

    @staticmethod
    def _wave_tensor(audio):
        if isinstance(audio, torch.Tensor):
            wave = audio.detach().float().cpu()
        else:
            wave = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        if wave.ndim == 0:
            wave = wave.reshape(1, 1)
        elif wave.ndim == 1:
            wave = wave.unsqueeze(0)
        elif wave.ndim == 2 and wave.shape[0] > wave.shape[1] and wave.shape[1] <= 2:
            wave = wave.transpose(0, 1)
        if wave.ndim > 2:
            wave = wave.reshape(1, -1)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        return wave.clamp(-1.0, 1.0).contiguous()

    def _finish_audio(self, wave, call_plugin=True):
        wave = self._wave_tensor(wave)
        if settings.GetOption("tts_normalize") and wave.numel():
            wave, _ = audio_tools.normalize_audio_lufs(
                wave, self.sample_rate, -24.0, -16.0, 1.3, verbose=True
            )
            wave = self._wave_tensor(wave)
        volume = self._float(settings.GetOption("tts_volume"), 1.0)
        if volume != 1.0:
            wave = self._wave_tensor(audio_tools.change_volume(wave, volume))
        if call_plugin:
            import Plugins

            result = Plugins.plugin_custom_event_call(
                "plugin_tts_after_audio", {"audio": wave, "sample_rate": self.sample_rate}
            )
            if isinstance(result, dict) and result.get("audio") is not None:
                wave = self._wave_tensor(result["audio"])
        return wave

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        del remove_silence, silence_after_segments, normalize
        if not text or not text.strip():
            return torch.zeros((1, self.sample_rate // 10)), self.sample_rate
        with self.generation_lock:
            self.stop_event.clear()
            self.load()
            wave = self._finish_audio(self._generate_one(text.strip(), ref_audio))
            self.last_generation = {
                "audio": wave,
                "sample_rate": self.sample_rate,
                "text": text.strip(),
            }
            return wave, self.sample_rate

    @staticmethod
    def _split_long_unit(unit, limit):
        parts = []
        remaining = unit.strip()
        while len(remaining) > limit:
            split_at = max(
                remaining.rfind(marker, 0, limit + 1)
                for marker in (" ", ",", ";", ":", "，", "、", "；", "：")
            )
            if split_at < max(1, limit // 3):
                split_at = limit
            parts.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            parts.append(remaining)
        return parts

    def _segments_with_limit(self, text, limit):
        units = re.split(
            r"(?<=[!?。！？…])\s*|(?<=\.)\s+|[\r\n]+",
            text.strip(),
        )
        segments = []
        current = ""
        for unit in units:
            for part in self._split_long_unit(unit, limit):
                candidate = f"{current} {part}".strip()
                if current and len(candidate) > limit:
                    segments.append(current)
                    current = part
                else:
                    current = candidate
        if current:
            segments.append(current)
        return segments or [text.strip()]

    def _streaming_segments(self, text):
        limit = max(
            20,
            min(
                1000,
                self._int(
                    self.special_settings.get("streaming_segment_characters"),
                    180,
                ),
            ),
        )
        return self._segments_with_limit(text, limit)

    def _lookahead_segments(self, text):
        limit = max(
            30,
            min(
                500,
                self._int(
                    self.special_settings.get("streaming_lookahead_characters"),
                    80,
                ),
            ),
        )
        return self._segments_with_limit(text, limit)

    @staticmethod
    def _lookahead_groups(segments, maximum_batch_size):
        segments = list(segments)
        maximum_batch_size = max(1, int(maximum_batch_size))
        if not segments:
            return []
        group_count = (len(segments) + maximum_batch_size - 1) // maximum_batch_size
        base_size, larger_groups = divmod(len(segments), group_count)
        groups = []
        cursor = 0
        for group_index in range(group_count):
            size = base_size + (1 if group_index < larger_groups else 0)
            groups.append(segments[cursor:cursor + size])
            cursor += size
        return groups

    def _silence(self):
        milliseconds = max(0, min(5000, self._int(self.special_settings.get("pause_between_segments_ms"), 120)))
        return torch.zeros((1, int(self.sample_rate * milliseconds / 1000.0)), dtype=torch.float32)

    def _lookahead_silence(self):
        milliseconds = max(
            0,
            min(
                1000,
                self._int(
                    self.special_settings.get("streaming_lookahead_pause_ms"),
                    0,
                ),
            ),
        )
        return torch.zeros(
            (1, int(self.sample_rate * milliseconds / 1000.0)),
            dtype=torch.float32,
        )

    def _estimated_spoken_duration(self, text):
        """Conservative speech-duration estimate used only for stream buffering."""
        text = str(text or "").strip()
        if not text:
            return 0.0

        non_space_characters = len(re.sub(r"\s+", "", text))
        words = re.findall(r"[\w']+", text, flags=re.UNICODE)
        cjk_characters = re.findall(
            r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]",
            text,
        )
        language = str(self.special_settings.get("language", "auto") or "auto").lower()
        mostly_cjk = language.split("-")[0] in {"zh", "ja", "ko"}
        mostly_cjk = mostly_cjk or len(cjk_characters) >= max(1, non_space_characters // 4)

        if mostly_cjk:
            # Chinese/Japanese scripts do not have dependable whitespace word
            # boundaries. Four visible characters per second is deliberately
            # conservative for natural Qwen voices.
            base_duration = non_space_characters / 4.0
        else:
            # Use both word and character rates so long compounds and URLs do
            # not make the estimate unrealistically short.
            base_duration = max(len(words) / 2.2, non_space_characters / 16.0)

        sentence_pauses = len(re.findall(r"[.!?\u3002\uff01\uff1f\u2026]", text)) * 0.16
        short_pauses = len(re.findall(r"[,;:\uff0c\u3001\uff1b\uff1a]", text)) * 0.07
        rate_factor = 1.0
        if bool(self.special_settings.get("apply_prosody_to_instruction", True)):
            rate_factor = {
                "x-fast": 0.7,
                "fast": 0.85,
                "medium": 1.0,
                "slow": 1.3,
                "x-slow": 1.7,
            }.get(str(settings.GetOption("tts_prosody_rate") or "").lower(), 1.0)
        return max(
            0.4,
            (base_duration + sentence_pauses + short_pauses) * rate_factor,
        )

    def _streaming_rtf_key(self, batch_size=1):
        configuration = getattr(self, "loaded_configuration", None)
        if configuration:
            key = tuple(configuration)
        else:
            key = (DEFAULT_MODEL, getattr(self, "compute_device_str", "cuda"))
        if int(batch_size) > 1:
            key += ("batch", int(batch_size))
        return key

    def _expected_streaming_rtf(self, batch_size=1):
        batch_size = max(1, int(batch_size))
        configuration = getattr(self, "loaded_configuration", None)
        model_name = configuration[0] if configuration else DEFAULT_MODEL
        device = str(getattr(self, "compute_device_str", "cuda"))
        default_rtf = 2.2 if "0.6B" in model_name else 3.0
        if not device.startswith("cuda"):
            default_rtf = 8.0
        default_rtf /= batch_size
        history = getattr(self, "streaming_rtf_history", {}).get(
            self._streaming_rtf_key(batch_size)
        )
        if history is not None:
            # A small margin absorbs normal prompt-length and scheduler jitter.
            default_rtf = max(default_rtf, float(history) * 1.1)
        return max(0.1, min(10.0, default_rtf))

    def _adaptive_buffer_target(self, text, expected_rtf=None):
        configured_minimum = max(
            0.0,
            self._float(settings.GetOption("tts_streamed_min_play_time"), 0.0),
        )
        estimated_duration = self._estimated_spoken_duration(text)
        expected_rtf = max(
            1.0,
            min(10.0, float(expected_rtf or self._expected_streaming_rtf())),
        )
        safety_seconds = max(
            0.0,
            min(
                5.0,
                self._int(
                    self.special_settings.get("streaming_buffer_safety_ms"),
                    500,
                )
                / 1000.0,
            ),
        )

        # If generation takes R wall seconds for one second of audio, playback
        # can start after D * (1 - 1/R) seconds without exhausting the buffer
        # before the remaining D seconds have been generated. The estimate is
        # capped at the whole utterance; short phrases therefore simply wait
        # until generation completes instead of starting and immediately
        # underrunning.
        generated_lead = estimated_duration * (1.0 - 1.0 / expected_rtf)
        adaptive_target = min(
            estimated_duration,
            generated_lead + safety_seconds,
        )
        return max(configured_minimum, adaptive_target), estimated_duration

    def _remember_streaming_rtf(self, elapsed_seconds, audio_seconds, batch_size=1):
        if audio_seconds <= 0.0:
            return None
        observed = max(0.01, float(elapsed_seconds) / float(audio_seconds))
        history = getattr(self, "streaming_rtf_history", None)
        if history is None:
            history = {}
            self.streaming_rtf_history = history
        key = self._streaming_rtf_key(batch_size)
        previous = history.get(key)
        history[key] = observed if previous is None else previous * 0.7 + observed * 0.3
        return observed

    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(
                audio_device,
                source_sample_rate=self.sample_rate,
                start_playback_timeout=1.0,
                min_buffer_play_time=float(settings.GetOption("tts_streamed_min_play_time")),
                playback_channels=2,
                buffer_size=settings.GetOption("tts_streamed_chunk_size"),
                input_channels=1,
                dtype="float32",
                tag="tts",
            )

    def tts_streaming(self, text, ref_audio=None):
        self._ensure_special_settings()
        mode = str(self.special_settings.get("streaming_mode", "codec")).lower()
        if mode == "segment":
            return self.tts_streaming_segments(text, ref_audio)
        if mode in {"lookahead", "parallel", "parallel_lookahead"}:
            return self.tts_streaming_lookahead(text, ref_audio)
        return self.tts_streaming_codec(text, ref_audio)

    def tts_streaming_codec(self, text, ref_audio=None):
        if not text or not text.strip():
            return torch.zeros((1, 0)), self.sample_rate
        with self.generation_lock:
            self.stop_event.clear()
            self.load()
            self.init_audio_stream_playback()
            chunks = []
            streaming_volume = self._float(settings.GetOption("tts_volume"), 1.0)
            buffer_mode = str(
                self.special_settings.get("streaming_buffer_mode", "adaptive")
            ).lower()
            adaptive_buffering = buffer_mode != "fixed"
            expected_rtf = self._expected_streaming_rtf()
            buffer_target, estimated_duration = self._adaptive_buffer_target(
                text,
                expected_rtf,
            )
            generation_started = time.perf_counter()
            pending_playback = []
            pending_samples = 0
            first_packet_elapsed = None
            first_packet_audio_seconds = 0.0
            playback_started = bool(
                self.audio_streamer is not None
                and getattr(self.audio_streamer, "playback_thread", None) is not None
                and self.audio_streamer.playback_thread.is_alive()
            )

            if adaptive_buffering and not playback_started:
                print(
                    "Qwen3-TTS adaptive playback buffer: "
                    f"{buffer_target:.2f}s target for about "
                    f"{estimated_duration:.2f}s of speech (expected RTF {expected_rtf:.2f})."
                )

            def flush_pending_playback(force=False):
                nonlocal pending_samples, playback_started
                if not pending_playback:
                    return
                streamer = self.audio_streamer
                if streamer is None:
                    pending_playback.clear()
                    pending_samples = 0
                    return

                pending_wave = torch.cat(pending_playback, dim=-1)
                old_minimum = getattr(streamer, "min_buffer_play_time", None)
                if force and old_minimum is not None:
                    pending_duration = pending_samples / max(1, self.sample_rate)
                    # Make a final phrase shorter than either buffer target
                    # start immediately instead of waiting for the streamer's
                    # one-second inactivity timeout.
                    streamer.min_buffer_play_time = min(
                        float(old_minimum),
                        max(0.0, pending_duration - 1e-6),
                    )
                try:
                    streamer.add_audio_chunk(self.return_pcm_audio(pending_wave))
                finally:
                    if force and old_minimum is not None:
                        streamer.min_buffer_play_time = old_minimum
                pending_playback.clear()
                pending_samples = 0
                playback_started = True

            def emit_audio_chunk(audio, sample_rate):
                nonlocal buffer_target, pending_samples, playback_started
                nonlocal first_packet_elapsed, first_packet_audio_seconds
                self.sample_rate = int(sample_rate)
                # Retrospective loudness normalization and whole-wave plugins
                # cannot safely modify audio that has already been played.  As
                # with Chatterbox token streaming, apply only the live-safe
                # volume control and make last_generation exactly match what was
                # queued for playback.
                wave = self._wave_tensor(audio)
                if streaming_volume != 1.0:
                    wave = self._wave_tensor(
                        audio_tools.change_volume(wave, streaming_volume)
                    )
                if wave.numel() == 0:
                    return
                chunks.append(wave)
                if self.audio_streamer is None:
                    return
                if not adaptive_buffering or playback_started:
                    self.audio_streamer.add_audio_chunk(self.return_pcm_audio(wave))
                    return

                pending_playback.append(wave)
                pending_samples += int(wave.shape[-1])
                generated_seconds = pending_samples / max(1, self.sample_rate)

                elapsed = time.perf_counter() - generation_started
                if first_packet_elapsed is None:
                    first_packet_elapsed = elapsed
                    first_packet_audio_seconds = generated_seconds
                measured_audio = generated_seconds - first_packet_audio_seconds
                # Use the sustained post-prefill slope. Dividing total elapsed
                # time by an early 160 ms packet makes prompt setup look like a
                # permanently slow producer and unnecessarily buffers the
                # complete utterance.
                if measured_audio >= 1.0:
                    measured_rtf = min(
                        10.0,
                        max(
                            1.0,
                            (elapsed - first_packet_elapsed) / measured_audio,
                        ),
                    )
                    refined_target, _ = self._adaptive_buffer_target(
                        text,
                        max(expected_rtf, measured_rtf * 1.05),
                    )
                    buffer_target = max(buffer_target, refined_target)

                if generated_seconds + 1e-9 >= buffer_target:
                    flush_pending_playback()

            frames_per_chunk = max(
                1,
                min(
                    50,
                    self._int(self.special_settings.get("streaming_codec_frames"), 2),
                ),
            )
            context_frames = max(
                0,
                min(
                    100,
                    self._int(
                        self.special_settings.get("streaming_decoder_context_frames"),
                        25,
                    ),
                ),
            )
            codec_streamer = _QwenCodecAudioStreamer(
                self.model.model.speech_tokenizer,
                emit_audio_chunk,
                self.model.model.config.talker_config.codec_eos_token_id,
                frames_per_chunk=frames_per_chunk,
                context_frames=context_frames,
            )
            fallback_wave = self._generate_one(
                text.strip(),
                ref_audio,
                codec_streamer=codec_streamer,
            )
            if not chunks and fallback_wave.numel():
                emit_audio_chunk(fallback_wave, self.sample_rate)
            if adaptive_buffering and not playback_started:
                flush_pending_playback(force=True)

            final_wave = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, 0))
            audio_seconds = final_wave.shape[-1] / max(1, self.sample_rate)
            observed_rtf = self._remember_streaming_rtf(
                time.perf_counter() - generation_started,
                audio_seconds,
            )
            if observed_rtf is not None:
                print(
                    f"Qwen3-TTS generated {audio_seconds:.2f}s of streamed audio at "
                    f"RTF {observed_rtf:.2f} "
                    f"({'slower' if observed_rtf > 1.0 else 'faster'} than real time)."
                )
            self.last_generation = {
                "audio": final_wave,
                "sample_rate": self.sample_rate,
                "text": text.strip(),
            }
            return final_wave, self.sample_rate

    def tts_streaming_lookahead(self, text, ref_audio=None):
        if not text or not text.strip():
            return torch.zeros((1, 0)), self.sample_rate
        text = text.strip()
        segments = self._lookahead_segments(text)
        if len(segments) < 2:
            # A short indivisible utterance has no independent future work to
            # batch and retains the coherent single-lane codec path.
            return self.tts_streaming_codec(text, ref_audio)

        with self.generation_lock:
            self.stop_event.clear()
            self.load()
            self.init_audio_stream_playback()

            maximum_batch_size = max(
                2,
                min(
                    4,
                    self._int(
                        self.special_settings.get(
                            "streaming_lookahead_batch_size"
                        ),
                        3,
                    ),
                ),
            )
            groups = self._lookahead_groups(segments, maximum_batch_size)
            first_batch_size = len(groups[0])
            frames_per_chunk = max(
                1,
                min(
                    50,
                    self._int(
                        self.special_settings.get(
                            "streaming_lookahead_codec_frames"
                        ),
                        6,
                    ),
                ),
            )
            context_frames = max(
                0,
                min(
                    100,
                    self._int(
                        self.special_settings.get(
                            "streaming_decoder_context_frames"
                        ),
                        25,
                    ),
                ),
            )
            streaming_volume = self._float(settings.GetOption("tts_volume"), 1.0)
            adaptive_buffering = str(
                self.special_settings.get("streaming_buffer_mode", "adaptive")
            ).lower() != "fixed"
            expected_rtf = self._expected_streaming_rtf(first_batch_size)
            first_lane_expected_rtf = self._expected_streaming_rtf()
            buffer_target, estimated_duration = self._adaptive_buffer_target(
                segments[0],
                first_lane_expected_rtf,
            )
            generation_started = time.perf_counter()
            all_chunks = []
            pending_playback = []
            pending_samples = 0
            ordered_generated_samples = 0
            released_segments = 0
            first_ordered_packet_elapsed = None
            first_ordered_packet_audio_seconds = 0.0
            playback_started = bool(
                self.audio_streamer is not None
                and getattr(self.audio_streamer, "playback_thread", None) is not None
                and self.audio_streamer.playback_thread.is_alive()
            )

            print(
                "Qwen3-TTS parallel lookahead: "
                f"{len(segments)} text segments in batches "
                f"{[len(group) for group in groups]} (expected aggregate RTF "
                f"{expected_rtf:.2f})."
            )
            if adaptive_buffering and not playback_started:
                print(
                    "Qwen3-TTS lookahead playback buffer: "
                    f"{buffer_target:.2f}s target for about "
                    f"{estimated_duration:.2f}s in the first lane "
                    f"(expected lane RTF {first_lane_expected_rtf:.2f})."
                )

            def flush_pending_playback(force=False):
                nonlocal pending_samples, playback_started
                if not pending_playback:
                    return
                streamer = self.audio_streamer
                if streamer is None:
                    pending_playback.clear()
                    pending_samples = 0
                    return
                pending_wave = torch.cat(pending_playback, dim=-1)
                old_minimum = getattr(streamer, "min_buffer_play_time", None)
                if force and old_minimum is not None:
                    pending_duration = pending_samples / max(1, self.sample_rate)
                    streamer.min_buffer_play_time = min(
                        float(old_minimum),
                        max(0.0, pending_duration - 1e-6),
                    )
                try:
                    streamer.add_audio_chunk(self.return_pcm_audio(pending_wave))
                finally:
                    if force and old_minimum is not None:
                        streamer.min_buffer_play_time = old_minimum
                pending_playback.clear()
                pending_samples = 0
                playback_started = True

            def queue_ordered_wave(wave):
                nonlocal buffer_target, pending_samples, playback_started
                nonlocal ordered_generated_samples
                nonlocal first_ordered_packet_elapsed
                nonlocal first_ordered_packet_audio_seconds
                if wave.numel() == 0:
                    return
                all_chunks.append(wave)
                ordered_generated_samples += int(wave.shape[-1])
                if self.audio_streamer is None:
                    return
                if not adaptive_buffering or playback_started:
                    self.audio_streamer.add_audio_chunk(
                        self.return_pcm_audio(wave)
                    )
                    return

                pending_playback.append(wave)
                pending_samples += int(wave.shape[-1])
                ordered_seconds = pending_samples / max(1, self.sample_rate)
                generated_seconds = ordered_generated_samples / max(1, self.sample_rate)
                elapsed = time.perf_counter() - generation_started
                if first_ordered_packet_elapsed is None:
                    first_ordered_packet_elapsed = elapsed
                    first_ordered_packet_audio_seconds = generated_seconds
                measured_audio = (
                    generated_seconds - first_ordered_packet_audio_seconds
                )
                if measured_audio >= 1.0:
                    measured_rtf = min(
                        10.0,
                        max(
                            1.0,
                            (elapsed - first_ordered_packet_elapsed)
                            / measured_audio,
                        ),
                    )
                    refined_target, _ = self._adaptive_buffer_target(
                        segments[0],
                        max(first_lane_expected_rtf, measured_rtf * 1.05),
                    )
                    buffer_target = max(buffer_target, refined_target)
                if ordered_seconds + 1e-9 >= buffer_target:
                    flush_pending_playback()

            talker_config = self.model.model.config.talker_config
            for group in groups:
                if self.stop_event.is_set():
                    break
                lane_chunks = [[] for _ in group]
                lane_chunk_cursors = [0] * len(group)
                lane_complete = [False] * len(group)
                next_lane_to_release = 0

                def release_available_lane_chunks(lane_index):
                    cursor = lane_chunk_cursors[lane_index]
                    available = lane_chunks[lane_index][cursor:]
                    if not available:
                        return
                    wave = (
                        available[0]
                        if len(available) == 1
                        else torch.cat(available, dim=-1)
                    )
                    queue_ordered_wave(wave)
                    lane_chunk_cursors[lane_index] = len(lane_chunks[lane_index])

                def advance_completed_lanes():
                    nonlocal next_lane_to_release, released_segments
                    while (
                        next_lane_to_release < len(group)
                        and lane_complete[next_lane_to_release]
                    ):
                        lane = next_lane_to_release
                        release_available_lane_chunks(lane)
                        released_segments += 1
                        if lane_chunks[lane]:
                            if released_segments < len(segments):
                                silence = self._lookahead_silence()
                                if silence.numel():
                                    queue_ordered_wave(silence)
                        next_lane_to_release += 1
                        if next_lane_to_release < len(group):
                            release_available_lane_chunks(next_lane_to_release)

                def emit_lane_audio(lane_index, audio, sample_rate):
                    self.sample_rate = int(sample_rate)
                    wave = self._wave_tensor(audio)
                    if streaming_volume != 1.0:
                        wave = self._wave_tensor(
                            audio_tools.change_volume(wave, streaming_volume)
                        )
                    if wave.numel():
                        lane_chunks[lane_index].append(wave)
                        if lane_index == next_lane_to_release:
                            release_available_lane_chunks(lane_index)

                def finish_lane(lane_index):
                    lane_complete[lane_index] = True
                    if lane_index == next_lane_to_release:
                        advance_completed_lanes()

                codec_streamer = _QwenBatchCodecAudioStreamer(
                    self.model.model.speech_tokenizer,
                    emit_lane_audio,
                    finish_lane,
                    talker_config.codec_eos_token_id,
                    talker_config.codec_pad_id,
                    batch_size=len(group),
                    frames_per_chunk=frames_per_chunk,
                    context_frames=context_frames,
                )
                self._generate_batch(
                    group,
                    ref_audio,
                    codec_streamer=codec_streamer,
                )
                codec_streamer.end()
                advance_completed_lanes()
                if (
                    next_lane_to_release != len(group)
                    and not self.stop_event.is_set()
                ):
                    raise RuntimeError(
                        "Qwen3-TTS lookahead generation did not finish every batch lane."
                    )

            if adaptive_buffering and not playback_started:
                flush_pending_playback(force=True)

            final_wave = (
                torch.cat(all_chunks, dim=-1)
                if all_chunks
                else torch.zeros((1, 0), dtype=torch.float32)
            )
            audio_seconds = final_wave.shape[-1] / max(1, self.sample_rate)
            observed_rtf = self._remember_streaming_rtf(
                time.perf_counter() - generation_started,
                audio_seconds,
                first_batch_size,
            )
            if observed_rtf is not None:
                print(
                    f"Qwen3-TTS lookahead generated {audio_seconds:.2f}s of "
                    f"ordered audio at aggregate RTF {observed_rtf:.2f}."
                )
            self.last_generation = {
                "audio": final_wave,
                "sample_rate": self.sample_rate,
                "text": text,
            }
            return final_wave, self.sample_rate

    def tts_streaming_segments(self, text, ref_audio=None):
        if not text or not text.strip():
            return torch.zeros((1, 0)), self.sample_rate
        with self.generation_lock:
            self.stop_event.clear()
            self.load()
            self.init_audio_stream_playback()
            chunks = []
            segments = self._streaming_segments(text)
            for index, segment in enumerate(segments):
                if self.stop_event.is_set():
                    break
                wave = self._finish_audio(self._generate_one(segment, ref_audio))
                chunks.append(wave)
                if self.audio_streamer is not None and wave.numel():
                    self.audio_streamer.add_audio_chunk(self.return_pcm_audio(wave))
                if index < len(segments) - 1:
                    silence = self._silence()
                    if silence.numel():
                        chunks.append(silence)
                        if self.audio_streamer is not None:
                            self.audio_streamer.add_audio_chunk(self.return_pcm_audio(silence))
            final_wave = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, 0))
            self.last_generation = {
                "audio": final_wave,
                "sample_rate": self.sample_rate,
                "text": text.strip(),
            }
            return final_wave, self.sample_rate

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def save_voice(self):
        audio = self.last_generation.get("audio")
        if audio is None or self._wave_tensor(audio).numel() == 0:
            raise RuntimeError("No Qwen3-TTS generation is available to save as a voice reference.")
        stem = "qwen3_tts_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        audio_path = VOICES_PATH / f"{stem}.wav"
        text_path = VOICES_PATH / f"{stem}.txt"
        audio_path.write_bytes(self.return_wav_file_binary(audio, self.last_generation.get("sample_rate")))
        reference_text = str(self.last_generation.get("text") or "").strip()
        if reference_text:
            text_path.write_text(reference_text, encoding="utf-8")
        print(f"Saved Qwen3-TTS clone reference to {audio_path}")
        return str(audio_path)

    def play_audio(self, audio, device=None):
        if device is None:
            device = settings.GetOption("device_default_out_index")
        secondary = None
        if settings.GetOption("tts_use_secondary_playback"):
            secondary = settings.GetOption("tts_secondary_playback_device")
            if secondary == -1:
                secondary = settings.GetOption("device_default_out_index")
        audio_tools.play_audio(
            self._wave_tensor(audio),
            device,
            source_sample_rate=self.sample_rate,
            audio_device_channel_num=1,
            target_channels=1,
            input_channels=1,
            dtype="float32",
            tensor_sample_with=4,
            tensor_channels=1,
            secondary_device=secondary,
            stop_play=not settings.GetOption("tts_allow_overlapping_audio"),
            tag="tts",
        )

    def return_wav_file_binary(self, audio, sample_rate=SAMPLE_RATE):
        array = self._wave_tensor(audio).squeeze(0).numpy()
        array = np.rint(np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        write_wav(buffer, int(sample_rate or self.sample_rate), array)
        return buffer.getvalue()

    def return_pcm_audio(self, audio):
        return self._wave_tensor(audio).squeeze(0).numpy().astype("<f4", copy=False).tobytes()


MODEL_LIST_FLAT = {
    model_name
    for model_names in MODEL_LIST.values()
    for model_name in model_names
}
