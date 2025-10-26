import gc
import io
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

import Plugins
import Utilities
import audio_tools
import downloader
import settings
# from .chatterbox.tts import ChatterboxTTS
from .chatterbox import SUPPORTED_LANGUAGES
from .chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from .chatterbox.vc import ChatterboxVC
from .chatterbox.models.s3tokenizer import SPEECH_VOCAB_SIZE
from .. import languageClassification
from ..Singleton import SingletonMeta
import re

# ONNX helper imports (lazy-used inside methods to avoid hard dependency at import time)
try:
    # Use reusable engine class and constants; engine is instantiated lazily
    from .chatterbox.onnx import (
        ChatterboxOnnxTTS as OnnxEngine,
        S3GEN_SR as ONNX_SAMPLE_RATE,
    )
except Exception:
    OnnxEngine = None
    ONNX_SAMPLE_RATE = 24000

cache_path = Path(Path.cwd() / ".cache" / "chatterbox-tts-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

TTS_MODEL_LINKS = {
    # Models
    "chatterbox-multilingual": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/chatterbox-tts/chatterbox-multilingual.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/chatterbox-tts/chatterbox-multilingual.zip",
            "https://s3.libs.space:9000/ai-models/chatterbox-tts/chatterbox-multilingual.zip",
        ],
        "checksum": "34140a601d03cc1faad454d894389a4d4876b84caca046493e6b270a689ce8af",
        "file_checksums": {
            "Cangjie5_TC.json": "7073fd9de919443ae88e0bd2449917a65fe54898a4413ed1edcc4b67f28bce8c",
            "conds.pt": "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e",
            "grapheme_mtl_merged_expanded_v1.json": "df81a7ca7c31796cbe97f7a7142d5a53b12e88e12417ebe98f66602cafaf0461",
            "mtl_tokenizer.json": "e7f9364e2c279b2de19f417a83624d9887532a56daec2ddddac470cc71693253",
            "s3gen.safetensors": "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e",
            "spacy_ontonotes\\features.msgpack": "fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18",
            "spacy_ontonotes\\weights.npz": "5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e",
            "t3_mtl23ls_v2.safetensors": "b1237586127ce98e7800a68e49938eb5092846862aabcb6e17b2fda7889a6c75",
            "tokenizer.json": "d71e3a44eabb1784df9a68e9f95b251ecbf1a7af6a9f50835856b2ca9d8c14a5",
            "ve.safetensors": "f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c"
        },
        "path": "chatterbox-multilingual",
    },
    "chatterbox-multilingual-onnx": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/chatterbox-tts/chatterbox-multilingual-onnx.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/chatterbox-tts/chatterbox-multilingual-onnx.zip",
            "https://s3.libs.space:9000/ai-models/chatterbox-tts/chatterbox-multilingual-onnx.zip",
        ],
        "checksum": "7e8223dea46bcf81d43301374b8da5aa0950c5218eb2623c2bf663e9b2cc726f",
        "file_checksums": {
            "Cangjie5_TC.json": "7073fd9de919443ae88e0bd2449917a65fe54898a4413ed1edcc4b67f28bce8c",
            "default_voice.wav": "3ebc531cdaba358a327099c1c4f0448026719957bcf4d8e9868767f227e02f4e",
            "generation_config.json": "1b6fbb953861089ebe7da64df46eeef570d53f47a44b7cc1b4d543669fc9cd50",
            "onnx\\conditional_decoder.onnx": "1656d0d31332bae1854839959a3139300ebb67c178651dfa3f8c5fbfa5351351",
            "onnx\\conditional_decoder.onnx_data": "51d58345a272747665ec9d5bb61e01835258a940e321a288582ac4c18cf01b5a",
            "onnx\\embed_tokens.onnx": "f785819ca4f6271262d5bb8971d62796c3a909e3b031982c113dbe83a4c3b854",
            "onnx\\embed_tokens.onnx_data": "2a15f7dd73b2ee47f6edf87740324011594b5a528ed6471ae55e327ed6cad68c",
            "onnx\\language_model.onnx": "861a34585605e8ad671051788afc495dcbeaee833a41523a1b33aded9c3babc7",
            "onnx\\language_model.onnx_data": "b3556d41085196c122b7197e4d44ec4475b6d7cfe0971a70faa95caa38ad787a",
            "onnx\\language_model_fp16.onnx": "0c36a5bbbc2a4ed8c345033896612cd320fd0971a0f5e6447ab4cdd2d7f22e36",
            "onnx\\language_model_fp16.onnx_data": "16dca11ae994e78427fa3090cc6faf347a15988ca40809c1bd9f2721f3b759a0",
            "onnx\\language_model_q4.onnx": "7f8cdca83b2493536cbf3acf421199808a3d68736f55f4eabd20ef8a99da4313",
            "onnx\\language_model_q4.onnx_data": "e79ab8784122a501718868b9631ff46e151c552d9b24e50f25d721f375e3526c",
            "onnx\\language_model_q4f16.onnx": "3b78e9235be5e2e2a811e482399155cb30415f6d87c98c21d12bf48843fc928f",
            "onnx\\language_model_q4f16.onnx_data": "bdbc79504d20742b5d028074b4f1cdca8872e013fdfbbcea6b8b03154fe85a42",
            "onnx\\speech_encoder.onnx": "8f1c8a0f89b77bf9cd5dd8f2e034eb2c79dc00fe70d41196b28c257643b00ccb",
            "onnx\\speech_encoder.onnx_data": "92f8f290fc9720e169bc2412c507209e20b03f6564bc3243739e25c56f7dfb8f",
            "tokenizer.json": "4abe9b558c1ea02170c2f11de6b1ec9e0dc6f75bc63566913a3e93929d91d035",
            "tokenizer_config.json": "b35967f93e30313d05fc9d520721ca9f671aaa5b3edbb03059aed3ff68b4c4c0"
        },
        "path": "chatterbox-multilingual-onnx",
    },
    # Default Voices
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
            "tiktok_jessie.wav": "5a26de921ea3e7c1ce1bfd2344fb107781def9366b56e2f583c7500a1052dbbd"
        },
        "path": "voices",
    },
}

model_list = {
    "Default": ["chatterbox-multilingual"],
    "ONNX": ["chatterbox-multilingual-onnx"],
}

class Chatterbox(metaclass=SingletonMeta):
    model = None
    vc_model = None
    sample_rate = 24000
    last_generation = {"audio": None, "sample_rate": None}
    voice_list = []
    audio_streamer = None
    split_patterns = {
        'fast': r'(?:[\.?！。？！,，;；](?=[ \n])|[\n])+',
        'slow': r'(?:\.(?=[ \n])|[\n?!。])+',
    }
    language_code_converter = None
    # Configurable chunking parameters
    chunk_goal_length = 130
    chunk_max_length = 170
    chunk_jitter = 0
    chunk_custom_split_chars = ","
    chunk_valid_ending_chars = ".;!?！。？！\n\""

    download_state = {"is_downloading": False}
    compute_device = "cpu"
    special_settings = {
        "precision": "float32",  # can be "float16" or "float32"
        "language": "en",
        "streaming_mode": "segment", # can be "segment" or "token"

        "max_new_tokens": 512,
        "repetition_penalty": 1.9,

        "seed": -1,
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg": 0.5,
    }

    # Cache for prepared voice conditionals: key -> Conditionals
    voice_conds_cache = {}
    _last_device_str = None
    # Track last loaded precision dtype to conditionally reload
    _loaded_precision_dtype = None

    # ONNX runtime members (lazy)
    _onnx_sessions = None  # dict of sessions (legacy)
    _onnx_tokenizer = None
    _onnx_tokenizer_dir = None
    _onnx_model_dir = None
    _onnx_mapping_path = None
    _onnx_engine = None

    def __init__(self):
        self.set_compute_device(settings.GetOption("tts_ai_device"))
        if not self.voice_list:
            self.update_voices()
        self.language_code_converter = Utilities.LanguageCodeConverter()

    def _get_model_type(self):
        model = self._get_model_name()
        if model.endswith("-onnx"):
            return "onnx"
        else:
            return "transformer"

    def set_compute_device(self, device):
        prev_device = getattr(self, 'compute_device_str', None)
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            #device = torch.device(self.compute_device_str)
            device = self.compute_device_str
        # Clear cached conditionals if device changes (device-bound tensors)
        if prev_device is not None and prev_device != self.compute_device_str:
            self.voice_conds_cache.clear()
        self.compute_device = device

    def set_special_setting(self, special_settings):
        self.special_settings = special_settings

    def _ensure_special_settings(self):
        # ensure special settings are in global settings
        special_settings = settings.GetOption("special_settings")
        if not isinstance(special_settings, dict):
            special_settings = {}

        tts_cfg = special_settings.get("tts_chatterbox")
        if isinstance(tts_cfg, dict):
            # Merge defaults to ensure new keys exist
            merged = {**self.special_settings, **tts_cfg}
            self.special_settings = merged
        else:
            # add without dropping other keys
            special_settings["tts_chatterbox"] = self.special_settings
            settings.SetOption("special_settings", special_settings)

    def _split_segment(self, segment, goal_length, custom_chars, valid_ending_chars):
        # Improved splitting for segments that are too long
        segments = []
        while len(segment) > goal_length:
            split_point = -1
            # Try custom split chars first
            if custom_chars:
                split_points = [segment.rfind(char, 0, goal_length) for char in custom_chars]
                split_points = [p for p in split_points if p != -1]
                if split_points:
                    split_point = max(split_points) + 1
            # Fallback to space
            if split_point == -1:
                split_point = segment.rfind(' ', 0, goal_length)
                if split_point == -1:
                    split_point = goal_length
            new_segment = segment[:split_point].strip()
            segment = segment[split_point:].strip()
            if new_segment:
                segments.append(new_segment)
        if segment:
            segments.append(segment)
        return segments

    def chunk_up_text(self, text, goal_length=None, max_length=None, jitter=None, custom_split_chars=None):
        # Bark-inspired chunking, improved for Zonos
        if goal_length is None:
            goal_length = self.chunk_goal_length
        if max_length is None:
            max_length = self.chunk_max_length
        if jitter is None:
            jitter = self.chunk_jitter
        if custom_split_chars is None:
            custom_split_chars = self.chunk_custom_split_chars
        valid_ending_chars = self.chunk_valid_ending_chars + custom_split_chars

        if jitter > 0:
            import random
            goal_length = random.randint(goal_length - jitter, goal_length + jitter)
            max_length = random.randint(max_length - jitter, max_length + jitter)

        # Normalize text
        import re
        text = re.sub(r"\n\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[“”]", '"', text)

        # Bark's split_general_purpose logic
        rv = []
        in_quote = False
        current = ""
        split_pos = []
        pos = -1
        end_pos = len(text) - 1

        def seek(delta):
            nonlocal pos, in_quote, current
            is_neg = delta < 0
            for _ in range(abs(delta)):
                if is_neg:
                    pos -= 1
                    current = current[:-1]
                else:
                    pos += 1
                    current += text[pos]
                if text[pos] == '"':
                    in_quote = not in_quote
            return text[pos]

        def peek(delta):
            p = pos + delta
            return text[p] if p < end_pos and p >= 0 else ""

        def commit():
            nonlocal rv, current, split_pos
            rv.append(current)
            current = ""
            split_pos = []

        while pos < end_pos:
            c = seek(1)
            # force split if too long
            if len(current) >= max_length:
                if len(split_pos) > 0 and len(current) > (goal_length / 2):
                    d = pos - split_pos[-1]
                    seek(-d)
                else:
                    while c not in ";!?.\n " and pos > 0 and len(current) > goal_length:
                        c = seek(-1)
                commit()
            # sentence boundaries
            elif not in_quote and (c in ";!?\n" or (c == "." and peek(1) in "\n ")):
                while (
                        pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?."
                ):
                    c = seek(1)
                split_pos.append(pos)
                if len(current) >= goal_length:
                    commit()
            elif in_quote and peek(1) == '"' and peek(2) in "\n ":
                seek(2)
                split_pos.append(pos)
        rv.append(current)

        # Clean up
        rv = [s.strip() for s in rv]
        rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s.,;:!?]*$", s)]

        # Post-process: merge/split segments as needed
        i = 0
        while i < len(rv):
            if not rv[i][-1] in valid_ending_chars:
                if any(char in custom_split_chars for char in rv[i]) and custom_split_chars:
                    if i < len(rv) - 1:
                        rv[i] += ' ' + rv[i + 1]
                        rv.pop(i + 1)
                    continue
            i += 1

        final_segments = []
        i = 0
        while i < len(rv):
            current_segment = rv[i]
            if i < len(rv) - 1 and current_segment[-1] not in valid_ending_chars:
                next_segment = rv[i + 1]
                combined_segment = current_segment + " " + next_segment
                if len(combined_segment) <= max_length:
                    rv[i] = combined_segment
                    rv.pop(i + 1)
                    continue
                else:
                    if len(current_segment) > max_length:
                        current_segment = self._split_segment(current_segment, goal_length, custom_split_chars, valid_ending_chars)
            elif len(current_segment) > max_length:
                current_segment = self._split_segment(current_segment, goal_length, custom_split_chars, valid_ending_chars)
            if current_segment:
                final_segments.extend(current_segment if isinstance(current_segment, list) else [current_segment])
            i += 1

        return final_segments

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": cache_path,
            "model_link_dict": TTS_MODEL_LINKS,
            "model_name": model_name,
            "title": "Text 2 Speech (Chatterbox TTS)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    def stop(self):
        print("TTS Stop requested")
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def release_model(self):
        if self.model is not None:
            if hasattr(self.model, 'model'):
                del self.model.model
            if hasattr(self.model, 'feature_extractor'):
                del self.model.feature_extractor
            if hasattr(self.model, 'hf_tokenizer'):
                del self.model.hf_tokenizer
            del self.model
        # Clear cached conditionals to avoid holding device tensors
        self.voice_conds_cache.clear()
        # Tear down ONNX resources
        self._release_onnx()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Reset precision tracker on release
        self._loaded_precision_dtype = None

    def garbage_collect(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


    def _get_model_name(self):
        model = "chatterbox-multilingual"
        if len(settings.GetOption('tts_model')) == 2:
            #language = settings.GetOption('tts_model')[0]
            model = settings.GetOption('tts_model')[1]
            # remove language part from string example: " (en & zh)"
            model = re.sub(r'\(.*?\)', '', model).strip()

        if "custom" in model:
            return model

        if model == "" or model not in TTS_MODEL_LINKS:
            model = "chatterbox-multilingual"

        return model

    def load(self):
        desired_precision = self.special_settings.get("precision", "float32")
        desired_dtype = self._precision_string_to_dtype(desired_precision)
        self.load_model(dtype=desired_dtype)

    def _precision_string_to_dtype(self, precision: str):
        """Map a user precision string to a torch.dtype, with device-aware fallback."""
        if not isinstance(precision, str):
            precision = "float32"
        p = precision.strip().lower()
        dtype = None
        if p in ("fp16", "float16", "half", "16"):
            dtype = torch.float16
        elif p in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif p in ("fp32", "float32", "32", "single", "full"):
            dtype = torch.float32
        else:
            # Default based on device: prefer fp16 on CUDA, otherwise fp32
            dtype = torch.float16 if self.compute_device_str == "cuda" else torch.float32

        # Device-aware fallback: if not CUDA, stick to float32 for safety
        if self.compute_device_str != "cuda" and dtype != torch.float32:
            print(f"Precision '{precision}' not supported/performance-safe on {self.compute_device_str}; using float32.")
            dtype = torch.float32
        return dtype

    def _ensure_model_for_precision(self):
        """Ensure model is loaded in the precision requested by special_settings."""
        desired_precision = self.special_settings.get("precision", "float32")
        desired_dtype = self._precision_string_to_dtype(desired_precision)
        # If model not loaded or precision changed, (re)load
        if self.model is None or self._loaded_precision_dtype != desired_dtype:
            # Reload in new precision
            self.load_model(dtype=desired_dtype)

    def load_model(self, dtype=None):
        model = self._get_model_name()
        self.set_compute_device(settings.GetOption('tts_ai_device'))
        if "custom" not in model:
            model_directory = Path(cache_path / TTS_MODEL_LINKS[model]["path"])
        else:
            model_directory = Path(cache_path / model)
            os.makedirs(model_directory, exist_ok=True)
        if "custom" not in model:
            self.download_model(model)
        self.download_model("voices")

        # Determine dtype: from arg or special settings
        if dtype is None:
            desired_precision = self.special_settings.get("precision", "float32")
            dtype = self._precision_string_to_dtype(desired_precision)

        # If device cannot handle selected precision, fallback handled in helper
        self.release_model()
        if model.endswith("-onnx"):
            if self._onnx_engine is None:
                self._ensure_onnx(model)
                self._loaded_precision_dtype = dtype
        else:
            if self.model is None:
                print(f"Loading Chatterbox TTS model {model} on device {self.compute_device_str} with precision {dtype}")
                self.model = ChatterboxMultilingualTTS.from_local(ckpt_dir=str(Path(model_directory).resolve()), device=self.compute_device_str, dtype=dtype)
                self.vc_model = ChatterboxVC.from_local(ckpt_dir=str(Path(model_directory).resolve()), device=self.compute_device_str, dtype=dtype)
                self._loaded_precision_dtype = dtype

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        return tuple([{"language": language, "models": models} for language, models in self.list_models().items()])

    def _get_voices(self):
        return self.voice_list

    def update_voices(self):
        # find all voices that have a .wav or .mp3 file
        voice_files = [f.stem for f in voices_path.iterdir() if f.is_file() and (f.suffix == ".wav" or f.suffix == ".mp3")]

        voice_list = []
        for voice_id in voice_files:
            wav_file = voices_path / f"{voice_id}.wav"
            mp3_file = voices_path / f"{voice_id}.mp3"

            if wav_file.exists() or mp3_file.exists():
                audio_file = wav_file if wav_file.exists() else mp3_file
                voice_list.append({"name": voice_id, "audio_filename": str(audio_file.resolve())})
        self.voice_list = voice_list

    def list_voices(self):
        self.update_voices()
        return [voice["name"] for voice in self._get_voices()]

    def get_voice_by_name(self, voice_name):
        for voice in self._get_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def _voice_cache_key(self, ref_audio_path: str) -> str:
        try:
            abs_path = os.path.abspath(ref_audio_path)
            st = os.stat(abs_path)
            # include device in key because tensors are device-bound
            return f"{self.compute_device_str}|{abs_path}|{st.st_size}|{int(st.st_mtime)}"
        except Exception:
            return f"{self.compute_device_str}|{ref_audio_path}|unknown"

    def _ensure_voice_conditionals(self, ref_audio: str, exaggeration: float = 0.5):
        """Prepare or reuse cached conditionals for the given reference audio.
        Stores and reuses per-device cache to avoid repeated voice embedding work.
        """
        if ref_audio is None:
            return
        key = self._voice_cache_key(ref_audio)
        cached = self.voice_conds_cache.get(key)
        if cached is not None:
            # Reuse cached conditionals
            self.model.conds = cached
            return
        # Not cached yet: prepare and cache
        self.model.prepare_conditionals(ref_audio, exaggeration=exaggeration)
        # Cache currently prepared conds (already on the correct device)
        self.voice_conds_cache[key] = self.model.conds

    # --- Voice-tag utilities ---
    def _resolve_main_voice_audio(self, ref_audio):
        """Resolve the audio file for the 'main' voice. Prefer explicit ref_audio, otherwise use settings."""
        if ref_audio is not None:
            return ref_audio
        voice_name = settings.GetOption('tts_voice')
        selected_voice = self.get_voice_by_name(voice_name)
        if selected_voice is None:
            selected_voice = self.get_voice_by_name("en_1")
        if selected_voice is not None:
            return selected_voice.get("audio_filename")
        return None

    def _build_voice_map(self, main_ref_audio):
        """Build a map of voice name -> audio file path, including 'main'."""
        voices_map = {v["name"]: v["audio_filename"] for v in self._get_voices()}
        voices_map["main"] = self._resolve_main_voice_audio(main_ref_audio)
        return voices_map

    def _parse_voice_tagged_text(self, text):
        """Parse text containing [voice_name] tags into ordered (voice, content) pairs.
        Rules:
        - Only tags at the start of a line (optionally preceded by spaces) are recognized.
        - Inline [brackets] elsewhere are treated as normal text.
        - Consecutive lines without a new tag continue the current voice segment.
        - Leading text before the first tag belongs to 'main'.
        """
        if not isinstance(text, str) or text.strip() == "":
            return []
        # Normalize newlines to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")
        segments = []
        current_voice = "main"
        buf = []
        # Support optional BOM/zero-width spaces before the tag; capture [voice] and trailing text
        tag_re = re.compile(r'^[\ufeff\u200b\s]*\[([^]]+)\]\s*(.*)$')

        def flush():
            nonlocal buf, current_voice
            if buf:
                content = "\n".join(buf).strip()
                if content:
                    segments.append((current_voice, content))
                buf = []

        for raw_line in lines:
            # Strip leading BOM/zero-width spaces before matching
            line = raw_line.lstrip('\ufeff\u200b')
            m = tag_re.match(line)
            if m:
                # New voice tag at start of line: flush previous, switch voice
                flush()
                current_voice = m.group(1).strip()
                rest = m.group(2)
                if rest is not None and rest.strip():
                    buf.append(rest.strip())
            else:
                # Continuation of current voice text
                if line.strip():
                    buf.append(line.strip())
        # Flush remaining
        flush()
        return segments

    def _ensure_onnx(self, model: str):
        """Lazy-create ONNX TTS engine when backend is set to 'onnx'."""
        if self._onnx_engine is not None:
            return
        # Dynamic import fallback if top-level import failed
        Engine = None
        try:
            from .chatterbox.onnx import ChatterboxOnnxTTS as _Engine
            Engine = _Engine
        except Exception:
            try:
                import importlib
                mod = importlib.import_module('.chatterbox.onnx', package=__package__)
                Engine = getattr(mod, 'ChatterboxOnnxTTS', None)
            except Exception:
                Engine = None
        if Engine is None:
            # Last-resort: load by file path to avoid package import issues
            try:
                import importlib.util
                module_path = Path(__file__).parent / 'chatterbox' / 'onnx.py'
                spec = importlib.util.spec_from_file_location('chatterbox_onnx_dyn', str(module_path))
                mod = importlib.util.module_from_spec(spec)
                if spec and spec.loader:
                    spec.loader.exec_module(mod)
                    Engine = getattr(mod, 'ChatterboxOnnxTTS', None)
            except Exception:
                Engine = None
        if Engine is None:
            raise RuntimeError("ONNX backend selected but ONNX engine is not available.")
        try:
            import onnxruntime
        except Exception as ex:
            raise RuntimeError(f"ONNX backend dependency onnxruntime not available: {ex}")

        model_dir = Path(cache_path / model).resolve()
        # Select providers based on device
        try:
            avail = onnxruntime.get_available_providers()
            if self.compute_device_str == "cuda" and "CUDAExecutionProvider" in avail:
                print("Using ONNX Runtime with CUDAExecutionProvider")
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("Using ONNX Runtime with CPUExecutionProvider")
                providers = ["CPUExecutionProvider"]
        except Exception:
            providers = None
        self._onnx_engine = Engine(str(model_dir), providers=providers)

    def _release_onnx(self):
        self._onnx_sessions = None
        self._onnx_tokenizer = None
        self._onnx_tokenizer_dir = None
        self._onnx_model_dir = None
        self._onnx_mapping_path = None
        self._onnx_engine = None

    def tts_generator(self, text, ref_audio=None, language="en"):
        #with self.stop_flag_lock:
        #    self.stop_flag = False
        self._ensure_special_settings()

        backend = self._get_model_type().lower()
        if backend == "onnx":
            return self._tts_generator_onnx(text, ref_audio=ref_audio, language=language)

        try:
            self.set_compute_device(settings.GetOption('tts_ai_device'))
            # Ensure model precision matches current setting
            self._ensure_model_for_precision()

            tts_volume = settings.GetOption("tts_volume")
            tts_normalize = settings.GetOption("tts_normalize")

            if ref_audio is None:
                voice_name = settings.GetOption('tts_voice')
                selected_voice = self.get_voice_by_name(voice_name)
                if selected_voice is None:
                    print("No voice selected or does not exist. Using default voice 'en_1'.")
                    selected_voice = self.get_voice_by_name("en_1")
                if selected_voice is not None:
                    ref_audio = selected_voice.get("audio_filename")
                else:
                    # No voice file available; rely on builtin conds if present
                    ref_audio = None

            exaggeration = self.special_settings["exaggeration"]
            temperature = self.special_settings["temperature"]
            if self._loaded_precision_dtype == torch.float16 and temperature < 0.55:
                temperature = 0.55  # avoid cuda error in fp16 at low temps

            # Backward-compatible cfg weight lookup
            cfg_weight = self.special_settings.get("cfg_weight", self.special_settings.get("cfg", 0.5))
            max_new_tokens = int(self.special_settings.get(
                "max_new_tokens", 256,
            ))
            repetition_penalty = float(self.special_settings.get(
                "repetition_penalty", 1.2,
            ))

            with torch.inference_mode():
                # Prepare/reuse voice conditionals once instead of per segment
                if ref_audio is not None:
                    try:
                        self._ensure_voice_conditionals(ref_audio, exaggeration=exaggeration)
                    except Exception:
                        # Fallback: let generate handle assertion if no conds
                        pass

                self.set_seed()

                generate_kwargs = {
                    "exaggeration": exaggeration,
                    "temperature": temperature,
                    "cfg_weight": cfg_weight,
                    "max_new_tokens": max_new_tokens,
                    # Align repetition handling with ONNX backend to curb trailing speech
                    "repetition_penalty": repetition_penalty,
                }

                # Only pass language; avoid audio_prompt_path to prevent re-encoding
                wav = self.model.generate(text, language_id=language, **generate_kwargs)

                if tts_normalize:
                    wav, _ = audio_tools.normalize_audio_lufs(
                        wav, self.sample_rate, -24.0, -16.0,
                        1.3, verbose=True
                    )

                if tts_volume != 1.0:
                    wav = audio_tools.change_volume(wav, tts_volume)

                # call custom plugin event method
                plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': wav, 'sample_rate': self.sample_rate})
                if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                    wav = plugin_audio['audio']

                self.garbage_collect()

                return wav, self.sample_rate

            # return empty audio on stop or error
        except Exception as e:
            print(f"TTS generation error: {e}")
            traceback.print_exc()
            return np.array([], dtype=np.float32), self.sample_rate

    def _tts_generator_onnx(self, text, ref_audio=None, language="en"):
        """Generate TTS audio using ONNX backend and return (torch.Tensor [1,N], sample_rate)."""
        try:
            # Resolve voice audio path
            if ref_audio is None:
                ref_audio = self._resolve_main_voice_audio(None)
            if ref_audio is None or not os.path.isfile(ref_audio):
                raise FileNotFoundError("No valid reference voice audio found for ONNX TTS.")

            # Runtime settings
            tts_volume = settings.GetOption("tts_volume")
            tts_normalize = settings.GetOption("tts_normalize")
            exaggeration = float(self.special_settings.get("exaggeration", 0.5))
            max_new_tokens = int(self.special_settings.get("max_new_tokens", 256))
            repetition_penalty = float(self.special_settings.get("repetition_penalty", 1.2))

            # Generate waveform via engine
            wav_np = self._onnx_engine.generate(
                text=text,
                language_id=str(language).lower() if isinstance(language, str) else "en",
                target_voice_path=ref_audio,
                max_new_tokens=max_new_tokens,
                exaggeration=exaggeration,
                repetition_penalty=repetition_penalty,
                progress=False,
            )
            # Convert to torch tensor [1, N]
            wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)

            # Post-processing
            if tts_normalize:
                wav_t, _ = audio_tools.normalize_audio_lufs(
                    wav_t, self.sample_rate, -24.0, -16.0,
                    1.3, verbose=True
                )
            if tts_volume != 1.0:
                wav_t = audio_tools.change_volume(wav_t, tts_volume)

            plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': wav_t, 'sample_rate': self.sample_rate})
            if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                wav_t = plugin_audio['audio']

            self.garbage_collect()
            return wav_t, self.sample_rate
        except Exception as e:
            print(f"ONNX TTS generation error: {e}")
            traceback.print_exc()
            return torch.zeros(1, 0), self.sample_rate

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        print("TTS requested Chatterbox TTS")
        language = self._use_language(text)
        # Voice-tag parsing: split by [voice_name] and then chunk each section
        voices_map = self._build_voice_map(ref_audio)
        voice_sections = self._parse_voice_tagged_text(text)
        # Determine if there are any line-start tags present at all
        has_any_tags = re.search(r'(?m)^[\ufeff\u200b\s]*\[[^]]+\]', text) is not None
        # If no tags found, fall back to normal chunking with 'main'
        if not voice_sections:
            if has_any_tags:
                print("Found voice tags but no content associated; skipping tags and returning silence.")
                return torch.zeros(1, 0), self.sample_rate
            voice_sections = [("main", text)]

        audio_chunks = []
        for voice_name, voice_text in voice_sections:
            # Resolve voice audio, fallback to main if missing
            voice_audio = voices_map.get(voice_name)
            if voice_audio is None:
                print(f"Voice '{voice_name}' not found. Using 'main' voice.")
                voice_audio = voices_map.get("main")
            # Chunk the text for this voice
            segments = self.chunk_up_text(
                voice_text,
                goal_length=self.chunk_goal_length,
                max_length=self.chunk_max_length,
                jitter=self.chunk_jitter,
                custom_split_chars=self.chunk_custom_split_chars
            )
            for segment in segments:
                if not segment.strip():
                    continue
                wav, sample_rate = self.tts_generator(segment, voice_audio, language=language)
                audio_chunks.append(wav)

        if not audio_chunks:
            print("No audio generated (no valid segments). Returning silence.")
            final_wave = torch.zeros(1, self.sample_rate // 10)  # 0.1s silence fallback
            self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}
            return final_wave, self.sample_rate

        # Concatenate all segment waves
        final_wave = torch.cat(audio_chunks, dim=-1)
        # proactively free segment_waves list after concat
        audio_chunks.clear()

        self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}

        return final_wave, self.sample_rate

    def tts_streaming(self, text, ref_audio=None):
        self._ensure_special_settings()
        backend = self._get_model_type().lower()
        streaming_mode = self.special_settings.get("streaming_mode", "segment")
        if streaming_mode == "token":
            if backend == "onnx":
                return self.tts_streaming_tokens_onnx(text, ref_audio)
            return self.tts_streaming_tokens(text, ref_audio)
        else:
            return self.tts_streaming_segments(text, ref_audio)

    def tts_streaming_segments(self, text, ref_audio=None):
        print("TTS requested Chatterbox TTS (Streaming)")
        self._ensure_special_settings()

        language = self._use_language(text)

        chunk_size = settings.GetOption("tts_streamed_chunk_size")
        self.init_audio_stream_playback()

        # Voice-tag parsing
        voices_map = self._build_voice_map(ref_audio)
        voice_sections = self._parse_voice_tagged_text(text)
        has_any_tags = re.search(r'(?m)^[\ufeff\u200b\s]*\[[^]]+\]', text) is not None
        if not voice_sections:
            if has_any_tags:
                print("Found voice tags but no content associated; skipping streaming output.")
                self.last_generation = {"audio": torch.zeros(1, 0), "sample_rate": self.sample_rate}
                print("TTS generation finished")
                return torch.zeros(1, 0), self.sample_rate
            voice_sections = [("main", text)]

        audio_chunks = []
        for voice_name, voice_text in voice_sections:
            voice_audio = voices_map.get(voice_name)
            if voice_audio is None:
                print(f"Voice '{voice_name}' not found. Using 'main' voice.")
                voice_audio = voices_map.get("main")

            segments = self.chunk_up_text(
                voice_text,
                goal_length=self.chunk_goal_length,
                max_length=self.chunk_max_length,
                jitter=self.chunk_jitter,
                custom_split_chars=self.chunk_custom_split_chars
            )
            for segment in segments:
                if not segment.strip():
                    continue
                wav, sample_rate = self.tts_generator(segment, voice_audio, language=language)
                audio_chunks.append(wav)
                if self.audio_streamer is not None:
                    wav_bytes = self.return_pcm_audio(wav)  # convert to PCM bytes
                    self.audio_streamer.add_audio_chunk(wav_bytes)

        final_wave = torch.cat(audio_chunks, dim=-1) if audio_chunks else torch.zeros(1, 0)
        self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}
        print("TTS generation finished")
        return final_wave, self.sample_rate


    def tts_streaming_tokens(self, text, ref_audio=None):
        print("TTS requested Chatterbox TTS (Streaming)")
        self._ensure_special_settings()
        # Ensure model precision matches current setting
        self._ensure_model_for_precision()

        # Initialize audio streamer playback
        chunk_size = settings.GetOption("tts_streamed_chunk_size")
        self.init_audio_stream_playback()

        # Pull runtime audio settings
        tts_volume = settings.GetOption("tts_volume")
        # Streaming batching controls (to speed up synthesis)
        token_batch_size = settings.GetOption("tts_streamed_token_batch_size")
        if token_batch_size is None or token_batch_size <= 0:
            token_batch_size = 12  # ~480ms at 25 tok/sec
        min_start_tokens = settings.GetOption("tts_streamed_min_start_tokens")
        if min_start_tokens is None or min_start_tokens <= 0:
            min_start_tokens = 6  # ~240ms warm-up to avoid empty frames

        exaggeration = self.special_settings["exaggeration"]
        temperature = self.special_settings["temperature"]
        if self._loaded_precision_dtype == torch.float16 and temperature < 0.55:
            temperature = 0.55  # avoid cuda error in fp16 at low temps
        # Backward-compatible cfg weight lookup
        cfg_weight = self.special_settings.get("cfg_weight", self.special_settings.get("cfg", 0.5))
        self.set_seed()

        # Prepare voice map and parse voice-tagged sections
        language = self._use_language(text)
        voices_map = self._build_voice_map(ref_audio)
        voice_sections = self._parse_voice_tagged_text(text)
        has_any_tags = re.search(r'(?m)^[\ufeff\u200b\s]*\[[^]]+\]', text) is not None
        if not voice_sections:
            if has_any_tags:
                print("Found voice tags but no content associated; skipping streaming output.")
                self.last_generation = {"audio": torch.zeros(1, 0), "sample_rate": self.sample_rate}
                print("TTS generation finished (streaming)")
                return torch.zeros(1, 0), self.sample_rate
            voice_sections = [("main", text)]
        max_new_tokens = int(self.special_settings.get(
            "max_new_tokens", 256,
        ))
        repetition_penalty = float(self.special_settings.get(
            "repetition_penalty", 1.2,
        ))

        # For final return, also build the full waveform as we stream
        segment_wavs = []
        # Streaming hyperparameters aligned with generate()
        min_p = 0.05
        top_p = 1.0

        last_voice_audio = None
        for voice_name, voice_text in voice_sections:
            voice_audio = voices_map.get(voice_name)
            if voice_audio is None:
                print(f"Voice '{voice_name}' not found. Using 'main' voice.")
                voice_audio = voices_map.get("main")

            # Ensure conditionals for this voice (once per voice switch)
            if voice_audio != last_voice_audio and voice_audio is not None:
                try:
                    self._ensure_voice_conditionals(voice_audio, exaggeration=exaggeration)
                except Exception:
                    pass
                last_voice_audio = voice_audio

            # Split text into sentence-like chunks for this voice
            segments = self.chunk_up_text(
                voice_text,
                goal_length=self.chunk_goal_length,
                max_length=self.chunk_max_length,
                jitter=self.chunk_jitter,
                custom_split_chars=self.chunk_custom_split_chars
            )

            for idx, segment in enumerate(segments):
                if not segment.strip():
                    continue

                # Text preprocessing and tokenization (match mtl_tts.generate logic)
                seg_text = punc_norm(segment)
                text_tokens = self.model.tokenizer.text_to_tokens(seg_text, language_id=language.lower()).to(self.model.device)
                # CFG duplication
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
                # Add SOT/EOT
                sot = self.model.t3.hp.start_text_token
                eot = self.model.t3.hp.stop_text_token
                text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
                text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)

                # Streaming decode for this text chunk
                emitted_samples = 0
                cache_source = torch.zeros(1, 1, 0, device=self.model.device)
                token_stream = self.model.t3.inference_stream(
                    t3_cond=self.model.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=max_new_tokens,
                    stop_on_eos=True,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    cfg_weight=cfg_weight,
                    progress=False,
                )

                # Accumulate the synthesized audio for this segment
                seg_audio = []
                collected_tokens: list[int] = []
                last_synth_tok_count = 0

                for tid in token_stream:
                    # Skip SoS/EoS and out-of-range
                    if tid >= SPEECH_VOCAB_SIZE:
                        continue
                    collected_tokens.append(int(tid))

                    # Warm-up: wait until we have a few tokens to avoid empty frames
                    if len(collected_tokens) < min_start_tokens:
                        continue

                    # Synthesize only when we have a full batch of new tokens
                    if (len(collected_tokens) - last_synth_tok_count) < token_batch_size:
                        continue

                    # Run incremental synthesis for current tokens
                    stoks = torch.tensor(collected_tokens, dtype=torch.long, device=self.model.device).unsqueeze(0)

                    wav, cache_source = self.model.s3gen.inference(
                        speech_tokens=stoks,
                        ref_dict=self.model.conds.gen,
                        cache_source=cache_source,
                        finalize=False,
                    )
                    wav = wav.squeeze(0).detach()

                    # Emit only the newly synthesized portion since last push
                    if wav.size(-1) > emitted_samples:
                        new_chunk = wav[..., emitted_samples:]
                        emitted_samples = wav.size(-1)
                        if new_chunk.numel() > 0:
                            if tts_volume != 1.0:
                                new_chunk = new_chunk * float(tts_volume)
                            seg_audio.append(new_chunk.cpu())
                            if self.audio_streamer is not None:
                                pcm_bytes = self.return_pcm_audio(new_chunk.unsqueeze(0))
                                self.audio_streamer.add_audio_chunk(pcm_bytes)

                    last_synth_tok_count = len(collected_tokens)

                # Finalize to flush the model tail for this segment
                if len(collected_tokens) > 0:
                    stoks = torch.tensor(collected_tokens, dtype=torch.long, device=self.model.device).unsqueeze(0)
                    wav, _ = self.model.s3gen.inference(
                        speech_tokens=stoks,
                        ref_dict=self.model.conds.gen,
                        cache_source=cache_source,
                        finalize=True,
                    )
                    wav = wav.squeeze(0).detach()
                    if wav.size(-1) > emitted_samples:
                        new_chunk = wav[..., emitted_samples:]
                        if new_chunk.numel() > 0:
                            if tts_volume != 1.0:
                                new_chunk = new_chunk * float(tts_volume)
                            seg_audio.append(new_chunk.cpu())
                            if self.audio_streamer is not None:
                                pcm_bytes = self.return_pcm_audio(new_chunk.unsqueeze(0))
                                self.audio_streamer.add_audio_chunk(pcm_bytes)

                # Concatenate the audio for this segment and collect for final output
                if len(seg_audio) > 0:
                    seg_wave = torch.cat(seg_audio, dim=-1).unsqueeze(0)  # [1, N]
                else:
                    seg_wave = torch.zeros(1, 0)
                segment_wavs.append(seg_wave)

                self.garbage_collect()

        # Build final waveform for return
        if len(segment_wavs) == 0:
            final_wave = torch.zeros(1, self.sample_rate // 10)  # 0.1s silence fallback
        else:
            final_wave = torch.cat(segment_wavs, dim=-1)

        self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}
        print("TTS generation finished (streaming)")
        return final_wave, self.sample_rate

    def tts_streaming_tokens_onnx(self, text, ref_audio=None):
        """Token-level streaming for ONNX backend using engine.stream_audio."""
        print("TTS requested Chatterbox TTS (Streaming ONNX Tokens)")
        self._ensure_special_settings()

        # Initialize audio streamer playback
        self.init_audio_stream_playback()

        # Runtime settings
        tts_volume = settings.GetOption("tts_volume")
        emit_every = settings.GetOption("tts_streamed_token_batch_size")
        if emit_every is None or emit_every <= 0:
            emit_every = 12
        exaggeration = float(self.special_settings.get("exaggeration", 0.5))
        repetition_penalty = float(self.special_settings.get("repetition_penalty", 1.2))

        # Prepare voice map and parse voice-tagged sections
        language = self._use_language(text)
        voices_map = self._build_voice_map(ref_audio)
        voice_sections = self._parse_voice_tagged_text(text)
        has_any_tags = re.search(r'(?m)^[\ufeff\u200b\s]*\[[^]]+\]', text) is not None
        if not voice_sections:
            if has_any_tags:
                print("Found voice tags but no content associated; skipping streaming output.")
                self.last_generation = {"audio": torch.zeros(1, 0), "sample_rate": self.sample_rate}
                print("TTS generation finished (streaming ONNX)")
                return torch.zeros(1, 0), self.sample_rate
            voice_sections = [("main", text)]

        segment_wavs = []
        for voice_name, voice_text in voice_sections:
            voice_audio = voices_map.get(voice_name)
            if voice_audio is None:
                print(f"Voice '{voice_name}' not found. Using 'main' voice.")
                voice_audio = voices_map.get("main")

            segments = self.chunk_up_text(
                voice_text,
                goal_length=self.chunk_goal_length,
                max_length=self.chunk_max_length,
                jitter=self.chunk_jitter,
                custom_split_chars=self.chunk_custom_split_chars
            )
            for segment in segments:
                if not segment.strip():
                    continue

                emitted = 0
                seg_chunks = []
                # Stream chunks from engine
                for new_chunk in self._onnx_engine.stream_audio(
                    text=segment,
                    language_id=str(language).lower(),
                    target_voice_path=voice_audio,
                    max_new_tokens=int(self.special_settings.get("max_new_tokens", 256)),
                    exaggeration=exaggeration,
                    repetition_penalty=repetition_penalty,
                    emit_every_tokens=int(emit_every),
                    progress=False,
                ):
                    # Volume adjust per-chunk
                    if tts_volume != 1.0:
                        new_chunk = new_chunk * float(tts_volume)
                    seg_chunks.append(torch.from_numpy(new_chunk).float())
                    if self.audio_streamer is not None:
                        self.audio_streamer.add_audio_chunk(self.return_pcm_audio(new_chunk))
                    emitted += new_chunk.shape[0]

                # Build segment waveform
                if len(seg_chunks) > 0:
                    seg_wave = torch.cat(seg_chunks, dim=-1).unsqueeze(0)
                else:
                    seg_wave = torch.zeros(1, 0)
                segment_wavs.append(seg_wave)
                self.garbage_collect()

        # Final waveform
        if len(segment_wavs) == 0:
            final_wave = torch.zeros(1, self.sample_rate // 10)
        else:
            final_wave = torch.cat(segment_wavs, dim=-1)
        self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}
        print("TTS generation finished (streaming ONNX)")
        return final_wave, self.sample_rate

    def voice_conversion(self, audio, target_voice_path):
        # Ensure model precision matches current setting before VC
        self._ensure_special_settings()
        self._ensure_model_for_precision()
        wav = self.vc_model.generate(audio, target_voice_path=target_voice_path)
        return wav, self.sample_rate

    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")

        chunk_size = settings.GetOption("tts_streamed_chunk_size")
        #if self.audio_streamer is not None:
        #    self.audio_streamer.stop()
        #    self.audio_streamer = None
        #else:
        if self.audio_streamer is None:
            min_play_time = float(settings.GetOption("tts_streamed_min_play_time"))
            self.audio_streamer = audio_tools.AudioStreamer(audio_device,
                                                            source_sample_rate=self.sample_rate,
                                                            start_playback_timeout=1.0,
                                                            min_buffer_play_time=min_play_time,
                                                            playback_channels=2,
                                                            buffer_size=chunk_size,
                                                            input_channels=1,
                                                            dtype="float32",
                                                            tag="tts",
                                                            )

    def play_audio(self, audio, device=None):
        source_channels = 1

        if device is None:
            device = settings.GetOption("device_default_out_index")

        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and device != settings.GetOption("tts_secondary_playback_device"))):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        allow_overlapping_audio = settings.GetOption("tts_allow_overlapping_audio")
        #audio = np.int16(audio * 32767)  # Convert to 16-bit PCM
        #audio = audio_tools.convert_audio_datatype_to_integer(audio)

        # play audio tensor
        audio_tools.play_audio(audio, device,
                               source_sample_rate=int(self.sample_rate),
                               audio_device_channel_num=1,
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=1,
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio, sample_rate=sample_rate):
        """
        audio: PyTorch tensor shaped (T,) for mono or (C, T)/(T, C) for multi-channel.
        sample_rate: int
        Returns: bytes of a WAV file (float32, IEEE float)
        """
        # Convert input to NumPy float32
        if isinstance(audio, torch.Tensor):
            np_arr = audio.detach().cpu().float().numpy()
        elif isinstance(audio, np.ndarray):
            np_arr = audio.astype(np.float32, copy=False)
        else:
            # Accept lists/tuples; fall back to NumPy conversion
            np_arr = np.asarray(audio, dtype=np.float32)

        # Ensure shape is (samples,) mono or (samples, channels)
        if np_arr.ndim == 1:
            pass  # (T,)
        elif np_arr.ndim == 2:
            # If (C, T) with small C, transpose to (T, C)
            if np_arr.shape[0] <= 32 and np_arr.shape[1] > np_arr.shape[0]:
                np_arr = np_arr.T
        else:
            raise ValueError(
                f"Expected audio with 1D or 2D shape (T,) or (C,T)/(T,C), got {np_arr.shape}"
            )

        # Clip to valid float range for audio
        np_arr = np.clip(np_arr, -1.0, 1.0)

        # Write WAV into memory
        buff = io.BytesIO()
        write_wav(buff, int(sample_rate), np_arr)
        return buff.getvalue()

    def return_pcm_audio(self, audio):
        """Return raw PCM bytes (float32) suitable for AudioStreamer(dtype="float32").
        Accepts torch.Tensor, numpy.ndarray, list. Produces mono 1-D float32 stream.
        """
        # Convert input to NumPy float32
        if isinstance(audio, torch.Tensor):
            np_arr = audio.detach().cpu().float().numpy()
        elif isinstance(audio, np.ndarray):
            np_arr = np.float32(audio)
        else:
            np_arr = np.asarray(audio, dtype=np.float32)

        # Normalize/reshape to mono 1-D (samples,)
        if np_arr.ndim == 1:
            pass
        elif np_arr.ndim == 2:
            # Determine orientation; treat as (C, T) if channels dimension likely first
            if np_arr.shape[0] <= 32 and np_arr.shape[1] > np_arr.shape[0]:
                # (C, T) -> average channels to mono
                if np_arr.shape[0] > 1:
                    np_arr = np_arr.mean(axis=0)
                else:
                    np_arr = np_arr[0]
            else:
                # (T, C) -> average channels to mono
                if np_arr.shape[1] > 1:
                    np_arr = np_arr.mean(axis=1)
                else:
                    np_arr = np_arr[:, 0]
        else:
            # Unexpected shapes: flatten after averaging over non-time dims
            np_arr = np_arr.reshape(-1).astype(np.float32, copy=False)

        # Clip to float range and ensure contiguous
        np_arr = np.clip(np_arr, -1.0, 1.0).astype(np.float32, copy=False)
        if not np_arr.flags['C_CONTIGUOUS']:
            np_arr = np.ascontiguousarray(np_arr)

        # Convert numpy array to raw PCM bytes (float32 little-endian)
        pcm_bytes = np_arr.tobytes()
        return pcm_bytes

    @staticmethod
    def generate_random_seed():
        return torch.randint(0, 2 ** 32 - 1, (1,)).item()

    def _use_language(self, text):
        language = self.special_settings.get("language", "en")
        if language.lower() == "auto":
            try:
                lang_code, _ = languageClassification.classify(text)
                language = self.language_code_converter.convert(lang_code, "iso1")
                print(f"Auto-detected language: {language} (ISO-1 code from '{lang_code}')")
                if language not in SUPPORTED_LANGUAGES:
                    print(f"Detected language '{language}' not supported by Chatterbox TTS. Falling back to English.")
                    language = "en"
            except Exception as e:
                print(f"Language auto-detection error: {e}. Falling back to English.")
                language = "en"
        return language

    def set_seed(self):
        seed = self.special_settings["seed"]
        try:
            seed = int(seed)
        except ValueError:
            seed = -1
        if seed <= -1:
            seed = self.generate_random_seed()

        torch.manual_seed(seed)
        if self.compute_device_str == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
