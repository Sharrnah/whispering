import gc
import hashlib
import importlib.util
import io
import re
import sys
import threading
import types
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from transformers import StoppingCriteria, StoppingCriteriaList

import audio_tools
import settings
from Models.Singleton import SingletonMeta
from Models.TTS.text_segmentation import chunk_text
from Models.TTS.tts_config import get_tts_device, get_tts_precision


SAMPLE_RATE = 44100
DEFAULT_MODEL = "Audio8-TTS-Preview-0.6b"
CODEC_MODEL = "Audio8-TTS-Codec-44.1kHz"
MODEL_CACHE_PATH = Path.cwd() / ".cache" / "audio8-tts"
VOICES_PATH = MODEL_CACHE_PATH / "voices"

MODEL_LANGUAGES = {
    DEFAULT_MODEL: {
        "yue": "Cantonese",
        "zh": "Chinese",
        "nl": "Dutch",
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pl": "Polish",
        "es": "Spanish",
    },
}

MODEL_LIST = {
    "0.6B": [DEFAULT_MODEL],
}


def _hosted_urls(archive_name):
    return [
        f"https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/audio8-tts/{archive_name}",
        f"https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/audio8-tts/{archive_name}",
        f"https://s3.libs.space:9000/ai-models/audio8-tts/{archive_name}",
    ]


# The model archives contain the immutable custom Transformers sources as well
# as their weights and tokenizers. They deliberately omit the byte-identical
# 1.35 GB codec, which is installed once through CODEC_MODEL.
TTS_MODEL_LINKS = {
    # Audio8/Audio8-TTS-Preview-0.6b @ f07040f3d151f1ba0253bfb92cb2f5dd38b44594
    DEFAULT_MODEL: {
        "urls": _hosted_urls("audio8-tts-preview-0.6b.zip"),
        "checksum": "d9171cb5d8f7f61695c86e824fdf73404f64ef264a3043c1afdd1888d9b5c9f2",
        "file_checksums": {
            "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            "NOTICE": "819129644b28fb066221d782dd27a6fe1d35f33f46718d4e61570614fb38854e",
            "README.md": "c3252410cdce14fc206da10524cfaf7cc9112a1fe2d7e81549b2f059391281f3",
            "config.json": "92e19f10edb6c79ef2ccf4aebe218baa2c12743d648d75238041eca39439f8d3",
            "configuration_arktts.py": "27fa4fd4073ef1983a48919a4562300cace84687335035a9b9a077966e1ac04e",
            "generation_config.json": "bbb0b0059d787345929a56f60d284c1a03ea2f4bca7c405d713eed20eb74658a",
            "model.safetensors": "62dcff0adf6c2535b3260467a7c1d482b556da57266c96a444518b76e140d2c3",
            "modeling_arktts.py": "b320b6c875e6a57afb9745cc98fc72c116d9df6b3bf20d092a5ad3095026911f",
            "modeling_arktts_codec.py": "e1d6b0e1ef1e9ad303dc01d45d86229cf7f2e98d06a7b426234a7e4d7e3ede68",
            "preprocessor_config.json": "e626df9aa1cbe25c4e0999a470c6598d58f172169a5488357b594795d85b5208",
            "processing_arktts.py": "eba6ecb6b7f0e221bcd19c5e3fe5978e677faf6b362912ff4febc08c3dc7f9df",
            "processor_config.json": "4df1f6d2a8ca9d1976b75378847a9de6c3d4db6e34240788b7b9ffa053c00aa2",
            "special_tokens_map.json": "c2ff18fde6e43b7408435bc8ed079af74531befba549358a97cbc59ce606bc6b",
            "tokenizer.json": "f24e08099d45a8adf3f52f5f0b03276e433bb9d689bb15fcbcc48ce58744588b",
            "tokenizer_config.json": "b8d149343ae425b0da67e6708686aceb51be7815d9792f265fc12ff04d5e9856",
        },
        "path": DEFAULT_MODEL,
        "base_model": CODEC_MODEL,
        "source_revision": "f07040f3d151f1ba0253bfb92cb2f5dd38b44594",
        "license": "Apache-2.0",
    },
    # codec.pth is byte-identical in both official repositories. Its Apache
    # provenance here is the 0.6B release and Audio8_TTS source revision below.
    CODEC_MODEL: {
        "urls": _hosted_urls("audio8-tts-codec-44khz.zip"),
        "checksum": "93691f7352ba692ad33e090b0154848536ae631e5c873c9aa716d5f269341eae",
        "file_checksums": {
            "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            "NOTICE": "819129644b28fb066221d782dd27a6fe1d35f33f46718d4e61570614fb38854e",
            "codec.pth": "c310505aa11fe2f6cc63b8d3130dc7e77e73227774f5c62575769b1f47a8d048",
        },
        "path": CODEC_MODEL,
        "source_revision": "421f71559848572431bd6229af3e1a73f25986a7",
        "license": "Apache-2.0",
    },
}

VOICE_MODEL_LINKS = {
    "voices": {
        "urls": _hosted_urls("voices.zip"),
        "checksum": "8377c12ae94c6f644e9e79cc9f36e6cb4595665f74453775bf61ad4810ff1e5d",
        "file_checksums": {
            "Announcer_Ahri.txt": "0b072cd9fcf16a0d836c66490f13f0c54af7ede0dca41d547e9a4bb78deb9943",
            "Announcer_Ahri.wav": "2a3fd17d45b3c5633dd64e2d80a6e3fc924fa829da405e3b591a5bacdc88f9fc",
            "Attenborough.txt": "23d83f548c9370fe58097e38cdebb8c398d1e544e24de89eb462a5fe4a5037a3",
            "Attenborough.wav": "358540c89932baf1103960d99949b78ea7466f95b2225fdcd8f8bb8b976f09ee",
            "Jane.txt": "5ca8096d0be793096512f61f32a15bee6ea53f7221f70b1df520ca1c22cd734f",
            "Jane.wav": "d1d2235af1a4408c641a765427978486f5cca9b369fc6456d8086449f1f92fe3",
            "Justin.txt": "97a13d31554fdee767050c2d65bfc8fdbd6f02a2322a404a071334173d2ecfa7",
            "Justin.wav": "a83c37f408b53efaeb9189f166c6669d1a0dc6cf779e85913fa9cbbbbe0d5aaf",
            "Xiaochen.txt": "a127b00fefed1100c25214d37251f998898e270971e9fbe5e5e6de0108f1fe15",
            "Xiaochen.wav": "7f0b735e188a06dc9f104eeb3fd71a3ef580d1f2133c95630c92a244dd253732",
            "default_voice.txt": "5032a80688c36608720745d7977ce6a90829c8027540d73ef31945d3dd0649b9",
            "default_voice.wav": "3ebc531cdaba358a327099c1c4f0448026719957bcf4d8e9868767f227e02f4e",
            "en_0.txt": "784c015b12f794c14fdc08a7f828fd4cbf84e65add36489abb2fce9efd7e6d0d",
            "en_0.wav": "f006e2e9c76523bde4f5bbe67a7be9a600786d7432cbcc9486bc9501053298b7",
            "en_1.txt": "0c8420c072540fc4b803369e87e79d03e8c4963434464006b76f6d7e24406f2f",
            "en_1.wav": "b0e22048e72414fcc1e6b6342e47a774d748a195ed34e4a5b3fcf416707f2b71",
            "fallback_audio.txt": "3d95afacabcfc8d4c18f7aafa532647d1eb705df06f6d01e5c7ab5f6a27c2f80",
            "fallback_audio.wav": "eaa7796d2c44424c645a0b384d82f09aac48fab2c9977de6f53b6a4f9d0e0da1",
            "female_shadowheart.txt": "5ec343e79fd7ab97e03a1467da28aaf964bea233604f9ea23200c3bfcf8e037e",
            "female_shadowheart.wav": "8abb726ad6aaa5203e62de4c92ac2aab3d3fa1fdb509c9b76d254722178ab70a",
            "test_zh_1_ref_short.txt": "96a12abab72997cf37c37546ec3b649b89393e2f138b86829814b482ea40dc96",
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede",
            "tiktok_adam.txt": "adcc76f9a70210ee0bc15a5e3d6a2372103533ff4d5864d35c439dc9cd17fdb6",
            "tiktok_adam.wav": "2ed130b6dd069ee4c306f6cb8fedb94db75567aefa084085c6a069bd2c34662d",
            "tiktok_jessie.txt": "de18a3119cf66ca9b3f3f29bd3e66fcf310557d0093a5151b5cb4e8519243251",
            "tiktok_jessie.wav": "5a26de921ea3e7c1ce1bfd2344fb107781def9366b56e2f583c7500a1052dbbd"
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


def _load_verified_runtime(model_directory):
    """Import Audio8's manifest-verified local runtime without a Hub code cache."""
    model_directory = Path(model_directory).resolve()
    required = (
        "configuration_arktts.py",
        "modeling_arktts.py",
        "modeling_arktts_codec.py",
        "processing_arktts.py",
    )
    missing = [name for name in required if not (model_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Audio8 runtime is incomplete in {model_directory}: {', '.join(missing)}"
        )

    digest = hashlib.sha256(str(model_directory).encode("utf-8")).hexdigest()[:12]
    package_name = f"_whispering_audio8_{digest}"
    configuration_name = package_name + ".configuration_arktts"
    modeling_name = package_name + ".modeling_arktts"
    processing_name = package_name + ".processing_arktts"
    if all(name in sys.modules for name in (configuration_name, modeling_name, processing_name)):
        return (
            sys.modules[configuration_name],
            sys.modules[modeling_name],
            sys.modules[processing_name],
        )

    package_spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    package = importlib.util.module_from_spec(package_spec)
    package.__path__ = [str(model_directory)]
    sys.modules[package_name] = package

    def load_module(short_name):
        full_name = package_name + "." + short_name
        spec = importlib.util.spec_from_file_location(
            full_name,
            model_directory / f"{short_name}.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Audio8 runtime module {short_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    try:
        configuration = load_module("configuration_arktts")
        modeling = load_module("modeling_arktts")
        processing = load_module("processing_arktts")
    except Exception:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        raise
    return configuration, modeling, processing


def _restore_rotary_buffers(model, modeling):
    """Restore Audio8 buffers left uninitialized by Transformers 5 meta loading."""
    specs = (
        ("freqs_cis", model.config.max_seq_len, model.config.head_dim),
        ("fast_freqs_cis", model.config.num_codebooks, model.config.fast_head_dim),
    )
    for name, length, head_dim in specs:
        current = getattr(model, name, None)
        if current is None:
            continue
        restored = modeling._precompute_rope(
            length,
            head_dim,
            model.config.rope_base,
        ).to(device=current.device, dtype=current.dtype)
        setattr(model, name, restored)


def _get_flash_attention_function():
    """Load the already-packaged FlashAttention kernel only when it is usable."""
    from flash_attn import flash_attn_func

    return flash_attn_func


def _resolve_attention_backend(requested, device, dtype):
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "flash_attention_2", "sdpa"}:
        requested = "auto"
    if requested == "sdpa":
        return "sdpa", None

    reason = None
    flash_function = None
    if device.type != "cuda":
        reason = "it requires CUDA"
    elif dtype not in {torch.float16, torch.bfloat16}:
        reason = "it requires float16 or bfloat16"
    else:
        try:
            flash_function = _get_flash_attention_function()
        except Exception as exc:
            reason = f"the installed kernel could not be loaded ({exc})"

    if flash_function is not None:
        return "flash_attention_2", flash_function
    if requested == "flash_attention_2":
        print(f"Audio8 FlashAttention 2 is unavailable because {reason}; using PyTorch attention.")
    return "sdpa", None


def _audio8_flash_attention_forward(
    self,
    x,
    rope,
    attention_mask,
    cache_position=None,
):
    """Use FA2 for the single-token slow-AR decode and preserve all other paths."""
    state = self._audio8_flash_state
    fallback = self._audio8_original_forward
    fallback_function = getattr(fallback, "__func__", fallback)
    if getattr(fallback_function, "__name__", "") == _audio8_flash_attention_forward.__name__:
        # Recover an instance that was patched twice by an older adapter. The
        # verified runtime class itself is never modified, so its method is the
        # authoritative compatible implementation.
        fallback = type(self).forward.__get__(self, type(self))
        self._audio8_original_forward = fallback
    batch, length, _ = x.shape
    valid_length = int(self._audio8_valid_cache_length)
    can_use_flash = (
        state["enabled"]
        and not self.training
        and x.device.type == "cuda"
        and x.dtype in {torch.float16, torch.bfloat16}
        and batch == 1
        and length == 1
        and self.kv_cache is not None
        and cache_position is not None
        and valid_length > 0
    )
    if not can_use_flash:
        return fallback(x, rope, attention_mask, cache_position)

    try:
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split(
            (query_size, kv_size, kv_size),
            dim=-1,
        )
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = self._audio8_apply_rope(query, rope)
        key = self._audio8_apply_rope(key, rope).transpose(1, 2)
        value = value.transpose(1, 2)
        key, value = self.kv_cache.update(cache_position, key, value)
        valid_length = min(valid_length, key.shape[2])
        key = key[:, :, :valid_length].transpose(1, 2).contiguous()
        value = value[:, :, :valid_length].transpose(1, 2).contiguous()
        output = self._audio8_flash_function(
            query.contiguous(),
            key,
            value,
            dropout_p=0.0,
            causal=False,
        )
        output = output.contiguous().view(batch, length, query_size)
        return self.wo(output)
    except Exception as exc:
        state["enabled"] = False
        if not state["warned"]:
            state["warned"] = True
            print(
                "Audio8 FlashAttention 2 failed during generation; "
                f"using PyTorch attention for this model instance. ({exc})"
            )
        # Repeating an update at the same cache position is safe and lets the
        # verified upstream implementation finish the current token.
        return fallback(x, rope, attention_mask, cache_position)


def _enable_audio8_flash_attention(model, modeling, flash_function):
    """Patch only Audio8's slow incremental AR path without editing cached code."""
    if getattr(model, "_audio8_flash_patch_installed", False):
        state = model._audio8_flash_state
        state.update(enabled=True, warned=False)
        model._audio8_valid_cache_length = 0
        for layer in model.layers:
            attention = layer.attention
            attention._audio8_flash_function = flash_function
            attention._audio8_flash_state = state
            attention._audio8_valid_cache_length = 0
        return

    model._audio8_flash_patch_installed = True
    state = {"enabled": True, "warned": False}
    model._audio8_flash_state = state
    model._audio8_valid_cache_length = 0

    original_setup = model._setup_generation_caches
    original_slow_step = model._slow_step

    def setup_generation_caches(instance, *args, **kwargs):
        instance._audio8_valid_cache_length = 0
        return original_setup(*args, **kwargs)

    def slow_step(instance, input_ids, *args, **kwargs):
        instance._audio8_valid_cache_length = min(
            instance.config.max_seq_len,
            instance._audio8_valid_cache_length + int(input_ids.shape[-1]),
        )
        for layer in instance.layers:
            layer.attention._audio8_valid_cache_length = (
                instance._audio8_valid_cache_length
            )
        return original_slow_step(input_ids, *args, **kwargs)

    model._setup_generation_caches = types.MethodType(setup_generation_caches, model)
    model._slow_step = types.MethodType(slow_step, model)
    for layer in model.layers:
        attention = layer.attention
        original_forward = attention.forward
        current_function = getattr(original_forward, "__func__", original_forward)
        if getattr(current_function, "__name__", "") == _audio8_flash_attention_forward.__name__:
            original_forward = type(attention).forward.__get__(attention, type(attention))
        attention._audio8_original_forward = original_forward
        attention._audio8_apply_rope = modeling._apply_rope
        attention._audio8_flash_function = flash_function
        attention._audio8_flash_state = state
        attention._audio8_valid_cache_length = 0
        attention.forward = types.MethodType(_audio8_flash_attention_forward, attention)


class Audio8TTS(metaclass=SingletonMeta):
    sample_rate = SAMPLE_RATE
    special_settings_defaults = {
        "precision": "auto",
        "attention": "auto",
        "clone_mode": "auto",
        "reference_text": "",
        "seed": -1,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 512,
        "streaming_segment_characters": 140,
        "pause_between_segments_ms": 120,
    }

    def __init__(self):
        self.model = None
        self.processor = None
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
        self.reference_code_cache = OrderedDict()
        MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        VOICES_PATH.mkdir(parents=True, exist_ok=True)
        self.set_compute_device(get_tts_device())

    def list_models(self):
        return MODEL_LIST

    def list_models_indexed(self):
        return tuple(
            {"language": group, "models": models}
            for group, models in self.list_models().items()
        )

    def set_special_setting(self, special_settings):
        if isinstance(special_settings, dict):
            self.special_settings = {
                **self.special_settings_defaults,
                **special_settings,
            }

    def _ensure_special_settings(self):
        all_settings = settings.GetOption("special_settings")
        if not isinstance(all_settings, dict):
            all_settings = {}
        configured = all_settings.get("tts_audio8")
        if isinstance(configured, dict):
            self.special_settings = {
                **self.special_settings_defaults,
                **configured,
            }
        else:
            self.special_settings = dict(self.special_settings_defaults)
            all_settings["tts_audio8"] = dict(self.special_settings)
            settings.SetOption("special_settings", all_settings)

    def set_compute_device(self, requested):
        device = str(requested or "").strip().lower()
        if device in {"", "auto", "cuda"}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("direct-ml"):
            raise ValueError("Audio8 TTS supports CUDA and CPU, but not DirectML.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"Unsupported Audio8 TTS device: {requested}")
        self.compute_device_str = device
        self.compute_device = torch.device(device)

    def _effective_dtype(self):
        requested = get_tts_precision(self.special_settings.get("precision", "auto"))
        if requested in {"float32", "fp32"}:
            return torch.float32
        if self.compute_device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested in {"bfloat16", "bf16"}:
            print("Audio8 BF16 is unavailable on this device; using float32.")
        return torch.float32

    def _get_model_name(self):
        selected = settings.GetOption("tts_model")
        model_name = DEFAULT_MODEL
        if isinstance(selected, (list, tuple)) and len(selected) == 2:
            model_name = re.sub(r"\(.*?\)", "", str(selected[1])).strip()
        return model_name if model_name in MODEL_LANGUAGES else DEFAULT_MODEL

    @staticmethod
    def _model_directory(model_name):
        return MODEL_CACHE_PATH / TTS_MODEL_LINKS[model_name]["path"]

    def download_model(self, model_name=DEFAULT_MODEL, force_non_ui_dl=False):
        import downloader

        entry = TTS_MODEL_LINKS[model_name]
        base_model = entry.get("base_model")
        if base_model and not self.download_model(base_model, force_non_ui_dl):
            return False
        directory = self._model_directory(model_name)
        if not downloader.model_needs_download(directory, entry["file_checksums"]):
            return True
        if entry["checksum"] == "0" * 64:
            raise RuntimeError(
                f"{model_name} is not currently available for automatic download. "
                f"Install its verified files in {directory.resolve()}."
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
                "model_path": VOICES_PATH.parent,
                "model_link_dict": VOICE_MODEL_LINKS,
                "model_name": "voices",
                "title": "Voice samples (Audio8 / Chatterbox)",
                "alt_fallback": False,
                "force_non_ui_dl": force_non_ui_dl,
                "extract_format": "zip",
            },
            self.voice_download_state,
        )

    def release_model(self):
        self.processor = None
        if self.model is not None:
            del self.model
            self.model = None
        self.loaded_configuration = None
        self.reference_code_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self):
        # Startup loading and a first TTS request can otherwise overlap and
        # install the instance-level attention hook twice on the same model.
        with self.generation_lock:
            self._load_model_locked()

    def _load_model_locked(self):
        self._ensure_special_settings()
        self.set_compute_device(get_tts_device())
        model_name = self._get_model_name()
        dtype = self._effective_dtype()
        requested_attention = self.special_settings.get("attention", "auto")
        attention_backend, flash_function = _resolve_attention_backend(
            requested_attention,
            self.compute_device,
            dtype,
        )
        desired = (
            model_name,
            self.compute_device_str,
            str(dtype),
            str(requested_attention),
            attention_backend,
        )
        if self.model is not None and self.loaded_configuration == desired:
            return

        self.download_model(model_name)
        self.download_voices()
        model_directory = self._model_directory(model_name).resolve()
        codec_directory = self._model_directory(CODEC_MODEL).resolve()
        configuration, modeling, processing = _load_verified_runtime(model_directory)

        self.release_model()
        print(
            f"Loading Audio8 TTS {model_name} on {self.compute_device_str} "
            f"with {dtype} and {attention_backend} attention."
        )
        config = configuration.ArkttsConfig.from_pretrained(
            str(model_directory),
            local_files_only=True,
        )
        self.processor = processing.ArkttsProcessor.from_pretrained(
            str(model_directory),
            local_files_only=True,
        )
        self.model = modeling.ArkttsModel.from_pretrained(
            str(model_directory),
            config=config,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval()
        _restore_rotary_buffers(self.model, modeling)
        self.model.to(self.compute_device)
        if attention_backend == "flash_attention_2":
            _enable_audio8_flash_attention(self.model, modeling, flash_function)
        # The verified codec is installed separately from the generator model.
        # Audio8 resolves it from _name_or_path lazily.
        self.model.config._name_or_path = str(codec_directory)
        self.sample_rate = int(self.model.config.codec_sample_rate)
        self.loaded_configuration = desired
        print("Audio8 TTS model loaded.")

    def load(self):
        self.load_model()

    def update_voices(self):
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        return [
            {"name": path.stem, "audio_filename": str(path.resolve())}
            for path in sorted(VOICES_PATH.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]

    def list_voices(self):
        voices = [
            {"name": voice["name"], "value": voice["name"]}
            for voice in self.update_voices()
        ]
        voices.append(
            {"name": "open_voice_dir", "value": "open_dir:" + str(VOICES_PATH.resolve())}
        )
        return voices

    def get_voice_by_name(self, voice_name):
        for voice in self.update_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def _resolve_reference(self, ref_audio=None):
        if ref_audio:
            path = Path(ref_audio).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Audio8 reference audio does not exist: {path}")
            return path
        selected = self.get_voice_by_name(settings.GetOption("tts_voice"))
        if selected is None:
            return None
        return Path(selected["audio_filename"])

    def _reference_text(self, reference_path):
        configured = str(self.special_settings.get("reference_text", "") or "").strip()
        if configured:
            return configured
        if reference_path is None:
            return ""
        sidecar = reference_path.with_suffix(".txt")
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8-sig").strip()
        return ""

    def _clone_reference(self, ref_audio=None):
        mode = str(self.special_settings.get("clone_mode", "auto") or "auto").lower()
        if mode == "disabled":
            return None, ""
        reference_path = self._resolve_reference(ref_audio)
        reference_text = self._reference_text(reference_path)
        if reference_path is not None and reference_text:
            return reference_path, reference_text
        if mode == "required":
            raise ValueError(
                "Audio8 voice cloning requires an exact reference transcript. "
                "Set reference_text or add a same-stem .txt file beside the selected voice."
            )
        if reference_path is not None:
            print(
                "Audio8 selected voice has no exact transcript; generating without "
                "reference conditioning."
            )
        return None, ""

    def _reference_cache_key(self, reference_path):
        stat = reference_path.stat()
        return (
            str(reference_path.resolve()).casefold(),
            stat.st_size,
            stat.st_mtime_ns,
            self.loaded_configuration,
        )

    def _reference_codes(self, reference_path, reference_text):
        key = self._reference_cache_key(reference_path)
        cached = self.reference_code_cache.get(key)
        if cached is not None:
            self.reference_code_cache.move_to_end(key)
            return cached

        probe = self.processor(
            text=["."],
            reference_audio=[str(reference_path)],
            reference_text=[reference_text],
            return_tensors="pt",
        )
        audio_values = probe["reference_audio_values"].to(self.compute_device)
        audio_lengths = probe["reference_audio_lengths"].to(self.compute_device)
        with torch.inference_mode():
            codes, code_lengths = self.model.encode_audio(audio_values, audio_lengths)
        length = int(code_lengths[0])
        codes = codes[0, :, :length].long().cpu()
        self.reference_code_cache[key] = codes
        while len(self.reference_code_cache) > 8:
            self.reference_code_cache.popitem(last=False)
        return codes

    def _generation_generator(self):
        seed = int(self.special_settings.get("seed", -1))
        if seed < 0:
            return None
        generator = torch.Generator(device=self.compute_device)
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _wave_tensor(audio):
        tensor = torch.as_tensor(audio).detach().float().cpu().squeeze()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
        tensor = torch.nan_to_num(tensor).clamp(-1.0, 1.0)
        return tensor.reshape(1, -1).contiguous()

    def _finish_audio(self, audio):
        import Plugins

        wave = self._wave_tensor(audio)
        volume = float(settings.GetOption("tts_volume") or 1.0)
        if volume != 1.0:
            wave = self._wave_tensor(audio_tools.change_volume(wave.numpy(), volume))
        plugin_audio = Plugins.plugin_custom_event_call(
            "plugin_tts_after_audio",
            {"audio": wave, "sample_rate": self.sample_rate},
        )
        if isinstance(plugin_audio, dict) and plugin_audio.get("audio") is not None:
            wave = self._wave_tensor(plugin_audio["audio"])
        return wave

    def _generate_segment(self, text, reference_path, reference_text, generator):
        processor_kwargs = {"text": [text], "return_tensors": "pt"}
        if reference_path is not None:
            processor_kwargs.update(
                reference_codes=[self._reference_codes(reference_path, reference_text)],
                reference_text=[reference_text],
            )
        inputs = self.processor(**processor_kwargs)
        inputs = {
            name: value.to(self.compute_device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        output = self.model.generate(
            **inputs,
            max_new_tokens=max(1, int(self.special_settings.get("max_new_tokens", 512))),
            temperature=max(0.001, float(self.special_settings.get("temperature", 0.8))),
            top_p=min(1.0, max(0.001, float(self.special_settings.get("top_p", 0.95)))),
            top_k=max(1, int(self.special_settings.get("top_k", 50))),
            do_sample=bool(self.special_settings.get("do_sample", True)),
            return_dict_in_generate=True,
            generator=generator,
            stopping_criteria=StoppingCriteriaList([_StopOnEvent(self.stop_event)]),
        )
        code_length = int(output.code_lengths[0])
        if code_length == 0:
            if self.stop_event.is_set():
                return torch.zeros((1, 0))
            raise RuntimeError("Audio8 generated zero acoustic frames.")
        waveforms, waveform_lengths = self.model.decode_audio(output.codes)
        waveform_length = int(waveform_lengths[0])
        if waveform_length == 0:
            raise RuntimeError("Audio8 decoded an empty waveform.")
        return self._finish_audio(waveforms[0, :waveform_length])

    def _segments(self, text):
        goal = int(self.special_settings.get("streaming_segment_characters", 140))
        goal = min(150, max(20, goal))
        return chunk_text(text, goal_length=goal, max_length=150) or [text.strip()]

    def _silence(self):
        pause_ms = max(0, int(self.special_settings.get("pause_between_segments_ms", 120)))
        return torch.zeros((1, int(self.sample_rate * pause_ms / 1000.0)))

    def _synthesize(self, text, streamed, ref_audio=None):
        self.stop_event.clear()
        self.load()
        reference_path, reference_text = self._clone_reference(ref_audio)
        generator = self._generation_generator()
        segments = self._segments(text)
        if streamed:
            self.init_audio_stream_playback()
        chunks = []
        for index, segment in enumerate(segments):
            if self.stop_event.is_set():
                break
            wave = self._generate_segment(
                segment,
                reference_path,
                reference_text,
                generator,
            )
            if wave.numel():
                chunks.append(wave)
                if streamed and self.audio_streamer is not None:
                    self.audio_streamer.add_audio_chunk(self.return_pcm_audio(wave))
            if index < len(segments) - 1 and not self.stop_event.is_set():
                silence = self._silence()
                if silence.numel():
                    chunks.append(silence)
                    if streamed and self.audio_streamer is not None:
                        self.audio_streamer.add_audio_chunk(self.return_pcm_audio(silence))
        final_wave = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, 0))
        self.last_generation = {
            "audio": final_wave,
            "sample_rate": self.sample_rate,
            "text": text.strip(),
        }
        return final_wave, self.sample_rate

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        del remove_silence, silence_after_segments, normalize
        if not text or not text.strip():
            return torch.zeros((1, 0)), self.sample_rate
        with self.generation_lock:
            self._ensure_special_settings()
            return self._synthesize(text, streamed=False, ref_audio=ref_audio)

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
        if not text or not text.strip():
            return torch.zeros((1, 0)), self.sample_rate
        with self.generation_lock:
            self._ensure_special_settings()
            return self._synthesize(text, streamed=True, ref_audio=ref_audio)

    def stop(self):
        self.stop_event.set()
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def save_voice(self):
        audio = self.last_generation.get("audio")
        if audio is None or self._wave_tensor(audio).numel() == 0:
            raise RuntimeError("No Audio8 generation is available to save as a voice reference.")
        stem = "audio8_tts_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        audio_path = VOICES_PATH / f"{stem}.wav"
        text_path = VOICES_PATH / f"{stem}.txt"
        audio_path.write_bytes(
            self.return_wav_file_binary(audio, self.last_generation.get("sample_rate"))
        )
        transcript = str(self.last_generation.get("text") or "").strip()
        if transcript:
            text_path.write_text(transcript, encoding="utf-8")
        print(f"Saved Audio8 clone reference to {audio_path}")
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
