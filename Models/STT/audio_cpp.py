"""Speech-to-text families served by the native audio.cpp GGUF runtime."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import settings
from Models.audio_cpp_runtime import AudioCppServer, normalize_backend_device


DEFAULT_MODEL = "Qwen3-ASR-0.6B-GGUF"
MODEL_CACHE_PATH = Path.cwd() / ".cache" / "audio.cpp" / "models"
GGUF_REVISION = "78f9d27aa214792b77256affe774eea57e35b9ae"
GGUF_REPOSITORY_ROOT = (
    "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/"
    f"{GGUF_REVISION}"
)
SPECIAL_SETTINGS_NAME = "stt_audio_cpp"


def _variant(path: str, sha256: str, size: int) -> dict:
    filename = Path(path).name
    return {
        "urls": [f"{GGUF_REPOSITORY_ROOT}/{path}"],
        "checksum": sha256,
        "file_checksums": {filename: sha256},
        "filename": filename,
        "size": size,
    }


STT_MODELS = {
    "Qwen3-ASR-0.6B-GGUF": {
        "family": "qwen3_asr", "description": "Recommended multilingual general ASR",
        "streaming": True, "timestamps": False, "settings_key": "qwen3_asr",
        "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf", "6c44ec2fb4cee513892d7863c1fcc3ea6b699ffa4d899b0ef4ab19956d9544f7", 1_151_272_416),
            "f16": _variant("Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-f16.gguf", "5472337d26df8e58fdfabc3ae58a149fbb200578c576b4c660f2985c07da1736", 1_880_642_016),
        },
    },
    "Qwen3-ASR-1.7B-GGUF": {
        "family": "qwen3_asr", "description": "Higher-accuracy multilingual general ASR",
        "streaming": True, "timestamps": False, "settings_key": "qwen3_asr",
        "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf", "da4fc2ac7f24dee784d1684eb1f35836cdbf559519452ae11777670734c0a4f8", 2_473_010_048),
            "f16": _variant("Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-f16.gguf", "f12537d4ea56df4e1dcca64a902e0b37fb1111f8ef7fc8e554fd00110be047d0", 4_087_653_248),
        },
    },
    "Nemotron-3.5-ASR-Streaming-0.6B-GGUF": {
        "family": "nemotron_asr", "description": "Fast 40-locale streaming RNNT with token timestamps",
        "streaming": True, "timestamps": True, "settings_key": "nemotron_asr",
        "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf", "7026c80d8b94b5184c3acc20b34c410787a35babab7ec33931bd2913cd022afe", 930_620_256),
            "f16": _variant("Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf", "8bef32306425ebe62484160ee1450fa477f92d0e6e03b13cf4e5fa158254137e", 1_277_710_880),
        },
    },
    "VibeVoice-ASR-GGUF": {
        "family": "vibevoice_asr", "description": "Large meeting ASR with segments and speaker turns",
        "streaming": False, "timestamps": True, "settings_key": "vibevoice_asr",
        "default_precision": "q8_0",
        "variants": {
            "q8_0": _variant("VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf", "71a821bcfe36370b906feae84b413572599d30c766f114a9505fd2716efe12af", 9_858_644_224),
            "f16": _variant("VibeVoice-ASR-GGUF/vibevoice-asr-f16.gguf", "4a1f7da42b495e0be5b85c7a51ca843e1b5d1811f4c9e930d71811fcec386019", 17_361_090_304),
        },
    },
    "Voxtral-Mini-4B-Realtime-2602-GGUF": {
        "family": "voxtral_realtime", "description": "Realtime multilingual ASR; no timestamps",
        "streaming": True, "timestamps": False, "settings_key": "voxtral_realtime",
        "default_precision": "q4_k",
        "variants": {
            "q4_k": _variant("Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q4_k.gguf", "8cafef18ea3e4cad81da8ffc4e72b69d2eab2c159c2e68428e2e088accbfc7f8", 3_097_662_432),
            "q8_0": _variant("Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf", "0312a5ceafc6ee4a19a32da458cab6485e3b86b1b563c7bb7150aeae295c1769", 5_104_567_264),
            "bf16": _variant("Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-bf16.gguf", "0e98f67a3f2cad98a5488f139a23551b7743e582f84f82a0767257dd52a0f3ab", 8_874_402_784),
        },
    },
    "Audio8-ASR-0.1B-GGUF": {
        "family": "audio8_asr", "description": "Compact local-only multilingual ASR (CC-BY-NC-4.0)",
        "streaming": False, "timestamps": False, "settings_key": "audio8_asr",
        "default_precision": "q8_0", "local_only": True,
        "variants": {
            "q8_0": {"filename": "audio8-asr-0.1b-q8_0.gguf", "size": 345_000_000},
            "f16": {"filename": "audio8-asr-0.1b-f16.gguf", "size": 600_000_000},
        },
    },
    "Kroko-ASR-English-64L-GGUF": {
        "family": "kroko_asr", "description": "Very small English streaming ASR with word timestamps",
        "streaming": True, "timestamps": True, "settings_key": "kroko_asr",
        "default_precision": "q8_0", "language": "en",
        "variants": {
            "q8_0": _variant("Kroko-ASR-GGUF/kroko-en-community-64-l-q8_0.gguf", "596c749fcb6c582f2d7aaa9aefcff2929e405867a4d04864a55b2c9c2963baa5", 167_756_928),
        },
    },
}

MODEL_VARIANTS = {name: definition["variants"] for name, definition in STT_MODELS.items()}
MODEL_ALIASES = {
    "Qwen3-ASR-0.6B-hf": "Qwen3-ASR-0.6B-GGUF",
    "Qwen3-ASR-1.7B-hf": "Qwen3-ASR-1.7B-GGUF",
    "Qwen3-ASR-0.6B": "Qwen3-ASR-0.6B-GGUF",
    "Qwen3-ASR-1.7B": "Qwen3-ASR-1.7B-GGUF",
    "nemotron_asr": "Nemotron-3.5-ASR-Streaming-0.6B-GGUF",
    "vibevoice_asr": "VibeVoice-ASR-GGUF",
    "voxtral_realtime": "Voxtral-Mini-4B-Realtime-2602-GGUF",
    "audio8_asr": "Audio8-ASR-0.1B-GGUF",
    "kroko_asr": "Kroko-ASR-English-64L-GGUF",
}
DOWNLOAD_STATES = {
    model: {precision: {"is_downloading": False} for precision in definition["variants"]}
    for model, definition in STT_MODELS.items() if not definition.get("local_only")
}

NEMOTRON_LOCALES = (
    "ar-AR", "bg-BG", "cs-CZ", "da-DK", "de-DE", "el-GR", "en-GB", "en-US",
    "es-ES", "es-US", "et-EE", "fi-FI", "fr-CA", "fr-FR", "he-IL", "hi-IN",
    "hr-HR", "hu-HU", "it-IT", "ja-JP", "ko-KR", "lt-LT", "lv-LV", "mt-MT",
    "nb-NO", "nl-NL", "nn-NO", "pl-PL", "pt-BR", "pt-PT", "ro-RO", "ru-RU",
    "sk-SK", "sl-SI", "sv-SE", "th-TH", "tr-TR", "uk-UA", "vi-VN", "zh-CN",
)
DEFAULT_LOCALE_BY_LANGUAGE = {
    locale.split("-", 1)[0].lower(): locale for locale in NEMOTRON_LOCALES
}
DEFAULT_LOCALE_BY_LANGUAGE.update({"en": "en-US", "es": "es-ES", "fr": "fr-FR", "pt": "pt-BR"})

STT_SETTINGS_DEFAULTS = {
    "qwen3_asr": {
        "mode": "auto", "max_tokens": 512, "audio_chunk_seconds": 30.0,
        "audio_chunk_mode": "auto", "weight_type": "native",
        "audio_encoder_weight_type": "native", "thinker_weight_type": "native",
        "audio_encoder_graph_arena_mb": 128, "thinker_prefill_graph_arena_mb": 256,
        "thinker_decode_graph_arena_mb": 256, "thinker_weight_context_mb": 64,
        "vad_model_path": "",
    },
    "nemotron_asr": {
        "mode": "auto", "lookahead_tokens": 0, "max_tokens": 0,
        "keep_language_tags": False, "weight_type": "native",
        "matmul_weight_type": "native", "conv_weight_type": "native",
        "weight_context_mb": 0, "encoder_graph_arena_mb": 0,
        "decoder_graph_arena_mb": 0, "mem_saver": False,
    },
    "vibevoice_asr": {
        "mode": "offline", "max_tokens": 0, "temperature": 0.0, "top_p": 1.0,
        "top_k": 0, "num_beams": 1, "repetition_penalty": 1.0, "seed": -1,
        "audio_chunk_mode": "auto", "audio_chunk_seconds": 1200.0,
        "weight_type": "native", "tokenizer_weight_type": "native",
        "connector_weight_type": "native", "decoder_weight_type": "native",
        "tokenizer_weight_context_mb": 0, "connector_weight_context_mb": 0,
        "decoder_weight_context_mb": 0, "vad_model_path": "",
    },
    "voxtral_realtime": {
        "mode": "auto", "max_new_tokens": 0, "do_sample": False,
        "temperature": 1.0, "top_p": 1.0, "top_k": 50, "seed": 1234,
        "stream_batch_tokens": 4, "stream_decode_cache_steps": 1024,
        "weight_type": "native", "audio_encoder_weight_type": "native",
        "text_decoder_weight_type": "native", "audio_encoder_graph_arena_mb": 0,
        "audio_encoder_weight_context_mb": 0, "text_decoder_prefill_graph_arena_mb": 0,
        "text_decoder_decode_graph_arena_mb": 0, "text_decoder_weight_context_mb": 0,
    },
    "audio8_asr": {
        "mode": "offline", "weight_type": "native", "audio_encoder_weight_type": "native",
        "encoder_graph_arena_mb": 128, "projector_graph_arena_mb": 256,
        "prefill_graph_arena_mb": 256, "decode_graph_arena_mb": 256,
    },
    "kroko_asr": {
        "mode": "auto", "decoding_method": "greedy_search", "num_beams": 4,
        "blank_penalty": 0.0, "hotwords": "", "hotwords_score": 1.5,
        "enable_endpoint": False, "rule1_min_trailing_silence_sec": 2.4,
        "rule2_min_trailing_silence_sec": 1.2, "rule3_min_utterance_length_sec": 20.0,
    },
}

REQUEST_KEYS = {
    "qwen3_asr": ("max_tokens", "audio_chunk_seconds", "audio_chunk_mode"),
    "nemotron_asr": ("lookahead_tokens", "max_tokens", "keep_language_tags"),
    "vibevoice_asr": ("max_tokens", "temperature", "top_p", "top_k", "num_beams", "repetition_penalty", "seed", "audio_chunk_mode", "audio_chunk_seconds"),
    "voxtral_realtime": ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "seed"),
    "audio8_asr": (),
    "kroko_asr": ("decoding_method", "num_beams", "blank_penalty", "hotwords", "hotwords_score", "enable_endpoint", "rule1_min_trailing_silence_sec", "rule2_min_trailing_silence_sec", "rule3_min_utterance_length_sec"),
}

SESSION_KEYS = {
    "qwen3_asr": ("weight_type", "audio_encoder_weight_type", "thinker_weight_type", "audio_encoder_graph_arena_mb", "thinker_prefill_graph_arena_mb", "thinker_decode_graph_arena_mb", "thinker_weight_context_mb", "vad_model_path"),
    "nemotron_asr": ("weight_type", "matmul_weight_type", "conv_weight_type", "weight_context_mb", "encoder_graph_arena_mb", "decoder_graph_arena_mb", "mem_saver"),
    "vibevoice_asr": ("weight_type", "tokenizer_weight_type", "connector_weight_type", "decoder_weight_type", "tokenizer_weight_context_mb", "connector_weight_context_mb", "decoder_weight_context_mb", "vad_model_path"),
    "voxtral_realtime": ("stream_batch_tokens", "stream_decode_cache_steps", "weight_type", "audio_encoder_weight_type", "text_decoder_weight_type", "audio_encoder_graph_arena_mb", "audio_encoder_weight_context_mb", "text_decoder_prefill_graph_arena_mb", "text_decoder_decode_graph_arena_mb", "text_decoder_weight_context_mb"),
    "audio8_asr": ("weight_type", "audio_encoder_weight_type", "encoder_graph_arena_mb", "projector_graph_arena_mb", "prefill_graph_arena_mb", "decode_graph_arena_mb"),
}


def get_languages():
    # Reuse Qwen's canonical names, then add locale selectors used by Nemotron.
    from Models.STT.qwen3_asr import get_languages as qwen_get_languages
    languages = list(qwen_get_languages())
    known = {item["code"].casefold() for item in languages}
    for locale in NEMOTRON_LOCALES:
        if locale.casefold() not in known:
            languages.append({"code": locale, "name": locale})
    return tuple(languages)


def normalize_model(model: str | None) -> str:
    selected = str(model or DEFAULT_MODEL).strip()
    if selected in STT_MODELS:
        return selected
    alias = MODEL_ALIASES.get(selected) or MODEL_ALIASES.get(selected.casefold())
    if alias:
        return alias
    raise ValueError(f"Unknown audio.cpp ASR model: {model}")


def precision_options(model: str = DEFAULT_MODEL) -> tuple[str, ...]:
    return tuple(STT_MODELS[normalize_model(model)]["variants"])


def normalize_precision(precision: str | None, model: str = DEFAULT_MODEL) -> str:
    model = normalize_model(model)
    variants = STT_MODELS[model]["variants"]
    selected = str(precision or "auto").strip().lower()
    selected = {
        "q8": "q8_0", "int8": "q8_0", "8bit": "q8_0", "int8_float16": "q8_0",
        "int8_bfloat16": "q8_0", "4bit": "q4_k", "q4": "q4_k", "float16": "f16",
        "fp16": "f16", "float32": "f16", "bfloat16": "bf16", "int16": "f16",
    }.get(selected, selected)
    if selected in {"auto", "default"}:
        return STT_MODELS[model]["default_precision"]
    if selected in variants:
        return selected
    if selected in {"f16", "bf16"}:
        for fallback in ("bf16", "f16", "q8_0", "q4_k"):
            if fallback in variants:
                return fallback
    if selected in {"q8_0", "q4_k"}:
        for fallback in ("q8_0", "q4_k", "f16", "bf16"):
            if fallback in variants:
                return fallback
    return STT_MODELS[model]["default_precision"]


def is_managed_model(model: str) -> bool:
    return not bool(STT_MODELS[normalize_model(model)].get("local_only"))


def get_model_path(model: str, precision: str) -> Path:
    model = normalize_model(model)
    precision = normalize_precision(precision, model)
    entry = STT_MODELS[model]["variants"][precision]
    nested = MODEL_CACHE_PATH / model / precision / entry["filename"]
    if STT_MODELS[model].get("local_only"):
        flat = MODEL_CACHE_PATH / model / entry["filename"]
        if nested.is_file() or not flat.is_file():
            return nested
        return flat
    return nested


def needs_download(model: str, precision: str) -> bool:
    model = normalize_model(model)
    if not is_managed_model(model):
        return False
    precision = normalize_precision(precision, model)
    entry = STT_MODELS[model]["variants"][precision]
    import downloader
    return downloader.model_needs_download(MODEL_CACHE_PATH / model / precision, entry["file_checksums"])


def download_model(model: str, precision: str, force_non_ui_dl: bool = False) -> bool:
    model = normalize_model(model)
    precision = normalize_precision(precision, model)
    if not is_managed_model(model):
        path = get_model_path(model, precision)
        if path.is_file():
            return True
        raise FileNotFoundError(
            "Audio8-ASR is CC-BY-NC-4.0 and audio.cpp does not redistribute its converted "
            f"weights. Convert or privately self-host the checkpoint, then place it at {path.resolve()}."
        )
    entry = STT_MODELS[model]["variants"][precision]
    if not needs_download(model, precision):
        return True
    import downloader
    return downloader.download_model({
        "model_path": MODEL_CACHE_PATH,
        "model_link_dict": {"model": {**entry, "path": str(Path(model) / precision)}},
        "model_name": "model", "title": f"Speech to Text (audio.cpp) - {model} {precision}",
        "alt_fallback": False, "force_non_ui_dl": force_non_ui_dl, "extract_format": "none",
    }, DOWNLOAD_STATES[model][precision])


def _dict_value(value):
    return value if isinstance(value, dict) else {}


def _special_settings_for(model: str) -> dict:
    key = STT_MODELS[model]["settings_key"]
    defaults = STT_SETTINGS_DEFAULTS[key]
    configured = _configured_special_settings_for(model)
    return {**defaults, **configured}


def _configured_special_settings_for(model: str) -> dict:
    key = STT_MODELS[model]["settings_key"]
    root = _dict_value(_dict_value(settings.GetOption("special_settings")).get(SPECIAL_SETTINGS_NAME))
    return _dict_value(root.get(key))


def _session_options(model: str, configured: dict) -> dict:
    definition = STT_MODELS[model]
    family = definition["family"]
    key = definition["settings_key"]
    result = {}
    for name in SESSION_KEYS.get(key, ()):
        # Precision chooses the actual pre-converted GGUF file. Per-component
        # storage overrides can conflict with that file's native tensor types,
        # so legacy profile values are intentionally ignored.
        if name == "weight_type" or name.endswith("_weight_type"):
            continue
        value = configured.get(name)
        if value is None or value == "":
            continue
        # Zero means "let audio.cpp use the model-derived arena default" for
        # optional memory settings. Audio8's documented defaults are non-zero.
        if name.endswith(("_mb", "_steps")) and value == 0:
            continue
        result[f"{family}.{name}"] = value
    return result


def _audio_to_wav(audio_sample) -> bytes:
    audio = np.asarray(audio_sample, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


@contextmanager
def _temporary_wav(audio_sample):
    handle = tempfile.NamedTemporaryFile(
        prefix="whispering-tiger-audiocpp-asr-", suffix=".wav", delete=False
    )
    path = Path(handle.name)
    try:
        handle.write(_audio_to_wav(audio_sample))
        handle.close()
        yield str(path.resolve())
    finally:
        try:
            handle.close()
        except Exception:
            pass
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _qwen_language(language):
    from Models.STT.qwen3_asr import Qwen3ASR

    code, name = Qwen3ASR._resolve_language(language)
    return name or "", code


def _basic_language(language):
    selected = str(language or "").strip()
    if selected.casefold() in {"", "auto", "none", "null"}:
        return "", None
    return selected, selected.lower()


def _language_for_model(model: str, language):
    family = STT_MODELS[model]["family"]
    if family == "qwen3_asr":
        return _qwen_language(language)
    native, code = _basic_language(language)
    if family == "nemotron_asr":
        if not native:
            return "auto", None
        for locale in NEMOTRON_LOCALES:
            if native.casefold() == locale.casefold():
                return locale, locale
        default_locale = DEFAULT_LOCALE_BY_LANGUAGE.get(code.split("-", 1)[0])
        if default_locale:
            return default_locale, default_locale
        raise ValueError(f"Unsupported Nemotron ASR language/locale: {language}")
    if family == "kroko_asr":
        if native and code != "en":
            raise ValueError(
                "The bundled Kroko audio.cpp checkpoint is English-only; "
                "select Auto/English or another ASR model."
            )
        return "en", "en"
    if family in {"audio8_asr", "voxtral_realtime"}:
        # These checkpoints auto-detect and expose no forced-language control.
        return "", None
    return native, code


def _result_language(model: str, value, requested_code):
    detected = str(value or "").strip()
    if not detected:
        return requested_code
    if STT_MODELS[model]["family"] == "qwen3_asr":
        from Models.STT.qwen3_asr import Qwen3ASR

        return Qwen3ASR._language_code_from_name(detected) or detected
    return detected


def _timed_items(items):
    normalized = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        value = dict(item)
        try:
            if "start_sample" in value:
                value["start"] = float(value["start_sample"]) / 16_000.0
            if "end_sample" in value:
                value["end"] = float(value["end_sample"]) / 16_000.0
        except (TypeError, ValueError):
            pass
        normalized.append(value)
    return normalized


class AudioCppASR:
    """Whispering Tiger adapter for audio.cpp ASR model families."""

    def __init__(
        self,
        compute_type="q8_0",
        device="cpu",
        device_index=0,
        cpu_threads=0,
        role="stt",
    ):
        MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.server = AudioCppServer(role)
        self.role = role
        self.compute_type = compute_type
        self.compute_device = "cpu"
        self.device_index = 0
        self.cpu_threads = max(0, int(cpu_threads or 0))
        self.loaded_configuration = None
        self.current_model = DEFAULT_MODEL
        self.lock = threading.RLock()
        self._timestamp_warning_models = set()
        self.set_compute_device(device, device_index)

    def set_compute_type(self, compute_type):
        self.compute_type = str(compute_type or "auto")

    def set_compute_device(self, device, device_index=0):
        self.compute_device, self.device_index = normalize_backend_device(device, device_index)

    def _mode(self, model: str, configured: dict) -> str:
        definition = STT_MODELS[model]
        selected = str(configured.get("mode", "auto") or "auto").strip().lower()
        if selected not in {"auto", "offline", "streaming"}:
            selected = "auto"
        if selected == "auto":
            selected = "streaming" if self.role == "realtime-stt" else "offline"
        if selected == "streaming" and not definition["streaming"]:
            return "offline"
        return selected

    def load_model(self, model=DEFAULT_MODEL, compute_type=None, device=None, device_index=None):
        model = normalize_model(model)
        if compute_type is not None:
            self.set_compute_type(compute_type)
        if device is not None:
            self.set_compute_device(device, self.device_index if device_index is None else device_index)
        precision = normalize_precision(self.compute_type, model)
        configured = _special_settings_for(model)
        mode = self._mode(model, configured)
        session_options = _session_options(model, configured)
        desired = (
            model,
            precision,
            self.compute_device,
            self.device_index,
            self.cpu_threads,
            mode,
            json.dumps(session_options, sort_keys=True),
        )
        if desired == self.loaded_configuration and self.server.process is not None:
            if self.server.process.poll() is None:
                return True

        if is_managed_model(model):
            if not download_model(model, precision):
                raise RuntimeError(f"Could not download audio.cpp ASR model {model} ({precision}).")
        model_path = get_model_path(model, precision)
        if not model_path.is_file() and not model_path.is_dir():
            if STT_MODELS[model].get("local_only"):
                raise FileNotFoundError(
                    "Audio8-ASR is local-only under its CC-BY-NC-4.0 terms. Convert your "
                    f"own checkpoint and place the GGUF at {model_path.resolve()}."
                )
            raise FileNotFoundError(f"audio.cpp ASR model does not exist: {model_path.resolve()}")

        self.server.configure(
            backend=self.compute_device,
            device_index=self.device_index,
            family=STT_MODELS[model]["family"],
            model_path=model_path,
            task="asr",
            mode=mode,
            threads=self.cpu_threads,
            session_options=session_options,
        )
        self.current_model = model
        self.loaded_configuration = desired
        print(
            f"audio.cpp {model} loaded on {self.compute_device}:{self.device_index} "
            f"with {precision} GGUF weights ({mode})."
        )
        return True

    def transcribe(
        self,
        audio_sample,
        model=DEFAULT_MODEL,
        task="transcribe",
        language=None,
        return_timestamps=False,
        beam_size=1,
        prompt=None,
        repetition_penalty=1.0,
        **kwargs,
    ):
        if task not in {None, "", "transcribe"}:
            raise ValueError("audio.cpp ASR models support transcription, not speech translation.")
        selected_model = normalize_model(model)
        audio = np.asarray(audio_sample, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "type": "transcribe", "language": language}

        with self.lock:
            self.load_model(selected_model)
            definition = STT_MODELS[selected_model]
            configured = _special_settings_for(selected_model)
            explicit_settings = _configured_special_settings_for(selected_model)
            options = {}
            for name in REQUEST_KEYS.get(definition["settings_key"], ()):
                value = configured.get(name)
                if value is None or value == "":
                    continue
                if name in {"max_tokens", "max_new_tokens", "lookahead_tokens"} and value == 0:
                    continue
                if name == "seed" and int(value) < 0:
                    continue
                options[name] = value

            family = definition["family"]
            if family in {"vibevoice_asr", "kroko_asr"}:
                if "num_beams" not in explicit_settings:
                    options["num_beams"] = max(1, int(beam_size or options.get("num_beams", 1)))
            if family == "vibevoice_asr":
                if "repetition_penalty" not in explicit_settings:
                    try:
                        penalty = float(repetition_penalty)
                    except (TypeError, ValueError):
                        penalty = 1.0
                    if penalty > 0:
                        options["repetition_penalty"] = penalty

            native_language, requested_code = _language_for_model(selected_model, language)
            wants_timestamps = bool(return_timestamps and definition["timestamps"])
            if return_timestamps and not definition["timestamps"] and selected_model not in self._timestamp_warning_models:
                print(f"{selected_model} does not provide timestamps; returning transcript text only.")
                self._timestamp_warning_models.add(selected_model)
            if wants_timestamps:
                options["return_timestamps"] = True

            request_payload = {
                "audio": None,
                "text": str(prompt or "").strip(),
                "options": options,
            }
            if native_language:
                request_payload["language"] = native_language

            with _temporary_wav(audio) as audio_path:
                request_payload["audio"] = audio_path
                response = self.server.request(
                    "POST",
                    "/v1/tasks/run",
                    json={"model": self.server.model_id, "request": request_payload},
                )
                try:
                    payload = response.json()
                finally:
                    response.close()

        if not isinstance(payload, dict):
            raise RuntimeError("audio.cpp returned an invalid ASR response.")
        result = {
            "text": str(payload.get("text") or "").strip(),
            "type": "transcribe",
            "language": _result_language(selected_model, payload.get("language"), requested_code),
        }
        segments = _timed_items(payload.get("segments"))
        turns = _timed_items(payload.get("speaker_turns"))
        words = _timed_items(payload.get("words"))
        if segments:
            result["segments"] = segments
        if turns:
            result["speaker_turns"] = turns
        if words:
            result["words"] = words
        return result

    def release_model(self):
        with self.lock:
            self.server.stop()
            self.loaded_configuration = None
