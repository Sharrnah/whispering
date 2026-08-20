import gc
import math
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from packaging.version import Version
from transformers import AutoProcessor, BitsAndBytesConfig

import downloader
from Models.transformers_attention import (
    get_preferred_attention_implementation,
    load_with_attention_fallback,
)


MINIMUM_TRANSFORMERS_VERSION = Version("5.13.0")
DEFAULT_MODEL = "Qwen3-ASR-0.6B-hf"
MODEL_CACHE_PATH = Path(".cache/qwen3-asr")

# Qwen's native inference helper publishes codes for these 30 languages.
STANDARD_LANGUAGES = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

# The Qwen3-ASR checkpoint cards additionally advertise recognition and language
# identification for these 22 Chinese dialects. Qwen publishes their names, but
# no corresponding language codes, so these are stable, application-local
# selector IDs rather than ISO language codes.
DIALECTS = {
    "zh-anhui": "Anhui",
    "zh-dongbei": "Dongbei",
    "zh-fujian": "Fujian",
    "zh-gansu": "Gansu",
    "zh-guizhou": "Guizhou",
    "zh-hebei": "Hebei",
    "zh-henan": "Henan",
    "zh-hubei": "Hubei",
    "zh-hunan": "Hunan",
    "zh-jiangxi": "Jiangxi",
    "zh-ningxia": "Ningxia",
    "zh-shandong": "Shandong",
    "zh-shaanxi": "Shaanxi",
    "zh-shanxi": "Shanxi",
    "zh-sichuan": "Sichuan",
    "zh-tianjin": "Tianjin",
    "zh-yunnan": "Yunnan",
    "zh-zhejiang": "Zhejiang",
    "yue-hk": "Cantonese (HK)",
    "yue-guangdong": "Cantonese (Guangdong)",
    "zh-wu": "Wu",
    "zh-minnan": "Minnan",
}

# Callers use this public mapping to build the language selector, so it includes
# all 52 variants advertised for Qwen3-ASR.
LANGUAGES = {**STANDARD_LANGUAGES, **DIALECTS}

# The official repository and the Hugging Face checkpoint cards use slightly
# different spellings for four dialects. Accept both without duplicate UI rows.
DIALECT_NAME_ALIASES = {
    "cantonese (hong kong accent)": "yue-hk",
    "cantonese (guangdong accent)": "yue-guangdong",
    "wu language": "zh-wu",
    "minnan language": "zh-minnan",
}


# The ZIP SHA-256 values intentionally remain zero until the application-hosted
# archives are created. Each ZIP must contain these files at its root (without
# another Qwen3-ASR-* directory). Replace the corresponding checksum before
# publishing the archive; a missing archive never falls back to Hugging Face.
ARCHIVE_CHECKSUM_PLACEHOLDER = "0" * 64

MODEL_LINKS = {
    # Source checkpoint used for the ZIP:
    # Qwen/Qwen3-ASR-0.6B-hf @ 7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c
    "Qwen3-ASR-0.6B-hf": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Qwen3-ASR/Qwen3-ASR-0.6B-hf.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Qwen3-ASR/Qwen3-ASR-0.6B-hf.zip",
            "https://s3.libs.space:9000/ai-models/Qwen3-ASR/Qwen3-ASR-0.6B-hf.zip",
        ],
        "checksum": "0000000000000000000000000000000000000000000000000000000000000000",
        "file_checksums": {
            "chat_template.jinja": "f50e6b694fbf4a683206e37869990d68333fe95d285730f084c838a34b0d98c2",
            "config.json": "9eecf6f1b383e343889c2e6010e632590fa57d4bc678e151c7d6a160a0dfb04a",
            "generation_config.json": "9939fc9388b79bd70757f938b87381e817173d6a6158f5af6506c0b73e775c3c",
            "model.safetensors": "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35",
            "processor_config.json": "bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a",
            "tokenizer.json": "fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05",
            "tokenizer_config.json": "945e980986de2ca7768f3326bfdbb4fbea3406f972b8ae0be233089f2b253c11",
        },
        "path": "Qwen3-ASR-0.6B-hf",
    },
    # Source checkpoint used for the ZIP:
    # Qwen/Qwen3-ASR-1.7B-hf @ bcd2b5b7f32b480ab5790554cfa8347f246a14f3
    "Qwen3-ASR-1.7B-hf": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Qwen3-ASR/Qwen3-ASR-1.7B-hf.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Qwen3-ASR/Qwen3-ASR-1.7B-hf.zip",
            "https://s3.libs.space:9000/ai-models/Qwen3-ASR/Qwen3-ASR-1.7B-hf.zip",
        ],
        "checksum": "0000000000000000000000000000000000000000000000000000000000000000",
        "file_checksums": {
            "chat_template.jinja": "f50e6b694fbf4a683206e37869990d68333fe95d285730f084c838a34b0d98c2",
            "config.json": "117ac8e63e2af7cae3665e5a632d6eb03f5f384915519ceb6403c15ec6533f63",
            "generation_config.json": "9939fc9388b79bd70757f938b87381e817173d6a6158f5af6506c0b73e775c3c",
            "model.safetensors": "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1",
            "processor_config.json": "bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a",
            "tokenizer.json": "fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05",
            "tokenizer_config.json": "945e980986de2ca7768f3326bfdbb4fbea3406f972b8ae0be233089f2b253c11",
        },
        "path": "Qwen3-ASR-1.7B-hf",
    },
}

DOWNLOAD_STATE = {"is_downloading": False}


def get_languages():
    return tuple(
        [{"code": "", "name": "Auto"}]
        + [{"code": code, "name": name} for code, name in LANGUAGES.items()]
    )


def get_model_path(model_name):
    if model_name == "custom":
        return MODEL_CACHE_PATH / "custom"
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown Qwen3-ASR model: {model_name}")
    return MODEL_CACHE_PATH / MODEL_LINKS[model_name]["path"]


def needs_download(model_name):
    if model_name == "custom":
        return False
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown Qwen3-ASR model: {model_name}")
    model_path = get_model_path(model_name)
    expected_hashes = MODEL_LINKS[model_name]["file_checksums"]
    return downloader.model_needs_download(model_path, expected_hashes)


def download_model(model_name, force_non_ui_dl=False):
    if model_name == "custom":
        return get_model_path(model_name).is_dir()
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown Qwen3-ASR model: {model_name}")
    if not needs_download(model_name):
        return True

    model_entry = MODEL_LINKS[model_name]
    if model_entry["checksum"] == ARCHIVE_CHECKSUM_PLACEHOLDER:
        archive_name = Path(model_entry["urls"][0]).name
        raise RuntimeError(
            f"The Qwen3-ASR model archive {archive_name} is not currently "
            "available for automatic download."
        )

    return downloader.download_model(
        {
            "model_path": MODEL_CACHE_PATH,
            "model_link_dict": MODEL_LINKS,
            "model_name": model_name,
            "title": f"Speech to Text (Qwen3-ASR) - {model_name}",
            "alt_fallback": False,
            "force_non_ui_dl": force_non_ui_dl,
            "extract_format": "zip",
        },
        DOWNLOAD_STATE,
    )


class Qwen3ASR:
    def __init__(self, compute_type="float32", device="cpu"):
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        self.model = None
        self.processor = None
        self.compute_type = compute_type
        self.compute_device = torch.device("cpu")
        self.compute_device_str = "cpu"
        self.loaded_configuration = None
        self.set_compute_device(device)

    @staticmethod
    def _dtype_settings(compute_type):
        if compute_type == "float16":
            return {"dtype": torch.float16, "4bit": False, "8bit": False}
        if compute_type == "bfloat16":
            return {"dtype": torch.bfloat16, "4bit": False, "8bit": False}
        if compute_type == "float32":
            return {"dtype": torch.float32, "4bit": False, "8bit": False}
        if compute_type == "4bit":
            return {"dtype": torch.float16, "4bit": True, "8bit": False}
        if compute_type == "8bit":
            return {"dtype": torch.float16, "4bit": False, "8bit": True}
        raise ValueError(
            f"Unsupported Qwen3-ASR precision '{compute_type}'. "
            "Use float32, float16, bfloat16, 8bit, or 4bit."
        )

    def set_compute_type(self, compute_type):
        self._dtype_settings(compute_type)
        self.compute_type = compute_type

    def set_compute_device(self, device):
        device_name = str(device or "").lower()
        if device_name in {"", "none", "auto", "cuda"}:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name.startswith("direct-ml"):
            raise ValueError("Qwen3-ASR currently supports CUDA and CPU devices, but not DirectML.")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        if device_name != "cpu" and not device_name.startswith("cuda"):
            raise ValueError(f"Unsupported Qwen3-ASR device: {device}")
        self.compute_device_str = device_name
        self.compute_device = torch.device(device_name)

    @staticmethod
    def _check_transformers_version():
        installed = Version(transformers.__version__)
        if installed < MINIMUM_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Qwen3-ASR requires Transformers 5.13.0 or newer; "
                f"found {transformers.__version__}."
            )

    def load_model(self, model=DEFAULT_MODEL, compute_type="float32", device="cpu"):
        self._check_transformers_version()
        # Keep the Qwen-only auto class lazy so installations that have not yet
        # applied the dependency update can still start with another STT type.
        from transformers import AutoModelForMultimodalLM

        self.set_compute_type(compute_type)
        self.set_compute_device(device)
        load_configuration = (model, self.compute_type, self.compute_device_str)
        if self.model is not None and self.processor is not None and load_configuration == self.loaded_configuration:
            return

        model_path = get_model_path(model)
        if model != "custom" and not download_model(model):
            raise RuntimeError(f"Could not download Qwen3-ASR model '{model}'.")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Qwen3-ASR model directory does not exist: {model_path.resolve()}")

        self.release_model()
        dtype_settings = self._dtype_settings(self.compute_type)
        compute_dtype = dtype_settings["dtype"]
        quantization_config = None
        if dtype_settings["4bit"] or dtype_settings["8bit"]:
            if self.compute_device.type != "cuda":
                raise ValueError("Qwen3-ASR 4-bit and 8-bit inference requires CUDA.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=dtype_settings["4bit"],
                load_in_8bit=dtype_settings["8bit"],
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        attention_type = get_preferred_attention_implementation(
            self.compute_device,
            compute_dtype,
        )

        resolved_path = str(model_path.resolve())
        print(
            f"Loading Qwen3-ASR model: {model} on {self.compute_device_str} "
            f"with {self.compute_type} precision using {attention_type}..."
        )
        def model_loader(attention_implementation):
            return AutoModelForMultimodalLM.from_pretrained(
                resolved_path,
                dtype=compute_dtype,
                quantization_config=quantization_config,
                device_map=self.compute_device,
                attn_implementation=attention_implementation,
                local_files_only=True,
            )

        loaded_model, attention_type = load_with_attention_fallback(
            model_loader,
            attention_type,
            f"Qwen3-ASR {model}",
        )
        loaded_processor = AutoProcessor.from_pretrained(
            resolved_path,
            local_files_only=True,
        )
        loaded_model.eval()
        self.model = loaded_model
        self.processor = loaded_processor
        self.loaded_configuration = load_configuration
        print(f"Qwen3-ASR model loaded successfully using {attention_type}.")

    @staticmethod
    def _resolve_language(language):
        if language is None:
            return None, None
        language = str(language).strip()
        if not language or language.lower() in {"auto", "none", "null"}:
            return None, None
        code = language.lower()
        if code in LANGUAGES:
            return code, LANGUAGES[code]
        for known_code, known_name in LANGUAGES.items():
            if language.casefold() == known_name.casefold():
                return known_code, known_name
        alias_code = DIALECT_NAME_ALIASES.get(language.casefold())
        if alias_code is not None:
            return alias_code, LANGUAGES[alias_code]
        raise ValueError(f"Unsupported Qwen3-ASR language: {language}")

    @staticmethod
    def _language_code_from_name(language_name):
        normalized_name = str(language_name).strip().casefold()
        for language_code, canonical_name in LANGUAGES.items():
            if canonical_name.casefold() == normalized_name:
                return language_code
        return DIALECT_NAME_ALIASES.get(normalized_name)

    def _prepare_inputs(self, audio_sample, prompt, language_name):
        messages = []
        if prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": prompt}]})
        messages.append(
            {"role": "user", "content": [{"type": "audio", "audio": audio_sample}]}
        )

        template_options = {"add_generation_prompt": True}
        if language_name:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"language {language_name}<asr_text>"}],
                }
            )
            template_options = {"continue_final_message": True}

        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            **template_options,
        ).to(self.model.device, self.model.dtype)

    def transcribe(
        self,
        audio_sample,
        model=DEFAULT_MODEL,
        task="transcribe",
        language=None,
        return_timestamps=False,
        beam_size=1,
        prompt=None,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        **kwargs,
    ):
        del return_timestamps, kwargs  # Timestamps require the separate forced-aligner checkpoint.
        if task not in {None, "", "transcribe"}:
            raise ValueError("Qwen3-ASR supports transcription, not speech translation.")

        self.load_model(model, self.compute_type, self.compute_device_str)
        audio = np.asarray(audio_sample, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "type": "transcribe", "language": language}

        requested_code, requested_name = self._resolve_language(language)
        clean_prompt = None if prompt is None else str(prompt).strip()
        inputs = self._prepare_inputs(audio, clean_prompt, requested_name)
        # Qwen's official Transformers helper uses greedy decoding. Reuse the
        # application's existing Whisper beam controls: 1 preserves that fast,
        # low-memory path, while values above 1 opt into Transformers beam search.
        beam_count = max(1, int(beam_size or 1))
        generation_options = {
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": beam_count,
            # This is already the model default, but keep it explicit because
            # reprocessing the complete decoder prefix for every token is costly.
            "use_cache": True,
        }

        # These are generic Transformers decoding controls, not Qwen/Whisper
        # temperature-fallback options. Keep neutral values out of generate()
        # so greedy decoding retains Qwen's published generation defaults and
        # avoids installing unnecessary logits processors.
        try:
            normalized_repetition_penalty = float(repetition_penalty)
        except (TypeError, ValueError):
            normalized_repetition_penalty = 1.0
        if not math.isfinite(normalized_repetition_penalty) or normalized_repetition_penalty <= 0:
            normalized_repetition_penalty = 1.0
        if normalized_repetition_penalty != 1.0:
            generation_options["repetition_penalty"] = normalized_repetition_penalty

        try:
            normalized_no_repeat_ngram_size = max(0, int(no_repeat_ngram_size or 0))
        except (TypeError, ValueError):
            normalized_no_repeat_ngram_size = 0
        if normalized_no_repeat_ngram_size > 0:
            generation_options["no_repeat_ngram_size"] = normalized_no_repeat_ngram_size

        # Transformers only uses length_penalty for beam-based generation.
        if beam_count > 1:
            try:
                normalized_length_penalty = float(length_penalty)
            except (TypeError, ValueError):
                normalized_length_penalty = 1.0
            if math.isfinite(normalized_length_penalty) and normalized_length_penalty != 1.0:
                generation_options["length_penalty"] = normalized_length_penalty

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_options)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(generated_ids, return_format="parsed")
        if isinstance(parsed, list):
            parsed = parsed[0]

        transcription = parsed.get("transcription", "").strip()
        detected_name = parsed.get("language")
        detected_code = requested_code
        if detected_code is None and detected_name:
            detected_code = self._language_code_from_name(detected_name) or detected_name

        return {
            "text": transcription,
            "type": "transcribe",
            "language": detected_code,
        }

    def release_model(self):
        if self.model is not None:
            print("Releasing Qwen3-ASR model...")
        self.model = None
        self.processor = None
        self.loaded_configuration = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
