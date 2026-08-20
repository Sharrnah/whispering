import gc
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import downloader
from Models import languageClassification
from Models.transformers_attention import (
    get_preferred_attention_implementation,
    load_with_attention_fallback,
)


DEFAULT_MODEL = "MiLMMT-46-1B-v1.0"
MODEL_CACHE_PATH = Path(".cache/milmmt")
ARCHIVE_CHECKSUM_PLACEHOLDER = "0" * 64

# MiLMMT's prompt requires these exact language names. Keep the display names
# synchronized with Xiaomi's model card and official demo.
LANGUAGES = {
    "Arabic": "ar",
    "Azerbaijani": "az",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Catalan": "ca",
    "Czech": "cs",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Persian": "fa",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kazakh": "kk",
    "Khmer": "km",
    "Korean": "ko",
    "Lao": "lo",
    "Malay": "ms",
    "Burmese": "my",
    "Norwegian": "no",
    "Dutch": "nl",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Swedish": "sv",
    "Tamil": "ta",
    "Thai": "th",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Cantonese": "yue",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-Hant",
}

LANGUAGE_CODES = set(LANGUAGES.values())
LANGUAGE_NAMES_BY_CODE = {code: name for name, code in LANGUAGES.items()}

# Accept codes already stored by other translators and language detectors. The
# canonical value returned to the rest of the app is always the MiLMMT code.
LANGUAGE_CODE_ALIASES = {
    "arb": "ar",
    "arb_arab": "ar",
    "ara": "ar",
    "azb_arab": "az",
    "aze": "az",
    "azj_latn": "az",
    "bul": "bg",
    "bul_cyrl": "bg",
    "ben": "bn",
    "ben_beng": "bn",
    "cat": "ca",
    "cat_latn": "ca",
    "ces": "cs",
    "ces_latn": "cs",
    "cze": "cs",
    "dan": "da",
    "dan_latn": "da",
    "deu": "de",
    "deu_latn": "de",
    "ger": "de",
    "ell": "el",
    "ell_grek": "el",
    "gre": "el",
    "eng": "en",
    "eng_latn": "en",
    "spa": "es",
    "spa_latn": "es",
    "fas": "fa",
    "per": "fa",
    "pes_arab": "fa",
    "fin": "fi",
    "fin_latn": "fi",
    "fra": "fr",
    "fra_latn": "fr",
    "fre": "fr",
    "heb": "he",
    "heb_hebr": "he",
    "hin": "hi",
    "hin_deva": "hi",
    "hrv": "hr",
    "hrv_latn": "hr",
    "hun": "hu",
    "hun_latn": "hu",
    "ind": "id",
    "ind_latn": "id",
    "ita": "it",
    "ita_latn": "it",
    "jpn": "ja",
    "jpn_jpan": "ja",
    "kaz": "kk",
    "kaz_cyrl": "kk",
    "khm": "km",
    "khm_khmr": "km",
    "kor": "ko",
    "kor_hang": "ko",
    "lao": "lo",
    "lao_laoo": "lo",
    "msa": "ms",
    "zsm_latn": "ms",
    "mya": "my",
    "mya_mymr": "my",
    "bur": "my",
    "nb": "no",
    "nn": "no",
    "nno": "no",
    "nno_latn": "no",
    "nob": "no",
    "nob_latn": "no",
    "nld": "nl",
    "nld_latn": "nl",
    "dut": "nl",
    "pol": "pl",
    "pol_latn": "pl",
    "por": "pt",
    "por_latn": "pt",
    "ron": "ro",
    "ron_latn": "ro",
    "rum": "ro",
    "rus": "ru",
    "rus_cyrl": "ru",
    "slk": "sk",
    "slk_latn": "sk",
    "slo": "sk",
    "slv": "sl",
    "slv_latn": "sl",
    "swe": "sv",
    "swe_latn": "sv",
    "tam": "ta",
    "tam_taml": "ta",
    "tha": "th",
    "tha_thai": "th",
    "fil": "tl",
    "tgl": "tl",
    "tgl_latn": "tl",
    "tur": "tr",
    "tur_latn": "tr",
    "urd": "ur",
    "urd_arab": "ur",
    "uzb": "uz",
    "uzn_latn": "uz",
    "vie": "vi",
    "vie_latn": "vi",
    "yue_hant": "yue",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zho_hans": "zh",
    "zh-tw": "zh-Hant",
    "zh_hant": "zh-Hant",
    "zho_hant": "zh-Hant",
}

# These archives are intentionally application-hosted. The source revisions
# are provenance for assembling each ZIP, never runtime download locations.
# MiLMMT is derived from Gemma and distributed under the Gemma terms; retain
# README.md in every hosted archive along with the runtime files below.
MODEL_LINKS = {
    # xiaomi-research/MiLMMT-46-1B-v1.0
    # revision 4fc480b6c58dec29c159dcdf9fde0f6d5c354995
    "MiLMMT-46-1B-v1.0": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/MiLMMT-46/MiLMMT-46-1B-v1.0.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/MiLMMT-46/MiLMMT-46-1B-v1.0.zip",
            "https://s3.libs.space:9000/ai-models/MiLMMT-46/MiLMMT-46-1B-v1.0.zip",
        ],
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "README.md": "b947eb325d6069558269bc82df9980e23883d618867c5abb98fb2333fea26810",
            "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
            "chat_template.jinja": "61fbedab465ef2a7f1b123fc7e4bc06a995bd09ac381579affeda60f02ebd940",
            "config.json": "2e9dcab9d9875f7acd33af240f4d0377ee22aa7e2849404dc5f0b36bca9bd027",
            "generation_config.json": "72e999861fcdc6c43485d04e12c47a78f4c6037db1234452e7226575662b72bb",
            "model.safetensors": "60d322f9b330b231a41f1d4fec4d6f3f5ed99c9631b4ea546478d1f335cc27c6",
            "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
            "tokenizer.json": "33753cc9825494361904313ed469063a8b3e05f1648c18e4b2936f5aa3c78202",
            "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
            "tokenizer_config.json": "76397efdd4e6c5dd3bb7994a8aa05c12484acae42dac33b7cadc1a38a8781643",
        },
        "path": "MiLMMT-46-1B-v1.0",
    },
    # xiaomi-research/MiLMMT-46-4B-v1.0
    # revision aa3262750cf493cc638fc9b82fcd26de8b0068fb
    "MiLMMT-46-4B-v1.0": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/MiLMMT-46/MiLMMT-46-4B-v1.0.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/MiLMMT-46/MiLMMT-46-4B-v1.0.zip",
            "https://s3.libs.space:9000/ai-models/MiLMMT-46/MiLMMT-46-4B-v1.0.zip",
        ],
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "README.md": "b10bf681d8dd0b01eb64c6d5ce01337a332135bc0009f4b6d66e312ba0170e9e",
            "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
            "chat_template.jinja": "61fbedab465ef2a7f1b123fc7e4bc06a995bd09ac381579affeda60f02ebd940",
            "config.json": "0c04f1335ef0e10f66d5b5dab51e76939000ad4b04914455ced9c2e0d5e061da",
            "generation_config.json": "72e999861fcdc6c43485d04e12c47a78f4c6037db1234452e7226575662b72bb",
            "model-00001-of-00002.safetensors": "0d93882d08d0b6d1bf410392f64247166d7bb3d2091e10827a78981fa964191a",
            "model-00002-of-00002.safetensors": "c54efb12b6f3ea4f1d3afa6ecc60ef7826d96bee92c1977b8419a945ecfd2bd8",
            "model.safetensors.index.json": "c20a286ce49c31383e1c33d8c7613342fb6d83a62f4074a0ee14906bea19c4eb",
            "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
            "tokenizer.json": "33753cc9825494361904313ed469063a8b3e05f1648c18e4b2936f5aa3c78202",
            "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
            "tokenizer_config.json": "513cefdf8eefbd9d1a39534875c471da170829f867b3c9b270e3e7ea9fc1c4db",
        },
        "path": "MiLMMT-46-4B-v1.0",
    },
    # xiaomi-research/MiLMMT-46-12B-v1.0
    # revision a27dbbb37142ff076990820a1c9f0827beb5d6ea
    "MiLMMT-46-12B-v1.0": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/MiLMMT-46/MiLMMT-46-12B-v1.0.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/MiLMMT-46/MiLMMT-46-12B-v1.0.zip",
            "https://s3.libs.space:9000/ai-models/MiLMMT-46/MiLMMT-46-12B-v1.0.zip",
        ],
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "README.md": "619351bad3f6e763c975f03e8cc2735cfd0c14829e5b70955bf97269989b5a11",
            "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
            "chat_template.jinja": "61fbedab465ef2a7f1b123fc7e4bc06a995bd09ac381579affeda60f02ebd940",
            "config.json": "95d9b498308a63bd6929fd6998fc51dc8d74ba2f530d3acdc06853f3a71386a1",
            "generation_config.json": "5332762cc865caf9917367cfddbfc9aa78acbf44bc3905d73d543b74ff6a303a",
            "model-00001-of-00005.safetensors": "e5ff2b59d221902eace3f596387a38c25d3de3ad7f99dfd70c39047e7fa31855",
            "model-00002-of-00005.safetensors": "b0361b79188a99faa44a10ef012941f75cdbc435bfac4d5dfa4a7127acad3d92",
            "model-00003-of-00005.safetensors": "f96bea8abe14f3301fefecf205dcd0ababc7bc113eebb1411ee64a8bc98dc40d",
            "model-00004-of-00005.safetensors": "3d148a44efc27d852f4cb9d568a7f3fbbee880b7f11dc3b59ecbf8df6d4368ca",
            "model-00005-of-00005.safetensors": "bd66b4aa89579ab18b52fcb2652fd8067e69ccbd78c0befebe6abcc0dd5bb314",
            "model.safetensors.index.json": "7bde3b089ae4e63cf690f222e50fab8dce4bc1d1e8f415b8ca1cd681c873fc1d",
            "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
            "tokenizer.json": "33753cc9825494361904313ed469063a8b3e05f1648c18e4b2936f5aa3c78202",
            "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
            "tokenizer_config.json": "513cefdf8eefbd9d1a39534875c471da170829f867b3c9b270e3e7ea9fc1c4db",
        },
        "path": "MiLMMT-46-12B-v1.0",
    },
}

MODEL_ALIASES = {
    "small": "MiLMMT-46-1B-v1.0",
    "medium": "MiLMMT-46-4B-v1.0",
    "large": "MiLMMT-46-12B-v1.0",
}

model = None
tokenizer = None
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_configuration = None
download_state = {"is_downloading": False}


def get_installed_language_names():
    return tuple(
        {"code": code, "name": language}
        for language, code in LANGUAGES.items()
    )


def _canonical_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def get_model_path(model_name):
    model_name = _canonical_model_name(model_name)
    if model_name == "custom":
        return MODEL_CACHE_PATH / "custom"
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown MiLMMT model: {model_name}")
    return MODEL_CACHE_PATH / MODEL_LINKS[model_name]["path"]


def needs_download(model_name):
    model_name = _canonical_model_name(model_name)
    if model_name == "custom":
        return False
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown MiLMMT model: {model_name}")
    return downloader.model_needs_download(
        get_model_path(model_name),
        MODEL_LINKS[model_name]["file_checksums"],
    )


def download_model(model_name, force_non_ui_dl=False):
    model_name = _canonical_model_name(model_name)
    if model_name == "custom":
        return get_model_path(model_name).is_dir()
    if model_name not in MODEL_LINKS:
        raise ValueError(f"Unknown MiLMMT model: {model_name}")
    if not needs_download(model_name):
        return True

    model_entry = MODEL_LINKS[model_name]
    if model_entry["checksum"] == ARCHIVE_CHECKSUM_PLACEHOLDER:
        archive_name = Path(model_entry["urls"][0]).name
        raise RuntimeError(
            f"The MiLMMT model archive {archive_name} is not currently "
            "available for automatic download."
        )

    return downloader.download_model(
        {
            "model_path": MODEL_CACHE_PATH,
            "model_link_dict": MODEL_LINKS,
            "model_name": model_name,
            "title": f"Text Translation (MiLMMT-46) - {model_name}",
            "alt_fallback": False,
            "force_non_ui_dl": force_non_ui_dl,
            "extract_format": "zip",
        },
        download_state,
    )


def _dtype_settings(compute_type):
    if compute_type == "float16":
        return {"dtype": torch.float16, "8bit": False}
    if compute_type == "bfloat16":
        return {"dtype": torch.bfloat16, "8bit": False}
    if compute_type == "float32":
        return {"dtype": torch.float32, "8bit": False}
    if compute_type == "8bit":
        return {"dtype": torch.float16, "8bit": True}
    raise ValueError(
        f"Unsupported MiLMMT precision '{compute_type}'. "
        "Use float32, float16, bfloat16, or 8bit."
    )


def _effective_compute_type(compute_type):
    # The v1.0 checkpoints emit an endless stream of padding tokens in FP16.
    # Keep old profiles working by migrating their request to a stable dtype.
    if compute_type != "float16":
        return compute_type
    if torch_device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def set_device(device):
    global torch_device

    device_name = str(device or "").lower()
    if device_name in {"", "none", "auto", "cuda"}:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("direct-ml"):
        raise ValueError("MiLMMT currently supports CUDA and CPU devices, but not DirectML.")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name != "cpu" and not device_name.startswith("cuda"):
        raise ValueError(f"Unsupported MiLMMT device: {device}")
    torch_device = torch.device(device_name)


def release_model():
    global model
    global tokenizer
    global loaded_configuration

    model = None
    tokenizer = None
    loaded_configuration = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def is_model_loaded():
    return model is not None and tokenizer is not None


def load_model(size=DEFAULT_MODEL, compute_type="float32"):
    global model
    global tokenizer
    global loaded_configuration

    os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
    model_name = _canonical_model_name(size)
    requested_compute_type = compute_type
    compute_type = _effective_compute_type(compute_type)
    if compute_type != requested_compute_type:
        print(
            f"MiLMMT does not produce valid output with {requested_compute_type}; "
            f"using {compute_type} instead."
        )
    dtype_settings = _dtype_settings(compute_type)
    if dtype_settings["8bit"] and torch_device.type != "cuda":
        raise ValueError("MiLMMT 8-bit inference requires CUDA.")

    load_configuration = (model_name, compute_type, str(torch_device))
    if model is not None and tokenizer is not None and loaded_configuration == load_configuration:
        return

    model_path = get_model_path(model_name)
    if model_name != "custom" and not download_model(model_name):
        raise RuntimeError(f"Could not download MiLMMT model '{model_name}'.")
    if not model_path.is_dir():
        raise FileNotFoundError(f"MiLMMT model directory does not exist: {model_path.resolve()}")

    release_model()
    quantization_config = None
    if dtype_settings["8bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    attention_type = get_preferred_attention_implementation(
        torch_device,
        dtype_settings["dtype"],
    )

    resolved_path = str(model_path.resolve())
    print(
        f"Loading MiLMMT model: {model_name} on {torch_device} "
        f"with {compute_type} precision using {attention_type}..."
    )
    def model_loader(attention_implementation):
        return AutoModelForCausalLM.from_pretrained(
            resolved_path,
            dtype=dtype_settings["dtype"],
            quantization_config=quantization_config,
            device_map=torch_device,
            attn_implementation=attention_implementation,
            local_files_only=True,
        )

    loaded_model, attention_type = load_with_attention_fallback(
        model_loader,
        attention_type,
        f"MiLMMT {model_name}",
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        local_files_only=True,
    )
    loaded_model.eval()
    model = loaded_model
    tokenizer = loaded_tokenizer
    loaded_configuration = load_configuration
    print(f"MiLMMT model loaded successfully using {attention_type}.")


def _generation_token_limit(input_length):
    # A normal translation is similar in length to its source. This generous
    # bound prevents a corrupt generation from running for 2,048 pad tokens.
    return min(2048, max(64, int(input_length) * 4))


def _resolve_language(language):
    if language is None:
        return None, None
    normalized = str(language).strip()
    if not normalized or normalized.casefold() in {"auto", "none", "null"}:
        return None, None

    for language_name, language_code in LANGUAGES.items():
        if normalized.casefold() == language_name.casefold():
            return language_code, language_name
        if normalized.casefold() == language_code.casefold():
            return language_code, language_name

    canonical_code = LANGUAGE_CODE_ALIASES.get(normalized.casefold())
    if canonical_code is not None:
        return canonical_code, LANGUAGE_NAMES_BY_CODE[canonical_code]
    raise ValueError(f"Unsupported MiLMMT language: {language}")


def _detect_source_language(text):
    detected_code, _ = languageClassification.classify(text)
    return _resolve_language(detected_code)


def build_prompt(source_language, target_language, text):
    return (
        f"Translate this from {source_language} to {target_language}:\n"
        f"{source_language}: {text}\n"
        f"{target_language}:"
    )


def translate_language(text, from_code, to_code, as_iso1=False):
    del as_iso1

    if text is None:
        text = ""
    text = str(text)
    try:
        source_code, source_language = _resolve_language(from_code)
        target_code, target_language = _resolve_language(to_code)
        if source_code is None and text.strip():
            source_code, source_language = _detect_source_language(text)
    except (TypeError, ValueError) as error:
        print(f"MiLMMT translation language error: {error}")
        return text, from_code, to_code

    if target_code is None:
        print("MiLMMT translation requires a target language.")
        return text, source_code or from_code, to_code
    if not text.strip() or source_code == target_code:
        return text, source_code or from_code, target_code
    if source_code is None:
        print("MiLMMT could not detect a supported source language.")
        return text, from_code, target_code
    if model is None or tokenizer is None:
        raise RuntimeError("MiLMMT model is not loaded.")

    prompt = build_prompt(source_language, target_language, text)
    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=_generation_token_limit(input_length),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    translation_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()
    if not translation_text:
        precision = loaded_configuration[1] if loaded_configuration else "unknown"
        raise RuntimeError(
            f"MiLMMT generated an empty translation using {precision} precision."
        )
    return translation_text, source_code, target_code
