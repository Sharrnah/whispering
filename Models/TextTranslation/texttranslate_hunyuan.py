import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import downloader
from pathlib import Path

# https://huggingface.co/spaces/SkynetM1/HY-MT1.5-1.8B/tree/main

LANGUAGES = {
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "Portuguese": "pt",
    "Spanish": "es",
    "Japanese": "ja",
    "Turkish": "tr",
    "Russian": "ru",
    "Arabic": "ar",
    "Korean": "ko",
    "Thai": "th",
    "Italian": "it",
    "German": "de",
    "Vietnamese": "vi",
    "Malay": "ms",
    "Indonesian": "id",
    "Filipino": "tl",
    "Hindi": "hi",
    "Traditional Chinese": "zh-Hant",
    "Polish": "pl",
    "Czech": "cs",
    "Dutch": "nl",
    "Khmer": "km",
    "Burmese": "my",
    "Persian": "fa",
    "Gujarati": "gu",
    "Urdu": "ur",
    "Telugu": "te",
    "Marathi": "mr",
    "Hebrew": "he",
    "Bengali": "bn",
    "Tamil": "ta",
    "Ukrainian": "uk",
    "Tibetan": "bo",
    "Kazakh": "kk",
    "Mongolian": "mn",
    "Uyghur": "ug",
    "Cantonese": "yue"
}

language_codes = list(LANGUAGES.values())

MODEL_LINKS = {
    "small": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/HY-MT1.5/small.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/HY-MT1.5/small.zip",
            "https://s3.libs.space:9000/ai-models/HY-MT1.5/small.zip",
        ],
        "checksum": "e6046a96a25a35e41c4c4b71f9e46a463f98fd71dcd32083054cee0bcf1fe363",
        "file_checksums": {
            "chat_template.jinja": "b7491ec0e9c869dfce20f2176758099bf248d979dd05530ede99deb21698acee",
            "config.json": "6d364cd4d0967604ccf91b4bddf3af966d8348cbfb999068f3ff80cdf8495fa7",
            "generation_config.json": "726f6353ca88628275c13ecab6e7f337c8573a9aa70776eae0bbd3eed6888035",
            "hf_quant_config.json": "53ee00e0d886e5d4ff8b67aac0a7a96197ecad61774c04c3263c6d69dd596e7c",
            "model.safetensors": "d318a9df24c583666fbb910c0d461edcc42847b9b1a27e06bd709c9a95eaf75c",
            "special_tokens_map.json": "bb9f59990034dae326581b9c62471523975417869f78a244b7ae2ce8cbb085eb",
            "tokenizer.json": "3c4fb9a1848b935921efa9c5fc2ca8e89ba5db36fbaf601893882e5546dae551",
            "tokenizer_config.json": "53bd8581b601a8ee9caefeb988207de50b3fc0b733295bdf5ad68dec4cc0b07c"
        },
        "path": "small",
    },
    "medium": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/HY-MT1.5/medium.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/HY-MT1.5/medium.zip",
            "https://s3.libs.space:9000/ai-models/HY-MT1.5/medium.zip",
        ],
        "checksum": "7bb7762d5eaa221d0a5605dadab5f15ef3651c811001e385d2aa4be62dd3d9fc",
        "file_checksums": {
            "angelslim_config.json": "9aa1b6edfeb2cd42c00a35f37cdf72da99b54261fa27efed59352d2c8c7ebeb4",
            "chat_template.jinja": "b7491ec0e9c869dfce20f2176758099bf248d979dd05530ede99deb21698acee",
            "config.json": "a1788df3224420f43ed1a424ad58bfacc34f689b0e477ce69d1298fa6d26292b",
            "generation_config.json": "3586ba4829d9769b89523523cb562f2e894c519274f8a0e9b970287a0b1388a9",
            "model.safetensors": "07736f560253d8c991616060fb2d855420957c268fa7d32fa8593df2f83b21ab",
            "special_tokens_map.json": "bb9f59990034dae326581b9c62471523975417869f78a244b7ae2ce8cbb085eb",
            "tokenizer.json": "b475bbef1b0b2fd57dcb865332b546475bd1ede2deb3bb91bafd0c047a8a530a",
            "tokenizer_config.json": "53bd8581b601a8ee9caefeb988207de50b3fc0b733295bdf5ad68dec4cc0b07c"
        },
        "path": "medium",
    },
    "large": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/HY-MT1.5/large.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/HY-MT1.5/large.zip",
            "https://s3.libs.space:9000/ai-models/HY-MT1.5/large.zip",
        ],
        "checksum": "62c06f3397c7e03c7213b91b36a83c5cf1acc1c6cc6f6738edb7263c2bcafa78",
        "file_checksums": {
            "chat_template.jinja": "788ac16c5d7bfefc28655928ad524c8f378a44cb24d24fb125d6a5859b167677",
            "config.json": "946636f2ac831e4e142d82f285e770bcf8cd4e8f2c9acf3b5309ca7f672b6cef",
            "generation_config.json": "f786ed6734246caf78549360455c8bd56f6a8dae891a8e276235edb4a02bef47",
            "hf_quant_config.json": "53ee00e0d886e5d4ff8b67aac0a7a96197ecad61774c04c3263c6d69dd596e7c",
            "model-00001-of-00002.safetensors": "d734a70b9475d61be105fd129eaca0f940d87b27b0b837e3c815b17fc2889545",
            "model-00002-of-00002.safetensors": "b1175e5d8bfb303af2557b836474b551f973cd8d8f000a64acb609af23d06553",
            "model.safetensors.index.json": "9197bc9890101ed8c637840563c3da66c5bd9d90a6b8a52e3f1c604839716632",
            "special_tokens_map.json": "c6571f2ee67c36fa93d7f688ee6fd43f5b84784e895f3e6a4eb8c2657379758e",
            "tokenizer.json": "a07abfc4611977f262d1789910d53ee76af28620f285af80cc4051a8db09b509",
            "tokenizer_config.json": "1fffdb809549222938f4a732645dd984c7c0b2973d592c2d01e05425274317da"
        },
        "path": "large",
    }
}

# Set paths to the models
cache_path = Path(Path.cwd() / ".cache" / "hunyuan_mt15")
os.makedirs(cache_path, exist_ok=True)

model = AutoModelForCausalLM
tokenizer = AutoTokenizer

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

download_state = {"is_downloading": False}

def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGES.items()])


def set_device(device: str):
    global torch_device
    if device == "cuda" or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = device


def load_model(size="small", compute_type="float32"):
    global model
    global tokenizer
    global torch_device

    model_path = Path(cache_path / size)

    print(f"HY-MT1.5 {size} is Loading to {torch_device} using {compute_type} precision...")

    downloader.download_model({
        "model_path": cache_path,
        "model_link_dict": MODEL_LINKS,
        "model_name": size,
        "title": "Text Translation (HY-MT1.5)",

        "alt_fallback": False,
        "force_non_ui_dl": False,
        "extract_format": "zip",
    }, download_state)

    model_path_string = str(model_path.resolve())

    torch_dtype = torch.float16 if compute_type == "float16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path_string, dtype=torch_dtype).to(torch_device)

    tokenizer = AutoTokenizer.from_pretrained(model_path_string, token=True, return_tensors="pt")

    print(f"HY-MT1.5 model loaded.")


def translate_language(text, from_code, to_code, as_iso1=False):
    global model
    global tokenizer
    global torch_device

    language_unsupported = False
    if to_code not in language_codes:
        print(f"error translating. {to_code} not supported.")
        language_unsupported = True
    if language_unsupported:
        print(f"Language {to_code} not supported.")
        return text, from_code, to_code

    # find key by value in LANGUAGES dict
    to_language = next((language for language, code in LANGUAGES.items() if code == to_code), to_code)

    messages = [
        {
            "role": "user",
            "content": f"Translate the following segment into {to_language}, without additional explanation.\n\n{text}"
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = tokenized_chat["input_ids"].to(model.device)
    input_length = input_ids.shape[1]

    outputs = model.generate(input_ids, max_new_tokens=2048)
    generated_tokens = outputs[0][input_length:]

    translation_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return translation_text, from_code, to_code
