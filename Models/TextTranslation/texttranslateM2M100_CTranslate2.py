import ctranslate2
import sentencepiece as spm
import os
import downloader
from pathlib import Path

LANGUAGES = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Breton": "br",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greeek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Fulah": "ff",
    "Finnish": "fi",
    "French": "fr",
    "Western Frisian": "fy",
    "Irish": "ga",
    "Gaelic": "gd",
    "Galician": "gl",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Haitian": "ht",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Igbo": "ig",
    "Iloko": "ilo",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Central Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Luxembourgish": "lb",
    "Ganda": "lg",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malagasy": "mg",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Northern Sotho": "ns",
    "Occitan (post 1500)": "oc",
    "Oriya": "or",
    "Panjabi": "pa",
    "Polish": "pl",
    "Pushto": "ps",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swati": "ss",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Thai": "th",
    "Tagalog": "tl",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Chinese": "zh",
    "Zulu": "zu"
}

# Download CTranslate2 models:
# • M2M-100 418M-parameter model: https://bit.ly/33fM1AO
# • M2M-100 1.2B-parameter model: https://bit.ly/3GYiaed
MODEL_LINKS = {
    "small": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/M2M100_ctranslate2/m2m100_ct2_418m.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/M2M100_ctranslate2/m2m100_ct2_418m.zip",
            "https://s3.libs.space:9000/ai-models/M2M100_ctranslate2/m2m100_ct2_418m.zip",
        ],
        "checksum": "aa2dc13ad93664021a5f3c16bf91028bd1145ccad622668a99292b11d832ebc6"
    },
    "large": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/M2M100_ctranslate2/m2m100_ct2_12b.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/M2M100_ctranslate2/m2m100_ct2_12b.zip",
            "https://s3.libs.space:9000/ai-models/M2M100_ctranslate2/m2m100_ct2_12b.zip",
        ],
        "checksum": "67657e00b42f1a12ce0b85e377212eae6747057c735d6edb8fe920d5eb583deb"
    }
}

# [Modify] Set the device and beam size
device = "auto"  # "cpu" or "cuda" for GPU, auto = automatic
beam_size = 5

# [Modify] Set paths to the CTranslate2 and SentencePiece models
ct_model_path = Path(Path.cwd() / ".cache" / "m2m100_ct2")
os.makedirs(ct_model_path, exist_ok=True)

# default small model path. (should be loaded using load_model function)
model_path = Path(ct_model_path / "m2m100_418m")

model = spm.SentencePieceProcessor()


def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGES.items()])


def load_model(size="small"):
    global model_path
    match size:
        case "small":
            model_file = "m2m100_418m"

        case "large":
            model_file = "m2m100_12b"

        case _:
            model_file = "m2m100_418m"

    model_path = Path(ct_model_path / model_file)
    sp_model_path = Path(ct_model_path / model_file / "sentencepiece.model")

    if not sp_model_path.exists():
        print(f"Downloading {size} text translation model...")
        downloader.download_extract(MODEL_LINKS[size]["urls"], str(ct_model_path.resolve()), MODEL_LINKS[size]["checksum"])

    model.load(str(sp_model_path.resolve()))


def set_device(option):
    global device
    device = option


def translate_language(text, from_code, to_code):
    src_prefix = "__" + from_code + "__"
    tgt_prefix = "__" + to_code + "__"
    target_prefix = [[tgt_prefix]] * len(text)

    # Subword the source sentences
    source_sents_subworded = model.encode(text, out_type="str")
    source_sents_subworded = [[src_prefix] + source_sents_subworded]

    # Translate the source sentences
    translator = ctranslate2.Translator(str(model_path.resolve()), device=device)
    translations = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=2024, beam_size=beam_size, target_prefix=target_prefix, normalize_scores=True, max_input_length=2048)
    translations = [translation[0]['tokens'] for translation in translations]

    # Desubword the target sentences
    translations_desubword = model.decode(translations)
    translations_desubword = [sent[len(tgt_prefix):] for sent in translations_desubword]

    return ' '.join(translations_desubword)
