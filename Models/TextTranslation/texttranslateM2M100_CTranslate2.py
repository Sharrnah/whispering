import sys

import ctranslate2
import sentencepiece as spm
import os
import downloader
from pathlib import Path
import torch
from Models import sentence_split

nltk_path = Path(Path.cwd() / ".cache" / "nltk")
os.makedirs(nltk_path, exist_ok=True)
os.environ["NLTK_DATA"] = str(nltk_path.resolve())
import nltk

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


# List from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.xml
NLTK_LANGUAGE_CODES = {
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "ell_Grek": "Greek",  # ??
    "it": "Italian",
    "ml": "Malayalam",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "slv_Latn": "Slovene",  # ??
    "es": "Spanish",
    "sv": "Swedish",
    "tr": "Turkish",
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
torch_device = "cuda" if torch.cuda.is_available() else "cpu"  # "cpu" or "cuda" for GPU, auto = automatic
beam_size = 5

# [Modify] Set paths to the models
ct_model_path = Path(Path.cwd() / ".cache" / "m2m100_ct2")
os.makedirs(ct_model_path, exist_ok=True)

# default small model path. (should be loaded using load_model function)
model_path = Path(ct_model_path / "m2m100_418m")

sentencepiece = spm.SentencePieceProcessor()

translator = None


def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGES.items()])


def load_model(size="small", compute_type="float32"):
    global model_path
    global sentencepiece
    global translator

    match size:
        case "small":
            model_file = "m2m100_418m"

        case "large":
            model_file = "m2m100_12b"

        case _:
            model_file = "m2m100_418m"

    model_path = Path(ct_model_path / model_file)
    sp_model_path = Path(ct_model_path / model_file / "sentencepiece.model")

    print(f"M2M100_CTranslate2 {size} is Loading to {torch_device} using {compute_type} precision...")

    if not sp_model_path.exists() or not Path(ct_model_path / model_file / "model.bin").exists():
        print(f"Downloading {size} text translation model...")
        downloader.download_extract(MODEL_LINKS[size]["urls"], str(ct_model_path.resolve()),
                                    MODEL_LINKS[size]["checksum"], title="M2M100CT2")

    sentencepiece.load(str(sp_model_path.resolve()))

    # only if not running as pyinstaller bundle (pyinstaller places tokenizer folder in distribution "nlpk_data")
    if not getattr(sys, 'frozen', False) and not hasattr(sys, '_MEIPASS'):
        # load nltk sentence splitting dependency
        if not Path(nltk_path / "tokenizers" / "punkt").is_dir() or not Path(nltk_path / "tokenizers" / "punkt" / "english.pickle").is_file():
            nltk.download('punkt', download_dir=str(nltk_path.resolve()))

    translator = ctranslate2.Translator(str(model_path.resolve()), device=torch_device, compute_type="float32")

    print(f"M2M100_CTranslate2 model loaded.")


def set_device(device: str):
    global torch_device
    if device == "cuda" or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = device


def translate_language(text, from_code, to_code):
    global sentencepiece

    src_prefix = "__" + from_code + "__"
    tgt_prefix = "__" + to_code + "__"
    target_prefix = [[tgt_prefix]] * len(text)

    # Split the source text into sentences
    sentences = sentence_split.split_text(text, language=from_code)
    translated_sentences = []

    for sentence in sentences:
        # Subword the source sentences
        source_sents_subworded = sentencepiece.encode(sentence, out_type=str)
        source_sents_subworded = [[src_prefix] + source_sents_subworded]

        # Translate the source sentences
        translations = translator.translate_batch(source_sents_subworded, target_prefix=target_prefix, batch_type="tokens",
                                                  max_batch_size=2024, beam_size=beam_size, max_input_length=2048)
        translations = [translation[0]['tokens'] for translation in translations]

        # Desubword the target sentences
        translations_desubword = sentencepiece.decode(translations)
        translations_desubword = [sent[len(tgt_prefix):] for sent in translations_desubword]

        translated_sentences.append(' '.join(translations_desubword))

    return ' '.join(translated_sentences)
