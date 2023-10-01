import os
import sys
from pathlib import Path
from collections import defaultdict

nltk_path = Path(Path.cwd() / ".cache" / "nltk")
os.makedirs(nltk_path, exist_ok=True)
os.environ["NLTK_DATA"] = str(nltk_path.resolve())
import nltk

# List from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.xml
NLTK_LANGUAGE_CODES_GROUPED = {
    ("czech", "ces_Latn", "ces", "cs",): "Czech",
    ("danish", "dan_Latn", "dan", "da",): "Danish",
    ("dutch", "nld_Latn", "nld", "nl",): "Dutch",
    ("english", "eng_Latn", "eng", "en",): "English",
    ("estonian", "est_Latn", "est", "et",): "Estonian",
    ("finnish", "fin_Latn", "fin", "fi",): "Finnish",
    ("french", "fra_Latn", "fra", "fr",): "French",
    ("german", "deu_Latn", "deu", "de",): "German",
    ("greek", "ell_Grek", "ell", "el",): "Greek",
    ("italian", "ita_Latn", "ita", "it",): "Italian",
    ("malayalam", "mal_Mlym", "mal", "ml",): "Malayalam",
    ("norwegian", "nno_Latn", "nno", "nob_Latn", "nob", "no",): "Norwegian",
    ("polish", "pol_Latn", "pol", "pl",): "Polish",
    ("portuguese", "por_Latn", "por", "pt",): "Portuguese",
    ("russian", "rus_Cyrl", "rus", "ru",): "Russian",
    ("slovene", "slv_Latn", "slv", "sl",): "Slovene",
    ("spanish", "spa_Latn", "spa", "es",): "Spanish",
    ("swedish", "swe_Latn", "swe", "sv",): "Swedish",
    ("turkish", "tur_Latn", "tur", "tr",): "Turkish",
}

# Create a flat dictionary for easier lookup
NLTK_LANGUAGE_CODES = {key: value for keys, value in NLTK_LANGUAGE_CODES_GROUPED.items() for key in keys}


def get_nltk_language_code(input_code: str):
    # Use input_code as is if it exists in the dictionary
    return NLTK_LANGUAGE_CODES.get(input_code.lower(), "English")


def load_model():
    # only if not running as pyinstaller bundle (pyinstaller places tokenizer folder in distribution "nlpk_data")
    if not getattr(sys, 'frozen', False) and not hasattr(sys, '_MEIPASS'):
        # load nltk sentence splitting dependency
        if not Path(nltk_path / "tokenizers" / "punkt").is_dir() or not Path(nltk_path / "tokenizers" / "punkt" / "english.pickle").is_file():
            nltk.download('punkt', download_dir=str(nltk_path.resolve()))


def split_text(text, language='english'):
    load_model()

    # Split the source text into sentences
    nltk_sentence_split_lang = get_nltk_language_code(language)
    return nltk.tokenize.sent_tokenize(text, language=nltk_sentence_split_lang)


def remove_repeated_sentences(text, language='english', max_repeat=1, additional_split_chars=None):
    load_model()

    sentence_count = defaultdict(int)
    cleaned_parts = []

    sentences = split_text(text, language=language)

    for sentence in sentences:
        if additional_split_chars:
            for char in additional_split_chars:
                phrases = sentence.split(char)
                for i, phrase in enumerate(phrases):
                    phrase = phrase.strip()
                    sentence_count[phrase] += 1
                    if sentence_count[phrase] <= max_repeat:
                        cleaned_parts.append(phrase)
                        if i < len(phrases) - 1:
                            cleaned_parts.append(char)
        else:
            sentence_count[sentence.strip()] += 1
            if sentence_count[sentence.strip()] <= max_repeat:
                cleaned_parts.append(sentence)

    return ''.join(cleaned_parts)
