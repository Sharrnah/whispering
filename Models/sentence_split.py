import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List

nltk_path = Path(Path.cwd() / ".cache" / "nltk")
os.makedirs(nltk_path, exist_ok=True)
os.environ["NLTK_DATA"] = str(nltk_path.resolve())
import nltk

# from nltk.tokenize import RegexpTokenizer

# List from https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt.xml
NLTK_LANGUAGE_CODES_GROUPED = {
    ("czech", "ces_latn", "ces", "cs",): "Czech",
    ("danish", "dan_latn", "dan", "da",): "Danish",
    ("dutch", "nld_latn", "nld", "nl",): "Dutch",
    ("english", "eng_latn", "eng", "en",): "English",
    ("estonian", "est_latn", "est", "et",): "Estonian",
    ("finnish", "fin_latn", "fin", "fi",): "Finnish",
    ("french", "fra_latn", "fra", "fr",): "French",
    ("german", "deu_latn", "deu", "de",): "German",
    ("greek", "ell_grek", "ell", "el",): "Greek",
    ("italian", "ita_latn", "ita", "it",): "Italian",
    ("malayalam", "mal_mlym", "mal", "ml",): "Malayalam",
    ("norwegian", "nno_latn", "nno", "nob_Latn", "nob", "no",): "Norwegian",
    ("polish", "pol_latn", "pol", "pl",): "Polish",
    ("portuguese", "por_latn", "por", "pt",): "Portuguese",
    ("russian", "rus_cyrl", "rus", "ru",): "Russian",
    ("slovene", "slv_latn", "slv", "sl",): "Slovene",
    ("spanish", "spa_latn", "spa", "es",): "Spanish",
    ("swedish", "swe_latn", "swe", "sv",): "Swedish",
    ("turkish", "tur_latn", "tur", "tr",): "Turkish",
    # Custom languages (see custom_split_text() function)
    #("japanese", "jpn_jpan", "jpn", "ja",): "Japanese",
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
        if not Path(nltk_path / "tokenizers" / "punkt").is_dir() or not Path(
                nltk_path / "tokenizers" / "punkt" / "english.pickle").is_file():
            nltk.download('punkt', download_dir=str(nltk_path.resolve()))


def custom_split_text(input_text, language='japanese'):
    match language:
        case "Japanese":
            # custom nltk tokenizer
            tokenizer = nltk.RegexpTokenizer(r'[^!?。\n]+[!?。\n]?')
            tokenized_sentences = tokenizer.tokenize(input_text)
            return tokenized_sentences

        case _:
            return None


def split_text(text, language='english') -> List[str]:
    load_model()

    nltk_sentence_split_lang = get_nltk_language_code(language)

    #custom_split = custom_split_text(text, nltk_sentence_split_lang)
    #if custom_split is not None:
    #    print("returning custom split text")
    #    return custom_split

    # Split the source text into sentences
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

    return ' '.join(cleaned_parts)
