from windowcapture import WindowCapture
import easyocr
from easyocr import config
import os
from pathlib import Path

model_path = Path(Path.cwd() / ".cache" / ".EasyOCR")
os.makedirs(model_path, exist_ok=True)

CURRENT_LANGUAGES = ['en']

LANGUAGES = config.all_lang_list

# list from https://www.jaided.ai/easyocr
LANGUAGE_CODES = {
    "Abaza": "abq",
    "Adyghe": "ady",
    "Afrikaans": "af",
    "Angika": "ang",
    "Arabic": "ar",
    "Assamese": "as",
    "Avar": "ava",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bihari": "bh",
    "Bhojpuri": "bho",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Simplified Chinese": "ch_sim",
    "Traditional Chinese": "ch_tra",
    "Chechen": "che",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "Dargwa": "dar",
    "German": "de",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian (Farsi)": "fa",
    "French": "fr",
    "Irish": "ga",
    "Goan Konkani": "gom",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Ingush": "inh",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Kabardian": "kbd",
    "Kannada": "kn",
    "Korean": "ko",
    "Kurdish": "ku",
    "Latin": "la",
    "Lak": "lbe",
    "Lezghian": "lez",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Magahi": "mah",
    "Maithili": "mai",
    "Maori": "mi",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Maltese": "mt",
    "Nepali": "ne",
    "Newari": "new",
    "Dutch": "nl",
    "Norwegian": "no",
    "Occitan": "oc",
    "Pali": "pi",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian (cyrillic)": "rs_cyrillic",
    "Serbian (latin)": "rs_latin",
    "Nagpuri": "sck",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Albanian": "sq",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Tabassaran": "tab",
    "Telugu": "te",
    "Thai": "th",
    "Tajik": "tjk",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Uyghur": "ug",
    "Ukranian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi"
}

reader: easyocr.Reader = None  # type: ignore


def init_reader(languages):
    global reader, CURRENT_LANGUAGES

    if reader is None or CURRENT_LANGUAGES != languages:
        CURRENT_LANGUAGES = languages
        reader = easyocr.Reader(CURRENT_LANGUAGES, model_storage_directory=str(model_path.resolve()))


def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGE_CODES.items()])


def initialize_window_capture(window_name):
    win_cap = WindowCapture(window_name)
    return win_cap


def run_image_processing(window_name, src_languages):
    init_reader(src_languages)

    win_cap = initialize_window_capture(window_name)

    # get an updated image of the game
    screenshot = win_cap.get_screenshot_mss()

    result = reader.readtext(screenshot, detail=0, paragraph=True)

    return result
