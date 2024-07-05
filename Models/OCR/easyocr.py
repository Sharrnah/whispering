import numpy
import requests

import websocket
from windowcapture import WindowCapture
import easyocr
from easyocr import config
import os
from pathlib import Path

from PIL import Image

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
        websocket.set_loading_state("ocr_loading", True)
        CURRENT_LANGUAGES = languages
        try:
            # disable verbose logging (which also disables the progress bar) to prevent charset error: see https://github.com/JaidedAI/EasyOCR/issues/1017
            reader = easyocr.Reader(CURRENT_LANGUAGES, model_storage_directory=str(model_path.resolve()), detect_network="craft", verbose=False)
        except Exception as e:
            print(str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
            websocket.set_loading_state("ocr_loading", False)
            return False
        websocket.set_loading_state("ocr_loading", False)


def get_installed_language_names():
    return tuple([{"code": code, "name": language} for language, code in LANGUAGE_CODES.items()])


def initialize_window_capture(window_name):
    win_cap = WindowCapture(window_name)
    return win_cap


def convert_bounding_box(coords):
    # Extract the minimum and maximum x and y coordinates
    min_x = min(coords, key=lambda x: x[0])[0]
    min_y = min(coords, key=lambda x: x[1])[1]
    max_x = max(coords, key=lambda x: x[0])[0]
    max_y = max(coords, key=lambda x: x[1])[1]

    # Calculate the width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Return the absolute pixel coordinates of the bounding box
    return min_x, min_y, min_x + width, min_y + height


def run_image_processing(window_name, src_languages):
    init_reader(src_languages)
    screenshot_png = None
    result_lines = []
    bounding_boxes = []
    if reader is not None:
        try:
            win_cap = initialize_window_capture(window_name)

            # get an updated image of the game
            screenshot, screenshot_png = win_cap.get_screenshot_mss()
            if screenshot is None:
                return None, None, None

            # unitialize
            win_cap.unitialize()

            result_lines, _, bounding_boxes = run_image_processing_from_image(screenshot, src_languages)

        except Exception as e:
            print(e)

    return result_lines, screenshot_png, bounding_boxes


def run_image_processing_from_image(image_src, src_languages):
    image_pth = image_src
    image = None
    if isinstance(image_src, str) and image_src.startswith("http"):
        print("fetching image url...")
        image_pth = requests.get(image_src, stream=True).raw
    elif hasattr(image_src, "file"):
        print("getting image from file...")
        image_pth = image_src.file
    try:
        print("converting image...")
        image = Image.open(image_pth).convert('RGB')
    except Exception as e:
        if not isinstance(image_pth, numpy.ndarray) and not isinstance(image_pth, bytes):
            print("failed to convert image: " + str(e))

    if image is None:
        image = image_src

    print("OCR Started...")

    init_reader(src_languages)
    result_lines = []
    bounding_boxes = []
    if reader is not None:
        try:
            result_data = reader.readtext(image, paragraph=True)
            if len(result_data) > 0:
                for line in result_data:
                    # bbox to pixel value bbox
                    bbox_pixels = convert_bounding_box(line[0])

                    # bounding_box = line[0]
                    bounding_box = bbox_pixels
                    text_detection = line[1]
                    result_lines.append(text_detection)
                    bounding_boxes.append(bounding_box)

        except Exception as e:
            print(e)

    print("OCR Finished.")

    return result_lines, image, bounding_boxes
