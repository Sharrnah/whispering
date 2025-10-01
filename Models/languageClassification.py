# noinspection PyPackageRequirements
import fasttext
import os
import downloader
from pathlib import Path

import websocket

from Utilities.iso_converter import LanguageCodeConverter
from Models.TextTranslation.texttranslateNLLB200_CTranslate2 import LANGUAGES

ct_model_path = Path(Path.cwd() / ".cache" / "lid")
os.makedirs(ct_model_path, exist_ok=True)

MODEL_LINKS = {
    "lid218e": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/LID/LID_lid218e.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/LID/LID_lid218e.zip",
            "https://s3.libs.space:9000/ai-models/LID/LID_lid218e.zip",
        ],
        "checksum": "0d61e4ec75de7c08f17395d93a612a9e7835c137e0cc6cb60213babd6916432c",
    }
}

model = None

code_converter = LanguageCodeConverter()

def download_model():
    pretrained_lang_model_file = Path(ct_model_path / "lid218e.bin")
    if not pretrained_lang_model_file.is_file():
        print(f"Downloading LID (language identification) model...")
        downloader.download_extract(MODEL_LINKS["lid218e"]["urls"], str(ct_model_path.resolve()), MODEL_LINKS["lid218e"]["checksum"], "language identification")


def classify(text, to_code='nllb'):
    global model
    pretrained_lang_model_file = Path(ct_model_path / "lid218e.bin")
    if not pretrained_lang_model_file.is_file():
        print(f"Error: LID (language identification model missing.")
        return ""

    if model is None:
        websocket.set_loading_state("lid_loading", True)
        print(f"Loading LID (language identification) model...")
        model = fasttext.load_model(str(pretrained_lang_model_file.resolve()))
        websocket.set_loading_state("lid_loading", False)
        print(f"LID loaded.")

    text = text.replace("\n", " ")
    predictions = model.predict(text, k=1)

    predicted_label = predictions[0][0]
    predicted_label = predicted_label.replace('__label__', '')
    predicted_probability = predictions[1][0]

    # convert to other language code if needed
    if to_code != 'nllb':
        predicted_label = code_converter.convert(predicted_label, to_code)

    return predicted_label, predicted_probability
