# noinspection PyPackageRequirements
import fasttext
import os
import downloader
from pathlib import Path

ct_model_path = Path(Path.cwd() / ".cache" / "lid")
os.makedirs(ct_model_path, exist_ok=True)

MODEL_LINKS = {
    "lid218e": "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/LID%2Flid218e.zip"
}


def classify(text):
    pretrained_lang_model_file = Path(ct_model_path / "lid218e.bin")
    if not pretrained_lang_model_file.exists():
        print(f"Downloading LID (language identification) model...")
        downloader.download_extract(MODEL_LINKS["lid218e"], str(ct_model_path.resolve()))

    model = fasttext.load_model(str(pretrained_lang_model_file.resolve()))
    predictions = model.predict(text, k=1)

    return predictions[0][0].replace('__label__', '')
