# noinspection PyPackageRequirements
import fasttext
import os
import downloader
from pathlib import Path

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


def classify(text):
    pretrained_lang_model_file = Path(ct_model_path / "lid218e.bin")
    if not pretrained_lang_model_file.exists():
        print(f"Downloading LID (language identification) model...")
        downloader.download_extract(MODEL_LINKS["lid218e"]["urls"], str(ct_model_path.resolve()), MODEL_LINKS["lid218e"]["checksum"])

    model = fasttext.load_model(str(pretrained_lang_model_file.resolve()))

    text = text.replace("\n", " ")
    predictions = model.predict(text, k=1)

    return predictions[0][0].replace('__label__', '')
