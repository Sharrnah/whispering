import io

import numpy
import requests
import torch
from attr.validators import instance_of

import downloader
import websocket
from Models.Singleton import SingletonMeta
import os
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

from PIL import Image

model_path = Path(Path.cwd() / ".cache" / ".GOT-OCR-2.0")
os.makedirs(model_path, exist_ok=True)


MODEL_LINKS = {
    "GOT_OCR_2.0": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/GOT_OCR_2.0/GOT-OCR-2.0.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/GOT_OCR_2.0/GOT-OCR-2.0.zip",
            "https://s3.libs.space:9000/ai-models/GOT_OCR_2.0/GOT-OCR-2.0.zip",
        ],
        "checksum": "d98db661dd7d76943807b316685d9561b4cf85403fee1ba749fb691e038a587b",
        "file_checksums": {
            "config.json": "cbe8aacd6cd84a2d58eafcd0045c6ac40e02e3a448f24b8cee51cc81d8bdccf2",
            "generation_config.json": "31915c5a692f43c5765a20cfc5f9403bcd250f5721a0d931bb703169c08993b4",
            "model.safetensors": "6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4",
            "preprocessor_config.json": "ef9a0dc0935cac11f4230ca30d00a52bedfa52b6633e409e9fbd2ea56373aa7e",
            "special_tokens_map.json": "7c2368a3889fdfb37c24cabeb031b53f47934f357b54e56e8e389909a338ea47",
            "tokenizer.json": "36b382a3c48c9a143c30139dac6c8230ddfb0b46a3dc43082af6052abe99d9de",
            "tokenizer_config.json": "8b0542937d32a67da8ea2d1288b870e325be383a962c65d201864299560a2b8e"
        },
        "path": "",
    },
}


class Got_ocr_20(metaclass=SingletonMeta):
    model = None
    processor = None
    LANGUAGE_CODES = {
        "Auto": "",
    }
    device = "cpu"
    torch_dtype = torch.float32

    download_state = {"is_downloading": False}

    def __init__(self, device=""):
        self.set_compute_device(device)

    def set_compute_device(self, device):
        if device != "" and device is not None and device != "auto":
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch_dtype = torch.float32
        if self.device == "cuda" or self.device.startswith("cuda:"):
            self.torch_dtype = torch.bfloat16

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": model_path,
            "model_link_dict": MODEL_LINKS,
            "model_name": model_name,
            "title": "Image-to-Text (GOT OCR 2.0)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    def init_reader(self, languages):

        try:
            self.download_model("GOT_OCR_2.0")

            websocket.set_loading_state("ocr_loading", True)
            self.model = AutoModelForImageTextToText.from_pretrained(str(model_path.resolve()), dtype=self.torch_dtype, device_map=self.device)
            self.processor = AutoProcessor.from_pretrained(str(model_path.resolve()))
            websocket.set_loading_state("ocr_loading", False)
        except Exception as e:
            print(str(e).encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
            websocket.set_loading_state("ocr_loading", False)
            return False

    def get_installed_language_names(self):
        return tuple([{"code": code, "name": language} for language, code in self.LANGUAGE_CODES.items()])

    def convert_bounding_box(self, coords):
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

    def run_image_processing_from_image(self, image_src, src_languages):
        image_pth = image_src
        image = None
        if isinstance(image_src, str) and image_src.startswith("http"):
            print("fetching image url...")
            image_pth = requests.get(image_src, stream=True).raw
        elif hasattr(image_src, "file"):
            print("getting image from file...")
            image_pth = image_src.file

        if isinstance(image_pth, numpy.ndarray):
            image = Image.fromarray(image_pth)
        elif isinstance(image_pth, bytes) or isinstance(image_pth, bytearray):
            buff = io.BytesIO()
            buff.write(image_pth)
            buff.seek(0)
            image = Image.open(buff).convert('RGB')

        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image_pth).convert('RGB')
            except Exception as e:
                print("failed to convert image: " + str(e))

        if image is None:
            image = image_src

        print("OCR Started...")
        if self.model is None or self.processor is None:
            self.init_reader(src_languages)

        result_lines = []
        bounding_boxes = []
        if self.model is not None or self.processor is not None:
            try:
                inputs = self.processor(image, return_tensors="pt", format=False, crop_to_patches=False, max_patches=5).to(self.device)
                generate_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self.processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=4096,
                )
                result_data = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                result_lines = result_data.split("\n")

            except Exception as e:
                print(e)

        print("OCR Finished.")

        return result_lines, image, bounding_boxes
