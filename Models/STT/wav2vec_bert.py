import os

import torch
import gc

import yaml
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
from Models.Singleton import SingletonMeta
from Models.TextCorrection import T5

from pathlib import Path
import downloader


class Wav2VecBert(metaclass=SingletonMeta):
    model = None
    previous_model = None
    processor = None
    compute_type = "float32"
    compute_device = "cpu"

    text_correction_model = None

    currently_downloading = False
    model_cache_path = Path(".cache/wav2vec-bert2.0")
    MODEL_LINKS = {}
    MODELS_LIST_URLS = [
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Wav2VecBert/models.yaml",
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Wav2VecBert/models.yaml",
        "https://s3.libs.space:9000/ai-models/Wav2VecBert/models.yaml",
    ]
    _debug_skip_dl = False

    def __init__(self, compute_type="float32", device="cpu"):
        os.makedirs(self.model_cache_path, exist_ok=True)
        self.compute_type = compute_type
        self.compute_device = device
        self.load_model_list()

        #if self._debug_skip_dl:
        #    # generate models.yaml
        #    self.generate_models_yaml(self.model_cache_path, "models.yaml")

    def _str_to_dtype_dict(self, dtype_str):
        if dtype_str == "float16":
            return {'dtype': torch.float16, '4bit': False, '8bit': False}
        elif dtype_str == "float32":
            return {'dtype': torch.float32, '4bit': False, '8bit': False}
        elif dtype_str == "4bit":
            return {'dtype': torch.float32, '4bit': True, '8bit': False}
        elif dtype_str == "8bit":
            return {'dtype': torch.float32, '4bit': False, '8bit': True}
        else:
            return {'dtype': torch.float32, '4bit': False, '8bit': False}

    def set_compute_type(self, compute_type):
        self.compute_type = compute_type

    def set_compute_device(self, device):
        self.compute_device = device

    def load_model_list(self):
        if not self._debug_skip_dl:
            if not downloader.download_extract(self.MODELS_LIST_URLS,
                                               str(self.model_cache_path.resolve()),
                                               '', title="Speech 2 Text (Wav2VecBert2 Model list)", extract_format="none"):
                print("Model list not downloaded. Using cached version.")

        # Load model list
        if Path(self.model_cache_path / "models.yaml").exists():
            with open(self.model_cache_path / "models.yaml", "r") as file:
                self.MODEL_LINKS = yaml.load(file, Loader=yaml.FullLoader)
                file.close()

    def get_languages(self):
        if not self.MODEL_LINKS:
            # Return a default value or message. Here, we return an empty tuple as a fallback.
            return ()

        # Generate a list of dictionaries, each containing the language code and language name
        languages = []
        for language, details in self.MODEL_LINKS.items():
            # Extract the lang_code for the current language entry
            lang_name = details.get("lang_name", "")  # Fallback to an empty string if not found
            languages.append({"code": language, "name": lang_name})
        return tuple(languages)

    def download_model(self, model_name):
        model_directory = Path(self.model_cache_path / model_name)
        os.makedirs(str(model_directory.resolve()), exist_ok=True)

        # if one of the files does not exist, break the loop and download the files
        needs_download = False
        for file in self.MODEL_LINKS[model_name]["files"]:
            if not Path(model_directory / Path(file["urls"][0]).name).exists():
                needs_download = True
                break

        if not needs_download:
            for file in self.MODEL_LINKS[model_name]["files"]:
                if Path(file["urls"][0]).name == "WS_VERSION":
                    checksum = downloader.sha256_checksum(str(model_directory.resolve() / Path(file["urls"][0]).name))
                    if checksum != file["checksum"]:
                        needs_download = True
                        break

        # iterate over all self.MODEL_LINKS[model_name]["files"] entries and download them
        if needs_download and not self.currently_downloading:
            self.currently_downloading = True
            for file in self.MODEL_LINKS[model_name]["files"]:
                if not downloader.download_extract(file["urls"],
                                                   str(model_directory.resolve()),
                                                   file["checksum"], title="Speech 2 Text (Wav2VecBert2) - " + model_name, extract_format="none"):
                    print(f"Download failed: {file}")

        self.currently_downloading = False

    def load_model(self, model='english', compute_type="float32", device="cpu"):
        if self.previous_model is None or model != self.previous_model:
            compute_dtype = self._str_to_dtype_dict(compute_type).get('dtype', torch.float32)
            compute_4bit = self._str_to_dtype_dict(self.compute_type).get('4bit', False)
            compute_8bit = self._str_to_dtype_dict(self.compute_type).get('8bit', False)
            self.compute_type = compute_type

            self.compute_device = device

            if not self._debug_skip_dl:
                self.download_model(model)

            if self.model is None or model != self.previous_model:
                if self.model is not None:
                    self.release_model()

                self.previous_model = model
                self.release_model()
                print(f"Loading wav2vec model: {model} on {device} with {compute_type} precision...")
                self.model = Wav2Vec2BertForCTC.from_pretrained(str(Path(self.model_cache_path / model).resolve()), torch_dtype=compute_dtype, load_in_8bit=compute_8bit, load_in_4bit=compute_4bit)
                if not compute_8bit and not compute_4bit:
                    self.model = self.model.to(self.compute_device)
                self.processor = Wav2Vec2BertProcessor.from_pretrained(str(Path(self.model_cache_path / model).resolve()))

                # load text correction model
                self.text_correction_model = T5.TextCorrectionT5(compute_type, device)

    def transcribe(self, audio_sample, task, language) -> dict:
        self.load_model(language, self.compute_type, self.compute_device)

        compute_dtype = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

        if self.model is not None and self.processor is not None:
            input_features = self.processor(audio=audio_sample, sampling_rate=16000, return_tensors="pt").to(self.compute_device).to(compute_dtype)

            with torch.no_grad():
                logits = self.model(**input_features).logits

            pred_ids = torch.argmax(logits, dim=-1)

            result_text = self.processor.batch_decode(pred_ids)

            if self.text_correction_model is not None and result_text[0] != "":
                result_text[0] = self.text_correction_model.translate(result_text[0], language)

            return {
                'text': result_text[0],
                'type': task,
                'language': language
            }
        else:
            return {
                'text': "",
                'type': task,
                'language': language
            }

    def release_model(self):
        if self.model is not None:
            print("Releasing wav2vec model...")
            if hasattr(self.model, 'model'):
                del self.model.model
            if hasattr(self.model, 'feature_extractor'):
                del self.model.feature_extractor
            if hasattr(self.model, 'hf_tokenizer'):
                del self.model.hf_tokenizer
            del self.model
        if self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def generate_models_yaml(self, directory, filename):
        # Prepare the data
        data = {}

        # Iterate through the directory
        for root, dirs, files in os.walk(directory):
            ws_version_file = None
            # Get the model name from the directory name
            model_name = os.path.basename(root)
            for file in files:
                # Calculate the SHA256 checksum
                checksum = downloader.sha256_checksum(os.path.join(root, file))

                # Initialize the model in the data dictionary if it doesn't exist
                if model_name not in data:
                    data[model_name] = {
                        'lang_name': model_name.capitalize(),
                        'files': []
                    }

                # Add the file details to the model's files list
                file_data = {
                    'urls': [
                        f'https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Wav2VecBert/{model_name}/{file}',
                        f'https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Wav2VecBert/{model_name}/{file}',
                        f'https://s3.libs.space:9000/ai-models/Wav2VecBert/{model_name}/{file}'
                    ],
                    'checksum': checksum
                }
                if file == "WS_VERSION":
                    ws_version_file = file_data
                else:
                    data[model_name]['files'].append(file_data)

            if ws_version_file is not None:
                data[model_name]['files'].insert(0, ws_version_file)

        # Write to YAML file
        with open(os.path.join(directory, filename), 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
