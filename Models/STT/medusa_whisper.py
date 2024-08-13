import os

import torch
import gc

import torchaudio
from transformers import WhisperProcessor
from .whisper_medusa.models import WhisperMedusaModel
from Models.Singleton import SingletonMeta

from pathlib import Path
import downloader


class MedusaWhisper(metaclass=SingletonMeta):
    model = None
    previous_model = None
    processor = None
    pipe = None
    compute_type = "float32"
    compute_device = "cpu"
    compute_device_str = "cpu"

    text_correction_model = None

    currently_downloading = False
    model_cache_path = Path(".cache/whisper-medusa")
    MODEL_LINKS = {}
    _debug_skip_dl = False

    def __init__(self, compute_type="float32", device="cpu"):
        os.makedirs(self.model_cache_path, exist_ok=True)
        self.compute_type = compute_type
        self.set_compute_device(device)

    def _str_to_dtype_dict(self, dtype_str):
        if dtype_str == "float16":
            return {'dtype': torch.float16, '4bit': False, '8bit': False}
        elif dtype_str == "float32":
            return {'dtype': torch.float32, '4bit': False, '8bit': False}
        elif dtype_str == "4bit":
            return {'dtype': torch.float32, '4bit': True, '8bit': False}
        elif dtype_str == "8bit":
            return {'dtype': torch.float16, '4bit': False, '8bit': True}
        else:
            return {'dtype': torch.float16, '4bit': False, '8bit': False}

    def set_compute_type(self, compute_type):
        self.compute_type = compute_type

    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(self.compute_device_str)
        elif device == "cpu":
            device = torch.device("cpu")
        elif device.startswith("direct-ml"):
            device_id = 0
            device_id_split = device.split(":")
            if len(device_id_split) > 1:
                device_id = int(device_id_split[1])
            import torch_directml
            device = torch_directml.device(device_id)
        self.compute_device = device

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
                                                   file["checksum"], title="Speech 2 Text (Whisper-Transformer) - " + model_name, extract_format="none"):
                    print(f"Download failed: {file}")

        self.currently_downloading = False

    def load_model(self, model='small', compute_type="float32", device="cpu"):
        if self.model is None:
            compute_dtype = self._str_to_dtype_dict(compute_type).get('dtype', torch.float32)
            self.set_compute_device(device)
            #self.model = WhisperMedusaModel.from_pretrained("aiola/whisper-medusa-v1", torch_dtype=compute_dtype, device_map=self.compute_device)
            self.model = WhisperMedusaModel.from_pretrained("aiola/whisper-medusa-v1")
            self.model = self.model.to(self.compute_device)
            self.model = self.model.to(compute_dtype)
        if self.processor is None:
            self.processor = WhisperProcessor.from_pretrained("aiola/whisper-medusa-v1")

    def transcribe(self, audio_sample, model, task, language,
                   return_timestamps=False, beam_size=4) -> dict:
        self.load_model(model, self.compute_type, self.compute_device_str)

        compute_dtype = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

        # convert audio_sample numpy array to torch tensor
        #audio_sample = torch.tensor(audio_sample).to(compute_dtype).to('cpu').squeeze()

        if self.model is not None and self.processor is not None:
            input_features = self.processor(audio_sample, sampling_rate=16_000, return_tensors="pt").input_features
            input_features = input_features.to(self.compute_device).to(compute_dtype)

            transcriptions = [""]
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features,
                                                    task=task, language=language,
                                                    num_beams=1,
                                                    return_timestamps=return_timestamps,
                                                    )
                transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                #transcriptions[0] = self.processor.decode(predicted_ids[0], skip_special_tokens=True)

                print("transcriptions")
                print(transcriptions)

            result_text = ''.join(transcriptions).strip()

            return {
                'text': result_text,
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
        print("Releasing Whisper-Transformer model...")
        if self.model is not None:
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
