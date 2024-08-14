import os

import torch
import gc

import torchaudio
from transformers import WhisperProcessor
from .whisper_medusa.models import WhisperMedusaModel
from Models.Singleton import SingletonMeta

from pathlib import Path
import downloader

MODEL_LINKS = {
    "v1": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-medusa/v1.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-medusa/v1.zip",
            "https://s3.libs.space:9000/ai-models/Whisper-medusa/v1.zip",
        ],
        "checksum": "d9242ba4acc10b20d6082a25c6805938b91336d56bddcb55a88e4a05bf7abdc9",
        "file_checksums": {
            "added_tokens.json": "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1",
            "config.json": "16346762b14c116eeda12b48f20e2281b327a11b516f8b004ce065fcb1450186",
            "generation_config.json": "98b06d803db59396298013f19cb1a206b17eee3b56b7d52357d48a09169eec05",
            "merges.txt": "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6",
            "model-00001-of-00002.safetensors": "b09e03326f4a9e3cd9bac17a55e17c60a3463e720a1cf0a51b8ba246a2b70b67",
            "model-00002-of-00002.safetensors": "6c496a29e2d131f999bbec815e4bd7a38b2ca436ce0d902237fdbd2971b35b74",
            "model.safetensors.index.json": "0b80666c06d5054aa425a07d9f2f4ecabf9e6d7b8333f0dc5d85d4f79c9ff449",
            "normalizer.json": "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
            "preprocessor_config.json": "a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d",
            "special_tokens_map.json": "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691",
            "tokenizer_config.json": "2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce",
            "vocab.json": "50d6a919f0a0601d56a04eb583c780d18553aa388254ba3158eb6a00f13e2c1a"
        }
    }
}

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

    def _needs_download(self, model_name):
        if not self.model_cache_path.exists():
            return True
        model_path = Path(self.model_cache_path / model_name)
        if not model_path.exists():
            return True

        expected_hashes = MODEL_LINKS[model_name]["file_checksums"]
        actual_hashes = downloader.load_hashes(model_path)

        if not actual_hashes:
            if downloader.check_file_hashes(model_path, expected_hashes):
                return False
            else:
                return True

        for file_name, expected_hash in expected_hashes.items():
            actual_hash = actual_hashes.get(file_name)
            if actual_hash.lower() != expected_hash.lower():
                if downloader.sha256_checksum(model_path / file_name).lower() == expected_hash.lower():
                    actual_hashes[file_name] = expected_hash.lower()
                else:
                    return True
        return False

    def download_model(self, model_name):
        os.makedirs(self.model_cache_path, exist_ok=True)

        model_path = Path(self.model_cache_path / model_name)
        os.makedirs(model_path, exist_ok=True)

        file_checksums_check_need_dl = False
        hash_checked_file = model_path / "hash_checked"

        if "file_checksums" in MODEL_LINKS[model_name]:
            if not hash_checked_file.is_file():
                file_checksums_check_need_dl = True

        if not model_path.exists() or file_checksums_check_need_dl:
            print("downloading Medusa Whisper model: " + model_name)
            if not downloader.download_extract(
                    MODEL_LINKS[model_name]["urls"],
                    str(model_path.resolve()),
                    MODEL_LINKS[model_name]["checksum"],
                    title="Speech 2 Text (Medusa Whisper) - " + model_name
            ):
                print("Model download failed")
            if file_checksums_check_need_dl:
                downloader.save_hashes(model_path, MODEL_LINKS[model_name]["file_checksums"])


    def load_model(self, model='v1', compute_type="float32", device="cpu"):
        model_path = Path(self.model_cache_path / model)
        if self.model is None:
            if self._needs_download(model):
                self.download_model(model)

            compute_dtype = self._str_to_dtype_dict(compute_type).get('dtype', torch.float32)
            self.set_compute_device(device)
            #self.model = WhisperMedusaModel.from_pretrained("aiola/whisper-medusa-v1", torch_dtype=compute_dtype, device_map=self.compute_device)
            self.model = WhisperMedusaModel.from_pretrained(model_path)
            self.model = self.model.to(self.compute_device)
            self.model = self.model.to(compute_dtype)
        if self.processor is None:
            self.processor = WhisperProcessor.from_pretrained(model_path)

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
        print("Releasing Medusa Whisper model...")
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
