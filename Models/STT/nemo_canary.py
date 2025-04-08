import json
import os
import signal
# patch (/nemo/utils/exp_manager.py) - https://github.com/NVIDIA/NeMo/issues/12858
signal.SIGKILL = signal.SIGTERM

import torch

import yaml
from nemo.collections.asr.models import EncDecMultiTaskModel
from Models.Singleton import SingletonMeta

from pathlib import Path
import downloader

import soundfile as sf
import tempfile

#try:
#    from pytorch_quantization import nn as quant_nn
#    from pytorch_quantization import quant_modules
#except ImportError:
#    raise ImportError(
#        "pytorch-quantization is not installed. Install from "
#        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
#    )


LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}


class NemoCanary(metaclass=SingletonMeta):
    model = None
    previous_model = None
    processor = None
    compute_type = "float32"
    compute_device = "cpu"

    sample_rate = 16000

    text_correction_model = None

    currently_downloading = False
    model_cache_path = Path(".cache/nemo-canary")
    MODEL_LINKS = {}
    MODELS_LIST_URLS = [
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/nemo-canary/models.yaml",
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/nemo-canary/models.yaml",
        "https://s3.libs.space:9000/ai-models/nemo-canary/models.yaml",
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

    @staticmethod
    def get_languages():
        return tuple([{"code": code, "name": language} for code, language in LANGUAGES.items()])

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
        self.compute_device = device

    def load_model_list(self):
        if not self._debug_skip_dl:
            if not downloader.download_extract(self.MODELS_LIST_URLS,
                                               str(self.model_cache_path.resolve()),
                                               '', title="Speech 2 Text (NeMo Canary Model list)", extract_format="none"):
                print("Model list not downloaded. Using cached version.")

        # Load model list
        if Path(self.model_cache_path / "models.yaml").exists():
            with open(self.model_cache_path / "models.yaml", "r") as file:
                self.MODEL_LINKS = yaml.load(file, Loader=yaml.FullLoader)
                file.close()

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
                                                   file["checksum"], title="Speech 2 Text (NeMo Canary) - " + model_name, extract_format="none"):
                    print(f"Download failed: {file}")

        self.currently_downloading = False

    def load_model(self, model='canary-1b', compute_type="float32", device="cpu"):
        #self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        #if self.model is None:

        if not self._debug_skip_dl:
            self.download_model(model)

        torch.set_grad_enabled(False)

        #quant_modules.initialize()

        if self.previous_model is None or self.model is None or model != self.previous_model:
            print(f"Loading NeMo Canary model: {model} on {device} with {compute_type} precision...")
            self.model = EncDecMultiTaskModel.restore_from(str(Path(self.model_cache_path / model / (model+".nemo")).resolve()), map_location=torch.device(device))
            #self.model.half()
            #self.model.cuda()
            self.model.eval()
            self.previous_model = model

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
                        'files': []
                    }

                # Add the file details to the model's files list
                file_data = {
                    'urls': [
                        f'https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/nemo-canary/{model_name}/{file}',
                        f'https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/nemo-canary/{model_name}/{file}',
                        f'https://s3.libs.space:9000/ai-models/nemo-canary/{model_name}/{file}'
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

    def transcribe(self, audio_sample, task, source_lang=None, target_lang='en',
                   return_timestamps=False, **kwargs) -> dict:

        model = "canary-1b"
        if "model" in kwargs:
            model = kwargs["model"]

        self.load_model(model, self.compute_type, self.compute_device)

        beam_size = 4
        if "beam_size" in kwargs:
            beam_size = kwargs["beam_size"]
        length_penalty = 1.0
        if "length_penalty" in kwargs:
            length_penalty = kwargs["length_penalty"]
        temperature = 1.0
        if "temperature" in kwargs:
            temperature = kwargs["temperature"]

        #taskname = "asr"
        #if task == "transcription":
        #    taskname = "asr"
        #    source_lang = target_lang
        #if task == "translation":
        #    taskname = "ast"

        # transcription
        if source_lang == target_lang:
            taskname = "asr"
        # translation
        else:
            taskname = "s2t_translation"

        self.model.change_decoding_strategy(None)
        decode_cfg = self.model.cfg.decoding
        changed_cfg = False
        if beam_size != decode_cfg.beam.beam_size:
            decode_cfg.beam.beam_size = beam_size
            changed_cfg = True
        if length_penalty != decode_cfg.beam.len_pen:
            decode_cfg.beam.len_pen = length_penalty
            changed_cfg = True
        if temperature != decode_cfg.temperature:
            decode_cfg.temperature = temperature
            changed_cfg = True

        if changed_cfg:
            self.model.change_decoding_strategy(decode_cfg)

        # setup for buffered inference
        self.model.cfg.preprocessor.dither = 0.0
        self.model.cfg.preprocessor.pad_to = 0

        #feature_stride = self.model.cfg.preprocessor['window_stride']
        #model_stride_in_secs = feature_stride * 8  # 8 = model stride, which is 8 for FastConformer

        #transcript = self.model.transcribe([audio_sample], batch_size=8, num_workers=2, taskname=task, source_lang=source_lang, target_lang=target_lang,)
        #transcript = self.model.transcribe([audio_sample], batch_size=8, num_workers=2,)

        with tempfile.TemporaryDirectory() as tmpdirname:
            audio_path = os.path.join(tmpdirname, "audio.wav")
            # Save the numpy array as a WAV file
            sf.write(audio_path, audio_sample, self.sample_rate, 'PCM_16')

            # calculate audio duration
            number_of_samples = audio_sample.shape[0]
            duration_seconds = number_of_samples / self.sample_rate

            # Prepare the manifest data
            manifest_data = [{
                "audio_filepath": audio_path,
                "duration": duration_seconds,
                "taskname": taskname,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "pnc": "yes",
                #"answer": "na",
                "answer": "predict",
            }]

            manifest_path = os.path.join(tmpdirname, "manifest.json")
            with open(manifest_path, "w") as manifest_file:
                for entry in manifest_data:
                    manifest_file.write(json.dumps(entry) + "\n")

            compute_type = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

            # Transcribe using the model
            if not self.compute_device.startswith("cuda"):
                with torch.no_grad():
                    predicted_text = self.model.transcribe(manifest_path, batch_size=16)
            else:
                with torch.cuda.amp.autocast(dtype=compute_type):
                    with torch.no_grad():
                        predicted_text = self.model.transcribe(manifest_path, batch_size=16)

            result = {
                'text': "".join(predicted_text),
                'type': "transcribe",
                'language': source_lang,
                'target_lang': target_lang
            }

            return result
