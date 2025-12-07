import copy
import json
import os
import signal
# patch (/nemo/utils/exp_manager.py) - https://github.com/NVIDIA/NeMo/issues/12858
signal.SIGKILL = signal.SIGTERM

import torch

import yaml
import nemo
from nemo.collections.asr.models import EncDecMultiTaskModel
import nemo.collections.asr as nemo_asr
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

# parekeet v3 languages:
# PARAKEET_V3_LANGUAGES = {
#     "bg": "Bulgarian",
#     "hr": "Croatian",
#     "cs": "Czech",
#     "da": "Danish",
#     "nl": "Dutch",
#     "en": "English",
#     "et": "Estonian",
#     "fi": "Finnish",
#     "fr": "French",
#     "de": "German",
#     "el": "Greek",
#     "hu": "Hungarian",
#     "it": "Italian",
#     "lv": "Latvian",
#     "lt": "Lithuanian",
#     "mt": "Maltese",
#     "pl": "Polish",
#     "pt": "Portuguese",
#     "ro": "Romanian",
#     "sk": "Slovak",
#     "sl": "Slovenian",
#     "es": "Spanish",
#     "sv": "Swedish",
#     "ru": "Russian",
#     "uk": "Ukrainian",
# }
PARAKEET_LANGUAGES = {
    "auto": "Auto",
}

LANGUAGES = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
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
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/nemo-canary/list_250701/models.yaml",
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/nemo-canary/list_250701/models.yaml",
        "https://s3.libs.space:9000/ai-models/nemo-canary/models.yaml",
    ]
    _debug_skip_dl = False

    def __init__(self, compute_type="float32", device="cpu"):
        print("nemo-toolkit:", nemo.__version__)
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

    @staticmethod
    def get_parakeet_languages():
        return tuple([{"code": code, "name": language} for code, language in PARAKEET_LANGUAGES.items()])

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

    def load_model(self, model='canary-1b-v2', compute_type="float32", device="cpu"):
        #self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        #if self.model is None:

        if not self._debug_skip_dl:
            self.download_model(model)

        torch.set_grad_enabled(False)

        #quant_modules.initialize()

        if self.previous_model is None or self.model is None or model != self.previous_model:
            print(f"Loading NeMo Canary model: {model} on {device} with {compute_type} precision...")
            if model.startswith("canary-"):
                self.model = EncDecMultiTaskModel.restore_from(str(Path(self.model_cache_path / model / (model+".nemo")).resolve()), map_location=torch.device(device))
            elif model.startswith("parakeet-"):
                self.model = nemo_asr.models.ASRModel.restore_from(str(Path(self.model_cache_path / model / (model+".nemo")).resolve()), map_location=torch.device(device))
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

    @torch.inference_mode()
    def transcribe(self, audio_sample, task, source_lang=None, target_lang='en',
                   without_timestamps=False, **kwargs) -> dict:

        model = "canary-1b"
        if "model" in kwargs:
            model = kwargs["model"]

        self.load_model(model, self.compute_type, self.compute_device)

        config_kwargs = {}

        if model.startswith("canary-"):
            beam_size = 4
            if "beam_size" in kwargs:
                beam_size = kwargs["beam_size"]
            length_penalty = 1.0
            if "length_penalty" in kwargs:
                length_penalty = kwargs["length_penalty"]
            temperature = 1.0
            if "temperature" in kwargs:
                temperature = kwargs["temperature"]

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

            # -------------------------------

            # Prepare the manifest data
            config_kwargs = {
                "task": taskname,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "pnc": "yes",
                #"answer": "na",
                "answer": "predict",
                "timestamps": "no" if without_timestamps else "yes",
            }

        result_text = ""

        compute_type = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

        segment_list = []

        if model.startswith("canary-"):
            # Transcribe using the model
            if not self.compute_device.startswith("cuda"):
                predicted_text = self.model.transcribe([audio_sample], batch_size=16, verbose=False, **config_kwargs)
            else:
                with torch.cuda.amp.autocast(dtype=compute_type):
                    predicted_text = self.model.transcribe([audio_sample], batch_size=16, verbose=False, **config_kwargs)

            if len(predicted_text) > 0:
                result_text = predicted_text[0].text

            if not without_timestamps:
                timestamp = predicted_text[0].timestamp
                segments = timestamp['segment']
                for single_segment in segments:
                    segment_list.append({'text': single_segment['segment'], 'start': single_segment['start'], 'end': single_segment['end']})

        elif model.startswith("parakeet-"):
            config_kwargs = {}
            if not without_timestamps:
                config_kwargs["timestamps"] = True
                predicted_text = self.model.transcribe([audio_sample], verbose=False, **config_kwargs)
                timestamp = predicted_text[0].timestamp
                segments = timestamp['segment']
                for single_segment in segments:
                    segment_list.append({'text': single_segment['segment'], 'start': single_segment['start'], 'end': single_segment['end']})
            else:
                predicted_text = self.model.transcribe([audio_sample], verbose=False)
            result_text = predicted_text[0].text

        result = {
            'text': result_text,
            'type': "transcribe",
            'language': source_lang,
            'target_lang': target_lang
        }
        # add segments if they exist
        if len(segment_list) > 0:
            result['segments'] = segment_list

        return result

    # https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py
    def long_form_transcribe(self, audio_sample, task, source_lang=None, target_lang='en',
                                 without_timestamps=False, **kwargs):
        filepaths = audio_sample

        from omegaconf import OmegaConf
        from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchMultiTaskAED
        from nemo.collections.asr.parts.utils.transcribe_utils import (
            compute_output_filename,
            get_buffered_pred_feat_multitaskAED,
            setup_model,
            write_transcription,
        )

        cfg = self.model.cfg
        cfg.chunk_len_in_secs = 10.0
        torch.set_grad_enabled(False)

        # setup GPU
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        if cfg.cuda is None:
            if torch.cuda.is_available():
                device = [0]  # use 0th CUDA device
                accelerator = 'gpu'
            else:
                device = 1
                accelerator = 'cpu'
        else:
            device = [cfg.cuda]
            accelerator = 'gpu'
        map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')

        asr_model, model_name = setup_model(cfg, map_location)
        model_cfg = copy.deepcopy(asr_model._cfg)

        OmegaConf.set_struct(model_cfg.preprocessor, False)
        # some changes for streaming scenario
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0

        # Disable config overwriting
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        # Compute output filename
        cfg = compute_output_filename(cfg, model_name)

        asr_model.change_decoding_strategy(cfg.decoding)

        asr_model.eval()
        asr_model = asr_model.to(asr_model.device)

        feature_stride = model_cfg.preprocessor['window_stride']
        model_stride_in_secs = feature_stride * cfg.model_stride

        frame_asr = FrameBatchMultiTaskAED(
            asr_model=asr_model,
            frame_len=cfg.chunk_len_in_secs,
            total_buffer=cfg.chunk_len_in_secs,
            batch_size=cfg.batch_size,
        )

        amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

        manifest = cfg.dataset_manifest

        with torch.amp.autocast(asr_model.device.type, enabled=cfg.amp, dtype=amp_dtype):
            with torch.no_grad():
                hyps = get_buffered_pred_feat_multitaskAED(
                    frame_asr,
                    model_cfg.preprocessor,
                    model_stride_in_secs,
                    asr_model.device,
                    manifest,
                    filepaths,
                    timestamps=cfg.timestamps,
                )
        print("hyps")
        print(hyps)