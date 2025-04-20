import os

import torch
import gc

import transformers.utils
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from Models.Singleton import SingletonMeta

from pathlib import Path
import downloader


class TransformerWhisper(metaclass=SingletonMeta):
    model = None
    previous_model = None
    processor = None
    pipe = None
    compute_type = "float32"
    compute_device = "cpu"
    compute_device_str = "cpu"

    text_correction_model = None

    currently_downloading = False
    model_cache_path = Path(".cache/whisper-transformer")
    MODEL_LINKS = {}
    MODELS_LIST_URLS = [
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/whisper-transformer/models.yaml",
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/whisper-transformer/models.yaml",
        "https://s3.libs.space:9000/ai-models/whisper-transformer/models.yaml",
    ]
    _debug_skip_dl = False

    def __init__(self, compute_type="float32", device="cpu"):
        os.makedirs(self.model_cache_path, exist_ok=True)
        self.compute_type = compute_type
        self.set_compute_device(device)
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

    def load_model_list(self):
        if not self._debug_skip_dl:
            if not downloader.download_extract(self.MODELS_LIST_URLS,
                                               str(self.model_cache_path.resolve()),
                                               '', title="Speech 2 Text (Whisper-Transformer Model list)", extract_format="none"):
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
                                                   file["checksum"], title="Speech 2 Text (Whisper-Transformer) - " + model_name, extract_format="none"):
                    print(f"Download failed: {file}")

        self.currently_downloading = False

    def load_model(self, model='small', compute_type="float32", device="cpu"):
        if self.previous_model is None or model != self.previous_model:
            compute_dtype = self._str_to_dtype_dict(compute_type).get('dtype', torch.float32)
            compute_4bit = self._str_to_dtype_dict(self.compute_type).get('4bit', False)
            compute_8bit = self._str_to_dtype_dict(self.compute_type).get('8bit', False)
            self.compute_type = compute_type

            self.set_compute_device(device)

            if not self._debug_skip_dl and not model == "custom":
                self.download_model(model)

            if self.model is None or model != self.previous_model:
                if self.model is not None:
                    self.release_model()

                self.previous_model = model
                self.release_model()
                attention_type = "sdpa"
                if transformers.utils.is_flash_attn_2_available():
                    attention_type = "flash_attention_2"
                print(f"Loading Whisper-Transformer model: {model} on {device} with {compute_type} precision using {attention_type}...")
                self.model = WhisperForConditionalGeneration.from_pretrained(str(Path(self.model_cache_path / model).resolve()), torch_dtype=compute_dtype, load_in_8bit=compute_8bit, load_in_4bit=compute_4bit, device_map=self.compute_device, attn_implementation=attention_type)
                #try:
                #    # Enable static cache and compile the forward pass
                #    self.model.generation_config.cache_implementation = "static"
                #    self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
                #except Exception as e:
                #    print(f"Warning: Failed to enable static cache and compile the forward pass: {e}")

                #if not compute_8bit and not compute_4bit:
                #self.model = self.model.to(self.compute_device)
                self.processor = WhisperProcessor.from_pretrained(str(Path(self.model_cache_path / model).resolve()))

                # self.pipe = pipeline(
                #     "automatic-speech-recognition",
                #     model=self.model,
                #     tokenizer=self.processor.tokenizer,
                #     feature_extractor=self.processor.feature_extractor,
                #     chunk_length_s=30,
                #     return_language=True,
                #     torch_dtype=compute_dtype,
                # )

                #self.model.config.forced_decoder_ids = None

    def transcribe(self, audio_sample, model, task, language,
                   return_timestamps=False, beam_size=4) -> dict:
        self.load_model(model, self.compute_type, self.compute_device_str)

        compute_dtype = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

        return_language = language

        if self.model is not None and self.processor is not None:
            processor_result = self.processor(audio_sample, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to(self.compute_device).to(compute_dtype)
            input_features = processor_result.input_features
            attention_mask = processor_result.attention_mask

            transcriptions = [""]
            with torch.no_grad():

                # result = self.pipe(audio_sample, return_timestamps="word", generate_kwargs={"task": task, "language": language, "num_beams": beam_size})
                # print("result")
                # print(result)

                predicted_ids = self.model.generate(inputs=input_features,
                                                    task=task, language=language, num_beams=beam_size,
                                                    return_timestamps=True,
                                                    forced_decoder_ids=None,
                                                    attention_mask=attention_mask,
                                                    )
                transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

                # @todo this does not work anymore to get the detected language
                #pred_language = self.processor.batch_decode(predicted_ids[:, 1:2], skip_special_tokens=False)
                # remove "<|" and "|>" from pred_language
                #if len(pred_language) > 0:
                #    pred_language = [lang.strip("<|>") for lang in pred_language]
                #    return_language = pred_language[0]

            result_text = ''.join(transcriptions).strip()

            return {
                'text': result_text,
                'type': task,
                'language': return_language
            }
        else:
            return {
                'text': "",
                'type': task,
                'language': return_language
            }

    def release_model(self):
        if self.model is not None:
            print("Releasing Whisper-Transformer model...")
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
                        'files': []
                    }

                # Add the file details to the model's files list
                file_data = {
                    'urls': [
                        f'https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/whisper-transformer/{model_name}/{file}',
                        f'https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/whisper-transformer/{model_name}/{file}',
                        f'https://s3.libs.space:9000/ai-models/whisper-transformer/{model_name}/{file}'
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
