import os

import torch
import gc

import transformers.utils
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration, BitsAndBytesConfig
from Models.Singleton import SingletonMeta

from pathlib import Path
import downloader


MODEL_LINKS = {
    "VibeVoice-ASR-HF": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/VibeVoice-ASR/VibeVoice-ASR-HF.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/VibeVoice-ASR/VibeVoice-ASR-HF.zip",
            "https://s3.libs.space:9000/ai-models/VibeVoice-ASR/VibeVoice-ASR-HF.zip",
        ],
        "checksum": "b5c448af52ba6f738fadfe9895607940bd1c614906f31e678af5924b1bc1b2e4",
        "file_checksums": {
            "chat_template.jinja": "facaa74472ce1cc68fd19be60062e202ec5ce8b7b87c28a28a7d27f6adfef58d",
            "config.json": "89e5f9e4932e72cd8ace355e80b8c5c0e4ac6f6a92e97bf47e8b8ffdca1035e0",
            "generation_config.json": "0d18916d2ae79a3cebc7b05d99fd66203c6180a64dfeb4586856c104bcd46e32",
            "model-00001-of-00008.safetensors": "580189757a1c737ecc6fad16e633b922bc73220268356e31362c24365f081095",
            "model-00002-of-00008.safetensors": "65e419d28fc8c87d6938ec611c7b425b38e391c80a2df59c2b4935f6fd65c7c2",
            "model-00003-of-00008.safetensors": "30f574c764550d26c654cc3f5dd000b90b0c3305db955267c577001c1013fdf7",
            "model-00004-of-00008.safetensors": "8fa3d9350f0a8cc97524617ff582e451159bfdba2ef0b2e372a627518ef681b1",
            "model-00005-of-00008.safetensors": "5015ecacc86d897d20d5e42ba52ef83355d42f587d3b158ff897ec94847bd083",
            "model-00006-of-00008.safetensors": "7463d75607185c118925cea6a60a5698aaa6f428e626bcee774572bdc28ef55d",
            "model-00007-of-00008.safetensors": "b52aa2fab9640bcac5201a278f540b018c439ed4e99516d9ff30ea1ae399525e",
            "model-00008-of-00008.safetensors": "a66ced85b619507e1970dc51be45786ff55803ccc63658765e412ea23d9a8ace",
            "model.safetensors.index.json": "c807b82f9bb711f0fd3cc9cc138abccb3bcb1e27a002605cc4fbb3c113d322f3",
            "processor_config.json": "918e2554dde40a1558a2c9c5f1ae4e077b594c00c14276f776ad67c73a2eb6a6",
            "tokenizer.json": "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8",
            "tokenizer_config.json": "64029a57ca4f977c2f50fe95dda323a923d4b5f9e31836257610c0362c8e683c"
        },
        "path": "VibeVoice-ASR-HF",
    }
}

class TransformerVibeVoiceASR(metaclass=SingletonMeta):
    model = None
    previous_model = None
    processor = None
    compute_type = "float32"
    compute_device = "cpu"
    compute_device_str = "cpu"

    text_correction_model = None

    model_cache_path = Path(".cache/vibevoice-asr-cache")

    download_state = {"is_downloading": False}

    def __init__(self, compute_type="float32", device="cpu"):
        os.makedirs(self.model_cache_path, exist_ok=True)
        self.compute_type = compute_type
        self.set_compute_device(device)

        #if self._debug_skip_dl:
        #    # generate models.yaml
        #    self.generate_models_yaml(self.model_cache_path, "models.yaml")

    def _str_to_dtype_dict(self, dtype_str):
        if dtype_str == "float16":
            return {'dtype': torch.float16, '4bit': False, '8bit': False}
        if dtype_str == "bfloat16":
            return {'dtype': torch.bfloat16, '4bit': False, '8bit': False}
        elif dtype_str == "float32":
            return {'dtype': torch.float32, '4bit': False, '8bit': False}
        elif dtype_str == "4bit":
            return {'dtype': torch.float16, '4bit': True, '8bit': False}
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
        downloader.download_model({
            "model_path": self.model_cache_path,
            "model_link_dict": MODEL_LINKS,
            "model_name": model_name,
            "title": "Text Translation (HY-MT1.5)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    def load_model(self, model='VibeVoice-ASR-HF', compute_type="float32", device="cpu"):
        if self.previous_model is None or model != self.previous_model:
            self.compute_type = compute_type

            compute_dtype = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

            self.set_compute_device(device)

            if not model == "custom":
                self.download_model(model)

            if self.model is None or model != self.previous_model:
                if self.model is not None:
                    self.release_model()

                self.previous_model = model
                self.release_model()
                attention_type = "sdpa"

                # build quantization configuration
                quantization_config = None
                if self.compute_device_str.startswith("cuda"):
                    if self.compute_type == "4bit" or self.compute_type == "8bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=self._str_to_dtype_dict(self.compute_type)['4bit'],
                            load_in_8bit=self._str_to_dtype_dict(self.compute_type)['8bit'],
                            bnb_4bit_use_double_quant=False,
                            bnb_4bit_quant_type="nf4",
                            #bnb_4bit_compute_dtype=self._str_to_dtype_dict(self.compute_type)['dtype']
                            bnb_4bit_compute_dtype=torch.float16
                        )

                print(f"Loading VibeVoice-ASR model: {model} on {device} with {compute_type} precision...")
                self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(str(Path(self.model_cache_path / model).resolve()), dtype=compute_dtype, quantization_config=quantization_config, device_map=self.compute_device)
                #try:
                #    # Enable static cache and compile the forward pass
                #    self.model.generation_config.cache_implementation = "static"
                #    self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
                #except Exception as e:
                #    print(f"Warning: Failed to enable static cache and compile the forward pass: {e}")

                #if not compute_8bit and not compute_4bit:
                #self.model = self.model.to(self.compute_device)
                self.processor = AutoProcessor.from_pretrained(str(Path(self.model_cache_path / model).resolve()))

                print("VibeVoice-ASR model loaded successfully.")

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

    def transcribe(self, audio_sample, task, language,
                   return_timestamps=False, beam_size=4, **kargs) -> dict:
        self.load_model("VibeVoice-ASR-HF", self.compute_type, self.compute_device_str)

        compute_dtype = self._str_to_dtype_dict(self.compute_type).get('dtype', torch.float32)

        return_language = language

        prompt = None
        if "prompt" in kargs and kargs["prompt"] is not None and kargs["prompt"] != "":
            prompt = kargs["prompt"]

        if self.model is not None and self.processor is not None:

            # Prepare inputs using `apply_transcription_request`
            inputs = self.processor.apply_transcription_request(
                audio=audio_sample,
                prompt=prompt
            ).to(self.model.device, self.model.dtype)

            output_ids = self.model.generate(**inputs)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            transcription = self.processor.decode(generated_ids, return_format="transcription_only")[0]

            print("transcription")
            print(transcription)

            return {
                'text': transcription,
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
