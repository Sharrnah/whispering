import numpy as np
from pathlib import Path
import torch
import transformers
from transformers import VoxtralForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

import downloader
from Models.Singleton import SingletonMeta

supported_audio_languages = {
    "en": "English",
    "de": "German",
    "nl": "Dutch",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ar": "Arabic",
}
supported_text_languages = {
    "en": "English",
    "de": "German",
    "nl": "Dutch",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ar": "Arabic",
}

transcribe_ignore_results = [
    "I'm sorry, but I'm unable to assist with that. Could you please provide more context or clarify your request?",
]

MODEL_LINKS = {
    "Voxtral-Mini-3B-2507": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/voxtral/Voxtral-Mini-3B-2507.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/voxtral/Voxtral-Mini-3B-2507.zip",
            "https://s3.libs.space:9000/ai-models/voxtral/Voxtral-Mini-3B-2507.zip",
        ],
        "checksum": "",
        "file_checksums": {
        },
        "path": "Voxtral-Mini-3B-2507",
    },
}

class Voxtral(metaclass=SingletonMeta):
    model_path = Path(Path.cwd() / ".cache" / "voxtral")
    download_state = {"is_downloading": False}

    current_model = "Voxtral-Mini-3B-2507"

    compute_type = "float32"
    compute_device = "cpu"  # cuda:0
    compute_device_str = "cpu"  # cuda:0

    processor = None
    model = None
    generation_config = None

    def __init__(self, compute_type="", device=""):
        if device == "" or device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_compute_device(device)

        if compute_type == "":
            compute_type = "float32"
            if self.compute_device_str == "cuda" or self.compute_device_str.startswith("cuda:"):
                compute_type = "bfloat16"
        self.set_compute_type(compute_type)

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": self.model_path,
            "model_link_dict": MODEL_LINKS,
            "model_name": model_name,
            "title": "Multimodal (Voxtral)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    @staticmethod
    def get_languages():
        return tuple([{"code": code, "name": language} for code, language in supported_text_languages.items()])

    def _str_to_dtype_dict(self, dtype_str):
        if dtype_str == "float16":
            return {'dtype': torch.float16, '4bit': False, '8bit': False}
        if dtype_str == "bfloat16":
            return {'dtype': torch.bfloat16, '4bit': False, '8bit': False}
        elif dtype_str == "float32":
            return {'dtype': torch.float32, '4bit': False, '8bit': False}
        #elif dtype_str == "float8":
        #    return {'dtype': torch.float8_e4m3fn, '4bit': False, '8bit': False}
        elif dtype_str == "4bit":
            return {'dtype': torch.float16, '4bit': True, '8bit': False}
        elif dtype_str == "8bit":
            return {'dtype': torch.float16, '4bit': False, '8bit': True}
        else:
            return {'dtype': torch.float32, '4bit': False, '8bit': False}

    def set_compute_type(self, compute_type):
        self.compute_type = compute_type

    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(self.compute_device_str)
        elif device.startswith("cuda:"):
            device_id = 0
            device_id_split = device.split(":")
            if len(device_id_split) > 1:
                device_id = int(device_id_split[1])
            device = torch.device(f"cuda:{device_id}")
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

    @torch.no_grad()
    def load_model(self, model='Voxtral-Mini-3B-2507', compute_type="float32", device="cpu"):
        if self.model is not None and self.processor is not None:
            return
        self.current_model = model

        self.download_model(model)

        self.processor = AutoProcessor.from_pretrained(Path(self.model_path / model).resolve(), trust_remote_code=True)

        attention_implementation = 'sdpa'
        quantization_config = None
        main_torch_dtype = self._str_to_dtype_dict(self.compute_type)['dtype']

        if self.compute_device_str.startswith("cuda"):
            if self.compute_type == "4bit" or self.compute_type == "8bit":
                # @todo not working currently for audio. only for text
                print("Loading model in 4bit or 8bit mode")
                print(self._str_to_dtype_dict(self.compute_type))
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self._str_to_dtype_dict(self.compute_type)['4bit'],
                    load_in_8bit=self._str_to_dtype_dict(self.compute_type)['8bit'],
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                attention_implementation = 'sdpa'
            elif self.compute_type == "float32":
                attention_implementation = 'sdpa'
            elif (self.compute_type == "float16" or self.compute_type == "bfloat16") and transformers.utils.is_flash_attn_2_available():
                attention_implementation = 'flash_attention_2'

            self.model = VoxtralForConditionalGeneration.from_pretrained(
                Path(self.model_path / model).resolve(),
                trust_remote_code=True,
                device_map=self.compute_device_str,
                torch_dtype=main_torch_dtype,
                _attn_implementation=attention_implementation,
                quantization_config=quantization_config,
            )

            if quantization_config is None:
                self.model = self.model.cuda()
        else:
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                Path(self.model_path / model).resolve(),
                trust_remote_code=True,
                device_map=self.compute_device_str,
                torch_dtype=main_torch_dtype,
                _attn_implementation='sdpa',
            ).cpu()

    @torch.no_grad()
    def transcribe(self, audio_sample, task, language='', chat_message='', image_sample=None, system_prompt='',
                   return_timestamps=False, beam_size=4) -> dict:

        # https://huggingface.co/mistralai/Voxtral-Mini-3B-2507

        response_dict = {
            'text': '',
            'type': '',
            'language': '',
        }

        # require model, processor, and audio_sample
        if self.model is None or self.processor is None or audio_sample is None:
            return response_dict

        if language == '' or language is None:
            language = 'auto'

        additional_args = {}
        if language != 'auto' and language != '' and language != None:
            # provide additional arguments for processor
            additional_args = {
                'language': language,
            }

        # prepare inputs in-memory
        inputs = self.processor.apply_transcription_request(
            audio=audio_sample,
            sampling_rate=16000,
            format=['WAV'],
            model_id=str(Path(self.model_path / self.current_model).resolve()),
             **additional_args
        ).to(self.compute_device_str)

        # generate and decode
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=500,
            generation_config=self.generation_config,
            num_beams=1
        )
        response = self.processor.batch_decode(
            generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        # check if response is in ignore list and set to empty string if so
        if response in transcribe_ignore_results:
            response = ""

        response_dict = {'text': response, 'type': task, 'language': language}

        return response_dict
