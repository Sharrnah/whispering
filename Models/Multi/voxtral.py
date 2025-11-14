import io
import soundfile as sf
import base64
from pathlib import Path
import torch
import transformers
from transformers import VoxtralForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

import downloader
import settings
from Models.Singleton import SingletonMeta

# patching optional language field
from typing import Optional
from pydantic_extra_types.language_code import LanguageAlpha2
from mistral_common.protocol.transcription.request import TranscriptionRequest as _TR

class TranscriptionRequest(_TR):
    # make it really optional
    language: Optional[LanguageAlpha2] = None

supported_audio_languages = {
    # "en": "English",
    # "de": "German",
    # "nl": "Dutch",
    # "fr": "French",
    # "it": "Italian",
    # "es": "Spanish",
    # "pt": "Portuguese",
    # "hi": "Hindi",
    # "ar": "Arabic",
    "en": "English",
    "de": "German",
    "nl": "Dutch",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "pl": "Polish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ar": "Arabic",
    "zh": "Chinese (Mandarin/Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
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
        "checksum": "d9a392232b0beb4b1ce16169d97cd48bc7346378c6f350434d2acb82929f6edc",
        "file_checksums": {
            "config.json": "368124c9a9171b3a6c0cdb35cd7fedeff465eefecf315e2a621684bbed7dcd7e",
            "generation_config.json": "cebeed3d4a1680c9b311863385d32b91008480dfbf8cf7abc2142447b8f73b76",
            "model-00001-of-00002.safetensors": "1facdc4c5a0e84f2881a59c4756441b7030a1f1036b2124c117540408d0e5fe9",
            "model-00002-of-00002.safetensors": "0a103ba715bc0d656e94bf79d637d4f8f0f4fe6d76fb1d4db7ee46c6b940f631",
            "model.safetensors.index.json": "7506794dd65e4324685f9634b3d83f20d36a723543650c1e6b72d81965289930",
            "params.json": "4a37e19f2524a44bebf36a3da37a55dde31b8df9eab9b35d359b9a315c74e991",
            "preprocessor_config.json": "86d67d926e17d3a9bb0bae334c9fc46bc163181992357a0357c0aec2c7e131d1",
            "tekken.json": "4aaf3836c2a5332f029ce85a7a62255c966f47b6797ef81dedd0ade9c862e4a8"
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
    last_chat_message = ''

    def __init__(self, compute_type="", device=""):
        if device == "" or device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_compute_device(device)

        if compute_type == "":
            compute_type = "float32"
            if self.compute_device_str == "cuda" or self.compute_device_str.startswith("cuda:"):
                compute_type = "bfloat16"
        self.set_compute_type(compute_type)

        self.last_chat_message = settings.GetOption("stt_llm_prompt") if settings.GetOption("stt_llm_prompt") else ""

    def download_model(self, model_name):
        downloader.download_model({
            "model_path": self.model_path,
            "model_link_dict": MODEL_LINKS,
            "model_name": model_name,
            "title": "Voxtral (Multimodal)",

            "alt_fallback": False,
            "force_non_ui_dl": False,
            "extract_format": "zip",
        }, self.download_state)

    @staticmethod
    def get_languages():
        return tuple([{"code": code, "name": language} for code, language in supported_audio_languages.items()])

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
                dtype=main_torch_dtype,
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
                dtype=main_torch_dtype,
                _attn_implementation='sdpa',
            ).cpu()

    def _numpy_to_wav(self, audio_sample):
        # serialize NumPy array to WAV in-memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_sample, 16000, format='WAV')
        wav_buffer.seek(0)
        return wav_buffer

    def _model_transcribe(self, audio_sample, task, language=''):
        if language == '' or language is None:
            language = 'auto'

        additional_args = {}
        if language != 'auto' and language != '' and language != None:
            # provide additional arguments for processor
            additional_args = {
                'language': language,
            }

        #### alternative way to transcribe using the mistral helper directly
        wav_buffer = self._numpy_to_wav(audio_sample)

        # build "OpenAI" request using mistral-common helper
        openai_req = {
            "model": str(Path(self.model_path / self.current_model).resolve()),
            "file":  wav_buffer,
            "temperature": 0.0,
            **additional_args,
        }
        tr = TranscriptionRequest.from_openai(openai_req)

        # Tokenise + feature‑extract like the helper does, but by hand
        tok = self.processor.tokenizer.tokenizer.encode_transcription(tr)
        audio_feats = self.processor.feature_extractor(
            audio_sample, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.model.device)

        # Generate
        ids = self.model.generate(
            input_features=audio_feats,
            input_ids     = torch.tensor([tok.tokens], device=self.model.device),
            max_new_tokens=500,
            num_beams=1
        )
        response = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

        ####### Original Huggingface Transformer code. (commented out until language argument is optional)
        # # prepare inputs in-memory
        # inputs = self.processor.apply_transcription_request(
        #     audio=audio_sample,
        #     sampling_rate=16000,
        #     format=['WAV'],
        #     model_id=str(Path(self.model_path / self.current_model).resolve()),
        #      **additional_args
        # ).to(self.compute_device_str)
        #
        # # generate and decode
        # generate_ids = self.model.generate(
        #     **inputs,
        #     max_new_tokens=500,
        #     generation_config=self.generation_config,
        #     num_beams=1
        # )
        # response = self.processor.batch_decode(
        #     generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        # )[0]
        #######
        # check if response is in ignore list and set to empty string if so

        response = response.strip()

        # remove "lang:" prefix if it exists
        if language is not None and language != '' and response.startswith("lang:"+language):
            response = response[len("lang:"+language):].strip()

        if response in transcribe_ignore_results:
            response = ""

        response_dict = {'text': response, 'type': task, 'language': language}

        return response_dict

    def _model_question_answering(self, audio_sample, task, chat_message=''):
        conversation = [
            {
                "role": "user",
                "content": [],
            }
        ]
        if audio_sample is not None:
            wav_buffer = self._numpy_to_wav(audio_sample)
            # convert wav_buffer to base64
            wav_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
            conversation[0]["content"].append({
                "type": "audio",
                "base64": wav_base64,
            })

        if chat_message:
            conversation[0]["content"].append({
                "type": "text",
                "text": chat_message,
            })

        inputs = self.processor.apply_chat_template(conversation)
        if self.compute_device_str.startswith("cuda"):
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        else:
            inputs = inputs.to(self.model.device, dtype=torch.float32)


        outputs = self.model.generate(**inputs, max_new_tokens=500) # temperature=0.2, top_p=0.95
        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        response = decoded_outputs[0]

        response_dict = {'llm_answer': response, 'text': response, 'type': 'llm_answer'}
        return response_dict

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
        if self.model is None or self.processor is None:
            return response_dict

        # switch case for task
        match task:
            case 'transcribe':
                if audio_sample is None:
                    return response_dict
                return self._model_transcribe(audio_sample, task, language)
            case 'question_answering':
                if chat_message == '':
                    chat_message = self.last_chat_message
                if audio_sample is None and chat_message == '':
                    return response_dict
                return self._model_question_answering(audio_sample, task, chat_message)
            case 'translate':
                prompt = f'Only Translate audio into {supported_audio_languages[language]}. Just write the translation without explanations.'
                response_dict = self._model_question_answering(audio_sample, task, prompt)
                response_dict["language"] = language
                return response_dict
            case 'text_translate':
                prompt = f'Do not answer or explain. Just write the translation without explanations. Translate the following text into {supported_audio_languages[language]}: {chat_message}'
                response_dict = self._model_question_answering(None, task, prompt)

                translation = response_dict["text"].strip()
                # trim 「」" from the start and end of the translation
                pairs = {'「': '」', '"': '"'}
                if translation and pairs.get(translation[0]) == translation[-1]:
                    translation = translation[1:-1].strip()

                response_dict["language"] = language
                response_dict["text"] = translation
                response_dict["txt_translation"] = translation
                response_dict["txt_translation_target"] = language
                return response_dict
            case 'update_llm_prompt':
                if chat_message == '':
                    return response_dict
                self.last_chat_message = chat_message
                return response_dict
