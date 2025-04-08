import io
import json
import os
import re
from pathlib import Path

import numpy
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig

import Plugins
import downloader
import settings
from Models.Singleton import SingletonMeta

supported_audio_languages = {
    "en": "English",
    "ch": "Chinese",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "es": "Spanish",
    "pt": "Portuguese",
}
supported_text_languages = {
    "ar": "Arabic",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "he": "Hebrew",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
}

MODEL_LINKS = {
    "Phi-4": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Phi-4/phi4.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Phi-4/phi4.zip",
            "https://s3.libs.space:9000/ai-models/Phi-4/phi4.zip",
        ],
        "checksum": "57f641617c0051f731e07eed0de24cadacd9876e7714003f855e833b91d712d3",
        "file_checksums": {
            "added_tokens.json": "d4f2aceb0f20b71dd1f4bcc7e052e4412946bf281840b8f83d39f259571af486",
            "config.json": "49e1c05f93d43d7f17715b779a2576235b019f587285d7d914e5b05156253f62",
            "configuration_phi4mm.py": "bd9609bd47ba0c87788011e5158a8bd3e1e93165a82a9b764eb7cb048006c949",
            "generation_config.json": "757daa0d0e89171fe48fc3286341833e95b90bdb7dd3b02a2f8920fb09f85a38",
            "ignite.wav": "5f704027965a5b01f51ddfeb29727ead21d817a908d01f746e716fad7619bb0a",
            "merges.txt": "856ce61180bb689282eed6b3a6838bb1f438399be23aefe9d20eb379791fb4ad",
            "model-00001-of-00003.safetensors": "c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094",
            "model-00002-of-00003.safetensors": "b3e812c0c8acef4e7f5e34d6c9f77a7640ee4a2b93ea351921365ac62f19918d",
            "model-00003-of-00003.safetensors": "7be96b7339303752634b202d3f377bcf312a03046586eca6cea23347ace1e65a",
            "model.safetensors.index.json": "b67dbc7062e1ccf472faba4222d631dc42929c827fbdaed1ec8e34fe0601819a",
            "modeling_phi4mm.py": "e2b44eb7a66d6cc54524cee1ff9ba92d0658d435ea8900329ea0dbdb85c6439d",
            "preprocessor_config.json": "9db19b9663fb86f04c0f11d3a9b7f65a19f13d4543fb4a15bd33f82b0c92d64f",
            "processing_phi4mm.py": "84914d3e12256b4e2186e040c9830c11408468b6774f42afe85e6f8de2626d50",
            "processor_config.json": "798fc4cd09c067053af27f07f0d2b329b471c5b2eb923ceb06efa41dee660c05",
            "special_tokens_map.json": "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
            "speech-lora\\adapter_config.json": "ed252a6ae210888ee69f5720bd7e8d8261f0abfda90b18ea6452316c71336df8",
            "speech-lora\\adapter_model.safetensors": "1c2237461a4d1f9292cd128147bd3f0f70326a48d5d79c8e0f7583b26c095b30",
            "speech-lora\\added_tokens.json": "d4f2aceb0f20b71dd1f4bcc7e052e4412946bf281840b8f83d39f259571af486",
            "speech-lora\\special_tokens_map.json": "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
            "speech-lora\\tokenizer.json": "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea",
            "speech-lora\\tokenizer_config.json": "d51bf28bbdc9915926a0b1908f95cec05f3d10b71af0d27f85344d9a486b03ae",
            "speech-lora\\vocab.json": "6cb65a857824fa6615bb1782d95d882617a8bbce1da0317118586b36f39e98bd",
            "speech_conformer_encoder.py": "3742827e945732cc5deea4a95e14004da037044431a94e3f3fac26239e614e3a",
            "tokenizer.json": "4c1b9f641d4f8b7247b8d5007dd3b6a9f6a87cb5123134fe0d326f14d10c0585",
            "tokenizer_config.json": "a5da2e45718db78924ad5135a58a80b0303596acf54a1dc5c912c98436ddcaf3",
            "vision-lora\\adapter_config.json": "efbff3b978dd25e0c3abd7e71a1e9acb9a4f25e13f7430ce035654c0cb159484",
            "vision-lora\\adapter_model.safetensors": "1620b16722edf701038bf66e3cd46412c7cc5458e58df89e9f92cedb71fcbde8",
            "vision-lora\\added_tokens.json": "d4f2aceb0f20b71dd1f4bcc7e052e4412946bf281840b8f83d39f259571af486",
            "vision-lora\\special_tokens_map.json": "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
            "vision-lora\\tokenizer.json": "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea",
            "vision-lora\\tokenizer_config.json": "d51bf28bbdc9915926a0b1908f95cec05f3d10b71af0d27f85344d9a486b03ae",
            "vision-lora\\vocab.json": "6cb65a857824fa6615bb1782d95d882617a8bbce1da0317118586b36f39e98bd",
            "vision_siglip_navit.py": "7d5c053341ee9c099126fe675d5dcdc0ed5c0246f92fffdec78a1ab2f804e28d",
            "vocab.json": "6cb65a857824fa6615bb1782d95d882617a8bbce1da0317118586b36f39e98bd"
        },
        "path": "",
    },
}

class Phi4(metaclass=SingletonMeta):
    #model_path = "microsoft/Phi-4-multimodal-instruct"
    model_path = Path(Path.cwd() / ".cache" / "phi4")
    download_state = {"is_downloading": False}

    prompt_types = {
        #'transcribe': "Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
        'transcribe': "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>",
        'translate': "<|user|><|audio_1|>Translate the audio to {language_name}.<|end|><|assistant|>",
        'transcribe_translate': "<|user|><|audio_1|>Transcribe the audio to text, and then translate the audio to {language_name}. Use <sep> as a separator between the original transcript and the translation.<|end|><|assistant|>",
        'question_answering': "<|system|>You are a helpful assistant.<|end|><|user|><|audio_1|>{chat_message}<|end|><|assistant|>",
        #'question_answering': "<|user|><|audio_1|>{chat_message}<|end|><|assistant|>",
        #'chat': "<|system|>You are a helpful assistant.<|end|><|user|><|audio_1|>{chat_message}<|end|><|assistant|>",
        'text_translate': "<|system|>You are a helpful assistant.<|end|><|user|>translate \"{chat_message}\" into {language_name}. Only reply with the translation.<|end|><|assistant|>",
        'image_recognition': "<|user|><|image_1|>What text is shown in this image? Only reply with the text.<|end|><|assistant|>",
        'function_calling': '<|system|>You are a helpful assistant with some tools.{tools_definition}<|end|><|user|><|audio_1|>{chat_message}<|end|><|assistant|>',
        #'function_calling_reply': '<|system|>You are a helpful assistant. You called {function_name} with the result {function_answer} to {function_description}<|end|><|user|><|audio_1|>{chat_message}<|end|><|assistant|>',
        'function_calling_reply': '<|system|>You are a helpful assistant. {system_prompt}<|end|><|user|><|audio_1|>{chat_message}<|end|><|assistant|>',
        # tools are defined like this:
        # <|tool|>[{"name": "get_weather_updates", "description": "Fetches weather updates for a given city using the RapidAPI Weather API.", "parameters": {"city": {"description": "The name of the city for which to retrieve weather information.", "type": "str", "default": "London"}}}]<|/tool|>
    }

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
            "title": "Multimodal (Phi-4)",

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
    def load_model(self, model='small', compute_type="float32", device="cpu"):
        if self.model is not None and self.processor is not None:
            return
        self.download_model("Phi-4")

        self.processor = AutoProcessor.from_pretrained(self.model_path.resolve(), trust_remote_code=True, use_fast=False)

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
            elif self.compute_type == "float16" or self.compute_type == "bfloat16":
                attention_implementation = 'flash_attention_2'

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path.resolve(),
                trust_remote_code=True,
                device_map=self.compute_device_str,
                torch_dtype=main_torch_dtype,
                _attn_implementation=attention_implementation,
                quantization_config=quantization_config,
            )

            if quantization_config is None:
                self.model = self.model.cuda()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path.resolve(),
                trust_remote_code=True,
                device_map=self.compute_device_str,
                torch_dtype=main_torch_dtype,
                _attn_implementation='sdpa',
            ).cpu()

        self.generation_config = GenerationConfig.from_pretrained(self.model_path.resolve(), 'generation_config.json')

    @torch.no_grad()
    def transcribe(self, audio_sample, task, language='', chat_message='', image_sample=None, system_prompt='',
                   return_timestamps=False, beam_size=4) -> dict:
        self.load_model()

        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct#speech-language-format

        separator = settings.GetOption('txt_second_translation_wrap')
        separator = separator.strip(' ')

        language_code = ""
        language_name = ""
        if task == 'translate' or task == 'transcribe_translate' or task == 'text_translate':
            if language in supported_text_languages.keys():
                language_code = language
                language_name = supported_text_languages[language]

        tools_definition = ""
        tools_definitions_list = []
        if task == 'function_calling':
            # function call definition event call
            plugin_function_registration_result_list = Plugins.plugin_custom_event_call_all('plugin_llm_function_registration', {'model': 'phi4', 'task': task})
            print("plugin_function_registration_result_list: ")
            print(plugin_function_registration_result_list)
            if plugin_function_registration_result_list is not None and plugin_function_registration_result_list:
                for plugin_function_registration_result in plugin_function_registration_result_list:
                    if plugin_function_registration_result is not None:
                        tools_definitions_list.append(json.dumps(plugin_function_registration_result['tool_definition']))
                if tools_definitions_list:
                    tools_definition = '<|tool|>[' + ','.join(tools_definitions_list) + ']<|/tool|>'

        print("tools_definition: " + str(tools_definition))

        prompt = self.prompt_types['transcribe']
        if task in self.prompt_types.keys():
            prompt = self.prompt_types[task].format(language_code=language_code, language_name=language_name, chat_message=chat_message, system_prompt=system_prompt, tools_definition=tools_definition)

        if audio_sample is None:
            # remove the <|audio_*|> from the prompt if no audio sample is provided
            prompt = re.sub(r'<\|audio_\d+\|>', '', prompt)
        if image_sample is None:
            # remove the <|image_*|> from the prompt if no image is provided
            prompt = re.sub(r'<\|image_\d+\|>', '', prompt)

        response_dict = {
            'text': '',
            'type': '',
            'language': '',
        }

        if self.model is not None and self.processor is not None:
            if audio_sample is not None:
                inputs = self.processor(text=prompt, audios=[(audio_sample,16_000)], return_tensors='pt').to(self.compute_device_str)
            elif image_sample is not None:
                inputs = self.processor(text=prompt, images=image_sample, return_tensors='pt').to(self.compute_device_str)
            else:
                inputs = self.processor(text=prompt, return_tensors='pt').to(self.compute_device_str)

            # fix error = TypeError: bad operand type for unary -: 'NoneType' (possibly due to wrong flash-attn version)
            inputs["num_logits_to_keep"] = torch.tensor([50], device=self.compute_device_str)
            inputs = {k: v for k, v in inputs.items() if v is not None and v.numel() > 0}

            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                generation_config=self.generation_config,
                num_beams=1
            )

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]

            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            transcript = None
            translation = None

            if task == 'transcribe_translate':
                if "<sep>" in response:
                    # split the response by <sep>
                    response_parts = response.split("<sep>")
                    transcript = response_parts[0].strip()
                    translation = response_parts[1].strip()

            response = response.replace("<sep>", separator)

            response_dict = {
                'text': response,
                'type': task,
                'language': language_code
            }

            if translation is not None:
                response_dict["txt_translation"] = translation
                response_dict["txt_translation_target"] = language_code
            if transcript is not None:
                response_dict["text"] = transcript
                response_dict["language"] = ''


            if task == 'question_answering' or task == 'chat' or task == 'function_calling':
                response_dict['llm_answer'] = response
                response_dict['text'] = response
                response_dict['type'] = 'llm_answer'

            # function event call
            if tools_definition != "" and response is not None:
                response = re.sub(r'<\|tool_call\|>', '', response)
                if response == "":
                    return response_dict
                try :
                    tool_call = json.loads(response)[0]
                except :
                    try:
                        json_part = re.search(r'\[.*]|\{.*}', response, re.DOTALL).group(0)
                        tool_call = json.loads(json_part)
                        if isinstance(tool_call, list):
                            tool_call = tool_call[0]
                    except :
                        print("Invalid tool call response 1: " + str(response))
                        return response_dict

                if "name" not in tool_call or ("arguments" not in tool_call and "parameters" not in tool_call):
                    print("Invalid tool call response 2: " + str(response))
                    return response_dict

                function_name = tool_call["name"]
                if "arguments" not in tool_call:
                    if "parameters" in tool_call:
                        tool_call["arguments"] = tool_call["parameters"]
                    else:
                        print("Invalid tool call response 3: " + str(response))
                        return response_dict

                arguments = tool_call["arguments"]

                plugin_function_call_result = Plugins.plugin_custom_event_call('plugin_llm_function_process_'+function_name, {
                    'model': 'phi4', 'task': task, 'response': response, 'function_name': function_name, 'arguments': arguments
                })
                print("cccc:")
                print(plugin_function_call_result)
                if plugin_function_call_result is not None:
                    # send function call result back to LLM to get the final answer
                    if 'function_calling_reply' in self.prompt_types.keys() and 'reply_prompt' in plugin_function_call_result:
                        function_reply_prompt = plugin_function_call_result["reply_prompt"]
                        function_call_reply_obj = self.transcribe(audio_sample, 'function_calling_reply', language=language, chat_message=chat_message, image_sample=image_sample, system_prompt=function_reply_prompt,)
                        print("function_call_reply_obj:")
                        print(function_call_reply_obj)
                        plugin_function_call_result["text"] = function_call_reply_obj['text']
                        plugin_function_call_result["llm_answer"] = function_call_reply_obj['text']

                    response_dict = plugin_function_call_result

        return response_dict

    def run_image_processing_from_image(self, image_src, src_languages=None):
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

        response = self.transcribe(None, 'image_recognition', image_sample=image, language=src_languages)
        result_lines = response['text'].split("\n")
        return result_lines, image, None
