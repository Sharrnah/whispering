from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

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

class Phi4(metaclass=SingletonMeta):
    #model_cache_path = "microsoft/Phi-4-multimodal-instruct"
    model_cache_path = Path(Path.cwd() / ".cache" / "phi4")

    prompt_types = {
        #'transcribe': "Based on the attached audio, generate a comprehensive text transcription of the spoken content.",
        'transcribe': "Transcribe the audio clip into text.",
        'translate': "Translate the audio to {language}.",
        'transcribe_translate': "Transcribe the audio to text, and then translate the audio to {language}. Use <sep> as a separator between the original transcript and the translation.",
        'question_answering': "",
    }

    compute_type = "float16"
    compute_device = "cpu"
    compute_device_str = "cpu"

    processor = None
    model = None
    generation_config = None

    def __init__(self, compute_type="float32", device="cpu"):
        self.load_model()

    @staticmethod
    def get_languages():
        return tuple([{"code": code, "name": language} for code, language in supported_text_languages.items()])

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

    def load_model(self, model='small', compute_type="float32", device="cpu"):
        self.processor = AutoProcessor.from_pretrained(self.model_cache_path.resolve(), trust_remote_code=True, use_fast=False)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cache_path.resolve(),
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).cuda()

        #gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

        #self.generation_config = GenerationConfig.from_pretrained(self.model_cache_path, 'generation_config.json')
        self.generation_config = GenerationConfig.from_pretrained(self.model_cache_path.resolve(), 'generation_config.json')

    def transcribe(self, audio_sample, task, language='',
                   return_timestamps=False, beam_size=4) -> dict:
        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct#speech-language-format

        separator = settings.GetOption('txt_second_translation_wrap')
        separator = separator.strip(' ')

        language_code = ""
        language_name = ""
        if task == 'translate' or task == 'transcribe_translate':
            if language in supported_text_languages.keys():
                language_code = language
                language_name = supported_text_languages[language]

        speech_prompt = self.prompt_types['transcribe']
        if task in self.prompt_types.keys():
            speech_prompt = self.prompt_types[task].format(language=language_name)

        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'

        prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

        response_dict = {
            'text': '',
            'type': '',
            'language': '',
        }

        if self.model is not None and self.processor is not None:
            inputs = self.processor(text=prompt, audios=[(audio_sample,16_000)], return_tensors='pt').to('cuda:0')

            # fix error = TypeError: bad operand type for unary -: 'NoneType' (possibly due to wrong flash-attn version)
            inputs["num_logits_to_keep"] = torch.tensor([50], device='cuda:0')
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

            # replace <sep> with separator
            response = response.replace("<sep>", separator)

            response_dict = {
                'text': response,
                'type': task,
                'language': language_code
            }

            if task == 'question_answering':
                response_dict['llm_answer'] = response

        return response_dict
