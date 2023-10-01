import os

import scipy
from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4TConfig

from pathlib import Path
import torch

import downloader
from Models import languageClassification

from Models.Singleton import SingletonMeta

model_cache_path = Path(".cache/seamlessm4t-cache")

LANGUAGES = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cmn_Hant": "Mandarin Chinese (Traditional)",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokmål",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}

MODEL_LINKS = {
    "medium": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/seamless-m4t/medium.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/seamless-m4t/medium.zip",
            "https://s3.libs.space:9000/ai-models/seamless-m4t/medium.zip",
        ],
        "checksum": "678c6dc97899a5a34835dfb2315fdc05cff62332cd6efca4c0cfafe7a7553738"
    },
    "large": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/seamless-m4t/large.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/seamless-m4t/large.zip",
            "https://s3.libs.space:9000/ai-models/seamless-m4t/large.zip",
        ],
        "checksum": "0ba6b31c223d4cebdf865e42d04a3b29b891f3286dd0550bcc7eb5c7f410d6eb"
    },
}


class SeamlessM4T(metaclass=SingletonMeta):
    model = None
    processor = None
    device = None
    precision = torch.float32
    compute_type_name = "float32"  # just for output
    load_in_8bit = False

    def __init__(self, model='medium', compute_type="float32", device="cpu"):
        if self.model is not None and self.processor is not None:
            return

        self.compute_type_name = compute_type
        if compute_type == "float32":
            self.precision = torch.float32
        elif compute_type == "float16":
            self.precision = torch.float16
        elif compute_type == "int8_float16":
            self.precision = torch.float16
            self.load_in_8bit = True
        elif compute_type == "bfloat16":
            self.precision = torch.bfloat16
        elif compute_type == "int8_bfloat16":
            self.precision = torch.bfloat16
            self.load_in_8bit = True

        if self.device is None:
            self.device = device
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None or self.processor is None:
            self.load_model(model_size=model)

    def set_device(self, device: str):
        if device == "cuda" or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    @staticmethod
    def needs_download(model: str):
        model_path = Path(model_cache_path / model)
        pretrained_lang_model_file = Path(model_path / "pytorch_model.bin")
        if not Path(model_path).exists() or not pretrained_lang_model_file.is_file():
            return True
        return False

    @staticmethod
    def download_model(model: str):
        os.makedirs(model_cache_path, exist_ok=True)
        model_path = Path(model_cache_path / model)
        pretrained_lang_model_file = Path(model_path / "pytorch_model.bin")
        if not Path(model_path).exists() or not pretrained_lang_model_file.is_file():
            print("downloading Seamless M4T...")
            if not downloader.download_extract(MODEL_LINKS[model]["urls"],
                                           str(model_path.resolve()),
                                           MODEL_LINKS[model]["checksum"], title="Speech 2 Text (Seamless M4T)"):
                print("Model download failed")

    def load_model(self, model_size='medium'):
        self.download_model(model_size)

        model_path = Path(model_cache_path / model_size)

        configuration = SeamlessM4TConfig()

        print(f"Seamless-M4T {model_size} is Loading to {self.device} using {self.compute_type_name} precision...")
        # facebook/hf-seamless-m4t-medium
        self.processor = AutoProcessor.from_pretrained(str(model_path.resolve()),
                                                       torch_dtype=self.precision)
        self.model = SeamlessM4TModel.from_pretrained(str(model_path.resolve()),
                                                      torch_dtype=self.precision,
                                                      low_cpu_mem_usage=True,
                                                      load_in_8bit=self.load_in_8bit,
                                                      config=configuration)

        if not self.load_in_8bit:
            self.model.to(self.device)

    @staticmethod
    def get_languages():
        return tuple([{"code": code, "name": language} for code, language in LANGUAGES.items()])

    def text_cleanup(self, text):
        cleanup_texts = ["[filler/]", "[n_s/]", "[UNK]", "<unk>", "(Video)"]
        for cleanup_text in cleanup_texts:
            text = text.replace(cleanup_text, "")
        return text.strip()

    # this always translates to the target langauge
    def transcribe(self, audio_sample, source_lang=None, target_lang='eng', beam_size=5, generate_speech=False,
                   repetition_penalty: float = 1, length_penalty: float = 1, no_repeat_ngram_size: int = 0) -> dict:
        if source_lang is not None and (source_lang == '' or source_lang.lower() == 'auto'):
            source_lang = None

        #self.model.config.temperature = 1.0
        self.model.config.repetition_penalty = repetition_penalty
        self.model.config.length_penalty = length_penalty
        self.model.config.no_repeat_ngram_size = no_repeat_ngram_size

        inputs = self.processor(audios=audio_sample, src_lang=source_lang, sampling_rate=16000, return_tensors="pt")
        inputs = {name: tensor.to(dtype=self.precision).to(self.device) for name, tensor in inputs.items()}

        output_tokens = self.model.generate(**inputs, tgt_lang=target_lang,
                                            text_num_beams=beam_size, speech_do_sample=True,
                                            return_intermediate_token_ids=True,
                                            generate_speech=generate_speech,
                                            spkr_id=0
                                            )

        if generate_speech:
            scipy.io.wavfile.write("seamless_m4t_out.wav",
                                   rate=self.model.config.sampling_rate,
                                   data=output_tokens.waveform.cpu().numpy().squeeze()
                                   )

        transcription = self.processor.decode(output_tokens.sequences.tolist()[0], skip_special_tokens=True)

        transcription = self.text_cleanup(transcription)

        result = {
            'text': transcription,
            'type': "transcribe",
            'language': source_lang
        }

        return result

    def remove_writing_system_from_lang_code(self, lang_code):
        writing_systems = ["Arab", "Latn", "Ethi", "Beng", "Deva", "Cyrl", "Tibt", "Grek", "Gujr", "Hebr", "Armn", "Jpan", "Knda", "Geor", "Khmr", "Hang", "Laoo", "Mlym", "Mymr", "Orya", "Guru", "Sinh", "Taml", "Telu", "Thai", "Tfng", "Hant", "Hans"]
        # delete writing system suffix from target_lang (changes lang code in form of "__eng_Latn__" to "eng"
        for writing_system in writing_systems:
            if lang_code.endswith("_"+writing_system+"__"):
                lang_code = lang_code[:-len("_"+writing_system)].strip('__')
                break
        return lang_code

    def text_translate(self, text, source_lang='eng', target_lang='eng', beam_size=5, generate_speech=False) -> tuple:
        if source_lang == "auto":
            source_lang = languageClassification.classify(text)

        inputs = self.processor(text=text, src_lang=source_lang, sampling_rate=16000, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        output_tokens = self.model.generate(**inputs, tgt_lang=target_lang,
                                            text_num_beams=beam_size, speech_do_sample=True,
                                            return_intermediate_token_ids=True,
                                            generate_speech=generate_speech,
                                            spkr_id=0
                                            )
        translation = self.processor.decode(output_tokens.sequences.tolist()[0], skip_special_tokens=True)
        translation = self.text_cleanup(translation)
        return translation, source_lang, target_lang
