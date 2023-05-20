from faster_whisper import WhisperModel

from pathlib import Path
import os
import downloader

MODEL_LINKS = {
    "tiny": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny-ct2-fp16.zip",
            ],
            "checksum": "3c7c0512b7b881ecb4cb0693d543aed2a9178968bef255fa0ca8b880541ec789"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny-ct2.zip",
            ],
            "checksum": "18f4d5a6dbb9d27b748ee7a58ef455ff6640f230e5d64781e9cfb16181136b04"
        }
    },
    "tiny.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny.en-ct2-fp16.zip",
            ],
            "checksum": "a14fedc8e57090505ec46119d346895604f5a6b5a8a44a7a137c44169544ea99"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny.en-ct2.zip",
            ],
            "checksum": "814c670c9922574c9e0e3be8d7f616e53347ec2dee099648523e2f88ec436eec"
        }
    },
    "base": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base-ct2-fp16.zip",
            ],
            "checksum": "fa863d01b4ef07bab0467d13b33221c8e6273362078ec6268bbc6398f40c0ab4"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base-ct2.zip",
            ],
            "checksum": "e95001e10c40b57797e208f2e915e16d86bac67f204742bac2b8950e6eeb3539"
        }
    },
    "base.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base.en-ct2-fp16.zip",
            ],
            "checksum": "ec00c31ef78f035950c276ff01e5da96b4e9761bc15e872b2ec02371ac357484"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base.en-ct2.zip",
            ],
            "checksum": "5113b44b8f4fe1927f935d85326df5bbe708ab269144fc9399234f9e9b9d61d1"
        }
    },
    "small": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small-ct2-fp16.zip",
            ],
            "checksum": "9f0618523bf19dc68d99109ba319f2faba2c94ef9d063aa300115935f3d09f14"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small-ct2.zip",
            ],
            "checksum": "b887054992cf42abddad057e4b52f3ef6b1a079485244d786f1941a6fec8c02e"
        }
    },
    "small.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small.en-ct2-fp16.zip",
            ],
            "checksum": "9f0618523bf19dc68d99109ba319f2faba2c94ef9d063aa300115935f3d09f14"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small.en-ct2.zip",
            ],
            "checksum": "c7eeb56070467bfad17ec774f66ce8dfc0b601d9c2ad5f96b3e4da9331552692"
        }
    },
    "medium": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium-ct2-fp16.zip",
            ],
            "checksum": "13d2d91bdd2c3722c0592cbffca468992257eb3ddb782b1779c59091a4d91dd4"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium-ct2.zip",
            ],
            "checksum": "5682a3833f4c87ed749778a844ccc9da6d8b3e3a2fef338cf5e66b495050e2e6"
        }
    },
    "medium.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium.en-ct2-fp16.zip",
            ],
            "checksum": "13d2d91bdd2c3722c0592cbffca468992257eb3ddb782b1779c59091a4d91dd4"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium.en-ct2.zip",
            ],
            "checksum": "8bf93eb5018c44c9115b6b942f8bc518790f88c2db93920f2da1a6a1efefe002"
        }
    },
    "large-v1": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v1-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v1-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v1-ct2-fp16.zip",
            ],
            "checksum": "42ecc70522602e69fe6365ef73173bbb1178ff8fd99210b96ea9025a205014bb"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v1-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v1-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v1-ct2.zip",
            ],
            "checksum": "82bd59ee73d7b52f60de5566e8e3e429374bd2dd1bce3e2f6fc18b620dbcf0cf"
        }
    },
    "large-v2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v2-ct2-fp16.zip",
            ],
            "checksum": "2397ed6433a08d4b6968852bc1b761b488c3149a3a52f49b62b2ac60d1d5cef0"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v2-ct2.zip",
            ],
            "checksum": "c9e889f59cacfef9ebe76a1db5d80befdcf0043195c07734f6984d19e78c8253"
        }
    },
}

TOKENIZER_LINKS = {
    "normal": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tokenizer.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tokenizer.zip",
            "https://s3.libs.space:9000/ai-models/Whisper-CT2/tokenizer.zip",
        ],
        "checksum": "f6233d181a04abce6e2ba20189d5872b58ce2e14917af525a99feb5619777d7d"
    },
    "en": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tokenizer.en.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tokenizer.en.zip",
            "https://s3.libs.space:9000/ai-models/Whisper-CT2/tokenizer.en.zip",
        ],
        "checksum": "fb364e7cae84eedfd742ad116a397daa75e4eebba38f27e3f391ae4fee19afa9"
    }
}


def needs_download(model: str, compute_type: str = "float32"):
    model_cache_path = Path(".cache/whisper")
    model_path = Path(model_cache_path / (model + "-ct2"))
    if compute_type == "float16" or compute_type == "int8_float16" or compute_type == "int16" or compute_type == "int8":
        model_path = Path(model_cache_path / (model + "-ct2-fp16"))

    pretrained_lang_model_file = Path(model_path / "model.bin")

    if not model_cache_path.exists() or not Path(model_path).exists() or not pretrained_lang_model_file.is_file():
        return True

    tokenizer_file = Path(model_path / "tokenizer.json")
    if not tokenizer_file.is_file():
        return True

    return False


def download_model(model: str, compute_type: str = "float32"):
    model_cache_path = Path(".cache/whisper")
    os.makedirs(model_cache_path, exist_ok=True)
    model_path = Path(model_cache_path / (model + "-ct2"))
    if compute_type == "float16" or compute_type == "int8_float16" or compute_type == "int16" or compute_type == "int8":
        compute_type = "float16"
        model_path = Path(model_cache_path / (model + "-ct2-fp16"))

    pretrained_lang_model_file = Path(model_path / "model.bin")

    if not Path(model_path).exists() or not pretrained_lang_model_file.is_file():
        print("downloading faster-whisper...")
        if not downloader.download_extract(MODEL_LINKS[model][compute_type]["urls"],
                                           str(model_cache_path.resolve()),
                                           MODEL_LINKS[model][compute_type]["checksum"], title="Speech 2 Text (faster whisper)"):
            print("Model download failed")

    tokenizer_file = Path(model_path / "tokenizer.json")
    if not tokenizer_file.is_file() and Path(model_path).exists():
        tokenizer_type = "normal"
        if ".en" in model:
            tokenizer_type = "en"
        print("downloading tokenizer...")
        if not downloader.download_extract(TOKENIZER_LINKS[tokenizer_type]["urls"],
                                           str(model_path.resolve()),
                                           TOKENIZER_LINKS[tokenizer_type]["checksum"], title="tokenizer"):
            print("Tokenizer download failed")
    elif not Path(model_path).exists():
        print("no model downloaded for tokenizer.")


class FasterWhisper:
    model = None

    def __init__(self, model: str, device: str = "cpu", compute_type: str = "float32", cpu_threads: int = 0,
                 num_workers: int = 1):
        if self.model is None:
            self.load_model(model, device, compute_type, cpu_threads, num_workers)

    def load_model(self, model: str, device: str = "cpu", compute_type: str = "float32", cpu_threads: int = 0,
                   num_workers: int = 1):
        model_cache_path = Path(".cache/whisper")
        os.makedirs(model_cache_path, exist_ok=True)
        model_path = Path(model_cache_path / (model + "-ct2"))
        if compute_type == "float16" or compute_type == "int8_float16":
            model_path = Path(model_cache_path / (model + "-ct2-fp16"))

        print(f"faster-whisper {model} is Loading to {device} using {compute_type} precision...")
        self.model = WhisperModel(str(Path(model_path).resolve()), device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, num_workers=num_workers)

    def transcribe(self, audio_sample, task, language, condition_on_previous_text,
                   initial_prompt, logprob_threshold, no_speech_threshold,
                   temperature, beam_size) -> dict:

        result_segments, audio_info = self.model.transcribe(audio_sample, task=task,
                                                            language=language,
                                                            condition_on_previous_text=condition_on_previous_text,
                                                            initial_prompt=initial_prompt,
                                                            log_prob_threshold=logprob_threshold,
                                                            no_speech_threshold=no_speech_threshold,
                                                            temperature=temperature,
                                                            beam_size=beam_size,
                                                            without_timestamps=True
                                                            )

        result = {
            'text': " ".join([segment.text for segment in result_segments]),
            'type': task,
            'language': audio_info.language
        }

        return result
