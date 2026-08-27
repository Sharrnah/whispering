import torch
from torch.package import PackageImporter
import io
from pathlib import Path
import os

from functools import partial

import Plugins
import audio_tools
import downloader
import settings
from scipy.io.wavfile import write
import re
import num2words

from Models.Singleton import SingletonMeta

tts = None
failed = False

cache_path = Path(Path.cwd() / ".cache" / "silero-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

# The mirrored files are self-contained torch.package archives. Loading them
# directly avoids Silero's remote torch.hub/model-catalog download path. Keep
# this manifest limited to application-controlled mirrors and expose only entries
# that have a pinned checksum here.
SILERO_TTS_MODELS = {
    "v4_ru": {
        "language": "ru",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_ru.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_ru.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_ru.pt",
        ],
        "sha256": "896ab96347d5bd781ab97959d4fd6885620e5aab52405d3445626eb7c1414b00",
    },
    "v3_1_ru": {
        "language": "ru",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_1_ru.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_1_ru.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_1_ru.pt",
        ],
        "sha256": "cf60b47ec8a9c31046021d2d14b962ea56b8a5bf7061c98accaaaca428522f85",
    },
    "ru_v3": {
        "language": "ru",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/ru_v3.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/ru_v3.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/ru_v3.pt",
        ],
        "sha256": "bf2bcab8e814edb17569503b23bd74e8cc8f584b0d2f7c7e08e2720cc48dc08c",
    },
    "v3_de": {
        "language": "de",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_de.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_de.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_de.pt",
        ],
        "sha256": "2e22f38619e1d1da96d963bda5fab6d53843e8837438cb5a45dc376882b0354b",
    },
    "v3_en": {
        "language": "en",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_en.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_en.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_en.pt",
        ],
        "sha256": "02b71034d9f13bc4001195017bac9db1c6bb6115e03fea52983e8abcff13b665",
    },
    "v3_en_indic": {
        "language": "en",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_en_indic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_en_indic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_en_indic.pt",
        ],
        "sha256": "8ebf6b8bc4a762117e5f8d9a6ba30ffcbb65eb669f57cecd6954b0f563095429",
    },
    "v3_es": {
        "language": "es",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_es.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_es.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_es.pt",
        ],
        "sha256": "36206add75fb89d0be16d5ce306ba7a896c6fa88bab7e3247403f4f4a520eced",
    },
    "v3_fr": {
        "language": "fr",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_fr.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_fr.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_fr.pt",
        ],
        "sha256": "02ed062cfff1c7097324929ca05c455a25d4f610fd14d51b89483126e50f15cb",
    },
    "v4_ua": {
        "language": "ua",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_ua.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_ua.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_ua.pt",
        ],
        "sha256": "ee14ace1b9ef79ab6af53cf14fdba17d80de209ee6c34dc69efc65a5a5458165",
    },
    "v3_ua": {
        "language": "ua",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_ua.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_ua.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_ua.pt",
        ],
        "sha256": "025c53797e730142816c9ce817518977c29d7a75adefece9f3c707a4f4b569cb",
    },
    "v4_indic": {
        "language": "indic",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_indic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_indic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_indic.pt",
        ],
        "sha256": "8c0d0055340a9789a7ff8e5f7610bbc8d82355e577e483acb8a1fe4f2df0caa6",
    },
    "v3_indic": {
        "language": "indic",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_indic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_indic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_indic.pt",
        ],
        "sha256": "f82129e01d4ccdfb6044ad642224be756c754dd0d82056971ff140ff7f60f87f",
    },
    "v3_tt": {
        "language": "tt",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_tt.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_tt.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_tt.pt",
        ],
        "sha256": "368c8f55e6de1b54dc5a393f0f5bcd328f84b3d544ac6f8b9654fc23730e925d",
    },
    "v4_uz": {
        "language": "uz",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_uz.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_uz.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_uz.pt",
        ],
        "sha256": "46c7977beccf2f3c9f730de281f8efefe60ee8f293a2047e89aebe567b3ed4d7",
    },
    "v3_uz": {
        "language": "uz",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_uz.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_uz.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_uz.pt",
        ],
        "sha256": "cbd93dca034adb84c3f914709e7ad4f5936b3594282ea200d3dc97758f6a56ce",
    },
    "v3_xal": {
        "language": "xal",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_xal.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_xal.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_xal.pt",
        ],
        "sha256": "fcababc14c6dbbffb14d04e490e4d2d85087f4aa42b2ae9d33f147cd4b868b76",
    },
    "v4_cyrillic": {
        "language": "cyrillic",
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_cyrillic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_cyrillic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_cyrillic.pt",
        ],
        "sha256": "5e3862319e13883ea105cd4db835273c7febde62ff82d98d1ccf596607f8673f",
    },
}

SILERO_LANGUAGE_ORDER = (
    "ru", "en", "de", "es", "fr", "xal", "tt", "uz", "ua", "indic", "cyrillic"
)
SILERO_MODEL_CACHE_RELATIVE_PATH = (
    Path("snakers4_silero-models_master") / "src" / "silero" / "model"
)


def is_inside_xml_tag(match, text):
    open_tag_pos = text.rfind('<', 0, match.start())
    close_tag_pos = text.rfind('>', 0, match.start())
    return open_tag_pos > close_tag_pos


def replace_numbers(match, lang, text):
    if is_inside_xml_tag(match, text):
        return match.group(0)
    else:
        return num2words.num2words(int(match.group(0)), lang=lang)


class Silero(metaclass=SingletonMeta):
    lang = 'en'
    model_id = 'v3_en'
    model = None
    sample_rate = 48000
    speaker = 'random'
    device = "cpu"  # cpu, cuda or direct-ml
    rate = ""
    pitch = ""

    last_speaker = None
    last_voice = str(Path(voices_path / "last_voice.pt").resolve())

    last_generation = {"audio": None, "sample_rate": None}

    def __init__(self):
        self._verified_model_files = set()
        self._package_importer = None
        self.device = "cuda" if settings.GetOption("tts_ai_device") == "cuda" or settings.GetOption(
            "tts_ai_device") == "auto" else "cpu"
        # if cuda is not available, use cpu
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU for TTS Model")
            self.device = "cpu"

    def list_languages(self):
        models_by_language = self.list_models()
        return list(models_by_language.keys())

    def list_models(self):
        model_list = {language: [] for language in SILERO_LANGUAGE_ORDER}
        for model_id, model_info in SILERO_TTS_MODELS.items():
            language = model_info["language"]
            model_list.setdefault(language, []).append(model_id)
        return {language: models for language, models in model_list.items() if models}

    def list_models_indexed(self):
        model_list = self.list_models()
        return tuple([{"language": language, "models": models} for language, models in model_list.items()])

    def list_voices(self):
        if self.model is None or not hasattr(self.model, 'speakers'):
            return []
        speaker_list = self.model.speakers
        speaker_list.append('last')

        # build json list
        voice_list_dict = []
        for speaker in speaker_list:
            voice_list_dict.append({"name": speaker, "value": speaker})
        return voice_list_dict

    def set_language(self, lang):
        self.lang = lang

    def set_model(self, model_id):
        self.model_id = model_id

    def set_rate(self, rate):
        self.rate = rate

    def set_pitch(self, pitch):
        self.pitch = pitch

    def _ensure_local_model(self):
        model_info = SILERO_TTS_MODELS.get(self.model_id)
        if model_info is None:
            raise ValueError(
                f"Silero TTS model '{self.model_id}' is not available from the application mirrors."
            )
        if model_info["language"] != self.lang:
            raise ValueError(
                f"Silero TTS model '{self.model_id}' belongs to language "
                f"'{model_info['language']}', not '{self.lang}'."
            )

        model_dir = Path(cache_path) / SILERO_MODEL_CACHE_RELATIVE_PATH
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filename = os.path.basename(model_info["urls"][0])
        model_path = model_dir / model_filename
        expected_sha256 = model_info["sha256"].lower()
        verification_key = (str(model_path.resolve()), expected_sha256)

        if model_path.is_file() and verification_key not in self._verified_model_files:
            actual_sha256 = downloader.sha256_checksum(model_path).lower()
            if actual_sha256 != expected_sha256:
                print(
                    f"Cached Silero TTS model '{self.model_id}' failed its SHA-256 check; "
                    "downloading a verified copy."
                )
                model_path.unlink()
            else:
                self._verified_model_files.add(verification_key)

        if not model_path.is_file():
            download_success = downloader.download_extract(
                model_info["urls"],
                str(model_dir.resolve()),
                model_info["sha256"],
                alt_fallback=False,
                force_non_ui_dl=True,
                title="Silero TTS Language " + self.model_id,
                extract_format="none",
            )
            if not download_success or not model_path.is_file():
                raise RuntimeError(
                    f"Could not download Silero TTS model '{self.model_id}' from the application mirrors."
                )

            actual_sha256 = downloader.sha256_checksum(model_path).lower()
            if actual_sha256 != expected_sha256:
                model_path.unlink()
                raise RuntimeError(
                    f"Downloaded Silero TTS model '{self.model_id}' failed its SHA-256 check."
                )
            self._verified_model_files.add(verification_key)

        return model_path

    def _load_model(self):
        try:
            model_path = self._ensure_local_model()
            self._package_importer = PackageImporter(str(model_path))
            self.model = self._package_importer.load_pickle("tts_models", "model")
        except Exception as e:
            self.model = None
            self._package_importer = None
            print("Error loading Silero TTS model from the local verified cache.")
            print(e)
            return False
        return True

    def load(self):
        if len(settings.GetOption('tts_model')) == 2:
            self.set_language(settings.GetOption('tts_model')[0])
            self.set_model(settings.GetOption('tts_model')[1])

        if self.device.startswith("direct-ml"):
            device_id = 0
            device_id_split = self.device.split(":")
            if len(device_id_split) > 1:
                device_id = int(device_id_split[1])
            import torch_directml
            device = torch_directml.device(device_id)
        else:
            device = torch.device(self.device)

        if not self._load_model():
            return False

        self.model.to(device)
        if self.device == "cpu":
            torch.set_num_threads(4)

        print(f"Model silero_tts loaded successfully.")

        return True

    def save_voice(self, voice_path=last_voice):
        if settings.GetOption('tts_voice') == 'random':
            self.model.save_random_voice(voice_path)
            self.last_voice = voice_path
        else:
            print("No generated random voice to save")

    def _preprocess_tts(self, text):
        # replace all numbers with their word representations
        replace_numbers_with_lang = partial(replace_numbers, lang=self.lang, text=text)
        text = re.sub(r"\d+", replace_numbers_with_lang, text)

        # replace parts the tts has trouble with
        text = text.replace("...", ".")

        if not text.endswith(".") and not text.endswith("!") and not text.endswith("?") and not text.endswith(
                ",") and not text.endswith(";") and not text.endswith(
            ":") and not text.endswith(")") and not text.endswith("]"):
            text += "."

        return text

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def tts(self, text):
        voice_path = None
        if settings.GetOption('tts_voice') == 'last':
            voice_path = self.last_voice
            self.speaker = 'random'
        else:
            self.speaker = settings.GetOption('tts_voice')

        tts_volume = settings.GetOption("tts_volume")

        # Load the checksum-verified model package from the managed local cache.
        if not self.load():
            return None, None

        # preprocess text
        text = self._preprocess_tts(text)

        # configure prosody tag
        self.set_rate(settings.GetOption('tts_prosody_rate'))
        self.set_pitch(settings.GetOption('tts_prosody_pitch'))
        prosody_tag = ""
        if self.rate != "" and self.pitch != "":
            prosody_tag = f'<prosody rate="{self.rate}" pitch="{self.pitch}">'
        elif self.rate != "":
            prosody_tag = f'<prosody rate="{self.rate}">'
        elif self.pitch != "":
            prosody_tag = f'<prosody pitch="{self.pitch}">'

        if prosody_tag != "":
            text = f"{prosody_tag}{text}</prosody>"

        # Try to generate tts
        try:
            audio = self.model.apply_tts(ssml_text="<speak>" + text + "</speak>",
                                         speaker=self.speaker,
                                         sample_rate=self.sample_rate,
                                         voice_path=voice_path,
                                         put_accent=True,
                                         put_yo=True)

            # change volume
            if tts_volume != 1.0:
                audio = audio_tools.change_volume(audio, tts_volume)

            # call custom plugin event method
            plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': audio, 'sample_rate': self.sample_rate})
            if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                audio = plugin_audio['audio']

        except Exception as e:
            print(e)
            return None, None

        # save last generation in memory
        self.last_generation = {"audio": audio, "sample_rate": self.sample_rate}

        return audio, self.sample_rate

    def play_audio(self, audio, device=None):
        source_sample_rate = 24000
        source_channels = 2

        if device is None:
            device = settings.GetOption("device_default_out_index")

        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and device != settings.GetOption("tts_secondary_playback_device"))):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        allow_overlapping_audio = settings.GetOption("tts_allow_overlapping_audio")

        # play audio tensor
        audio_tools.play_audio(audio, device,
                               source_sample_rate=source_sample_rate,
                               audio_device_channel_num=2,
                               target_channels=2,
                               input_channels=source_channels,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=2,
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to wav file
        buff = io.BytesIO()
        write(buff, self.sample_rate, np_arr)

        return buff.read()
