import torch
from omegaconf import OmegaConf
import io
from pathlib import Path
import os

from functools import partial

import Plugins
import audio_tools
import downloader
import websocket
import settings
from scipy.io.wavfile import write
import re
import num2words

tts = None
failed = False

cache_path = Path(Path.cwd() / ".cache" / "silero-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

#  https://github.com/snakers4/silero-models#standalone-use


tts_fallback_server = {
    "urls": [
        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/silero-tts.zip",
        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/silero-tts.zip",
        "https://s3.libs.space:9000/ai-models/silero/silero-tts.zip"
    ],
    "sha256": "5e486c77d9295e03c4a026f8edfff397672dc49718f4cd20fd3a7abeec6d2451",
}
tts_models_fallback_server = {
    "v4_ru": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_ru.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_ru.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_ru.pt",
        ],
        "sha256": "896ab96347d5bd781ab97959d4fd6885620e5aab52405d3445626eb7c1414b00",
    },
    "v3_1_ru": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_1_ru.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_1_ru.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_1_ru.pt",
        ],
        "sha256": "cf60b47ec8a9c31046021d2d14b962ea56b8a5bf7061c98accaaaca428522f85",
    },
    "ru_v3": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/ru_v3.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/ru_v3.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/ru_v3.pt",
        ],
        "sha256": "bf2bcab8e814edb17569503b23bd74e8cc8f584b0d2f7c7e08e2720cc48dc08c",
    },
    "v3_de": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_de.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_de.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_de.pt",
        ],
        "sha256": "2e22f38619e1d1da96d963bda5fab6d53843e8837438cb5a45dc376882b0354b",
    },
    "v3_en": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_en.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_en.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_en.pt",
        ],
        "sha256": "02b71034d9f13bc4001195017bac9db1c6bb6115e03fea52983e8abcff13b665",
    },
    "v3_en_indic": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_en_indic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_en_indic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_en_indic.pt",
        ],
        "sha256": "8ebf6b8bc4a762117e5f8d9a6ba30ffcbb65eb669f57cecd6954b0f563095429",
    },
    "v3_es": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_es.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_es.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_es.pt",
        ],
        "sha256": "36206add75fb89d0be16d5ce306ba7a896c6fa88bab7e3247403f4f4a520eced",
    },
    "v3_fr": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_fr.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_fr.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_fr.pt",
        ],
        "sha256": "02ed062cfff1c7097324929ca05c455a25d4f610fd14d51b89483126e50f15cb",
    },
    "v4_ua": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_ua.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_ua.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_ua.pt",
        ],
        "sha256": "ee14ace1b9ef79ab6af53cf14fdba17d80de209ee6c34dc69efc65a5a5458165",
    },
    "v3_ua": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_ua.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_ua.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_ua.pt",
        ],
        "sha256": "025c53797e730142816c9ce817518977c29d7a75adefece9f3c707a4f4b569cb",
    },
    "v4_indic": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_indic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_indic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_indic.pt",
        ],
        "sha256": "8c0d0055340a9789a7ff8e5f7610bbc8d82355e577e483acb8a1fe4f2df0caa6",
    },
    "v3_indic": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_indic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_indic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_indic.pt",
        ],
        "sha256": "f82129e01d4ccdfb6044ad642224be756c754dd0d82056971ff140ff7f60f87f",
    },
    "v3_tt": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_tt.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_tt.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_tt.pt",
        ],
        "sha256": "368c8f55e6de1b54dc5a393f0f5bcd328f84b3d544ac6f8b9654fc23730e925d",
    },
    "v4_uz": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_uz.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_uz.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_uz.pt",
        ],
        "sha256": "46c7977beccf2f3c9f730de281f8efefe60ee8f293a2047e89aebe567b3ed4d7",
    },
    "v3_uz": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_uz.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_uz.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_uz.pt",
        ],
        "sha256": "cbd93dca034adb84c3f914709e7ad4f5936b3594282ea200d3dc97758f6a56ce",
    },
    "v3_xal": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v3_xal.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v3_xal.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v3_xal.pt",
        ],
        "sha256": "fcababc14c6dbbffb14d04e490e4d2d85087f4aa42b2ae9d33f147cd4b868b76",
    },
    "v4_cyrillic": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/tts_models/v4_cyrillic.pt",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/tts_models/v4_cyrillic.pt",
            "https://s3.libs.space:9000/ai-models/silero/tts_models/v4_cyrillic.pt",
        ],
        "sha256": "5e3862319e13883ea105cd4db835273c7febde62ff82d98d1ccf596607f8673f",
    },
}


def is_inside_xml_tag(match, text):
    open_tag_pos = text.rfind('<', 0, match.start())
    close_tag_pos = text.rfind('>', 0, match.start())
    return open_tag_pos > close_tag_pos


def replace_numbers(match, lang, text):
    if is_inside_xml_tag(match, text):
        return match.group(0)
    else:
        return num2words.num2words(int(match.group(0)), lang=lang)


class Silero:
    lang = 'en'
    model_id = 'v3_en'
    model = None
    sample_rate = 48000
    speaker = 'random'
    models = []
    device = "cpu"  # cpu, cuda or direct-ml
    rate = ""
    pitch = ""

    last_speaker = None
    last_voice = str(Path(voices_path / "last_voice.pt").resolve())

    def __init__(self):
        global failed
        self.device = "cuda" if settings.GetOption("tts_ai_device") == "cuda" or settings.GetOption(
            "tts_ai_device") == "auto" else "cpu"
        # if cuda is not available, use cpu
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU for TTS Model")
            self.device = "cpu"

        models_config_file = str(Path(cache_path / 'latest_silero_models.yml').resolve())

        websocket.set_loading_state("tts_loading", True)
        try:
            torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                           models_config_file,
                                           progress=False)
        except:
            print("could not load latest TTS models file. trying to fetch from fallback server")
            try:
                if not Path(cache_path / "latest_silero_models.yml").is_file():
                    downloader.download_extract(tts_fallback_server["urls"],
                                                str(Path(cache_path).resolve()),
                                                tts_fallback_server["sha256"],
                                                alt_fallback=False,
                                                force_non_ui_dl=True,
                                                fallback_extract_func=downloader.extract_zip,
                                                fallback_extract_func_args=(
                                                    str(Path(cache_path / "silero-tts.zip")),
                                                    str(Path(cache_path).resolve()),
                                                ),
                                                title="Silero TTS", extract_format="zip")
            except Exception as e:
                print("could not load latest TTS models file. using existing offline file.")
                print(e)
                failed = True

        self.models = OmegaConf.load(models_config_file)
        websocket.set_loading_state("tts_loading", False)

    def list_languages(self):
        return list(self.models.tts_models.keys())

    def list_models(self):
        model_list = {}
        available_languages = self.list_languages()
        for lang in available_languages:
            _models = list(self.models.tts_models.get(lang).keys())
            model_list[lang] = _models
        return model_list

    def list_models_indexed(self):
        model_list = self.list_models()
        return tuple([{"language": language, "models": models} for language, models in model_list.items()])

    def list_voices(self):
        if self.model is None or not hasattr(self.model, 'speakers'):
            return []
        speaker_list = self.model.speakers
        speaker_list.append('last')
        return speaker_list

    def set_language(self, lang):
        self.lang = lang

    def set_model(self, model_id):
        self.model_id = model_id

    def set_rate(self, rate):
        self.rate = rate

    def set_pitch(self, pitch):
        self.pitch = pitch

    def _load_model(self, repo_or_dir, model, source='github', trust_repo=None, verbose=False, skip_validation=False,
                    fallback_local_dir=None):
        try:
            self.model, _ = torch.hub.load(trust_repo=trust_repo, skip_validation=skip_validation,
                                           source=source,
                                           repo_or_dir=repo_or_dir,
                                           model=model,
                                           language=self.lang,
                                           speaker=self.model_id,
                                           verbose=verbose)
        except Exception as e:
            print(e)
            try:
                self.model, _ = torch.hub.load(trust_repo=trust_repo, skip_validation=skip_validation,
                                               source="local", model=model,
                                               language=self.lang,
                                               speaker=self.model_id,
                                               repo_or_dir=fallback_local_dir,
                                               verbose=verbose)
            except Exception as e:
                print("Error loading TTS model. trying to load from fallback server...")
                print(e)

                try:
                    if not Path(cache_path / "snakers4_silero-models_master" / "src" / "silero" / "silero.py").is_file():
                        downloader.download_extract(tts_fallback_server["urls"],
                                                    str(Path(cache_path).resolve()),
                                                    tts_fallback_server["sha256"],
                                                    alt_fallback=False,
                                                    force_non_ui_dl=True,
                                                    fallback_extract_func=downloader.extract_zip,
                                                    fallback_extract_func_args=(
                                                        str(Path(cache_path / "silero-tts.zip")),
                                                        str(Path(cache_path).resolve()),
                                                    ),
                                                    title="Silero TTS", extract_format="zip")

                    if not Path(cache_path / "snakers4_silero-models_master" / "src" / "silero" / "model" / (self.model_id+".pt")).is_file():
                        model_path = Path(cache_path / "snakers4_silero-models_master" / "src" / "silero" / "model")
                        os.makedirs(model_path, exist_ok=True)
                        downloader.download_extract(tts_models_fallback_server[self.model_id]["urls"],
                                                    str(model_path.resolve()),
                                                    tts_models_fallback_server[self.model_id]["sha256"],
                                                    alt_fallback=False,
                                                    force_non_ui_dl=True,
                                                    title="Silero TTS Language " + self.model_id, extract_format="none")

                    self.model, _ = torch.hub.load(trust_repo=True, skip_validation=True,
                                                   source="local", model=model,
                                                   language=self.lang,
                                                   speaker=self.model_id,
                                                   repo_or_dir=fallback_local_dir,
                                                   verbose=verbose)
                except Exception as e:
                    print("Error loading tts model.")
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

        # set cache path
        torch.hub.set_dir(str(Path(cache_path).resolve()))

        # load model
        load_error = self._load_model(trust_repo=True, skip_validation=True,
                                      repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      fallback_local_dir=str(
                                          Path(cache_path / "snakers4_silero-models_master").resolve()))

        self.model.to(device)
        if self.device == "cpu":
            torch.set_num_threads(4)

        return load_error

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

    def tts(self, text):
        voice_path = None
        if settings.GetOption('tts_voice') == 'last':
            voice_path = self.last_voice
            self.speaker = 'random'
        else:
            self.speaker = settings.GetOption('tts_voice')

        tts_volume = settings.GetOption("tts_volume")

        # Try to load model repo from GitHub or locally
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
            plugin_audio = Plugins.plugin_custom_event_call('silero_tts_after_audio', {'audio': audio})
            if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
                audio = plugin_audio['audio']

        except Exception as e:
            print(e)
            return None, None

        return audio, self.sample_rate

    def play_audio(self, audio, device=None):
        source_sample_rate = 24000
        source_channels = 1

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
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=1,
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
