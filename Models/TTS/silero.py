import torch
from omegaconf import OmegaConf
import io
from pathlib import Path
import os

from functools import partial

import audio_tools
import websocket
import settings
from scipy.io.wavfile import write
import re
import num2words

tts = None

cache_path = Path(Path.cwd() / ".cache" / "silero-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)


#  https://github.com/snakers4/silero-models#standalone-use

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
    device = "cpu"  # cpu or cuda
    rate = ""
    pitch = ""

    last_speaker = None
    last_voice = str(Path(voices_path / "last_voice.pt").resolve())

    def __init__(self):
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
            print("could not load latest TTS models file. using existing offline file.")

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
        except:
            try:
                self._load_model(trust_repo=trust_repo, skip_validation=skip_validation,
                                 source="local",
                                 repo_or_dir=fallback_local_dir,
                                 model=model,
                                 verbose=verbose)
            except:
                print("Error loading TTS model")
                return False

        return True

    def load(self):
        if len(settings.GetOption('tts_model')) == 2:
            self.set_language(settings.GetOption('tts_model')[0])
            self.set_model(settings.GetOption('tts_model')[1])

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
        except Exception as e:
            return None, None

        return audio, self.sample_rate

    def play_audio(self, audio, device=None):
        source_sample_rate = 24000
        source_is_mono = False

        if device is None:
            device = settings.GetOption("device_default_out_index")

        secondary_audio_device = None
        if settings.GetOption("tts_use_secondary_playback") and (
                (settings.GetOption("tts_secondary_playback_device") == -1 and device != settings.GetOption("device_default_out_index")) or
                (settings.GetOption("tts_secondary_playback_device") > -1 and device != settings.GetOption("tts_secondary_playback_device"))):
            secondary_audio_device = settings.GetOption("tts_secondary_playback_device")
            if secondary_audio_device == -1:
                secondary_audio_device = settings.GetOption("device_default_out_index")

        # play audio tensor
        audio_tools.play_audio(audio, device,
                               source_sample_rate=source_sample_rate,
                               audio_device_channel_num=2,
                               target_channels=2,
                               is_mono=source_is_mono,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=2,
                               secondary_device=secondary_audio_device
                               )

    def return_wav_file_binary(self, audio):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to wav file
        buff = io.BytesIO()
        write(buff, self.sample_rate, np_arr)

        return buff.read()


def init():
    global tts
    if settings.GetOption("tts_enabled") and tts is None:
        tts = Silero()
        return True
    else:
        if tts is not None:
            return True
        else:
            return False
