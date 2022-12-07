import torch
from omegaconf import OmegaConf
import pyaudio
import wave
import io
from pathlib import Path
import os
from pydub import AudioSegment
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


class Silero:
    lang = 'en'
    model_id = 'v3_en'
    model = None
    sample_rate = 48000
    speaker = 'random'
    models = []
    device = "cuda" if settings.GetOption("tts_ai_device") == "cuda" or settings.GetOption("tts_ai_device") == "auto" else "cpu"

    last_speaker = None
    last_voice = str(Path(voices_path / "last_voice.pt").resolve())

    def __init__(self):
        models_config_file = str(Path(cache_path / 'latest_silero_models.yml').resolve())

        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                       models_config_file,
                                       progress=False)
        self.models = OmegaConf.load(models_config_file)

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

    def _load_model(self, repo_or_dir, model, source='github', trust_repo=None, verbose=False, skip_validation=False, fallback_local_dir=None):
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
        self.set_language(settings.GetOption('tts_model')[0])
        self.set_model(settings.GetOption('tts_model')[1])

        device = torch.device(self.device)

        # set cache path
        torch.hub.set_dir(str(Path(cache_path).resolve()))

        # load model
        load_error = self._load_model(trust_repo=True, skip_validation=True,
                                      repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      fallback_local_dir=str(Path(cache_path / "snakers4_silero-models_master").resolve()))

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
        text = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0)), lang=self.lang), text)

        # replace parts the tts has trouble with
        text = text.replace("...", ".")

        if not text.endswith(".") or not text.endswith("!") or not text.endswith("?") or not text.endswith(",") or not text.endswith(";") or not text.endswith(":") or not text.endswith(")") or not text.endswith("]"):
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

        # Try to generate tts
        try:
            text = self._preprocess_tts(text)
            audio = self.model.apply_tts(text=text,
                                         speaker=self.speaker,
                                         sample_rate=self.sample_rate,
                                         voice_path=voice_path,
                                         put_accent=True,
                                         put_yo=True)
        except Exception as e:
            return None, None

        return audio, self.sample_rate

    @staticmethod
    def tensor_to_buffer(tensor):
        buff = io.BytesIO()
        torch.save(tensor, buff)
        buff.seek(0)
        return buff

    def play_audio(self, audio, device=None):
        buff = self._generate_wav_buffer(audio)

        # Set chunk size of 1024 samples per data frame
        chunk = 1024

        # Open the sound file
        wf = wave.open(buff, 'rb')

        # Create an interface to PortAudio
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output_device_index=device,
                        output=True)

        # Read data in chunks
        data = wf.readframes(chunk)

        # Play the sound by writing the audio data to the stream
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(chunk)

        # Close and terminate the stream
        stream.close()
        wf.close()
        p.terminate()

    def _generate_wav_buffer(self, audio):
        audio = Silero.tensor_to_buffer(audio)

        wav_file = AudioSegment.from_file(audio, format="raw", frame_rate=self.sample_rate, channels=1, sample_width=4)

        buff = io.BytesIO()
        wav_file.export(buff, format="wav")

        return buff

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
