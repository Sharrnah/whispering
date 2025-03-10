import io
import os
import threading
from pathlib import Path

import numpy as np

import Plugins
import audio_tools
import settings
from Models.Singleton import SingletonMeta

cache_path = Path(Path.cwd() / ".cache" / "zonos-tts-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

#os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(Path(cache_path / "eSpeak NG" / "libespeak-ng.dll").resolve())

from scipy.io.wavfile import write as write_wav
import torch
import torchaudio
from Models.TTS.zonos.model import Zonos
from Models.TTS.zonos.conditioning import make_cond_dict, supported_language_codes
from Models.TTS.zonos.utils import DEFAULT_DEVICE


failed = False

speed_mapping = {
    "": 15.0,        # Default speed when empty
    "x-slow": 5.0,  # Extra slow speed
    "slow": 10.0,   # Slow speed
    "medium": 15.0,  # Medium speed (default)
    "fast": 20.0,   # Fast speed
    "x-fast": 30.0   # Extra fast speed
}

model_list = {
    "Default": ["v0.1-transformer"],
}


# patching EspeakWrapper of phonemizer library.
# See https://github.com/open-mmlab/Amphion/issues/323#issuecomment-2646709006



class ZonosTTS(metaclass=SingletonMeta):
    model = None
    sample_rate = 44100
    last_generation = {"audio": None, "sample_rate": None}
    voice_list = []
    audio_streamer = None

    last_speaker_embedding = None
    last_speaker_audio = None

    compute_device = "cpu"

    stop_flag = False
    stop_flag_lock = threading.Lock()

    special_settings = {
        "language": "en-us",

        "happiness": 0.3077,
        "sadness": 0.0256,
        "disgust": 0.0256,
        "fear": 0.0256,
        "surprise": 0.0256,
        "anger": 0.0256,
        "other": 0.2564,
        "neutral": 0.3077,

        "ignore_list": [],
        "seed": -1,
    }

    def __init__(self):
        self.compute_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
        # self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=DEFAULT_DEVICE)
        model_type = "v0.1-transformer"
        #model_type = "v0.1-hybrid"
        self.model = Zonos.from_local(
            config_path=str(Path(cache_path / model_type / "config.json").resolve()),
            model_path=str(Path(cache_path / model_type / "model.safetensors").resolve()),
            device=DEFAULT_DEVICE
        )
        self.target_sample_rate = self.model.autoencoder.sampling_rate

        if not self.voice_list:
            self.update_voices()
        pass

    def stop(self):
        print("TTS Stop requested")
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def set_compute_device(self, device):
        self.compute_device_str = device
        if device is None or device == "cuda" or device == "auto" or device == "":
            self.compute_device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            #device = torch.device(self.compute_device_str)
            device = self.compute_device_str
        self.compute_device = device

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        return tuple([{"language": language, "models": models} for language, models in self.list_models().items()])

    def download_model(self, model_name):
        pass

    def load(self):
        pass

    def _get_voices(self):
        return self.voice_list

    def update_voices(self):
        # find all voices that have a .wav or .mp3 file
        voice_files = [f.stem for f in voices_path.iterdir() if f.is_file() and (f.suffix == ".wav" or f.suffix == ".mp3")]

        voice_list = []
        for voice_id in voice_files:
            wav_file = voices_path / f"{voice_id}.wav"
            mp3_file = voices_path / f"{voice_id}.mp3"

            if wav_file.exists() or mp3_file.exists():
                audio_file = wav_file if wav_file.exists() else mp3_file
                voice_list.append({"name": voice_id, "audio_filename": str(audio_file.resolve())})
        self.voice_list = voice_list

    def list_voices(self):
        self.update_voices()
        return [voice["name"] for voice in self._get_voices()]

    def get_voice_by_name(self, voice_name):
        for voice in self._get_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    @staticmethod
    def generate_random_seed():
        #min_seed = -0x8000_0000_0000_0000
        #max_seed = 0xffff_ffff_ffff_ffff
        #seed = random.randint(min_seed, max_seed)
        #if seed < 0:
        #    seed = max_seed + seed + 1
        #return seed
        return torch.randint(0, 2 ** 32 - 1, (1,)).item()

    def set_special_setting(self, special_settings):
        self.special_settings = special_settings

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        #with self.stop_flag_lock:
        #    self.stop_flag = False
        print("TTS requested Zonos TTS")
        self.set_compute_device(settings.GetOption('tts_ai_device'))

        tts_volume = settings.GetOption("tts_volume")

        if ref_audio is None:
            voice_name = settings.GetOption('tts_voice')
            selected_voice = self.get_voice_by_name(voice_name)
            if selected_voice is None:
                print("No voice selected or does not exist. Using default voice 'en_1'.")
                voice_name = "en_1"
                selected_voice = self.get_voice_by_name(voice_name)
            ref_audio = selected_voice["audio_filename"]

        # reuse speaker embedding
        if self.last_speaker_audio != ref_audio or self.last_speaker_embedding is None:
            wav, sampling_rate = torchaudio.load(ref_audio)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            self.last_speaker_audio = ref_audio
            self.last_speaker_embedding = speaker
        else:
            speaker = self.last_speaker_embedding

        seed = self.special_settings["seed"]
        # convert string to integer. If its invalid, default to -1
        try:
            seed = int(seed)
        except ValueError:
            seed = -1
        if seed <= -1:
            seed = self.generate_random_seed()
        torch.manual_seed(seed)

        print("Using seed:", seed)

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        language = self.special_settings["language"]
        emotion = [
            self.special_settings["happiness"],
            self.special_settings["sadness"],
            self.special_settings["disgust"],
            self.special_settings["fear"],
            self.special_settings["surprise"],
            self.special_settings["anger"],
            self.special_settings["other"],
            self.special_settings["neutral"],
        ]

        # fix if all emotions are either set to 1.0 or 0.0, or all non-zero emotions are set to 1.0
        if all(emotion_value == 0.0 for emotion_value in emotion) or all(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.0001 if emotion_value == 0.0 else 0.9999 for emotion_value in emotion]
        elif all(emotion_value in (0.0, 1.0) for emotion_value in emotion) and any(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.9999 if emotion_value == 1.0 else 0.0001 for emotion_value in emotion]

        unconditional_keys = self.special_settings["ignore_list"]

        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, emotion=emotion, speaking_rate=tts_speed, unconditional_keys=unconditional_keys, device=self.compute_device)
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning, batch_size=1, disable_torch_compile=True) # €todo with torch_compile Not yet supported. Probably when updating to pytorch 2.6.0 which does not support Direct-ML (yet)
        wavs = self.model.autoencoder.decode(codes).cpu()

        final_wave = wavs[0]
        # change volume
        if tts_volume != 1.0:
            final_wave = audio_tools.change_volume(wavs[0], tts_volume)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': final_wave, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            final_wave = plugin_audio['audio']

        final_wave = final_wave.unsqueeze(0)
        # save last generation in memory
        self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return final_wave, self.sample_rate

    def tts_streaming(self, text, ref_audio=None):
        #self.stop_flag = False
        print("TTS requested Zonos TTS (Streaming)")
        self.set_compute_device(settings.GetOption('tts_ai_device'))

        chunk_size = settings.GetOption("tts_streamed_chunk_size")

        self.init_audio_stream_playback()

        tts_volume = settings.GetOption("tts_volume")

        if ref_audio is None:
            voice_name = settings.GetOption('tts_voice')
            selected_voice = self.get_voice_by_name(voice_name)
            if selected_voice is None:
                print("No voice selected or does not exist. Using default voice 'en_1'.")
                voice_name = "en_1"
                selected_voice = self.get_voice_by_name(voice_name)
            ref_audio = selected_voice["audio_filename"]

        # reuse speaker embedding
        if self.last_speaker_audio != ref_audio or self.last_speaker_embedding is None:
            wav, sampling_rate = torchaudio.load(ref_audio)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            self.last_speaker_audio = ref_audio
            self.last_speaker_embedding = speaker
        else:
            speaker = self.last_speaker_embedding

        seed = self.generate_random_seed()
        torch.manual_seed(seed)

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        language = self.special_settings["language"]
        emotion = [
            self.special_settings["happiness"],
            self.special_settings["sadness"],
            self.special_settings["disgust"],
            self.special_settings["fear"],
            self.special_settings["surprise"],
            self.special_settings["anger"],
            self.special_settings["other"],
            self.special_settings["neutral"],
        ]

        # fix if all emotions are either set to 1.0 or 0.0, or all non-zero emotions are set to 1.0
        if all(emotion_value == 0.0 for emotion_value in emotion) or all(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.0001 if emotion_value == 0.0 else 0.9999 for emotion_value in emotion]
        elif all(emotion_value in (0.0, 1.0) for emotion_value in emotion) and any(emotion_value == 1.0 for emotion_value in emotion):
            emotion = [0.9999 if emotion_value == 1.0 else 0.0001 for emotion_value in emotion]

        unconditional_keys = self.special_settings["ignore_list"]

        cond_dict = make_cond_dict(text=text, speaker=speaker, language=language, emotion=emotion, speaking_rate=tts_speed, unconditional_keys=unconditional_keys, device=self.compute_device)
        conditioning = self.model.prepare_conditioning(cond_dict)

        stream_generator = self.model.stream(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,
            chunk_size=chunk_size,
            batch_size=1,
            disable_torch_compile=True,
        )

        audio_chunks = []
        for sr_out, codes_chunk in stream_generator:
            if self.audio_streamer is None:
                break
            audio_chunk = self.model.autoencoder.decode(codes_chunk).cpu()
            return_audio_chunk = audio_chunk[0]
            # change volume
            if tts_volume != 1.0:
                return_audio_chunk = audio_tools.change_volume(return_audio_chunk, tts_volume)
            audio_chunks.append(return_audio_chunk)
            # torch tensor to pcm bytes
            wav_bytes = self.return_pcm_audio(return_audio_chunk)
            if self.audio_streamer is not None:
                self.audio_streamer.add_audio_chunk(wav_bytes)

        full_audio = np.concatenate(audio_chunks, axis=-1)
        # numpy array to torch.Tensor
        full_audio = torch.from_numpy(full_audio).float()
        full_audio = full_audio.unsqueeze(0)

        self.last_generation = {"audio": full_audio, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return full_audio, self.sample_rate


    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")

        chunk_size = settings.GetOption("tts_streamed_chunk_size")
        #if self.audio_streamer is not None:
        #    self.audio_streamer.stop()
        #    self.audio_streamer = None
        #else:
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(audio_device,
                                                            source_sample_rate=self.sample_rate,
                                                            playback_channels=2,
                                                            buffer_size=chunk_size*2,
                                                            input_channels=1,
                                                            dtype="float32",
                                                            tag="tts",
                                                            )

    def play_audio(self, audio, device=None):
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
        #audio = np.int16(audio * 32767)  # Convert to 16-bit PCM
        #audio = audio_tools.convert_audio_datatype_to_integer(audio)

        # play audio tensor
        audio_tools.play_audio(audio, device,
                               source_sample_rate=int(self.target_sample_rate),
                               audio_device_channel_num=1,
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="float32",
                               tensor_sample_with=4,
                               tensor_channels=1,
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio, sample_rate=sample_rate):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to wav file
        buff = io.BytesIO()
        write_wav(buff, sample_rate, np_arr)

        return buff.read()

    def return_pcm_audio(self, audio):
        # convert pytorch tensor to numpy array
        np_arr = audio.detach().cpu().numpy()

        # convert numpy array to raw PCM bytes
        pcm_bytes = np_arr.tobytes()

        return pcm_bytes
