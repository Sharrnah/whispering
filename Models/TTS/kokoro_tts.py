import io
import os
import re
from pathlib import Path

import numpy as np

import Plugins
import audio_tools
import settings
from Models.Singleton import SingletonMeta

cache_path = Path(Path.cwd() / ".cache" / "kokoro-tts-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

#os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(Path(cache_path / "eSpeak NG" / "libespeak-ng.dll").resolve())

from Models.TTS.kokoro import KPipeline, KModel

from scipy.io.wavfile import write as write_wav
import torch
import threading

failed = False


TTS_MODEL_LINKS = {
    # Models
    "kokoro-v1_0": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/kokoro-tts/models/kokoro-v1_0.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/kokoro-tts/models/kokoro-v1_0.zip",
            "https://s3.libs.space:9000/ai-models/kokoro-tts/models/kokoro-v1_0.zip",
        ],
        "checksum": "240dd23db1057c28695d90e095c4f9e21428d4a98ff213b9821a0f39ecc754bc",
        "file_checksums": {
            "config.json": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
            "kokoro-v1_0.pth": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
        },
        "path": "kokoro-v1_0",
    },
    # Default Voices
    "voices": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/kokoro-tts/voices.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/kokoro-tts/voices.zip",
            "https://s3.libs.space:9000/ai-models/kokoro-tts/voices.zip",
        ],
        "checksum": "f26daa7101269d395997bf531cb1a91d8bce476d27cb86106f0935b8d8fa2160",
        "file_checksums": {
            "af_alloy.pt": "6d877149dd8b348fbad12e5845b7e43d975390e9f3b68a811d1d86168bef5aa3",
            "af_aoede.pt": "c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4",
            "af_bella.pt": "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6",
            "af_heart.pt": "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
            "af_jessica.pt": "cdfdccb8cc975aa34ee6b89642963b0064237675de0e41a30ae64cc958dd4e87",
            "af_kore.pt": "8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d",
            "af_nicole.pt": "c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da",
            "af_nova.pt": "e0233676ddc21908c37a1f102f6b88a59e4e5c1bd764983616eb9eda629dbcd2",
            "af_river.pt": "e149459bd9c084416b74756b9bd3418256a8b839088abb07d463730c369dab8f",
            "af_sarah.pt": "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9",
            "af_sky.pt": "c799548aed06e0cb0d655a85a01b48e7f10484d71663f9a3045a5b9362e8512c",
            "am_adam.pt": "ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34",
            "am_echo.pt": "8bcfdc852bc985fb45c396c561e571ffb9183930071f962f1b50df5c97b161e8",
            "am_eric.pt": "ada66f0eefff34ec921b1d7474d7ac8bec00cd863c170f1c534916e9b8212aae",
            "am_fenrir.pt": "98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a",
            "am_liam.pt": "c82550757ddb31308b97f30040dda8c2d609a9e2de6135848d0a948368138518",
            "am_michael.pt": "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37",
            "am_onyx.pt": "e8452be16cd0f6da7b4579eaf7b1e4506e92524882053d86d72b96b9a7fed584",
            "am_puck.pt": "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223",
            "am_santa.pt": "7f2f7582fa2b1f160e90aafe6d0b442a685e773608b6667e545d743b073e97a7",
            "bf_alice.pt": "d292651b6af6c0d81705c2580dcb4463fccc0ff7b8d618a471dbb4e45655b3f3",
            "bf_emma.pt": "d0a423deabf4a52b4f49318c51742c54e21bb89bbbe9a12141e7758ddb5da701",
            "bf_isabella.pt": "cdd4c37003805104d1d08fb1e05855c8fb2c68de24ca6e71f264a30aaa59eefd",
            "bf_lily.pt": "6e09c2e481e2d53004d7e5ae7d3a325369e130a6f45c35a6002de75084be9285",
            "bm_daniel.pt": "fc3fce4e9c12ed4dbc8fa9680cfe51ee190a96444ce7c3ad647549a30823fc5d",
            "bm_fable.pt": "d44935f3135257a9064df99f007fc1342ff1aa767552b4a4fa4c3b2e6e59079c",
            "bm_george.pt": "f1bc812213dc59774769e5c80004b13eeb79bd78130b11b2d7f934542dab811b",
            "bm_lewis.pt": "b5204750dcba01029d2ac9cec17aec3b20a6d64073c579d694a23cb40effbd0e",
            "ef_dora.pt": "d9d69b0f8a2b87a345f269d89639f89dfbd1a6c9da0c498ae36dd34afcf35530",
            "em_alex.pt": "5eac53f767c3f31a081918ba531969aea850bed18fe56419b804d642c6973431",
            "em_santa.pt": "aa8620cb96cec705823efca0d956a63e158e09ad41aca934d354b7f0778f63cb",
            "ff_siwis.pt": "8073bf2d2c4b9543a90f2f0fd2144de4ed157e2d4b79ddeb0d5123066171fbc9",
            "hf_alpha.pt": "06906fe05746d13a79c5c01e21fd7233b05027221a933c9ada650f5aafc8f044",
            "hf_beta.pt": "63c0a1a6272e98d43f4511bba40e30dd9c8ceaf5f39af869509b9f51a71c503e",
            "hm_omega.pt": "b55f02a8e8483fffe0afa566e7d22ed8013acf47ad4f6bbee2795a840155703e",
            "hm_psi.pt": "2f0f055cea4f1083f4ef127ece48d71606347f6557dbb961c0ca5740a2da485b",
            "if_sara.pt": "6c0b253b955fe32f1a1a86006aebe83d050ea95afd0e7be15182f087deedbf55",
            "im_nicola.pt": "234ed06648649f9bd874b37508ea17560b9c993ef85b4ddb3e3a71e062bd2c12",
            "jf_alpha.pt": "1bf4c9dc69e45ee46183b071f4db766349aac5592acbcfeaf051018048a5d787",
            "jf_gongitsune.pt": "1b171917f18f351e65f2bf9657700cd6bfec4e65589c297525b9cf3c20105770",
            "jf_nezumi.pt": "d83f007a7f01783b77014561a7d493d327a0210e143440e91c9b697590d27661",
            "jf_tebukuro.pt": "0d6917904438aec85f73a6fa1f7ac2be6481aae47c697834936930a91796c576",
            "jm_kumo.pt": "98340afd68b1cee84fe0cd95528cfa6d4b39e416aa75a9df64049d52c8b55896",
            "pf_dora.pt": "07e4ff987c5d5a8c3995efd15cc4f0db7c4c15e881b198d8ab7f67ecf51f5eb7",
            "pm_alex.pt": "cf0ba8c573c2480fc54123683a35cf1e2ae130428e441eb91f9149bdb188a526",
            "pm_santa.pt": "d42103169c5c872abbafb9129133af7e942bb9d272c3cc3b95c203e7d7198c29",
            "zf_xiaobei.pt": "9b76be63dab4f4f96962030acc0126a9aee9728608fbbe115e2b58a2bd504df6",
            "zf_xiaoni.pt": "95b49f169bf1640f4f43c25e13daa39f7b98d15d00823e83ef5c14b3ced59e49",
            "zf_xiaoxiao.pt": "cfaf6f2ded1ee56f1ff94fcd2b0e6cdf32e5b794bdc05b44e7439d44aef5887c",
            "zf_xiaoyi.pt": "b5235dbaeef85a4c613bf78af9a88ff63c25bac5f26ba77e36186d8b7ebf05e2",
            "zm_yunjian.pt": "76cbf8bad35901d011d9628a2fdceb7b4f1f127e7a3269cb393b3941eb7fc417",
            "zm_yunxi.pt": "dbe6e1ce7c3dbaf2f5667432947b638b1c6831ccbe154c4610dbcc44f431e27b",
            "zm_yunxia.pt": "bb2b03b08e84d64e1214440ce3b624987fac177f2eeb5bab8571799a3d980acd",
            "zm_yunyang.pt": "5238ac22e0c7f8b6cdd2eddd6e444b8a700b73c4674d9a047d59a94ff96379a2"
        },
        "path": "voices",
    },
}

speed_mapping = {
    "": 1.0,        # Default speed when empty
    "x-slow": 0.1,  # Extra slow speed
    "slow": 0.5,   # Slow speed
    "medium": 1.0,  # Medium speed (default)
    "fast": 2.0,   # Fast speed
    "x-fast": 3.0   # Extra fast speed
}

model_list = {
    "Default": ["kokoro-v1_0"],
}


# patching EspeakWrapper of phonemizer library.
# See https://github.com/open-mmlab/Amphion/issues/323#issuecomment-2646709006



class KokoroTTS(metaclass=SingletonMeta):
    model = None
    sample_rate = 24000
    last_generation = {"audio": None, "sample_rate": None}
    voice_list = []
    audio_streamer = None

    pipeline = None

    last_language = ""

    special_settings = {
        "language": 'a',
    }

    compute_device = "cpu"

    def __init__(self):
        self.compute_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        if not self.pipeline:
            self.load()

        if not self.voice_list:
            self.update_voices()
        pass

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

    def set_special_setting(self, special_settings):
        self.special_settings = special_settings

    def stop(self):
        print("TTS Stop requested")
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    def _get_model_name(self):
        model = "kokoro-v1_0"
        if len(settings.GetOption('tts_model')) == 2:
            #language = settings.GetOption('tts_model')[0]
            model = settings.GetOption('tts_model')[1]
            # remove language part from string example: " (en & zh)"
            model = re.sub(r'\(.*?\)', '', model).strip()

        if model == "" or model not in TTS_MODEL_LINKS:
            model = "kokoro-v1_0"
        return model

    def load(self, lang='a'):
        self.load_model(lang)

    def load_model(self, lang='a'):
        if self.model is None:
            self.model = KModel(
                str(Path(cache_path / "kokoro-v1_0" / "config.json").resolve()),
                str(Path(cache_path / "kokoro-v1_0" / "kokoro-v1_0.pth").resolve())
            )
        if self.pipeline is None or self.last_language != lang:
            self.pipeline = KPipeline(lang_code=lang, model=self.model, device="cuda")
        pass

    def _get_voices(self):
        return self.voice_list

    def update_voices(self):
        # find all voices that have a .wav or .mp3 file
        voice_files = [f.stem for f in voices_path.iterdir() if f.is_file() and (f.suffix == ".pt")]

        voice_list = []
        for voice_id in voice_files:
            voice_file = voices_path / f"{voice_id}.pt"

            if voice_file.exists():
                voice_list.append({"name": voice_id, "voice_filename": str(voice_file.resolve())})
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

    def tts(self, text, remove_silence=True, silence_after_segments=0.2, normalize=True):
        print("TTS requested Kokoro TTS")
        self.set_compute_device(settings.GetOption('tts_ai_device'))

        lang = self.special_settings["language"]
        self.load(lang)

        tts_volume = settings.GetOption("tts_volume")

        voice_name = settings.GetOption('tts_voice')
        selected_voice = self.get_voice_by_name(voice_name)
        if selected_voice is None:
            print("No voice selected or does not exist. Using default voice 'af_heart'.")
            voice_name = "af_heart"
            selected_voice = self.get_voice_by_name(voice_name)
        ref_voice = selected_voice["voice_filename"]

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        voice_tensor = torch.load(ref_voice, weights_only=True)
        generator = self.pipeline(
            text, voice=voice_tensor,
            speed=tts_speed, split_pattern=r'\n+'
        )

        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            #with self.stop_flag_lock:
            #    if self.stop_flag:
            #        break
            # change volume
            if tts_volume != 1.0:
                audio = audio_tools.change_volume(audio, tts_volume)
            audio_chunks.append(audio)

        full_audio = np.concatenate(audio_chunks, axis=-1)
        # numpy array to torch.Tensor
        full_audio = torch.from_numpy(full_audio).float()
        full_audio = full_audio.unsqueeze(0)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio', {'audio': full_audio, 'sample_rate': self.sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            full_audio = plugin_audio['audio']

        # save last generation in memory
        self.last_generation = {"audio": full_audio, "sample_rate": self.sample_rate}

        print("TTS generation finished")

        return full_audio, self.sample_rate

    def tts_streaming(self, text, ref_audio=None):
        print("TTS requested Kokoro TTS (Streaming)")
        self.set_compute_device(settings.GetOption('tts_ai_device'))

        lang = self.special_settings["language"]
        self.load(lang)

        self.init_audio_stream_playback()

        tts_volume = settings.GetOption("tts_volume")

        voice_name = settings.GetOption('tts_voice')
        selected_voice = self.get_voice_by_name(voice_name)
        if selected_voice is None:
            print("No voice selected or does not exist. Using default voice 'af_heart'.")
            voice_name = "af_heart"
            selected_voice = self.get_voice_by_name(voice_name)
        ref_voice = selected_voice["voice_filename"]

        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)

        voice_tensor = torch.load(ref_voice, weights_only=True)
        generator = self.pipeline(
            text, voice=voice_tensor,
            speed=tts_speed, split_pattern=r'\n+'
        )

        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            if self.audio_streamer is None:
                break
            #with self.stop_flag_lock:
            #    if self.stop_flag:
            #        break
            # change volume
            if tts_volume != 1.0:
                audio = audio_tools.change_volume(audio, tts_volume)
            audio_chunks.append(audio)
            # torch tensor to pcm bytes
            wav_bytes = self.return_pcm_audio(audio)
            if self.audio_streamer is not None:
                self.audio_streamer.add_audio_chunk(wav_bytes)

        full_audio = np.concatenate(audio_chunks, axis=-1)
        # numpy array to torch.Tensor
        full_audio = torch.from_numpy(full_audio).float()
        #full_audio = full_audio.unsqueeze(0)

        # save last generation in memory
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
                                                            source_sample_rate=int(self.sample_rate),
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
                               source_sample_rate=int(self.sample_rate),
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
