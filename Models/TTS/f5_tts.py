import io
import os
import re
from pathlib import Path

import numpy
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

import Plugins
import audio_tools
import downloader
#from cached_path import cached_path

import settings

from Models.TTS.F5TTS.model.backbones.unett import UNetT
from Models.TTS.F5TTS.model.backbones.dit import DiT

from Models.TTS.F5TTS.model.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)


def estimate_remaining_time(total_segments, segment_times, segments_for_estimate=3, last_x_segments=None):
    """
    Estimates the remaining time based on the average time of specified segment times.

    Parameters:
    total_segments (int): Total number of segments.
    segment_times (list): List of times taken for each segment.
    segments_for_estimate (int): Minimum number of segments needed to start estimating.
    last_x_segments (int, optional): Number of recent segments to use for estimating the average time.
                                     If None, all available segments are used.

    Returns:
    str: Formatted string of estimated remaining time or "[estimating...]" if not enough data.
    """
    if len(segment_times) < segments_for_estimate:
        return " [estimating...]"

    if last_x_segments is not None:
        # Use only the last x segments for estimation
        relevant_segment_times = segment_times[-last_x_segments:]
    else:
        # Use all available segment times for estimation
        relevant_segment_times = segment_times

    # Calculate the average time per segment
    avg_time_per_segment = sum(relevant_segment_times) / len(relevant_segment_times)

    # Calculate the remaining segments
    remaining_segments = total_segments - len(segment_times)

    # Estimate the remaining time
    estimated_remaining_time = avg_time_per_segment * remaining_segments

    # Convert estimated time to hours, minutes, and seconds
    hours, rem = divmod(estimated_remaining_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format the estimated time
    estimated_time_str = ""
    if hours > 0:
        estimated_time_str += f"{int(hours)} hrs. "
    if minutes > 0:
        estimated_time_str += f"{int(minutes)} min. "
    estimated_time_str += f"{int(seconds)} sec."

    if estimated_time_str:
        return f" [~ {estimated_time_str} remaining]"
    else:
        return ""


cache_path = Path(Path.cwd() / ".cache" / "f5tts-cache")
os.makedirs(cache_path, exist_ok=True)
voices_path = Path(cache_path / "voices")
os.makedirs(voices_path, exist_ok=True)

vocos_local_path = Path(cache_path / "vocos")

tts_proc = None
failed = False

TTS_MODEL_LINKS = {
    # Models
    "F5-TTS": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/F5TTS_Base.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/F5TTS_Base.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/F5TTS_Base.zip",
        ],
        "checksum": "bc1a704885caf25d0d6878c21ea8e525697f2e380d81bc7630744dbac17fe728",
        "file_checksums": {
            "model.safetensors": "4180310f91d592cee4bc14998cd37c781f779cf105e8ca8744d9bd48ca7046ae"
        },
        "path": "F5TTS_Base",
    },
    "E2-TTS": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/models/E2TTS_Base.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/models/E2TTS_Base.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/models/E2TTS_Base.zip",
        ],
        "checksum": "dd5c01d9801274d94ed5d2a74d6f5a9e550325ec6addc56ff8e1c2259ac25d6c",
        "file_checksums": {
            "model.safetensors": "8a813cd26fd21b298734eece9474bfea842a585adff98d2bb89fd384b1c00ac7"
        },
        "path": "E2TTS_Base",
    },
    # Others
    "vocab": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/vocab.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/vocab.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/vocab.zip",
        ],
        "checksum": "d7f6e519de25c45235d46a040465966c7904c560a26341d399e0754f485d08cd",
        "file_checksums": {
            "Emilia_ZH_EN_pinyin\\vocab.txt": "4e173934be56219eb38759fa8d4c48132d5a34454f0c44abce409bcf6a07ec46",
            "librispeech_pc_test_clean_cross_sentence.lst": "a232b6fe539fb3bbde5714a13c3f2333900997be274c19ca7558f31615679b21",
        },
        "path": "vocab",
    },
    "vocos": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/vocos.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/vocos.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/vocos.zip",
        ],
        "checksum": "e9af55494b4729ca10b10e7c3d1e1769fe12dc1fcca75cdbd0219699dce55677",
        "file_checksums": {
            "README.md": "5858497d76d58914b29958a0f448e7b4fd3bb54940100baaa7520d4a56d8b3df",
            "config.yaml": "da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc",
            "pytorch_model.bin": "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
        },
        "path": "vocos",
    },
    # Default Voices
    "voices": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/f5-tts/voices.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/f5-tts/voices.zip",
            "https://s3.libs.space:9000/ai-models/f5-tts/voices.zip",
        ],
        "checksum": "3997e6e1c7a7c0255bac3fd6ad9493098cfda410091755399e873e848459ac96",
        "file_checksums": {
            #"Info.txt": "19455228a1ef69f6abe1dfa7f8477a1da600ed371ca237032263786d8a7a51ec",
            "Announcer_Ahri.txt": "65cdbe885b89037dc651bea9bb7c41077471a2d7168e25c905c7034da7de285d",
            "Announcer_Ahri.wav": "2a3fd17d45b3c5633dd64e2d80a6e3fc924fa829da405e3b591a5bacdc88f9fc",
            "Attenborough.txt": "4c617f7adc60b992de93abd18bd447da5c882db7d04d9d51b241cdf79cbda6a1",
            "Attenborough.wav": "358540c89932baf1103960d99949b78ea7466f95b2225fdcd8f8bb8b976f09ee",
            "Jane.txt": "58e939100b6422f76e631d445a957047fa915ba6727f984ebdcecfa3418f5d08",
            "Jane.wav": "d1d2235af1a4408c641a765427978486f5cca9b369fc6456d8086449f1f92fe3",
            "Justin.txt": "6ce2802c88bd83ef12ecb3338f1bf6f8bc5bc12212b3cd1d2863d0d3ab93632b",
            "Justin.wav": "a83c37f408b53efaeb9189f166c6669d1a0dc6cf779e85913fa9cbbbbe0d5aaf",
            "Xiaochen.txt": "1316b1e27871565b1d7cd4f64b0521a37632cc15d1ea0944d18394bdaf76d8e2",
            "Xiaochen.wav": "7f0b735e188a06dc9f104eeb3fd71a3ef580d1f2133c95630c92a244dd253732",
            "en_0.txt": "3bb999d455ca88b8eca589bd16d3b99db8b324b5f8c57e3283e4bb4db8593243",
            "en_0.wav": "f006e2e9c76523bde4f5bbe67a7be9a600786d7432cbcc9486bc9501053298b7",
            "en_1.txt": "79cccada817b316fa855dc8ca04823f59a11c956b5780fbb3267ddf684c8e145",
            "en_1.wav": "b0e22048e72414fcc1e6b6342e47a774d748a195ed34e4a5b3fcf416707f2b71",
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede"
        },
        "path": "voices",
    },
}

model_list = {
    "en & zh": ["F5-TTS", "E2-TTS"],
}

speed_mapping = {
    "": 1.0,        # Default speed when empty
    "x-slow": 0.5,  # Extra slow speed
    "slow": 0.75,   # Slow speed
    "medium": 1.0,  # Medium speed (default)
    "fast": 1.25,   # Fast speed
    "x-fast": 1.5   # Extra fast speed
}

class F5TTS:
    lang = 'en'
    model_id = 'F5-TTS'
    model = None

    target_sample_rate = 24000
    target_rms = 0.1
    n_mel_channels = 100
    hop_length = 256
    ode_method = "euler"
    speed = 1.0

    device = None
    ema_model = None
    vocos = None

    currently_downloading = False

    config = {
        "model": "F5-TTS",
        #"model": "E2-TTS",
        "ref_audio": str(Path(cache_path / "voices" / "en_1.wav").resolve()),
        # If an empty "", transcribes the reference audio automatically.
        "ref_text": "Some call me nature, others call me mother nature.",
        #"gen_text": "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.",
        "gen_text": "This is a test.",
        # File with text to generate. Ignores the text above.
        "gen_file": "",
        "remove_silence": True,
        "output_dir": "tests",
        "voices": {},
    }

    def __init__(self):
        pass

    def download_model(self, model_name):
        model_directory = Path(cache_path / TTS_MODEL_LINKS[model_name]["path"])
        os.makedirs(str(model_directory.resolve()), exist_ok=True)

        # if one of the files does not exist, break the loop and download the files
        needs_download = False
        for file, expected_checksum in TTS_MODEL_LINKS[model_name]["file_checksums"].items():
            if not Path(model_directory / file).exists():
                needs_download = True
                break

        if not needs_download:
            for file, expected_checksum in TTS_MODEL_LINKS[model_name]["file_checksums"].items():
                checksum = downloader.sha256_checksum(str(model_directory.resolve() / file))
                if checksum != expected_checksum:
                    needs_download = True
                    break

        # iterate over all TTS_MODEL_LINKS[model_name]["files"] entries and download them
        if needs_download and not self.currently_downloading:
            self.currently_downloading = True
            if not downloader.download_extract(TTS_MODEL_LINKS[model_name]["urls"],
                                               str(model_directory.resolve()),
                                               TTS_MODEL_LINKS[model_name]["checksum"], title="Text 2 Speech (F5/E2 TTS) - " + model_name, extract_format="zip"):
                print(f"Download failed: Text 2 Speech (F5/E2 TTS) - {model_name}")

        self.currently_downloading = False

    def load(self):

        model = "F5-TTS"
        if len(settings.GetOption('tts_model')) == 2:
            #language = settings.GetOption('tts_model')[0]
            model = settings.GetOption('tts_model')[1]
            # remove language part from string example: " (en & zh)"
            model = re.sub(r'\(.*?\)', '', model).strip()

        if model == "" or model not in TTS_MODEL_LINKS:
            model = "F5-TTS"

        self.download_model("vocab")
        self.download_model("vocos")
        self.download_model("voices")
        self.download_model(model)

        self.model_id = model
        # load models
        if model == "F5-TTS":
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            exp_name = "F5TTS_Base"
            #ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            ckpt_file = str(Path(cache_path / exp_name / f"model.safetensors").resolve())
            # ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

        elif model == "E2-TTS":
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            exp_name = "E2TTS_Base"
            #ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            ckpt_file = str(Path(cache_path / exp_name / f"model.safetensors").resolve())
            # ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

        vocab_file = str(Path(cache_path / "vocab" / "Emilia_ZH_EN_pinyin" / "vocab.txt").resolve())

        print(f"Using {model}...")
        self.ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file)

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        model_list = self.list_models()
        return tuple([{"language": language, "models": models} for language, models in model_list.items()])

    def _get_voices(self):
        voices_path = Path(cache_path / "voices")
        # find all voices that have both a .wav and .txt file
        voice_files = [f.stem for f in voices_path.iterdir() if f.is_file() and f.suffix == ".wav"]
        voice_list = []
        for voice_id in voice_files:
            wav_file = voices_path / f"{voice_id}.wav"
            txt_file = voices_path / f"{voice_id}.txt"
            if wav_file.exists() and txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as file:
                    text_content = file.read().strip()
                voice_list.append({"name": voice_id, "wav_filename": str(wav_file.resolve()), "text_content": text_content})
        return voice_list

    def list_voices(self):
        return [voice["name"] for voice in self._get_voices()]

    def get_voice_by_name(self, voice_name):
        for voice in self._get_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def load_vocos(self):
        self.vocos = load_vocoder(is_local=True, local_path=vocos_local_path)

    def tts(self, text, ref_audio=None, ref_text=None, remove_silence=True):
        tts_speed = speed_mapping.get(settings.GetOption('tts_prosody_rate'), 1)
        return_sample_rate = self.target_sample_rate
        if ref_audio is None and ref_text is None:
            voice_name = settings.GetOption('tts_voice')
            selected_voice = self.get_voice_by_name(voice_name)
            ref_audio = selected_voice["wav_filename"]
            ref_text = selected_voice["text_content"]

        if self.ema_model is None:
            print("loading ema_model...")
            self.load_model(self.model_id)
        if self.vocos is None:
            print("loading vocos model...")
            self.load_vocos()

        if ref_audio is None:
            ref_audio = self.config["ref_audio"]
        if ref_text is None:
            ref_text = self.config["ref_text"]

        main_voice = {"ref_audio":ref_audio, "ref_text":ref_text}

        # get voice list from _get_voices function
        voices_list = self._get_voices()
        for voice_entry in voices_list:
            self.config["voices"][voice_entry["name"]] = {'ref_audio': voice_entry['wav_filename'], 'ref_text': voice_entry['text_content']}

        if "voices" not in self.config or self.config["voices"] == {}:
            voices = {"main": main_voice}
        else:
            voices = self.config["voices"]
            voices["main"] = main_voice
        #for voice in voices:
            #voices[voice]['ref_audio'], voices[voice]['ref_text'] = preprocess_ref_audio_text(voices[voice]['ref_audio'], voices[voice]['ref_text'])
            #print("Voice:", voice)
            #print("Ref_audio:", voices[voice]['ref_audio'])
            #print("Ref_text:", voices[voice]['ref_text'])

        segment_times = []

        generated_audio_segments = []
        reg1 = r'(?=\[\w+\])'
        chunks = re.split(reg1, text)
        reg2 = r'\[(\w+)\]'

        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        for i, text_chunk in enumerate(chunks):
            text_chunk = text_chunk.strip()  # Remove leading/trailing whitespace
            if not text_chunk:
                continue  # Skip empty chunks

            match = re.match(reg2, text_chunk)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text_chunk = re.sub(reg2, "", text_chunk)
            gen_text = text_chunk.strip()

            if not gen_text:
                continue  # Skip processing if there's no text to generate

            ref_audio = voices[voice]['ref_audio']
            ref_text = voices[voice]['ref_text']

            #print(f"Voice: {voice}")
            audio, final_sample_rate, spectragram = infer_process(ref_audio, ref_text, gen_text, self.ema_model, speed=tts_speed, show_info=None)
            return_sample_rate = final_sample_rate
            generated_audio_segments.append(audio)

            estimate_time_full_str = estimate_remaining_time(len(chunks), segment_times, 3)
            print(f"TTS progress: {int((i+1) / len(chunks) * 100)}% ({i+1} of {len(chunks)} segments){estimate_time_full_str}")

        if generated_audio_segments:
            #wave_path = Path(Path.cwd() / "temp_tts_f5.wav")
            final_wave = np.concatenate(generated_audio_segments)
            return final_wave, return_sample_rate
            # with open(wave_path, "wb") as f:
            #     sf.write(f.name, final_wave, final_sample_rate)
            #     # Remove silence
            #     if remove_silence:
            #         remove_silence_for_generated_wav(f.name)
            #     print(f.name)

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
        audio = audio_tools.convert_audio_datatype_to_integer(audio)

        audio_bytes = self.return_wav_file_binary(audio, int(self.target_sample_rate))

        # play audio tensor
        audio_tools.play_audio(audio_bytes, device,
                               source_sample_rate=int(self.target_sample_rate),
                               audio_device_channel_num=1,
                               target_channels=1,
                               input_channels=source_channels,
                               dtype="int16",
                               secondary_device=secondary_audio_device,
                               stop_play=not allow_overlapping_audio,
                               tag="tts"
                               )

    def return_wav_file_binary(self, audio, sample_rate=24000):
        # convert numpy array to wav file
        buff = io.BytesIO()
        write(buff, sample_rate, audio)

        # call custom plugin event method
        plugin_audio = Plugins.plugin_custom_event_call('plugin_tts_after_audio',
                                                        {'audio': buff, 'sample_rate': sample_rate})
        if plugin_audio is not None and 'audio' in plugin_audio and plugin_audio['audio'] is not None:
            print("applied plugin_tts_after_audio")
            buff = plugin_audio['audio']

        return buff.read()
