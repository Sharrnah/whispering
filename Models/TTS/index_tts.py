import gc
import io
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

import Utilities
import audio_tools
import settings
from Models.Singleton import SingletonMeta
from Models.TTS.text_segmentation import chunk_text, has_voice_tags, parse_voice_tagged_text


SAMPLE_RATE = 22050
DEFAULT_MODEL = "IndexTTS-2.5"
MODEL_CACHE_PATH = Path.cwd() / ".cache" / "index-tts"

VOICES_PATH = MODEL_CACHE_PATH / "voices"

SUPPORTED_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ar": "Arabic",
    "es": "Spanish",
}

# The ZIP hash intentionally remains zero until the application-hosted archive
# has been assembled. The ZIP must contain this exact layout at its root. Model
# files are sourced from the immutable revisions recorded below; Whispering
# Tiger never contacts Hugging Face to install them at runtime.
TTS_MODEL_LINKS = {
    # IndexTeam/IndexTTS-2.5 @ c39ce5ba981572cb187443877ff559dfb246ce63
    # facebook/w2v-bert-2.0 @ da985ba0987f70aaeb84a80f2851cfac8c697a7b
    # amphion/MaskGCT @ 265c6cef07625665d0c28d2faafb1415562379dc
    # funasr/campplus @ e4b6ede7ce16997aff4ae69fbca1f0175e2afede
    # nvidia/bigvgan_v2_22khz_80band_256x @ 633ff708ed5b74903e86ff1298cf4a98e921c513
    DEFAULT_MODEL: {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/index-tts/index-tts-2.5.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/index-tts/index-tts-2.5.zip",
            "https://s3.libs.space:9000/ai-models/index-tts/index-tts-2.5.zip",
        ],
        "checksum": "db4c1a599172976d0fcc8ebdefafabfed61d4275233485b96d1e9e7a9527372a",
        "file_checksums": {
            "LICENSE": "cc7da9ea0f8a97ef15ab3bf0389e636ce79ffca1aef5489520796ac87d87a87b",
            "README.md": "d21e2ed55013ce0c456b9aa6d4fab7222365bfad10dd839155da0ce36fa93001",
            "codec.pth": "d15cbed16a40f478438c961fb043f68dfa6353bf56c966761315db3433e9722c",
            "config.yaml": "18adf417be3e8f5e2e48e30f7420c719170a6870619436250f360d626877870e",
            "feat1.pt": "f219cb447d80216ba615666da2ff8d63ac544eee26657f3a7b278692bf7a67c4",
            "feat2.pt": "9c4292e96dee535aea9a6206e9a0c856dd578dde9212acdb16dd3ada4d12bf80",
            "gpt.pth": "43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85",
            "multilingual_zh_ja_yue_char_del.tiktoken": "747979631e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718",
            "s2mel.pth": "9b1b0003fc189c94cc349758d7ebc25f903b7eb2de4602879959cc64ce816456",
            "wav2vec2bert_stats.pt": "c9c176c2b8850ab2e3ba828bbfa969deaf4566ce55db5f2687b8430b87526ad2",
            "hf_cache\\w2v-bert-2.0\\config.json": "f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675",
            "hf_cache\\w2v-bert-2.0\\model.safetensors": "eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043",
            "hf_cache\\w2v-bert-2.0\\preprocessor_config.json": "8e6281aad64f97e40534135a59dcc5d33571efae376f2a25adf5551951897ab4",
            "hf_cache\\semantic_codec_model.safetensors": "ec947271175d8cad75ec37e83aa487e27c97a0f72a303393772da5ffa84bddf2",
            "hf_cache\\campplus_cn_common.bin": "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
            "hf_cache\\bigvgan\\config.json": "88a1f47acf747db0b21e97a389d838566147f7a5464583ff5c8d819d870f03ee",
            "hf_cache\\bigvgan\\bigvgan_generator.pt": "e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced",
        },
        "path": DEFAULT_MODEL,
    },
}

VOICE_MODEL_LINKS = {
    "voices": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/index-tts/voices.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/index-tts/voices.zip",
            "https://s3.libs.space:9000/ai-models/index-tts/voices.zip",
        ],
        "checksum": "1219fc592b50118807d54e3049e6b019d248e2e1a6be2324e398b3edd6df19a9",
        "file_checksums": {
            "Announcer_Ahri.wav": "2a3fd17d45b3c5633dd64e2d80a6e3fc924fa829da405e3b591a5bacdc88f9fc",
            "Attenborough.wav": "358540c89932baf1103960d99949b78ea7466f95b2225fdcd8f8bb8b976f09ee",
            "Jane.wav": "d1d2235af1a4408c641a765427978486f5cca9b369fc6456d8086449f1f92fe3",
            "Justin.wav": "a83c37f408b53efaeb9189f166c6669d1a0dc6cf779e85913fa9cbbbbe0d5aaf",
            "Xiaochen.wav": "7f0b735e188a06dc9f104eeb3fd71a3ef580d1f2133c95630c92a244dd253732",
            "default_voice.wav": "3ebc531cdaba358a327099c1c4f0448026719957bcf4d8e9868767f227e02f4e",
            "en_0.wav": "f006e2e9c76523bde4f5bbe67a7be9a600786d7432cbcc9486bc9501053298b7",
            "en_1.wav": "b0e22048e72414fcc1e6b6342e47a774d748a195ed34e4a5b3fcf416707f2b71",
            "fallback_audio.wav": "eaa7796d2c44424c645a0b384d82f09aac48fab2c9977de6f53b6a4f9d0e0da1",
            "female_shadowheart.wav": "8abb726ad6aaa5203e62de4c92ac2aab3d3fa1fdb509c9b76d254722178ab70a",
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede",
            "tiktok_adam.wav": "2ed130b6dd069ee4c306f6cb8fedb94db75567aefa084085c6a069bd2c34662d",
            "tiktok_jessie.wav": "5a26de921ea3e7c1ce1bfd2344fb107781def9366b56e2f583c7500a1052dbbd",
        },
        "path": "voices",
    },
}

MODEL_LIST = {"Multilingual voice cloning": [DEFAULT_MODEL]}


class IndexTTS(metaclass=SingletonMeta):
    sample_rate = SAMPLE_RATE
    special_settings_defaults = {
        "precision": "bfloat16",
        "language": "auto",
        "duration_factor": 1.0,
        "seed": -1,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 30,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
        "max_text_tokens_per_segment": 120,
        "streaming_segment_goal_length": 120,
        "pause_between_segments_ms": 200,
        "pause_between_voice_change_ms": 400,
        "text_normalization": True,
        "emotion_enabled": False,
        "emotion_happy": 0.0,
        "emotion_angry": 0.0,
        "emotion_sad": 0.0,
        "emotion_afraid": 0.0,
        "emotion_disgusted": 0.0,
        "emotion_melancholic": 0.0,
        "emotion_surprised": 0.0,
        "emotion_calm": 0.0,
        "emotion_strength": 1.0,
        "emotion_random_reference": False,
    }

    def __init__(self):
        self.model = None
        self.loaded_configuration = None
        self.compute_device_str = "cpu"
        self.compute_device = torch.device("cpu")
        self.special_settings = dict(self.special_settings_defaults)
        self.last_generation = {"audio": None, "sample_rate": None}
        self.audio_streamer = None
        self.download_state = {"is_downloading": False}
        self.voice_download_state = {"is_downloading": False}
        self.generation_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.language_code_converter = Utilities.LanguageCodeConverter()
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        os.makedirs(VOICES_PATH, exist_ok=True)
        self.set_compute_device(settings.GetOption("tts_ai_device"))

    def list_models(self):
        return MODEL_LIST

    def list_models_indexed(self):
        return tuple(
            {"language": language, "models": models}
            for language, models in self.list_models().items()
        )

    def set_special_setting(self, special_settings):
        if isinstance(special_settings, dict):
            self.special_settings = {**self.special_settings_defaults, **special_settings}

    def _ensure_special_settings(self):
        all_settings = settings.GetOption("special_settings")
        if not isinstance(all_settings, dict):
            all_settings = {}
        configured = all_settings.get("tts_index_tts")
        if isinstance(configured, dict):
            self.special_settings = {**self.special_settings_defaults, **configured}
        else:
            self.special_settings = dict(self.special_settings_defaults)
            all_settings["tts_index_tts"] = dict(self.special_settings)
            settings.SetOption("special_settings", all_settings)

    def set_compute_device(self, requested):
        device = str(requested or "").strip().lower()
        if device in {"", "auto", "cuda"}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("direct-ml"):
            raise ValueError("IndexTTS 2.5 supports CUDA and CPU, but not DirectML.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"Unsupported IndexTTS device: {requested}")
        self.compute_device_str = device
        self.compute_device = torch.device(device)

    def _get_model_name(self):
        selected = settings.GetOption("tts_model")
        model_name = DEFAULT_MODEL
        if isinstance(selected, (list, tuple)) and len(selected) == 2:
            model_name = re.sub(r"\(.*?\)", "", str(selected[1])).strip()
        return model_name if model_name in TTS_MODEL_LINKS else DEFAULT_MODEL

    @staticmethod
    def _model_directory(model_name=DEFAULT_MODEL):
        return MODEL_CACHE_PATH / TTS_MODEL_LINKS[model_name]["path"]

    def download_model(self, model_name=DEFAULT_MODEL, force_non_ui_dl=False):
        import downloader

        model_entry = TTS_MODEL_LINKS[model_name]
        model_directory = self._model_directory(model_name)
        if not downloader.model_needs_download(model_directory, model_entry["file_checksums"]):
            return True
        return downloader.download_model(
            {
                "model_path": MODEL_CACHE_PATH,
                "model_link_dict": TTS_MODEL_LINKS,
                "model_name": model_name,
                "title": "Text to Speech (IndexTTS 2.5)",
                "alt_fallback": False,
                "force_non_ui_dl": force_non_ui_dl,
                "extract_format": "zip",
            },
            self.download_state,
        )

    def download_voices(self, force_non_ui_dl=False):
        import downloader

        entry = VOICE_MODEL_LINKS["voices"]
        if not downloader.model_needs_download(VOICES_PATH, entry["file_checksums"]):
            return True
        return downloader.download_model(
            {
                "model_path": MODEL_CACHE_PATH,
                "model_link_dict": VOICE_MODEL_LINKS,
                "model_name": "voices",
                "title": "Voice samples (IndexTTS / Chatterbox)",
                "alt_fallback": False,
                "force_non_ui_dl": force_non_ui_dl,
                "extract_format": "zip",
            },
            self.voice_download_state,
        )

    def update_voices(self):
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        return [
            {"name": path.stem, "audio_filename": str(path.resolve())}
            for path in sorted(VOICES_PATH.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]

    def list_voices(self):
        voices = [
            {"name": voice["name"], "value": voice["name"]}
            for voice in self.update_voices()
        ]
        voices.append({"name": "open_voice_dir", "value": "open_dir:" + str(VOICES_PATH.resolve())})
        return voices

    def get_voice_by_name(self, voice_name):
        for voice in self.update_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    def _resolve_voice(self, ref_audio):
        if ref_audio:
            path = Path(ref_audio).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"IndexTTS voice reference does not exist: {path}")
            return str(path)
        selected = self.get_voice_by_name(settings.GetOption("tts_voice"))
        if selected is None:
            selected = self.get_voice_by_name("default_voice")
        if selected is None:
            voices = self.update_voices()
            selected = voices[0] if voices else None
        if selected is None:
            raise FileNotFoundError(
                f"IndexTTS needs a voice-cloning reference. Add a WAV/MP3/FLAC/OGG file to {VOICES_PATH}."
            )
        return selected["audio_filename"]

    def _build_voice_map(self, main_voice):
        voices = {
            voice["name"]: voice["audio_filename"]
            for voice in self.update_voices()
        }
        voices["main"] = main_voice
        return voices

    def _segment_plan(self, text, main_voice, split_for_streaming):
        """Return ordered ``(voice name, reference path, text)`` segments."""
        tagged = has_voice_tags(text)
        sections = parse_voice_tagged_text(text) if tagged else [("main", text.strip())]
        voices = self._build_voice_map(main_voice)
        plan = []

        goal_length = max(
            40,
            min(
                1000,
                self._int(
                    self.special_settings.get("streaming_segment_goal_length"),
                    120,
                ),
            ),
        )
        for requested_voice, section_text in sections:
            voice_name = requested_voice
            voice = voices.get(voice_name)
            if voice is None:
                print(f"IndexTTS voice '{voice_name}' not found; using the main voice.")
                voice_name = "main"
                voice = main_voice

            segments = (
                chunk_text(section_text, goal_length=goal_length)
                if split_for_streaming
                else [section_text.strip()]
            )
            for segment in segments:
                if segment and segment.strip():
                    plan.append((voice_name, voice, segment.strip()))
        if split_for_streaming:
            print(f"IndexTTS streamed playback prepared {len(plan)} text segment(s).")
        if tagged:
            voice_names = list(dict.fromkeys(item[0] for item in plan))
            print(f"IndexTTS voice-tag sequence: {', '.join(voice_names) or 'empty'}.")
        return plan

    def _pause_after_segment(self, plan, index):
        if index >= len(plan) - 1:
            return 0
        current_voice = plan[index][0]
        next_voice = plan[index + 1][0]
        setting = (
            "pause_between_voice_change_ms"
            if current_voice != next_voice
            else "pause_between_segments_ms"
        )
        default = 400 if current_voice != next_voice else 200
        return max(0, min(5000, self._int(self.special_settings.get(setting), default)))

    def _make_silence(self, duration_ms):
        sample_count = int(self.sample_rate * max(0, duration_ms) / 1000)
        return torch.zeros((1, sample_count), dtype=torch.float32)

    def _effective_precision(self):
        precision = str(self.special_settings.get("precision", "bfloat16")).lower()
        if precision not in {"bfloat16", "float32"}:
            precision = "bfloat16"
        if self.compute_device.type != "cuda":
            return "float32"
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            print("IndexTTS bfloat16 is unavailable on this GPU; using float32.")
            return "float32"
        return precision

    @staticmethod
    def _runtime_class():
        runtime_parent = Path(__file__).resolve().parent
        runtime_parent_string = str(runtime_parent)
        if runtime_parent_string not in sys.path:
            sys.path.insert(0, runtime_parent_string)
        from indextts.infer_v2_5 import IndexTTS2

        return IndexTTS2

    def load(self):
        with self.generation_lock:
            self._ensure_special_settings()
            self.set_compute_device(settings.GetOption("tts_ai_device"))
            precision = self._effective_precision()
            configuration = (self.compute_device_str, precision)
            if self.model is not None and self.loaded_configuration == configuration:
                return
            if self.model is not None:
                self.release_model()

            model_name = self._get_model_name()
            if not self.download_model(model_name):
                raise RuntimeError("IndexTTS 2.5 model installation failed.")
            # The shared pack is optional when the user already supplied a
            # reference path, but prepare it for normal profile voice selection.
            if not self.download_voices():
                print("IndexTTS voice sample pack could not be installed; custom references remain usable.")

            model_directory = self._model_directory(model_name).resolve()
            print(
                f"Loading IndexTTS 2.5 on {self.compute_device_str} "
                f"with {precision} precision..."
            )
            runtime_class = self._runtime_class()
            self.model = runtime_class(
                cfg_path=str(model_directory / "config.yaml"),
                model_dir=str(model_directory),
                use_bf16=precision == "bfloat16",
                device=self.compute_device_str,
                use_cuda_kernel=False,
                use_deepspeed=False,
                use_accel=False,
                use_torch_compile=False,
                use_qwen_emo=False,
            )
            self.loaded_configuration = configuration
            print("IndexTTS 2.5 loaded.")

    def release_model(self):
        self.model = None
        self.loaded_configuration = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop(self):
        self.stop_event.set()
        if self.audio_streamer is not None:
            self.audio_streamer.stop()
            self.audio_streamer = None

    @staticmethod
    def _float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _language(self, text):
        language = str(self.special_settings.get("language", "auto")).lower()
        if language == "auto":
            try:
                from Models import languageClassification

                detected, _ = languageClassification.classify(text)
                language = self.language_code_converter.convert(detected, "iso1").lower()
                language = language.split("-")[0]
                print(f"IndexTTS auto-detected language: {language}")
            except Exception as exc:
                print(f"IndexTTS language detection failed ({exc}); using English.")
                language = "en"
        if language not in SUPPORTED_LANGUAGES:
            print(f"IndexTTS does not support '{language}'; using English.")
            language = "en"
        return language

    def _duration_factor(self):
        factor = self._float(self.special_settings.get("duration_factor"), 1.0)
        prosody = str(settings.GetOption("tts_prosody_rate") or "")
        factor *= {
            "x-fast": 0.7,
            "fast": 0.85,
            "medium": 1.0,
            "slow": 1.2,
            "x-slow": 1.45,
        }.get(prosody, 1.0)
        return max(0.5, min(2.0, factor))

    def _emotion_vector(self):
        if not bool(self.special_settings.get("emotion_enabled", False)):
            return None
        names = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
        return [
            max(0.0, min(1.0, self._float(self.special_settings.get("emotion_" + name), 0.0)))
            for name in names
        ]

    def _seed_generation(self):
        seed = self._int(self.special_settings.get("seed"), -1)
        if seed < 0:
            seed = int(torch.randint(1, 2**31 - 1, (1,)).item())
        torch.manual_seed(seed)
        if self.compute_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _inference_kwargs(self, text):
        return {
            "lang": self._language(text),
            "emo_vector": self._emotion_vector(),
            "emo_alpha": max(0.0, min(1.0, self._float(self.special_settings.get("emotion_strength"), 1.0))),
            "use_random": bool(self.special_settings.get("emotion_random_reference", False)),
            "interval_silence": max(1, min(5000, self._int(self.special_settings.get("pause_between_segments_ms"), 200))),
            "verbose": False,
            "max_text_tokens_per_segment": max(20, min(500, self._int(self.special_settings.get("max_text_tokens_per_segment"), 120))),
            "duration_factor": self._duration_factor(),
            "text_normalization": bool(self.special_settings.get("text_normalization", True)),
            "do_sample": bool(self.special_settings.get("do_sample", True)),
            "temperature": max(0.01, min(2.0, self._float(self.special_settings.get("temperature"), 0.8))),
            "top_p": max(0.01, min(1.0, self._float(self.special_settings.get("top_p"), 0.8))),
            "top_k": max(0, min(200, self._int(self.special_settings.get("top_k"), 30))),
            "num_beams": max(1, min(5, self._int(self.special_settings.get("num_beams"), 3))),
            "repetition_penalty": max(1.0, min(20.0, self._float(self.special_settings.get("repetition_penalty"), 10.0))),
            "max_mel_tokens": max(128, min(1815, self._int(self.special_settings.get("max_mel_tokens"), 1500))),
        }

    @staticmethod
    def _wave_tensor(audio):
        if isinstance(audio, torch.Tensor):
            wave = audio.detach().float().cpu()
        else:
            wave = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        if wave.ndim == 0:
            wave = wave.reshape(1, 1)
        elif wave.ndim == 1:
            wave = wave.unsqueeze(0)
        elif wave.ndim == 2 and wave.shape[0] > wave.shape[1] and wave.shape[1] <= 2:
            wave = wave.transpose(0, 1)
        if wave.ndim > 2:
            wave = wave.reshape(1, -1)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        # Upstream returns int16 arrays and int16-scaled float tensors.
        if wave.numel() and wave.abs().max().item() > 1.5:
            wave = wave / 32767.0
        return wave.clamp(-1.0, 1.0).contiguous()

    def _finish_audio(self, wave, apply_normalization=True, call_plugin=True):
        wave = self._wave_tensor(wave)
        if apply_normalization and settings.GetOption("tts_normalize") and wave.numel():
            wave, _ = audio_tools.normalize_audio_lufs(
                wave, self.sample_rate, -24.0, -16.0, 1.3, verbose=True
            )
            wave = self._wave_tensor(wave)
        volume = self._float(settings.GetOption("tts_volume"), 1.0)
        if volume != 1.0:
            wave = self._wave_tensor(audio_tools.change_volume(wave, volume))
        if call_plugin:
            import Plugins

            result = Plugins.plugin_custom_event_call(
                "plugin_tts_after_audio", {"audio": wave, "sample_rate": self.sample_rate}
            )
            if isinstance(result, dict) and result.get("audio") is not None:
                wave = self._wave_tensor(result["audio"])
        return wave

    def _release_temporary_cuda_memory(self):
        device = getattr(self, "compute_device", torch.device("cpu"))
        if device.type == "cuda":
            # IndexTTS has length-dependent GPT/diffusion/vocoder working sets.
            # Return their now-unused allocator blocks so another application
            # does not have to compete with this process's high-water mark.
            torch.cuda.empty_cache()

    def _generate_segment(self, text, voice):
        try:
            result = self.model.infer(
                spk_audio_prompt=voice,
                text=text,
                output_path=None,
                stream_return=False,
                **self._inference_kwargs(text),
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("IndexTTS returned no audio.")
            sample_rate, audio = result
            self.sample_rate = int(sample_rate)
            return self._finish_audio(audio)
        finally:
            self._release_temporary_cuda_memory()

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        del remove_silence, silence_after_segments, normalize
        if not text or not text.strip():
            return torch.zeros((1, self.sample_rate // 10)), self.sample_rate
        with self.generation_lock:
            self.stop_event.clear()
            self.load()
            self._seed_generation()
            main_voice = self._resolve_voice(ref_audio)
            plan = self._segment_plan(
                text,
                main_voice,
                split_for_streaming=False,
            )
            chunks = []
            for index, (_, voice, segment) in enumerate(plan):
                if self.stop_event.is_set():
                    break
                chunks.append(self._generate_segment(segment, voice))
                pause_ms = self._pause_after_segment(plan, index)
                if pause_ms:
                    chunks.append(self._make_silence(pause_ms))
            wave = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, 0))
            self.last_generation = {"audio": wave, "sample_rate": self.sample_rate}
            return wave, self.sample_rate

    def init_audio_stream_playback(self):
        audio_device = settings.GetOption("device_out_index")
        if audio_device is None or audio_device == -1:
            audio_device = settings.GetOption("device_default_out_index")
        if self.audio_streamer is None:
            self.audio_streamer = audio_tools.AudioStreamer(
                audio_device,
                source_sample_rate=self.sample_rate,
                start_playback_timeout=1.0,
                min_buffer_play_time=float(settings.GetOption("tts_streamed_min_play_time")),
                playback_channels=2,
                buffer_size=settings.GetOption("tts_streamed_chunk_size"),
                input_channels=1,
                dtype="float32",
                tag="tts",
            )

    def tts_streaming(self, text, ref_audio=None):
        if not text or not text.strip():
            return torch.zeros((1, 0)), self.sample_rate
        with self.generation_lock:
            self.stop_event.clear()
            self.load()
            self._seed_generation()
            main_voice = self._resolve_voice(ref_audio)
            self.init_audio_stream_playback()
            chunks = []
            plan = self._segment_plan(
                text,
                main_voice,
                split_for_streaming=True,
            )
            for index, (_, voice, segment) in enumerate(plan):
                if self.stop_event.is_set():
                    break
                wave = self._generate_segment(segment, voice)
                if self.stop_event.is_set():
                    break
                chunks.append(wave)
                if self.audio_streamer is not None and wave.numel():
                    self.audio_streamer.add_audio_chunk(self.return_pcm_audio(wave))
                pause_ms = self._pause_after_segment(plan, index)
                if pause_ms:
                    silence = self._make_silence(pause_ms)
                    chunks.append(silence)
                    if self.audio_streamer is not None and silence.numel():
                        self.audio_streamer.add_audio_chunk(self.return_pcm_audio(silence))
            final_wave = torch.cat(chunks, dim=-1) if chunks else torch.zeros((1, 0))
            self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}
            return final_wave, self.sample_rate

    def get_last_generation(self):
        return self.last_generation["audio"], self.last_generation["sample_rate"]

    def play_audio(self, audio, device=None):
        if device is None:
            device = settings.GetOption("device_default_out_index")
        secondary = None
        if settings.GetOption("tts_use_secondary_playback"):
            secondary = settings.GetOption("tts_secondary_playback_device")
            if secondary == -1:
                secondary = settings.GetOption("device_default_out_index")
        audio_tools.play_audio(
            self._wave_tensor(audio),
            device,
            source_sample_rate=self.sample_rate,
            audio_device_channel_num=1,
            target_channels=1,
            input_channels=1,
            dtype="float32",
            tensor_sample_with=4,
            tensor_channels=1,
            secondary_device=secondary,
            stop_play=not settings.GetOption("tts_allow_overlapping_audio"),
            tag="tts",
        )

    def return_wav_file_binary(self, audio, sample_rate=SAMPLE_RATE):
        # Runtime playback uses float32, but exported files use conventional
        # signed PCM16 so Go audio packages and external WAV readers can open
        # them without requiring IEEE-float WAV support.
        array = self._wave_tensor(audio).squeeze(0).numpy()
        array = np.rint(np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        write_wav(buffer, int(sample_rate or self.sample_rate), array)
        return buffer.getvalue()

    def return_pcm_audio(self, audio):
        return self._wave_tensor(audio).squeeze(0).numpy().astype("<f4", copy=False).tobytes()
