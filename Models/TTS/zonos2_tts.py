import gc
import io
import os
import re
import threading
from pathlib import Path

import torch
import torchaudio
from scipy.io.wavfile import write as write_wav

import audio_tools
import settings
from ..Singleton import SingletonMeta
from .zonos2.native import SamplingOptions
from .zonos2.emotion import emotion_hidden_delta
from .zonos2.runtime import (
    DAC_SAMPLE_RATE,
    extract_speaker_embedding,
    generate_audio,
    load_bundle,
    resolve_attention_backend,
    unload_bundle,
)


ARCHIVE_CHECKSUM_PLACEHOLDER = "0" * 64
ZONOS_CACHE_PATH = Path.cwd() / ".cache" / "zonos-tts-cache"
MODEL_CACHE_PATH = ZONOS_CACHE_PATH / "zonos2"
VOICES_PATH = ZONOS_CACHE_PATH / "voices"


def _hosted_urls(archive_name):
    return [
        f"https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/zonos-tts/{archive_name}",
        f"https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/zonos-tts/{archive_name}",
        f"https://s3.libs.space:9000/ai-models/zonos-tts/{archive_name}",
    ]


# Extracted-file hashes were generated from the locally validated checkpoints.
# The application-hosted archive hashes remain zero until the ZIPs are created
# with maximum compression and uploaded. Each archive must contain the listed
# files directly at its ZIP root and must omit downloader receipts such as
# ``hash_checked``. Runtime installation never contacts Hugging Face.
TTS_MODEL_LINKS = {
    # drbaph/ZONOS2-BF16 @ c8470cd6b2c43639780635fac10921fd74a381e0
    "zonos2-bf16": {
        "urls": _hosted_urls("zonos2-bf16.zip"),
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "zonos2-bf16.safetensors": "cd2fd4c9e867750e27ccd0544d2ff4cc86b5c403d6bff4ceb7540068d31b26b4",
        },
        "path": "bf16",
        "source_revision": "c8470cd6b2c43639780635fac10921fd74a381e0",
    },
    # drbaph/ZONOS2-FP8 @ 6b713d14604a648939388f83935aed746d1be00e
    "zonos2-fp8-mixed": {
        "urls": _hosted_urls("zonos2-fp8-mixed.zip"),
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "zonos2-fp8-mixed.safetensors": "326e1d68c66cde0af14c27570edfdbf2d339be50ce3d6fafb8778ceaf3f9381a",
        },
        "path": "fp8",
        "source_revision": "6b713d14604a648939388f83935aed746d1be00e",
    },
    # Reuse the already hosted descript/dac_44khz archive used by Zonos v0.1.
    "dac_44khz": {
        "urls": _hosted_urls("dac_44khz.zip"),
        "checksum": "c7da3a820e5600a45dc0bb22ff7327a1c91f669b14151b55319afb584e8b5787",
        "file_checksums": {
            "config.json": "4eb55fb9af1990b8d608184ad29b70e358589719af7ea8d3c06998f7c2264a64",
            "model.safetensors": "6128ebff483a41422b0164d079a3773b0d8d82e64c4293d775994cbf8baf913a",
            "preprocessor_config.json": "c7d295758ce5777d6d88fef1996e94adc8ef3e2237ddfc5ecc24d1407aaddd7d",
        },
        "path": "dac_44khz",
        "source_revision": "c1bc521685adf9cfe247bc39a5ca58917eda1ac4",
    },
    # Speaker encoder files distributed with drbaph/ZONOS2-BF16.
    "speaker_encoder": {
        "urls": _hosted_urls("zonos2-speaker-encoder.zip"),
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "config.json": "297ac64afea59191e5aa446cb8acfdfdecfcd34771edda47d6948d1be5834ae9",
            "configuration_ecapa_tdnn.py": "6e187fd0adb8245829c855614e880551e2ec14c2372b2e5ad3c7e6565726d860",
            "feature_extraction_ecapa_tdnn.py": "4a353b0dc0171f640dc367f2e55e1ede045195d5d2134fa6124642ff3eb74927",
            "model.safetensors": "df60a638e7f4a29331c0af2bd2984ee5b992fee9d5923c776f7e4bdc3dedea48",
            "modeling_ecapa_tdnn.py": "88281ea40ad4792943d598d416476faa4b311acedf153350a556ceaa6b55805a",
            "preprocessor_config.json": "c67e710a6d1609f774e112459e7b47638ca185ae1b147d2e790c45c9cde3652f",
            "tokenizer_config.json": "30ac911d59755bbaa7f0986d8cbd4cc98ff73f4752bae8829936a573e663d6b0",
            "tokenizer_ecapa_tdnn.py": "299cf9528498ef093e6cfbb88d8738368c133342362e2cbe52e99ef59e39b13e",
        },
        "path": "speaker_encoder",
        "source_revision": "c8470cd6b2c43639780635fac10921fd74a381e0",
    },
    # Zyphra/ZONOS2 projected emotion directions @
    # 194c0a3ab67b90383a67646289f28d4ecb1c1f64
    "emotion_directions": {
        "urls": _hosted_urls("zonos2-emotion-directions.zip"),
        "checksum": ARCHIVE_CHECKSUM_PLACEHOLDER,
        "file_checksums": {
            "angry.npy": "78e7d1381a21a6903046391ff7401c49bd32e9ac348d40f903c4a04f06064592",
            "arousal.npy": "afa3eae5b74c4e495406bee7024f817f7655d4ba7e171e5ba03e92e885037c9f",
            "calibration.json": "622d9f6e6b6af88039a5586479d17f009a507cbefef49a2131ed3807327471d9",
            "happy.npy": "b180fb9a62fc7adbb06c024006f596853eb886547fe10ed921185ca1767e3917",
            "manifest.json": "7ad8b8f09a13e1efd6829dde02cc4398826ee901e624d5aef1a50239592054d7",
            "sad.npy": "ee2f1885eabe89782429fe5291ffb15550c39f6c20fbdc37c1efb755d04db018",
            "surprised.npy": "e5e26a4aa9670c78d6860e933f877a11f0469589a5f9dbd0ca9602cf99cf6f27",
            "valence.npy": "21f6d7d39dd3e402741ba10297377b4d248ccac45edd218a335f5f03486fcbe9",
        },
        "path": "emotion_directions",
        "source_revision": "194c0a3ab67b90383a67646289f28d4ecb1c1f64",
    },
}


# Share the existing Zonos v0.1 voice archive without importing that adapter and
# its plugin/runtime dependencies during ZONOS2 startup.
VOICE_MODEL_LINKS = {
    "voices": {
        "urls": _hosted_urls("voices.zip"),
        "checksum": "3997e6e1c7a7c0255bac3fd6ad9493098cfda410091755399e873e848459ac96",
        "file_checksums": {
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
            "test_zh_1_ref_short.wav": "96724a113240d1f82c6ded1334122f0176b96c9226ccd3c919e625bcfd2a3ede",
        },
        "path": "voices",
    },
}


model_list = {
    "Multilingual": ["zonos2-bf16", "zonos2-fp8-mixed"],
}


class Zonos2TTS(metaclass=SingletonMeta):
    """Whispering Tiger adapter for managed drbaph ZONOS2 checkpoints."""

    sample_rate = DAC_SAMPLE_RATE
    cache_path = MODEL_CACHE_PATH
    voices_path = VOICES_PATH

    special_settings = {
        "attention": "auto",
        "seed": -1,
        "max_new_tokens": 1024,
        "temperature": 1.15,
        "top_k": 106,
        "top_p": 0.0,
        "min_p": 0.18,
        "repetition_window": 50,
        "repetition_penalty": 1.2,
        "repetition_codebooks": 8,
        "speaking_rate": -1,
        "quality_enabled": True,
        "loudness_lufs": -1,
        "estimated_snr": -1,
        "maximum_pause": -1,
        "estimated_bandlimit_hz": -1,
        "leading_silence": -1,
        "trailing_silence": 3,
        "fade_out_ms": 0.0,
        "clean_speaker_background": False,
        "accurate_mode": True,
        "emotion_enabled": False,
        "emotion_happy": 0.0,
        "emotion_sad": 0.0,
        "emotion_angry": 0.0,
        "emotion_surprised": 0.0,
        "emotion_valence": 0.0,
        "emotion_arousal": 0.0,
        "emotion_strength": 1.0,
        "emotion_cfg_scale": 1.0,
    }

    def __init__(self):
        self.bundle = None
        self.compute_device = torch.device("cpu")
        self.last_generation = {"audio": None, "sample_rate": None}
        self.last_speaker_audio = None
        self.last_speaker_embedding = None
        self.download_state = {"is_downloading": False}
        self.voice_download_state = {"is_downloading": False}
        self.gen_lock = threading.Lock()
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.voices_path, exist_ok=True)

    def list_models(self):
        return model_list

    def list_models_indexed(self):
        return tuple(
            {"language": language, "models": models}
            for language, models in self.list_models().items()
        )

    @staticmethod
    def _model_directory(model_name):
        entry = TTS_MODEL_LINKS[model_name]
        relative_path = entry.get("path", "")
        return MODEL_CACHE_PATH / relative_path if relative_path else MODEL_CACHE_PATH

    def download_model(self, model_name, force_non_ui_dl=False):
        import downloader

        if model_name not in TTS_MODEL_LINKS:
            raise ValueError(f"Unsupported managed ZONOS2 model component: {model_name!r}")
        entry = TTS_MODEL_LINKS[model_name]
        model_directory = self._model_directory(model_name)
        if not downloader.model_needs_download(model_directory, entry["file_checksums"]):
            return True
        if entry["checksum"] == ARCHIVE_CHECKSUM_PLACEHOLDER:
            archive_name = os.path.basename(entry["urls"][0])
            raise RuntimeError(
                f"The ZONOS2 model archive {archive_name} is not currently "
                "available for automatic download."
            )
        return downloader.download_model(
            {
                "model_path": MODEL_CACHE_PATH,
                "model_link_dict": TTS_MODEL_LINKS,
                "model_name": model_name,
                "title": f"Text to Speech (ZONOS2: {model_name})",
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
                "model_path": ZONOS_CACHE_PATH,
                "model_link_dict": VOICE_MODEL_LINKS,
                "model_name": "voices",
                "title": "Voice samples (ZONOS2 / Zonos)",
                "alt_fallback": False,
                "force_non_ui_dl": force_non_ui_dl,
                "extract_format": "zip",
            },
            self.voice_download_state,
        )

    def _ensure_special_settings(self):
        all_settings = settings.GetOption("special_settings")
        if not isinstance(all_settings, dict):
            all_settings = {}
        configured = all_settings.get("tts_zonos2")
        if isinstance(configured, dict):
            self.special_settings = {**type(self).special_settings, **configured}
        else:
            all_settings["tts_zonos2"] = dict(self.special_settings)
            settings.SetOption("special_settings", all_settings)

    def set_special_setting(self, special_settings):
        if isinstance(special_settings, dict):
            self.special_settings = {**type(self).special_settings, **special_settings}

    def set_compute_device(self, requested):
        if requested in (None, "", "auto", "cuda"):
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            requested = "cpu"
        self.compute_device = torch.device(requested)

    def _get_model_name(self):
        model = "zonos2-bf16"
        selected = settings.GetOption("tts_model")
        if isinstance(selected, (list, tuple)) and len(selected) == 2:
            model = re.sub(r"\(.*?\)", "", str(selected[1])).strip()
        if model not in model_list["Multilingual"]:
            model = "zonos2-bf16"
        return model

    def load(self):
        self._ensure_special_settings()
        self.set_compute_device(settings.GetOption("tts_ai_device"))
        model_name = self._get_model_name()
        attention = str(self.special_settings.get("attention", "auto"))
        if self.bundle is not None:
            if (
                self.bundle.device == self.compute_device
                and self.bundle.model_name == model_name
            ):
                self.bundle.attention_preference = attention.strip().lower()
                self.bundle.attention_backend = resolve_attention_backend(
                    attention,
                    self.compute_device,
                )
                return

        required_components = (
            model_name,
            "dac_44khz",
            "speaker_encoder",
            "emotion_directions",
        )
        for component in required_components:
            if not self.download_model(component):
                raise RuntimeError(f"Could not install the ZONOS2 component {component}.")
        if not self.download_voices():
            raise RuntimeError("Could not install the shared ZONOS2 voice samples.")

        if self.bundle is not None:
            self.release_model()

        print(f"Loading ZONOS2 {model_name} on {self.compute_device}")
        self.bundle = load_bundle(
            self.cache_path,
            self.compute_device,
            model_name=model_name,
            attention=attention,
        )
        self.sample_rate = DAC_SAMPLE_RATE
        self.last_speaker_audio = None
        self.last_speaker_embedding = None
        print(
            f"ZONOS2 {model_name} loaded with "
            f"{self.bundle.attention_backend} attention"
        )

    def release_model(self):
        unload_bundle(self.bundle)
        self.bundle = None
        self.last_speaker_audio = None
        self.last_speaker_embedding = None

    def stop(self):
        # Native generation is synchronous; playback is stopped by the shared audio layer.
        return None

    def update_voices(self):
        extensions = {".wav", ".mp3", ".flac", ".ogg"}
        return [
            {"name": path.stem, "audio_filename": str(path.resolve())}
            for path in sorted(self.voices_path.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        ]

    def list_voices(self):
        voices = [{"name": "Default (no voice clone)", "value": "default"}]
        voices.extend(
            {"name": voice["name"], "value": voice["name"]}
            for voice in self.update_voices()
        )
        voices.append(
            {"name": "open_voice_dir", "value": "open_dir:" + str(self.voices_path.resolve())}
        )
        return voices

    def get_voice_by_name(self, voice_name):
        if voice_name in (None, "", "default"):
            return None
        for voice in self.update_voices():
            if voice["name"] == voice_name:
                return voice
        return None

    @staticmethod
    def _int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _split_text(text, maximum_utf8_bytes):
        """Split on sentence/word boundaries while respecting the byte-token prompt."""
        pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
        chunks = []
        current = ""

        def append_piece(piece):
            nonlocal current
            candidate = f"{current} {piece}".strip()
            if len(candidate.encode("utf-8")) <= maximum_utf8_bytes:
                current = candidate
                return
            if current:
                chunks.append(current)
                current = ""
            words = piece.split()
            if len(words) > 1:
                for word in words:
                    append_piece(word)
                return
            part = ""
            for character in piece:
                candidate_part = part + character
                if len(candidate_part.encode("utf-8")) > maximum_utf8_bytes:
                    if part:
                        chunks.append(part)
                    part = character
                else:
                    part = candidate_part
            current = part

        for piece in pieces:
            if piece.strip():
                append_piece(piece.strip())
        if current:
            chunks.append(current)
        return chunks

    def _speaker_embedding(self, ref_audio):
        if not ref_audio:
            selected = self.get_voice_by_name(settings.GetOption("tts_voice"))
            ref_audio = selected["audio_filename"] if selected else None
        if not ref_audio:
            return None
        ref_audio = str(Path(ref_audio).resolve())
        if ref_audio == self.last_speaker_audio and self.last_speaker_embedding is not None:
            return self.last_speaker_embedding
        waveform, sample_rate = torchaudio.load(ref_audio)
        embedding = extract_speaker_embedding(
            self.bundle,
            self.cache_path,
            waveform,
            sample_rate,
        )
        self.last_speaker_audio = ref_audio
        self.last_speaker_embedding = embedding
        return embedding

    def tts(self, text, ref_audio=None, remove_silence=True, silence_after_segments=0.2, normalize=True):
        del remove_silence, silence_after_segments, normalize
        if not text or not text.strip():
            return torch.zeros((1, self.sample_rate // 10)), self.sample_rate

        with self.gen_lock:
            self._ensure_special_settings()
            self.load()

            configured_seed = self._int(self.special_settings.get("seed"), -1)
            seed = configured_seed
            if seed < 0:
                seed = int(torch.randint(1, 2**31 - 1, (1,)).item())
            max_new_tokens = max(
                32,
                min(6000, self._int(self.special_settings.get("max_new_tokens"), 1024)),
            )
            options = SamplingOptions(
                max_new_tokens=max_new_tokens,
                temperature=max(0.0, min(2.0, self._float(self.special_settings.get("temperature"), 1.15))),
                top_k=max(0, min(1026, self._int(self.special_settings.get("top_k"), 106))),
                top_p=max(0.0, min(1.0, self._float(self.special_settings.get("top_p"), 0.0))),
                min_p=max(0.0, min(1.0, self._float(self.special_settings.get("min_p"), 0.18))),
                repetition_window=max(0, min(512, self._int(self.special_settings.get("repetition_window"), 50))),
                repetition_penalty=max(1.0, min(2.0, self._float(self.special_settings.get("repetition_penalty"), 1.2))),
                repetition_codebooks=max(-1, min(9, self._int(self.special_settings.get("repetition_codebooks"), 8))),
                seed=seed,
            )
            speaking_rate = max(-1, min(7, self._int(self.special_settings.get("speaking_rate"), -1)))
            quality_limits = (11, 11, 11, 7, 7, 7)
            quality_defaults = (-1, -1, -1, -1, -1, 3)
            quality_names = (
                "loudness_lufs",
                "estimated_snr",
                "maximum_pause",
                "estimated_bandlimit_hz",
                "leading_silence",
                "trailing_silence",
            )
            quality_buckets = []
            for name, default, maximum in zip(
                quality_names, quality_defaults, quality_limits
            ):
                value = max(
                    -1,
                    min(maximum, self._int(self.special_settings.get(name), default)),
                )
                quality_buckets.append(None if value < 0 else value)
            if not bool(self.special_settings.get("quality_enabled", True)):
                quality_buckets = None
            speaker_embedding = self._speaker_embedding(ref_audio)
            emotion_values = {
                "happy": max(-1.0, min(1.0, self._float(self.special_settings.get("emotion_happy"), 0.0))),
                "sad": max(-1.0, min(1.0, self._float(self.special_settings.get("emotion_sad"), 0.0))),
                "angry": max(-1.0, min(1.0, self._float(self.special_settings.get("emotion_angry"), 0.0))),
                "surprised": max(-1.0, min(1.0, self._float(self.special_settings.get("emotion_surprised"), 0.0))),
            }
            emotion_valence = max(-1.0, min(1.0, self._float(self.special_settings.get("emotion_valence"), 0.0)))
            emotion_arousal = max(-1.0, min(1.0, self._float(self.special_settings.get("emotion_arousal"), 0.0)))
            emotion_requested = bool(self.special_settings.get("emotion_enabled", False)) and (
                any(value != 0.0 for value in emotion_values.values())
                or emotion_valence != 0.0
                or emotion_arousal != 0.0
            )
            speaker_emotion_delta = None
            emotion_cfg_scale = 1.0
            if emotion_requested:
                if speaker_embedding is None:
                    raise ValueError(
                        "ZONOS2 emotion control requires a selected voice reference."
                    )
                if self.bundle.emotion_directions is None:
                    raise FileNotFoundError(
                        "ZONOS2 emotion directions are missing from "
                        f"{self.cache_path / 'emotion_directions'}."
                    )
                speaker_emotion_delta = emotion_hidden_delta(
                    self.bundle.emotion_directions,
                    sliders=emotion_values,
                    valence=emotion_valence,
                    arousal=emotion_arousal,
                    strength=max(0.0, min(3.0, self._float(self.special_settings.get("emotion_strength"), 1.0))),
                )
                emotion_cfg_scale = max(
                    1.0,
                    min(3.0, self._float(self.special_settings.get("emotion_cfg_scale"), 1.0)),
                )

            maximum_text_bytes = max(64, self.bundle.config.max_seqlen - max_new_tokens - 64)
            segments = self._split_text(text, maximum_text_bytes)
            segment_audio = []
            print(f"ZONOS2 generation using seed {seed}, {len(segments)} segment(s)")
            for index, segment in enumerate(segments):
                options.seed = seed + index if seed > 0 else 0
                segment_audio.append(
                    generate_audio(
                        self.bundle,
                        segment,
                        options,
                        speaking_rate_bucket=speaking_rate,
                        quality_buckets=quality_buckets,
                        speaker_embedding=speaker_embedding,
                        clean_speaker_background=bool(self.special_settings.get("clean_speaker_background", False)),
                        accurate_mode=bool(self.special_settings.get("accurate_mode", True)),
                        speaker_emotion_delta=speaker_emotion_delta,
                        emotion_cfg_scale=emotion_cfg_scale,
                    )
                )

            final_wave = torch.cat(segment_audio, dim=-1).float().cpu()
            fade_out_ms = max(
                0.0,
                min(2000.0, self._float(self.special_settings.get("fade_out_ms"), 0.0)),
            )
            fade_samples = min(
                final_wave.shape[-1],
                int(self.sample_rate * fade_out_ms / 1000.0),
            )
            if fade_samples > 0:
                fade = 0.5 * (
                    1.0
                    + torch.cos(torch.linspace(0.0, torch.pi, fade_samples))
                )
                final_wave[..., -fade_samples:] *= fade
            if settings.GetOption("tts_normalize"):
                final_wave, _ = audio_tools.normalize_audio_lufs(
                    final_wave, self.sample_rate, -24.0, -16.0, 1.3, verbose=True
                )
            volume = settings.GetOption("tts_volume")
            if volume != 1.0:
                final_wave = audio_tools.change_volume(final_wave, volume)
            # Import lazily: importing Plugins while the TTS registry is being built
            # causes plugin modules that import websocket.py to re-enter this module.
            import Plugins

            plugin_audio = Plugins.plugin_custom_event_call(
                "plugin_tts_after_audio",
                {"audio": final_wave, "sample_rate": self.sample_rate},
            )
            if plugin_audio is not None and plugin_audio.get("audio") is not None:
                final_wave = plugin_audio["audio"]

            if final_wave.ndim == 1:
                final_wave = final_wave.unsqueeze(0)
            self.last_generation = {"audio": final_wave, "sample_rate": self.sample_rate}
            gc.collect()
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
            audio,
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

    def return_wav_file_binary(self, audio, sample_rate=DAC_SAMPLE_RATE):
        array = audio.detach().float().cpu().squeeze().numpy()
        buffer = io.BytesIO()
        write_wav(buffer, sample_rate, array)
        return buffer.getvalue()
