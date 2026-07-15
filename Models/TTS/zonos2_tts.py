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


model_list = {
    "Multilingual": ["zonos2-bf16", "zonos2-fp8-mixed"],
}


class Zonos2TTS(metaclass=SingletonMeta):
    """Whispering Tiger adapter for local drbaph ZONOS2 checkpoints."""

    sample_rate = DAC_SAMPLE_RATE
    cache_path = Path.cwd() / ".cache" / "zonos-tts-cache" / "zonos2"
    voices_path = Path.cwd() / ".cache" / "zonos-tts-cache" / "voices"

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
