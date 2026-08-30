import os
import threading
from pathlib import Path

from df.enhance import enhance, init_df
from df.model import ModelParams
from libdf import DF
from Models.Singleton import SingletonMeta
import numpy as np
import torch
import audio_tools
from typing import Union

import downloader
from Models.STS.AudioEnhancer import float32_to_pcm16

cache_df_path = Path(Path.cwd() / ".cache" / "deepfilternet")

DEEP_FILTER_LINK = {
    "DeepFilterNet3": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/DeepFilterNet/DeepFilterNet3.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/DeepFilterNet/DeepFilterNet3.zip",
            "https://s3.libs.space:9000/ai-models/DeepFilterNet/DeepFilterNet3.zip",
        ],
        "checksum": "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2"
    }
}


class DeepFilterNet(metaclass=SingletonMeta):
    df_model = None
    df_state = None

    def __init__(self, post_filter=False, epoch: Union[str, int, None] = "best"):
        self._inference_lock = threading.RLock()
        os.makedirs(cache_df_path, exist_ok=True)

        model = "DeepFilterNet3"

        model_path = Path(cache_df_path / model / "checkpoints/model_120.ckpt.best")
        model_config_path = Path(cache_df_path / model / "config.ini")
        if not Path(cache_df_path).exists() or not model_path.is_file() or not model_config_path.is_file():
            print("downloading DeepFilterNet3...")
            if not downloader.download_extract(DEEP_FILTER_LINK[model]["urls"],
                                               str(cache_df_path.resolve()),
                                               DEEP_FILTER_LINK[model]["checksum"], title="DeepFilterNet3 (A.I. Denoise)"):
                print("Model download failed")
        self.df_model, self.df_state, _ = init_df(model_base_dir=str(Path(cache_df_path / model).resolve()), post_filter=post_filter, epoch=epoch, log_level="none")
        model_params = ModelParams()
        self._df_state_kwargs = {
            "sr": model_params.sr,
            "fft_size": model_params.fft_size,
            "hop_size": model_params.hop_size,
            "nb_bands": model_params.nb_erb,
            "min_nb_erb_freqs": model_params.min_nb_freqs,
        }

        # original part (downloads to %LOCALAPPDATA%\DeepFilterNet)
        # self.df_model, self.df_state, _ = init_df(post_filter=post_filter, epoch=epoch, log_level="none")
        pass

    def int2float(self, sound):
        """Convert PCM to normalized float audio without peak-normalizing noise."""
        sound = np.asarray(sound)
        if np.issubdtype(sound.dtype, np.integer):
            scale = float(max(abs(np.iinfo(sound.dtype).min), np.iinfo(sound.dtype).max))
            sound = sound.astype(np.float32) / scale
        else:
            sound = sound.astype(np.float32, copy=False)
        return sound.squeeze()

    def enhance_audio(self, audio_bytes, sample_rate=16000, output_sample_rate=16000, input_channels=1, output_channels=1, strength=1.0):
        strength = float(np.clip(strength, 0.0, 1.0))
        input_channels = max(1, int(input_channels or 1))
        output_channels = max(1, int(output_channels or 1))

        audio_full_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio_full_int16.size % input_channels:
            raise ValueError("PCM sample count is not divisible by input_channels.")
        if (
            strength <= 0.0
            and int(sample_rate) == int(output_sample_rate)
            and input_channels == output_channels
        ):
            return audio_full_int16.copy()
        audio_float = (
            audio_full_int16.reshape(-1, input_channels).astype(np.float32)
            / 32768.0
        ).mean(axis=1)
        dry_output = audio_tools.resample_audio(
            audio_float,
            sample_rate,
            output_sample_rate,
            target_channels=output_channels,
            input_channels=1,
            dtype="float32",
        )
        if strength <= 0.0 or audio_float.size == 0:
            return float32_to_pcm16(dry_output)

        enhanced_sample_rate = int(self._df_state_kwargs["sr"])
        model_audio = audio_tools.resample_audio(
            audio_float,
            sample_rate,
            enhanced_sample_rate,
            target_channels=1,
            input_channels=1,
            dtype="float32",
        )

        # libDF's analysis/synthesis state retains overlap and normalization
        # history. A fresh state per independent window prevents one route or
        # utterance from bleeding into another. Upstream notes that reset() is
        # not equivalent to a newly initialized normalization state.
        with self._inference_lock:
            df_state = DF(**self._df_state_kwargs)
            audio_tensor = torch.from_numpy(
                np.ascontiguousarray(model_audio, dtype=np.float32)
            ).unsqueeze(0)
            enhanced_audio = enhance(
                self.df_model,
                df_state,
                audio_tensor,
                pad=True,
            )
            enhanced_audio = (
                torch.as_tensor(enhanced_audio)
                .detach()
                .cpu()
                .squeeze()
                .numpy()
                .astype(np.float32, copy=False)
            )

        wet_output = audio_tools.resample_audio(
            enhanced_audio,
            enhanced_sample_rate,
            output_sample_rate,
            target_channels=output_channels,
            input_channels=1,
            dtype="float32",
        )
        target_length = min(dry_output.size, wet_output.size)
        mixed = np.asarray(dry_output, dtype=np.float32).copy()
        mixed[:target_length] = (
            dry_output[:target_length] * (1.0 - strength)
            + wet_output[:target_length] * strength
        )
        return float32_to_pcm16(mixed)
