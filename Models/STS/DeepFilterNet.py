import os
from pathlib import Path

from df.enhance import enhance, init_df
from Models.Singleton import SingletonMeta
import numpy as np
import torch
import audio_tools
from typing import Union

import downloader

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

        # original part (downloads to %LOCALAPPDATA%\DeepFilterNet)
        # self.df_model, self.df_state, _ = init_df(post_filter=post_filter, epoch=epoch, log_level="none")
        pass

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def enhance_audio(self, audio_bytes, sample_rate=16000, output_sample_rate=16000, input_channels=1, output_channels=1, strength=1.0):
        enhanced_sample_rate = self.df_state.sr()
        audio_bytes = audio_tools.resample_audio(audio_bytes, sample_rate, enhanced_sample_rate, 1,
                                                 input_channels=input_channels).tobytes()

        audio_full_int16 = np.frombuffer(audio_bytes, np.int16)
        audio_bytes = self.int2float(audio_full_int16)

        audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.float32).unsqueeze_(0)
        # convert bytes to torch tensor
        enhanced_audio = enhance(self.df_model, self.df_state, torch.as_tensor(audio_tensor))
        # convert torch tensor to bytes
        enhanced_audio = torch.as_tensor(enhanced_audio)

        if enhanced_audio.ndim == 1:
            enhanced_audio.unsqueeze_(0)

        if enhanced_audio.dtype != torch.int16:
            enhanced_audio = (enhanced_audio * (1 << 15)).to(torch.int16)
        elif enhanced_audio.dtype != torch.float32:
            enhanced_audio = enhanced_audio.to(torch.float32) / (1 << 15)

        enhanced_audio = enhanced_audio.squeeze().numpy().astype(np.int16).tobytes()

        audio_bytes = audio_tools.resample_audio(enhanced_audio, enhanced_sample_rate, output_sample_rate,
                                                 input_channels=1,
                                                 target_channels=output_channels)

        # clear variables
        enhanced_audio = None
        del enhanced_audio
        audio_tensor = None
        del audio_tensor
        audio_full_int16 = None
        del audio_full_int16

        return audio_bytes
