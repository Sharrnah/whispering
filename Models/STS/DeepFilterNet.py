from df.enhance import enhance, init_df
import numpy as np
import torch
import audio_tools
from typing import Union


class DeepFilterNet:
    df_model = None
    df_state = None

    def __init__(self, post_filter=False, epoch: Union[str, int, None] = "best"):
        self.df_model, self.df_state, _ = init_df(post_filter=post_filter, epoch=epoch)
        pass

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def enhance_audio(self, audio_bytes, sample_rate=16000):
        audio_bytes = audio_tools.resample_audio(audio_bytes, sample_rate, self.df_state.sr(), -1,
                                                 is_mono=True).tobytes()

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

        audio_bytes = audio_tools.resample_audio(enhanced_audio, self.df_state.sr(), sample_rate, -1,
                                                 is_mono=True).tobytes()
        return audio_bytes
