import numpy as np
import torch
from scipy.io import wavfile

import audio_tools
from Models.Singleton import SingletonMeta
import noisereduce as nr
# from noisereduce.torchgate import TorchGate as TG

class Noisereduce(metaclass=SingletonMeta):
    #torch_gate = None
    def __init__(self):
        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Create TorchGating instance
        #self.torch_gate = TG(sr=8000, nonstationary=True).to(device)
        pass

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def enhance_audio(self, audio_bytes, sample_rate=16000, output_sample_rate=16000, input_channels=1, output_channels=1, strength=1.0):
        audio_full_int16 = np.frombuffer(audio_bytes, np.int16)
        audio_bytes = self.int2float(audio_full_int16)

        audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.float32).unsqueeze_(0)
        # reduce noise on tensor
        enhanced_audio = nr.reduce_noise(y=audio_tensor, sr=sample_rate, prop_decrease=strength)

        # convert torch tensor to bytes
        enhanced_audio = torch.as_tensor(enhanced_audio)

        if enhanced_audio.ndim == 1:
            enhanced_audio.unsqueeze_(0)

        if enhanced_audio.dtype != torch.int16:
            enhanced_audio = (enhanced_audio * (1 << 15)).to(torch.int16)
        elif enhanced_audio.dtype != torch.float32:
            enhanced_audio = enhanced_audio.to(torch.float32) / (1 << 15)

        enhanced_audio = enhanced_audio.squeeze().numpy().astype(np.int16).tobytes()

        audio_bytes = audio_tools.resample_audio(enhanced_audio, sample_rate, output_sample_rate, output_channels,
                                                 input_channels=input_channels)

        # clear variables
        enhanced_audio = None
        del enhanced_audio
        audio_tensor = None
        del audio_tensor
        audio_full_int16 = None
        del audio_full_int16

        return audio_bytes

    def noise_reduction_file(self, path) -> bytes:
        """
        Perform noise reduction on an audio file and save the output.

        This function reads an audio file from the given path, performs noise reduction using the noisereduce library,
        and saves the processed audio to a new file.

        Args:
            path (str): Path to the input audio file.
                Example: "path/to/input_audio.wav"

        Returns:
            bytes wav

        Example usage:
            noise_reduction("input.wav")
        """
        rate, data = wavfile.read(path)
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate)

        # convert numpy data to wav bytes
        wav_bytes = audio_tools.numpy_array_to_wav_bytes(reduced_noise, rate)
        return wav_bytes.getvalue()
