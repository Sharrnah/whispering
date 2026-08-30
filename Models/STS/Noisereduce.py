import numpy as np
from scipy.io import wavfile

import audio_tools
from Models.Singleton import SingletonMeta
from Models.STS.AudioEnhancer import float32_to_pcm16
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
        """Convert PCM to normalized float audio without boosting its noise floor."""
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

        if strength <= 0.0 or audio_float.size == 0:
            enhanced_audio = audio_float
        else:
            enhanced_audio = nr.reduce_noise(
                y=audio_float,
                sr=sample_rate,
                prop_decrease=strength,
                use_tqdm=False,
            )
            enhanced_audio = np.asarray(enhanced_audio, dtype=np.float32).reshape(-1)

        enhanced_audio = audio_tools.resample_audio(
            enhanced_audio,
            sample_rate,
            output_sample_rate,
            target_channels=output_channels,
            input_channels=1,
            dtype="float32",
        )
        return float32_to_pcm16(enhanced_audio)

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
