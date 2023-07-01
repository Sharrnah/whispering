from transformers import SpeechT5Processor, SpeechT5ForSpeechToText

from pathlib import Path
import torch

model_cache_path = Path(".cache/speecht5-cache")


class SpeechT5STT:
    model = None
    processor = None
    device = None

    def __init__(self, device="cpu"):
        self.device = device
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None:
            self.load_model()

    def load_model(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr",
                                                           cache_dir=str(model_cache_path.resolve()))
        self.model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr",
                                                             cache_dir=str(model_cache_path.resolve()))
        self.model.to(self.device)

    def transcribe(self, audio_sample) -> dict:

        inputs = self.processor(audio=audio_sample, sampling_rate=16000, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        predicted_ids = self.model.generate(**inputs, max_length=100)

        transcription = self.processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)

        result = {
            'text': transcription[0],
            'type': "transcribe",
            'language': "en"
        }

        return result
