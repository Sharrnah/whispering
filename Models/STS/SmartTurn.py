import os
from pathlib import Path

import numpy as np

import onnxruntime as ort
import torch
from transformers import WhisperFeatureExtractor

import downloader

cache_path = Path(Path.cwd() / ".cache" / "smart-turn")
os.makedirs(cache_path, exist_ok=True)

# dl_server = {
#     "urls": [
#         "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/smart-turn/smart-turn3.2.zip",
#         "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/smart-turn/smart-turn3.2.zip",
#         "https://s3.libs.space:9000/ai-models/smart-turn/smart-turn3.2.zip"
#     ],
#     "sha256": "3aa2e455ee805de3d49ebce9dd5006f47f7fcf31ebb7c015a2b732fbcbf54aa5",
#     "zip_file_name": "smart-turn3.2.zip",
#     "path": str(cache_path / "smart-turn-v3.2-cpu.onnx")
# }

MODEL_LINKS = {
    # Models
    "smart-turn3.2": {
        "urls": [
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/smart-turn/smart-turn3.2.zip",
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/smart-turn/smart-turn3.2.zip",
            "https://s3.libs.space:9000/ai-models/smart-turn/smart-turn3.2.zip"
        ],
        "checksum": "3aa2e455ee805de3d49ebce9dd5006f47f7fcf31ebb7c015a2b732fbcbf54aa5",
        "file_checksums": {
            "smart-turn-v3.2-cpu.onnx": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
            "smart-turn-v3.2-gpu.onnx": "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be"
        },
        "path": "",
    }
}

def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Pad with zeros at the beginning
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    return audio_array


class SmartTurn:
    feature_extractor = None
    session = None
    audio_array = np.array([])

    device = "cpu"
    providers = None

    sample_rate = 16000
    min_audio_length = 2  # in seconds

    download_state = {"is_downloading": False}

    def __init__(self, sample_rate=16000, min_audio_length=2):
        self.sample_rate = sample_rate
        self.min_audio_length = min_audio_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            downloader.download_model({
                "model_path": cache_path,
                "model_link_dict": MODEL_LINKS,
                "model_name": "smart-turn3.2",
                "title": "Smart Turn",

                "alt_fallback": False,
                "force_non_ui_dl": False,
                "extract_format": "zip",
            }, self.download_state)
        except Exception as e:
            print("Error loading smart turn model.")
            print(e)

        if self.feature_extractor is None:
            self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        if self.session is None:
            self.providers = ["CPUExecutionProvider"]
            try:
                avail = ort.get_available_providers()
                if self.device == "cuda" and "CUDAExecutionProvider" in avail:
                    print("Loading Smart Turn using ONNX Runtime with CUDAExecutionProvider")
                    self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self.session = self.build_session(str(Path(cache_path / "smart-turn-v3.2-gpu.onnx").resolve()))
                else:
                    print("Loading Smart Turn using ONNX Runtime with CPUExecutionProvider")
                    self.providers = ["CPUExecutionProvider"]
                    self.session = self.build_session(str(Path(cache_path / "smart-turn-v3.2-cpu.onnx").resolve()))
            except Exception as e:
                print("Error initializing ONNX Runtime session for Smart Turn.")
                print(e)

    def set_min_audio_length(self, min_audio_length: float):
        """Set the minimum audio length required for prediction."""
        self.min_audio_length = min_audio_length

    def build_session(self, onnx_path):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return ort.InferenceSession(onnx_path, providers=self.providers, sess_options=so)

    def clear_session(self):
        """Clear the current audio session."""
        self.audio_array = np.array([])  # Reset audio array to empty

    def predict(self, audio_array: np.ndarray, probability_threshold: float = 0.5) -> dict:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            probability_threshold: Threshold for classifying completion
            audio_array: Numpy array containing audio samples at 16kHz

        Returns:
            Dictionary containing prediction results:
            - prediction: 1 for complete, 0 for incomplete
            - probability: Probability of completion (sigmoid output)
        """
        # Append new audio to existing audio array
        self.audio_array = np.concatenate((self.audio_array, audio_array), axis=0)

        # Check if audio length is sufficient
        if len(self.audio_array) < self.min_audio_length * self.sample_rate:
            return {
                "prediction": False,
                "probability": 0.0,
            }

        # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
        self.audio_array = truncate_audio_to_last_n_seconds(self.audio_array, n_seconds=8, sample_rate=self.sample_rate)

        # Process audio using Whisper's feature extractor
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=8 * self.sample_rate,
            truncation=True,
            do_normalize=True,
        )

        # Extract features and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

        # Run ONNX inference
        outputs = self.session.run(None, {"input_features": input_features})

        # Extract probability (ONNX model returns sigmoid probabilities)
        probability = outputs[0][0].item()

        # Make prediction (True for Complete, False for Incomplete)
        prediction = True if probability > probability_threshold else False

        return {
            "prediction": prediction,
            "probability": probability,
        }
