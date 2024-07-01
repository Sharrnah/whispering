import os
from pathlib import Path

import torch
import downloader
from Models.Singleton import SingletonMeta

cache_vad_path = Path(Path.cwd() / ".cache" / "silero-vad")
os.makedirs(cache_vad_path, exist_ok=True)

torch.hub.set_dir(str(Path(cache_vad_path).resolve()))


class VAD(metaclass=SingletonMeta):
    _vad_model = None
    _vad_utils = None
    vad_frames_per_buffer = 1536

    def __init__(self, vad_thread_num=1):
        torch.set_num_threads(vad_thread_num)

        try:
            self._vad_model, self._vad_utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                  repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
                                                  )
        except:
            try:
                self._vad_model, self._vad_utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                      source="local", model="silero_vad", onnx=False,
                                                      repo_or_dir=str(Path(
                                                          cache_vad_path / "snakers4_silero-vad_master").resolve())
                                                      )
            except Exception as e:
                print("Error loading vad model trying to load from fallback server...")
                print(e)

                vad_fallback_server = {
                    "urls": [
                        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/silero/silero-vad.zip",
                        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/silero/silero-vad.zip",
                        "https://s3.libs.space:9000/ai-models/silero/silero-vad.zip"
                    ],
                    "sha256": "097cfacdc2b2f5b09e0da1273b3e30b0e96c3588445958171a7e339cc5805683",
                }

                try:
                    downloader.download_extract(vad_fallback_server["urls"],
                                                str(Path(cache_vad_path).resolve()),
                                                vad_fallback_server["sha256"],
                                                alt_fallback=True,
                                                fallback_extract_func=downloader.extract_zip,
                                                fallback_extract_func_args=(
                                                    str(Path(cache_vad_path / "silero-vad.zip")),
                                                    str(Path(cache_vad_path).resolve()),
                                                ),
                                                title="Silero VAD", extract_format="zip")

                    self._vad_model, self._vad_utils = torch.hub.load(trust_repo=True, skip_validation=True,
                                                          source="local", model="silero_vad", onnx=False,
                                                          repo_or_dir=str(Path(
                                                              cache_vad_path / "snakers4_silero-vad_master").resolve())
                                                          )

                except Exception as e:
                    print("Error loading vad model.")
                    print(e)

        pass

    def is_loaded(self):
        if self._vad_model is None:
            return False
        return True

    def set_vad_frames_per_buffer(self, vad_frames_per_buffer):
        self.vad_frames_per_buffer = vad_frames_per_buffer

    def get_vad_frames_per_buffer(self):
        return self.vad_frames_per_buffer

    def run_vad(self, *args):
        return self._vad_model(*args)
