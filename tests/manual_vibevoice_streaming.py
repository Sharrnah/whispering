"""Run real local-only VibeVoice streaming inference without touching dependencies.

Example: venv\Scripts\python.exe tests/manual_vibevoice_streaming.py --audio Andrew_Dessler.wav
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from Models.STT.vibevoice_asr_streaming import DEFAULT_MODEL, VibeVoiceStreamingASR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--context", default="")
    parser.add_argument("--paced", action="store_true", help="Deliver new snapshots at recording speed")
    args = parser.parse_args()
    audio, rate = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    if not len(audio):
        raise ValueError("The audio file is empty.")
    adapter = VibeVoiceStreamingASR(args.precision, args.device)
    started = time.monotonic()
    adapter.load_model(args.model)
    print(json.dumps({"ready_seconds": time.monotonic() - started}), flush=True)
    started = time.monotonic()
    try:
        for end in range(4000, len(audio) + 4000, 4000):
            end = min(end, len(audio))
            if args.paced:
                time.sleep(max(0, started + end / 16000 - time.monotonic()))
            for result in adapter.process_snapshot(
                audio[:end], stream_id="manual-test", final=end == len(audio), context=args.context,
            ):
                print(json.dumps({"elapsed_seconds": time.monotonic() - started,
                                  "received_seconds": end / 16000, **result}, ensure_ascii=False), flush=True)
        print(json.dumps({"inference_seconds": time.monotonic() - started,
                          "audio_seconds": len(audio) / 16000,
                          "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2
                          if args.device.startswith("cuda") else None}), flush=True)
    finally:
        adapter.release_model()


if __name__ == "__main__":
    main()
