"""Derive the Linux stack from current shared pins; never install on the host."""
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
flavor = sys.argv[3] if len(sys.argv) > 3 else "cu128"
# Windows distributions and the duplicate ONNX namespace are not Linux inputs.
omit = {"pyaudiowpatch", "pywin32", "winsdk", "triton-windows", "onnxruntime",
        "onnxruntime-genai-cuda", "jitaer"}
lines = ["# Generated from requirements.txt by builder/linux-requirements.py"]
for line in source.read_text(encoding="utf-8-sig").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name = re.split(r"[<>=@\[\s]", line, maxsplit=1)[0].lower().replace("_", "-")
    if name in omit or line == "tiktoken==0.5.2":
        continue
    # 1.24.3 has no Linux GPU wheel. Use its next patch release on Linux.
    if name == "onnxruntime-gpu":
        line = ("onnxruntime-gpu" if flavor == "cu128" else "onnxruntime") + "==1.24.4"
    if name == "descript-audiotools":
        line = "descript-audiotools @ file:///tmp/linux-wheels/patched/descript_audiotools-0.7.2%2Bwtlinux1-py2.py3-none-any.whl"
    if line.startswith("git+https://github.com/NVIDIA/NeMo.git"):
        line = "nemo-toolkit[asr] @ file:///tmp/linux-wheels/patched/nemo_toolkit-2.6.0%2Bwtlinux1-py3-none-any.whl"
    if name == "protobuf":
        line = "protobuf==5.29.5"
    if name == "wandb":
        line = "wandb==0.19.11"
    if flavor == "cu128" and name in {"faster-whisper", "ruaccent"}:
        wheels = sorted(Path('/tmp/linux-wheels/patched').glob(name.replace('-', '_') + '-*.whl'))
        if len(wheels) != 1:
            raise RuntimeError(f'Expected one CUDA ONNX metadata wheel for {name}: {wheels}')
        line = f"{name} @ {wheels[0].as_uri()}"
    # spaCy distributes its trained English pipeline on GitHub, not PyPI.
    if name == "en-core-web-sm":
        line = "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    lines.append(line)
lines.extend([f"torch==2.7.1+{flavor}", f"torchaudio==2.7.1+{flavor}", f"torchvision==0.22.1+{flavor}", "cuda-bindings>=12.9.1,<13", "pulsectl==24.12.0"])
lines.extend(["torchdiffeq==0.2.4", "x-transformers==1.40.2", "ema-pytorch==0.7.0", "vocos==0.1.0",
              "Resemblyzer @ file:///tmp/linux-wheels/patched/Resemblyzer-0.1.4%2Bwtlinux1-py3-none-any.whl"])
# NeMo imports numba-cuda's local runtime-version probe even on CPU.
if flavor == "cpu":
    lines.append("nvidia-cuda-runtime-cu12==12.8.90")
# CUDA builds use Torch's exact NVIDIA dependency pins (including cudart).
# webrtcvad (used by Resemblyzer) still imports pkg_resources, removed in 81+.
lines.append("setuptools==80.9.0")
destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
