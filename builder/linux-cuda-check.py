"""Verify the selected build runtime without requiring a GPU or CUDA driver."""
import argparse
import importlib.metadata
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--flavor', choices=('cpu', 'cu128'), required=True)
parser.add_argument('--output', type=Path)
args = parser.parse_args()

import torch
import torchaudio
import torchvision
import onnxruntime
import ctranslate2

expected_cuda = '12.8' if args.flavor == 'cu128' else None
if torch.version.cuda != expected_cuda:
    raise RuntimeError(f'Expected CUDA {expected_cuda!r}, found {torch.version.cuda!r}')
for name, module in [('torch', torch), ('torchaudio', torchaudio), ('torchvision', torchvision)]:
    if not module.__version__.endswith('+' + args.flavor):
        raise RuntimeError(f'{name} has the wrong build flavor: {module.__version__}')
if args.flavor == 'cu128':
    if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        raise RuntimeError('ONNX Runtime was built without CUDA support')
    try:
        importlib.metadata.distribution('onnxruntime')
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError('CPU ONNX Runtime must not coexist with the GPU distribution')
    for name in ('nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'nvidia-cuda-runtime-cu12',
                 'nvidia-cuda-nvrtc-cu12', 'nvidia-nvjitlink-cu12', 'triton'):
        importlib.metadata.version(name)
report = json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                  # Diagnostic of the pinned wheel, independent of visible GPUs.
                  'cuda_architectures': torch._C._cuda_getArchFlags().split() if expected_cuda else [],
                  'cudnn': torch.backends.cudnn.version(),
                  'onnxruntime': onnxruntime.__version__,
                  'onnx_providers': onnxruntime.get_available_providers(),
                  'ctranslate2': ctranslate2.__version__,
                  'gpu_visible': torch.cuda.is_available()}, indent=2)
print(report)
if args.output:
    args.output.write_text(report + '\n')
