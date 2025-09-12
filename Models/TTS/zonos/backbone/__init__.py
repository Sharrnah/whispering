BACKBONES = {}

def _has_cuda_toolkit_on_windows() -> bool:
    import os, sys
    if sys.platform != "win32":
        return True  # not our concern here
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    lib_ok = False
    inc_ok = False
    if cuda_path:
        lib_ok = os.path.exists(os.path.join(cuda_path, "lib", "x64", "cuda.lib"))
        inc_ok = os.path.exists(os.path.join(cuda_path, "include", "cuda.h"))
    return bool(lib_ok and inc_ok)

TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

CUDA_OK = _has_cuda_toolkit_on_windows()

if not (TRITON_AVAILABLE and CUDA_OK):
    # Either disable Triton paths or select a CPU code path before importing modules that require Triton
    USE_TRT_BACKENDS = False
else:
    USE_TRT_BACKENDS = True

if USE_TRT_BACKENDS:
    try:
        from ._mamba_ssm import MambaSSMZonosBackbone

        BACKBONES["mamba_ssm"] = MambaSSMZonosBackbone
    except ImportError:
        pass
else:
    print("Triton is not available or CUDA toolkit is not properly installed on Windows. Triton-based backbones are disabled.")
    import warnings
    warnings.warn("Triton is not available or CUDA toolkit is not properly installed on Windows. Triton-based backbones are disabled.", ImportWarning)

from ._torch import TorchZonosBackbone

BACKBONES["torch"] = TorchZonosBackbone
