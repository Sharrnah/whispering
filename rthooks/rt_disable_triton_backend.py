# rthooks/rt_disable_triton_backend.py
import os

# Disable Triton CUDA backend entirely so it won't try to compile CPython/PYD at runtime.
os.environ.setdefault("TRITON_DISABLE_BACKENDS", "nvidia")

# Strongly discourage libraries from using Triton/Flash-Attention paths:
# (These are safe no-ops if a given lib doesn't check them.)
os.environ.setdefault("HF_FLASH_ATTENTION_FORCE_DISABLE", "1")  # huggingface Transformers
os.environ.setdefault("FLASH_ATTENTION_SKIP", "1")               # some integrations
os.environ.setdefault("PYTORCH_SDP_FORCE_FALLBACK", "1")         # force SDP fallback

os.environ.setdefault("TRANSFORMERS_NO_TRITON", "1")

# Optional: ensure no stray NVIDIA compiler selection
os.environ.setdefault("TRITON_WINDOWS_COMPILER", "cl")
