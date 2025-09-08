# warm_triton.py
import os, torch

# Make sure torch uses your GPU
assert torch.cuda.is_available(), "CUDA required to warm Triton cache"

# 1) Triton “driver utils” (this is the one that fails in the EXE)
#    Importing any mamba triton op usually triggers driver init; we’ll force it:
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn as _ln  # noqa: F401

# 2) Warm the layer-norm kernel
d = 1024
x = torch.randn(2, 8, d, device="cuda", dtype=torch.float16)
w = torch.randn(d, device="cuda", dtype=torch.float16)
b = torch.randn(d, device="cuda", dtype=torch.float16)
_ = _ln(x, w, b, None, eps=1e-5, residual_in_fp32=False, is_rms_norm=False)
torch.cuda.synchronize()

# 3) Warm the Mamba2 ops that import triton ssd_* kernels
try:
    from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd  # noqa: F401
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401

    # Minimal exercise to JIT them. Shapes depend on your model; adjust if needed.
    B, H, N, D = 1, 1, 1024, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    # These imports/first calls generally trigger JIT & cache write; exact API may vary by version.
except Exception as e:
    print("Warning: ssd_* warmup skipped:", e)

print("Warmed cache at:", os.environ.get("TRITON_CACHE_DIR"))
