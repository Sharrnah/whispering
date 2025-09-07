# Runtime hook to provide a minimal flash_attn.layers.rotary implementation in frozen builds.
# Ensures mamba_ssm -> flash_attn import works without Triton/JIT.
import sys
import types

try:
    import torch  # noqa: F401
    from torch import nn  # noqa: F401
except Exception:
    # If torch is not available here, do nothing; main app will fail earlier anyway.
    torch = None
    nn = None

# Only install shim if the real flash_attn isn't available
if 'flash_attn' not in sys.modules:
    try:
        import flash_attn  # type: ignore
    except Exception:
        if torch is not None and nn is not None:
            flash_attn_mod = types.ModuleType("flash_attn")
            layers_mod = types.ModuleType("flash_attn.layers")
            rotary_mod = types.ModuleType("flash_attn.layers.rotary")

            class RotaryEmbedding(nn.Module):
                def __init__(self, dim: int, base: float = 10000.0, interleaved: bool = False, **kwargs):
                    super().__init__()
                    self.dim = dim
                    self.interleaved = interleaved
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                    t = torch.arange(16384, dtype=torch.float32)
                    freqs = torch.einsum("i,j->ij", t, inv_freq)
                    self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
                    self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

                def forward(self, x: torch.Tensor, seq_len: int | None = None):
                    if seq_len is None:
                        seq_len = x.shape[1] if x.dim() >= 2 else 0
                    seq_len = max(1, min(seq_len, self.cos_cached.size(0)))
                    cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device).view(seq_len, 1, 1, -1)
                    sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device).view(seq_len, 1, 1, -1)
                    return cos, sin

            rotary_mod.RotaryEmbedding = RotaryEmbedding
            layers_mod.rotary = rotary_mod
            flash_attn_mod.layers = layers_mod

            sys.modules.setdefault("flash_attn", flash_attn_mod)
            sys.modules.setdefault("flash_attn.layers", layers_mod)
            sys.modules.setdefault("flash_attn.layers.rotary", rotary_mod)

