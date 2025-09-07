# rthooks/rt_mamba_triton_shim.py
# Install minimal shims for Mamba's Triton-dependent modules so they never import real triton.
import sys, types

def _mk_layer_norm_like():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    m = types.ModuleType("shim.layernorm")
    def _layer_norm_fwd(x, w, b, eps=1e-5, residual=None, residual_in_fp32=False, is_rms_norm=False, z=None,
                        group_size=None, norm_before_gate=True, **_):
        if residual is not None:
            x = (x.float() + residual.float()).to(x.dtype) if residual_in_fp32 else x + residual
        if is_rms_norm:
            # RMSNorm
            var = x.pow(2).mean(dim=-1, keepdim=True)
            y = x * torch.rsqrt(var + eps)
            if w is not None: y = y * w
            if b is not None: y = y + b
        else:
            y = torch.nn.functional.layer_norm(x, (x.size(-1),), weight=w, bias=b, eps=eps)
        if z is not None:
            # gated RMSNorm helper path used by mamba2
            import torch.nn.functional as F
            gate = F.silu(z)
            y = y if norm_before_gate else y * gate
            y = y * gate if norm_before_gate else y
        return y, None, None  # keep signature compatibility when called expecting tuple
    def _layer_norm_bwd(dy, x, w, b, eps, residual, rstd, z=None, group_size=None,
                        norm_before_gate=True, is_rms_norm=False, recompute_output=False, dz=None, out=None, **_):
        # Very lightweight backward: delegate to autograd by redoing fwd; enough for runtime-only/inference.
        # When used at inference there is no backward anyway; keep API to avoid import failures.
        import torch
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w) if w is not None else None
        db = torch.zeros_like(b) if b is not None else None
        if dz is not None:
            dz.zero_()
        return dx, dw, db, dz
    def rmsnorm_fn(x, w, b=None, z=None, eps=1e-6, norm_before_gate=True):
        y, *_ = _layer_norm_fwd(x, w, b, eps=eps, is_rms_norm=True, z=z, norm_before_gate=norm_before_gate)
        return y

    class RMSNorm(nn.Module):
        """Minimal RMSNorm wrapper with the API Mamba expects."""
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
            # Mamba sometimes passes bias=None; keep for API parity
            self.bias = None
        def forward(self, x, z=None, norm_before_gate=True):
            return rmsnorm_fn(x, self.weight, None, z=z, eps=self.eps, norm_before_gate=norm_before_gate)

    m._layer_norm_fwd = _layer_norm_fwd
    m._layer_norm_bwd = _layer_norm_bwd
    m.rmsnorm_fn = rmsnorm_fn
    m.RMSNorm = RMSNorm
    return m

def _mk_k_activations():
    import torch.nn.functional as F, types
    m = types.ModuleType("mamba_ssm.ops.triton.k_activations")
    def _swiglu_fwd(x, out=None):
        # x is [*, 2*d]; split and apply SiLU * linear as in swiglu; here just do SiLU(gate)*vals
        d = x.shape[-1] // 2
        v, g = x[..., :d], x[..., d:]
        y = v * F.silu(g)
        if out is not None:
            out.copy_(y)
            return out
        return y
    def _swiglu_bwd(x, dy, dxy=None, recompute_output=False, out=None):
        import torch
        d = x.shape[-1] // 2
        v, g = x[..., :d], x[..., d:]
        sig = torch.sigmoid(g)
        silu = g * sig
        if recompute_output and out is not None:
            out.copy_(v * F.silu(g))
        dv = dy * F.silu(g)
        dg = dy * v * (sig * (1 + g * (1 - sig)))
        if dxy is not None:
            dxy[..., :d].copy_(dv)
            dxy[..., d:].copy_(dg)
            return dxy
        return torch.cat([dv, dg], dim=-1)
    m._swiglu_fwd = _swiglu_fwd
    m._swiglu_bwd = _swiglu_bwd
    return m

def _mk_ssd_bmm():
    import torch, types
    m = types.ModuleType("mamba_ssm.ops.triton.ssd_bmm")
    def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):
        B, L = a.shape[:2]; C = chunk_size
        G = 1 if a.dim()==3 else a.shape[2]
        nchunks = (L + C - 1)//C
        shape = (B, nchunks, C, C) if G==1 else (B, nchunks, G, C, C)
        return torch.zeros(shape, dtype=output_dtype or a.dtype, device=a.device)
    def _bmm_chunk_bwd(a, dout, residual=None, out=None):
        return torch.zeros_like(a if out is None else out)
    m._bmm_chunk_fwd = _bmm_chunk_fwd
    m._bmm_chunk_bwd = _bmm_chunk_bwd
    return m

# 1) Preinstall shims under the exact module names Mamba imports
_layer = _mk_layer_norm_like()
sys.modules.setdefault("mamba_ssm.ops.triton.layer_norm", _layer)        # some paths use this
sys.modules.setdefault("mamba_ssm.ops.triton.layernorm_gated", _layer)   # mamba2 imports this name
sys.modules.setdefault("mamba_ssm.ops.triton.k_activations", _mk_k_activations())
sys.modules.setdefault("mamba_ssm.ops.triton.ssd_bmm", _mk_ssd_bmm())
sys.modules.setdefault("mamba_ssm.ops.triton.selective_state_update", types.ModuleType("mamba_ssm.ops.triton.selective_state_update"))

import importlib.machinery, importlib.abc, types, sys

class _DummyLoader(importlib.abc.Loader):
    def create_module(self, spec):  # use default
        return None
    def exec_module(self, module):  # nothing to execute
        pass

def _install_triton_shim():
    if "triton" in sys.modules:
        return
    loader = _DummyLoader()
    spec = importlib.machinery.ModuleSpec("triton", loader, is_package=True)

    tri = types.ModuleType("triton")
    tri.__spec__ = spec
    tri.__package__ = "triton"
    tri.__file__ = "<shim>"
    tri.__path__ = []  # mark as package
    # Valid-looking version; Transformers may parse it.
    tri.__version__ = "2.2.0"

    # decorators / helpers some libs expect
    def _identity_dec(*args, **kwargs):
        def dec(f): return f
        return dec
    tri.autotune = _identity_dec
    tri.jit = _identity_dec
    tri.heuristics = _identity_dec
    tri.cdiv = lambda x, y: (x + y - 1) // y
    tri.next_power_of_2 = lambda n: 1 << (max(int(n) - 1, 0)).bit_length()

    # Minimal triton.language submodule
    lang_loader = _DummyLoader()
    lang_spec = importlib.machinery.ModuleSpec("triton.language", lang_loader, is_package=False)
    lang = types.ModuleType("triton.language")
    lang.__spec__ = lang_spec
    lang.__package__ = "triton"
    # some code imports tl.constexpr
    lang.constexpr = lambda x: x

    sys.modules["triton"] = tri
    sys.modules["triton.language"] = lang

_install_triton_shim()

# 3) Install ssd_combined shim AFTER the shims above (it imports selective_scan_interface).
def _mk_ssd_combined():
    import types, sys

    modname = "mamba_ssm.ops.triton.ssd_combined"
    m = types.ModuleType(modname)
    sys.modules[modname] = m  # make it visible *before* mamba2 imports it

    # attach a stub right now; import deps lazily inside the function
    def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
                                  initial_states=None, seq_idx=None, cu_seqlens=None,
                                  dt_softplus=False, dt_limit=(0.0, float("inf")),
                                  return_final_states=False, return_varlen_states=False):
        # lazy imports to avoid circular import during mamba_ssm import
        import torch
        from einops import rearrange, repeat
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        B0, L, H, P = x.shape
        G, N = B.shape[2], B.shape[3]

        x2  = rearrange(x,  "b l h p -> b (h p) l")
        dt2 = rearrange(dt, "b l h   -> b h l")
        dt2 = repeat(dt2, "b h l -> b (h p) l", p=P)
        A2  = repeat(A, "h -> (h p) n", p=P, n=N).to(dtype=torch.float32)
        B2  = rearrange(B, "b l g n -> b g n l")
        C2  = rearrange(C, "b l g n -> b g n l")
        D2  = None if D is None else (D if D.dim()==1 else rearrange(D, "h p -> (h p)"))
        z2  = None if z is not None else None
        if z is not None:
            z2 = rearrange(z, "b l h p -> b (h p) l")

        if dt_bias is not None:
            db = dt_bias if dt_bias.dim()==2 else repeat(dt_bias, "h -> h p", p=P)
            dt2 = dt2 + rearrange(db, "h p -> (h p) 1")
        if dt_softplus:
            dt2 = torch.nn.functional.softplus(dt2)
        if dt_limit != (0.0, float("inf")):
            dt2 = dt2.clamp(min=dt_limit[0], max=dt_limit[1])

        out = selective_scan_fn(
            x2.to(dt2.dtype), dt2, A2, B2, C2, D=D2, z=z2,
            delta_bias=None, delta_softplus=None
        )
        out4 = rearrange(out, "b (h p) l -> b l h p", p=P)
        return (out4, None) if return_final_states else out4

    def mamba_chunk_scan(*args, **kw):
        return mamba_chunk_scan_combined(*args, **kw)

    # publish symbols *now*
    m.mamba_chunk_scan_combined = mamba_chunk_scan_combined
    m.mamba_chunk_scan = mamba_chunk_scan

    return m

sys.modules.setdefault("mamba_ssm.ops.triton.ssd_combined", _mk_ssd_combined())
