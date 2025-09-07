# Models/TTS/zonos/backbone/_mamba_ssm.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.mixer_seq_simple import create_block
# *conditionally* import the triton kernel below.

from ..config import BackboneConfig, InferenceParams


def _torch_layer_norm_fn(
        hidden_states: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        residual: torch.Tensor | None,
        eps: float,
        residual_in_fp32: bool,
        is_rms_norm: bool,
) -> torch.Tensor:
    """
    Fallback that mimics the Triton fused add+norm behavior using PyTorch ops.
    Returns the normalized tensor (no residual returned).
    """
    x = hidden_states
    if residual is not None:
        if residual_in_fp32:
            x = (x.float() + residual.float()).to(x.dtype)
        else:
            x = x + residual

    if is_rms_norm:
        # RMSNorm: x * rsqrt(mean(x^2)) with optional weight/bias
        # (Some RMSNorm impls omit bias; we honor both if provided.)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + eps)
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias
        return x_norm
    else:
        # Standard LayerNorm over last dim
        return F.layer_norm(x, normalized_shape=(x.size(-1),), weight=weight, bias=bias, eps=eps)


# Decide whether to use Triton kernel
_USE_TRITON = True
if getattr(sys, "frozen", False):
    # In a PyInstaller EXE, Triton’s NVIDIA backend tries to compile a CPython extension
    # and fails (no pythonXY.lib). Use the Torch fallback.
    _USE_TRITON = False
if os.environ.get("TRITON_DISABLE_BACKENDS") == "nvidia":
    _USE_TRITON = False

if _USE_TRITON:
    try:
        from mamba_ssm.ops.triton.layer_norm import layer_norm_fn as _triton_layer_norm_fn  # type: ignore
        def layer_norm_fn(*args, **kwargs) -> torch.Tensor:
            # Keep a thin wrapper in case signatures diverge later.
            return _triton_layer_norm_fn(*args, **kwargs)
    except Exception:
        # If Triton import fails at runtime, fall back gracefully.
        layer_norm_fn = _torch_layer_norm_fn
else:
    layer_norm_fn = _torch_layer_norm_fn


class MambaSSMZonosBackbone(nn.Module):
    supported_architectures = ["transformer", "hybrid"]

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,
                    d_intermediate=(
                        config.d_intermediate
                        if (i not in config.attn_layer_idx)
                        else config.attn_mlp_d_intermediate
                    ),
                    ssm_cfg=config.ssm_cfg,
                    layer_idx=i,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=True,
                    rms_norm=config.rms_norm,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(
            self,
            batch_size: int,
            max_seqlen: int,
            dtype: torch.dtype = torch.bfloat16,
    ):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams | None = None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        return layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            residual,
            eps=self.norm_f.eps,
            residual_in_fp32=self.config.residual_in_fp32,
            is_rms_norm=self.config.rms_norm,
        )
