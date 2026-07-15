"""Native PyTorch ZONOS2 model, checkpoint loading, prompting, and sampling."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

try:
    from comfy.ops import (
        cast_bias_weight,
        uncast_bias_weight,
    )
except ImportError:
    def cast_bias_weight(module, input=None, **kwargs):
        return module.weight, module.bias, (None, None, None)

    def uncast_bias_weight(module, weight, bias, offload_stream):
        return None


def _linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    operations: Any = None,
) -> nn.Module:
    linear_class = operations.Linear if operations is not None else nn.Linear
    return linear_class(
        in_features,
        out_features,
        bias=bias,
    )


def _embedding(num_embeddings: int, embedding_dim: int) -> nn.Module:
    return nn.Embedding(num_embeddings, embedding_dim)


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger("Zonos2_TTS-ComfyUI")

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_TEXT_VOCAB_SIZE = 448

SILENCE_TOKENS_0_2S = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
]


@dataclass(frozen=True)
class Zonos2Config:
    n_layers: int
    dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    intermediate_size: int
    norm_eps: float
    rope_theta: float
    max_seqlen: int
    n_codebooks: int
    codebook_size: int
    eoa_id: int
    audio_pad_id: int
    text_vocab: int
    loss_softcap: float
    speaker_enabled: bool
    speaker_embedding_dim: int
    speaker_lda_dim: int | None
    speaker_background_token_enabled: bool
    accurate_mode_token_enabled: bool
    speaking_rate_num_buckets: int
    speaking_rate_buckets: tuple[str, ...]
    quality_features: tuple[str, ...]
    quality_bucket_counts: tuple[int, ...]
    moe_n_experts: int
    moe_router_topk: int
    special_topk_layers: dict[int, int]
    moe_router_dim: int
    moe_start_from_layer: int
    moe_end_from_layer: int

    @classmethod
    def from_dict(cls, raw: dict) -> "Zonos2Config":
        dim = int(raw["dim"])
        head_dim = int(raw["head_dim"])
        ffn_dim = int(float(raw.get("ffn_dim_multiplier", 4.0)) * dim)
        multiple = int(raw.get("multiple_of", 256))
        intermediate_size = multiple * ((ffn_dim + multiple - 1) // multiple)
        quality_features = tuple(str(item) for item in raw.get("quality_features", ()))
        quality_buckets = raw.get("quality_buckets", {}) or {}
        return cls(
            n_layers=int(raw["n_layers"]),
            dim=dim,
            head_dim=head_dim,
            n_heads=int(raw.get("n_heads") or (dim // head_dim)),
            n_kv_heads=int(raw.get("n_kv_heads") or (dim // head_dim)),
            intermediate_size=intermediate_size,
            norm_eps=float(raw.get("norm_eps", 1e-5)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_seqlen=int(raw.get("max_seqlen", 6144)),
            n_codebooks=int(raw.get("n_codebooks", 9)),
            codebook_size=int(raw.get("codebook_size", 1024)),
            eoa_id=int(raw.get("eoa_id", 1024)),
            audio_pad_id=int(raw.get("audio_pad_id", 1025)),
            text_vocab=int(raw["text_vocab"]),
            loss_softcap=float(raw.get("loss_softcap", 15.0)),
            speaker_enabled=bool(raw.get("speaker_enabled", False)),
            speaker_embedding_dim=int(raw.get("speaker_embedding_dim", 128)),
            speaker_lda_dim=(
                int(raw["speaker_lda_dim"])
                if raw.get("speaker_lda_dim") is not None
                else None
            ),
            speaker_background_token_enabled=bool(
                raw.get("speaker_background_token_enabled", False)
            ),
            accurate_mode_token_enabled=bool(
                raw.get("accurate_mode_token_enabled", False)
            ),
            speaking_rate_num_buckets=int(
                raw.get("speaking_rate_num_buckets", 0)
            ),
            speaking_rate_buckets=tuple(
                str(item) for item in raw.get("speaking_rate_buckets", ())
            ),
            quality_features=quality_features,
            quality_bucket_counts=tuple(
                len(quality_buckets.get(feature, ()) or ())
                for feature in quality_features
            ),
            moe_n_experts=int(raw.get("moe_n_experts", 1)),
            moe_router_topk=int(raw.get("moe_router_topk", 1)),
            special_topk_layers={
                int(key): int(value)
                for key, value in (raw.get("special_topk_layers", {}) or {}).items()
            },
            moe_router_dim=int(raw.get("moe_router_dim", 128)),
            moe_start_from_layer=int(raw.get("moe_start_from_layer", 0)),
            moe_end_from_layer=int(raw.get("moe_end_from_layer", 0)),
        )

    def is_moe_layer(self, layer_id: int) -> bool:
        if self.moe_n_experts <= 1:
            return False
        if layer_id < self.moe_start_from_layer:
            return False
        return (self.n_layers - layer_id) > self.moe_end_from_layer

    def top_k_for_layer(self, layer_id: int) -> int:
        return int(self.special_topk_layers.get(layer_id, self.moe_router_topk))


def read_config(path: Path) -> Zonos2Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("model_type") != "zonos2":
        raise ValueError(f"Unsupported model_type: {raw.get('model_type')!r}")
    return Zonos2Config.from_dict(raw)


class RMSNorm(nn.RMSNorm):
    def __init__(self, size: int, eps: float):
        super().__init__(size, eps=float(eps))
        self.weight.requires_grad_(False)


class TensorLinear(nn.Module):
    """Linear projection whose checkpoint weight is stored as a rank-3 tensor."""

    def __init__(self, chunks: int, out_per_chunk: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(chunks, out_per_chunk, in_features),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.flatten(0, 1))


class MultiEmbedding(nn.Module):
    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.embedders = nn.ModuleList(
            [
                _embedding(config.codebook_size + 2, config.dim)
                for _ in range(config.n_codebooks)
            ]
            + [_embedding(config.text_vocab + 1, config.dim)]
        )
        self.output_dtype: torch.dtype | None = None

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        def embed(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
            if self.output_dtype is None:
                return module(values)
            try:
                return module(values, out_dtype=self.output_dtype)
            except TypeError:
                return module(values).to(dtype=self.output_dtype)

        result = embed(self.embedders[0], codes[..., 0].long())
        for index in range(1, codes.shape[-1]):
            result = result + embed(
                self.embedders[index],
                codes[..., index].long(),
            )
        return result


class LayerKVCache:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (batch_size, max_length, num_kv_heads, head_dim)
        self.key = torch.empty(shape, device=device, dtype=dtype)
        self.value = torch.empty(shape, device=device, dtype=dtype)
        self.length = 0


def _apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even = pair[..., 0]
    odd = pair[..., 1]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin),
        dim=-1,
    )
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.num_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.wq = _linear(
            config.dim,
            config.n_heads * config.head_dim,
            bias=False,
        )
        self.wkv = TensorLinear(
            2,
            config.n_kv_heads * config.head_dim,
            config.dim,
        )
        self.wo = _linear(
            config.n_heads * config.head_dim,
            config.dim,
            bias=False,
        )
        self.temp = nn.Parameter(
            torch.empty(1, config.n_heads, 1),
            requires_grad=False,
        )
        self.gater = _linear(
            config.dim,
            config.n_heads,
            bias=False,
        )

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        positions = torch.arange(config.max_seqlen, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("_rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("_rope_sin", freqs.sin(), persistent=False)

    def materialize_rope(
        self,
        max_seqlen: int,
        rope_theta: float,
        device: torch.device,
    ) -> None:
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_dim
            )
        )
        positions = torch.arange(
            max_seqlen,
            dtype=torch.float32,
            device=device,
        )
        freqs = torch.outer(positions, inv_freq)
        self._rope_cos = freqs.cos()
        self._rope_sin = freqs.sin()

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        try:
            output = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                dropout_p=0.0,
                is_causal=causal,
                enable_gqa=True,
            )
        except TypeError:
            repeat = self.num_heads // self.num_kv_heads
            output = F.scaled_dot_product_attention(
                q_t,
                k_t.repeat_interleave(repeat, dim=1),
                v_t.repeat_interleave(repeat, dim=1),
                dropout_p=0.0,
                is_causal=causal,
            )
        return output.transpose(1, 2)

    def _flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func

        return flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerKVCache,
        attention_backend: str,
    ) -> torch.Tensor:
        batch, query_length, _ = x.shape
        start = cache.length
        end = start + query_length
        if end > cache.key.shape[1]:
            raise RuntimeError(
                f"ZONOS2 KV cache exhausted: need {end}, "
                f"allocated {cache.key.shape[1]}."
            )

        gate = torch.sigmoid(self.gater(x))
        q = self.wq(x).view(
            batch, query_length, self.num_heads, self.head_dim
        )
        kv = self.wkv(x).view(
            batch, query_length, 2, self.num_kv_heads, self.head_dim
        )
        k = kv[:, :, 0]
        v = kv[:, :, 1]

        q = F.rms_norm(q, (self.head_dim,), eps=1e-6)
        q = q * self.temp.abs().to(device=q.device, dtype=q.dtype)
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)

        cos = self._rope_cos[start:end].to(device=q.device, dtype=q.dtype)
        sin = self._rope_sin[start:end].to(device=q.device, dtype=q.dtype)
        cos = cos.view(1, query_length, 1, self.head_dim // 2)
        sin = sin.view(1, query_length, 1, self.head_dim // 2)
        q = _apply_interleaved_rope(q, cos, sin)
        k = _apply_interleaved_rope(k, cos, sin)

        cache.key[:, start:end].copy_(k)
        cache.value[:, start:end].copy_(v)
        cache.length = end
        full_k = cache.key[:, :end]
        full_v = cache.value[:, :end]
        causal = start == 0 and query_length > 1

        if attention_backend == "flash_attention":
            attended = self._flash(q, full_k, full_v, causal)
        else:
            attended = self._sdpa(q, full_k, full_v, causal)

        attended = attended * gate.unsqueeze(-1)
        return self.wo(attended.reshape(batch, query_length, -1))


class DenseFeedForward(nn.Module):
    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.w_in = TensorLinear(2, config.intermediate_size, config.dim)
        self.w_out = _linear(
            config.intermediate_size,
            config.dim,
            bias=False,
        )
        self.intermediate_size = config.intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.w_in(x)
        up = projected[..., : self.intermediate_size]
        gate = projected[..., self.intermediate_size :]
        return self.w_out(up * F.silu(gate))


class SonicExpert(nn.Module):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.w13_shape = (
            config.intermediate_size * 2,
            config.dim,
        )
        self.w2_shape = (
            config.dim,
            config.intermediate_size,
        )
        self.weight = nn.Parameter(
            torch.empty(self.w13_shape),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(self.w2_shape),
            requires_grad=False,
        )
        self.intermediate_size = config.intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not hasattr(self, "_v")
            and self.weight.device == x.device
            and self.bias.device == x.device
            and not self.weight_function
            and not self.bias_function
        ):
            fused = F.linear(x, self.weight)
            gate = fused[..., 0::2]
            up = fused[..., 1::2]
            return F.linear(F.silu(gate) * up, self.bias)

        weight, bias, offload_stream = cast_bias_weight(
            self,
            x,
            offloadable=True,
        )
        try:
            fused = F.linear(x, weight)
            gate = fused[..., 0::2]
            up = fused[..., 1::2]
            return F.linear(F.silu(gate) * up, bias)
        finally:
            uncast_bias_weight(self, weight, bias, offload_stream)


class FP8Linear(nn.Module):
    """Mixed-FP8 expert projection used by drbaph's ZONOS2 checkpoint."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if bias:
            raise ValueError("ZONOS2 FP8 expert projections do not use bias.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False,
        )
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32))
        # This checkpoint marker is retained for exact safetensors compatibility.
        self.register_buffer("comfy_quant", torch.empty(26, dtype=torch.uint8))
        self._scaled_mm_supported: bool | None = None

    def _can_use_scaled_mm(self, x: torch.Tensor) -> bool:
        if self._scaled_mm_supported is not None:
            return self._scaled_mm_supported
        supported = (
            x.device.type == "cuda"
            and self.weight.dtype == torch.float8_e4m3fn
            and hasattr(torch, "_scaled_mm")
        )
        if supported:
            major, minor = torch.cuda.get_device_capability(x.device)
            supported = (major, minor) >= (8, 9)
        self._scaled_mm_supported = supported
        return supported

    def _scaled_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        activation_scale = (
            flat.abs().amax().float() / fp8_max
        ).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (flat.float() / activation_scale).clamp(
            -fp8_max,
            fp8_max,
        ).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(
            quantized,
            self.weight.t(),
            activation_scale,
            self.weight_scale,
            out_dtype=x.dtype,
        )
        return output.reshape(*original_shape, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._can_use_scaled_mm(x):
            try:
                return self._scaled_forward(x)
            except RuntimeError as exc:
                self._scaled_mm_supported = False
                logger.warning(
                    "Native ZONOS2 FP8 compute failed; using BF16 "
                    "dequantization fallback: %s",
                    exc,
                )
        weight = self.weight.to(dtype=x.dtype) * self.weight_scale.to(dtype=x.dtype)
        return F.linear(x, weight)


class QuantizedSonicExpert(nn.Module):
    def __init__(self, config: Zonos2Config, operations: Any):
        super().__init__()
        del operations
        self.w13 = FP8Linear(
            config.dim,
            config.intermediate_size * 2,
            bias=False,
        )
        self.w2 = _linear(
            config.intermediate_size,
            config.dim,
            bias=False,
        )
        self.intermediate_size = config.intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.w13(x)
        gate = fused[..., 0::2]
        up = fused[..., 1::2]
        return self.w2(F.silu(gate) * up)


class SonicExperts(nn.Module):
    def __init__(self, config: Zonos2Config, operations: Any = None):
        super().__init__()
        self.experts = nn.ModuleList(
            (
                QuantizedSonicExpert(config, operations)
                if operations is not None
                else SonicExpert(config)
            )
            for _ in range(config.moe_n_experts)
        )
        self.num_experts = config.moe_n_experts
        self.packed_w13_shape = (
            config.moe_n_experts,
            config.intermediate_size * 2,
            config.dim,
        )
        self.packed_w2_shape = (
            config.moe_n_experts,
            config.dim,
            config.intermediate_size,
        )

    def _forward_single_token(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        selected = topk_ids[0].tolist()
        output = torch.zeros_like(x)
        ordered_slots = sorted(
            enumerate(selected),
            key=lambda item: item[1],
        )
        for slot, expert_id in ordered_slots:
            expert_output = self.experts[expert_id](x)
            weight = topk_weights[0, slot].to(dtype=expert_output.dtype)
            output.add_(expert_output * weight)
        return output

    def _forward_grouped(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens, hidden = x.shape
        top_k = topk_ids.shape[-1]
        flat_experts = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            assignment = torch.nonzero(
                flat_experts == expert_id,
                as_tuple=False,
            ).flatten()
            if assignment.numel() == 0:
                continue
            token_ids = torch.div(assignment, top_k, rounding_mode="floor")
            expert_input = x.index_select(0, token_ids)
            expert_output = self.experts[expert_id](expert_input)
            weights = flat_weights.index_select(0, assignment).to(
                dtype=expert_output.dtype
            )
            output.index_add_(0, token_ids, expert_output * weights.unsqueeze(-1))
        return output.view(tokens, hidden)

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[0] == 1:
            return self._forward_single_token(x, topk_weights, topk_ids)
        return self._forward_grouped(x, topk_weights, topk_ids)


class Router(nn.Module):
    def __init__(self, config: Zonos2Config, layer_id: int):
        super().__init__()
        self.down_proj = _linear(
            config.dim,
            config.moe_router_dim,
            bias=True,
        )
        self.router_mlp = nn.Sequential(
            _linear(
                config.moe_router_dim,
                config.moe_router_dim,
                bias=True,
            ),
            nn.GELU(),
            _linear(
                config.moe_router_dim,
                config.moe_router_dim,
                bias=True,
            ),
            nn.GELU(),
            _linear(
                config.moe_router_dim,
                config.moe_n_experts,
                bias=False,
            ),
        )
        self.rmsnorm_eda = RMSNorm(config.moe_router_dim, config.norm_eps)
        self.use_eda = layer_id != config.moe_start_from_layer
        if self.use_eda:
            self.router_states_scale = nn.Parameter(
                torch.empty(config.moe_router_dim),
                requires_grad=False,
            )
        self.balancing_biases = nn.Parameter(
            torch.empty(config.moe_n_experts),
            requires_grad=False,
        )
        self.top_k = config.top_k_for_layer(layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.down_proj(hidden_states)
        if self.use_eda and router_states is not None:
            projected = projected + router_states * self.router_states_scale
        next_router_states = projected.clone()
        expert_prob = torch.softmax(
            self.router_mlp(self.rmsnorm_eda(projected)).float(),
            dim=-1,
        )
        routing_scores = expert_prob + self.balancing_biases.float()
        expert_choice = torch.topk(
            routing_scores,
            self.top_k,
            dim=-1,
        ).indices
        route_prob = torch.gather(expert_prob, -1, expert_choice)
        return route_prob, expert_choice, next_router_states


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        config: Zonos2Config,
        layer_id: int,
        operations: Any = None,
    ):
        super().__init__()
        self.router = Router(config, layer_id)
        self.experts = SonicExperts(config, operations=operations)

    def forward(
        self,
        x: torch.Tensor,
        router_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        previous = (
            router_states.reshape(-1, router_states.shape[-1])
            if router_states is not None
            else None
        )
        weights, expert_ids, next_states = self.router(flat, previous)
        output = self.experts(flat, weights, expert_ids)
        return output.view(original_shape), next_states.view(
            *original_shape[:-1], next_states.shape[-1]
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: Zonos2Config,
        layer_id: int,
        expert_operations: Any = None,
    ):
        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.is_moe = config.is_moe_layer(layer_id)
        self.feed_forward = (
            MoEFeedForward(config, layer_id, operations=expert_operations)
            if self.is_moe
            else DenseFeedForward(config)
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerKVCache,
        attention_backend: str,
        router_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.attention(
            self.attention_norm(x),
            cache,
            attention_backend,
        )
        normalized = self.ffn_norm(x)
        if self.is_moe:
            feed_forward, router_states = self.feed_forward(
                normalized,
                router_states,
            )
        else:
            feed_forward = self.feed_forward(normalized)
            router_states = None
        return x + feed_forward, router_states


class Zonos2Model(nn.Module):
    def __init__(
        self,
        config: Zonos2Config,
        expert_operations: Any = None,
        quantization: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.runtime_dtype: torch.dtype | None = None
        self.quantization = quantization
        self.multi_embedder = MultiEmbedding(config)
        if config.speaker_enabled and config.speaker_lda_dim is not None:
            self.speaker_lda_projection = _linear(
                config.speaker_embedding_dim,
                config.speaker_lda_dim,
                bias=True,
            )
            speaker_input = config.speaker_lda_dim
        else:
            self.speaker_lda_projection = None
            speaker_input = config.speaker_embedding_dim
        self.speaker_projection = (
            _linear(speaker_input, config.dim, bias=True)
            if config.speaker_enabled
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    index,
                    expert_operations=expert_operations,
                )
                for index in range(config.n_layers)
            ]
        )
        self.out_norm = RMSNorm(config.dim, config.norm_eps)
        self.multi_output = _linear(
            config.dim,
            config.n_codebooks * (config.codebook_size + 2),
            bias=False,
        )

    def materialize_runtime_buffers(self, device: torch.device) -> None:
        for layer in self.layers:
            layer.attention.materialize_rope(
                self.config.max_seqlen,
                self.config.rope_theta,
                device,
            )

    def create_kv_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[LayerKVCache]:
        return [
            LayerKVCache(
                batch_size,
                max_length,
                self.config.n_kv_heads,
                self.config.head_dim,
                device,
                dtype,
            )
            for _ in range(self.config.n_layers)
        ]

    def _speaker_projection(self, embedding: torch.Tensor) -> torch.Tensor:
        value = embedding
        runtime_dtype = self.runtime_dtype or embedding.dtype
        if self.speaker_lda_projection is not None:
            value = self.speaker_lda_projection(
                value.to(dtype=runtime_dtype)
            )
        if self.speaker_projection is None:
            raise RuntimeError("This ZONOS2 model has no speaker projection.")
        return self.speaker_projection(
            value.to(dtype=runtime_dtype)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: list[LayerKVCache],
        attention_backend: str,
        speaker_embedding: torch.Tensor | None = None,
        speaker_position: int | None = None,
        speaker_emotion_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.multi_embedder(input_ids)
        if speaker_embedding is not None and speaker_position is not None:
            if 0 <= speaker_position < x.shape[1]:
                projected = self._speaker_projection(
                    speaker_embedding.to(device=x.device)
                )
                if speaker_emotion_delta is not None:
                    delta = speaker_emotion_delta.to(
                        device=x.device,
                        dtype=projected.dtype,
                    )
                    if delta.ndim == 1:
                        delta = delta.unsqueeze(0)
                    if delta.shape != projected.shape:
                        raise ValueError(
                            "ZONOS2 projected emotion delta shape "
                            f"{tuple(delta.shape)} does not match speaker projection "
                            f"{tuple(projected.shape)}."
                        )
                    projected = projected + delta
                x[:, speaker_position] = projected.to(dtype=x.dtype)
        x = F.rms_norm(
            x,
            (x.shape[-1],),
            weight=None,
            eps=self.config.norm_eps,
        )

        router_states = None
        for layer, cache in zip(self.layers, caches):
            x, router_states = layer(
                x,
                cache,
                attention_backend,
                router_states,
            )
        hidden = self.out_norm(x[:, -1])
        logits = self.multi_output(hidden).view(
            hidden.shape[0],
            self.config.n_codebooks,
            self.config.codebook_size + 2,
        )
        if self.config.loss_softcap > 0:
            cap = self.config.loss_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits


def _quantized_operations(
    quantization: str,
    compute_dtype: torch.dtype,
    load_device: torch.device,
) -> Any:
    if quantization != "fp8_e4m3":
        raise ValueError(f"Unsupported ZONOS2 quantization: {quantization}")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "This PyTorch build does not support FP8 E4M3 tensors."
        )
    del compute_dtype
    if load_device.type != "cuda":
        logger.warning(
            "Native FP8 E4M3 compute is unavailable on %s; ZONOS2 will "
            "dequantize expert weights for compatible fallback execution.",
            load_device,
        )
    # SonicExperts only needs a non-None marker to construct split mixed-FP8
    # experts. FP8Linear provides the standalone operations.
    return True


def build_native_model(
    config: Zonos2Config,
    quantization: str | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    load_device: torch.device = torch.device("cpu"),
) -> Zonos2Model:
    expert_operations = None
    if quantization is not None:
        expert_operations = _quantized_operations(
            quantization,
            compute_dtype,
            load_device,
        )
    with torch.device("meta"):
        model = Zonos2Model(
            config,
            expert_operations=expert_operations,
            quantization=quantization,
        )
    return model


def set_runtime_dtype(model: Zonos2Model, dtype: torch.dtype) -> None:
    model.runtime_dtype = dtype
    model.multi_embedder.output_dtype = dtype
    for module in model.modules():
        for name, value in module.named_parameters(recurse=False):
            if value.is_floating_point():
                setattr(module, f"{name}_comfy_model_dtype", dtype)
        for name, value in module.named_buffers(recurse=False):
            if value.is_floating_point():
                setattr(module, f"{name}_comfy_model_dtype", dtype)


def validate_quantized_runtime_model(model: Zonos2Model) -> None:
    if model.quantization != "fp8_e4m3":
        return

    expert_count = 0
    invalid: list[str] = []
    for layer_index, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue
        for expert_index, expert in enumerate(
            layer.feed_forward.experts.experts
        ):
            expert_count += 1
            if not isinstance(expert, QuantizedSonicExpert):
                invalid.append(
                    f"layers.{layer_index}.expert.{expert_index}: "
                    f"{expert.__class__.__name__}"
                )
                continue
            for projection_name in ("w13", "w2"):
                projection = getattr(expert, projection_name)
                weight = projection.weight
                if weight.ndim != 2:
                    invalid.append(
                        f"layers.{layer_index}.expert.{expert_index}."
                        f"{projection_name}.weight shape={tuple(weight.shape)}"
                    )
            if not isinstance(expert.w13, FP8Linear):
                invalid.append(
                    f"layers.{layer_index}.expert.{expert_index}.w13 "
                    "is not an FP8Linear"
                )

    if expert_count == 0:
        invalid.append("no MoE experts found")
    if invalid:
        raise RuntimeError(
            "ZONOS2 mixed FP8 runtime validation failed before inference. "
            "The checkpoint may have been produced by the retired all-layer "
            f"FP8 converter. Invalid entries: {invalid[:5]}"
        )


def _sonic_expert_modules(
    model: nn.Module,
) -> dict[str, SonicExperts]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SonicExperts)
    }


def checkpoint_layout(model: Zonos2Model) -> dict[str, tuple[int, ...]]:
    layout = {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
    }
    for prefix, module in _sonic_expert_modules(model).items():
        internal_prefix = f"{prefix}.experts."
        for name in tuple(layout):
            if name.startswith(internal_prefix):
                del layout[name]
        layout[f"{prefix}.w13"] = module.packed_w13_shape
        layout[f"{prefix}.w2"] = module.packed_w2_shape

    for name, module in model.named_modules():
        checkpoint_shape = getattr(module, "checkpoint_weight_shape", None)
        if checkpoint_shape is not None:
            layout[f"{name}.weight"] = tuple(checkpoint_shape)
    return layout


def _set_parameter(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    current = getattr(parent, leaf)
    if current.shape != tensor.shape:
        checkpoint_shape = getattr(
            parent,
            f"checkpoint_{leaf}_shape",
            None,
        )
        if (
            checkpoint_shape is None
            or tuple(tensor.shape) != tuple(checkpoint_shape)
            or tensor.numel() != current.numel()
        ):
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(current.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.reshape(current.shape)
    value = tensor
    if value.is_floating_point():
        value = value.to(dtype=dtype)
    value = value.to(device=device).contiguous()
    setattr(parent, leaf, nn.Parameter(value, requires_grad=False))


def _set_checkpoint_value(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    current = getattr(parent, leaf)
    if current.shape != tensor.shape:
        raise ValueError(
            f"Shape mismatch for {name}: expected {tuple(current.shape)}, "
            f"got {tuple(tensor.shape)}"
        )
    value = tensor.to(device=device).contiguous()
    if isinstance(current, nn.Parameter):
        value = nn.Parameter(value, requires_grad=False)
    setattr(parent, leaf, value)


def validate_checkpoint_layout(
    model: Zonos2Model,
    checkpoint_path: Path,
) -> tuple[int, set[str], set[str]]:
    expected = checkpoint_layout(model)
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        actual_names = set(handle.keys())
        expected_names = set(expected)
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names
        for name in sorted(expected_names & actual_names):
            shape = tuple(handle.get_slice(name).get_shape())
            if shape != expected[name]:
                raise ValueError(
                    f"Shape mismatch for {name}: checkpoint {shape}, "
                    f"model {expected[name]}"
                )
    return len(expected), missing, unexpected


def load_native_weights(
    model: Zonos2Model,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    expected = checkpoint_layout(model)
    expected_names = set(expected)
    sonic_modules = _sonic_expert_modules(model)
    packed_expert_names = {
        f"{prefix}.{suffix}": (prefix, suffix)
        for prefix in sonic_modules
        for suffix in ("w13", "w2")
    }
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        source_names = set(handle.keys())
        missing = expected_names - source_names
        unexpected = source_names - expected_names
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint layout mismatch. Missing={sorted(missing)[:10]}, "
                f"unexpected={sorted(unexpected)[:10]}"
            )

        names = sorted(expected_names)
        terminal_progress = (
            tqdm(
                total=len(names),
                desc="Loading ZONOS2 weights",
                unit="tensor",
                dynamic_ncols=True,
                leave=True,
            )
            if tqdm is not None
            else None
        )
        try:
            for index, name in enumerate(names, start=1):
                tensor = handle.get_tensor(name)
                packed = packed_expert_names.get(name)
                if packed is None:
                    _set_parameter(model, name, tensor, device, dtype)
                else:
                    prefix, projection = packed
                    module = sonic_modules[prefix]
                    for expert_index in range(module.num_experts):
                        expert = module.experts[expert_index]
                        source = tensor[expert_index].to(
                            device=device,
                            dtype=dtype,
                        )
                        if projection == "w13":
                            expert.weight = nn.Parameter(
                                source,
                                requires_grad=False,
                            )
                        else:
                            expert.bias = nn.Parameter(
                                source,
                                requires_grad=False,
                            )
                if terminal_progress is not None:
                    terminal_progress.update(1)
                if progress_callback is not None:
                    progress_callback(index, len(names))
                if terminal_progress is None and (
                    index == 1 or index % 32 == 0 or index == len(names)
                ):
                    logger.info(
                        "Loading ZONOS2 weights %d/%d",
                        index,
                        len(names),
                    )
        finally:
            if terminal_progress is not None:
                terminal_progress.close()

    model.eval()
    model.materialize_runtime_buffers(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta = [
        name
        for name, value in model.state_dict().items()
        if value.device.type == "meta"
    ]
    meta.extend(
        name
        for name, value in model.named_buffers()
        if value.device.type == "meta"
    )
    if meta:
        raise RuntimeError(f"ZONOS2 load left meta tensors: {meta[:10]}")
    logger.info("Loaded all %d ZONOS2 checkpoint tensors.", len(expected_names))


def load_quantized_weights(
    model: Zonos2Model,
    checkpoint_path: Path,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    expected = {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
    }
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        actual_names = set(names)
        expected_names = set(expected)
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            raise RuntimeError(
                "Quantized ZONOS2 checkpoint does not match the native "
                f"architecture. Missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        for name in names:
            shape = tuple(handle.get_slice(name).get_shape())
            if shape != expected[name]:
                raise ValueError(
                    f"Shape mismatch for {name}: checkpoint {shape}, "
                    f"model {expected[name]}"
                )
        terminal_progress = (
            tqdm(
                total=len(names),
                desc="Loading quantized ZONOS2 weights",
                unit="tensor",
                dynamic_ncols=True,
                leave=True,
            )
            if tqdm is not None
            else None
        )
        try:
            for index, name in enumerate(names, start=1):
                _set_checkpoint_value(
                    model,
                    name,
                    handle.get_tensor(name),
                    device,
                )
                if terminal_progress is not None:
                    terminal_progress.update(1)
                if progress_callback is not None:
                    progress_callback(index, len(names))
                if terminal_progress is None and (
                    index == 1 or index % 64 == 0 or index == len(names)
                ):
                    logger.info(
                        "Loading quantized ZONOS2 weights %d/%d",
                        index,
                        len(names),
                    )

        finally:
            if terminal_progress is not None:
                terminal_progress.close()

    model.eval()
    model.materialize_runtime_buffers(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    meta = [
        name
        for name, value in model.state_dict().items()
        if value.device.type == "meta"
    ]
    meta.extend(
        name
        for name, value in model.named_buffers()
        if value.device.type == "meta"
    )
    if meta:
        raise RuntimeError(
            f"Quantized ZONOS2 load left meta tensors: {meta[:10]}"
        )
    logger.info("Loaded all %d quantized ZONOS2 checkpoint tensors.", len(names))
    return len(names)


def text_to_byte_ids(text: str) -> list[int]:
    return [
        BOS_ID,
        *(byte + LEGACY_SYMBOL_VOCAB_SIZE for byte in text.encode("utf-8")),
        EOS_ID,
    ]


def _conditioning_base(config: Zonos2Config) -> int:
    background = 2 if config.speaker_background_token_enabled else 0
    accurate = 1 if config.accurate_mode_token_enabled and background else 0
    return (
        config.text_vocab
        - config.speaking_rate_num_buckets
        - sum(config.quality_bucket_counts)
        - background
        - accurate
    )


def speaking_rate_token(config: Zonos2Config, bucket: int) -> int:
    if bucket < 0 or bucket >= config.speaking_rate_num_buckets:
        raise ValueError(
            f"speaking_rate_bucket must be -1 or 0.."
            f"{config.speaking_rate_num_buckets - 1}"
        )
    return _conditioning_base(config) + bucket


def quality_token(
    config: Zonos2Config,
    feature_index: int,
    bucket: int,
) -> int:
    count = config.quality_bucket_counts[feature_index]
    if bucket < 0 or bucket >= count:
        raise ValueError(
            f"Quality bucket for {config.quality_features[feature_index]} "
            f"must be 0..{count - 1}"
        )
    return (
        _conditioning_base(config)
        + config.speaking_rate_num_buckets
        + sum(config.quality_bucket_counts[:feature_index])
        + bucket
    )


def speaker_background_token(config: Zonos2Config, clean: bool) -> int:
    return (
        _conditioning_base(config)
        + config.speaking_rate_num_buckets
        + sum(config.quality_bucket_counts)
        + (0 if clean else 1)
    )


def accurate_mode_token(config: Zonos2Config) -> int:
    return (
        _conditioning_base(config)
        + config.speaking_rate_num_buckets
        + sum(config.quality_bucket_counts)
        + 2
    )


def shear(codes: torch.Tensor, pad_id: int) -> torch.Tensor:
    frames, codebooks = codes.shape
    padded = codes.new_full((codebooks - 1 + frames, codebooks), pad_id)
    padded[codebooks - 1 :] = codes
    row_idx = (
        codebooks
        - 1
        + torch.arange(frames, device=codes.device).unsqueeze(1)
        - torch.arange(codebooks, device=codes.device)
    )
    return padded.gather(0, row_idx)


def shear_up(codes: torch.Tensor, pad_id: int) -> torch.Tensor:
    frames, codebooks = codes.shape[-2:]
    output = codes.new_full(codes.shape, pad_id)
    for index in range(codebooks):
        if frames > index:
            output[..., : frames - index, index] = codes[..., index:, index]
    return output


def build_prompt(
    config: Zonos2Config,
    text: str,
    speaking_rate_bucket: int = -1,
    quality_buckets: list[int | None] | tuple[int | None, ...] | None = None,
    speaker_embedding: torch.Tensor | None = None,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
) -> tuple[torch.Tensor, int | None]:
    rows: list[list[int]] = []
    audio_padding = [config.audio_pad_id] * config.n_codebooks

    speaker_position = None
    if speaker_embedding is not None:
        speaker_position = 0
        rows.append(audio_padding + [config.text_vocab])
        if config.speaker_background_token_enabled:
            rows.append(
                audio_padding
                + [speaker_background_token(config, clean_speaker_background)]
            )
        if config.accurate_mode_token_enabled and accurate_mode:
            rows.append(audio_padding + [accurate_mode_token(config)])

    if speaking_rate_bucket >= 0:
        rows.append(
            audio_padding + [speaking_rate_token(config, speaking_rate_bucket)]
        )

    if quality_buckets is not None:
        if len(quality_buckets) != len(config.quality_features):
            raise ValueError(
                f"Expected {len(config.quality_features)} quality buckets, "
                f"got {len(quality_buckets)}."
            )
        for feature_index, bucket in enumerate(quality_buckets):
            if bucket is None or int(bucket) < 0:
                continue
            rows.append(
                audio_padding
                + [
                    quality_token(
                        config,
                        feature_index,
                        int(bucket),
                    )
                ]
            )

    rows.extend(audio_padding + [token] for token in text_to_byte_ids(text))
    silence = torch.tensor(SILENCE_TOKENS_0_2S, dtype=torch.int32)
    silence = shear(silence[:, : config.n_codebooks], config.audio_pad_id)
    silence_text = torch.full(
        (silence.shape[0], 1),
        config.text_vocab,
        dtype=torch.int32,
    )
    prompt = torch.cat(
        [
            torch.tensor(rows, dtype=torch.int32),
            torch.cat([silence, silence_text], dim=1),
        ],
        dim=0,
    )
    return prompt.unsqueeze(0), speaker_position


@dataclass
class SamplingOptions:
    max_new_tokens: int = 1024
    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int = 0


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated: list[torch.Tensor],
    options: SamplingOptions,
) -> torch.Tensor:
    if (
        options.repetition_window <= 0
        or options.repetition_penalty <= 1.0
        or not generated
    ):
        return logits
    result = logits.clone()
    codebook_count = (
        result.shape[1]
        if options.repetition_codebooks < 0
        else min(result.shape[1], options.repetition_codebooks)
    )
    recent = torch.stack(generated[-options.repetition_window :]).to(result.device)
    for codebook in range(codebook_count):
        token_ids = recent[:, codebook].long().unique()
        valid = token_ids[
            (token_ids >= 0) & (token_ids < result.shape[-1])
        ]
        if valid.numel() == 0:
            continue
        values = result[0, codebook, valid]
        adjusted = torch.where(
            values > 0,
            values / options.repetition_penalty,
            values * options.repetition_penalty,
        )
        result[0, codebook, valid] = adjusted
    return result


def sample_codes(
    logits: torch.Tensor,
    generated: list[torch.Tensor],
    options: SamplingOptions,
    generator: torch.Generator | None,
) -> torch.Tensor:
    logits = _apply_repetition_penalty(logits.float(), generated, options)
    if options.temperature <= 1e-5:
        return logits.argmax(dim=-1)[0].to(torch.long)

    logits = logits / max(options.temperature, 1e-8)
    vocab = logits.shape[-1]
    if 0 < options.top_k < vocab:
        threshold = torch.topk(logits, options.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probabilities = torch.softmax(logits, dim=-1)
    if 0.0 < options.top_p < 1.0:
        sorted_probs, sorted_indices = probabilities.sort(
            dim=-1,
            descending=True,
        )
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative - sorted_probs > options.top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1,
            sorted_indices,
            sorted_probs,
        )

    if options.min_p > 0:
        maximum = probabilities.max(dim=-1, keepdim=True).values
        probabilities = probabilities.masked_fill(
            probabilities < maximum * options.min_p,
            0.0,
        )

    sums = probabilities.sum(dim=-1, keepdim=True)
    invalid = sums <= 0
    probabilities = probabilities / sums.clamp_min(1e-8)
    if invalid.any():
        greedy = logits.argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(probabilities).scatter(-1, greedy, 1.0)
        probabilities = torch.where(invalid, fallback, probabilities)

    rows = probabilities[0]
    return torch.multinomial(
        rows,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)


@torch.inference_mode()
def generate_audio_codes(
    model: Zonos2Model,
    prompt: torch.Tensor,
    attention_backend: str,
    options: SamplingOptions,
    speaker_embedding: torch.Tensor | None = None,
    speaker_position: int | None = None,
    speaker_emotion_delta: torch.Tensor | None = None,
    emotion_cfg_scale: float = 1.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, int | None]:
    device = torch.device(
        getattr(model, "device", next(model.parameters()).device)
    )
    dtype = model.runtime_dtype or next(model.parameters()).dtype
    prompt = prompt.to(device=device)
    use_emotion_cfg = (
        speaker_emotion_delta is not None
        and abs(float(emotion_cfg_scale) - 1.0) > 1e-6
    )
    batch_size = 2 if use_emotion_cfg else prompt.shape[0]
    if use_emotion_cfg:
        prompt = prompt.repeat(2, 1, 1)
        if speaker_embedding is None:
            raise ValueError("ZONOS2 emotion guidance requires a speaker embedding.")
        if speaker_embedding.shape[0] == 1:
            speaker_embedding = speaker_embedding.repeat(2, 1)
        delta = speaker_emotion_delta.reshape(1, -1)
        speaker_emotion_delta = torch.cat([delta, torch.zeros_like(delta)], dim=0)
    total_length = prompt.shape[1] + int(options.max_new_tokens)
    if total_length > model.config.max_seqlen:
        raise ValueError(
            f"Prompt ({prompt.shape[1]}) + max_new_tokens "
            f"({options.max_new_tokens}) exceeds ZONOS2 max sequence length "
            f"{model.config.max_seqlen}."
        )
    caches = model.create_kv_cache(
        batch_size=batch_size,
        max_length=total_length,
        device=device,
        dtype=dtype,
    )
    logits = model(
        prompt,
        caches,
        attention_backend,
        speaker_embedding=speaker_embedding,
        speaker_position=speaker_position,
        speaker_emotion_delta=speaker_emotion_delta,
    )

    def apply_emotion_guidance(value: torch.Tensor) -> torch.Tensor:
        if not use_emotion_cfg:
            return value
        conditional = value[0:1]
        unconditional = value[1:2]
        return unconditional + float(emotion_cfg_scale) * (
            conditional - unconditional
        )

    logits = apply_emotion_guidance(logits)
    generated: list[torch.Tensor] = []
    eos_frame: int | None = None
    eos_countdown = 0
    generator = None
    if options.seed > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(options.seed))

    terminal_progress = (
        tqdm(
            total=int(options.max_new_tokens),
            desc="ZONOS2 audio tokens",
            unit="tok",
            dynamic_ncols=True,
            leave=True,
        )
        if tqdm is not None
        else None
    )
    try:
        for step in range(int(options.max_new_tokens)):
            codes = sample_codes(logits, generated, options, generator)
            generated.append(codes.detach().to("cpu", torch.long))
            if terminal_progress is not None:
                terminal_progress.update(1)
            if progress_callback is not None:
                progress_callback(step + 1, int(options.max_new_tokens))
            if terminal_progress is None and (
                step == 0 or (step + 1) % 64 == 0
            ):
                logger.info(
                    "ZONOS2 audio tokens %d/%d",
                    step + 1,
                    options.max_new_tokens,
                )

            if eos_frame is None:
                eos_codebooks = torch.nonzero(
                    codes == model.config.eoa_id,
                    as_tuple=False,
                ).flatten()
                if eos_codebooks.numel() > 0:
                    eos_frame = max(
                        0,
                        step - int(eos_codebooks.max().item()),
                    )
                    eos_countdown = model.config.n_codebooks + 1
            if eos_frame is not None:
                eos_countdown -= 1
                if eos_countdown <= 0:
                    break

            next_row = torch.cat(
                [
                    codes.to(device=device, dtype=torch.int32),
                    torch.tensor(
                        [model.config.text_vocab],
                        device=device,
                        dtype=torch.int32,
                    ),
                ]
            ).view(1, 1, -1)
            if use_emotion_cfg:
                next_row = next_row.repeat(2, 1, 1)
            logits = model(
                next_row,
                caches,
                attention_backend,
            )
            logits = apply_emotion_guidance(logits)
    finally:
        if terminal_progress is not None:
            terminal_progress.close()

    if not generated:
        raise RuntimeError("ZONOS2 generated no audio token frames.")
    return torch.stack(generated), eos_frame
