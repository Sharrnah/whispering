"""Standalone ZONOS2 loading, voice cloning, and audio decoding.

The native model implementation is adapted from Saganaki22/Zonos2_TTS-ComfyUI
(MIT). This module intentionally contains no ComfyUI integration and never
downloads model files: Whispering Tiger uses the user's local cache only.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .native import (
    SamplingOptions,
    Zonos2Config,
    Zonos2Model,
    build_native_model,
    build_prompt,
    generate_audio_codes,
    load_native_weights,
    load_quantized_weights,
    read_config,
    set_runtime_dtype,
    shear_up,
    validate_checkpoint_layout,
    validate_quantized_runtime_model,
)
from .emotion import EmotionDirections

logger = logging.getLogger(__name__)

DAC_SAMPLE_RATE = 44_100
SPEAKER_SAMPLE_RATE = 24_000
MAX_REFERENCE_SECONDS = 60.0


@dataclass
class Zonos2Bundle:
    model: Zonos2Model
    config: Zonos2Config
    device: torch.device
    codec: "Zonos2DAC"
    model_name: str
    attention_preference: str
    attention_backend: str
    speaker_encoder: "Zonos2SpeakerEncoder | None" = None
    emotion_directions: "EmotionDirections | None" = None


class Zonos2DAC(nn.Module):
    sample_rate = DAC_SAMPLE_RATE

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model.eval().to(device)
        self.requires_grad_(False)

    @torch.inference_mode()
    def decode(
        self,
        delayed_codes: torch.Tensor,
        pad_id: int,
        eos_frame: int | None,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        codes = shear_up(delayed_codes.to(device=device, dtype=torch.long), pad_id)
        if eos_frame is not None:
            codes = codes[:max(0, int(eos_frame))]
        else:
            complete = max(0, codes.shape[0] - (codes.shape[1] - 1))
            codes = codes[:complete]
        if codes.numel() == 0:
            raise RuntimeError("ZONOS2 ended before producing decodable audio.")
        codes = codes.clamp_(0, 1023).transpose(0, 1).unsqueeze(0)
        audio = self.model.decode(audio_codes=codes).audio_values
        if audio.ndim == 3:
            audio = audio.mean(dim=1)
        return audio.float().cpu()


class Zonos2SpeakerEncoder(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model.eval().to(device)
        self.device = device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SPEAKER_SAMPLE_RATE,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=12_000.0,
            n_mels=128,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        self._resamplers = nn.ModuleDict()
        self.requires_grad_(False)

    def _prepare_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio = waveform
        if audio.ndim == 3:
            audio = audio[0]
        if audio.ndim == 2:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            raise ValueError(f"Reference audio must be 1D, 2D, or 3D, got {audio.ndim}D.")

        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Reference audio sample rate must be positive.")
        max_samples = round(MAX_REFERENCE_SECONDS * sample_rate)
        if audio.shape[-1] > max_samples:
            logger.warning("Clipping ZONOS2 reference audio to %.0f seconds.", MAX_REFERENCE_SECONDS)
            audio = audio[..., :max_samples]

        audio = audio.to(device=self.device, dtype=torch.float32)
        if sample_rate != SPEAKER_SAMPLE_RATE:
            key = str(sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    sample_rate, SPEAKER_SAMPLE_RATE
                ).to(self.device)
            audio = self._resamplers[key](audio)
        if audio.shape[-1] < 1024:
            audio = F.pad(audio, (0, 1024 - audio.shape[-1]))
        return audio

    @torch.inference_mode()
    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio = self._prepare_audio(waveform, sample_rate)
        padding = (1024 - 256) // 2
        audio = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)
        return self.model(input_values=mel).last_hidden_state.float()


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ZONOS2 {description}: {path}")
    return path


def flash_attention_available(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return False
    if torch.cuda.get_device_capability(device) < (8, 0):
        return False
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError):
        return False
    return callable(flash_attn_func)


def resolve_attention_backend(preference: str, device: torch.device) -> str:
    preference = str(preference or "auto").strip().lower()
    if preference not in {"auto", "flash_attention", "sdpa"}:
        raise ValueError(
            "ZONOS2 attention must be auto, flash_attention, or sdpa; "
            f"got {preference!r}."
        )
    available = flash_attention_available(device)
    if preference == "auto":
        return "flash_attention" if available else "sdpa"
    if preference == "flash_attention" and not available:
        raise RuntimeError(
            "ZONOS2 FlashAttention was selected, but flash-attn is not "
            "installed or cannot run on the selected device. Choose auto or sdpa."
        )
    return preference


def load_bundle(
    cache_path: Path,
    device: torch.device,
    model_name: str = "zonos2-bf16",
    attention: str = "auto",
) -> Zonos2Bundle:
    """Load a local drbaph ZONOS2 checkpoint without downloading anything."""
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("ZONOS2 BF16 requires a CUDA GPU with BF16 support.")

    package_dir = Path(__file__).resolve().parent
    if model_name == "zonos2-bf16":
        checkpoint = _require_file(
            cache_path / "bf16" / "zonos2-bf16.safetensors",
            "BF16 checkpoint",
        )
        quantization = None
        checkpoint_label = "BF16"
    elif model_name == "zonos2-fp8-mixed":
        checkpoint = _require_file(
            cache_path / "fp8" / "zonos2-fp8-mixed.safetensors",
            "mixed FP8 checkpoint",
        )
        quantization = "fp8_e4m3"
        checkpoint_label = "mixed FP8"
    else:
        raise ValueError(f"Unsupported ZONOS2 model: {model_name!r}")

    params = _require_file(package_dir / "params.json", "architecture configuration")
    dac_dir = cache_path / "dac_44khz"
    _require_file(dac_dir / "config.json", "DAC configuration")
    _require_file(dac_dir / "model.safetensors", "DAC weights")

    config = read_config(params)
    model = build_native_model(
        config,
        quantization=quantization,
        compute_dtype=torch.bfloat16,
        load_device=device,
    )
    logger.info("Loading ZONOS2 %s from %s onto %s", checkpoint_label, checkpoint, device)
    if quantization is None:
        _, missing, unexpected = validate_checkpoint_layout(model, checkpoint)
        if missing or unexpected:
            raise RuntimeError(
                "ZONOS2 checkpoint does not match the supported BF16 architecture. "
                f"Missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}"
            )
        load_native_weights(model, checkpoint, device, torch.bfloat16)
    else:
        validate_quantized_runtime_model(model)
        load_quantized_weights(model, checkpoint, device)
    set_runtime_dtype(model, torch.bfloat16)
    attention_backend = resolve_attention_backend(attention, device)
    attention_preference = str(attention or "auto").strip().lower()
    logger.info(
        "ZONOS2 attention backend: %s (requested %s)",
        attention_backend,
        attention_preference,
    )

    from transformers import DacModel

    codec_model = DacModel.from_pretrained(str(dac_dir), local_files_only=True)
    codec = Zonos2DAC(codec_model, device)
    emotion_directions = EmotionDirections.load(cache_path / "emotion_directions")
    if emotion_directions is not None and emotion_directions.dim != config.dim:
        raise ValueError(
            f"ZONOS2 emotion directions have dim {emotion_directions.dim}; expected {config.dim}."
        )
    return Zonos2Bundle(
        model=model,
        config=config,
        device=device,
        codec=codec,
        model_name=model_name,
        attention_preference=attention_preference,
        attention_backend=attention_backend,
        emotion_directions=emotion_directions,
    )


def ensure_speaker_encoder(bundle: Zonos2Bundle, cache_path: Path) -> Zonos2SpeakerEncoder:
    if bundle.speaker_encoder is not None:
        return bundle.speaker_encoder

    encoder_dir = cache_path / "speaker_encoder"
    _require_file(encoder_dir / "config.json", "speaker encoder configuration")
    _require_file(encoder_dir / "model.safetensors", "speaker encoder weights")
    _require_file(encoder_dir / "configuration_ecapa_tdnn.py", "speaker encoder configuration code")
    _require_file(encoder_dir / "modeling_ecapa_tdnn.py", "speaker encoder model code")

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(encoder_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    bundle.speaker_encoder = Zonos2SpeakerEncoder(model, bundle.device)
    return bundle.speaker_encoder


@torch.inference_mode()
def extract_speaker_embedding(
    bundle: Zonos2Bundle,
    cache_path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    embedding = ensure_speaker_encoder(bundle, cache_path)(waveform, sample_rate)
    expected = (1, bundle.config.speaker_embedding_dim)
    if tuple(embedding.shape) != expected:
        raise RuntimeError(
            f"ZONOS2 speaker encoder returned {tuple(embedding.shape)}, expected {expected}."
        )
    return embedding


@torch.inference_mode()
def generate_audio(
    bundle: Zonos2Bundle,
    text: str,
    options: SamplingOptions,
    speaking_rate_bucket: int = -1,
    quality_buckets: list[int | None] | None = None,
    speaker_embedding: torch.Tensor | None = None,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
    speaker_emotion_delta: torch.Tensor | None = None,
    emotion_cfg_scale: float = 1.0,
) -> torch.Tensor:
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    prompt, speaker_position = build_prompt(
        bundle.config,
        text=text,
        speaking_rate_bucket=speaking_rate_bucket,
        quality_buckets=quality_buckets,
        speaker_embedding=speaker_embedding,
        clean_speaker_background=clean_speaker_background,
        accurate_mode=accurate_mode,
    )
    delayed_codes, eos_frame = generate_audio_codes(
        bundle.model,
        prompt,
        attention_backend=bundle.attention_backend,
        options=options,
        speaker_embedding=speaker_embedding,
        speaker_position=speaker_position,
        speaker_emotion_delta=speaker_emotion_delta,
        emotion_cfg_scale=emotion_cfg_scale,
    )
    return bundle.codec.decode(
        delayed_codes,
        pad_id=bundle.config.audio_pad_id,
        eos_frame=eos_frame,
    )


def unload_bundle(bundle: Zonos2Bundle | None) -> None:
    if bundle is None:
        return
    bundle.speaker_encoder = None
    del bundle.codec
    del bundle.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
