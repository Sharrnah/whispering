"""Shared profile configuration for integrated TTS adapters."""

from __future__ import annotations

import settings
from Models.ai_device import get_torch_device


def get_tts_device() -> str:
    # audio.cpp consumes Vulkan/HIP/Metal directly and does not call this
    # helper.  Chatterbox and other PyTorch TTS consumers (including plugins)
    # need a Torch-compatible fallback when they share an audio.cpp profile.
    return get_torch_device("tts_ai_device", "tts_ai_device_index")


def get_tts_precision(default="auto") -> str:
    precision = settings.GetOption("tts_precision")
    if not isinstance(precision, str) or not precision.strip():
        precision = default
    precision = str(precision).strip().lower()
    legacy_precision = str(default or "").strip().lower()
    # ``auto`` is also the migration/default sentinel.  Preserve a concrete
    # precision supplied by an older per-adapter settings block; the UI keeps
    # that legacy field synchronized when the new selector is changed.
    if precision == "auto" and legacy_precision not in {"", "auto"}:
        return legacy_precision
    return precision
