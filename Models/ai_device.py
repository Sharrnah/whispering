"""Resolve profile AI device selections to one concrete accelerator.

The profile stores the device family (for example ``cuda``) separately from
the CUDA adapter index.  PyTorch consumers receive ``cuda:N`` while
CTranslate2 consumers receive the family and index as separate arguments.
"""

from __future__ import annotations

from contextlib import contextmanager

import settings


DEVICE_INDEX_SETTINGS = {
    "ai_device": "ai_device_index",
    "txt_translator_device": "txt_translator_device_index",
    "tts_ai_device": "tts_ai_device_index",
    "ocr_ai_device": "ocr_ai_device_index",
}


def normalize_device_index(value) -> int:
    """Return a non-negative CUDA adapter index."""
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = 0
    if index < 0:
        raise ValueError(f"CUDA GPU index must be non-negative, got {index}.")
    return index


def _cuda_capabilities(cuda_available=None, cuda_device_count=None):
    if cuda_available is not None and cuda_device_count is not None:
        return bool(cuda_available), int(cuda_device_count)

    try:
        import torch

        available = bool(torch.cuda.is_available()) if cuda_available is None else bool(cuda_available)
        count = int(torch.cuda.device_count()) if cuda_device_count is None and available else int(cuda_device_count or 0)
        return available, count
    except (ImportError, RuntimeError):
        return False, 0


def resolve_device(device, device_index=0, *, cuda_available=None, cuda_device_count=None) -> str:
    """Resolve a configured device to ``cpu``, ``cuda:N``, or another explicit backend."""
    device_name = str(device or "").strip().lower()
    selected_index = normalize_device_index(device_index)
    available, count = _cuda_capabilities(cuda_available, cuda_device_count)

    if device_name in {"", "none", "auto", "cuda"}:
        if not available:
            return "cpu"
        cuda_index = selected_index
    elif device_name.startswith("cuda:"):
        try:
            cuda_index = normalize_device_index(device_name.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"Invalid CUDA device {device!r}.") from error
        if not available:
            return "cpu"
    else:
        return device_name

    if count > 0 and cuda_index >= count:
        raise ValueError(
            f"CUDA GPU index {cuda_index} is unavailable; detected {count} CUDA device(s)."
        )
    return f"cuda:{cuda_index}"


def get_device(device_setting: str, index_setting: str | None = None, settings_source=None) -> str:
    """Resolve a device and its companion index from a settings provider."""
    if index_setting is None:
        index_setting = DEVICE_INDEX_SETTINGS[device_setting]
    get_option = settings.GetOption
    if settings_source is not None:
        get_option = getattr(settings_source, "GetOption", None)
        if get_option is None:
            get_option = settings_source.get_option
    return resolve_device(
        get_option(device_setting),
        get_option(index_setting),
    )


def get_ctranslate2_device(
    device_setting: str, index_setting: str | None = None, settings_source=None
) -> tuple[str, int]:
    """Return CTranslate2's device family and separate adapter index."""
    if index_setting is None:
        index_setting = DEVICE_INDEX_SETTINGS[device_setting]
    resolved = get_device(device_setting, index_setting, settings_source)
    return split_ctranslate2_device(resolved)


def split_ctranslate2_device(device) -> tuple[str, int]:
    """Split an already resolved device for a CTranslate2 constructor."""
    resolved = str(device or "cpu").strip().lower()
    if resolved.startswith("cuda:"):
        return "cuda", normalize_device_index(resolved.split(":", 1)[1])
    return resolved, 0


@contextmanager
def cuda_device_context(device):
    """Make unqualified PyTorch ``cuda`` allocations use the selected adapter."""
    device_name = str(device or "").strip().lower()
    if not device_name.startswith("cuda:"):
        yield
        return

    import torch

    if not torch.cuda.is_available():
        yield
        return
    with torch.cuda.device(device_name):
        yield
