import gc

import torch
from transformers.utils import is_flash_attn_2_available


FLASH_ATTENTION_2 = "flash_attention_2"
SDPA = "sdpa"
FLASH_ATTENTION_DTYPES = {torch.float16, torch.bfloat16}


def get_preferred_attention_implementation(device, dtype):
    """Choose FA2 only when its local package, device, and dtype are usable."""
    compute_device = torch.device(device)
    if compute_device.type != "cuda" or dtype not in FLASH_ATTENTION_DTYPES:
        return SDPA
    if not is_flash_attn_2_available():
        return SDPA

    try:
        major_version, _ = torch.cuda.get_device_capability(compute_device)
    except (AssertionError, RuntimeError, ValueError):
        return SDPA

    # The standard CUDA FA2 kernels require Ampere-class hardware or newer.
    if major_version < 8:
        return SDPA
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return SDPA
    return FLASH_ATTENTION_2


def _clear_accelerator_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_with_attention_fallback(loader, preferred_implementation, model_label):
    """Load with FA2 when selected, retrying once with SDPA on runtime failure."""
    try:
        return loader(preferred_implementation), preferred_implementation
    except Exception as error:
        if preferred_implementation != FLASH_ATTENTION_2:
            raise
        flash_error = str(error)

    print(
        f"{model_label} could not initialize FlashAttention 2 ({flash_error}); "
        "retrying with SDPA."
    )
    _clear_accelerator_cache()
    return loader(SDPA), SDPA
