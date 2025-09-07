# rthooks/rt_fix_flash_attn_spec.py
import sys
import importlib

# If a broken stub is present (no __spec__), remove it BEFORE anything imports transformers.
m = sys.modules.get("flash_attn")
if m is not None and getattr(m, "__spec__", None) is None:
    del sys.modules["flash_attn"]

# Try to import the real bundled package so it has a proper spec.
# If it fails here, that's fine — find_spec() will still work; we just avoid the spec-less state.
try:
    import flash_attn  # noqa: F401
except Exception:
    pass
