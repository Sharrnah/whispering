# rthooks/rt_triton_env.py
import os, sys, importlib.util, warnings

# Put Triton cache next to the exe so it's writable.
base = os.path.dirname(getattr(sys, "_MEIPASS", os.getcwd()))
cache_dir = os.path.join(base, "triton_cache")
os.environ.setdefault("TRITON_CACHE_DIR", cache_dir)
os.environ.setdefault("TRITON_HOME", cache_dir)

# Ensure the cache directory exists, but do not force Triton to import here.
try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    # Non-fatal: if this fails, we'll try a user-writable fallback below
    pass

# If the chosen cache_dir is not writable, fall back to a user cache under LOCALAPPDATA
if not os.path.isdir(cache_dir) or not os.access(cache_dir, os.W_OK):
    fallback_root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    fallback_dir = os.path.join(fallback_root, "WhisperingVRC", "triton_cache")
    try:
        os.makedirs(fallback_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = fallback_dir
        os.environ["TRITON_HOME"] = fallback_dir
    except Exception:
        # Still non-fatal: Triton may attempt to create its own default cache
        pass

# Avoid importing Triton at runtime-hook load time — it can cascade into heavy imports
# (e.g., triton.runtime.autotuner) that may be unavailable in some frozen builds.
# If a strict verification is desired, gate it behind an env var and use find_spec only.
if os.environ.get("WT_STRICT_TRITON_CHECK") == "1":
    spec = importlib.util.find_spec("triton")
    if spec is None:
        warnings.warn(
            "Triton not found; GPU kernels requiring Triton will be unavailable.",
            RuntimeWarning,
        )
    else:
        # Probe for submodules without importing them
        if importlib.util.find_spec("triton.runtime.jit") is None:
            warnings.warn(
                "Triton 'runtime.jit' not found; the package may be incomplete in this build.",
                RuntimeWarning,
            )
