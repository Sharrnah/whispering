# rthooks/rt_inspect_fallback.py
import inspect
from types import ModuleType

_orig_getsourcelines = inspect.getsourcelines

def _fallback_getsourcelines(obj):
    # Try normal path first
    try:
        return _orig_getsourcelines(obj)
    except OSError:
        pass

    # Try to get source via the module loader (works in PyInstaller)
    mod = inspect.getmodule(obj)
    if not isinstance(mod, ModuleType):
        raise

    spec = getattr(mod, "__spec__", None)
    loader = getattr(spec, "loader", None) if spec else None

    if loader and hasattr(loader, "get_source"):
        src = loader.get_source(mod.__name__)
        if src:
            lines = src.splitlines(True)
            # Try to locate the object's definition; if not found, return whole module
            name = getattr(obj, "__name__", None)
            if name:
                import re
                pat = re.compile(rf"^\s*(def|class)\s+{re.escape(name)}\b", re.M)
                m = pat.search(src)
                if m:
                    # line numbers are 1-based; count newlines up to match start
                    start_line = src.count("\n", 0, m.start()) + 1
                    # Slice from the matched line; TorchScript only needs a consistent view
                    return lines[start_line-1:], start_line
            return lines, 1

    # No luck: re-raise original error for visibility
    raise OSError("could not get source code (PyInstaller fallback)")  # noqa: B904

# Apply the patch very early
inspect.getsourcelines = _fallback_getsourcelines
