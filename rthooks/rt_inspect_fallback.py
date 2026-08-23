# rthooks/rt_inspect_fallback.py
import inspect
from pathlib import Path
from types import ModuleType

_orig_getsourcelines = inspect.getsourcelines


def _get_module_source(mod):
    """Load source from a normal loader or a PyInstaller-collected .py file."""
    spec = getattr(mod, "__spec__", None)
    loader = getattr(spec, "loader", None) if spec else None

    if loader and hasattr(loader, "get_source"):
        try:
            source = loader.get_source(mod.__name__)
        except (ImportError, OSError):
            source = None
        if source:
            return source

    # PyInstaller's frozen-module loader does not expose source from the PYZ.
    # Selected packages can bundle their .py files as data at the same path
    # represented by module.__file__, which gives TorchScript source access.
    module_file = getattr(mod, "__file__", None)
    if module_file:
        source_path = Path(module_file).with_suffix(".py")
        try:
            return source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass

    return None


def _fallback_getsourcelines(obj):
    # Try normal path first
    try:
        return _orig_getsourcelines(obj)
    except OSError:
        pass

    # Try a source-aware loader, then a source file collected beside the module.
    mod = inspect.getmodule(obj)
    if not isinstance(mod, ModuleType):
        raise OSError("could not resolve the object's module (PyInstaller fallback)")

    src = _get_module_source(mod)
    if src:
        lines = src.splitlines(True)
        name = getattr(obj, "__name__", None)
        if name:
            import re

            pattern = re.compile(
                rf"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+{re.escape(name)}\b",
                re.M,
            )
            match = pattern.search(src)
            if match:
                # Return only this definition. TorchScript's parser rejects a
                # function followed by unrelated top-level definitions.
                start_line = src.count("\n", 0, match.start()) + 1
                return inspect.getblock(lines[start_line - 1:]), start_line
        return lines, 1

    # No luck: re-raise original error for visibility
    raise OSError("could not get source code (PyInstaller fallback)")  # noqa: B904

# Apply the patch very early
inspect.getsourcelines = _fallback_getsourcelines
