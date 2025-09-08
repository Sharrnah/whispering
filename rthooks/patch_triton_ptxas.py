# rthooks/patch_triton_ptxas.py
import sys, os, subprocess, importlib, importlib.util, importlib.abc

def _patch(module):
    # Patch only if needed
    try:
        src = module.get_ptxas_version.__code__.co_names
    except Exception:
        src = ()
    # proceed; even if already patched this is idempotent
    def _patched_get_ptxas_version(arch):
        exe = module.get_ptxas(arch)[0]
        out = subprocess.run(
            [exe, "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=False,          # important on Windows frozen apps
            creationflags=0x08000000  # CREATE_NO_WINDOW (optional)
        )
        return out.stdout.decode("utf-8", "ignore")
    module.get_ptxas_version = _patched_get_ptxas_version

def _try_patch_now():
    try:
        mod = sys.modules.get("triton.backends.nvidia.compiler")
        if mod is None:
            return False
        _patch(mod)
        sys.stderr.write("[rthook] patched triton.backends.nvidia.compiler.get_ptxas_version (already imported)\n")
        return True
    except Exception as e:
        sys.stderr.write(f"[rthook] immediate patch failed: {e}\n")
        return False

class _PostImportHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self._orig_loader = None

    def find_spec(self, fullname, path, target=None):
        if fullname == "triton.backends.nvidia.compiler":
            spec = importlib.util.find_spec(fullname)
            if spec and spec.loader:
                self._orig_loader = spec.loader
                # Return a spec that reuses us as the loader so we can patch after exec
                return importlib.util.spec_from_loader(fullname, self, origin=spec.origin)
        return None

    def create_module(self, spec):
        # Defer to original loader
        return None

    def exec_module(self, module):
        # Load the real module first
        self._orig_loader.exec_module(module)
        _patch(module)
        sys.stderr.write("[rthook] patched triton.backends.nvidia.compiler.get_ptxas_version (post-import)\n")

# Install post-import hook, then try to patch immediately if already imported
sys.meta_path.insert(0, _PostImportHook())
_try_patch_now()
