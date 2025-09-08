# rthooks/rt_triton_env.py
import os, sys

# Put Triton cache next to the exe so it's writable.
base = os.path.dirname(getattr(sys, "_MEIPASS", os.getcwd()))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(base, "triton_cache"))
os.environ.setdefault("TRITON_HOME", os.path.join(base, "triton_cache"))
