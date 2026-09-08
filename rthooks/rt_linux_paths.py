"""Expose bundled Linux media tools to subprocess-based audio decoders."""
import os
import sys

if sys.platform.startswith("linux") and getattr(sys, "frozen", False):
    bundled_bin = os.path.join(sys._MEIPASS, "bin")
    os.environ["PATH"] = bundled_bin + os.pathsep + os.environ.get("PATH", "")
    # sounddevice calls find_library before dlopen. Linux's implementation
    # normally relies on the system ldconfig cache (or a linker), neither of
    # which knows about this portable application's private PortAudio build.
    import ctypes.util

    original_find_library = ctypes.util.find_library

    def find_bundled_library(name):
        if name == "portaudio":
            bundled_portaudio = os.path.join(sys._MEIPASS, "libportaudio.so.2")
            if os.path.isfile(bundled_portaudio):
                return bundled_portaudio
        return original_find_library(name)

    ctypes.util.find_library = find_bundled_library
