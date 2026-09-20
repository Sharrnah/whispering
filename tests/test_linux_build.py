import importlib.util
import os
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]


def load_builder(name):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "builder" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


staging = load_builder("linux-stage")
startup = load_builder("linux-startup-check")


class LinuxBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="linux build ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_snapshot_includes_untracked_streaming_dependencies(self):
        source = self.root / "source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        (source / "audioprocessor.py").write_text("from streaming_text import rolling_text\nimport streaming_display\n")
        (source / "deleted.py").touch()
        subprocess.run(["git", "-C", str(source), "add", "audioprocessor.py", "deleted.py"], check=True)
        (source / "deleted.py").unlink()
        for name in ("streaming_text.py", "streaming_display.py"):
            shutil.copy2(REPOSITORY / name, source / name)
        (source / "private-experiment.py").write_text("raise RuntimeError('Do not ship')\n")
        destination = self.root / "snapshot"
        staging.snapshot_backend(source, destination)
        self.assertFalse((destination / "private-experiment.py").exists())
        self.assertFalse((destination / "deleted.py").exists())
        for name in ("streaming_text.py", "streaming_display.py"):
            self.assertEqual((destination / name).read_bytes(), (source / name).read_bytes())
        result = subprocess.run([sys.executable, "-E", "-c", "import audioprocessor"],
                                cwd=destination, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_linux_uses_synchronous_inductor_without_changing_other_platforms(self):
        import ctypes.util
        hook = REPOSITORY / "rthooks/rt_linux_paths.py"
        for platform, frozen, expected in (("linux", True, "1"), ("linux", False, None), ("win32", True, None)):
            with self.subTest(platform=platform, frozen=frozen), \
                    mock.patch.object(sys, "platform", platform), \
                    mock.patch.object(sys, "frozen", frozen, create=True), \
                    mock.patch.object(sys, "_MEIPASS", str(self.root), create=True), \
                    mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(ctypes.util, "find_library"):
                runpy.run_path(str(hook))
                self.assertEqual(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"), expected)

    @unittest.skipUnless(sys.platform == "linux", "Linux executable startup gate")
    def test_startup_rejects_import_failure_even_if_help_was_printed(self):
        executable = self.root / "backend"
        executable.write_text("#!/bin/sh\necho 'Usage: audioWhisper'\necho \"ModuleNotFoundError: streaming_text\" >&2\nexit 1\n")
        executable.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "streaming_text"):
            startup.check_startup(executable, self.root / "failed.log")

    @unittest.skipUnless(sys.platform == "linux", "Linux executable startup gate")
    def test_startup_runs_help_outside_source_directory(self):
        executable = self.root / "backend"
        executable.write_text("#!/bin/sh\n[ \"$1\" = --help ] || exit 2\n[ ! -e source-only.txt ] || exit 3\necho 'Usage: audioWhisper'\n")
        executable.chmod(0o755)
        (self.root / "source-only.txt").touch()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            startup.check_startup(executable, self.root / "passed.log")
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
