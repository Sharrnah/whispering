import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("windows_package", REPOSITORY / "builder/windows-package.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class WindowsPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="windows package ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        for name in (
            "LICENSE", "ignorelist.txt", "dist_files/help.bat", "dist_files/get-device-list.bat",
            "markers/OKW-MRK.wav", "websocket_clients/index.html",
            "dist_files/Plugins/place_plugins_here.txt", "dist_files/Plugins/__pycache__/old.pyc",
            "toolchain/ffmpeg/bin/ffmpeg.exe", "toolchain/ffmpeg/bin/ffprobe.exe",
            "toolchain/ffmpeg/LICENSE", "toolchain/tcc/tcc.exe", "toolchain/tcc/lib/libtcc1.a",
            "Plugins/private_plugin.py", "Profiles/private.yaml", ".cache/private.bin",
            "toolchain/experiments/private.bin",
        ):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        self.backend = self.root / "frozen/audioWhisper"
        (self.backend / "_internal").mkdir(parents=True)
        (self.backend / "audioWhisper.exe").write_bytes(b"MZ backend")
        (self.backend / "_internal/python311.dll").write_bytes(b"MZ Python runtime")
        self.ui = self.root / "Whispering Tiger.exe"
        self.ui.write_bytes(b"MZ UI")
        self.output = self.root / "release/windows"

    def package(self, version="1.2.3"):
        return packager.package(self.backend, self.ui, self.output, self.source, version)

    def test_complete_archive_excludes_personal_data_and_has_matching_hashes(self):
        archive = self.package()
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            self.assertTrue({
                "audioWhisper/audioWhisper.exe", "audioWhisper/_internal/python311.dll",
                "LICENSE", "ignorelist.txt", "help.bat", "get-device-list.bat",
                "markers/OKW-MRK.wav", "websocket_clients/index.html",
                "Plugins/place_plugins_here.txt", "toolchain/ffmpeg/bin/ffmpeg.exe",
                "toolchain/ffmpeg/bin/ffprobe.exe", "toolchain/ffmpeg/LICENSE",
                "toolchain/tcc/tcc.exe", "toolchain/tcc/lib/libtcc1.a",
            } <= names)
            self.assertEqual(bundle.read(".current_platform.yaml"), b'version: "1.2.3"\n')
            self.assertFalse(any("private" in name or "__pycache__" in name for name in names))
            self.assertNotIn("Whispering Tiger.exe", names)
            self.assertIsNone(bundle.testzip())
        self.assertEqual((self.output / self.ui.name).read_bytes(), self.ui.read_bytes())
        for line in (self.output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(digest, hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        self.assertFalse(archive.with_suffix(".zip.partial").exists())

    def test_missing_media_tool_fails_before_creating_archive(self):
        (self.source / "toolchain/ffmpeg/bin/ffprobe.exe").unlink()
        with self.assertRaisesRegex(ValueError, "Missing release file"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_existing_release_cannot_be_overwritten(self):
        archive = self.package()
        original = archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.package()
        self.assertEqual(archive.read_bytes(), original)

    def test_invalid_version_or_non_windows_binary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid backend version"):
            self.package("../other")
        self.ui.write_bytes(b"\x7fELF")
        with self.assertRaisesRegex(ValueError, "Windows executable"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_interrupted_archive_is_never_given_the_finished_name(self):
        with mock.patch.object(zipfile.ZipFile, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.package()
        self.assertFalse((self.output / "whispering-tiger1.2.3_win.zip").exists())
        self.assertFalse((self.output / "SHA256SUMS").exists())


if __name__ == "__main__":
    unittest.main()
