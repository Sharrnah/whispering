import importlib.util
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("linux_package", REPOSITORY / "builder/linux-package.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


@unittest.skipUnless(sys.platform == "linux", "Linux distribution helpers")
class LinuxPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="linux package ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        for name in ("LICENSE", "ignorelist.txt", "markers/OKW-MRK.wav", "markers/WOK-MRK.wav",
                     "websocket_clients/simple/index.html", "websocket_clients/streaming-transcripts.js",
                     "dist_files/Plugins/place_plugins_here.txt", "Plugins/private_plugin.py", ".cache/private.bin"):
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("test data")
        shutil.copytree(REPOSITORY / "dist_files/linux", self.source / "dist_files/linux")
        self.backend = self.root / "dist/audioWhisper"
        (self.backend / "_internal/bin").mkdir(parents=True)
        for name in ("audioWhisper", "_internal/bin/ffmpeg", "_internal/bin/ffprobe"):
            binary = self.backend / name
            binary.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$*"\n')
            binary.chmod(0o755)
        self.archive = self.root / "backend.zip"

    def test_complete_distribution_and_portable_helpers(self):
        packager.write_archive(self.backend, self.archive, self.source, "1.2.3")
        target = self.root / "portable app"
        with zipfile.ZipFile(self.archive) as bundle:
            names = set(bundle.namelist())
            self.assertTrue({"LICENSE", "ignorelist.txt", "Plugins/place_plugins_here.txt", "markers/OKW-MRK.wav",
                             "websocket_clients/streaming-transcripts.js", ".current_platform.yaml"} <= names)
            self.assertNotIn("Plugins/private_plugin.py", names)
            self.assertFalse(any(".cache/" in name for name in names))
            for library in packager.WAYLAND_RUNTIME_LIBRARIES:
                self.assertIn("audioWhisper/_internal/" + library, names)
            self.assertEqual(yaml.safe_load(bundle.read(".current_platform.yaml")), {"version": "1.2.3"})
            self.assertFalse({"README-LINUX.txt", "PACKAGE-CONTENTS.txt", "toolchain/README-LINUX.txt",
                              "linux-runtime-info.json", "linux-python-packages.txt"} & names)
            bundle.extractall(target)
            for entry in bundle.infolist():
                (target / entry.filename).chmod(stat.S_IMODE(entry.external_attr >> 16))
        for helper, expected in (("help.sh", "--help"), ("get-device-list.sh", "--devices true"),
                                 ("toolchain/ffmpeg/ffmpeg", "--version"), ("toolchain/ffmpeg/ffprobe", "--version")):
            command = [str(target / helper)]
            if helper.startswith("toolchain"):
                command.append("--version")
            output = subprocess.check_output(command, cwd=self.root, text=True)
            self.assertIn(expected, output)
            if helper.endswith(".sh"):
                self.assertEqual(output.splitlines()[0], str(target))

    def test_missing_release_file_fails(self):
        (self.source / "dist_files/Plugins/place_plugins_here.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing release files"):
            packager.write_archive(self.backend, self.archive, self.source, "1.2.3")

    def test_missing_bundled_ffmpeg_fails(self):
        (self.backend / "_internal/bin/ffmpeg").unlink()
        with self.assertRaisesRegex(ValueError, "Missing bundled media tool"):
            packager.write_archive(self.backend, self.archive, self.source, "1.2.3")

    def test_audio_cpp_bundle_and_version_mismatch(self):
        (self.source / "Models").mkdir()
        version_source = self.source / "Models/audio_cpp_runtime.py"
        version_source.write_text('AUDIO_CPP_VERSION = "0.7.1"\nLINUX_BUNDLE_REVISION = 2\n')
        native_root = self.root / "native"
        folder = native_root / "v0.7.1-r2-linux-x86_64"
        folder.mkdir(parents=True)
        for name in ("audiocpp_server", "libggml.so.0", "libggml-base.so.0", "libggml-vulkan.so",
                     "libggml-cpu-x64.so", "libvulkan.so.1", "libgomp.so.1", "LICENSE"):
            target = folder / name
            target.write_bytes(b"\x7fELF\x02" + b"\0" * 13 + (62).to_bytes(2, "little"))
            target.chmod(0o755)
        packager.write_archive(self.backend, self.archive, self.source, "1.2.3", native_root)
        with zipfile.ZipFile(self.archive) as bundle:
            server = bundle.getinfo("toolchain/audio.cpp/v0.7.1-r2-linux-x86_64/audiocpp_server")
            self.assertTrue(stat.S_IMODE(server.external_attr >> 16) & stat.S_IXUSR)
            self.assertIn("toolchain/audio.cpp/v0.7.1-r2-linux-x86_64/libvulkan.so.1", bundle.namelist())
        for library in ("libstdc++.so.6", "libgcc_s.so.1"):
            (folder / library).touch()
            with self.assertRaisesRegex(ValueError, r"system C\+\+/GCC"):
                packager.write_archive(self.backend, self.archive, self.source, "1.2.3", native_root)
            (folder / library).unlink()
        version_source.write_text('AUDIO_CPP_VERSION = "0.7.2"\nLINUX_BUNDLE_REVISION = 2\n')
        with self.assertRaisesRegex(ValueError, "Missing bundled audio.cpp 0.7.2"):
            packager.write_archive(self.backend, self.archive, self.source, "1.2.3", native_root)

    def test_versioned_download_fragment_and_url_override(self):
        filename = packager.archive_filename("1.2.3", "cu128")
        self.assertEqual(filename, "whispering-tiger1.2.3_linux_amd64_cu128.zip")
        url = packager.backend_download_url(packager.DEFAULT_DOWNLOAD_BASE_URL, filename)
        self.assertEqual(url, "https://s3.libs.space:9000/projects/whispering/" + filename)
        self.assertEqual(packager.backend_download_url("https://mirror.example/releases", filename),
                         "https://mirror.example/releases/" + filename)
        fragment = self.root / "fragment.yaml"
        packager.write_update_fragment(fragment, "cu128", "1.2.3", "a" * 64, url)
        entry = yaml.safe_load(fragment.read_text())["packages"]["ai_platform_linux_amd64_cu128"]
        self.assertEqual(entry, {"version": "1.2.3", "locationUrls": {"DEFAULT": [url]}, "SHA256": "a" * 64})
        for version in ("../escape", "bad/version", "bad\\version", "", "two words"):
            with self.assertRaises(ValueError):
                packager.archive_filename(version, "cu128")


if __name__ == "__main__":
    unittest.main()
