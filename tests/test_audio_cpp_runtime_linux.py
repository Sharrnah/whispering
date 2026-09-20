import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Models import audio_cpp_runtime as runtime


class BundledAudioCppTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server = self.root / "toolchain/audio.cpp" / f"v{runtime.AUDIO_CPP_VERSION}-r{runtime.LINUX_BUNDLE_REVISION}-linux-x86_64/audiocpp_server"
        self.server.parent.mkdir(parents=True)
        self.server.touch()

    def test_linux_bundle_precedes_managed_downloads(self):
        with mock.patch.object(runtime, "_normalized_system", return_value="linux"), \
                mock.patch.object(runtime, "_normalized_machine", return_value="x86_64"), \
                mock.patch.object(runtime, "_explicit_server_path", return_value=None), \
                mock.patch.object(runtime, "RUNTIME_PACKAGES", {}), \
                mock.patch.object(runtime.Path, "cwd", return_value=self.root):
            self.assertEqual(runtime.ensure_runtime("cpu"), self.server.resolve())
            self.assertEqual(runtime.ensure_runtime("vulkan"), self.server.resolve())
            self.assertIsNone(runtime._bundled_server_path("cuda"))

    def test_custom_server_override_is_preserved(self):
        custom = self.root / "custom-server"
        custom.touch()
        with mock.patch.dict(os.environ, {"WHISPERING_TIGER_AUDIOCPP_SERVER": str(custom)}):
            self.assertEqual(runtime.ensure_runtime("vulkan"), custom.resolve())

    def test_old_bundle_cannot_shadow_system_graphics_driver_libraries(self):
        old = self.root / "toolchain/audio.cpp" / f"v{runtime.AUDIO_CPP_VERSION}-linux-x86_64"
        old.mkdir()
        (old / "audiocpp_server").touch()
        (old / "libstdc++.so.6").touch()
        with mock.patch.object(runtime, "_normalized_system", return_value="linux"), \
                mock.patch.object(runtime, "_normalized_machine", return_value="x86_64"), \
                mock.patch.object(runtime.Path, "cwd", return_value=self.root):
            self.assertEqual(runtime._bundled_server_path("vulkan"), self.server.resolve())

    def test_frozen_bundle_can_be_found_from_another_working_directory(self):
        with mock.patch.object(runtime, "_normalized_system", return_value="linux"), \
                mock.patch.object(runtime, "_normalized_machine", return_value="x86_64"), \
                mock.patch.object(runtime.sys, "frozen", True, create=True), \
                mock.patch.object(runtime.sys, "executable", str(self.root / "audioWhisper/audioWhisper")), \
                mock.patch.object(runtime.Path, "cwd", return_value=self.root / "elsewhere"):
            self.assertEqual(runtime._bundled_server_path("cpu"), self.server.resolve())

    def test_windows_does_not_use_the_linux_bundle_or_library_environment(self):
        with mock.patch.object(runtime, "_normalized_system", return_value="windows"):
            self.assertIsNone(runtime._bundled_server_path("cpu"))
            self.assertIsNone(runtime._server_environment(self.server))

    def test_native_server_does_not_inherit_pyinstaller_libraries(self):
        with mock.patch.object(runtime, "_normalized_system", return_value="linux"), \
                mock.patch.object(runtime.sys, "frozen", True, create=True), \
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/frozen/_internal", "LD_LIBRARY_PATH_ORIG": "/user/lib"}):
            self.assertEqual(runtime._server_environment(self.server),
                             {"LD_LIBRARY_PATH": str(self.server.parent) + os.pathsep + "/user/lib"})
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/frozen/_internal")
        with mock.patch.object(runtime, "_normalized_system", return_value="linux"), \
                mock.patch.object(runtime.sys, "frozen", True, create=True), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime._server_environment(self.server), {"LD_LIBRARY_PATH": str(self.server.parent)})


if __name__ == "__main__":
    unittest.main()
