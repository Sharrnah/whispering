"""Export the isolated Linux UI edits as a reviewable companion-repo patch."""
import difflib
import sys
from pathlib import Path

original, staged, output = map(Path, sys.argv[1:])
files = (
    "RuntimeBackend/jobobject_stub.go", "RuntimeBackend/jobobject_windows.go", "RuntimeBackend/process_group_linux_test.go",
    "Utilities/Hardwareinfo/NVIDIAMemory.go", "Utilities/Hardwareinfo/UnknownGPU_test.go", "Pages/Profiles.go",
    "main.go", "RuntimeBackend/Whisper.go", "Updater/Unzip.go",
    "Updater/Unzip_test.go", "UpdateUtility/UpdateCheck.go",
    "Utilities/BackendPath.go", "Utilities/BackendPath_test.go",
    "Updater/Platform.go", "Updater/Platform_test.go",
    "Pages/Ocr.go", "ProfileForm/Builder.go",
    "ProfileForm/Schema.go", "ProfileForm/Schema_audio_cpp_test.go",
)
patch = []
for name in files:
    before = original / name
    after = staged / name
    before_lines = before.read_text(encoding="utf-8").splitlines(keepends=True) if before.exists() else []
    after_lines = after.read_text(encoding="utf-8").splitlines(keepends=True)
    patch.extend(difflib.unified_diff(before_lines, after_lines,
                                    fromfile="a/" + name if before.exists() else "/dev/null",
                                    tofile="b/" + name))
output.write_text("".join(patch), encoding="utf-8")
