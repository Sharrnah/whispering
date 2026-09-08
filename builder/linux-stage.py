"""Create build snapshots inside Docker; inputs are mounted read-only."""
import shutil
import subprocess
import sys
from pathlib import Path


def snapshot(source, destination, extras):
    destination.mkdir(parents=True, exist_ok=False)
    listing = subprocess.check_output([
        "git", "-c", f"safe.directory={source}", "-C", str(source), "ls-files", "-z"
    ]).decode().split("\0")
    paths = {name for name in listing if name}
    for pattern in extras:
        paths.update(str(path.relative_to(source)) for path in source.glob(pattern) if path.is_file())
    for name in sorted(paths):
        original = source / name
        if not original.is_file():  # Respect locally deleted tracked files.
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)


backend, ui, work = map(Path, sys.argv[1:])
snapshot(backend, work / "backend", [
    "builder/linux-*.py", "builder/ui-linux.patch", "builder/build-linux.ps1",
    "builder/Dockerfile-linux64.dockerignore", "Utilities/linux_audio.py",
    "rthooks/rt_linux_paths.py", "tests/test_linux_audio.py", "tests/test_process_environment.py",
    "tests/test_websocket_server_compatibility.py",
    # Active imports in this developer checkout that are not yet tracked.
    "Models/STT/vibevoice_asr.py", "Models/STT/higgs_audio.py",
    "Models/STT/boson_multimodal/**/*.py", "Models/STT/boson_multimodal/**/*.json",
    "Models/STT/boson_multimodal/**/*.txt", "Models/TTS/compat_parler_transformers.py",
])
snapshot(ui, work / "ui", ["Resources/fonts.go", "Resources/fonts/**/*", "**/*_test.go"])
patch = backend / "builder/ui-linux.patch"
# A clean companion checkout needs the patch; an already updated checkout does not.
check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=work / "ui", capture_output=True)
if check.returncode == 0:
    subprocess.run(["git", "apply", str(patch)], cwd=work / "ui", check=True)
else:
    subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=work / "ui", check=True)
print(f"Build snapshots: {work}")
