"""Create build snapshots inside Docker; inputs are mounted read-only."""
import shutil
import importlib.util
import json
import os
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


def snapshot_backend(backend, destination):
    snapshot(backend, destination, [
        "builder/linux-*.py", "builder/build-linux.ps1", "dist_files/linux/**/*",
        "builder/Dockerfile-linux64.dockerignore", "Utilities/linux_audio.py",
        "rthooks/rt_linux_paths.py", "tests/test_linux_audio.py", "tests/test_process_environment.py",
        "tests/test_websocket_server_compatibility.py", "tests/test_linux_packaging.py",
        "tests/test_audio_cpp_runtime_linux.py", "tests/test_linux_build.py",
        "tests/test_audio_playback.py",
        # Active imports in this developer checkout that are not yet tracked.
        "streaming_text.py", "streaming_display.py",
        "Models/STT/vibevoice*.py", "Models/STT/vibevoice_streaming_runtime/**/*", "Models/STT/higgs_audio.py",
        "Models/STT/boson_multimodal/**/*.py", "Models/STT/boson_multimodal/**/*.json",
        "Models/STT/boson_multimodal/**/*.txt", "Models/STT/boson_multimodal/**/LICENSE",
        "Models/TTS/compat_parler_transformers.py", "websocket_clients/**/*.js",
    ])


def main():
    backend, ui, work = map(Path, sys.argv[1:])
    ui_builder = ui / "BuildTools/build.py"
    if not ui_builder.is_file() or not (ui / "Updater/Platform.go").is_file():
        raise RuntimeError("Update the UI repository: its Linux support and BuildTools/build.py are required")
    snapshot_backend(backend, work / "backend")
    spec = importlib.util.spec_from_file_location("ui_build", ui_builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.snapshot(ui, work / "ui")
    provenance = {}
    for name, source in (("backend", backend), ("ui", ui)):
        command = ["git", "-c", f"safe.directory={source}", "-C", str(source)]
        provenance[name] = {
            "repository": os.environ.get(f"WT_{name.upper()}_REPOSITORY", str(source)),
            "commit": subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip(),
            "worktree_changes": bool(subprocess.check_output(command + ["status", "--porcelain"])),
        }
    (work / "build-sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Sources copied from the backend and UI repositories into the Docker build volume; no patches applied.")


if __name__ == "__main__":
    main()
