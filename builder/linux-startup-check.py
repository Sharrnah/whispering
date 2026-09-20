"""Reject a broken frozen backend before spending time building/packaging the UI."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def check_startup(executable, log_path, timeout=180):
    executable = executable.resolve()
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Run away from source files, profiles and model caches. --help imports the
    # real application but exits before model downloads or recording. The build
    # container supplies PulseAudio because PortAudio initializes during imports.
    with tempfile.TemporaryDirectory(prefix="wt-frozen-startup-") as work:
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run([str(executable), "--help"], cwd=work,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Frozen backend startup timed out; see {log_path}") from error
    output = log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0 or "usage:" not in output.lower():
        tail = "\n".join(output.splitlines()[-60:])
        raise RuntimeError(f"Frozen backend startup failed (exit {result.returncode}); see {log_path}\n{tail}")
    print(f"PASS: frozen backend imports and --help without source files; log: {log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    check_startup(args.executable, args.log)
