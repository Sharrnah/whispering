"""Compile the UI in Docker without passing linker flags through nested shells."""
import os
import subprocess
import sys
from pathlib import Path

flavor = os.environ.get("TORCH_FLAVOR", "cu128")
preview = os.environ.get("WT_LINUX_PREVIEW", "true")
if flavor not in ("cpu", "cu128") or preview not in ("true", "false"):
    raise ValueError("Invalid Linux build flavor or preview mode")
# Updated UI checkouts own their build metadata and platform validation.
ui_builder = Path("BuildTools/build.py")
if ui_builder.is_file():
    command = [sys.executable, str(ui_builder), "--in-place", "--target", "linux",
               "--flavor", flavor, "--test", "--output", "Build/whispering-tiger-linux-amd64"]
    if preview == "false":
        command.append("--release")
    subprocess.run(command, check=True)
    sys.exit(0)
subprocess.run(["xvfb-run", "-a", "go", "test", "./..."], check=True)
subprocess.run([
    "go", "build", "-buildvcs=false", "-ldflags",
    f"-X whispering-tiger-ui/Updater.LinuxBackendFlavor={flavor} "
    f"-X whispering-tiger-ui/Updater.LinuxPreview={preview}",
    "-o", "Build/whispering-tiger-linux-amd64", ".",
], check=True)
