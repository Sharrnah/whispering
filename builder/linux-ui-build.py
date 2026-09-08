"""Compile the UI in Docker without passing linker flags through nested shells."""
import os
import subprocess

flavor = os.environ.get("TORCH_FLAVOR", "cu128")
preview = os.environ.get("WT_LINUX_PREVIEW", "true")
if flavor not in ("cpu", "cu128") or preview not in ("true", "false"):
    raise ValueError("Invalid Linux build flavor or preview mode")
subprocess.run(["xvfb-run", "-a", "go", "test", "./..."], check=True)
subprocess.run([
    "go", "build", "-buildvcs=false", "-ldflags",
    f"-X whispering-tiger-ui/Updater.LinuxBackendFlavor={flavor} "
    f"-X whispering-tiger-ui/Updater.LinuxPreview={preview}",
    "-o", "Build/whispering-tiger-linux-amd64", ".",
], check=True)
