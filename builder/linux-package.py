"""Package the Docker-built ELF files; ZIP entries retain Unix permissions.

Symlinks are dereferenced because the application's Go ZIP reader creates files.
Never copy caches, local profiles, or plugins into the distributable archive.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path


def elf(path):
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:5] != b"\x7fELF\x02" or int.from_bytes(header[18:20], "little") != 62:
        raise ValueError(f"Expected a Linux x86-64 executable: {path}")


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_cuda_linkage(internal):
    """Check bundled CUDA dependency closure; only the target driver is external.

    This is a packaging check, not GPU inference. The frozen startup test also
    checks Torch's actual runtime search paths without a CUDA installation.
    """
    directories = [internal, internal / 'torch/lib'] + sorted((internal / 'nvidia').glob('*/lib'))
    environment = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(map(str, directories)))
    for name in ('libtorch_cuda.so', 'libc10_cuda.so', 'libonnxruntime_providers_cuda.so',
                 'libbitsandbytes_cuda128.so'):
        library = next(internal.rglob(name))
        result = subprocess.run(['ldd', str(library)], env=environment,
                                text=True, capture_output=True, check=True)
        missing = {line.strip().split()[0] for line in result.stdout.splitlines() if '=> not found' in line}
        missing -= {'libcuda.so.1', 'libnvidia-ml.so.1'}
        if missing:
            raise ValueError(f'{name} has missing runtime dependencies: {sorted(missing)}')


def write_archive(backend, archive):
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source in sorted(backend.rglob("*")):
            if source.is_file():
                # ZipFile.write follows the symlink's content and stat mode.
                bundle.write(source, Path("audioWhisper") / source.relative_to(backend))
        for notice in sorted(Path("/usr/share/doc").glob("*/copyright")):
            if notice.is_file():
                bundle.write(notice, Path("audioWhisper/_internal/licenses/debian") / notice.parent.name / "copyright")
    with zipfile.ZipFile(archive) as bundle:
        if not stat.S_IMODE(bundle.getinfo("audioWhisper/audioWhisper").external_attr >> 16) & stat.S_IXUSR:
            raise ValueError("ZIP lost the backend's executable permission")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", type=Path)
    parser.add_argument("ui", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flavor", choices=("cpu", "cu128"), default="cu128")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    executable = args.backend / "audioWhisper"
    elf(executable)
    elf(args.ui)
    runtime_info = args.backend.parent / 'linux-runtime-info.json'
    if args.flavor == 'cu128':
        runtime = json.loads(runtime_info.read_text())
        if runtime.get('torch_cuda') != '12.8' or 'CUDAExecutionProvider' not in runtime.get('onnx_providers', []):
            raise ValueError('CUDA package requires verified CUDA Torch and ONNX Runtime')
        library_names = {path.name for path in (args.backend / '_internal').rglob('*') if path.is_file()}
        required = {'libtorch_cuda.so', 'libc10_cuda.so', 'libcudart.so.12',
                    'libcublas.so.12', 'libcublasLt.so.12', 'libcudnn.so.9',
                    'libnvrtc.so.12', 'libnvJitLink.so.12',
                    'libonnxruntime_providers_cuda.so', 'libbitsandbytes_cuda128.so'}
        if missing := required - library_names:
            raise ValueError(f'CUDA package is missing runtime libraries: {sorted(missing)}')
        if {'libcuda.so', 'libcuda.so.1'} & library_names:
            raise ValueError('NVIDIA driver/stub libraries must come from the target system')
        validate_cuda_linkage(args.backend / '_internal')
    if not (args.backend / "_internal/nltk_data/tokenizers/punkt_tab/english").is_dir():
        raise ValueError("The frozen backend is missing its NLTK sentence tokenizer")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"audioWhisper_linux_amd64_{args.flavor}.zip"
    # Compress on Docker's native filesystem; shared Windows paths are slow
    # for the many small writes produced by the ZIP compressor.
    with tempfile.TemporaryDirectory(prefix="wt-linux-package-") as temporary:
        native_archive = Path(temporary) / archive.name
        write_archive(args.backend, native_archive)
        with native_archive.open("rb") as source, archive.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    ui_output = args.output / f"whispering-tiger-linux-amd64-{args.flavor}"
    shutil.copy2(args.ui, ui_output)
    ui_output.chmod(0o755)
    package_inventory = args.backend.parent / "linux-python-packages.txt"
    if package_inventory.is_file():
        shutil.copy2(package_inventory, args.output / package_inventory.name)
    if runtime_info.is_file():
        shutil.copy2(runtime_info, args.output / runtime_info.name)
    hashes = {path.name: checksum(path) for path in (archive, ui_output)}
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")
    # Reviewable publication fragment only. It deliberately has no download URL.
    manifest = {"packages": {f"ai_platform_linux_amd64_{args.flavor}": {
        "version": args.version, "locationUrls": {}, "SHA256": hashes[archive.name]
    }}}
    # JSON is valid YAML and avoids another dependency in this packaging step.
    (args.output / "latest-linux.fragment.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "README-LINUX.txt").write_text(
        "Private Linux test build - no updates are published.\n"
        "Extract the backend ZIP beside the UI executable in a fresh writable folder.\n"
        f"Run: chmod +x {ui_output.name} audioWhisper/audioWhisper\n"
        f"Then run ./{ui_output.name} as your normal desktop user.\n"
        "Requires x86-64 Linux with glibc 2.36+, an OpenGL-capable X11/XWayland desktop,\n"
        "and PulseAudio or PipeWire's PulseAudio compatibility service.\n"
        f"Backend flavor: {args.flavor}.\n"
        + ("CUDA 12.8 runtime, cuBLAS and cuDNN are bundled; no separate CUDA Toolkit is needed.\n"
           "GPU use requires a compatible NVIDIA GPU and Linux NVIDIA driver (570.26+ recommended).\n"
           "This Torch wheel targets compute capability 7.5 and newer (Turing or newer).\n"
           "This package was built without a GPU; CUDA inference needs manual Linux validation.\n"
           if args.flavor == 'cu128' else "CPU build; GPU execution requires the separate CUDA package.\n")
        +
        "Model weights download on demand into this folder's .cache directory.\n",
        encoding="utf-8")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
