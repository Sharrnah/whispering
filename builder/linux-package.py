"""Package the Docker-built ELF files; ZIP entries retain Unix permissions.

Symlinks are dereferenced because the application's Go ZIP reader creates files.
Never copy caches, local profiles, or plugins into the distributable archive.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit


DEFAULT_DOWNLOAD_BASE_URL = "https://s3.libs.space:9000/projects/whispering/"
WAYLAND_RUNTIME_LIBRARIES = (
    "libwayland-egl.so.1", "libwayland-cursor.so.0",
    "libwayland-client.so.0", "libwayland-server.so.0",
)


def elf(path):
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:5] != b"\x7fELF\x02" or int.from_bytes(header[18:20], "little") != 62:
        raise ValueError(f"Expected a Linux x86-64 executable: {path}")


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_filename(version, flavor):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("Version must contain only letters, digits, dots, underscores or hyphens")
    return f"whispering-tiger{version}_linux_amd64_{flavor}.zip"


def backend_download_url(base_url, filename):
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Download base URL must be an HTTP(S) directory URL without a query or fragment")
    return base_url.rstrip("/") + "/" + quote(filename, safe="")


def write_update_fragment(path, flavor, version, archive_sha256, download_url):
    # JSON-quoted scalar values are also valid YAML strings; no YAML dependency
    # is needed to write this fixed schema in the same style as latest.yaml.
    path.write_text(
        "# Build output only: the UI fetches the hosted latest.yaml, not this file.\n"
        "# Add the Linux entry below to its existing packages mapping.\n"
        "# Keep the existing app and ai_platform entries.\n"
        "# Upload the backend ZIP to the URL below before publishing this entry.\n"
        "packages:\n"
        f"  ai_platform_linux_amd64_{flavor}:\n"
        f"    version: {json.dumps(version)}\n"
        "    locationUrls:\n"
        "      DEFAULT:\n"
        f"        - {json.dumps(download_url)}\n"
        f"    SHA256: {json.dumps(archive_sha256)}\n",
        encoding="utf-8",
    )


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


def release_files(source_root):
    """Use distribution templates, never the developer's installed Plugins."""
    files = [(source_root / name, name) for name in ("LICENSE", "ignorelist.txt")]
    for source_dir, destination in (("markers", "markers"), ("websocket_clients", "websocket_clients"),
                                    ("dist_files/Plugins", "Plugins"), ("dist_files/linux", "")):
        directory = source_root / source_dir
        if not directory.is_dir():
            raise ValueError(f"Missing release directory: {directory}")
        for source in sorted(directory.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                files.append((source, (Path(destination) / source.relative_to(directory)).as_posix()))
    for source, name in files:
        if not source.is_file() or not source.resolve().is_relative_to(source_root.resolve()):
            raise ValueError(f"Missing or external release file: {source}")
    required = {"LICENSE", "ignorelist.txt", "Plugins/place_plugins_here.txt", "help.sh", "get-device-list.sh",
                "toolchain/ffmpeg/ffmpeg", "toolchain/ffmpeg/ffprobe", "markers/OKW-MRK.wav",
                "markers/WOK-MRK.wav", "websocket_clients/simple/index.html"}
    if missing := required - {name for _, name in files}:
        raise ValueError(f"Missing release files: {sorted(missing)}")
    return files


def write_text_member(bundle, name, data, mode=0o644):
    header = zipfile.ZipInfo(name)
    header.create_system = 3
    header.external_attr = (stat.S_IFREG | mode) << 16
    header.compress_type = zipfile.ZIP_DEFLATED
    bundle.writestr(header, data)


def media_runtime_files(backend):
    # PyInstaller excludes Wayland libraries as system dependencies. Debian's
    # FFmpeg loads SDL, which requires them even for audio-only command usage.
    # Include these client libraries without bundling display/GPU drivers.
    files = []
    for name in WAYLAND_RUNTIME_LIBRARIES:
        if (backend / "_internal" / name).is_file():
            continue
        source = Path("/usr/lib/x86_64-linux-gnu") / name
        if not source.is_file():
            raise ValueError(f"Missing FFmpeg runtime dependency in the Linux builder: {name}")
        files.append((source, f"audioWhisper/_internal/{name}"))
    return files


def audio_cpp_runtime_files(source_root, runtime_root):
    tree = ast.parse((source_root / "Models/audio_cpp_runtime.py").read_text(encoding="utf-8"))
    constants = {target.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)
                 and target.id in {"AUDIO_CPP_VERSION", "LINUX_BUNDLE_REVISION"}}
    version = constants["AUDIO_CPP_VERSION"]
    directory = runtime_root / f"v{version}-r{constants['LINUX_BUNDLE_REVISION']}-linux-x86_64"
    required = {"audiocpp_server", "libggml.so.0", "libggml-base.so.0", "libggml-vulkan.so",
                "libggml-cpu-x64.so", "libvulkan.so.1", "libgomp.so.1", "LICENSE"}
    if missing := {name for name in required if not (directory / name).is_file()}:
        raise ValueError(f"Missing bundled audio.cpp {version} runtime files (rebuild the Linux image): {sorted(missing)}")
    elf(directory / "audiocpp_server")
    if any(directory.glob("libstdc++*")) or any(directory.glob("libgcc_s*")):
        raise ValueError("audio.cpp must use system C++/GCC libraries so current Vulkan drivers can load")
    return [(source, (Path("toolchain/audio.cpp") / directory.name / source.relative_to(directory)).as_posix())
            for source in sorted(directory.rglob("*")) if source.is_file()]


def installation_notes(flavor, version, release):
    return (
        f"Whispering Tiger Linux backend {version} ({flavor})\n"
        "Extract this ZIP beside the Whispering Tiger Linux UI in a writable folder.\n"
        "Run ./whispering-tiger-linux-amd64 as your normal desktop user.\n"
        + ("Release UI builds can check the application's update feed.\n" if release else
           "Preview UI builds have automatic updates disabled.\n")
        + "Requires x86-64 Linux with glibc 2.36+, an OpenGL-capable X11/XWayland desktop,\n"
        "and PulseAudio or PipeWire's PulseAudio compatibility service.\n"
        "Run ./help.sh for backend options or ./get-device-list.sh for audio devices.\n"
        "FFmpeg/FFprobe and their libraries are bundled; terminal launchers are in toolchain/ffmpeg/.\n"
        "audio.cpp CPU/Vulkan and its libraries are bundled in toolchain/audio.cpp/.\n"
        "Vulkan GPU use needs the system graphics driver; no Vulkan SDK is needed.\n"
        + ("CUDA 12.8 runtime, cuBLAS and cuDNN are bundled; no separate CUDA Toolkit is needed.\n"
           "GPU use requires a compatible NVIDIA GPU (Turing or newer) and Linux driver (570.26+ recommended).\n"
           if flavor == "cu128" else "This package uses the CPU backend.\n")
        + "Place your plugins in Plugins/ and use websocket_clients/ for browser overlays.\n"
        "Profiles and model weights are created/downloaded when needed.\n"
    )


def write_archive(backend, archive, source_root, version, audio_cpp_root=None):
    extras = release_files(source_root)
    media_libraries = media_runtime_files(backend)
    audio_cpp_files = audio_cpp_runtime_files(source_root, audio_cpp_root) if audio_cpp_root is not None else []
    for tool in ("ffmpeg", "ffprobe"):
        if not (backend / "_internal/bin" / tool).is_file():
            raise ValueError(f"Missing bundled media tool: {tool}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source in sorted(backend.rglob("*")):
            if source.is_file():
                # ZipFile.write follows the symlink's content and stat mode.
                bundle.write(source, Path("audioWhisper") / source.relative_to(backend))
        for source, name in media_libraries:
            bundle.write(source, name)
        for source, name in audio_cpp_files:
            bundle.write(source, name)
        for notice in sorted(Path("/usr/share/doc").glob("*/copyright")):
            if notice.is_file():
                bundle.write(notice, Path("audioWhisper/_internal/licenses/debian") / notice.parent.name / "copyright")
        for source, name in extras:
            executable = name.endswith(".sh") or name in ("toolchain/ffmpeg/ffmpeg", "toolchain/ffmpeg/ffprobe")
            data = source.read_bytes()
            if executable:
                data = data.replace(b"\r\n", b"\n")
            write_text_member(bundle, name, data, 0o755 if executable else 0o644)
        # Only the installed version is needed here. The ZIP's own hash cannot
        # be embedded inside itself; the UI writes the full receipt on download.
        write_text_member(bundle, ".current_platform.yaml", f"version: {json.dumps(version)}\n")
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
    parser.add_argument("--download-base-url", default=DEFAULT_DOWNLOAD_BASE_URL,
                        help="Backend ZIP directory on the download server (default: beside latest.yaml)")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--audio-cpp-root", type=Path, default=Path("/opt/audio-cpp"))
    parser.add_argument("--release", action="store_true", help="Write release rather than preview installation notes")
    args = parser.parse_args()
    archive = args.output / archive_filename(args.version, args.flavor)
    download_url = backend_download_url(args.download_base_url, archive.name)
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
    notes = installation_notes(args.flavor, args.version, args.release)
    # Compress on Docker's native filesystem; shared Windows paths are slow
    # for the many small writes produced by the ZIP compressor.
    with tempfile.TemporaryDirectory(prefix="wt-linux-package-") as temporary:
        native_archive = Path(temporary) / archive.name
        write_archive(args.backend, native_archive, args.source_root, args.version, args.audio_cpp_root)
        with native_archive.open("rb") as source, archive.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    ui_output = args.output / "whispering-tiger-linux-amd64"
    shutil.copy2(args.ui, ui_output)
    ui_output.chmod(0o755)
    package_inventory = args.backend.parent / "linux-python-packages.txt"
    if package_inventory.is_file():
        shutil.copy2(package_inventory, args.output / package_inventory.name)
    if runtime_info.is_file():
        shutil.copy2(runtime_info, args.output / runtime_info.name)
    provenance = args.source_root.parent / "build-sources.json"
    if provenance.is_file():
        shutil.copy2(provenance, args.output / provenance.name)
    hashes = {path.name: checksum(path) for path in (archive, ui_output)}
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")
    write_update_fragment(args.output / "latest-linux.fragment.yaml",
                          args.flavor, args.version, hashes[archive.name], download_url)
    (args.output / "README-LINUX.txt").write_text(notes, encoding="utf-8")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
