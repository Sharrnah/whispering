"""Package a frozen Windows backend using distribution templates, never profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


def release_files(source):
    for name in ("LICENSE", "ignorelist.txt"):
        yield source / name, name
    for name in ("help.bat", "get-device-list.bat"):
        yield source / "dist_files" / name, name
    for folder, target in (
        ("markers", "markers"), ("websocket_clients", "websocket_clients"),
        ("dist_files/Plugins", "Plugins"),
        ("toolchain/ffmpeg", "toolchain/ffmpeg"), ("toolchain/tcc", "toolchain/tcc"),
    ):
        directory = source / folder
        if not directory.is_dir():
            raise ValueError(f"Missing release directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield path, (Path(target) / path.relative_to(directory)).as_posix()


def package(backend, ui, output, source, version):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("Invalid backend version")
    for executable in (backend / "audioWhisper.exe", ui):
        with executable.open("rb") as stream:
            if stream.read(2) != b"MZ":
                raise ValueError(f"Expected a Windows executable: {executable}")
    extras = list(release_files(source))
    required = [path for path, _ in extras] + [
        source / "toolchain/ffmpeg/bin/ffmpeg.exe",
        source / "toolchain/ffmpeg/bin/ffprobe.exe",
        source / "toolchain/tcc/tcc.exe",
    ]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing release file: {path}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"whispering-tiger{version}_win.zip"
    ui_output = output / "Whispering Tiger.exe"
    if archive.exists() or ui_output.exists():
        raise ValueError(f"Refusing to overwrite existing release artifacts in {output}")
    # ZIP64 supports the multi-gigabyte backend and hidden version receipt.
    # A partial build never appears under the finished archive's name.
    temporary = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(backend.rglob("*")):
            if path.is_file():
                bundle.write(path, (Path("audioWhisper") / path.relative_to(backend)).as_posix())
        for path, name in extras:
            bundle.write(path, name)
        bundle.writestr(".current_platform.yaml", f"version: {json.dumps(version)}\n")
    temporary.replace(archive)
    shutil.copy2(ui, ui_output)
    hashes = {}
    for path in (archive, ui_output):
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8"
    )
    print(json.dumps(hashes, indent=2), flush=True)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", type=Path)
    parser.add_argument("ui", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    package(args.backend, args.ui, args.output, args.source_root, args.version)
