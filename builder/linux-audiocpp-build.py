"""Build the pinned CPU/Vulkan server on Debian 12; no GPU is needed."""
import argparse
from pathlib import Path
import shutil
import subprocess

VERSION = "0.8.1"
BUNDLE_REVISION = 2
REVISION = "f2b4937306daa25f5c78520f3c626ed31495a37a"
REPOSITORY = "https://github.com/0xShug0/audio.cpp.git"
VULKAN_HEADERS_REVISION = "2cd90f9d20df57eac214c148f3aed885372ddcfe"
VULKAN_HPP_REVISION = "78243585183d42c93eeb2c57f9a194c7cac40bcc"


def run(*command):
    subprocess.run(list(map(str, command)), check=True)


def checkout(directory, repository, expected):
    if not directory.exists():
        run("git", "init", directory)
        run("git", "-C", directory, "fetch", "--depth", "1", repository, expected)
        run("git", "-C", directory, "checkout", "--detach", "FETCH_HEAD")
    revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected:
        raise ValueError(f"Expected {expected} in {directory}, found {revision}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/tmp/audio-cpp-source"))
    parser.add_argument("--build", type=Path, default=Path("/tmp/audio-cpp-build"))
    parser.add_argument("--output", type=Path, default=Path("/opt/audio-cpp"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--package-only", action="store_true", help="Package an already validated build")
    args = parser.parse_args()
    checkout(args.source, REPOSITORY, REVISION)
    headers = args.build.parent / "audio-cpp-vulkan-headers"
    hpp = args.build.parent / "audio-cpp-vulkan-hpp"
    if not args.package_only:
        # Debian's loader is sufficient at runtime, but ggml needs newer C/C++
        # declarations than Debian 12's SDK headers. Pin both to v1.4.321.
        checkout(headers, "https://github.com/KhronosGroup/Vulkan-Headers.git", VULKAN_HEADERS_REVISION)
        checkout(hpp, "https://github.com/KhronosGroup/Vulkan-Hpp.git", VULKAN_HPP_REVISION)
        include = args.build.parent / "audio-cpp-vulkan-include"
        shutil.copytree(headers / "include", include, dirs_exist_ok=True)
        shutil.copytree(hpp / "vulkan", include / "vulkan", dirs_exist_ok=True)
        run("cmake", "-S", args.source, "-B", args.build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DAUDIOCPP_DEPLOYMENT_BUILD=ON",
            "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF", "-DENGINE_ENABLE_NATIVE_CPU=OFF",
            "-DENGINE_ENABLE_CPU_ALL_VARIANTS=ON", "-DENGINE_ENABLE_CUDA=OFF",
            "-DENGINE_ENABLE_VULKAN=ON", "-DENGINE_BUILD_TESTS=OFF", "-DENGINE_BUILD_EXAMPLES=OFF",
            f"-DVulkan_INCLUDE_DIR={include}")
        run("cmake", "--build", args.build, "--parallel", args.jobs, "--target", "audiocpp_server")
    output = args.output / f"v{VERSION}-r{BUNDLE_REVISION}-linux-x86_64"
    output.mkdir(parents=True, exist_ok=True)
    binaries = [args.build / "bin/audiocpp_server", *sorted((args.build / "bin").glob("*.so*"))]
    # Use the target's C++/GCC runtime: Mesa/LLVM may require newer GLIBCXX or
    # GCC symbols than Debian 12 provides. Keep the app's OpenMP dependency.
    for name in ("libvulkan.so.1", "libgomp.so.1"):
        binaries.append(Path("/usr/lib/x86_64-linux-gnu") / name)
    for source in binaries:
        target = output / source.name
        shutil.copy2(source, target)  # Dereference symlinks for the Go ZIP extractor.
        target.chmod(0o755)
        run("strip", "--strip-unneeded", target)
        run("patchelf", "--set-rpath", "$ORIGIN", target)
    shutil.copy2(args.source / "LICENSE", output / "LICENSE")
    for source in (args.source / "external").rglob("*"):
        if source.is_file() and source.name.lower() in {"license", "license.txt", "license.md", "copying", "notice"}:
            target = output / "licenses" / source.relative_to(args.source / "external")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("libvulkan1", "libgomp1"):
        target = output / "licenses" / name / "copyright"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("/usr/share/doc") / name / "copyright", target)
    for source_root in (headers, hpp):
        for source in source_root.glob("LICENSE*"):
            target = output / "licenses" / source_root.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
    run(output / "audiocpp_server", "--list-devices")
    print(f"audio.cpp {VERSION} CPU/Vulkan runtime: {output}", flush=True)


if __name__ == "__main__":
    main()
