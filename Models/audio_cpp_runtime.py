"""Managed audio.cpp runtime and persistent local server process.

audio.cpp is a native GGML/GGUF runtime, not a PyTorch device.  The adapters in
this application therefore talk to its loopback-only HTTP server and keep that
server alive between requests.  Runtime archives are pinned to an upstream
release and verified before extraction; ``WHISPERING_TIGER_AUDIOCPP_SERVER``
can point at a custom build (notably a HIP/ROCm build).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

import processmanager


AUDIO_CPP_VERSION = "0.8.1"
# A new directory prevents C++ libraries left by older ZIP installs from
# shadowing the target system's newer Mesa/Vulkan driver dependencies.
LINUX_BUNDLE_REVISION = 2
CACHE_ROOT = Path.cwd() / ".cache" / "audio.cpp"
RUNTIME_ROOT = CACHE_ROOT / "runtime"
SERVER_CONFIG_ROOT = CACHE_ROOT / "server"
RUNTIME_RECEIPT = ".whispering_tiger_runtime.json"
SERVER_START_TIMEOUT_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 15 * 60

_RELEASE_ROOT = (
    "https://github.com/0xShug0/audio.cpp/releases/download/"
    f"v{AUDIO_CPP_VERSION}"
)


def _package(filename: str, checksum: str, extract_format: str) -> dict[str, Any]:
    return {
        "urls": [f"{_RELEASE_ROOT}/{filename}"],
        "checksum": checksum,
        "extract_format": extract_format,
        "filename": filename,
    }


# CPU support is included in every audio.cpp build.  Separate CPU packages are
# still used where available so CPU-only users do not download GPU libraries.
RUNTIME_PACKAGES: dict[tuple[str, str, str], tuple[dict[str, Any], ...]] = {
    ("windows", "x86_64", "cpu"): (
        _package(
            "audio-v0.8.1-bin-windows-x64-cpu-portable.zip",
            "fc6a20cc881b0882569d0eca060235a1904863b96b531f79145ce00acf8f8bfd",
            "zip",
        ),
    ),
    ("windows", "x86_64", "vulkan"): (
        _package(
            "audio-v0.8.1-bin-windows-x64-vulkan.zip",
            "c787971e025ba8ef900f0482a2cc36a049367081fe89f4841aae521a0b49de32",
            "zip",
        ),
    ),
    ("windows", "x86_64", "cuda"): (
        _package(
            "audio-v0.8.1-bin-windows-x64-cuda12.4.zip",
            "28bbe8ac62a06c5d9d42ba3066b051f433dc9a8f456c544e03e87202f0fa8c52",
            "zip",
        ),
        _package(
            "audio-v0.8.1-cudart-windows-x64-cuda12.4.zip",
            "025faacfdc3dec215ee07cb9be7d1ef2016402723f3721a30500ceee02cc4701",
            "zip",
        ),
    ),
    ("linux", "x86_64", "cpu"): (
        _package(
            "audio-v0.8.1-bin-ubuntu-x64-cpu-portable.tar.gz",
            "90e8d538338cc209875a18c940529302805563e54738489da1d684c6e0de12d0",
            "tar.gz",
        ),
    ),
    ("linux", "x86_64", "vulkan"): (
        _package(
            "audio-v0.8.1-bin-ubuntu-x64-vulkan.tar.gz",
            "54070c724b663e3387c750498eba1a152eba9ad711aa247fd9d8addc1a73de02",
            "tar.gz",
        ),
    ),
    ("darwin", "x86_64", "metal"): (
        _package(
            "audio-v0.8.1-bin-macos-x64-metal.tar.gz",
            "637fb5b47f92a8d01724e288a4d38da3b3d1a34799f2b7ff3bd0d01aabb2e563",
            "tar.gz",
        ),
    ),
    ("darwin", "arm64", "metal"): (
        _package(
            "audio-v0.8.1-bin-macos-arm64-metal.tar.gz",
            "a5995233c4e28297600c474eed24b734a3ff8f00393147915112b2b4d07ab593",
            "tar.gz",
        ),
    ),
}


def _normalized_system() -> str:
    return platform.system().strip().lower()


def _normalized_machine() -> str:
    machine = platform.machine().strip().lower()
    if machine in {"amd64", "x64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    return machine


def normalize_backend_device(device: Any, device_index: Any = 0) -> tuple[str, int]:
    """Return an audio.cpp backend name and its non-negative adapter index."""
    value = str(device or "cpu").strip().lower()
    if value in {"", "none", "auto"}:
        value = "cpu"
    if value.startswith("direct-ml"):
        raise ValueError(
            "audio.cpp does not use DirectML. Select Vulkan for an AMD or Intel GPU."
        )

    parsed_index = device_index
    if ":" in value:
        value, embedded_index = value.split(":", 1)
        parsed_index = embedded_index
    if value == "rocm":
        value = "hip"
    if value not in {"cpu", "cuda", "vulkan", "hip", "metal"}:
        raise ValueError(f"Unsupported audio.cpp backend: {device!r}")
    try:
        parsed_index = int(parsed_index or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid audio.cpp device index: {device_index!r}") from exc
    if parsed_index < 0:
        raise ValueError("audio.cpp device index must be non-negative.")
    return value, parsed_index


def _server_filename() -> str:
    return "audiocpp_server.exe" if _normalized_system() == "windows" else "audiocpp_server"


def _find_server(directory: Path) -> Path | None:
    direct = directory / _server_filename()
    if direct.is_file():
        return direct
    matches = list(directory.rglob(_server_filename())) if directory.is_dir() else []
    return matches[0] if matches else None


def _explicit_server_path() -> Path | None:
    for variable in (
        "WHISPERING_TIGER_AUDIOCPP_SERVER",
        "AUDIOCPP_SERVER_PATH",
    ):
        configured = str(os.environ.get(variable, "") or "").strip()
        if not configured:
            continue
        path = Path(configured).expanduser()
        if path.is_dir():
            path = path / _server_filename()
        if not path.is_file():
            raise FileNotFoundError(f"{variable} does not point to an audio.cpp server: {path}")
        return path.resolve()
    return None


def _receipt_payload(packages: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return {
        "version": AUDIO_CPP_VERSION,
        "archives": [
            {"filename": item["filename"], "sha256": item["checksum"]}
            for item in packages
        ],
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


_runtime_install_lock = threading.RLock()


def _bundled_server_path(backend: str) -> Path | None:
    if (_normalized_system(), _normalized_machine()) != ("linux", "x86_64") or backend not in {"cpu", "vulkan"}:
        return None
    roots = [Path.cwd()]
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent.parent)
    for root in roots:
        server = root / "toolchain" / "audio.cpp" / f"v{AUDIO_CPP_VERSION}-r{LINUX_BUNDLE_REVISION}-linux-x86_64" / "audiocpp_server"
        if server.is_file():
            return server.resolve()
    return None


def espeak_paths() -> tuple[str, str]:
    # Already shipped by the application, including its matching voice data.
    import espeakng_loader
    return str(espeakng_loader.get_library_path()), str(espeakng_loader.get_data_path())


def _server_environment(server: Path, family: str = "") -> dict[str, str] | None:
    environment = {}
    if family == "kokoro_tts":
        library, data = espeak_paths()
        environment["AUDIOCPP_ESPEAK_LIBRARY"] = os.environ.get("AUDIOCPP_ESPEAK_LIBRARY") or library
        environment["AUDIOCPP_ESPEAK_DATA"] = os.environ.get("AUDIOCPP_ESPEAK_DATA") or data
    if _normalized_system() != "linux":
        return environment or None
    # External native servers must not inherit PyInstaller's private library
    # directory. Prefer their own libraries and retain the original user path.
    variable = "LD_LIBRARY_PATH_ORIG" if getattr(sys, "frozen", False) else "LD_LIBRARY_PATH"
    original = os.environ.get(variable, "")
    environment["LD_LIBRARY_PATH"] = str(server.parent) + (os.pathsep + original if original else "")
    return environment


def ensure_runtime(backend: str, force_non_ui_dl: bool = False) -> Path:
    """Resolve or install the pinned audio.cpp server for ``backend``."""
    explicit = _explicit_server_path()
    if explicit is not None:
        return explicit

    bundled = _bundled_server_path(backend)
    if bundled is not None:
        return bundled

    system = _normalized_system()
    machine = _normalized_machine()
    package_backend = backend
    # Official macOS builds include both Metal and CPU support.
    if system == "darwin" and backend == "cpu":
        package_backend = "metal"

    key = (system, machine, package_backend)
    packages = RUNTIME_PACKAGES.get(key)
    if packages is None:
        path_server = shutil.which(_server_filename()) or shutil.which("audiocpp_server")
        if path_server:
            return Path(path_server).resolve()
        if backend == "hip":
            raise RuntimeError(
                "The official audio.cpp release has no HIP/ROCm binary for this platform. "
                "Build audio.cpp with HIP support and set "
                "WHISPERING_TIGER_AUDIOCPP_SERVER to audiocpp_server. "
                "For AMD and Intel GPUs, the managed Vulkan build is the easiest option."
            )
        raise RuntimeError(
            f"No managed audio.cpp {AUDIO_CPP_VERSION} runtime is published for "
            f"{system}/{machine}/{backend}. Set WHISPERING_TIGER_AUDIOCPP_SERVER "
            "to a compatible custom audiocpp_server binary."
        )

    runtime_directory = RUNTIME_ROOT / f"v{AUDIO_CPP_VERSION}-{system}-{machine}-{package_backend}"
    expected_receipt = _receipt_payload(packages)
    with _runtime_install_lock:
        server = _find_server(runtime_directory)
        receipt = _read_json(runtime_directory / RUNTIME_RECEIPT)
        if server is not None and receipt == expected_receipt:
            return server.resolve()

        runtime_directory.mkdir(parents=True, exist_ok=True)
        try:
            (runtime_directory / RUNTIME_RECEIPT).unlink()
        except FileNotFoundError:
            pass
        # downloader imports the WebSocket/TTS registry.  Keep it out of this
        # module's import path so that registry initialization can safely import
        # AudioCppServer without recursing through a partially defined module.
        import downloader

        for package_info in packages:
            ok = downloader.download_extract(
                package_info["urls"],
                str(runtime_directory.resolve()),
                package_info["checksum"],
                title=f"audio.cpp {AUDIO_CPP_VERSION} ({package_backend})",
                extract_format=package_info["extract_format"],
                force_non_ui_dl=force_non_ui_dl,
            )
            if not ok:
                raise RuntimeError(
                    f"Could not download the audio.cpp {package_backend} runtime."
                )

        server = _find_server(runtime_directory)
        if server is None:
            raise FileNotFoundError(
                f"The verified audio.cpp archive did not contain {_server_filename()}."
            )
        if system != "windows":
            server.chmod(server.stat().st_mode | 0o111)
        _write_json_atomic(runtime_directory / RUNTIME_RECEIPT, expected_receipt)
        return server.resolve()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if error:
                return str(error)
    except (ValueError, TypeError):
        pass
    return response.text.strip() or response.reason or f"HTTP {response.status_code}"


class AudioCppServer:
    """One loopback-only persistent audio.cpp process for one adapter role."""

    def __init__(self, role: str):
        self.role = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in role)
        self.process = None
        self.base_url: str | None = None
        self.configuration_key: tuple[Any, ...] | None = None
        self.model_id: str | None = None
        self._lock = threading.RLock()

    def configure(
        self,
        *,
        backend: str,
        device_index: int,
        family: str,
        model_path: Path,
        task: str,
        mode: str,
        threads: int = 0,
        load_options: dict[str, Any] | None = None,
        session_options: dict[str, Any] | None = None,
        default_request_options: dict[str, Any] | None = None,
        force_non_ui_dl: bool = False,
    ) -> None:
        backend, device_index = normalize_backend_device(backend, device_index)
        resolved_model = Path(model_path).resolve()
        if not resolved_model.is_file() and not resolved_model.is_dir():
            raise FileNotFoundError(f"audio.cpp model does not exist: {resolved_model}")
        server_path = ensure_runtime(backend, force_non_ui_dl=force_non_ui_dl)
        configuration_key = (
            str(server_path),
            backend,
            device_index,
            family,
            str(resolved_model),
            task,
            mode,
            int(threads or 0),
            json.dumps(load_options or {}, sort_keys=True),
            json.dumps(session_options or {}, sort_keys=True),
            json.dumps(default_request_options or {}, sort_keys=True),
        )

        with self._lock:
            if (
                configuration_key == self.configuration_key
                and self.process is not None
                and self.process.poll() is None
                and self.base_url is not None
            ):
                return
            self._stop_locked()

            port = _free_loopback_port()
            model_id = f"whispering-tiger-{self.role}"
            model_config: dict[str, Any] = {
                "id": model_id,
                "family": family,
                "path": str(resolved_model),
                "task": task,
                "mode": mode,
            }
            if load_options:
                model_config["load_options"] = load_options
            if session_options:
                model_config["session_options"] = session_options
            if default_request_options:
                model_config["default_request_options"] = default_request_options
            config = {
                "host": "127.0.0.1",
                "port": port,
                "backend": backend,
                "device": device_index,
                "threads": max(1, int(threads or (os.cpu_count() or 4))),
                "lazy_load": True,
                "max_loaded_models": 1,
                "busy_timeout_ms": REQUEST_TIMEOUT_SECONDS * 1000,
                "log_request_body": False,
                "models": [model_config],
            }
            config_path = SERVER_CONFIG_ROOT / f"{self.role}.json"
            _write_json_atomic(config_path, config)

            print(
                f"Starting audio.cpp {AUDIO_CPP_VERSION} for {self.role}: "
                f"{family} on {backend}:{device_index}"
            )
            self.process = processmanager.run_process(
                [str(server_path), "--config", str(config_path.resolve()), "--no-ui"],
                include_stdout=False,
                env=_server_environment(server_path, family),
            )
            if self.process is None:
                raise RuntimeError("Could not start audiocpp_server.")
            self.base_url = f"http://127.0.0.1:{port}"
            self.model_id = model_id
            self.configuration_key = configuration_key

            deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self._stop_locked()
                    raise RuntimeError(
                        "audiocpp_server exited during startup. Check the backend choice "
                        "and GPU driver, or select CPU."
                    )
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=1.0)
                    if response.ok:
                        return
                    last_error = RuntimeError(_error_message(response))
                except requests.RequestException as exc:
                    last_error = exc
                time.sleep(0.1)

            self._stop_locked()
            detail = f": {last_error}" if last_error is not None else ""
            raise RuntimeError(f"Timed out waiting for audiocpp_server{detail}")

    def request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        with self._lock:
            if self.process is None or self.process.poll() is not None or not self.base_url:
                raise RuntimeError("audiocpp_server is not running.")
            url = self.base_url + endpoint
        kwargs.setdefault("timeout", (10, REQUEST_TIMEOUT_SECONDS))
        response = requests.request(method, url, **kwargs)
        if not response.ok:
            message = _error_message(response)
            response.close()
            raise RuntimeError(f"audio.cpp request failed ({response.status_code}): {message}")
        return response

    def _stop_locked(self) -> None:
        process = self.process
        self.process = None
        self.base_url = None
        self.model_id = None
        self.configuration_key = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
        try:
            processmanager.all_processes.remove(process)
        except ValueError:
            pass

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
