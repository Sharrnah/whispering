"""Windows WASAPI capture for one process tree.

PyAudio/PortAudio exposes endpoint loopback devices, but it does not expose
``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK``.  This module implements that
small Windows-only gap directly with ``ActivateAudioInterfaceAsync`` and
provides the handful of stream methods used by ``audioWhisper.py``.

The module is safe to import on other platforms.  Constructing a stream there
raises a clear error before any Windows API is accessed.
"""

from __future__ import annotations

import ctypes
import ntpath
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


IS_WINDOWS = sys.platform == "win32"


class ProcessLoopbackError(RuntimeError):
    """Raised when a process-loopback stream cannot be created or operated."""


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent_pid: int
    executable: str


def _normalise_executable(executable: str) -> str:
    return ntpath.basename(str(executable or "").strip()).casefold()


def choose_process_id(
    executable: str,
    preferred_pid: int = 0,
    processes: Optional[Iterable[ProcessInfo]] = None,
) -> int:
    """Resolve a saved executable/PID pair to a currently running process.

    The UI stores the PID selected by the user as a fast, exact hint.  PIDs are
    ephemeral, so a later launch falls back to the root of a process family
    with the same executable name.  Selecting the root is important for
    browsers and Electron applications because WASAPI can include the target's
    child processes.
    """

    if not IS_WINDOWS and processes is None:
        raise ProcessLoopbackError(
            "Per-application WASAPI capture is only available on Windows."
        )

    requested_name = _normalise_executable(executable)
    entries = list(processes if processes is not None else _enumerate_processes())
    by_pid = {entry.pid: entry for entry in entries}

    try:
        preferred_pid = int(preferred_pid or 0)
    except (TypeError, ValueError):
        preferred_pid = 0

    current_pid = os.getpid()
    preferred = by_pid.get(preferred_pid) if preferred_pid != current_pid else None
    if preferred is not None and (
        not requested_name
        or _normalise_executable(preferred.executable) == requested_name
    ):
        # A visible browser/Electron window can occasionally belong to a
        # helper process. Walk through same-executable parents so the selected
        # target includes the application's complete process tree.
        target = preferred
        visited = set()
        while target.pid not in visited:
            visited.add(target.pid)
            parent = by_pid.get(target.parent_pid)
            if (
                parent is None
                or parent.pid == current_pid
                or _normalise_executable(parent.executable)
                != _normalise_executable(target.executable)
            ):
                break
            target = parent
        return target.pid

    if not requested_name:
        raise ProcessLoopbackError(
            "The selected application is no longer running and no executable "
            "name was saved for it."
        )

    matches = [
        entry
        for entry in entries
        if entry.pid != current_pid
        and _normalise_executable(entry.executable) == requested_name
    ]
    if not matches:
        raise ProcessLoopbackError(
            f"Cannot find a running process named {ntpath.basename(executable)!r}. "
            "Start the application or select it again in the profile."
        )

    matching_pids = {entry.pid for entry in matches}
    roots = [entry for entry in matches if entry.parent_pid not in matching_pids]
    if len(roots) > 1:
        raise ProcessLoopbackError(
            f"Multiple independent processes named {ntpath.basename(executable)!r} "
            "are running and the saved PID is stale. Select the intended "
            "application again in the profile."
        )
    candidates = roots or matches
    return min(candidates, key=lambda entry: entry.pid).pid


class _ChunkDispatcher:
    """Turn arbitrary WASAPI packets into fixed PyAudio-style callbacks."""

    def __init__(self, frames_per_buffer: int, channels: int, callback: Callable):
        self.frames_per_buffer = int(frames_per_buffer)
        self.channels = int(channels)
        self.callback = callback
        self.bytes_per_frame = self.channels * 2
        self.bytes_per_chunk = self.frames_per_buffer * self.bytes_per_frame
        self.pending = bytearray()
        self.last_emit_time = time.monotonic()

    def feed(self, data: bytes) -> bool:
        """Dispatch complete chunks; return false if the callback asks to stop."""

        if data:
            self.pending.extend(data)
        while len(self.pending) >= self.bytes_per_chunk:
            chunk = bytes(self.pending[: self.bytes_per_chunk])
            del self.pending[: self.bytes_per_chunk]
            if not self._emit(chunk):
                return False
        return True

    def emit_silence(self) -> bool:
        if self.pending:
            chunk = bytes(self.pending) + bytes(
                self.bytes_per_chunk - len(self.pending)
            )
            self.pending.clear()
            return self._emit(chunk)
        return self._emit(bytes(self.bytes_per_chunk))

    def _emit(self, data: bytes) -> bool:
        now = time.monotonic()
        result = self.callback(
            data,
            self.frames_per_buffer,
            {
                "input_buffer_adc_time": now,
                "current_time": now,
                "output_buffer_dac_time": 0.0,
            },
            0,
        )
        self.last_emit_time = now
        if isinstance(result, tuple) and len(result) > 1:
            # paContinue is zero.  Avoid importing PyAudio in this Windows API
            # bridge merely to compare the callback status.
            return int(result[1]) == 0
        return True


if IS_WINDOWS:
    from ctypes import wintypes

    HRESULT = ctypes.c_long
    REFERENCE_TIME = ctypes.c_longlong
    ULONG = wintypes.ULONG
    UINT32 = wintypes.UINT
    UINT64 = ctypes.c_ulonglong
    BYTE = ctypes.c_ubyte

    S_OK = 0
    E_NOINTERFACE = -2147467262  # 0x80004002
    COINIT_MULTITHREADED = 0
    VT_BLOB = 65

    AUDCLNT_SHAREMODE_SHARED = 0
    AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
    AUDCLNT_BUFFERFLAGS_SILENT = 0x00000002

    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258
    INFINITE = 0xFFFFFFFF
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    TH32CS_SNAPPROCESS = 0x00000002
    WAVE_FORMAT_PCM = 1

    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", BYTE * 8),
        ]

        @classmethod
        def from_string(cls, value: str) -> "GUID":
            import uuid

            raw = uuid.UUID(value).bytes_le
            return cls(
                int.from_bytes(raw[0:4], "little"),
                int.from_bytes(raw[4:6], "little"),
                int.from_bytes(raw[6:8], "little"),
                (BYTE * 8).from_buffer_copy(raw[8:16]),
            )

    IID_IUNKNOWN = GUID.from_string("00000000-0000-0000-C000-000000000046")
    IID_IAUDIOCLIENT = GUID.from_string("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
    IID_IAUDIOCAPTURECLIENT = GUID.from_string(
        "C8ADBD64-E71E-48A0-A4DE-185C395CD317"
    )
    IID_IACTIVATE_COMPLETION_HANDLER = GUID.from_string(
        "41D949AB-9862-444A-80F6-C261334DA5EB"
    )
    IID_IAGILE_OBJECT = GUID.from_string("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")

    class BLOB(ctypes.Structure):
        _fields_ = [("cbSize", ULONG), ("pBlobData", ctypes.POINTER(BYTE))]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [
            ("vt", ctypes.c_ushort),
            ("wReserved1", ctypes.c_ushort),
            ("wReserved2", ctypes.c_ushort),
            ("wReserved3", ctypes.c_ushort),
            ("blob", BLOB),
        ]

    class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
        _fields_ = [("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", ctypes.c_int)]

    class _ACTIVATION_UNION(ctypes.Union):
        _fields_ = [("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS)]

    class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
        _anonymous_ = ("parameters",)
        _fields_ = [("ActivationType", ctypes.c_int), ("parameters", _ACTIVATION_UNION)]

    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [
            ("wFormatTag", wintypes.WORD),
            ("nChannels", wintypes.WORD),
            ("nSamplesPerSec", wintypes.DWORD),
            ("nAvgBytesPerSec", wintypes.DWORD),
            ("nBlockAlign", wintypes.WORD),
            ("wBitsPerSample", wintypes.WORD),
            ("cbSize", wintypes.WORD),
        ]

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ole32 = ctypes.WinDLL("ole32")
    _mmdevapi = ctypes.WinDLL("Mmdevapi")

    _kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    _kernel32.Process32FirstW.restype = wintypes.BOOL
    _kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    _kernel32.Process32NextW.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD

    _ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _ole32.CoInitializeEx.restype = HRESULT
    _ole32.CoUninitialize.argtypes = []
    _ole32.CoUninitialize.restype = None

    _mmdevapi.ActivateAudioInterfaceAsync.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(GUID),
        ctypes.POINTER(PROPVARIANT),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _mmdevapi.ActivateAudioInterfaceAsync.restype = HRESULT

    _QUERY_INTERFACE = ctypes.WINFUNCTYPE(
        HRESULT, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)
    )
    _ADD_REF = ctypes.WINFUNCTYPE(ULONG, ctypes.c_void_p)
    _RELEASE = ctypes.WINFUNCTYPE(ULONG, ctypes.c_void_p)
    _ACTIVATE_COMPLETED = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p)

    class _CompletionHandlerVTable(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", _QUERY_INTERFACE),
            ("AddRef", _ADD_REF),
            ("Release", _RELEASE),
            ("ActivateCompleted", _ACTIVATE_COMPLETED),
        ]

    class _CompletionHandlerStruct(ctypes.Structure):
        pass

    _CompletionHandlerStruct._fields_ = [
        ("lpVtbl", ctypes.POINTER(_CompletionHandlerVTable)),
        ("refCount", ctypes.c_long),
        ("owner", ctypes.py_object),
    ]


    def _guid_equal(left: ctypes.POINTER(GUID), right: GUID) -> bool:
        return ctypes.string_at(left, ctypes.sizeof(GUID)) == bytes(right)


    @_QUERY_INTERFACE
    def _handler_query_interface(this, riid, result):
        if not result:
            return E_NOINTERFACE
        if (
            _guid_equal(riid, IID_IUNKNOWN)
            or _guid_equal(riid, IID_IACTIVATE_COMPLETION_HANDLER)
            or _guid_equal(riid, IID_IAGILE_OBJECT)
        ):
            result[0] = this
            _handler_add_ref(this)
            return S_OK
        result[0] = None
        return E_NOINTERFACE


    @_ADD_REF
    def _handler_add_ref(this):
        handler = ctypes.cast(this, ctypes.POINTER(_CompletionHandlerStruct)).contents
        handler.refCount += 1
        return handler.refCount


    @_RELEASE
    def _handler_release(this):
        handler = ctypes.cast(this, ctypes.POINTER(_CompletionHandlerStruct)).contents
        handler.refCount = max(0, handler.refCount - 1)
        return handler.refCount


    @_ACTIVATE_COMPLETED
    def _handler_activate_completed(this, operation):
        handler = ctypes.cast(this, ctypes.POINTER(_CompletionHandlerStruct)).contents
        owner = handler.owner
        try:
            activation_result = HRESULT()
            audio_client = ctypes.c_void_p()
            get_activate_result = _com_method(
                operation,
                3,
                HRESULT,
                ctypes.POINTER(HRESULT),
                ctypes.POINTER(ctypes.c_void_p),
            )
            _check_hresult(
                get_activate_result(
                    operation,
                    ctypes.byref(activation_result),
                    ctypes.byref(audio_client),
                ),
                "GetActivateResult",
            )
            _check_hresult(activation_result.value, "WASAPI process-loopback activation")
            owner._audio_client = audio_client
        except BaseException as exc:  # callback exceptions cannot cross the COM boundary
            owner._activation_error = exc
        finally:
            owner._activation_complete.set()
        return S_OK


    _COMPLETION_VTABLE = _CompletionHandlerVTable(
        _handler_query_interface,
        _handler_add_ref,
        _handler_release,
        _handler_activate_completed,
    )


    class _CompletionHandler:
        def __init__(self, owner):
            self.struct = _CompletionHandlerStruct(
                ctypes.pointer(_COMPLETION_VTABLE), 1, owner
            )

        @property
        def pointer(self):
            return ctypes.cast(ctypes.pointer(self.struct), ctypes.c_void_p)


    def _com_method(pointer, index: int, result_type, *argument_types):
        vtable = ctypes.cast(
            pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        return ctypes.WINFUNCTYPE(
            result_type, ctypes.c_void_p, *argument_types
        )(vtable[index])


    def _format_hresult(value: int) -> str:
        unsigned = ctypes.c_uint32(int(value)).value
        try:
            message = ctypes.FormatError(unsigned).strip()
        except Exception:
            message = ""
        return f"0x{unsigned:08X}" + (f" ({message})" if message else "")


    def _check_hresult(value: int, operation: str) -> None:
        signed = HRESULT(value).value
        if signed < 0:
            raise ProcessLoopbackError(
                f"{operation} failed with HRESULT {_format_hresult(signed)}."
            )


    def _release_com(pointer) -> None:
        if pointer and getattr(pointer, "value", pointer):
            try:
                release = _com_method(pointer, 2, ULONG)
                release(pointer)
            except Exception:
                pass


    def _enumerate_processes() -> list[ProcessInfo]:
        snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        entries: list[ProcessInfo] = []
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            success = _kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while success:
                entries.append(
                    ProcessInfo(
                        pid=int(entry.th32ProcessID),
                        parent_pid=int(entry.th32ParentProcessID),
                        executable=str(entry.szExeFile),
                    )
                )
                success = _kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            _kernel32.CloseHandle(snapshot)
        return entries


else:

    def _enumerate_processes() -> list[ProcessInfo]:
        raise ProcessLoopbackError(
            "Per-application WASAPI capture is only available on Windows."
        )


class ProcessLoopbackStream:
    """A callback capture stream with the PyAudio methods used by the app."""

    def __init__(
        self,
        process_id: int,
        callback: Callable,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 512,
        include_process_tree: bool = True,
        activation_timeout: float = 10.0,
    ):
        if not IS_WINDOWS:
            raise ProcessLoopbackError(
                "Per-application WASAPI capture is only available on Windows."
            )
        if callback is None:
            raise ValueError("A stream callback is required for process-loopback capture.")
        if int(process_id) <= 0:
            raise ValueError("process_id must be a positive running process ID.")
        if int(channels) != 1:
            raise ValueError("Whispering Tiger process capture currently emits mono audio.")

        self.process_id = int(process_id)
        self.callback = callback
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_buffer = int(frames_per_buffer)
        self.include_process_tree = bool(include_process_tree)
        self.activation_timeout = float(activation_timeout)

        self._dispatcher = _ChunkDispatcher(
            self.frames_per_buffer, self.channels, self.callback
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._startup_complete = threading.Event()
        self._activation_complete = threading.Event()
        self._active = False
        self._error: Optional[BaseException] = None
        self._activation_error: Optional[BaseException] = None
        self._audio_client = None
        self._capture_client = None
        self._async_operation = None
        self._sample_event = None
        self._completion_handler = None

    def start_stream(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._startup_complete.clear()
        self._activation_complete.clear()
        self._error = None
        self._activation_error = None
        self._dispatcher.pending.clear()
        self._dispatcher.last_emit_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._capture_worker,
            name=f"WASAPI process capture ({self.process_id})",
            daemon=True,
        )
        self._thread.start()
        if not self._startup_complete.wait(self.activation_timeout + 2.0):
            self.stop_stream()
            raise ProcessLoopbackError("Timed out while starting process-loopback capture.")
        if self._error is not None:
            raise ProcessLoopbackError(str(self._error)) from self._error

    def stop_stream(self) -> None:
        self._stop_requested.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def close(self) -> None:
        self.stop_stream()

    def is_active(self) -> bool:
        return self._active

    def is_stopped(self) -> bool:
        return not self._active

    def _capture_worker(self) -> None:
        com_initialised = False
        try:
            _check_hresult(_ole32.CoInitializeEx(None, COINIT_MULTITHREADED), "CoInitializeEx")
            com_initialised = True
            self._activate_audio_client()
            self._initialise_capture_client()
            self._active = True
            self._startup_complete.set()
            self._capture_packets()
        except BaseException as exc:
            self._error = exc
            if self._active:
                print(f"WASAPI process-loopback capture stopped: {exc}")
                traceback.print_exc()
        finally:
            self._active = False
            self._startup_complete.set()
            self._cleanup_native_resources()
            if com_initialised:
                _ole32.CoUninitialize()

    def _activate_audio_client(self) -> None:
        activation = AUDIOCLIENT_ACTIVATION_PARAMS()
        activation.ActivationType = 1  # AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
        activation.ProcessLoopbackParams.TargetProcessId = self.process_id
        activation.ProcessLoopbackParams.ProcessLoopbackMode = (
            0 if self.include_process_tree else 1
        )

        blob_bytes = (BYTE * ctypes.sizeof(activation)).from_buffer_copy(activation)
        propvariant = PROPVARIANT()
        propvariant.vt = VT_BLOB
        propvariant.blob.cbSize = ctypes.sizeof(activation)
        propvariant.blob.pBlobData = ctypes.cast(blob_bytes, ctypes.POINTER(BYTE))

        self._completion_handler = _CompletionHandler(self)
        async_operation = ctypes.c_void_p()
        _check_hresult(
            _mmdevapi.ActivateAudioInterfaceAsync(
                VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
                ctypes.byref(IID_IAUDIOCLIENT),
                ctypes.byref(propvariant),
                self._completion_handler.pointer,
                ctypes.byref(async_operation),
            ),
            "ActivateAudioInterfaceAsync",
        )
        self._async_operation = async_operation
        if not self._activation_complete.wait(self.activation_timeout):
            raise ProcessLoopbackError(
                "Timed out waiting for WASAPI process-loopback activation."
            )
        if self._activation_error is not None:
            raise self._activation_error
        if not self._audio_client:
            raise ProcessLoopbackError(
                "WASAPI process-loopback activation returned no audio client."
            )

    def _initialise_capture_client(self) -> None:
        wave_format = WAVEFORMATEX()
        wave_format.wFormatTag = WAVE_FORMAT_PCM
        wave_format.nChannels = self.channels
        wave_format.nSamplesPerSec = self.sample_rate
        wave_format.wBitsPerSample = 16
        wave_format.nBlockAlign = self.channels * 2
        wave_format.nAvgBytesPerSec = self.sample_rate * wave_format.nBlockAlign
        wave_format.cbSize = 0

        initialise = _com_method(
            self._audio_client,
            3,
            HRESULT,
            ctypes.c_int,
            wintypes.DWORD,
            REFERENCE_TIME,
            REFERENCE_TIME,
            ctypes.POINTER(WAVEFORMATEX),
            ctypes.POINTER(GUID),
        )
        stream_flags = (
            AUDCLNT_STREAMFLAGS_LOOPBACK
            | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
            | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
            | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
        )
        _check_hresult(
            initialise(
                self._audio_client,
                AUDCLNT_SHAREMODE_SHARED,
                stream_flags,
                0,
                0,
                ctypes.byref(wave_format),
                None,
            ),
            "IAudioClient.Initialize",
        )

        sample_event = _kernel32.CreateEventW(None, False, False, None)
        if not sample_event:
            raise ctypes.WinError(ctypes.get_last_error())
        self._sample_event = sample_event

        set_event_handle = _com_method(
            self._audio_client, 13, HRESULT, wintypes.HANDLE
        )
        _check_hresult(
            set_event_handle(self._audio_client, sample_event),
            "IAudioClient.SetEventHandle",
        )

        capture_client = ctypes.c_void_p()
        get_service = _com_method(
            self._audio_client,
            14,
            HRESULT,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        _check_hresult(
            get_service(
                self._audio_client,
                ctypes.byref(IID_IAUDIOCAPTURECLIENT),
                ctypes.byref(capture_client),
            ),
            "IAudioClient.GetService(IAudioCaptureClient)",
        )
        self._capture_client = capture_client

        start = _com_method(self._audio_client, 10, HRESULT)
        _check_hresult(start(self._audio_client), "IAudioClient.Start")

    def _capture_packets(self) -> None:
        chunk_duration = self.frames_per_buffer / self.sample_rate
        last_packet_time = time.monotonic()
        while not self._stop_requested.is_set():
            wait_result = _kernel32.WaitForSingleObject(self._sample_event, 20)
            if wait_result == WAIT_OBJECT_0:
                received_packet = self._drain_packets()
                if received_packet:
                    last_packet_time = time.monotonic()
            elif wait_result != WAIT_TIMEOUT:
                raise ProcessLoopbackError(
                    f"WaitForSingleObject failed with result {wait_result}."
                )

            now = time.monotonic()
            # A process with no active render stream may not wake the event.
            # Continue feeding silence so phrase-finalisation/VAD timers behave
            # like the ordinary PyAudio callback stream.
            if (
                now - last_packet_time >= chunk_duration * 2
                and now - self._dispatcher.last_emit_time >= chunk_duration
            ):
                if not self._dispatcher.emit_silence():
                    self._stop_requested.set()

    def _drain_packets(self) -> bool:
        get_next_packet_size = _com_method(
            self._capture_client, 5, HRESULT, ctypes.POINTER(UINT32)
        )
        get_buffer = _com_method(
            self._capture_client,
            3,
            HRESULT,
            ctypes.POINTER(ctypes.POINTER(BYTE)),
            ctypes.POINTER(UINT32),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(UINT64),
            ctypes.POINTER(UINT64),
        )
        release_buffer = _com_method(
            self._capture_client, 4, HRESULT, UINT32
        )

        received = False
        packet_frames = UINT32()
        _check_hresult(
            get_next_packet_size(self._capture_client, ctypes.byref(packet_frames)),
            "IAudioCaptureClient.GetNextPacketSize",
        )
        while packet_frames.value:
            received = True
            data_pointer = ctypes.POINTER(BYTE)()
            frame_count = UINT32()
            flags = wintypes.DWORD()
            _check_hresult(
                get_buffer(
                    self._capture_client,
                    ctypes.byref(data_pointer),
                    ctypes.byref(frame_count),
                    ctypes.byref(flags),
                    None,
                    None,
                ),
                "IAudioCaptureClient.GetBuffer",
            )
            try:
                byte_count = int(frame_count.value) * self.channels * 2
                if flags.value & AUDCLNT_BUFFERFLAGS_SILENT:
                    packet = bytes(byte_count)
                else:
                    packet = ctypes.string_at(data_pointer, byte_count)
            finally:
                _check_hresult(
                    release_buffer(self._capture_client, frame_count),
                    "IAudioCaptureClient.ReleaseBuffer",
                )
            if not self._dispatcher.feed(packet):
                self._stop_requested.set()
                return received
            _check_hresult(
                get_next_packet_size(
                    self._capture_client, ctypes.byref(packet_frames)
                ),
                "IAudioCaptureClient.GetNextPacketSize",
            )
        return received

    def _cleanup_native_resources(self) -> None:
        if self._audio_client:
            try:
                stop = _com_method(self._audio_client, 11, HRESULT)
                stop(self._audio_client)
            except Exception:
                pass
        _release_com(self._capture_client)
        _release_com(self._audio_client)
        _release_com(self._async_operation)
        self._capture_client = None
        self._audio_client = None
        self._async_operation = None
        if self._sample_event:
            _kernel32.CloseHandle(self._sample_event)
            self._sample_event = None
        self._completion_handler = None


class ReconnectingProcessLoopbackStream:
    """Keep process-loopback capture alive across ordinary application restarts.

    WASAPI binds a process-loopback client to one PID.  The saved executable is
    therefore retained as the durable identity and the PID is treated as a
    hint: while that PID exists it remains exact, and after it exits we switch
    only when the executable identifies one unambiguous process family.
    """

    def __init__(
        self,
        executable: str,
        preferred_pid: int,
        callback: Callable,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 512,
        include_process_tree: bool = True,
        activation_timeout: float = 10.0,
        process_check_interval: float = 0.5,
        _process_resolver: Callable = choose_process_id,
        _stream_factory: Callable = ProcessLoopbackStream,
    ):
        if callback is None:
            raise ValueError("A stream callback is required for process-loopback capture.")
        if not _normalise_executable(executable):
            raise ValueError("An executable name is required for process-loopback capture.")
        if int(channels) != 1:
            raise ValueError("Whispering Tiger process capture currently emits mono audio.")

        self.executable = str(executable)
        self.process_id = int(preferred_pid or 0)
        self.callback = callback
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_buffer = int(frames_per_buffer)
        self.include_process_tree = bool(include_process_tree)
        self.activation_timeout = float(activation_timeout)
        self.process_check_interval = max(float(process_check_interval), 0.01)
        self._process_resolver = _process_resolver
        self._stream_factory = _stream_factory

        self._silence_dispatcher = _ChunkDispatcher(
            self.frames_per_buffer, self.channels, self.callback
        )
        self._thread: Optional[threading.Thread] = None
        self._inner_stream = None
        self._inner_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._startup_complete = threading.Event()
        self._active = False
        self._error: Optional[BaseException] = None

    def start_stream(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._startup_complete.clear()
        self._error = None
        self._silence_dispatcher.pending.clear()
        self._silence_dispatcher.last_emit_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._capture_supervisor,
            name=f"WASAPI application capture ({ntpath.basename(self.executable)})",
            daemon=True,
        )
        self._thread.start()
        if not self._startup_complete.wait(self.activation_timeout + 3.0):
            self.stop_stream()
            raise ProcessLoopbackError("Timed out while starting application capture.")
        if self._error is not None:
            raise ProcessLoopbackError(str(self._error)) from self._error

    def stop_stream(self) -> None:
        self._stop_requested.set()
        with self._inner_lock:
            inner_stream = self._inner_stream
        if inner_stream is not None:
            inner_stream.stop_stream()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def close(self) -> None:
        self.stop_stream()

    def is_active(self) -> bool:
        return self._active

    def is_stopped(self) -> bool:
        return not self._active

    def _capture_supervisor(self) -> None:
        startup_succeeded = False
        last_resolution_error = ""
        chunk_duration = self.frames_per_buffer / self.sample_rate
        tick_duration = min(chunk_duration, 0.05)
        next_process_check = 0.0
        next_silence = time.monotonic()
        try:
            resolved_pid = self._process_resolver(self.executable, self.process_id)
            self._connect(resolved_pid)
            startup_succeeded = True
            self._active = True
            self._startup_complete.set()
            next_process_check = time.monotonic() + self.process_check_interval

            while not self._stop_requested.wait(tick_duration):
                inner_stream = self._get_inner_stream()
                if inner_stream is not None and not inner_stream.is_active():
                    # A callback can explicitly ask a PyAudio stream to stop.
                    # Preserve that contract instead of reconnecting forever.
                    if (
                        getattr(inner_stream, "_error", None) is None
                        and getattr(inner_stream, "_stop_requested", None) is not None
                        and inner_stream._stop_requested.is_set()
                    ):
                        self._stop_requested.set()
                        break
                    self._disconnect()
                    inner_stream = None
                    next_process_check = 0.0

                now = time.monotonic()
                if now >= next_process_check:
                    try:
                        resolved_pid = self._process_resolver(
                            self.executable, self.process_id
                        )
                    except Exception as exc:
                        self._disconnect()
                        inner_stream = None
                        message = str(exc)
                        if message != last_resolution_error:
                            print(
                                "WASAPI application capture is waiting for "
                                f"{ntpath.basename(self.executable)}: {message}"
                            )
                            last_resolution_error = message
                    else:
                        last_resolution_error = ""
                        if inner_stream is None or resolved_pid != self.process_id:
                            old_pid = self.process_id
                            self._disconnect()
                            try:
                                self._connect(resolved_pid)
                            except BaseException as exc:
                                print(
                                    "Could not reconnect WASAPI application capture "
                                    f"for {ntpath.basename(self.executable)}: {exc}"
                                )
                                traceback.print_exc()
                            else:
                                print(
                                    "Reconnected WASAPI application capture for "
                                    f"{ntpath.basename(self.executable)} "
                                    f"(PID {old_pid} -> {resolved_pid})."
                                )
                        inner_stream = self._get_inner_stream()
                    next_process_check = now + self.process_check_interval

                if inner_stream is None and now >= next_silence:
                    if not self._silence_dispatcher.emit_silence():
                        self._stop_requested.set()
                        break
                    next_silence = now + chunk_duration
        except BaseException as exc:
            self._error = exc
            if startup_succeeded:
                print(f"WASAPI application capture supervisor stopped: {exc}")
                traceback.print_exc()
        finally:
            self._active = False
            self._startup_complete.set()
            self._disconnect()

    def _connect(self, process_id: int) -> None:
        stream = self._stream_factory(
            process_id=int(process_id),
            callback=self.callback,
            sample_rate=self.sample_rate,
            channels=self.channels,
            frames_per_buffer=self.frames_per_buffer,
            include_process_tree=self.include_process_tree,
            activation_timeout=self.activation_timeout,
        )
        with self._inner_lock:
            self._inner_stream = stream
        try:
            stream.start_stream()
        except BaseException:
            stream.close()
            with self._inner_lock:
                if self._inner_stream is stream:
                    self._inner_stream = None
            raise
        self.process_id = int(process_id)

    def _get_inner_stream(self):
        with self._inner_lock:
            return self._inner_stream

    def _disconnect(self) -> None:
        with self._inner_lock:
            stream = self._inner_stream
            self._inner_stream = None
        if stream is not None:
            stream.close()


def create_process_loopback_stream(
    executable: str,
    preferred_pid: int,
    callback: Callable,
    sample_rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 512,
) -> ReconnectingProcessLoopbackStream:
    """Resolve an application and create (but do not start) its capture stream."""

    process_id = choose_process_id(executable, preferred_pid)
    print(
        "Using WASAPI application capture for "
        f"{ntpath.basename(executable)} (PID {process_id}, including child processes)."
    )
    return ReconnectingProcessLoopbackStream(
        executable=executable,
        preferred_pid=process_id,
        callback=callback,
        sample_rate=sample_rate,
        channels=channels,
        frames_per_buffer=frames_per_buffer,
        include_process_tree=True,
    )


__all__ = [
    "IS_WINDOWS",
    "ProcessInfo",
    "ProcessLoopbackError",
    "ProcessLoopbackStream",
    "ReconnectingProcessLoopbackStream",
    "choose_process_id",
    "create_process_loopback_stream",
]
