"""Process-wide coordination for CUDA model runtimes.

Calls on one explicitly identified model runtime may overlap. Different model
runtimes hand CUDA over synchronously, preventing incompatible third-party
kernels from running at the same time while retaining same-model STT parallelism.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import functools
import inspect
import sys
import threading
import time
from typing import Callable, Iterable

from Models.ai_device import cuda_device_context


_CUDA_INFERENCE_CONDITION = threading.Condition()
_CUDA_INFERENCE_THREAD_STATE = threading.local()

_waiting_runtimes = deque()
_active_runtime_key = None
_active_label: str | None = None
_active_since = 0.0
_active_count = 0
_handoff_in_progress = False

_WAIT_WARNING_SECONDS = 5.0
_WAIT_REPEAT_SECONDS = 30.0
_GUARDED_METHOD_MARKER = "__whispering_tiger_cuda_guarded__"


def _resolve(value):
    return value() if callable(value) else value


def _cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def is_cuda_device(device) -> bool:
    """Return whether a configured device resolves to CUDA."""
    if hasattr(device, "type"):
        return str(device.type).lower() == "cuda"

    device_name = str(device or "").strip().lower()
    if device_name.startswith("cuda"):
        return True
    if device_name in {"", "none", "auto"}:
        return _cuda_is_available()
    return False


def _synchronize_cuda(device) -> None:
    """Finish queued kernels before another model runtime takes over CUDA."""
    import torch

    if not torch.cuda.is_available():
        return

    device_name = str(device or "").strip().lower()
    synchronize_device = None if device_name in {"", "none", "auto", "cuda"} else device
    torch.cuda.synchronize(synchronize_device)


def _normalize_runtime_key(runtime_key):
    resolved_key = _resolve(runtime_key)
    if resolved_key is None:
        # Calls without an explicit runtime identity remain mutually exclusive.
        return object()
    try:
        hash(resolved_key)
    except TypeError:
        return ("object", id(resolved_key))
    return resolved_key


def _has_earlier_different_runtime(entry) -> bool:
    for queued_entry in _waiting_runtimes:
        if queued_entry is entry:
            return False
        if queued_entry[1] != entry[1]:
            return True
    return False


def _wait_for_gate(label: str, runtime_key) -> None:
    global _active_runtime_key, _active_label, _active_since, _active_count

    depth = getattr(_CUDA_INFERENCE_THREAD_STATE, "depth", 0)
    if depth:
        # Public methods on one adapter commonly call other guarded methods on
        # that adapter. Keep those calls reentrant rather than queueing behind
        # their own outer inference.
        _CUDA_INFERENCE_THREAD_STATE.depth = depth + 1
        return

    entry = (object(), runtime_key, label)
    wait_started = time.monotonic()
    next_warning = wait_started + _WAIT_WARNING_SECONDS
    with _CUDA_INFERENCE_CONDITION:
        _waiting_runtimes.append(entry)
        try:
            while True:
                can_claim_idle_gate = (
                    _active_runtime_key is None
                    and not _handoff_in_progress
                    and _waiting_runtimes[0] is entry
                )
                can_join_same_runtime = (
                    _active_runtime_key == runtime_key
                    and not _handoff_in_progress
                    and not _has_earlier_different_runtime(entry)
                )
                if can_claim_idle_gate or can_join_same_runtime:
                    _waiting_runtimes.remove(entry)
                    if _active_runtime_key is None:
                        _active_runtime_key = runtime_key
                        _active_label = label
                        _active_since = time.monotonic()
                    _active_count += 1
                    _CUDA_INFERENCE_THREAD_STATE.depth = 1
                    _CUDA_INFERENCE_THREAD_STATE.runtime_key = runtime_key
                    _CUDA_INFERENCE_CONDITION.notify_all()
                    return

                now = time.monotonic()
                wait_timeout = max(0.0, next_warning - now)
                _CUDA_INFERENCE_CONDITION.wait(timeout=wait_timeout)
                now = time.monotonic()
                if now >= next_warning:
                    active_label = _active_label or "another CUDA runtime"
                    active_for = max(0.0, now - _active_since) if _active_since else 0.0
                    print(
                        f"CUDA inference queued: {label} has waited {now - wait_started:.1f}s "
                        f"for {active_label} (active for {active_for:.1f}s)."
                    )
                    next_warning = now + _WAIT_REPEAT_SECONDS
        except BaseException:
            try:
                _waiting_runtimes.remove(entry)
            except ValueError:
                pass
            _CUDA_INFERENCE_CONDITION.notify_all()
            raise


def _leave_gate() -> bool:
    """Return whether this caller must synchronize the runtime hand-off."""
    global _active_count, _handoff_in_progress

    depth = getattr(_CUDA_INFERENCE_THREAD_STATE, "depth", 0)
    if depth <= 0:
        raise RuntimeError("CUDA inference gate released without being acquired.")
    depth -= 1
    _CUDA_INFERENCE_THREAD_STATE.depth = depth
    if depth:
        return False

    if hasattr(_CUDA_INFERENCE_THREAD_STATE, "runtime_key"):
        del _CUDA_INFERENCE_THREAD_STATE.runtime_key
    with _CUDA_INFERENCE_CONDITION:
        _active_count -= 1
        if _active_count < 0:
            raise RuntimeError("CUDA inference gate active count became negative.")
        if _active_count:
            return False
        _handoff_in_progress = True
        return True


def _complete_handoff() -> None:
    global _active_runtime_key, _active_label, _active_since, _handoff_in_progress

    with _CUDA_INFERENCE_CONDITION:
        _active_runtime_key = None
        _active_label = None
        _active_since = 0.0
        _handoff_in_progress = False
        _CUDA_INFERENCE_CONDITION.notify_all()


@contextmanager
def cuda_inference_guard(device, label: str = "CUDA inference", runtime_key=None):
    """Enter one CUDA runtime group and synchronize its outermost hand-off.

    Calls with the same explicit ``runtime_key`` may overlap. Calls without one
    are exclusive. The gate is reentrant so a public adapter method may call
    another guarded method without deadlocking. CPU and DirectML pass through.
    """
    resolved_device = _resolve(device)
    resolved_label = str(_resolve(label) or "CUDA inference")
    if not is_cuda_device(resolved_device):
        yield
        return

    resolved_runtime_key = _normalize_runtime_key(runtime_key)
    _wait_for_gate(resolved_label, resolved_runtime_key)
    try:
        # A few third-party runtimes still allocate on the unqualified
        # ``cuda`` device internally.  Keep those allocations on the same GPU
        # as the explicitly selected model device.
        with cuda_device_context(resolved_device):
            yield
    finally:
        handoff_required = _leave_gate()
        if handoff_required:
            try:
                exception_in_flight = sys.exc_info()[0] is not None
                try:
                    _synchronize_cuda(resolved_device)
                except Exception as synchronization_error:
                    if exception_in_flight:
                        print(
                            "CUDA synchronization also failed while handling another "
                            f"error in {resolved_label}: {synchronization_error}"
                        )
                    else:
                        raise
            finally:
                _complete_handoff()


def cuda_inference_guarded(device, label="CUDA inference", runtime_key=None):
    """Decorate a function with :func:`cuda_inference_guard`."""

    def decorate(function):
        if inspect.isgeneratorfunction(function):
            @functools.wraps(function)
            def guarded_generator(*args, **kwargs):
                with cuda_inference_guard(device, label, runtime_key):
                    yield from function(*args, **kwargs)

            return guarded_generator

        @functools.wraps(function)
        def guarded(*args, **kwargs):
            with cuda_inference_guard(device, label, runtime_key):
                return function(*args, **kwargs)

        return guarded

    return decorate


def _model_device(instance, fallback):
    for attribute_name in ("compute_device_str", "compute_device", "device"):
        try:
            candidate = getattr(instance, attribute_name)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(candidate, (str, bytes)) or hasattr(candidate, "type"):
            return candidate
    return fallback


def guard_cuda_model_methods(
    instance,
    method_names: Iterable[str],
    device,
    runtime_label,
    parallel_same_runtime: bool = True,
):
    """Install reentrant CUDA guards on an adapter instance's public methods.

    Inference methods normally share an adapter identity and may run in
    parallel. Lifecycle methods can pass ``parallel_same_runtime=False`` to
    remain exclusive even when they belong to that same adapter.
    """
    if instance is None:
        return None

    for method_name in method_names:
        method = getattr(instance, method_name, None)
        if not callable(method) or getattr(method, _GUARDED_METHOD_MARKER, False):
            continue

        method_label = lambda name=method_name: f"{_resolve(runtime_label)}.{name}"
        guarded_method = cuda_inference_guarded(
            device,
            method_label,
            runtime_key=("adapter", id(instance)) if parallel_same_runtime else None,
        )(method)
        setattr(guarded_method, _GUARDED_METHOD_MARKER, True)
        try:
            setattr(instance, method_name, guarded_method)
        except (AttributeError, TypeError) as error:
            raise RuntimeError(
                f"Cannot install the CUDA inference guard on "
                f"{type(instance).__name__}.{method_name}."
            ) from error

    return instance


def guard_cuda_model_loader(runtime_label: str, method_names=("transcribe",)):
    """Guard a ``(model, device)`` loader and the returned adapter methods."""

    def decorate(loader: Callable):
        @functools.wraps(loader)
        def guarded_loader(model, device, *args, **kwargs):
            load_label = f"{runtime_label}/{model}.load"
            with cuda_inference_guard(device, load_label):
                instance = loader(model, device, *args, **kwargs)

            return guard_cuda_model_methods(
                instance,
                method_names,
                device=lambda: _model_device(instance, device),
                runtime_label=lambda: f"{runtime_label}/{type(instance).__name__}",
            )

        return guarded_loader

    return decorate
