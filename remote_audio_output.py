"""Request-scoped audio destination, independent of devices and OS APIs."""
from contextlib import contextmanager
from contextvars import ContextVar

_destination = ContextVar("remote_audio_destination", default=None)


def current_destination():
    return _destination.get()


@contextmanager
def destination(sink):
    token = _destination.set(sink)
    try:
        yield
    finally:
        _destination.reset(token)
