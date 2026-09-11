"""Logging: an ASCII-safe stdout handler plus a ring buffer for ``GET /api/health.log_tail``.

SPEC c.29. Two public names: ``RingHandler`` (keeps the last ``capacity`` formatted lines) and
``setup_logging(level)`` (idempotent root-logger setup that returns the ring). The Windows console is
cp1254, so every byte that leaves this process through stdout is forced to ASCII: a ``'?'`` replaces
anything else instead of raising ``UnicodeEncodeError`` in the middle of a tick (SPEC 0.1).
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from typing import Final, TextIO

__all__ = ["RingHandler", "AsciiStream", "setup_logging", "get_ring", "LOG_FORMAT", "DATE_FORMAT",
           "RING_CAPACITY", "ROUTED_LOGGERS"]

#: ``'%(asctime)s %(levelname)s %(name)s: %(message)s'`` (SPEC c.29).
LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT: Final[str] = "%H:%M:%S"
RING_CAPACITY: Final[int] = 200

#: Third-party loggers that must flow through our handlers instead of their own (SPEC c.29).
ROUTED_LOGGERS: Final[tuple[str, ...]] = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi",
                                          "fastapi", "websockets", "websockets.server", "httpx", "httpcore")


class AsciiStream:
    """Write-through wrapper that replaces every non-ASCII character by ``'?'``.

    ``errors='replace'`` semantics without reconfiguring the process stdout (a reconfigure would leak
    into the user's shell and into uvicorn's own output). Unwritable streams are swallowed: logging must
    never raise inside the simulation loop.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    @property
    def stream(self) -> TextIO:
        return self._stream

    def write(self, text: str) -> int:
        try:
            safe = str(text).encode("ascii", "replace").decode("ascii")
            return self._stream.write(safe)
        except Exception:  # noqa: BLE001 - a closed/detached stdout must not break logging
            return 0

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:  # noqa: BLE001
            return False

    def __getattr__(self, name: str):  # pragma: no cover - delegation for logging internals
        return getattr(self._stream, name)


class RingHandler(logging.Handler):
    """Keeps the last ``capacity`` formatted lines in a deque (thread-safe); ``tail(n)`` for the API."""

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        super().__init__()
        self.capacity = max(1, int(capacity))
        self._lines: deque[str] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001 - never raise out of a log call
            try:
                line = f"{record.levelname} {record.name}: {record.getMessage()}"
            except Exception:  # noqa: BLE001
                return
        line = line.encode("ascii", "replace").decode("ascii")
        with self._lock:
            self._lines.append(line)

    def tail(self, n: int = 50) -> list[str]:
        """The last ``n`` lines, oldest first (``n <= 0`` -> empty list)."""
        count = int(n)
        if count <= 0:
            return []
        with self._lock:
            lines = list(self._lines)
        return lines[-count:]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)


_lock = threading.Lock()
_ring: RingHandler | None = None
_stream_handler: logging.StreamHandler | None = None


def setup_logging(level: str = "INFO") -> RingHandler:
    """Configure the root logger once and return the ``RingHandler`` (SPEC c.29).

    First call: one ``StreamHandler`` on an ASCII-forcing wrapper around ``sys.stdout`` plus the ring,
    both with ``LOG_FORMAT``; ``ROUTED_LOGGERS`` lose their own handlers and propagate into ours.
    Later calls only update the level (and re-attach the handlers if something cleared them), so
    ``create_app`` may call it on every start without duplicating lines. ``level`` accepts a name or a
    numeric string; an unknown value falls back to ``INFO`` with a warning.
    """
    global _ring, _stream_handler
    num = _level_number(level)
    with _lock:
        root = logging.getLogger()
        if _ring is None:
            _ring = RingHandler(RING_CAPACITY)
            _ring.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        if _stream_handler is None:
            _stream_handler = logging.StreamHandler(AsciiStream(sys.stdout))  # type: ignore[arg-type]
            _stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        for handler in (_stream_handler, _ring):
            handler.setLevel(logging.NOTSET)
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(num)
        for name in ROUTED_LOGGERS:
            lg = logging.getLogger(name)
            lg.handlers = []
            lg.propagate = True
            lg.setLevel(num)
        logging.captureWarnings(True)
        return _ring


def get_ring() -> RingHandler | None:
    """The ring created by ``setup_logging``, or ``None`` when logging was never set up."""
    return _ring


def _level_number(level: str | int) -> int:
    if isinstance(level, int):
        return level
    text = str(level or "INFO").strip().upper()
    if text.isdigit():
        return int(text)
    value = logging.getLevelNamesMapping().get(text) if hasattr(logging, "getLevelNamesMapping") else None
    if value is None:
        value = getattr(logging, text, None) if text.isalpha() else None
    if not isinstance(value, int):
        logging.getLogger("flybrain.log").warning("setup_logging: unknown level %r; using INFO", level)
        return logging.INFO
    return value
