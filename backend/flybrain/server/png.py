"""Pure-stdlib PNG writer: ``write_png(rgb)`` -> PNG bytes (``zlib`` + ``struct``, no PIL).

SPEC c.25. 8-bit truecolour (colour type 2), filter type 0 on every scanline, ``zlib`` level 6, three
chunks (IHDR, IDAT, IEND). Used by the server-side snapshot fallback (``agent/snapshot.py``) and by
``scripts/selftest.py``; the browser path never comes through here.

The output is deterministic for a given array (same bytes on every run), which is what
``tests/test_agent.py::test_render_snapshot_png`` and the snapshot tests rely on.
"""

from __future__ import annotations

import struct
import zlib
from typing import Final

import numpy as np

__all__ = ["write_png", "PNG_MAGIC", "ZLIB_LEVEL"]

PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"
ZLIB_LEVEL: Final[int] = 6
_MAX_DIM: Final[int] = 1 << 16  #: [E] sanity bound (a canvas is 800x500)


def _chunk(tag: bytes, data: bytes) -> bytes:
    """``length | tag | data | crc32(tag + data)`` big-endian (PNG spec)."""
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(rgb: np.ndarray) -> bytes:
    """``uint8[H, W, 3]`` -> PNG bytes (filter 0, zlib level 6, struct chunks; no PIL).

    Accepts anything numpy can view as ``uint8`` with shape ``(H, W, 3)``; ``(H, W)`` and ``(H, W, 1)``
    are broadcast to grey RGB and ``(H, W, 4)`` drops the alpha channel, so a caller never has to
    reshape. Raises ``ValueError`` for any other shape or an empty image.
    """
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"write_png expects uint8[H,W,3], got shape {tuple(np.asarray(rgb).shape)}")
    h, w, _ = (int(arr.shape[0]), int(arr.shape[1]), 3)
    if h <= 0 or w <= 0:
        raise ValueError(f"write_png: empty image ({h}x{w})")
    if h > _MAX_DIM or w > _MAX_DIM:
        raise ValueError(f"write_png: image too large ({w}x{h})")
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    # one filter byte (0 = None) in front of every scanline, then a single IDAT
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), arr.reshape(h, w * 3)], axis=1)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # bit depth 8, colour type 2, no interlace
    return (PNG_MAGIC
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(rows.tobytes(), ZLIB_LEVEL))
            + _chunk(b"IEND", b""))
