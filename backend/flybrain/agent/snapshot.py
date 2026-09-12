"""Snapshot images for tweets (SPEC section c.23).

* ``render_snapshot`` - a numpy RGB raster of the Paint window: white canvas inside a 3 px #c0c0c0 frame, a 20 px
  #000080 title bar and a 20 px status strip with a 5x7 bitmap-font caption ``FLYBRAIN | <mood> | <symbol> <price>
  <chg_m5>%``; the trail as Bresenham segments (x, y, '#rrggbb', width) and a 9x9 fly glyph rotated to 8 headings,
  scaled 2x. Encoded with ``flybrain.server.png.write_png`` (no PIL). Everything here is [E] presentation.
* ``SnapshotBroker`` - asks the most recently active browser for its composited canvas over the bus
  (``snapshot_request`` / ``snapshot`` frames, SPEC d.9) and falls back to ``render_snapshot`` on timeout or when no
  client is connected. Keeps the server-side trail ring (last 4000 points, written by the sim thread) and saves every
  snapshot under ``data/snapshots/<t_ms>_<mood>.png``.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flybrain.config import Settings
    from flybrain.decoder import FlyBody, Kinematics
    from flybrain.server.state import StateBus

__all__ = ["render_snapshot", "SnapshotBroker", "PNG_MAGIC", "TRAIL_RING", "FRAME_PX", "TITLE_PX", "STATUS_PX"]

log = logging.getLogger("flybrain.agent.snapshot")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TRAIL_RING = 4000
TRAIL_SAVE_INTERVAL_S = 10.0  # throttle for the periodic trail flush written by the sim thread
FRAME_PX = 3
TITLE_PX = 20
STATUS_PX = 20
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024

_FRAME = (0xC0, 0xC0, 0xC0)
_TITLE = (0x00, 0x00, 0x80)
_WHITE = (0xFF, 0xFF, 0xFF)
_BLACK = (0x00, 0x00, 0x00)

# 5x7 bitmap font (columns are 5 bits, rows top->bottom); classic CGA-style glyphs, uppercase only.
_FONT: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "01100", "00100", "01000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "$": ("00100", "01111", "10100", "01110", "00101", "11110", "00100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
}

# 9x9 fly glyph facing +x (right): head at the right, two wings up/down, three leg pairs. [E]
_FLY_GLYPH = (
    "000010100",
    "001111010",
    "010111100",
    "111111111",
    "011111110",
    "111111111",
    "010111100",
    "001111010",
    "000010100",
)


def _glyph_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[c == "1" for c in r] for r in rows], dtype=bool)


_FLY0 = _glyph_array(_FLY_GLYPH)


def _rotate45(g: np.ndarray) -> np.ndarray:
    """Nearest-neighbour rotation of a square boolean glyph by -45 deg (clockwise on screen, y down)."""
    n = g.shape[0]
    c = (n - 1) / 2.0
    out = np.zeros_like(g)
    ang = math.pi / 4.0
    ca, sa = math.cos(ang), math.sin(ang)
    for y in range(n):
        for x in range(n):
            # inverse map: destination (x,y) -> source
            dx, dy = x - c, y - c
            sx = ca * dx + sa * dy + c
            sy = -sa * dx + ca * dy + c
            ix, iy = int(round(sx)), int(round(sy))
            if 0 <= ix < n and 0 <= iy < n and g[iy, ix]:
                out[y, x] = True
    return out


def _fly_sprites() -> list[np.ndarray]:
    """8 headings, k*45 deg clockwise on screen (heading 0 = +x). Index = round(heading / 45deg) mod 8."""
    base0 = _FLY0
    base45 = _rotate45(_FLY0)
    sprites = []
    for k in range(8):
        arr = base0 if k % 2 == 0 else base45
        for _ in range(k // 2):
            arr = np.rot90(arr, k=-1)  # k=-1: clockwise in array coordinates == clockwise on a y-down screen
        sprites.append(arr)
    return sprites


_SPRITES = _fly_sprites()


def _hex_rgb(color: Any, default: tuple[int, int, int] = _BLACK) -> tuple[int, int, int]:
    if not isinstance(color, str):
        return default
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return default
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return default


def _draw_text(img: np.ndarray, x0: int, y0: int, text: str, rgb: tuple[int, int, int], scale: int = 2) -> None:
    """Draw ``text`` with the 5x7 font at (x0, y0), ``scale``x pixels, 1 glyph-px spacing; clipped to the image."""
    h, w = img.shape[:2]
    x = x0
    for ch in text.upper():
        rows = _FONT.get(ch, _FONT["?"])
        for r, bits in enumerate(rows):
            for cidx, bit in enumerate(bits):
                if bit == "1":
                    ya, xa = y0 + r * scale, x + cidx * scale
                    yb, xb = min(h, ya + scale), min(w, xa + scale)
                    if ya < h and xa < w and ya >= 0 and xa >= 0:
                        img[ya:yb, xa:xb] = rgb
        x += 6 * scale
        if x >= w:
            break


def _stamp(img: np.ndarray, x: int, y: int, half: int, rgb: tuple[int, int, int], x_min: int, y_min: int,
           x_max: int, y_max: int) -> None:
    ya, yb = max(y_min, y - half), min(y_max, y + half + 1)
    xa, xb = max(x_min, x - half), min(x_max, x + half + 1)
    if ya < yb and xa < xb:
        img[ya:yb, xa:xb] = rgb


def _bresenham(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, width: float, rgb: tuple[int, int, int],
               clip: tuple[int, int, int, int]) -> None:
    """Integer Bresenham segment; ``width`` px is realised as a square stamp of radius width//2 per pixel."""
    half = max(0, int(round(max(1.0, float(width)) / 2.0 - 0.5)))
    x_min, y_min, x_max, y_max = clip
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    steps = 0
    limit = dx - dy + 2
    while steps <= limit:
        if half == 0:
            if x_min <= x < x_max and y_min <= y < y_max:
                img[y, x] = rgb
        else:
            _stamp(img, x, y, half, rgb, x_min, y_min, x_max, y_max)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        steps += 1


def _num_or(v: Any, default: float) -> float:
    """Finite ``float(v)`` or ``default``."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _ipx(v: Any, default: int | None = None) -> int | None:
    """``int(round(float(v)))`` for a finite value, else ``default`` (``int()`` raises on NaN/inf: a single poisoned
    trail point must never blank every server-side snapshot)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    f = max(-1e7, min(1e7, f))
    return int(round(f))


def _fmt_price(p: Any) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v) or v <= 0:
        return "N/A"
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:.2f}"
    if v >= 0.01:
        return f"${v:.4f}"
    return f"${v:.7f}".rstrip("0")


def render_snapshot(trail: list[tuple[float, float, str, float]], kin: "Kinematics", mood: str, market: dict,
                    w: int = 800, h: int = 500) -> bytes:
    """numpy RGB raster: white canvas with a 3 px #c0c0c0 Paint frame, a 20 px title bar '#000080' and a 20 px status
    strip with a 5x7 bitmap-font caption 'FLYBRAIN | <mood> | <symbol> <price> <chg_m5>%'; Bresenham segments for the
    trail (x, y, '#rrggbb', width); fly sprite = 9x9 glyph rotated to 8 headings, scaled 2x; encoded with server.png.write_png."""
    from flybrain.server.png import write_png  # E6 module (a stub exists until it lands)

    w, h = max(16, int(w)), max(16, int(h))
    W = w + 2 * FRAME_PX
    H = h + 2 * FRAME_PX + TITLE_PX + STATUS_PX
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:, :] = _FRAME
    # title bar
    ty = FRAME_PX
    img[ty:ty + TITLE_PX, FRAME_PX:W - FRAME_PX] = _TITLE
    _draw_text(img, FRAME_PX + 6, ty + 3, "UNTITLED - PAINT", _WHITE, scale=2)
    # canvas
    cy0 = ty + TITLE_PX
    cy1 = cy0 + h
    cx0, cx1 = FRAME_PX, FRAME_PX + w
    img[cy0:cy1, cx0:cx1] = _WHITE
    clip = (cx0, cy0, cx1, cy1)

    # trail segments
    prev: tuple[int, int] | None = None
    for pt in trail or ():
        try:
            x, y, color, width = pt
        except (TypeError, ValueError):
            prev = None
            continue
        xv, yv = _ipx(x), _ipx(y)
        wv = _ipx(width, 1)
        if xv is None or yv is None:       # a non-finite point breaks the line instead of poisoning the raster
            prev = None
            continue
        xi, yi = xv + cx0, yv + cy0
        width = float(max(1, wv if wv is not None else 1))
        if prev is not None:
            if abs(xi - prev[0]) + abs(yi - prev[1]) <= 120:  # a jump/wrap breaks the line
                _bresenham(img, prev[0], prev[1], xi, yi, width, _hex_rgb(color), clip)
        else:
            _bresenham(img, xi, yi, xi, yi, width, _hex_rgb(color), clip)
        prev = (xi, yi)

    # fly sprite (2x) - non-finite kinematics fall back to the canvas centre / heading 0 (never raise)
    fxi = _ipx(getattr(kin, "x", None), w // 2)
    fyi = _ipx(getattr(kin, "y", None), h // 2)
    k8 = _ipx(_num_or(getattr(kin, "heading", 0.0), 0.0) / (math.pi / 4.0), 0)
    fxi = w // 2 if fxi is None else fxi
    fyi = h // 2 if fyi is None else fyi
    k = (k8 if k8 is not None else 0) % 8
    sprite = np.repeat(np.repeat(_SPRITES[k], 2, axis=0), 2, axis=1)
    sh, sw = sprite.shape
    ox, oy = fxi + cx0 - sw // 2, fyi + cy0 - sh // 2
    ys, xs = np.nonzero(sprite)
    for dy_, dx_ in zip(ys.tolist(), xs.tolist()):
        px, py = ox + dx_, oy + dy_
        if cx0 <= px < cx1 and cy0 <= py < cy1:
            img[py, px] = _BLACK

    # status strip
    sy0 = cy1
    img[sy0:sy0 + STATUS_PX, FRAME_PX:W - FRAME_PX] = _FRAME
    mkt = market if isinstance(market, dict) else {}
    symbol = str(mkt.get("symbol") or "FLY")[:12]
    chg = _num_or(mkt.get("chg_m5") or 0.0, 0.0)
    caption = (f"FLYBRAIN | {str(mood or 'CRUISING')[:24]} | {symbol} "
               f"{_fmt_price(mkt.get('price_usd'))} {chg:+.1f}%")
    _draw_text(img, FRAME_PX + 6, sy0 + 3, caption, _BLACK, scale=2)
    return write_png(img)


class SnapshotBroker:
    """Browser PNG over the bus with a server-side numpy fallback; owns the trail ring (SPEC c.23)."""

    def __init__(self, bus: "StateBus", body: "FlyBody", settings: "Settings") -> None:
        self.bus = bus
        self.body = body
        self.settings = settings
        self._trail: deque[tuple[float, float, str, float]] = deque(maxlen=TRAIL_RING)
        self._trail_lock = threading.Lock()
        self._trail_px = 0.0
        self._last: tuple[float, float] | None = None
        self._seq = 0
        self.last_source: str | None = None
        self.last_path: Path | None = None
        w = int(getattr(settings, "canvas_w", 800) or 800)
        h = int(getattr(settings, "canvas_h", 500) or 500)
        self.canvas = (w, h)
        # The trail is the shared painting: everyone who loads the page backfills it from GET /api/state, so it must
        # survive a backend restart (a redeploy would otherwise blank the canvas for every viewer). Persisted to
        # data/trail.json - the same volume that already holds snapshots/tweets - flushed on a throttle by the sim
        # thread and once more on shutdown. data_dir is on a Docker named volume in production (deploy/).
        self._trail_path = Path(getattr(settings, "data_dir", "data")) / "trail.json"
        self._last_save = 0.0
        self._load_trail()

    # -- trail (sim thread)
    def record_trail(self, x: float, y: float, color: str, width: float) -> None:
        """Ring of the last 4000 points (sim thread). Non-finite coordinates are dropped: one NaN/inf point would
        otherwise sit in the ring for 4000 points and blank every server-side snapshot taken meanwhile."""
        try:
            xf, yf, wf = float(x), float(y), float(width)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(xf) and math.isfinite(yf) and math.isfinite(wf)):
            log.debug("agent.snapshot: dropping non-finite trail point (%r, %r, %r)", x, y, width)
            return
        with self._trail_lock:
            if self._last is not None:
                d = math.hypot(xf - self._last[0], yf - self._last[1])
                if d < 120.0:
                    self._trail_px += d
            self._last = (xf, yf)
            self._trail.append((xf, yf, str(color or "#000000"), wf))

    def clear_trail(self) -> None:
        with self._trail_lock:
            self._trail.clear()
            self._last = None
            self._trail_px = 0.0
        # Persist immediately so a restart right after a clear cannot resurrect the old painting.
        self.save_trail(force=True)

    # -- persistence
    def _load_trail(self) -> None:
        """Restore the trail ring from data/trail.json on boot (best effort; a missing or corrupt file is ignored)."""
        try:
            raw = self._trail_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("agent.snapshot: could not read trail (%s)", exc)
            return
        try:
            doc = json.loads(raw)
            pts = doc.get("trail") if isinstance(doc, dict) else None
            if not isinstance(pts, list):
                return
            restored: list[tuple[float, float, str, float]] = []
            for pt in pts[-TRAIL_RING:]:
                x, y, color, width = float(pt[0]), float(pt[1]), str(pt[2]), float(pt[3])
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(width):
                    restored.append((x, y, color, width))
        except (ValueError, TypeError, IndexError) as exc:
            log.warning("agent.snapshot: ignoring corrupt trail file (%s)", exc)
            return
        if not restored:
            return
        with self._trail_lock:
            self._trail.clear()
            self._trail.extend(restored)
            self._last = (restored[-1][0], restored[-1][1])
            self._trail_px = float(doc.get("trail_px", 0.0) or 0.0) if isinstance(doc, dict) else 0.0
        log.info("agent.snapshot: restored %d trail point(s) from %s", len(restored), self._trail_path)

    def save_trail(self, force: bool = False) -> None:
        """Atomically write the trail ring to disk, throttled to one write per TRAIL_SAVE_INTERVAL_S unless forced.
        Safe to call every tick from the sim thread; a write failure never propagates (SPEC 0.1)."""
        now = time.monotonic()
        if not force and (now - self._last_save) < TRAIL_SAVE_INTERVAL_S:
            return
        self._last_save = now
        with self._trail_lock:
            pts = list(self._trail)
            px = self._trail_px
        doc = {"trail": [[round(x, 1), round(y, 1), c, round(w, 1)] for x, y, c, w in pts],
               "trail_px": px}  # kept full-precision (it is a cumulative meander distance, not a drawn coordinate)
        try:
            self._trail_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._trail_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc), encoding="utf-8")
            os.replace(tmp, self._trail_path)
        except OSError as exc:
            log.warning("agent.snapshot: could not save trail (%s)", exc)

    def trail(self) -> list[tuple[float, float, str, float]]:
        with self._trail_lock:
            return list(self._trail)

    def trail_wire(self) -> list[list[Any]]:
        """``[[x, y, color, width], ...]`` with positions at 1 dp for ``GET /api/state`` (SPEC c.28 / d rounding)."""
        with self._trail_lock:
            pts = list(self._trail)
        return [[round(x, 1), round(y, 1), c, round(w, 1)] for x, y, c, w in pts]

    @property
    def trail_px(self) -> float:
        with self._trail_lock:
            return self._trail_px

    # -- snapshots
    def _context(self) -> tuple[str, dict, int]:
        """(mood, market, t_ms) from the latest tick on the bus (never raises)."""
        tick: dict | None = None
        try:
            tick = self.bus.latest_tick() if self.bus is not None else None
        except Exception:
            tick = None
        tick = tick if isinstance(tick, dict) else {}
        mood = tick.get("mood") if isinstance(tick.get("mood"), dict) else {}
        market = tick.get("market") if isinstance(tick.get("market"), dict) else {}
        try:
            t_ms = int(tick.get("t_ms") or 0)
        except (TypeError, ValueError):
            t_ms = 0
        return str(mood.get("state") or "CRUISING"), market, t_ms

    def render(self, mood: str | None = None, market: dict | None = None) -> bytes:
        """Server-side raster of the current trail and fly pose."""
        m, mk, _ = self._context()
        kin = getattr(self.body, "kin", None) if self.body is not None else None
        if kin is None:
            from types import SimpleNamespace

            kin = SimpleNamespace(x=self.canvas[0] / 2, y=self.canvas[1] / 2, heading=0.0, mode="walk")
        return render_snapshot(self.trail(), kin, mood or m, market if market is not None else mk,
                               self.canvas[0], self.canvas[1])

    def request(self, timeout_s: float = 3.0) -> tuple[bytes, str]:
        """Ask the most recently active browser over the bus ({'type':'snapshot_request','id'}) and wait for
        {'type':'snapshot','id','png_b64'} (max 2 MB); on timeout/no client render_snapshot(). Returns (png, 'browser'|'server').
        Saves to data/snapshots/<t_ms>_<mood>.png."""
        self._seq += 1
        sid = f"snap-{self._seq}-{int(time.time() * 1000) % 100000}"
        png: bytes | None = None
        source = "server"
        try:
            n_clients = int(self.bus.client_count()) if self.bus is not None else 0
        except Exception:
            n_clients = 0
        if n_clients > 0:
            try:
                self.bus.request_snapshot(sid)
                got = self.bus.wait_snapshot(sid, float(timeout_s))
                if got and got[:8] == PNG_MAGIC and len(got) <= MAX_SNAPSHOT_BYTES:
                    png, source = bytes(got), "browser"
                elif got:
                    log.warning("agent.snapshot: browser snapshot rejected (%d bytes, magic %r)", len(got), got[:4])
                else:
                    log.info("agent.snapshot: no browser snapshot within %.1fs; rendering server-side", timeout_s)
            except Exception as exc:  # noqa: BLE001
                log.warning("agent.snapshot: bus round trip failed (%s); rendering server-side", exc)
        if png is None:
            try:
                png = self.render()
            except Exception as exc:  # noqa: BLE001 - last resort: a blank canvas
                log.error("agent.snapshot: render failed (%s); using blank canvas", exc)
                from flybrain.server.png import write_png

                png = write_png(np.full((self.canvas[1], self.canvas[0], 3), 255, dtype=np.uint8))
            source = "server"
        self.last_source = source
        self._save(png)
        return png, source

    def _save(self, png: bytes) -> None:
        try:
            mood, _, t_ms = self._context()
            data_dir = Path(getattr(self.settings, "data_dir", "data"))
            d = data_dir / "snapshots"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{t_ms}_{mood}.png"
            path.write_bytes(png)
            self.last_path = path
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.snapshot: could not save snapshot (%s)", exc)

    def deliver(self, id: str, png_b64: str) -> None:
        """Called by the WS handler (asyncio thread), thread-safe: decode, check size and magic, hand to the bus."""
        try:
            raw = base64.b64decode(png_b64, validate=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.snapshot: bad base64 for %s (%s)", id, exc)
            return
        if len(raw) > MAX_SNAPSHOT_BYTES or raw[:8] != PNG_MAGIC:
            log.warning("agent.snapshot: rejected snapshot %s (%d bytes)", id, len(raw))
            return
        try:
            self.bus.deliver_snapshot(id, raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.snapshot: deliver failed for %s (%s)", id, exc)
