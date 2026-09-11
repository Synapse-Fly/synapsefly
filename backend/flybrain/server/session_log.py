"""Session log: one JSONL line per tick, enough to replay a run bit-exactly.

SPEC c.25 / section 0 "Replay". ``data/sessions/<run_id>.jsonl`` starts with a ``header`` line (the
redacted settings, ``run_id``, ``connectome_key``, creation time) and then carries one ``tick`` line per
wall tick with every **input** the simulation consumed: the market snapshot, the trades, the pokes, the
commands -- plus ``total_spikes`` as the replay checksum. ``scripts/replay.py`` feeds these lines back
into ``SimulationLoop(replay=read_session(path))`` and asserts identical spike counts.

Line shapes (snake_case, JSON, one object per line):

``{"kind":"header","version":1,"created":"2026-09-11T10:00:00+00:00","run_id":"9f3a2c1b",
   "connectome_key":"bccd14effb17","settings":{...redacted...}}``

``{"kind":"tick","seq":1,"t_ms":50,"wall":1789051563.12,"total_spikes":812,
   "market":{...MarketSnapshot.to_wire() or null},"trades":[{kind,usd,ts,surrogate,price}],
   "pokes":[{stim,strength,side,until_ms}],"commands":[{"name":...}]}``

Writing never raises into the simulation loop: the first failure is logged, ``ok`` goes ``False`` and the
log goes quiet for the rest of the run (SPEC 0.1 "errors never stop the sim").
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Sequence, TYPE_CHECKING

__all__ = ["SessionLog", "read_session", "SESSION_LOG_VERSION"]

if TYPE_CHECKING:  # pragma: no cover - typing only (no import at module import time, SPEC 0.1)
    from flybrain.config import Settings
    from flybrain.market.base import MarketSnapshot, Trade

log = logging.getLogger("flybrain.server.session_log")

SESSION_LOG_VERSION: Final[int] = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _trade_dict(trade: Any) -> dict:
    """Wire form of a trade plus ``price`` (the replay needs the exact trade, not only its wire view)."""
    if isinstance(trade, Mapping):
        return dict(trade)
    out: dict[str, Any] = {}
    to_wire = getattr(trade, "to_wire", None)
    if callable(to_wire):
        try:
            out = dict(to_wire())
        except Exception:  # noqa: BLE001
            out = {}
    if not out:
        out = {k: getattr(trade, k, None) for k in ("kind", "usd", "ts", "surrogate")}
    price = getattr(trade, "price", None)
    if price is not None and "price" not in out:
        out["price"] = float(price)
    return out


def _poke_dict(poke: Any) -> dict:
    """``{stim, strength, side, until_ms}`` -- exactly the ``Poke`` constructor keywords."""
    if isinstance(poke, Mapping):
        return {k: poke.get(k) for k in ("stim", "strength", "side", "until_ms")}
    return {"stim": str(getattr(poke, "stim", "")), "strength": float(getattr(poke, "strength", 0.0) or 0.0),
            "side": int(getattr(poke, "side", 0) or 0), "until_ms": int(getattr(poke, "until_ms", 0) or 0)}


def _snapshot_dict(snap: Any) -> dict | None:
    if snap is None:
        return None
    if isinstance(snap, Mapping):
        return dict(snap)
    to_wire = getattr(snap, "to_wire", None)
    if callable(to_wire):
        try:
            return dict(to_wire())
        except Exception:  # noqa: BLE001
            return None
    return None


class SessionLog:
    """Append-only JSONL writer for one run (SPEC c.25).

    ``SessionLog(path, settings)`` creates the parent directory, opens the file for writing and emits
    the header line immediately. Thread-safe (the sim thread writes, ``close()`` may come from the
    asyncio shutdown path). ``ok`` is ``False`` once writing failed.
    """

    def __init__(self, path: Path, settings: "Settings") -> None:
        self.path = Path(path)
        self.settings = settings
        self.ok = True
        self.lines = 0
        self._lock = threading.Lock()
        self._fh = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # line buffered: a reader (a test, a tail -f, scripts/replay.py on a live run) sees every
            # tick as soon as it is written, without an fsync per tick
            self._fh = self.path.open("w", encoding="utf-8", newline="\n", buffering=1)
        except OSError as exc:
            self.ok = False
            log.warning("session_log: cannot open %s (%s); logging disabled", self.path, exc)
            return
        self._write(self._header())

    # ---------------------------------------------------------------- header
    def _header(self) -> dict:
        try:
            from flybrain.config import redacted  # local import (SPEC 0.1: no cycles at import time)

            settings_dict = redacted(self.settings)
        except Exception as exc:  # noqa: BLE001 - a duck-typed settings object in a test
            log.debug("session_log: redacted() unavailable (%s)", exc)
            settings_dict = {}
        return {"kind": "header", "version": SESSION_LOG_VERSION, "created": _now_iso(),
                "run_id": str(getattr(self.settings, "run_id", "") or ""),
                "connectome_key": str(getattr(self.settings, "connectome_key", "") or ""),
                "settings": settings_dict}

    # ---------------------------------------------------------------- writing
    def write_tick(self, seq: int, t_ms: int, wall: float, snap: "MarketSnapshot | None",
                   trades: Sequence[Any], pokes: Sequence[Any], commands: Sequence[Mapping[str, Any]],
                   total_spikes: int) -> None:
        """Append one tick line with every input of that tick plus its spike checksum."""
        if not self.ok or self._fh is None:
            return
        row = {
            "kind": "tick", "seq": int(seq), "t_ms": int(t_ms), "wall": round(float(wall), 3),
            "total_spikes": int(total_spikes),
            "market": _snapshot_dict(snap),
            "trades": [_trade_dict(t) for t in (trades or ())],
            "pokes": [_poke_dict(p) for p in (pokes or ())],
            "commands": [dict(c) for c in (commands or ())],
        }
        self._write(row)

    def write_record(self, row: Mapping[str, Any]) -> None:
        """Append an arbitrary record (``kind`` should be set by the caller)."""
        self._write(dict(row))

    def _write(self, row: Mapping[str, Any]) -> None:
        if self._fh is None or not self.ok:
            return
        try:
            text = json.dumps(row, separators=(",", ":"), allow_nan=False, default=str)
        except (TypeError, ValueError) as exc:
            log.warning("session_log: row %r not serialisable (%s)", row.get("kind"), exc)
            return
        with self._lock:
            if self._fh is None:
                return
            try:
                self._fh.write(text + "\n")
                self.lines += 1
            except (OSError, ValueError) as exc:
                self.ok = False
                log.warning("session_log: write failed (%s); logging disabled", exc)

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                except OSError:  # pragma: no cover
                    pass

    def close(self) -> None:
        """Flush and close (idempotent)."""
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.flush()
            fh.close()
        except OSError as exc:  # pragma: no cover
            log.debug("session_log: close failed (%s)", exc)
        log.info("session_log: %d line(s) written to %s", self.lines, self.path)

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_session(path: Path) -> Iterator[dict]:
    """Yield every JSON object of a session log, header first (SPEC c.25).

    Corrupt or truncated lines (a run killed mid-write) are skipped with a warning so a replay still
    works on everything that was flushed.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except ValueError as exc:
                log.warning("read_session: %s:%d skipped (%s)", p.name, lineno, exc)
                continue
            if isinstance(row, dict):
                yield row
            else:
                log.warning("read_session: %s:%d skipped (not an object)", p.name, lineno)
