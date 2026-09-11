#!/usr/bin/env python3
"""Stage demo for the simulated market (SPEC section h.4 ``scripts/demo_market.py``).

Drives the running FlyBrain server through ``POST /api/market/mode``:

    PUMP (60 s) -> RUG (40 s) -> CALM (60 s) -> DEAD (120 s) -> CHOP (45 s)

printing the fly's mood from ``GET /api/state`` every 5 s. Expected to observe FEEDING / EUPHORIA during the
pump, ESCAPE / PANIC during the rug, CRUISING in the calm, SLEEP in the dead market. ``--loop`` repeats the
stages forever, ``--dry-run`` only prints the plan (no network), ``--scale`` shortens every stage (rehearsal).

Exit code: 0 only when every **hard** stage observed one of its expected moods (and no poll failed), 1
otherwise - so CI sees a miss; ``--no-strict`` downgrades every miss to a warning (exit 0).

The ``DEAD -> SLEEP`` stage is **advisory** by default (``ADVISORY_STAGES``) because it cannot be reached in the
120 s the SPEC h.4 table allows: ``buys_m5`` / ``sells_m5`` are 5-minute windows, so after a busy stage the
f.1 ``activity`` only drops below the 0.35 of the f.6 SLEEP gate once ~300 s of quiet have flushed the window,
and the gate then needs a further ~90 s to raise ``sleep_pressure`` to 0.95 plus a 20 s hold. Pass
``--require-sleep`` (with ``--scale`` >= 4, or simply watch a longer DEAD stretch) to make it hard.

Usage: py -3 scripts/demo_market.py [--url http://127.0.0.1:4000] [--loop] [--dry-run] [--scale 1.0]
                                    [--no-strict] [--require-sleep]
"""

from __future__ import annotations

import argparse
import sys
import time

#: (regime, seconds, expected moods) - SPEC section h.4.
STAGES: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("PUMP", 60.0, ("FEEDING", "EUPHORIA")),
    ("RUG", 40.0, ("ESCAPE", "PANIC")),
    ("CALM", 60.0, ("CRUISING",)),
    ("DEAD", 120.0, ("SLEEP",)),
    ("CHOP", 45.0, ()),
)
#: Stages whose expected mood is not reachable inside the h.4 duration (module docstring): a miss is reported
#: but does not fail the run unless ``--require-sleep`` is given.
ADVISORY_STAGES: frozenset[str] = frozenset({"DEAD"})
POLL_S: float = 5.0
#: The server forces a regime for 60 s (SPEC d.7); re-post well inside that window.
REPOST_S: float = 30.0
HTTP_TIMEOUT_S: float = 5.0


def ascii_only(text: object) -> str:
    """Every byte printed by this script is ASCII (SPEC h.4): OS / exception messages are localised on
    Windows (cp1254 console), and a non-encodable character would raise ``UnicodeEncodeError``."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S") + "  " + ascii_only(msg), flush=True)


def plan_lines(scale: float, require_sleep: bool = False) -> list[str]:
    lines = ["stage | regime | seconds | expect"]
    for i, (regime, secs, expect) in enumerate(STAGES, 1):
        kind = "" if (not expect or regime not in ADVISORY_STAGES or require_sleep) else "  (advisory)"
        lines.append(f"{i:>5} | {regime:<6} | {secs * scale:>7.1f} | {'/'.join(expect) or '-'}{kind}")
    total = sum(s for _, s, _ in STAGES) * scale
    lines.append(f"total {total:.0f} s; POST /api/market/mode every {REPOST_S * scale:.0f} s, GET /api/state every {POLL_S * scale:.1f} s")
    return lines


class Server:
    """Tiny httpx wrapper (httpx is a runtime hard dependency, SPEC section 0.1)."""

    def __init__(self, url: str) -> None:
        import httpx

        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": "synapsefly/0.1 demo_market"})

    def close(self) -> None:
        self.client.close()

    def health(self) -> dict:
        r = self.client.get(self.url + "/api/health")
        r.raise_for_status()
        return r.json()

    def set_mode(self, mode: str) -> dict:
        r = self.client.post(self.url + "/api/market/mode", json={"mode": mode})
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self.client.get(self.url + "/api/state")
        r.raise_for_status()
        return r.json()


def _mood_line(state: dict) -> tuple[str, str]:
    tick = state.get("tick") or {}
    mood = (tick.get("mood") or {}).get("state", "?")
    fly = tick.get("fly") or {}
    market = tick.get("market") or {}
    drives = tick.get("drives") or {}
    price = market.get("price_usd")
    price_s = "?" if price is None else f"{price:.6g}"
    line = (f"mood={mood:<9} fly={fly.get('mode', '?'):<6} regime={market.get('regime') or '-':<5} "
            f"price={price_s:<10} chg_m5={market.get('chg_m5', 0.0):+6.2f}% b/s={market.get('buys_m5', 0)}/{market.get('sells_m5', 0)} "
            f"sugar={drives.get('sugar', 0.0):.2f} loom={drives.get('looming', 0.0):.2f} sleep={drives.get('sleep_pressure', 0.0):.2f}")
    return mood, line


def run_stage(srv: Server, regime: str, seconds: float, expect: tuple[str, ...], scale: float,
              advisory: bool = False) -> tuple[bool, set[str]]:
    """Force ``regime``, poll the mood for ``seconds * scale``. Returns (ok, observed moods); ``ok`` is False on
    a poll / post error or - unless ``advisory`` - when none of ``expect`` was observed."""
    dur = seconds * scale
    poll = max(0.5, POLL_S * scale)
    repost = max(poll, REPOST_S * scale)
    res = srv.set_mode(regime)
    if not res.get("ok", False):
        _log(f"stage {regime}: server refused mode ({res})")
        return False, set()
    _log(f"== stage {regime} for {dur:.0f} s (expect {'/'.join(expect) or 'anything'}) ==")
    seen: set[str] = set()
    t_start = time.monotonic()
    t_last_post = t_start
    ok = True
    while True:
        try:
            mood, line = _mood_line(srv.state())
            seen.add(mood)
            _log(f"   {line}")
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            ok = False
            _log(f"   state poll failed: {type(exc).__name__}: {ascii_only(exc)}")
        now = time.monotonic()
        if now - t_start >= dur:
            break
        if now - t_last_post >= repost:
            try:
                srv.set_mode(regime)
                t_last_post = now
            except Exception as exc:  # noqa: BLE001
                ok = False
                _log(f"   re-post {regime} failed: {type(exc).__name__}: {ascii_only(exc)}")
        time.sleep(min(poll, max(0.0, t_start + dur - now)))
    hit = (not expect) or any(m in seen for m in expect)
    verdict = "OK" if hit else ("MISSED (advisory)" if advisory else "MISSED")
    _log(f"== stage {regime} done: observed {sorted(seen)} -> {verdict} ==")
    return ok and (hit or advisory), seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drive the simulated market PUMP -> RUG -> CALM -> DEAD -> CHOP and watch the mood.")
    ap.add_argument("--url", default="http://127.0.0.1:4000", help="backend base URL (default http://127.0.0.1:4000)")
    ap.add_argument("--loop", action="store_true", help="repeat the stages forever (Ctrl+C stops)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without touching the server")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every stage duration (0.1 = quick rehearsal)")
    ap.add_argument("--strict", dest="strict", action="store_true", default=True,
                    help="exit 1 when a stage's expected moods were not observed (default)")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="report misses but always exit 0")
    ap.add_argument("--require-sleep", action="store_true",
                    help=f"treat the advisory stage(s) {sorted(ADVISORY_STAGES)} as hard too "
                         "(needs a much longer DEAD stretch: the m5 windows are 300 s)")
    args = ap.parse_args(argv)
    if args.scale <= 0:
        ap.error("--scale must be > 0")

    print("demo_market plan:")
    for line in plan_lines(args.scale, args.require_sleep):
        print("  " + line)
    if not args.require_sleep:
        print("  advisory stages do not fail the run (see --require-sleep; m5 windows are 300 s wide)")
    if args.dry_run:
        print("dry-run: nothing posted.")
        return 0

    try:
        srv = Server(args.url)
    except ImportError as exc:  # pragma: no cover
        print(f"httpx is required: {ascii_only(exc)}")
        return 1
    try:
        try:
            h = srv.health()
        except Exception as exc:  # noqa: BLE001
            print(f"cannot reach {args.url}/api/health: {type(exc).__name__}: {ascii_only(exc)}")
            print("start the backend first: py -3 backend\\run.py (or scripts\\dev.ps1)")
            return 1
        conn = h.get("connectome") or {}
        _log(f"server ok run_id={h.get('run_id')} connectome={conn.get('name')} market={(h.get('market') or {}).get('mode')} rtf={h.get('rtf')}")
        all_ok = True
        missed: list[str] = []
        rounds = 0
        while True:
            rounds += 1
            for regime, secs, expect in STAGES:
                advisory = bool(expect) and regime in ADVISORY_STAGES and not args.require_sleep
                ok, seen = run_stage(srv, regime, secs, expect, args.scale, advisory=advisory)
                if expect and not any(m in seen for m in expect):
                    missed.append(regime + (" (advisory)" if advisory else ""))
                all_ok = all_ok and ok
            if not args.loop:
                break
            _log(f"-- round {rounds} complete, looping --")
        try:
            srv.set_mode("sim")  # hand the regime back to the Markov chain
        except Exception:  # noqa: BLE001
            pass
        if missed:
            print("stages that did not observe an expected mood: " + ", ".join(missed))
        if not all_ok:
            print("FAILED: a hard stage missed its expected moods or a request failed"
                  + ("" if args.strict else " (--no-strict: exit 0)"))
            return 1 if args.strict else 0
        print("demo complete: every hard stage observed an expected mood")
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        try:
            srv.set_mode("sim")
        except Exception:  # noqa: BLE001
            pass
        return 130
    finally:
        srv.close()


if __name__ == "__main__":
    sys.exit(main())
