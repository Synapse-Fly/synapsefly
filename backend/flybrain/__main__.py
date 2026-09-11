"""``py -3 -m flybrain``: parse the overrides, load the settings, serve the app (SPEC section c.28).

Every flag is a thin override of the corresponding ``FLY_*`` environment variable of SPEC section b, so
``py -3 -m flybrain --n 4000 --market sim`` behaves exactly like exporting ``FLY_N_NEURONS=4000
FLY_MARKET=sim`` first. ``PYTHONUTF8=1`` is set (never overriding an explicit value) before anything is
imported that may print, and the banner is ASCII only (SPEC 0.1: the Windows console is cp1254).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Final

__all__ = ["main", "build_parser", "ARG_ENV", "bind_hosts"]

#: Loopback hosts whose ``localhost`` alias resolves ``::1`` before ``127.0.0.1`` on Windows.
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})


def _ipv6_loopback_ok() -> bool:
    """True when an ``::1`` TCP socket can actually be bound (IPv6 loopback is up)."""
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        sock.bind(("::1", 0))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def bind_hosts(host: str) -> str | list[str]:
    """The uvicorn ``host`` for ``settings.host`` (SPEC section b, ``FLY_HOST`` default ``127.0.0.1``).

    The frontend default (``NEXT_PUBLIC_WS_URL=ws://localhost:4000/ws``, SPEC section b) connects to
    ``localhost``, which on Windows resolves ``::1`` (IPv6) *before* ``127.0.0.1``; a server bound to the
    IPv4 loopback alone refuses that first attempt, so a fresh user sees a stream of
    ``ws://localhost:4000/ws failed`` console errors before the browser retries on IPv4. Binding both
    loopback families keeps ``FLY_HOST`` at its documented ``127.0.0.1`` yet lets ``localhost`` reach the
    server on the first try. Only the loopback default is broadened, and only when ``::1`` can actually be
    bound, so an IPv6-disabled host still starts (returns the single address unchanged); an explicit
    ``0.0.0.0`` / LAN / IPv6 host is never touched.
    """
    if str(host).strip() in _LOOPBACK_HOSTS and _ipv6_loopback_ok():
        return ["127.0.0.1", "::1"]
    return str(host)

#: ``--flag`` -> the ``FLY_*`` variable it overrides (SPEC section b).
ARG_ENV: Final[dict[str, str]] = {
    "host": "FLY_HOST",
    "port": "FLY_PORT",
    "seed": "FLY_SEED",
    "n": "FLY_N_NEURONS",
    "source": "FLY_CONNECTOME_SOURCE",
    "market": "FLY_MARKET",
    "realtime": "FLY_REALTIME",
    "log_level": "FLY_LOG_LEVEL",
    "replay": "FLY_REPLAY",
}


def build_parser() -> argparse.ArgumentParser:
    """The CLI of SPEC c.28 (``-h`` works everywhere, ASCII only)."""
    p = argparse.ArgumentParser(prog="flybrain",
                                description="SynapseFly / FlyBrain server: a spiking MaleCNS-shaped fly "
                                            "brain wired to a token market.")
    p.add_argument("--host", default=None, help="bind address (FLY_HOST, default 127.0.0.1)")
    p.add_argument("--port", type=int, default=None, help="TCP port (FLY_PORT, default 4000)")
    p.add_argument("--seed", type=int, default=None, help="master seed (FLY_SEED, default 1337)")
    p.add_argument("--n", type=int, default=None, help="synthetic neuron count (FLY_N_NEURONS)")
    p.add_argument("--source", choices=("synthetic", "csv", "neuprint"), default=None,
                   help="connectome source (FLY_CONNECTOME_SOURCE)")
    p.add_argument("--market", choices=("sim", "dexscreener"), default=None, help="market source (FLY_MARKET)")
    p.add_argument("--replay", default=None, help="session log to replay instead of live inputs (FLY_REPLAY)")
    p.add_argument("--log-level", dest="log_level", default=None,
                   choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"), help="FLY_LOG_LEVEL")
    rt = p.add_mutually_exclusive_group()
    rt.add_argument("--realtime", dest="realtime", action="store_const", const="1",
                    help="wall-clock 20 Hz ticks (default)")
    rt.add_argument("--no-realtime", dest="realtime", action="store_const", const="0",
                    help="run as fast as possible, fixed steps per tick (tests, replay)")
    p.set_defaults(realtime=None)
    p.add_argument("--no-gates", dest="gates", action="store_false",
                   help="skip the SPEC g.6 boot gate run (faster start)")
    p.set_defaults(gates=True)
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, apply the overrides to the environment, run uvicorn. Returns 0.

    A bad ``FLY_*`` value is an operator mistake, not a crash: ``load_settings`` already formats a
    one-line ASCII message naming the variable (SPEC c.1), so it is printed as ``flybrain: <message>``
    on stderr and ``main`` returns 2 instead of dumping a traceback (SPEC 0.1 / h.4).
    """
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    os.environ.setdefault("PYTHONUTF8", "1")
    for attr, var in ARG_ENV.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[var] = str(value)

    from .config import load_settings
    from .server.app import create_app

    try:
        settings = load_settings()
        app = create_app(settings, boot_gates=bool(args.gates))
    except (ValueError, OSError) as exc:          # bad FLY_* value, unreadable .env, unwritable data dir
        print(f"flybrain: {exc}", file=sys.stderr)
        return 2

    import uvicorn

    print(f"FlyBrain {settings.run_id}: http://{settings.host}:{settings.port}  "
          f"ws://{settings.host}:{settings.port}/ws")
    print(f"  connectome={settings.connectome_source} n={settings.n_neurons} seed={settings.seed} "
          f"dt={settings.dt_ms}ms tick={settings.tick_hz}Hz realtime={int(settings.realtime)}")
    print(f"  market={settings.market} llm={settings.llm} x={settings.x_mode} "
          f"data={settings.data_dir}")
    uvicorn.run(app, host=bind_hosts(settings.host), port=int(settings.port), log_level="info",
                ws="websockets")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
