"""Configuration: the frozen ``Settings`` record, ``load_settings``, ``gain_value``, ``redacted``.

SPEC section b is the normative table of environment variables (every variable, its default, its allowed
values); SPEC section c.1 is the API contract of this module. Nothing here reads the network and nothing
depends on the current working directory: ``data_dir`` / ``out_dir`` are resolved from ``__file__``
(three parents = the repository root, SPEC section a) and created on demand.

Precedence (SPEC section b): ``.env`` at the repository root is parsed first, then the process
environment wins. ``load_settings(env=...)`` replaces the process environment entirely so tests are
hermetic; ``load_settings(dotenv=None)`` skips the ``.env`` file (the default, ``_AUTO``, reads
``<repo>/.env`` when it exists).

Secrets (``ANTHROPIC_API_KEY``, ``X_API_KEY``, ``X_API_SECRET``, ``X_ACCESS_TOKEN``,
``X_ACCESS_TOKEN_SECRET``, ``NEUPRINT_APPLICATION_CREDENTIALS``) are deliberately **not** fields of
``Settings``: ``agent/llm.py`` and ``agent/x_client.py`` read them from ``os.environ`` so that a key
never reaches a log line, ``GET /api/config`` or a session-log header. When a ``.env`` file supplies
them during a real run (``env is None``) they are copied into ``os.environ`` with ``setdefault`` (an
already exported value always wins) so the Anthropic SDK and tweepy find them. ``redacted()`` drops any
secret-looking field anyway and ``mask_secrets()`` renders an environment mapping safe for logging.

All biological numbers live elsewhere; every value here is ``[E]`` engineered configuration.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Final, Mapping

__all__ = [
    "Settings",
    "load_settings",
    "gain_value",
    "redacted",
    "repo_root",
    "parse_bool",
    "parse_dotenv",
    "mask_secrets",
    "EnvVar",
    "ENV_VARS",
    "ENV_BY_NAME",
    "SECRET_ENV_VARS",
    "IGNORED_ENV_VARS",
    "CONNECTOME_SOURCES",
    "SYNTH_WEIGHT_MODES",
    "SUBSETS",
    "BACKENDS",
    "DRIVE_MODES",
    "MARKET_SOURCES",
    "LLM_MODES",
    "X_MODES",
    "TWEET_LANGS",
    "WALL_MODES",
    "DT_MS_ALLOWED",
    "LOG_LEVELS",
    "BOOL_TRUE",
    "BOOL_FALSE",
]

log = logging.getLogger("flybrain.config")

# --------------------------------------------------------------------------------------------------
# allowed value sets (SPEC section b)
# --------------------------------------------------------------------------------------------------

CONNECTOME_SOURCES: Final[tuple[str, ...]] = ("synthetic", "csv", "neuprint")
SYNTH_WEIGHT_MODES: Final[tuple[str, ...]] = ("calibrated", "literature")
SUBSETS: Final[tuple[str, ...]] = ("core", "all")
BACKENDS: Final[tuple[str, ...]] = ("numpy", "torch", "auto")
DRIVE_MODES: Final[tuple[str, ...]] = ("poisson", "current")
MARKET_SOURCES: Final[tuple[str, ...]] = ("sim", "dexscreener")
LLM_MODES: Final[tuple[str, ...]] = ("dryrun", "anthropic")
X_MODES: Final[tuple[str, ...]] = ("dryrun", "post")
TWEET_LANGS: Final[tuple[str, ...]] = ("en", "tr")
WALL_MODES: Final[tuple[str, ...]] = ("bounce", "wrap")
DT_MS_ALLOWED: Final[tuple[float, ...]] = (1.0, 0.5, 0.2, 0.1)
LOG_LEVELS: Final[tuple[str, ...]] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET")

BOOL_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on", "t"})
BOOL_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "no", "n", "off", "f"})

#: Never stored in ``Settings``, never logged, never returned by ``redacted()``.
SECRET_ENV_VARS: Final[tuple[str, ...]] = (
    "ANTHROPIC_API_KEY",
    "X_API_KEY",
    "X_API_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_TOKEN_SECRET",
    "NEUPRINT_APPLICATION_CREDENTIALS",
)

#: Recognised but not part of ``Settings`` (frontend / tooling keys that may sit in the same ``.env``).
IGNORED_ENV_VARS: Final[tuple[str, ...]] = (
    "NEXT_PUBLIC_WS_URL",
    "NEXT_PUBLIC_API_URL",
    "PYTHONUTF8",
)

_SECRET_FIELD_MARKERS: Final[tuple[str, ...]] = ("x_api", "x_access", "anthropic", "api_key", "secret",
                                                 "token_secret", "credential", "password")


# --------------------------------------------------------------------------------------------------
# Settings (SPEC c.1, field names and defaults verbatim)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable configuration of one process (SPEC c.1).

    Constructing ``Settings()`` with zero arguments yields exactly the offline / dry-run defaults of
    SPEC section b (synthetic connectome, simulated market, dry-run LLM and X): importable and usable
    with no environment at all. ``load_settings`` is the only place that reads the environment, and the
    only place that fills the four derived fields at the bottom.
    """

    # connectome
    connectome_source: str = "synthetic"      # FLY_CONNECTOME_SOURCE: synthetic|csv|neuprint
    connectome_dir: Path = Path("data/connectome/malecns")
    connectome_name: str = ""
    n_neurons: int = 20_000
    mean_outdeg: int = 25
    synth_weights: str = "calibrated"         # calibrated|literature
    subset: str = "core"                      # core|all
    min_weight: int = 3
    # engine
    dt_ms: float = 1.0
    seed: int = 1337
    backend: str = "numpy"                    # numpy|torch|auto
    gain: str = "auto"                        # "auto" or a float literal, e.g. "0.65"
    noise_mu: float = 0.5
    noise_sigma: float = 3.5
    drive_mode: str = "poisson"               # poisson|current
    tick_hz: int = 20
    realtime: bool = True
    # market
    market: str = "sim"                       # sim|dexscreener
    token_address: str = ""
    chain: str = "solana"
    dex_poll_s: int = 60
    sim_regime_s: float = 90.0
    # agent
    llm: str = "dryrun"                       # dryrun|anthropic
    llm_model: str = "claude-opus-5"
    llm_json: bool = True
    llm_fallbacks: bool = False
    tweet_lang: str = "en"
    x_mode: str = "dryrun"                    # FLY_X: dryrun|post
    tweet_cooldown_s: int = 900
    tweet_reason_cooldown_s: int = 2700
    tweets_per_day: int = 12
    # server / body
    port: int = 4000
    host: str = "127.0.0.1"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    canvas_w: int = 800
    canvas_h: int = 500
    walls: str = "bounce"                     # bounce|wrap
    raster_per_region: int = 48
    raster_cap: int = 2000
    mood_feedback: bool = True
    explore_baseline: float = 0.25
    wander_sigma: float = 0.6
    easter_eggs: bool = True
    session_log: bool = True
    replay: str = ""
    data_dir: Path = Path("data")             # absolute after load_settings
    out_dir: Path = Path("out")
    log_level: str = "INFO"
    # derived (filled by load_settings)
    tick_ms: float = 50.0
    steps_per_tick: int = 50
    run_id: str = ""
    connectome_key: str = ""

    # -- convenience (no state, no I/O) ------------------------------------------------------------
    @property
    def tick_s(self) -> float:
        """Wall seconds per tick (``tick_ms / 1000``)."""
        return float(self.tick_ms) / 1000.0

    @property
    def brain_ms_per_tick(self) -> float:
        """Brain ms simulated per tick at ``speed == 1`` (``steps_per_tick * dt_ms``)."""
        return float(self.steps_per_tick) * float(self.dt_ms)

    @property
    def sessions_dir(self) -> Path:
        """``<data_dir>/sessions`` (``SessionLog`` target, SPEC section a)."""
        return Path(self.data_dir) / "sessions"

    @property
    def snapshots_dir(self) -> Path:
        """``<data_dir>/snapshots``."""
        return Path(self.data_dir) / "snapshots"

    @property
    def cache_dir(self) -> Path:
        """``<data_dir>/cache``."""
        return Path(self.data_dir) / "cache"

    @property
    def tweets_dir(self) -> Path:
        """``<out_dir>/tweets`` (dry-run artifacts)."""
        return Path(self.out_dir) / "tweets"


# --------------------------------------------------------------------------------------------------
# environment variable registry (SPEC section b)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvVar:
    """One row of the SPEC section b table.

    ``field`` is the ``Settings`` field it fills (``""`` for secrets / frontend keys that are
    recognised but never stored). ``kind`` drives the parser: ``str|int|float|bool|path|choice|csv|
    gain|dt|level``. ``lo`` / ``hi`` are inclusive bounds for ``int`` / ``float``.
    """

    name: str
    field: str
    kind: str
    choices: tuple[str, ...] = ()
    lo: float | None = None
    hi: float | None = None
    note: str = ""


ENV_VARS: Final[tuple[EnvVar, ...]] = (
    # ---- connectome
    EnvVar("FLY_CONNECTOME_SOURCE", "connectome_source", "choice", CONNECTOME_SOURCES),
    EnvVar("FLY_CONNECTOME_DIR", "connectome_dir", "path",
           note="relative paths are resolved by connectome.cache.resolve_connectome_dir"),
    EnvVar("FLY_CONNECTOME_NAME", "connectome_name", "str"),
    EnvVar("FLY_N_NEURONS", "n_neurons", "int", lo=4_000, hi=200_000),
    EnvVar("FLY_MEAN_OUTDEG", "mean_outdeg", "int", lo=5, hi=200),
    EnvVar("FLY_SYNTH_WEIGHTS", "synth_weights", "choice", SYNTH_WEIGHT_MODES),
    EnvVar("FLY_SUBSET", "subset", "choice", SUBSETS),
    EnvVar("FLY_MIN_WEIGHT", "min_weight", "int", lo=1),
    # ---- engine
    EnvVar("FLY_DT_MS", "dt_ms", "dt"),
    EnvVar("FLY_SEED", "seed", "int", lo=0, hi=2**32 - 1),
    EnvVar("FLY_BACKEND", "backend", "choice", BACKENDS),
    EnvVar("FLY_GAIN", "gain", "gain"),
    EnvVar("FLY_NOISE_MU", "noise_mu", "float", lo=0.0, hi=1_000.0),
    EnvVar("FLY_NOISE_SIGMA", "noise_sigma", "float", lo=0.0, hi=1_000.0),
    EnvVar("FLY_DRIVE_MODE", "drive_mode", "choice", DRIVE_MODES),
    EnvVar("FLY_TICK_HZ", "tick_hz", "int", lo=5, hi=50),
    EnvVar("FLY_REALTIME", "realtime", "bool"),
    # ---- market
    EnvVar("FLY_MARKET", "market", "choice", MARKET_SOURCES),
    EnvVar("FLY_TOKEN_ADDRESS", "token_address", "str"),
    EnvVar("FLY_CHAIN", "chain", "str"),
    EnvVar("FLY_DEX_POLL_S", "dex_poll_s", "int", lo=15, hi=86_400),
    EnvVar("FLY_SIM_REGIME_S", "sim_regime_s", "float", lo=1.0, hi=86_400.0),
    # ---- agent (LLM)
    EnvVar("FLY_LLM", "llm", "choice", LLM_MODES),
    EnvVar("ANTHROPIC_API_KEY", "", "secret"),
    EnvVar("FLY_LLM_MODEL", "llm_model", "str"),
    EnvVar("FLY_LLM_JSON", "llm_json", "bool"),
    EnvVar("FLY_LLM_FALLBACKS", "llm_fallbacks", "bool"),
    EnvVar("FLY_TWEET_LANG", "tweet_lang", "choice", TWEET_LANGS),
    # ---- agent (X)
    EnvVar("FLY_X", "x_mode", "choice", X_MODES),
    EnvVar("X_API_KEY", "", "secret"),
    EnvVar("X_API_SECRET", "", "secret"),
    EnvVar("X_ACCESS_TOKEN", "", "secret"),
    EnvVar("X_ACCESS_TOKEN_SECRET", "", "secret"),
    EnvVar("FLY_TWEET_COOLDOWN_S", "tweet_cooldown_s", "int", lo=0, hi=86_400),
    EnvVar("FLY_TWEET_REASON_COOLDOWN_S", "tweet_reason_cooldown_s", "int", lo=0, hi=86_400),
    EnvVar("FLY_TWEETS_PER_DAY", "tweets_per_day", "int", lo=0, hi=1_000),
    # ---- server / body
    EnvVar("FLY_PORT", "port", "int", lo=1, hi=65_535),
    EnvVar("FLY_HOST", "host", "str"),
    EnvVar("FLY_CORS_ORIGINS", "cors_origins", "csv"),
    EnvVar("FLY_CANVAS_W", "canvas_w", "int", lo=64, hi=8_000),
    EnvVar("FLY_CANVAS_H", "canvas_h", "int", lo=64, hi=8_000),
    EnvVar("FLY_WALLS", "walls", "choice", WALL_MODES),
    EnvVar("FLY_RASTER_PER_REGION", "raster_per_region", "int", lo=8, hi=128),
    EnvVar("FLY_RASTER_CAP", "raster_cap", "int", lo=1, hi=200_000),
    EnvVar("FLY_MOOD_FEEDBACK", "mood_feedback", "bool"),
    EnvVar("FLY_EXPLORE_BASELINE", "explore_baseline", "float", lo=0.0, hi=1.0),
    EnvVar("FLY_WANDER_SIGMA", "wander_sigma", "float", lo=0.0, hi=100.0),
    EnvVar("FLY_EASTER_EGGS", "easter_eggs", "bool"),
    EnvVar("FLY_SESSION_LOG", "session_log", "bool"),
    EnvVar("FLY_REPLAY", "replay", "str"),
    EnvVar("FLY_DATA_DIR", "data_dir", "path"),
    EnvVar("FLY_OUT_DIR", "out_dir", "path"),
    EnvVar("FLY_LOG_LEVEL", "log_level", "level"),
    # ---- data scripts only
    EnvVar("NEUPRINT_APPLICATION_CREDENTIALS", "", "secret"),
    # ---- frontend (recognised, never stored)
    EnvVar("NEXT_PUBLIC_WS_URL", "", "ignored"),
    EnvVar("NEXT_PUBLIC_API_URL", "", "ignored"),
)

ENV_BY_NAME: Final[dict[str, EnvVar]] = {v.name: v for v in ENV_VARS}
_FIELD_TO_ENV: Final[dict[str, str]] = {v.field: v.name for v in ENV_VARS if v.field}


# --------------------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------------------


def repo_root() -> Path:
    """The repository root, computed from ``__file__`` (``backend/flybrain/config.py`` -> three parents).

    Never derived from the current working directory (SPEC section 0.1).
    """
    return Path(__file__).resolve().parents[2]


def parse_bool(raw: str | bool | int | None, var: str = "value") -> bool:
    """``1/0/true/false/yes/no`` (also ``on/off``, ``y/n``, ``t/f``), case-insensitive.

    Raises ``ValueError`` naming ``var`` for anything else (SPEC section b).
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    text = ("" if raw is None else str(raw)).strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    raise ValueError(f"{var}: expected a boolean (1/0/true/false/yes/no), got {raw!r}")


def parse_dotenv(path: Path | str) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file: ``#`` comments, blank lines skipped, keys/values stripped.

    Tolerates a UTF-8 BOM, a leading ``export``, and one layer of matching quotes around the value.
    A missing or unreadable file yields ``{}`` (configuration never stops the process). Lines without
    ``=`` or with a non-identifier key are ignored with a debug log line -- no extra dependency, no
    shell semantics (SPEC section b).
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except (OSError, UnicodeError) as exc:
        log.debug("config: cannot read %s (%s)", p, exc)
        return {}
    out: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[len("export "):].strip()
        if "=" not in s:
            log.debug("config: %s:%d ignored (no '=')", p.name, lineno)
            continue
        key, _, value = s.partition("=")
        key = key.strip()
        if not key.replace("_", "a").isalnum():
            log.debug("config: %s:%d ignored (bad key %r)", p.name, lineno, key)
            continue
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[key] = v
    return out


def mask_secrets(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``{name: "<set>" | ""}`` for every secret variable -- safe to log (values never included)."""
    src = os.environ if env is None else env
    return {name: ("<set>" if str(src.get(name, "")).strip() else "") for name in SECRET_ENV_VARS}


# --------------------------------------------------------------------------------------------------
# coercion
# --------------------------------------------------------------------------------------------------


def _err(var: str, raw: str, detail: str) -> ValueError:
    return ValueError(f"{var}: {detail} (got {raw!r})")


def _as_int(var: str, raw: str, lo: float | None, hi: float | None) -> int:
    try:
        value = int(str(raw).strip(), 10)
    except ValueError:
        try:  # tolerate "20000.0" coming out of a spreadsheet
            as_float = float(str(raw).strip())
        except ValueError:
            raise _err(var, raw, "expected an integer") from None
        if as_float != int(as_float):
            raise _err(var, raw, "expected an integer")
        value = int(as_float)
    if lo is not None and value < lo:
        raise _err(var, raw, f"expected an integer >= {int(lo)}")
    if hi is not None and value > hi:
        raise _err(var, raw, f"expected an integer <= {int(hi)}")
    return value


def _as_float(var: str, raw: str, lo: float | None, hi: float | None) -> float:
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise _err(var, raw, "expected a number") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise _err(var, raw, "expected a finite number")
    if lo is not None and value < lo:
        raise _err(var, raw, f"expected a number >= {lo}")
    if hi is not None and value > hi:
        raise _err(var, raw, f"expected a number <= {hi}")
    return value


def _as_choice(var: str, raw: str, choices: tuple[str, ...]) -> str:
    value = str(raw).strip()
    if value in choices:
        return value
    lowered = {c.lower(): c for c in choices}
    if value.lower() in lowered:
        return lowered[value.lower()]
    raise _err(var, raw, "expected one of " + "|".join(choices))


def _as_dt(var: str, raw: str) -> float:
    value = _as_float(var, raw, 0.0, 100.0)
    for allowed in DT_MS_ALLOWED:
        if abs(value - allowed) < 1e-9:
            return float(allowed)
    raise _err(var, raw, "expected one of " + "|".join(f"{d:g}" for d in DT_MS_ALLOWED))


def _as_gain(var: str, raw: str) -> str:
    value = str(raw).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        f = float(value)
    except ValueError:
        raise _err(var, raw, "expected 'auto' or a float") from None
    if not (f > 0.0) or f > 100.0:
        raise _err(var, raw, "expected 'auto' or a float in (0, 100]")
    return value


def _as_level(var: str, raw: str) -> str:
    value = str(raw).strip().upper()
    if value in LOG_LEVELS:
        return value
    if value.isdigit():  # a numeric level is legal for logging.setLevel
        return value
    raise _err(var, raw, "expected one of " + "|".join(LOG_LEVELS))


def _as_csv(var: str, raw: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in str(raw).split(",") if p.strip())
    if not parts:
        raise _err(var, raw, "expected a non-empty comma-separated list")
    return parts


def _coerce(spec: EnvVar, raw: str) -> Any:
    """Raw env string -> the typed field value (raises ``ValueError`` naming the variable)."""
    if spec.kind == "int":
        return _as_int(spec.name, raw, spec.lo, spec.hi)
    if spec.kind == "float":
        return _as_float(spec.name, raw, spec.lo, spec.hi)
    if spec.kind == "bool":
        return parse_bool(raw, spec.name)
    if spec.kind == "choice":
        return _as_choice(spec.name, raw, spec.choices)
    if spec.kind == "dt":
        return _as_dt(spec.name, raw)
    if spec.kind == "gain":
        return _as_gain(spec.name, raw)
    if spec.kind == "level":
        return _as_level(spec.name, raw)
    if spec.kind == "csv":
        return _as_csv(spec.name, raw)
    if spec.kind == "path":
        return Path(str(raw).strip())
    return str(raw).strip()


# --------------------------------------------------------------------------------------------------
# load_settings
# --------------------------------------------------------------------------------------------------


class _Auto:
    """Sentinel for "look for ``<repo>/.env``" (``dotenv=None`` means "no .env file at all")."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<auto>"


_AUTO: Final[_Auto] = _Auto()


def _resolve_dir(raw: Path | str | None, default: Path) -> Path:
    """Absolute directory: blank -> ``default``; relative -> relative to the repository root."""
    text = "" if raw is None else str(raw).strip()
    if not text:
        return default
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = repo_root() / p
    return Path(os.path.normpath(str(p)))


def _mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # configuration never stops the process (SPEC 0.1)
        log.warning("config: could not create %s (%s)", path, exc)


def _run_id(source: str, n_neurons: int, seed: int, dt_ms: float, synth_weights: str, market: str) -> str:
    """``sha1(f"{source}|{n_neurons}|{seed}|{dt_ms}|{synth_weights}|{market}")[:8]`` (SPEC section b)."""
    raw = f"{source}|{n_neurons}|{seed}|{dt_ms}|{synth_weights}|{market}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _connectome_key(settings: Settings) -> str:
    """``connectome.cache.cache_key(settings)`` (SPEC c.8), with a local fallback.

    Imported lazily: ``connectome.cache`` imports ``Settings`` only under ``TYPE_CHECKING``, so the
    lazy import keeps ``flybrain.config`` free of any package-level dependency.
    """
    try:
        from .connectome.cache import cache_key  # local import: avoids an import cycle

        return str(cache_key(settings))
    except Exception as exc:  # noqa: BLE001 - a key is always produced
        log.debug("config: cache_key unavailable (%s); using the local fallback", exc)
        raw = "|".join(str(x) for x in (settings.connectome_source, settings.connectome_dir,
                                        settings.n_neurons, settings.mean_outdeg, settings.synth_weights,
                                        settings.subset, settings.min_weight, settings.seed))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_settings(env: Mapping[str, str] | None = None, dotenv: Path | str | None = _AUTO) -> Settings:  # type: ignore[assignment]
    """Parse ``.env`` then the environment (environment wins), validate, return a frozen ``Settings``.

    ``env`` replaces the process environment when given (hermetic tests; SPEC h.1 passes exactly the
    seven ``FLY_*`` keys it cares about and expects section b defaults everywhere else). ``dotenv``
    defaults to ``<repo>/.env`` when that file exists; pass ``None`` to skip it entirely, or a path to
    read another file.

    Every value is validated against SPEC section b and a bad one raises ``ValueError`` whose message
    starts with the offending variable name. Blank values mean "use the default" (``.env.example``
    ships blank lines for the optional keys). ``data_dir`` / ``out_dir`` become absolute (relative to
    the repository root, never the CWD) and are created if missing; ``connectome_dir`` is left as
    given, because ``connectome.cache.resolve_connectome_dir`` owns its (data-dir aware) resolution.
    Derived fields: ``tick_ms``, ``steps_per_tick``, ``run_id``, ``connectome_key``.

    Side effect, real runs only (``env is None``): secrets found in the ``.env`` file are copied into
    ``os.environ`` with ``setdefault`` so the Anthropic SDK / tweepy can read them; an already
    exported value always wins and nothing is ever logged.
    """
    file_values: dict[str, str] = {}
    dotenv_path: Path | None = None
    if isinstance(dotenv, _Auto):
        candidate = repo_root() / ".env"
        dotenv_path = candidate if candidate.is_file() else None
    elif dotenv is not None:
        dotenv_path = Path(dotenv)
    if dotenv_path is not None:
        file_values = parse_dotenv(dotenv_path)
        if file_values:
            log.debug("config: %d keys from %s", len(file_values), dotenv_path)

    process_env: Mapping[str, str] = os.environ if env is None else env
    if env is None and file_values:  # make .env secrets visible to the SDKs (environ still wins)
        for name in SECRET_ENV_VARS:
            value = file_values.get(name, "").strip()
            if value:
                os.environ.setdefault(name, value)

    merged: dict[str, str] = {}
    for name in ENV_BY_NAME:
        raw = process_env.get(name, None)
        if raw is None or not str(raw).strip():
            raw = file_values.get(name, None)
        if raw is not None and str(raw).strip():
            merged[name] = str(raw)

    kwargs: dict[str, Any] = {}
    for spec in ENV_VARS:
        if not spec.field or spec.name not in merged:
            continue
        kwargs[spec.field] = _coerce(spec, merged[spec.name])

    root = repo_root()
    kwargs["data_dir"] = _resolve_dir(kwargs.get("data_dir"), root / "data")
    kwargs["out_dir"] = _resolve_dir(kwargs.get("out_dir"), root / "out")

    settings = Settings(**kwargs)

    # derived fields -------------------------------------------------------------------------------
    tick_ms = 1000.0 / float(settings.tick_hz)
    steps_per_tick = max(1, int(round(tick_ms / float(settings.dt_ms))))
    settings = replace(
        settings,
        tick_ms=tick_ms,
        steps_per_tick=steps_per_tick,
        run_id=_run_id(settings.connectome_source, settings.n_neurons, settings.seed, settings.dt_ms,
                       settings.synth_weights, settings.market),
    )
    settings = replace(settings, connectome_key=_connectome_key(settings))

    # cross-field checks ---------------------------------------------------------------------------
    if settings.market == "dexscreener" and not settings.token_address:
        log.warning("config: FLY_MARKET=dexscreener without FLY_TOKEN_ADDRESS; the feed will fall back to sim")
    if settings.llm == "anthropic" and not str(process_env.get("ANTHROPIC_API_KEY", "")).strip():
        log.warning("config: FLY_LLM=anthropic without ANTHROPIC_API_KEY; the agent degrades to templates")
    if settings.x_mode == "post":
        missing = [k for k in SECRET_ENV_VARS[1:5] if not str(process_env.get(k, "")).strip()]
        if missing:
            log.warning("config: FLY_X=post but %d OAuth1 variable(s) are unset; posting stays dry-run",
                        len(missing))
    if settings.replay:
        replay_path = Path(settings.replay)
        if not replay_path.is_absolute():
            replay_path = root / replay_path
        if not replay_path.is_file():
            log.warning("config: FLY_REPLAY=%s does not exist; running live inputs", settings.replay)

    _mkdir(settings.data_dir)
    _mkdir(settings.out_dir)
    return settings


# --------------------------------------------------------------------------------------------------
# derived helpers
# --------------------------------------------------------------------------------------------------


def gain_value(settings: Settings) -> float | None:
    """``None`` when ``settings.gain == 'auto'``, else ``float(settings.gain)`` (SPEC c.1)."""
    raw = str(getattr(settings, "gain", "auto") or "auto").strip()
    if raw.lower() == "auto":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"FLY_GAIN: expected 'auto' or a float (got {raw!r})") from None


def _is_secret_field(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in _SECRET_FIELD_MARKERS)


def redacted(settings: Settings) -> dict:
    """``asdict()`` with paths as strings and no secret field (``GET /api/config``, SPEC c.1).

    ``Settings`` holds no secret by construction, so this is mostly a type-normalising dump: ``Path``
    -> ``str``, ``tuple`` -> ``list``, and no key is added or renamed (``scripts/replay.py`` rebuilds a
    ``Settings`` from a session-log header written with this function). Any future field whose name
    looks like a credential (``*api_key*``, ``x_access*``, ``*secret*``, ``anthropic*``, ...) is
    dropped here rather than leaking into the API, the log or a session-log header.
    """
    try:
        raw = asdict(settings)
    except TypeError:  # a duck-typed settings object (tests / scripts)
        raw = {f.name: getattr(settings, f.name, None) for f in fields(Settings)}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if _is_secret_field(key):
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out
