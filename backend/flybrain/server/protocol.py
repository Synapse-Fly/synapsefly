"""WebSocket / REST wire contract: one pydantic v2 model per frame of SPEC section d.

SPEC c.26. Server -> client frames (``HelloMsg, TickMsg, MoodChangeMsg, TweetMsg, MarketMsg, EventMsg,
SnapshotRequestMsg, PongMsg, ErrorMsg``) are validated leniently (unknown keys ignored) so a future
additive field never kills a live connection; client -> server frames (``PokeMsg, SetMarketModeMsg,
ClearMsg, TweetTestMsg, PingMsg, SnapshotMsg``) use ``extra='forbid'`` and strict value ranges, because
they are untrusted input. ``parse_client`` (alias ``parse_client_frame``) turns a raw text/bytes frame
into one of those models and raises ``BadClientFrame`` -- a ``ValueError`` carrying the ``error`` frame
the handler must answer with (SPEC d.7/d.9).

Every key is snake_case and mirrors ``frontend/lib/types.ts`` 1:1 (SPEC e.1); the only Python-side
rename is ``MoodChangeMsg.from_`` -> wire ``from`` (a Python keyword), handled by an alias, so
``model_dump_json(by_alias=True)`` / ``wire_fields()`` always report the wire name.
``tests/test_protocol.py::test_types_ts_mirror`` diffs the two sides on every build.

Serialisation: ``tick_to_json`` is the hot path (once per tick, the same string goes to every client,
SPEC d.10) and applies the SPEC d rounding rules (positions/speeds 1 dp, angles 3 dp, rates 2 dp,
drives / mood scores 3 dp, prices untouched) before ``json.dumps(separators=(',',':'),
allow_nan=False)``. ``server_json`` serialises any model or plain dict frame.

No biological numbers live here; the few constants are wire budgets ``[E]``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Annotated, Final, Iterable, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

__all__ = [
    # constants
    "PROTOCOL_VERSION", "REGION_NAMES", "MOOD_STATES", "FLY_MODES", "INK_STYLE_NAMES", "STAMPS", "SIDES",
    "POKE_STIMS", "MARKET_MODES", "MARKET_SOURCE_NAMES", "CANDLES", "ERROR_CODES", "TICK_EVENT_KINDS",
    "OOB_EVENT_KINDS", "SNAPSHOT_DEADLINE_MS", "MAX_SNAPSHOT_B64", "POS_DP", "ANGLE_DP", "RATE_DP",
    "DRIVE_DP",
    # nested models
    "RasterRow", "Canvas", "HelloConnectome", "RasterBlock", "HelloMarket", "HelloAgent", "HelloFeatures",
    "SimStats", "FlyState", "InkStyle", "MoodState", "Trade", "MarketSnapshot", "TickMarket", "Drives",
    "Spikes", "Rates", "TickEvent",
    # server -> client
    "HelloMsg", "TickMsg", "MoodChangeMsg", "TweetMsg", "MarketMsg", "EventMsg", "SnapshotRequestMsg",
    "PongMsg", "ErrorMsg", "ServerMsg",
    # client -> server
    "PokeMsg", "SetMarketModeMsg", "ClearMsg", "TweetTestMsg", "PingMsg", "SnapshotMsg", "ClientMsg",
    # REST
    "PokeRequest", "MarketModeRequest", "SnapshotUpload", "HealthConnectome", "HealthMarket", "HealthAgent",
    "HealthResponse", "StateResponse",
    # functions
    "BadClientFrame", "parse_client", "parse_client_frame", "tick_to_json", "round_tick", "server_json",
    "wire_fields", "error_frame", "pong_frame", "snapshot_request_frame", "event_frame",
    "SERVER_MODELS", "CLIENT_MODELS",
]

# --------------------------------------------------------------------------------------------------
# wire vocabulary (SPEC d.1 / e.1)
# --------------------------------------------------------------------------------------------------

PROTOCOL_VERSION: Final[int] = 1

REGION_NAMES: Final[tuple[str, ...]] = ("optic_lobe", "antennal_lobe", "mushroom_body", "central_complex",
                                        "sez", "central_other", "descending_motor", "vnc")
MOOD_STATES: Final[tuple[str, ...]] = ("SLEEP", "CRUISING", "FEEDING", "EUPHORIA", "ANXIOUS", "PANIC",
                                       "ESCAPE", "COURTSHIP")
FLY_MODES: Final[tuple[str, ...]] = ("walk", "fly", "jump", "feed", "freeze", "groom", "court", "sleep")
INK_STYLE_NAMES: Final[tuple[str, ...]] = ("solid", "rainbow", "zigzag", "dotted", "hearts")
STAMPS: Final[tuple[str, ...]] = ("blob", "heart", "zzz", "dash", "bump")
SIDES: Final[tuple[str, ...]] = ("L", "R", "M")
POKE_STIMS: Final[tuple[str, ...]] = ("sugar", "bitter", "loom", "water", "dust", "pheromone", "sleep",
                                      "reward", "punish", "explore")
MARKET_MODES: Final[tuple[str, ...]] = ("sim", "dexscreener", "CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD")
MARKET_SOURCE_NAMES: Final[tuple[str, ...]] = ("sim", "dexscreener", "sim(fallback)")
CANDLES: Final[tuple[str, ...]] = ("up", "down", "flat")
ERROR_CODES: Final[tuple[str, ...]] = ("bad_message", "rate_limited", "forbidden")

#: ``tick.events[].kind`` (SPEC d.8). Not a ``Literal``: the field stays ``str`` so an additional kind
#: from a later workstream cannot break a live connection; use this tuple to check a kind explicitly.
TICK_EVENT_KINDS: Final[tuple[str, ...]] = ("jump", "takeoff", "landing", "freeze", "unfreeze", "feed_start",
                                            "feed_stop", "groom", "song", "saccade", "wander_floor", "sleep",
                                            "wake", "wall_bump", "wrap", "gf_spike", "mood")
#: ``event`` frame kinds of SPEC d.8 plus the agent's own ``tweet_suppressed`` / ``tweet_error``.
OOB_EVENT_KINDS: Final[tuple[str, ...]] = ("market_source", "homeostasis", "clear", "poke", "easter_egg",
                                           "whale", "calibration", "tweet_suppressed", "tweet_error")

SNAPSHOT_DEADLINE_MS: Final[int] = 3000          #: [E] browser snapshot deadline (SPEC d.9)
MAX_SNAPSHOT_B64: Final[int] = 3_000_000         #: [E] 2 MB PNG -> ~2.7 MB base64 (SPEC d.7)

POS_DP: Final[int] = 1      #: positions / speeds (SPEC d)
ANGLE_DP: Final[int] = 3    #: angles
RATE_DP: Final[int] = 2     #: rates (Hz)
DRIVE_DP: Final[int] = 3    #: drives / mood scores

_MoodName = Literal["SLEEP", "CRUISING", "FEEDING", "EUPHORIA", "ANXIOUS", "PANIC", "ESCAPE", "COURTSHIP"]
_FlyModeName = Literal["walk", "fly", "jump", "feed", "freeze", "groom", "court", "sleep"]
_InkStyleName = Literal["solid", "rainbow", "zigzag", "dotted", "hearts"]
_StampName = Literal["blob", "heart", "zzz", "dash", "bump"]
_SideName = Literal["L", "R", "M"]
_PokeStim = Literal["sugar", "bitter", "loom", "water", "dust", "pheromone", "sleep", "reward", "punish",
                    "explore"]
_MarketMode = Literal["sim", "dexscreener", "CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD"]
_MarketSource = Literal["sim", "dexscreener", "sim(fallback)"]
_Candle = Literal["up", "down", "flat"]
_ErrorCode = Literal["bad_message", "rate_limited", "forbidden"]
_ConnectomeSource = Literal["synthetic", "csv", "neuprint"]


class _Server(BaseModel):
    """Base of every server -> client model: lenient, alias-aware, snake_case on the wire."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, protected_namespaces=())


class _Client(BaseModel):
    """Base of every client -> server model: ``extra='forbid'`` (SPEC c.26)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


# --------------------------------------------------------------------------------------------------
# d.1 hello
# --------------------------------------------------------------------------------------------------


class RasterRow(_Server):
    """One raster lane row of ``hello.raster.rows``; ``slot`` equals the index in that list."""

    region: int
    slot: int
    neuron: int
    label: str
    side: _SideName
    star: bool


class Canvas(_Server):
    """``hello.canvas``: logical canvas bounds and the boundary rule."""

    w: int
    h: int
    walls: Literal["bounce", "wrap"]


class HelloConnectome(_Server):
    """``hello.connectome``: provenance block (SPEC d.1); ``note`` always labels synthetic data."""

    name: str
    source: _ConnectomeSource
    n: int
    e: int
    synapses: float
    gain: float
    weights_mode: str | None = None
    license: str
    citation: str | None = None
    note: str
    region_counts: dict[str, int]


class RasterBlock(_Server):
    """``hello.raster``: ``8 * per_region`` rows, ``cap`` raster events per tick."""

    per_region: int
    cap: int
    rows: list[RasterRow]


class HelloMarket(_Server):
    """``hello.market``: the configured source, not a snapshot.

    ``token_live`` (``FLY_TOKEN_LIVE``) is the launch disclosure and defaults to **false** in both directions:
    a backend that predates the field omits it, and a client then reads the safe value - the tracked pair is a
    third-party stand-in used only as sensory input and this project's own token has not launched. ``chain`` /
    ``symbol`` / ``token`` plus ``pair`` / ``dex`` are the identity of the pair actually being tracked, so a UI
    showing the numbers can name whose pair they are instead of implying they are this project's.
    """

    mode: str
    chain: str
    symbol: str
    token: str
    poll_s: float
    token_live: bool = False
    pair: str | None = None
    dex: str | None = None


class HelloAgent(_Server):
    """``hello.agent``: tweet plumbing the UI must display (never a credential)."""

    llm: Literal["dryrun", "anthropic"]
    x: Literal["dryrun", "post"]
    cooldown_s: int
    reason_cooldown_s: int
    tweets_per_day: int
    lang: Literal["en", "tr"]


class HelloFeatures(_Server):
    """``hello.features``: the puppeteering / noise knobs of SPEC section b."""

    explore_baseline: float
    wander_sigma: float
    mood_feedback: bool
    easter_eggs: bool
    drive_mode: Literal["poisson", "current"]
    noise_mu: float
    noise_sigma: float


class HelloMsg(_Server):
    """First frame of every connection and ``GET /api/state.hello`` (SPEC d.1)."""

    type: Literal["hello"] = "hello"
    v: int = PROTOCOL_VERSION
    run_id: str
    server_wall: float
    dt_ms: float
    tick_ms: float
    steps_per_tick: int
    tick_hz: int
    realtime: bool
    backend: str
    canvas: Canvas
    connectome: HelloConnectome
    regions: list[str]
    raster: RasterBlock
    pops: list[str]
    mood_states: list[_MoodName]
    channels: list[_PokeStim]
    market_modes: list[_MarketMode]
    market: HelloMarket
    agent: HelloAgent
    features: HelloFeatures


# --------------------------------------------------------------------------------------------------
# d.2 tick
# --------------------------------------------------------------------------------------------------


class SimStats(_Server):
    """``tick.sim``: engine telemetry of this tick (SPEC d.2)."""

    rtf: float
    speed: float
    steps: int
    step_ms: float
    spikes: int
    active_frac: float
    gain: float
    edge_visits: int
    forced: int
    noise: bool
    backend: str


class FlyState(_Server):
    """``tick.fly`` == ``Kinematics.to_wire()``: px, px/s, rad (0 = +x, clockwise positive)."""

    x: float
    y: float
    vx: float
    vy: float
    heading: float
    speed: float
    omega: float
    wing_hz: float
    wing_amp: float
    wing_ext: Literal[-1, 0, 1]
    mode: _FlyModeName
    leg_phase: float
    proboscis: float
    jump_t_ms: int | None = None


class InkStyle(_Server):
    """``tick.ink`` == ``InkStyle.to_wire()``: trail colour / width / alpha / style / stamp."""

    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    width: float
    alpha: float
    style: _InkStyleName
    stamp: _StampName | None = None


class MoodState(_Server):
    """``tick.mood`` == ``MoodState.to_wire()``: scores 0..1, ``valence`` -1..1."""

    state: _MoodName
    prev: _MoodName
    since_ms: int
    euphoria: float
    anxiety: float
    arousal: float
    valence: float
    fear: float
    hunger: float
    sleep: float
    dwell_left_ms: int


class Trade(_Server):
    """One trade as it appears on the wire (``Trade.to_wire()``: no internal ``price`` field)."""

    kind: Literal["buy", "sell"]
    usd: float
    ts: float
    surrogate: bool = False


class MarketSnapshot(_Server):
    """``MarketSnapshot.to_wire()`` (SPEC d.5): DexScreener-shaped, nullable where the pair omits data."""

    source: _MarketSource
    ts: float
    seq: int
    chain: str
    dex: str
    pair: str
    symbol: str
    price_usd: float | None = None
    price_native: float | None = None
    buys_m5: int
    sells_m5: int
    buys_h1: int
    sells_h1: int
    chg_m5: float
    chg_h1: float
    chg_h6: float
    chg_h24: float
    vol_m5: float
    vol_h1: float
    liq_usd: float | None = None
    fdv: float | None = None
    mcap: float | None = None
    regime: str | None = None


class TickMarket(MarketSnapshot):
    """``tick.market``: a snapshot plus the feed ``mode``, the last trade seen this session and the disclosure.

    ``token_live`` repeats ``hello.market.token_live`` on every tick so a client that connects late, misses the
    hello or reconnects still knows whether these numbers belong to this project's token. It defaults to false,
    so a backend that does not send it is read as "not live" (SPEC d.1 / d.2, fail safe).
    """

    mode: _MarketSource
    last_trade: Trade | None = None
    token_live: bool = False


class Drives(_Server):
    """``tick.drives`` == ``Features.to_wire()`` (SPEC f.1): the channels that drive the brain."""

    sugar: float
    bitter: float
    water: float
    looming: float
    loom_side: Literal[-1, 0, 1]
    flash: float
    odor: float
    chop: float
    courtship: float
    sleep_pressure: float
    explore: float
    up: float
    down: float
    activity: float
    hunger: float
    candle: _Candle
    any_max: float


class Spikes(_Server):
    """``tick.spikes``: raster sample of this tick; ``slots`` / ``dt`` are parallel and sorted by ``dt``."""

    t0_ms: int
    win_ms: int
    total: int
    capped: bool = False
    slots: list[int] = Field(default_factory=list)
    dt: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parallel(self) -> "Spikes":
        if len(self.slots) != len(self.dt):
            raise ValueError(f"spikes.slots and spikes.dt must be parallel "
                             f"({len(self.slots)} != {len(self.dt)})")
        return self


class Rates(_Server):
    """``tick.rates``: per-region Hz in ``REGIONS`` order plus the readout table (key order = hello.pops)."""

    regions: list[float]
    pops: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _eight_regions(self) -> "Rates":
        if len(self.regions) != len(REGION_NAMES):
            raise ValueError(f"rates.regions must have {len(REGION_NAMES)} entries, got {len(self.regions)}")
        return self


class TickEvent(_Server):
    """One in-pipeline event of this tick (SPEC d.8 ``tick.events``); ``kind`` in ``TICK_EVENT_KINDS``."""

    kind: str
    t_ms: int
    data: dict[str, Any] = Field(default_factory=dict)


class TickMsg(_Server):
    """One wall tick (SPEC d.2); ``t_ms`` is brain time at the END of the tick."""

    type: Literal["tick"] = "tick"
    seq: int
    t_ms: int
    wall: float
    sim: SimStats
    fly: FlyState
    ink: InkStyle
    mood: MoodState
    market: TickMarket
    drives: Drives
    rates: Rates
    spikes: Spikes
    events: list[TickEvent] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# d.3 - d.5, d.8, d.9 other server frames
# --------------------------------------------------------------------------------------------------


class MoodChangeMsg(_Server):
    """Mood transition frame (SPEC d.3); ``from_`` is the wire key ``from`` (a Python keyword)."""

    type: Literal["mood_change"] = "mood_change"
    seq: int
    t_ms: int
    wall: float
    from_: _MoodName = Field(alias="from")
    to: _MoodName
    reason: str
    mood: MoodState


class TweetMsg(_Server):
    """Agent fire frame, dry-run included (SPEC d.4); the same record is ``GET /api/tweets``."""

    type: Literal["tweet"] = "tweet"
    seq: int
    t_ms: int
    wall: float
    id: str
    reason: str
    text: str
    model: str
    dry_run: bool
    posted: bool
    url: str | None = None
    error: str | None = None
    snapshot_source: Literal["browser", "server"]
    neurons: list[str] = Field(default_factory=list)
    mood: _MoodName
    latency_ms: float


class MarketMsg(_Server):
    """New market snapshot plus the trades consumed since the previous frame (SPEC d.5)."""

    type: Literal["market"] = "market"
    seq: int
    t_ms: int
    wall: float
    mode: _MarketSource
    market: MarketSnapshot
    trades: list[Trade] = Field(default_factory=list)


class EventMsg(_Server):
    """Out-of-band event frame (SPEC d.8); ``kind`` in ``OOB_EVENT_KINDS``."""

    type: Literal["event"] = "event"
    seq: int
    t_ms: int
    wall: float
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class SnapshotRequestMsg(_Server):
    """Ask the most recently active client for a canvas PNG (SPEC d.9)."""

    type: Literal["snapshot_request"] = "snapshot_request"
    id: str
    deadline_ms: int = SNAPSHOT_DEADLINE_MS


class PongMsg(_Server):
    """Answer to a client ``ping`` (SPEC d.9); ``t`` is echoed verbatim for the latency measurement."""

    type: Literal["pong"] = "pong"
    t: float
    server_wall: float
    seq: int


class ErrorMsg(_Server):
    """Protocol error frame; the connection stays open (SPEC d.9)."""

    type: Literal["error"] = "error"
    code: _ErrorCode
    msg: str


ServerMsg = Union[HelloMsg, TickMsg, MoodChangeMsg, TweetMsg, MarketMsg, EventMsg, SnapshotRequestMsg,
                  PongMsg, ErrorMsg]


# --------------------------------------------------------------------------------------------------
# d.7 client frames (extra='forbid')
# --------------------------------------------------------------------------------------------------


class PokeMsg(_Client):
    """``{"type":"poke","stim":"sugar","strength":1.0,"side":"both","duration_ms":500}`` (SPEC d.7)."""

    type: Literal["poke"]
    stim: _PokeStim
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    side: Literal["L", "R", "both"] = "both"
    duration_ms: int = Field(default=500, ge=50, le=5000)

    @property
    def side_int(self) -> int:
        """``Poke.side``: ``L`` -> -1, ``R`` -> +1, ``both`` -> 0 (SPEC d.7)."""
        return {"L": -1, "R": 1, "both": 0}[self.side]

    def to_poke(self, t_ms: int) -> Any:
        """``flybrain.encoder.Poke`` with ``until_ms = t_ms + duration_ms`` (imported lazily)."""
        from flybrain.encoder import Poke  # local import: keeps protocol.py dependency-free

        return Poke(stim=self.stim, strength=float(self.strength), side=self.side_int,
                    until_ms=int(t_ms) + int(self.duration_ms))

    def to_event_data(self) -> dict[str, Any]:
        """``event`` frame payload echoed to every client (SPEC d.8 ``poke``)."""
        return {"stim": self.stim, "strength": float(self.strength), "side": self.side,
                "duration_ms": int(self.duration_ms)}


class SetMarketModeMsg(_Client):
    """``{"type":"set_market_mode","mode":"PUMP"}``: a source name or a forced sim regime (SPEC d.7)."""

    type: Literal["set_market_mode"]
    mode: _MarketMode


class ClearMsg(_Client):
    """``{"type":"clear"}``: reset the server trail ring and broadcast a ``clear`` event."""

    type: Literal["clear"]


class TweetTestMsg(_Client):
    """``{"type":"tweet_test"}``: one manual agent fire (never posts unless ``FLY_X=post``)."""

    type: Literal["tweet_test"]


class PingMsg(_Client):
    """``{"type":"ping","t":1789051563.1}``: answered immediately with ``pong``."""

    type: Literal["ping"]
    t: float = 0.0


class SnapshotMsg(_Client):
    """Reply to ``snapshot_request``: base64 PNG, <= 2 MB decoded (SPEC d.7)."""

    type: Literal["snapshot"]
    id: str = Field(min_length=1, max_length=128)
    png_b64: str = Field(min_length=1, max_length=MAX_SNAPSHOT_B64)


ClientMsg = Union[PokeMsg, SetMarketModeMsg, ClearMsg, TweetTestMsg, PingMsg, SnapshotMsg]

_CLIENT_ADAPTER: Final[TypeAdapter] = TypeAdapter(
    Annotated[ClientMsg, Field(discriminator="type")]
)

CLIENT_MODELS: Final[dict[str, type[BaseModel]]] = {
    "poke": PokeMsg, "set_market_mode": SetMarketModeMsg, "clear": ClearMsg, "tweet_test": TweetTestMsg,
    "ping": PingMsg, "snapshot": SnapshotMsg,
}
SERVER_MODELS: Final[dict[str, type[BaseModel]]] = {
    "hello": HelloMsg, "tick": TickMsg, "mood_change": MoodChangeMsg, "tweet": TweetMsg, "market": MarketMsg,
    "event": EventMsg, "snapshot_request": SnapshotRequestMsg, "pong": PongMsg, "error": ErrorMsg,
}


# --------------------------------------------------------------------------------------------------
# REST bodies and responses (SPEC c.28)
# --------------------------------------------------------------------------------------------------


class PokeRequest(_Client):
    """``POST /api/poke`` body: the same validation as the WS ``poke`` frame, without ``type``."""

    stim: _PokeStim
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    side: Literal["L", "R", "both"] = "both"
    duration_ms: int = Field(default=500, ge=50, le=5000)

    def as_frame(self) -> PokeMsg:
        """The equivalent WS frame (so both entry points share ``to_poke`` / ``to_event_data``)."""
        return PokeMsg(type="poke", stim=self.stim, strength=self.strength, side=self.side,
                       duration_ms=self.duration_ms)


class MarketModeRequest(_Client):
    """``POST /api/market/mode`` body."""

    mode: _MarketMode


class SnapshotUpload(_Client):
    """``POST /api/snapshot`` body (REST alternative to the WS snapshot reply)."""

    id: str = Field(min_length=1, max_length=128)
    png_b64: str = Field(min_length=1, max_length=MAX_SNAPSHOT_B64)


class HealthConnectome(_Server):
    """``GET /api/health.connectome``."""

    name: str
    source: str
    n: int
    e: int
    gain: float
    license: str


class HealthMarket(_Server):
    """``GET /api/health.market`` (``token_live`` defaults to false: an operator curling an old backend must not
    read silence as "the token is live")."""

    mode: str
    ok: bool
    last_poll: float | None = None
    failures: int = 0
    token_live: bool = False


class HealthAgent(_Server):
    """``GET /api/health.agent`` (never a credential, only modes and counters)."""

    llm: str
    x: str
    tweets_today: int
    last_tweet_wall: float | None = None
    disabled_reason: str | None = None


class HealthResponse(_Server):
    """``GET /api/health`` (SPEC c.28)."""

    ok: bool
    uptime_s: float
    run_id: str
    rtf: float
    speed: float
    seq: int
    t_ms: int
    backend: str
    connectome: HealthConnectome
    market: HealthMarket
    agent: HealthAgent
    clients: int
    log_tail: list[str] = Field(default_factory=list)


class StateResponse(_Server):
    """``GET /api/state`` (SPEC c.28): hello, the latest tick, the trail ring and recent events."""

    hello: HelloMsg
    tick: TickMsg | None = None
    trail: list[tuple[float, float, str, float]] = Field(default_factory=list)
    mood_history: list[tuple[int, str]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------------


class BadClientFrame(ValueError):
    """Unparsable / invalid client frame: carries the ``error`` frame the handler answers with.

    A ``ValueError`` so callers written against SPEC c.26 (``parse_client`` "raises ValueError") keep
    working, with ``code`` / ``msg`` / ``to_frame()`` for the handler (SPEC d.9).
    """

    def __init__(self, msg: str, code: str = "bad_message") -> None:
        super().__init__(msg)
        self.code = code if code in ERROR_CODES else "bad_message"
        self.msg = str(msg)

    def to_frame(self) -> dict[str, Any]:
        """``{"type":"error","code":...,"msg":...}``."""
        return {"type": "error", "code": self.code, "msg": self.msg}


def _first_error(exc: ValidationError) -> str:
    """``"poke.stim: Input should be one of ..."``-shaped message from the first pydantic error."""
    try:
        err = exc.errors()[0]
    except Exception:  # noqa: BLE001 - fall back to the raw string
        return str(exc).splitlines()[0]
    loc = ".".join(str(p) for p in err.get("loc", ()) if not str(p).startswith("function-"))
    message = str(err.get("msg", "invalid value"))
    return f"{loc}: {message}" if loc else message


def parse_client(raw: str | bytes | bytearray | Mapping[str, Any]) -> ClientMsg:
    """Validate one client frame into its model (SPEC c.26/d.7).

    Accepts text, bytes or an already decoded mapping; the union is discriminated on ``type`` and every
    client model forbids extra keys. Raises ``BadClientFrame`` (a ``ValueError``) whose ``to_frame()``
    is the ``error`` frame to send back -- the connection stays open (SPEC d.7).
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            text: Any = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BadClientFrame(f"frame is not valid UTF-8: {exc}") from None
    else:
        text = raw
    if isinstance(text, Mapping):
        data: Any = dict(text)
    else:
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise BadClientFrame(f"not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise BadClientFrame(f"expected a JSON object, got {type(data).__name__}")
    kind = data.get("type")
    if not isinstance(kind, str) or not kind:
        raise BadClientFrame("type: missing string discriminator")
    if kind not in CLIENT_MODELS:
        raise BadClientFrame("type: Input should be one of " + ", ".join(sorted(CLIENT_MODELS)))
    try:
        return _CLIENT_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise BadClientFrame(_first_error(exc)) from None


#: SPEC-compatible alias (the task brief calls it ``parse_client_frame``).
parse_client_frame = parse_client


# --------------------------------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------------------------------

_FLY_DP: Final[dict[str, int]] = {"x": POS_DP, "y": POS_DP, "vx": POS_DP, "vy": POS_DP, "speed": POS_DP,
                                  "wing_hz": POS_DP, "heading": ANGLE_DP, "omega": ANGLE_DP,
                                  "wing_amp": DRIVE_DP, "leg_phase": DRIVE_DP, "proboscis": DRIVE_DP}
_SIM_DP: Final[dict[str, int]] = {"rtf": RATE_DP, "speed": DRIVE_DP, "step_ms": DRIVE_DP,
                                  "active_frac": 4, "gain": DRIVE_DP}
_INK_DP: Final[dict[str, int]] = {"width": RATE_DP, "alpha": DRIVE_DP}
_MOOD_DP: Final[dict[str, int]] = {"euphoria": DRIVE_DP, "anxiety": DRIVE_DP, "arousal": DRIVE_DP,
                                   "valence": DRIVE_DP, "fear": DRIVE_DP, "hunger": DRIVE_DP,
                                   "sleep": DRIVE_DP}


def _round_map(src: Mapping[str, Any], dp: Mapping[str, int], default_dp: int | None = None) -> dict:
    """New dict with the named float fields rounded (ints, strings and ``None`` pass through)."""
    out: dict[str, Any] = {}
    for key, value in src.items():
        digits = dp.get(key, default_dp)
        if digits is not None and isinstance(value, float):
            out[key] = round(value, digits)
        else:
            out[key] = value
    return out


def round_tick(tick: Mapping[str, Any]) -> dict:
    """The SPEC d rounding pass: positions/speeds 1 dp, angles 3 dp, rates 2 dp, drives / mood 3 dp.

    Prices and every other ``market`` number keep full precision (SPEC d). The input is never mutated;
    unknown keys are copied through so a richer tick still serialises.
    """
    out: dict[str, Any] = dict(tick)
    wall = out.get("wall")
    if isinstance(wall, float):
        out["wall"] = round(wall, 3)
    for key, table in (("fly", _FLY_DP), ("sim", _SIM_DP), ("ink", _INK_DP), ("mood", _MOOD_DP)):
        value = out.get(key)
        if isinstance(value, Mapping):
            out[key] = _round_map(value, table)
    drives = out.get("drives")
    if isinstance(drives, Mapping):
        out["drives"] = _round_map(drives, {}, DRIVE_DP)
    rates = out.get("rates")
    if isinstance(rates, Mapping):
        new_rates = dict(rates)
        regions = new_rates.get("regions")
        if isinstance(regions, (list, tuple)):
            new_rates["regions"] = [round(float(r), RATE_DP) for r in regions]
        pops = new_rates.get("pops")
        if isinstance(pops, Mapping):
            new_rates["pops"] = {k: round(float(v), RATE_DP) for k, v in pops.items()}
        out["rates"] = new_rates
    return out


def tick_to_json(tick: Mapping[str, Any]) -> str:
    """Serialise one tick exactly once per wall tick (SPEC c.26/d.10).

    ``json.dumps(separators=(',',':'), allow_nan=False)`` over ``round_tick`` -- a NaN or infinity
    anywhere in the tick raises ``ValueError`` instead of emitting JSON the browser cannot parse.
    """
    return json.dumps(round_tick(tick), separators=(",", ":"), allow_nan=False)


def server_json(msg: BaseModel | Mapping[str, Any]) -> str:
    """Serialise any server frame (model or plain dict) with snake_case wire keys.

    Models go through ``model_dump_json(by_alias=True)`` (so ``MoodChangeMsg.from_`` emits ``from``),
    dicts through the same compact ``json.dumps`` as ``tick_to_json``.
    """
    if isinstance(msg, BaseModel):
        return msg.model_dump_json(by_alias=True)
    return json.dumps(dict(msg), separators=(",", ":"), allow_nan=False)


def wire_fields(model: type[BaseModel]) -> set[str]:
    """The wire key set of a model (alias when one is declared) -- used by the types.ts mirror test."""
    out: set[str] = set()
    for name, field in model.model_fields.items():
        alias = getattr(field, "alias", None) or getattr(field, "serialization_alias", None)
        out.add(str(alias) if alias else name)
    return out


# --------------------------------------------------------------------------------------------------
# small frame builders (one source of truth for the key sets app.py sends)
# --------------------------------------------------------------------------------------------------


def error_frame(code: str = "bad_message", msg: str = "") -> dict[str, Any]:
    """``{"type":"error","code":...,"msg":...}`` (SPEC d.9)."""
    return {"type": "error", "code": code if code in ERROR_CODES else "bad_message", "msg": str(msg)}


def pong_frame(t: float, seq: int, server_wall: float | None = None) -> dict[str, Any]:
    """``{"type":"pong","t":...,"server_wall":...,"seq":...}`` (SPEC d.9)."""
    return {"type": "pong", "t": float(t),
            "server_wall": float(time.time() if server_wall is None else server_wall), "seq": int(seq)}


def snapshot_request_frame(id: str, deadline_ms: int = SNAPSHOT_DEADLINE_MS) -> dict[str, Any]:
    """``{"type":"snapshot_request","id":...,"deadline_ms":...}`` (SPEC d.9)."""
    return {"type": "snapshot_request", "id": str(id), "deadline_ms": int(deadline_ms)}


def event_frame(kind: str, data: Mapping[str, Any] | None = None, seq: int = 0, t_ms: int = 0,
                wall: float | None = None) -> dict[str, Any]:
    """``{"type":"event","seq":...,"t_ms":...,"wall":...,"kind":...,"data":{...}}`` (SPEC d.8)."""
    return {"type": "event", "seq": int(seq), "t_ms": int(t_ms),
            "wall": float(time.time() if wall is None else wall), "kind": str(kind),
            "data": dict(data or {})}


def _iter_models(names: Iterable[str]) -> list[type[BaseModel]]:  # pragma: no cover - introspection aid
    table = {**SERVER_MODELS, **CLIENT_MODELS}
    return [table[n] for n in names if n in table]
