// 1:1 mirror of SPEC section d (wire protocol) - copied verbatim from SPEC section e.1.
// backend/tests/test_protocol.py::test_types_ts_mirror diffs the interface field names against the pydantic models.
export type Region = "optic_lobe"|"antennal_lobe"|"mushroom_body"|"central_complex"|"sez"|"central_other"|"descending_motor"|"vnc";
export const REGIONS: readonly Region[] = ["optic_lobe","antennal_lobe","mushroom_body","central_complex","sez","central_other","descending_motor","vnc"];
export type Mood = "SLEEP"|"CRUISING"|"FEEDING"|"EUPHORIA"|"ANXIOUS"|"PANIC"|"ESCAPE"|"COURTSHIP";
export type FlyMode = "walk"|"fly"|"jump"|"feed"|"freeze"|"groom"|"court"|"sleep";
export type InkStyleName = "solid"|"rainbow"|"zigzag"|"dotted"|"hearts";
export type Stamp = null|"blob"|"heart"|"zzz"|"dash"|"bump";
export type Side = "L"|"R"|"M";
export type PokeStim = "sugar"|"bitter"|"loom"|"water"|"dust"|"pheromone"|"sleep"|"reward"|"punish"|"explore";
export type MarketMode = "sim"|"dexscreener"|"CALM"|"PUMP"|"DUMP"|"CHOP"|"RUG"|"DEAD";
export type Candle = "up"|"down"|"flat";

export interface RasterRow { region: number; slot: number; neuron: number; label: string; side: Side; star: boolean }

export interface HelloMsg {
  type: "hello"; v: 1; run_id: string; server_wall: number;
  dt_ms: number; tick_ms: number; steps_per_tick: number; tick_hz: number; realtime: boolean; backend: string;
  canvas: { w: number; h: number; walls: "bounce"|"wrap" };
  connectome: { name: string; source: "synthetic"|"csv"|"neuprint"; n: number; e: number; synapses: number; gain: number;
                weights_mode: string|null; license: string; citation: string|null; note: string; region_counts: Record<Region, number> };
  regions: Region[];
  raster: { per_region: number; cap: number; rows: RasterRow[] };
  pops: string[];
  mood_states: Mood[];
  channels: PokeStim[];
  market_modes: MarketMode[];
  market: { mode: "sim"|"dexscreener"|"sim(fallback)"; chain: string; symbol: string; token: string; poll_s: number };
  agent: { llm: "dryrun"|"anthropic"; x: "dryrun"|"post"; cooldown_s: number; reason_cooldown_s: number; tweets_per_day: number; lang: "en"|"tr" };
  features: { explore_baseline: number; wander_sigma: number; mood_feedback: boolean; easter_eggs: boolean;
              drive_mode: "poisson"|"current"; noise_mu: number; noise_sigma: number };
}

export interface SimStats { rtf: number; speed: number; steps: number; step_ms: number; spikes: number; active_frac: number;
  gain: number; edge_visits: number; forced: number; noise: boolean; backend: string }
export interface FlyState { x: number; y: number; vx: number; vy: number; heading: number; speed: number; omega: number;
  wing_hz: number; wing_amp: number; wing_ext: -1|0|1; mode: FlyMode; leg_phase: number; proboscis: number; jump_t_ms: number|null }
export interface InkStyle { color: string; width: number; alpha: number; style: InkStyleName; stamp: Stamp }
export interface MoodState { state: Mood; prev: Mood; since_ms: number; euphoria: number; anxiety: number; arousal: number;
  valence: number; fear: number; hunger: number; sleep: number; dwell_left_ms: number }
export interface Trade { kind: "buy"|"sell"; usd: number; ts: number; surrogate: boolean }
export interface MarketSnapshot { source: "sim"|"dexscreener"|"sim(fallback)"; ts: number; seq: number; chain: string; dex: string;
  pair: string; symbol: string; price_usd: number|null; price_native: number|null; buys_m5: number; sells_m5: number;
  buys_h1: number; sells_h1: number; chg_m5: number; chg_h1: number; chg_h6: number; chg_h24: number; vol_m5: number; vol_h1: number;
  liq_usd: number|null; fdv: number|null; mcap: number|null; regime: string|null }
export interface TickMarket extends MarketSnapshot { mode: "sim"|"dexscreener"|"sim(fallback)"; last_trade: Trade|null }
export interface Drives { sugar: number; bitter: number; water: number; looming: number; loom_side: -1|0|1; flash: number; odor: number;
  chop: number; courtship: number; sleep_pressure: number; explore: number; up: number; down: number; activity: number;
  hunger: number; candle: Candle; any_max: number }
export interface Spikes { t0_ms: number; win_ms: number; total: number; capped: boolean; slots: number[]; dt: number[] }
export interface TickEvent { kind: TickEventKind; t_ms: number; data: Record<string, unknown> }
export type TickEventKind = "jump"|"takeoff"|"landing"|"freeze"|"unfreeze"|"feed_start"|"feed_stop"|"groom"|"song"|"saccade"
  |"wander_floor"|"sleep"|"wake"|"wall_bump"|"wrap"|"gf_spike"|"mood";
export interface TickMsg {
  type: "tick"; seq: number; t_ms: number; wall: number;
  sim: SimStats; fly: FlyState; ink: InkStyle; mood: MoodState; market: TickMarket; drives: Drives;
  rates: { regions: number[]; pops: Record<string, number> };
  spikes: Spikes; events: TickEvent[];
}
export interface MoodChangeMsg { type: "mood_change"; seq: number; t_ms: number; wall: number; from: Mood; to: Mood; reason: string; mood: MoodState }
export interface TweetMsg { type: "tweet"; seq: number; t_ms: number; wall: number; id: string; reason: string; text: string; model: string;
  dry_run: boolean; posted: boolean; url: string|null; error: string|null; snapshot_source: "browser"|"server"; neurons: string[];
  mood: Mood; latency_ms: number }
export interface MarketMsg { type: "market"; seq: number; t_ms: number; wall: number; mode: "sim"|"dexscreener"|"sim(fallback)";
  market: MarketSnapshot; trades: Trade[] }
export type OobEventKind = "market_source"|"homeostasis"|"clear"|"poke"|"easter_egg"|"whale"|"calibration";
export interface EventMsg { type: "event"; seq: number; t_ms: number; wall: number; kind: OobEventKind; data: Record<string, unknown> }
export interface SnapshotRequestMsg { type: "snapshot_request"; id: string; deadline_ms: number }
export interface PongMsg { type: "pong"; t: number; server_wall: number; seq: number }
export interface ErrorMsg { type: "error"; code: "bad_message"|"rate_limited"|"forbidden"; msg: string }
export type ServerMsg = HelloMsg|TickMsg|MoodChangeMsg|TweetMsg|MarketMsg|EventMsg|SnapshotRequestMsg|PongMsg|ErrorMsg;

export type ClientMsg =
  | { type: "poke"; stim: PokeStim; strength: number; side: "L"|"R"|"both"; duration_ms: number }
  | { type: "set_market_mode"; mode: MarketMode }
  | { type: "clear" }
  | { type: "tweet_test" }
  | { type: "ping"; t: number }
  | { type: "snapshot"; id: string; png_b64: string };

// REST shapes (lib/api.ts)
export interface TweetRecord extends Omit<TweetMsg, "type"|"seq"> { summary?: Record<string, unknown> }
export interface StateResponse { hello: HelloMsg; tick: TickMsg|null; trail: [number, number, string, number][];
  mood_history: [number, Mood][]; events: (TickEvent|EventMsg)[] }
export interface HealthResponse { ok: boolean; uptime_s: number; run_id: string; rtf: number; speed: number; seq: number; t_ms: number;
  backend: string; connectome: { name: string; source: string; n: number; e: number; gain: number; license: string };
  market: { mode: string; ok: boolean; last_poll: number|null; failures: number };
  agent: { llm: string; x: string; tweets_today: number; last_tweet_wall: number|null; disabled_reason: string|null };
  clients: number; log_tail: string[] }
