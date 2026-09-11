"use client";
// SPEC section e.5 `MarketTicker`: symbol + source badge + regime chip, price (subscript-zero notation, monospace),
// chg_m5/h1/h24 with up/down glyphs, buys-vs-sells bar, vol/liq/mcap (compact), last trade with age (~ = surrogate),
// a 5-minute price chart (the section e.5 line sparkline, or 15 s candles via the line/candle buttons) and a 12-row
// trade tape, all built client-side from `market` frames (section d.5; buffered <= 600 trades / 5 min per section e.8),
// plus the Options > Market regime buttons as .btn95 chips and the encoder's candle sign from `tick.drives.candle`.
// `market` frames arrive over the socket bus (~1 Hz sim, ~1/60 Hz DexScreener) and are folded into a small external
// store; the panel re-renders through useSyncExternalStore at <= 4 Hz. Nothing here calls setState from a tick handler.
import { useEffect, useState, useSyncExternalStore } from "react";
import type { FlySocket } from "@/lib/ws";
import type { TickStore } from "@/lib/store";
import type { Candle, MarketMode, MarketMsg, MarketSnapshot, TickMarket, TickMsg, Trade } from "@/lib/types";
import { fmtCompact, fmtPrice, postMarketMode } from "@/lib/api";

export interface MarketTickerProps { sock: FlySocket }

const SPARK_WINDOW_S = 300;     // 5-minute sparkline
const MAX_POINTS = 600;         // price samples kept (sim: ~1 Hz -> 300 in 5 min)
const MAX_TRADES = 600;         // section e.8: <= 600 trades / 5 min
const TAPE_N = 12;              // trade tape rows
const NOTIFY_MS = 250;          // 4 Hz re-render budget
const PENDING_MS = 2000;        // how long a clicked regime chip stays pressed without a server confirmation
const CANDLE_S = 15;            // one candle per 15 s -> 20 candles over the 5 min window
const DEFAULT_MODES: readonly MarketMode[] = ["CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD", "sim", "dexscreener"];
const REGIME_MODES: ReadonlySet<string> = new Set(["CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD"]);
const UP = "#008000", DOWN = "#a80000", FLAT = "#606060";

// ------------------------------------------------------------------------------------------------ formatting
// fmtPrice (subscript-zero notation) and fmtCompact live in lib/api.ts so the status bar, the fly overlay and this
// panel cannot drift apart; only the ticker-local helpers are defined here.

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  return `${v >= 0 ? "+" : ""}${v.toFixed(Math.abs(v) >= 100 ? 0 : 1)}%`;
}

function fmtClock(ts: number): string {
  if (!Number.isFinite(ts)) return "--:--:--";
  const d = new Date(ts * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function fmtAge(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "now";
  if (s < 60) return `${s.toFixed(0)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function sourceBadge(src: string | null | undefined): string {
  if (src === "dexscreener") return "DEX";
  if (src === "sim(fallback)") return "SIM(fallback)";
  return "SIM";
}

// ------------------------------------------------------------------------------------------------ history store

export interface PricePoint { ts: number; p: number }
export interface MarketSnap {
  version: number;
  last: MarketSnapshot | null;             // latest snapshot (market frame or tick copy, whichever is newer)
  mode: string | null;                     // MarketFeed.mode of the latest frame
  points: readonly PricePoint[];           // last 5 min of prices, oldest first
  trades: readonly Trade[];                // last <= 600 trades, oldest first
  frames: number;                          // market frames received this session
  tick: TickMsg | null;                    // latest 4 Hz tick snapshot (wall clock, drives.candle)
}

const EMPTY_SNAP: MarketSnap = { version: 0, last: null, mode: null, points: [], trades: [], frames: 0, tick: null };

/**
 * External store for useSyncExternalStore: folds `market` frames (bus) and the `tick.market` convenience copy
 * (store, 4 Hz) into one price ring + trade ring, and carries the latest tick. Both inputs share one 250 ms notify
 * throttle, so the panel re-renders at <= 4 Hz (section e.0) even though it is fed from two sources.
 */
export class MarketHistory {
  private readonly listeners = new Set<() => void>();
  private points: PricePoint[] = [];
  private trades: Trade[] = [];
  private last: MarketSnapshot | null = null;
  private mode: string | null = null;
  private frames = 0;
  private tick: TickMsg | null = null;
  private tickSeq = -1;
  private snap: MarketSnap = EMPTY_SNAP;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastNotify = -Infinity;

  constructor(private readonly store: TickStore) {}

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    const off = this.store.subscribe(() => this.ingestTick(this.store.snapshot()));
    this.ingestTick(this.store.snapshot());
    return () => { this.listeners.delete(fn); off(); };
  };

  getSnapshot = (): MarketSnap => this.snap;
  getServerSnapshot = (): MarketSnap => EMPTY_SNAP;

  /** A `market` frame (section d.5): snapshot + the trades consumed since the previous frame. */
  pushFrame(m: MarketMsg): void {
    try {
      this.frames += 1;
      this.mode = m.mode;
      this.addSnapshot(m.market);
      if (Array.isArray(m.trades) && m.trades.length) {
        for (const t of m.trades) {
          if (t && (t.kind === "buy" || t.kind === "sell") && Number.isFinite(t.usd)) this.trades.push(t);
        }
        if (this.trades.length > MAX_TRADES) this.trades.splice(0, this.trades.length - MAX_TRADES);
      }
      this.schedule();
    } catch (e) {
      console.error("[ticker] market frame failed", e);
    }
  }

  ingestTick(tick: TickMsg | null): void {
    if (!tick || tick.seq === this.tickSeq) return;
    try {
      this.tick = tick;
      this.tickSeq = tick.seq;
      const tm: TickMarket | null = tick.market ?? null;
      if (tm && this.acceptsCopy(tm)) {
        this.mode = tm.mode ?? this.mode;
        this.addSnapshot(tm);
      }
      this.schedule();
    } catch (e) {
      console.error("[ticker] tick ingest failed", e);
    }
  }

  /** Drops the stale `tick.market` copies of snapshots already folded in, without blocking a restarted server. */
  private acceptsCopy(tm: TickMarket): boolean {
    const last = this.last;
    if (!last) return true;
    if (tm.seq === last.seq) return false;        // same snapshot (the `market` frame usually arrives first)
    if (tm.seq > last.seq) return true;
    // `seq` went backwards: a new MarketFeed (server restart, section d.1 "a different run_id means a new session")
    // re-stamps seq from 1, so only a fresher wall clock distinguishes it from an out-of-order copy.
    return tm.ts > last.ts;
  }

  /** Forgets the previous session (new `run_id`): price ring, trade tape, regime and source all start over. */
  reset(): void {
    this.points = [];
    this.trades = [];
    this.last = null;
    this.mode = null;
    this.frames = 0;
    this.tick = null;
    this.tickSeq = -1;
    this.schedule();
  }

  private addSnapshot(s: MarketSnapshot): void {
    if (this.last && s.seq === this.last.seq && s.ts === this.last.ts) return;
    const p = s.price_usd ?? s.price_native;
    if (p !== null && p !== undefined && Number.isFinite(p) && Number.isFinite(s.ts)) {
      this.points.push({ ts: s.ts, p });
      const cutoff = s.ts - SPARK_WINDOW_S;
      let drop = 0;
      while (drop < this.points.length - 1 && this.points[drop].ts < cutoff) drop++;
      if (this.points.length - drop > MAX_POINTS) drop = this.points.length - MAX_POINTS;
      if (drop > 0) this.points.splice(0, drop);
    }
    this.last = s;
  }

  private schedule(): void {
    if (this.timer !== null) return;
    const wait = Math.max(0, NOTIFY_MS - (performance.now() - this.lastNotify));
    this.timer = setTimeout(() => this.notify(), wait);
  }

  private notify(): void {
    this.timer = null;
    this.lastNotify = performance.now();
    this.snap = {
      version: this.snap.version + 1, last: this.last, mode: this.mode,
      points: this.points.slice(), trades: this.trades.slice(-TAPE_N), frames: this.frames, tick: this.tick,
    };
    for (const fn of this.listeners) {
      try { fn(); } catch (e) { console.error("[ticker] listener failed", e); }
    }
  }
}

// ------------------------------------------------------------------------------------------------ widgets

function Sparkline({ points, w = 240, h = 36 }: { points: readonly PricePoint[]; w?: number; h?: number }) {
  const n = points.length;
  let lo = Infinity, hi = -Infinity;
  for (const q of points) { if (q.p < lo) lo = q.p; if (q.p > hi) hi = q.p; }
  const span = hi - lo;
  const t0 = n ? points[0].ts : 0;
  const t1 = n ? points[n - 1].ts : 1;
  const tspan = Math.max(1e-6, Math.max(t1 - t0, n > 1 ? 1 : 0));
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = n > 1 ? ((points[i].ts - t0) / tspan) * (w - 2) + 1 : w - 1;
    const y = span > 0 ? h - 2 - ((points[i].p - lo) / span) * (h - 4) : h / 2;
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const up = n > 1 ? points[n - 1].p >= points[0].p : true;
  const stroke = up ? "#008000" : "#a80000";
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="block h-[36px] w-full bg-white" preserveAspectRatio="none" aria-label="5 minute price sparkline" role="img">
      <line x1={0} y1={h / 2} x2={w} y2={h / 2} stroke="#e0e0e0" strokeWidth={1} />
      {n >= 2 ? <polyline points={pts.join(" ")} fill="none" stroke={stroke} strokeWidth={1.5} vectorEffect="non-scaling-stroke" /> : null}
      {n === 1 ? <circle cx={w - 1} cy={h / 2} r={1.5} fill={stroke} /> : null}
    </svg>
  );
}

export interface CandleBar { t0: number; o: number; h: number; l: number; c: number }

/** Buckets the 5-minute price ring into `bucketS`-second candles, oldest first (<= 20 bars at CANDLE_S). */
export function buildCandles(points: readonly PricePoint[], bucketS: number = CANDLE_S): CandleBar[] {
  const out: CandleBar[] = [];
  const step = bucketS > 0 ? bucketS : CANDLE_S;
  for (const q of points) {
    if (!Number.isFinite(q.p) || !Number.isFinite(q.ts)) continue;
    const t0 = Math.floor(q.ts / step) * step;
    const last = out.length ? out[out.length - 1] : null;
    if (last && last.t0 === t0) {
      last.c = q.p;
      if (q.p > last.h) last.h = q.p;
      if (q.p < last.l) last.l = q.p;
    } else {
      out.push({ t0, o: q.p, h: q.p, l: q.p, c: q.p });
    }
  }
  return out;
}

function candleColor(o: number, c: number): string {
  return c > o ? UP : c < o ? DOWN : FLAT;
}

/** Candle chart of the same 5-minute ring: one body per CANDLE_S seconds, right-aligned (newest at the right edge). */
function Candles({ bars: all, w = 240, h = 36 }: { bars: readonly CandleBar[]; w?: number; h?: number }) {
  const slots = Math.max(8, Math.ceil(SPARK_WINDOW_S / CANDLE_S));      // 20 slots -> stable bar width
  // The 5-minute ring spans the full 300 s, which straddles 21 buckets whenever it is full: keep the newest `slots`
  // bars so the oldest one is never placed at a negative x (clipped away).
  const bars = all.length > slots ? all.slice(all.length - slots) : all;
  const n = bars.length;
  let lo = Infinity, hi = -Infinity;
  for (const b of bars) { if (b.l < lo) lo = b.l; if (b.h > hi) hi = b.h; }
  const span = hi - lo;
  const yOf = (p: number) => (span > 0 ? h - 2 - ((p - lo) / span) * (h - 4) : h / 2);
  const step = w / slots;
  const bw = Math.max(2, step - 1);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="block h-[36px] w-full bg-white" preserveAspectRatio="none" aria-label="5 minute candles (15 s)" role="img">
      <line x1={0} y1={h / 2} x2={w} y2={h / 2} stroke="#e0e0e0" strokeWidth={1} />
      {bars.map((b, i) => {
        const x = w - (n - i) * step;
        const col = candleColor(b.o, b.c);
        const top = Math.min(yOf(b.o), yOf(b.c));
        const bot = Math.max(yOf(b.o), yOf(b.c));
        return (
          <g key={`${b.t0}-${i}`} fill={col} stroke="none">
            <rect x={x + bw / 2 - 0.5} y={yOf(b.h)} width={1} height={Math.max(1, yOf(b.l) - yOf(b.h))} />
            <rect x={x} y={top} width={bw} height={Math.max(1, bot - top)} />
          </g>
        );
      })}
      {n === 0 ? <text x={2} y={h - 3} fontSize={9} fill="#808080">no candles yet</text> : null}
    </svg>
  );
}

const CANDLE_GLYPH: Record<Candle, string> = { up: "▲", down: "▼", flat: "=" };
const CANDLE_COLOR: Record<Candle, string> = { up: UP, down: DOWN, flat: FLAT };

function Pct({ label, v }: { label: string; v: number | null | undefined }) {
  const ok = v !== null && v !== undefined && Number.isFinite(v);
  const up = ok && (v as number) >= 0;
  return (
    <span className="whitespace-nowrap font-mono" style={{ color: ok ? (up ? "#008000" : "#a80000") : "#404040" }} title={`${label}: ${ok ? v : "-"}`}>
      {ok ? (up ? "▲" : "▼") : "■"} {fmtPct(v)} <span className="text-[#404040]">{label}</span>
    </span>
  );
}

// ------------------------------------------------------------------------------------------------ panel

export function MarketTicker({ sock }: MarketTickerProps) {
  const hello = sock.hello;
  const on = sock.on;
  const send = sock.send;
  const [hist] = useState(() => new MarketHistory(sock.store));
  const snap = useSyncExternalStore(hist.subscribe, hist.getSnapshot, hist.getServerSnapshot);
  const tick = snap.tick;                            // 4 Hz snapshot (section e.5), same notify throttle as the rings
  const [pending, setPending] = useState<MarketMode | null>(null);
  const [chart, setChart] = useState<"line" | "candle">("line");

  // market frames from the socket bus (the `on` callback is stable across socket meta updates)
  useEffect(() => on("market", (m) => hist.pushFrame(m)), [on, hist]);

  // a new run_id is a new session (section d.1): the previous server's price ring, trade tape and regime are dropped
  const runId = hello?.run_id ?? null;
  useEffect(() => { hist.reset(); }, [runId, hist]);

  const m = snap.last;
  // wall reference for trade ages: the server's publish time of the latest 4 Hz tick snapshot (never the client clock)
  const nowWall = tick ? tick.wall : (m ? m.ts : 0);
  const lastTrade: Trade | null = snap.trades.length ? snap.trades[snap.trades.length - 1] : (tick?.market.last_trade ?? null);
  const symbol = m?.symbol ?? hello?.market.symbol ?? "FLY";
  const source = m?.source ?? hello?.market.mode ?? "sim";
  const regime = m?.regime ?? null;
  const price = m ? (m.price_usd ?? m.price_native) : null;
  const buys = m?.buys_m5 ?? 0, sells = m?.sells_m5 ?? 0;
  const total = buys + sells;
  const buyFrac = total > 0 ? buys / total : 0.5;
  const candle: Candle | null = tick?.drives.candle ?? null;
  const modes: readonly MarketMode[] = hello?.market_modes?.length ? hello.market_modes : DEFAULT_MODES;
  const tokenSet = Boolean(hello?.market.token);

  /** True when the server state already shows `mode` (regime forced / source switched). */
  const serverHas = (mode: MarketMode): boolean => {
    if (REGIME_MODES.has(mode)) return regime === mode;
    if (mode === "sim") return source === "sim" || source === "sim(fallback)";
    if (mode === "dexscreener") return source === "dexscreener";
    return false;
  };

  // The clicked chip stops being "pending" as soon as the server state confirms it (checked during render, below) and
  // in any case after PENDING_MS. The timeout is keyed on `pending` ONLY: adding the snapshot version (or any other
  // ~1 Hz market signal) to the deps makes React clear and re-arm the timer on every market frame, so it never fires
  // and the clicked chip stays pressed for the lifetime of the mount.
  const confirmed = pending !== null && serverHas(pending);
  useEffect(() => {
    if (pending === null) return;
    const t = setTimeout(() => setPending(null), PENDING_MS);
    return () => clearTimeout(t);
  }, [pending]);

  const setMode = (mode: MarketMode) => {
    setPending(mode);
    if (!send({ type: "set_market_mode", mode })) {
      postMarketMode(mode).catch((e: unknown) => console.warn("[ticker] set_market_mode via REST failed", e));
    }
  };

  // Once the request is confirmed the server state alone drives the chips (no chip can look pressed by a stale click).
  const chipPressed = (mode: MarketMode): boolean => serverHas(mode) || (!confirmed && pending === mode);

  return (
    <div className="flex h-full flex-col gap-1 overflow-auto bg-win-gray p-1 text-[11px]" data-testid="market-ticker">
      {/* header: symbol, source badge, regime chip */}
      <div className="flex items-center gap-1">
        <span className="text-[13px] font-bold">${symbol}</span>
        <span className="border border-black px-1 text-[9px] font-bold" style={{ background: source === "dexscreener" ? "#000080" : source === "sim(fallback)" ? "#a80000" : "#008080", color: "#fff" }} title={`market source: ${source}`} data-testid="source-badge">
          {sourceBadge(source)}
        </span>
        {regime ? (
          <span className="border border-black bg-white px-1 text-[9px] font-bold" title="simulated market regime" data-testid="regime-chip">{regime}</span>
        ) : null}
        {m ? <span className="ml-auto text-[#404040]" title={`${m.dex} ${m.pair} ${m.chain}`}>{m.chain}/{m.dex}</span> : <span className="ml-auto text-[#404040]">waiting for market</span>}
      </div>

      {/* price + changes */}
      <div className="font-mono text-[16px] font-bold leading-[18px]" data-testid="price">
        ${fmtPrice(price)}
      </div>
      <div className="flex flex-wrap gap-x-2 gap-y-0">
        <Pct label="m5" v={m?.chg_m5} />
        <Pct label="h1" v={m?.chg_h1} />
        <Pct label="h24" v={m?.chg_h24} />
      </div>

      {/* buys vs sells (m5) */}
      <div className="flex items-center gap-1" title={`buys ${buys} / sells ${sells} in the last 5 min`} data-testid="buys-sells">
        <span className="w-[44px] font-mono text-[10px] text-[#008000]">{buys} buy</span>
        <span className="bevel-in relative h-[9px] flex-1 overflow-hidden bg-white">
          <span className="absolute left-0 top-0 h-full" style={{ width: `${(buyFrac * 100).toFixed(1)}%`, background: "#00a800" }} />
          <span className="absolute right-0 top-0 h-full" style={{ width: `${((1 - buyFrac) * 100).toFixed(1)}%`, background: "#a80000" }} />
        </span>
        <span className="w-[44px] text-right font-mono text-[10px] text-[#a80000]">{sells} sell</span>
      </div>

      {/* volume / liquidity / mcap */}
      <div className="flex flex-wrap gap-x-2 text-[#404040]">
        <span>vol m5 <b className="font-mono text-black">{fmtCompact(m?.vol_m5)}</b></span>
        <span>liq <b className="font-mono text-black">{fmtCompact(m?.liq_usd)}</b></span>
        <span>mcap <b className="font-mono text-black">{fmtCompact(m?.mcap ?? m?.fdv)}</b></span>
      </div>

      {/* last trade */}
      <div className="flex items-center gap-1" data-testid="last-trade">
        <span className="text-[#404040]">last</span>
        {lastTrade ? (
          <>
            <span className="font-mono font-bold" style={{ color: lastTrade.kind === "buy" ? "#008000" : "#a80000" }}>
              {lastTrade.surrogate ? "~" : ""}{lastTrade.kind.toUpperCase()}
            </span>
            <span className="font-mono">{fmtCompact(lastTrade.usd)}</span>
            <span className="text-[#404040]">{fmtAge(nowWall - lastTrade.ts)}</span>
            {lastTrade.surrogate ? <span className="text-[9px] text-[#404040]" title="rate-matched surrogate trade between DexScreener polls">surrogate</span> : null}
          </>
        ) : <span className="text-[#404040]">none yet</span>}
      </div>

      {/* 5-minute chart from the `market` frames: line sparkline (default) or 15 s candles */}
      <div className="flex items-center gap-1">
        <span className="text-[#404040]">5 min</span>
        {candle ? (
          <span className="font-mono" style={{ color: CANDLE_COLOR[candle] }} title={`encoder candle sign (up/down drives, section f.1): ${candle}`} data-testid="candle-sign">
            {CANDLE_GLYPH[candle]} {candle}
          </span>
        ) : null}
        <span className="ml-auto flex gap-[2px]">
          {(["line", "candle"] as const).map((k) => (
            <button key={k} type="button" className="btn95 text-[9px] leading-[12px]" aria-pressed={chart === k}
              title={k === "line" ? "price line, 5 min" : `candles, ${CANDLE_S} s each`} onClick={() => setChart(k)}>
              {k}
            </button>
          ))}
        </span>
      </div>
      <div className="bevel-in p-0" title={`${snap.points.length} price samples, last 5 min`} data-testid="sparkline">
        {chart === "line" ? <Sparkline points={snap.points} /> : <Candles bars={buildCandles(snap.points)} />}
      </div>

      {/* trade tape */}
      <table className="w-full border-collapse font-mono text-[10px]" data-testid="trade-tape">
        <thead>
          <tr className="text-left text-[#404040]"><th className="font-normal">time</th><th className="font-normal">side</th><th className="text-right font-normal">usd</th></tr>
        </thead>
        <tbody>
          {snap.trades.length === 0 ? (
            <tr><td colSpan={3} className="text-[#404040]">no trades yet ({snap.frames} market frames)</td></tr>
          ) : (
            snap.trades.slice().reverse().map((t, i) => (
              <tr key={`${t.ts}-${i}`} style={{ color: t.kind === "buy" ? "#008000" : "#a80000" }}>
                <td>{fmtClock(t.ts)}</td>
                <td>{t.surrogate ? "~" : ""}{t.kind}</td>
                <td className="text-right">{fmtCompact(t.usd)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>

      {/* regime chips (= Options > Market regime) */}
      <div className="mt-auto flex flex-wrap gap-[2px] pt-[2px]" data-testid="regime-chips">
        {modes.map((mode) => {
          const disabled = mode === "dexscreener" && !tokenSet;
          return (
            <button
              key={mode}
              type="button"
              className="btn95 text-[9px] leading-[14px] disabled:text-[#808080]"
              aria-pressed={chipPressed(mode)}
              disabled={disabled}
              title={disabled ? "FLY_TOKEN_ADDRESS is empty (dexscreener refused)" : REGIME_MODES.has(mode) ? `force the simulated regime ${mode} for 60 s` : `switch the market source to ${mode}`}
              onClick={() => setMode(mode)}
            >
              {mode === "dexscreener" ? "dex" : mode}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default MarketTicker;
