"use client";
// SPEC section e.5 `MoodPanel`: 4 Hz snapshot of the mood machine (section f.6), drives (section f.1) and the
// population readouts. Row 1 mood + pixel face + timers, row 2 Win95 segmented score bars with the f.6 threshold
// ticks, row 3 drive meters, row 4 a 300 x 12 px mood timeline (last 5 min), row 5 the population table with 8 s
// sparklines, footer with the run/connectome facts. History (sparklines, timeline) lives in a small external store
// fed by the tick store's 4 Hz notifications, so nothing here calls setState from a tick handler.
import { useEffect, useState, useSyncExternalStore } from "react";
import type { FlySocket } from "@/lib/ws";
import type { TickStore } from "@/lib/store";
import type { Drives, Mood, TickMsg } from "@/lib/types";
import { moodColor } from "@/lib/ink";

export interface MoodPanelProps { sock: FlySocket }

/** The 8 moods in hello.mood_states order (section d.1). */
export const MOOD_LIST: readonly Mood[] = ["SLEEP", "CRUISING", "FEEDING", "EUPHORIA", "ANXIOUS", "PANIC", "ESCAPE", "COURTSHIP"];

/** Visitor-friendly tooltip per mood (lights up when the fly is in that state). */
const MOOD_TIP: Record<Mood, string> = {
  SLEEP: "SLEEP — flat, dead market, the fly dozes off (lights up when active)",
  CRUISING: "CRUISING — calm market, the fly just wanders and doodles",
  FEEDING: "FEEDING — a buy fed its sugar neurons; it stops to eat",
  EUPHORIA: "EUPHORIA — strong buying; the fly is loving it, loops and hearts",
  ANXIOUS: "ANXIOUS — selling pressure building; it gets jittery",
  PANIC: "PANIC — heavy selling; looming + escape circuits fire, it darts",
  ESCAPE: "ESCAPE — a giant-fiber jump; the fly bolted",
  COURTSHIP: "COURTSHIP — the male-specific song circuit lit up",
};
/** Population table rows (section e.5). */
export const PANEL_POPS: readonly string[] = ["gf_L", "gf_R", "dn_freeze", "dng100", "steer_a02_L", "steer_a02_R", "feed_mn", "flight_dn", "p1", "pam", "ppl1"];
const SPARK_N = 32;            // 8 s at 4 Hz
const TIMELINE_W = 300;        // px = seconds (last 5 min, one column per second)
const DRIVE_KEYS: readonly (keyof Drives)[] = ["sugar", "bitter", "looming", "flash", "odor", "chop", "courtship", "sleep_pressure", "explore"];
/** Section f.6 thresholds drawn as ticks on the score bars. */
const TICKS_EUPHORIA = [0.40, 0.70];
const TICKS_ANXIETY = [0.35, 0.45, 0.70];

// ------------------------------------------------------------------------------------------------ history store

export interface PanelSnap { tick: TickMsg | null; sparks: Record<string, number[]>; timeline: (Mood | null)[]; version: number }

const EMPTY_SNAP: PanelSnap = { tick: null, sparks: {}, timeline: [], version: 0 };

/** External store for useSyncExternalStore: rings of the last 8 s of population rates and the 5-min mood timeline. */
class PanelHistory {
  private readonly listeners = new Set<() => void>();
  private readonly sparks: Record<string, number[]> = {};
  private readonly timeline: (Mood | null)[] = new Array<Mood | null>(TIMELINE_W).fill(null);
  private lastSec: number | null = null;
  private lastSeq = -1;
  private snap: PanelSnap;

  constructor(private readonly store: TickStore) {
    for (const p of PANEL_POPS) this.sparks[p] = [];
    this.snap = { tick: null, sparks: this.sparks, timeline: this.timeline, version: 0 };
  }

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    const off = this.store.subscribe(() => this.ingest(this.store.snapshot()));
    this.ingest(this.store.snapshot());
    return () => { this.listeners.delete(fn); off(); };
  };

  getSnapshot = (): PanelSnap => this.snap;
  getServerSnapshot = (): PanelSnap => EMPTY_SNAP;

  ingest(tick: TickMsg | null): void {
    if (!tick || tick.seq === this.lastSeq) return;
    try {
      this.lastSeq = tick.seq;
      const pops = tick.rates?.pops ?? {};
      for (const p of PANEL_POPS) {
        const arr = this.sparks[p];
        arr.push(Number(pops[p]) || 0);
        if (arr.length > SPARK_N) arr.splice(0, arr.length - SPARK_N);
      }
      const sec = Math.floor(tick.wall);
      const mood = tick.mood.state;
      if (this.lastSec === null || sec < this.lastSec - 5) {
        this.timeline.fill(null);
      } else {
        const steps = Math.min(TIMELINE_W, Math.max(0, sec - this.lastSec));
        for (let i = 0; i < steps; i++) { this.timeline.shift(); this.timeline.push(mood); }
      }
      this.timeline[TIMELINE_W - 1] = mood;
      this.lastSec = sec;
      this.snap = { tick, sparks: this.sparks, timeline: this.timeline, version: this.snap.version + 1 };
    } catch (e) {
      console.error("[mood-panel] ingest failed", e);
      return;
    }
    for (const fn of this.listeners) {
      try { fn(); } catch (e) { console.error("[mood-panel] listener failed", e); }
    }
  }
}

// ------------------------------------------------------------------------------------------------ pixel faces

const FACE_BASE: readonly string[] = [
  "................",
  ".....OOOOOO.....",
  "....OFFFFFFO....",
  "...OFFFFFFFFO...",
  "..OFFFFFFFFFFO..",
  "..OFFFFFFFFFFO..",
  "..OFFFFFFFFFFO..",
  "..OFFFFFFFFFFO..",
  "..OFFFFFFFFFFO..",
  "..OFFFFFFFFFFO..",
  "...OFFFFFFFFO...",
  "....OFFFFFFO....",
  ".....OOOOOO.....",
  "................",
  "................",
  "................",
];
const PIX_COLORS: Record<string, string> = {
  O: "#000000", F: "#f2c078", E: "#000000", W: "#ffffff", R: "#c02020", P: "#ff69b4", B: "#1e90ff", Y: "#ffd700", L: "#404040",
};

function stamp(g: string[][], r: number, c: number, pattern: readonly string[]): void {
  for (let i = 0; i < pattern.length; i++) {
    for (let j = 0; j < pattern[i].length; j++) {
      const ch = pattern[i][j];
      if (ch === " ") continue;
      const rr = r + i, cc = c + j;
      if (rr >= 0 && rr < 16 && cc >= 0 && cc < 16) g[rr][cc] = ch;
    }
  }
}

const EYES_DOT = ["EE", "EE"];
const EYES_LINE = ["EE"];
const EYE_WIDE = ["WWW", "WEW", "WWW"];
const EYE_SPIRAL = ["EEEE", "   E", "EE E", "E  E"];
const EYE_HEART = ["P P", "PPP", " P "];
const MOUTH_FLAT = ["OOOO"];
const MOUTH_SMILE = ["O    O", " OOOO "];
const MOUTH_WAVE = ["O O O ", " O O O"];
const MOUTH_OPEN = ["EEEE", "EEEE"];

function faceGrid(mood: Mood): string[][] {
  const g = FACE_BASE.map((row) => row.split(""));
  switch (mood) {
    case "SLEEP":
      stamp(g, 6, 4, EYES_LINE); stamp(g, 6, 10, EYES_LINE); stamp(g, 9, 7, ["OO"]);
      break;
    case "CRUISING":
      stamp(g, 5, 4, EYES_DOT); stamp(g, 5, 10, EYES_DOT); stamp(g, 9, 6, MOUTH_FLAT);
      break;
    case "FEEDING":
      stamp(g, 5, 4, EYES_DOT); stamp(g, 5, 10, EYES_DOT); stamp(g, 9, 7, ["OO"]);
      stamp(g, 10, 7, ["OO", "OO", "OO", "OO", "YY", "YY"]);
      break;
    case "EUPHORIA":
      stamp(g, 4, 3, EYE_SPIRAL); stamp(g, 4, 9, EYE_SPIRAL); stamp(g, 8, 5, MOUTH_SMILE);
      break;
    case "ANXIOUS":
      stamp(g, 5, 4, EYES_DOT); stamp(g, 5, 10, EYES_DOT); stamp(g, 9, 5, MOUTH_WAVE);
      stamp(g, 3, 13, ["B", "BB", "BB"]);
      break;
    case "PANIC":
      stamp(g, 4, 3, EYE_WIDE); stamp(g, 4, 10, EYE_WIDE); stamp(g, 8, 6, MOUTH_OPEN);
      break;
    case "ESCAPE":
      stamp(g, 5, 4, EYES_LINE); stamp(g, 5, 10, EYES_LINE); stamp(g, 9, 6, MOUTH_FLAT);
      stamp(g, 4, 0, ["LL"]); stamp(g, 7, 0, ["LL"]); stamp(g, 10, 0, ["LL"]);
      stamp(g, 4, 14, ["LL"]); stamp(g, 7, 14, ["LL"]); stamp(g, 10, 14, ["LL"]);
      break;
    case "COURTSHIP":
      stamp(g, 4, 3, EYE_HEART); stamp(g, 4, 10, EYE_HEART); stamp(g, 8, 5, MOUTH_SMILE);
      break;
  }
  return g;
}

interface Px { x: number; y: number; w: number; fill: string }

/** Horizontal-run compression of a face grid into SVG rects (computed once per mood at module load). */
function faceRects(mood: Mood): Px[] {
  const g = faceGrid(mood);
  const out: Px[] = [];
  for (let y = 0; y < 16; y++) {
    let x = 0;
    while (x < 16) {
      const ch = g[y][x];
      if (ch === ".") { x++; continue; }
      let w = 1;
      while (x + w < 16 && g[y][x + w] === ch) w++;
      out.push({ x, y, w, fill: PIX_COLORS[ch] ?? "#000000" });
      x += w;
    }
  }
  return out;
}

const FACE_RECTS: Record<Mood, Px[]> = Object.fromEntries(MOOD_LIST.map((m) => [m, faceRects(m)])) as Record<Mood, Px[]>;

export function MoodFace({ mood, size = 32 }: { mood: Mood; size?: number }) {
  const rects = FACE_RECTS[mood] ?? FACE_RECTS.CRUISING;
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" shapeRendering="crispEdges" aria-label={`mood ${mood}`} role="img">
      {rects.map((p, i) => <rect key={i} x={p.x} y={p.y} width={p.w} height={1} fill={p.fill} />)}
      {mood === "SLEEP" ? (
        <g fill="#000080" fontFamily="monospace" fontWeight="bold">
          <text x={11} y={4} fontSize={4}>z</text>
          <text x={13} y={2.5} fontSize={3}>z</text>
        </g>
      ) : null}
    </svg>
  );
}

// ------------------------------------------------------------------------------------------------ widgets

const SEGMENTS = 20;

/** Win95 segmented progress bar. `centered` draws from the middle (valence -1..1). */
function SegBar({ value, min = 0, max = 1, ticks, centered = false, color = "#000080" }:
  { value: number; min?: number; max?: number; ticks?: readonly number[]; centered?: boolean; color?: string }) {
  const span = max - min || 1;
  const frac = Math.min(1, Math.max(0, (value - min) / span));
  const filled = Math.round(frac * SEGMENTS);
  const half = SEGMENTS / 2;
  const segs: boolean[] = [];
  for (let i = 0; i < SEGMENTS; i++) {
    if (!centered) segs.push(i < filled);
    else if (filled >= half) segs.push(i >= half && i < filled);
    else segs.push(i >= filled && i < half);
  }
  const fill = centered ? (filled >= half ? "#008000" : "#a80000") : color;
  return (
    <div className="bevel-in relative flex h-[10px] w-[102px] items-center gap-[1px] px-[1px]" style={{ background: "#ffffff" }}>
      {segs.map((on, i) => (
        <span key={i} className="h-[6px] flex-1" style={{ background: on ? fill : "transparent" }} />
      ))}
      {(ticks ?? []).map((t) => (
        <span key={t} className="absolute top-0 h-full w-px" style={{ left: `${((t - min) / span) * 100}%`, background: "#ff0000", opacity: 0.8 }} title={`threshold ${t}`} />
      ))}
    </div>
  );
}

function Sparkline({ values, w = 64, h = 12 }: { values: readonly number[]; w?: number; h?: number }) {
  const n = values.length;
  let max = 1;
  for (const v of values) if (v > max) max = v;
  const pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = n > 1 ? (i * (w - 1)) / (n - 1) : w - 1;
    const y = h - 1 - (Math.max(0, values[i]) / max) * (h - 2);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="block bg-white" aria-hidden="true">
      <polyline points={pts.join(" ")} fill="none" stroke="#000080" strokeWidth={1} />
    </svg>
  );
}

function timelineColor(m: Mood): string {
  return m === "EUPHORIA" ? "#ff00ff" : moodColor(m, 0);
}

/**
 * Runs of identical moods -> one CSS gradient (few stops; mood changes are rare). The stops are *percentages* of the
 * element width, never px: the 300-column ring has to map onto the real box, which in the section e.7 Fly Status window
 * (560 px wide, two columns) is only ~205-270 px - px stops would simply clip the newest ~1.5 min of the timeline.
 */
export function timelineGradient(tl: readonly (Mood | null)[]): string {
  const n = tl.length;
  if (n === 0) return "#c0c0c0";
  const stops: string[] = [];
  const pct = (i: number) => `${((i / n) * 100).toFixed(3)}%`;
  let start = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || tl[i] !== tl[start]) {
      const m = tl[start];
      const c = m ? timelineColor(m) : "#c0c0c0";
      stops.push(`${c} ${pct(start)} ${pct(i)}`);
      start = i;
    }
  }
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

function fmtHz(v: number | undefined): string {
  return v === undefined || !Number.isFinite(v) ? "-" : v.toFixed(1);
}

// ------------------------------------------------------------------------------------------------ panel

export function MoodPanel({ sock }: MoodPanelProps) {
  const store = sock.store;
  const hello = sock.hello;
  const on = sock.on;
  const [hist] = useState(() => new PanelHistory(store));      // one history per mount (never re-created)
  const snap = useSyncExternalStore(hist.subscribe, hist.getSnapshot, hist.getServerSnapshot);
  const tick = snap.tick;
  const mood = tick?.mood ?? null;
  // Section d.3: a `mood_change` frame repaints the label/face immediately - it does not wait for the next 4 Hz tick
  // (up to 250 ms later). The announced mood wins until a tick *newer than the one current when it was announced*
  // confirms it, so a reconnect into a different run can never leave a stale override behind. `mood_change` is rare
  // (a few per minute, section d.3), so this setState is not a per-tick update.
  const [announced, setAnnounced] = useState<{ to: Mood; seq: number } | null>(null);
  useEffect(() => on("mood_change", (m) => setAnnounced({ to: m.to, seq: store.latest()?.tick.seq ?? -1 })), [on, store]);
  const state: Mood = announced !== null && (tick?.seq ?? -1) <= announced.seq ? announced.to : (mood?.state ?? "CRUISING");
  const tSec = tick?.wall ?? 0;
  const moods: readonly Mood[] = hello?.mood_states?.length ? hello.mood_states : MOOD_LIST;
  const drives = tick?.drives ?? null;
  const pops = tick?.rates.pops ?? {};

  return (
    <div className="flex h-full flex-col gap-1 overflow-auto bg-win-gray p-1 text-[11px]" data-testid="mood-panel">
      {/* row 1: mood + face + timers */}
      <div className="flex items-center gap-2">
        <MoodFace mood={state} />
        <div className="flex flex-col">
          <span className="text-[14px] font-bold" style={{ color: moodColor(state, tSec) }} data-testid="mood-label">{state}</span>
          <span className="text-[#404040]">
            {mood ? `since ${(mood.since_ms / 1000).toFixed(0)}s` : "waiting for ticks"}
            {mood && mood.dwell_left_ms > 0 ? ` · dwell ${(mood.dwell_left_ms / 1000).toFixed(1)}s` : ""}
            {mood ? ` · prev ${mood.prev}` : ""}
          </span>
        </div>
        <div className="ml-auto flex flex-wrap justify-end gap-[2px]">
          {moods.map((m) => (
            <span key={m} className="btn95 flex items-center gap-1 text-[9px] leading-[14px]" aria-pressed={m === state} title={MOOD_TIP[m]}>
              <span className="inline-block h-[8px] w-[8px] border border-black" style={{ background: moodColor(m, tSec) }} />
              {m}
            </span>
          ))}
        </div>
      </div>

      {/* row 2: score bars */}
      <div className="grid grid-cols-[auto_auto_auto] items-center gap-x-1 gap-y-[2px]">
        <span>euphoria</span><SegBar value={mood?.euphoria ?? 0} ticks={TICKS_EUPHORIA} /><span className="font-mono">{(mood?.euphoria ?? 0).toFixed(2)}</span>
        <span>anxiety</span><SegBar value={mood?.anxiety ?? 0} ticks={TICKS_ANXIETY} /><span className="font-mono">{(mood?.anxiety ?? 0).toFixed(2)}</span>
        <span>arousal</span><SegBar value={mood?.arousal ?? 0} /><span className="font-mono">{(mood?.arousal ?? 0).toFixed(2)}</span>
        <span>hunger</span><SegBar value={mood?.hunger ?? 0} /><span className="font-mono">{(mood?.hunger ?? 0).toFixed(2)}</span>
        <span>sleep</span><SegBar value={mood?.sleep ?? 0} /><span className="font-mono">{(mood?.sleep ?? 0).toFixed(2)}</span>
        <span>valence</span><SegBar value={mood?.valence ?? 0} min={-1} max={1} centered /><span className="font-mono">{(mood?.valence ?? 0) >= 0 ? "+" : ""}{(mood?.valence ?? 0).toFixed(2)}</span>
      </div>

      {/* row 3: drives */}
      <div className="grid grid-cols-3 gap-x-2 gap-y-[1px]" data-testid="drive-table">
        {DRIVE_KEYS.map((k) => {
          const raw = drives ? drives[k] : 0;
          const v = typeof raw === "number" ? raw : 0;
          const arrow = k === "looming" && drives ? (drives.loom_side < 0 ? " ◄" : drives.loom_side > 0 ? " ►" : "") : "";
          return (
            <div key={k} className="flex items-center gap-1" title={`${k} = ${v.toFixed(3)}`}>
              <span className="w-[62px] truncate">{k}{arrow}</span>
              <span className="bevel-in h-[7px] w-[40px] bg-white">
                <span className="block h-full" style={{ width: `${Math.round(Math.min(1, Math.max(0, v)) * 100)}%`, background: v >= 0.5 ? "#a80000" : "#000080" }} />
              </span>
              <span className="font-mono text-[10px]">{v.toFixed(2)}</span>
            </div>
          );
        })}
        {drives ? (
          <div className="col-span-3 text-[#404040]">
            candle {drives.candle} · up {drives.up.toFixed(2)} · down {drives.down.toFixed(2)} · activity {drives.activity.toFixed(2)} · hunger {drives.hunger.toFixed(2)} · max {drives.any_max.toFixed(2)}
          </div>
        ) : null}
      </div>

      {/* row 4: mood timeline */}
      <div className="flex items-center gap-1">
        <span className="w-[62px]">last 5 min</span>
        <div className="bevel-in h-[12px] min-w-0 max-w-[300px] flex-1" style={{ background: timelineGradient(snap.timeline) }} title="mood timeline: 300 columns = the last 5 min (one column per second), scaled to the box" data-testid="mood-timeline" />
      </div>

      {/* row 5: population table */}
      <table className="w-full border-collapse font-mono text-[10px]" data-testid="pop-table">
        <thead>
          <tr className="text-left text-[#404040]"><th className="font-normal">pop</th><th className="font-normal">8 s</th><th className="text-right font-normal">Hz</th></tr>
        </thead>
        <tbody>
          {PANEL_POPS.map((p) => (
            <tr key={p}>
              <td className="pr-1">{p}</td>
              <td className="w-[66px]"><Sparkline values={snap.sparks[p] ?? []} /></td>
              <td className="text-right">{fmtHz(pops[p])}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* footer */}
      <div className="mt-auto border-t border-[#808080] pt-[2px] text-[10px] text-[#404040]" data-testid="mood-footer">
        {hello ? (
          <>
            run {hello.run_id} · {hello.connectome.name} ({hello.connectome.source}) · {hello.connectome.n.toLocaleString("en-US")} n / {hello.connectome.e.toLocaleString("en-US")} e
            · gain {hello.connectome.gain} · dt {hello.dt_ms} ms · {tick?.sim.backend ?? hello.backend}
            {tick ? ` · rtf ${tick.sim.rtf.toFixed(2)} · speed ${tick.sim.speed.toFixed(2)}x` : ""}
            {hello.connectome.source === "synthetic" ? <b> · synthetic stand-in</b> : null}
          </>
        ) : "no hello yet"}
      </div>
    </div>
  );
}

export default MoodPanel;
