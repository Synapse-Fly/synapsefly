"use client";
// SPEC section e.5 `StatusBar`: .bevel-in cells - help text (replaced by the last event kind for 2 s), fly position,
// document size, mood badge, price ticker, RTF / brain speed / n / e, market source badge, WS dot. Re-renders at 4 Hz
// through useTickSnapshot (never from a tick handler).
//
// WIDTH BEHAVIOUR (why this file is more than a row of divs). The Paint window is resizable and tiled, so this row is
// regularly narrower than eight cells want. A row of flex cells with fixed widths squeezes them all: at 1366 the first
// cell became a single clipped glyph inside a bevel and the connectome cell read "20000 n / 4" instead of
// "20000 n / 493249"; at 390 the whole row was eight unreadable slivers. Windows 95 never did that - a status-bar pane
// was either wide enough for its text or it was not there at all. So:
//
//   * every cell is INTRINSICALLY sized (flex: none) with a min-width floor that covers the widest value it can show.
//     A cell is therefore never narrower than its own text - nothing is clipped, no number is ever half-truncated -
//     and the floor stops a changing digit from making the whole row dance;
//   * a hidden "ruler" row (`ghostRef`) holds every cell, and every variant of the help text, at its natural width.
//     One layout-effect measurement per render plus a ResizeObserver turns that into real pixels. It measures THIS
//     PANE, not the viewport, which is the whole point: the window can be dragged to any width on any screen;
//   * cells are packed in PRIORITY order (CELL_PRIO below) and everything from the first one that does not fit is
//     dropped, so the row degrades from the least important end upward;
//   * the help / last-event pane is the stretchy one, exactly as in real Paint: it absorbs whatever is left over,
//     prints the longest of its texts that fits in that space, and is not rendered at all below FILLER_MIN px.
//
// Measurement lag cannot clip anything: the visible cells are content-sized by the browser, so a stale ruler width can
// only make the packing decision one 250 ms render old (the row, which is overflow-hidden, absorbs that), never make a
// cell too small for its text. The stretchy pane is the one flex-sized box here, so its variants are keyed by their
// TEXT - a text nobody has measured yet counts as "does not fit" and the pane simply stays empty for one render.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { NOT_LAUNCHED, TOKEN, feedSubject, tokenLive } from "@/lib/brand";
import type { TickStore } from "@/lib/store";
import { useTickSnapshot } from "@/lib/store";
import type { HelloMsg, Mood } from "@/lib/types";
import type { WsStatus } from "@/lib/ws";
import { fmtPrice } from "@/lib/api";
import { moodColor } from "@/lib/ink";

export interface StatusBarProps {
  store: TickStore; hello: HelloMsg | null; status: WsStatus; attempt: number; fps: number | null;
  /** SPEC d.3: the badge repaints on the `mood_change` frame, it does not wait for the next 4 Hz tick snapshot.
   *  PaintWindow (which owns the socket) feeds the frame's `to` + `t_ms` here; the tick wins again as soon as one
   *  newer than the transition has landed. */
  moodNow?: { to: Mood; t_ms: number } | null;
}

const HELP_TEXT = "For Help, click Help Topics on the Help Menu.";
/** Shorter stand-ins for the same sentence, longest first: a narrow pane gets as much of it as fits. */
const HELP_SHORT = ["For Help, click Help Topics", "Help Topics"];
const EVENT_MS = 2000;

const GAP = 2;            // gap-[2px] between cells
const PAD = 2;            // p-[2px] around the row
const SAFETY = 1;         // sub-pixel layout widths are ceiled, then padded, so packing never over-commits
const FILLER_MIN = 32;    // below this the stretchy pane is a sliver, so it is dropped and the slack stays gray

/** Display order is left-to-right; `prio` is the survival order (0 = the last one standing). */
const CELL_PRIO = { ws: 0, mood: 1, market: 2, size: 3, pos: 4, sim: 5, source: 6, fps: 7 } as const;
type CellKey = keyof typeof CELL_PRIO;

interface CellForm {
  /** px floor, covering the widest text this form can hold, so a changing value never re-flows the row. */
  min: number;
  body: ReactNode;
}

interface Cell extends CellForm {
  key: CellKey;
  title: string;
  className?: string;
  style?: CSSProperties;
  /** Narrower printings of the SAME cell, longest first, tried in order before the cell is dropped altogether.
   *  Only the market cell uses them, and only to shed its numbers: the "not launched" qualifier is in every form,
   *  so a price can never reach the screen without it - the cell goes dark first. */
  alts?: readonly CellForm[];
}

const CELL_CLASS = "bevel-in h-[18px] overflow-hidden whitespace-nowrap bg-win-gray px-1 leading-[14px]";

/** Every printing of a cell, longest first (index 0 is the cell's own `body`). */
function formsOf(c: Cell): readonly CellForm[] {
  return c.alts && c.alts.length ? [{ min: c.min, body: c.body }, ...c.alts] : [{ min: c.min, body: c.body }];
}

/** Ruler key for one form: the bare cell key for form 0 (so existing widths keep their names), `key:i` beyond. */
function formKey(key: CellKey, i: number): string {
  return i === 0 ? key : `${key}:${i}`;
}

/** One bevelled pane at its own content width (`flex: none`), never narrower than `min`. `grow` is for the last pane
 *  of a row whose stretchy help pane was dropped: it eats the few px of slack so the row still ends flush. */
function paneOf(c: Cell, grow = false, i = 0): ReactNode {
  const forms = formsOf(c);
  const f = forms[i] ?? forms[0];
  return (
    <div key={formKey(c.key, i)} data-cell={formKey(c.key, i)} title={c.title}
      style={{ minWidth: f.min, flexGrow: grow ? 1 : undefined, ...c.style }}
      className={`${CELL_CLASS} flex-none ${c.className ?? ""}`}>
      {f.body}
    </div>
  );
}

export function sourceBadge(source: string | undefined | null): string {
  if (source === "dexscreener") return "DEX";
  if (source === "sim(fallback)") return "SIM(fallback)";
  return "SIM";
}

/** useLayoutEffect, minus the "does nothing on the server" warning when this client component is pre-rendered. */
const useMeasureEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

interface Metrics { avail: number; w: Readonly<Record<string, number>> }
const NO_METRICS: Metrics = { avail: 0, w: {} };

function sameMetrics(a: Metrics, avail: number, w: Record<string, number>): boolean {
  if (a.avail !== avail) return false;
  const keys = Object.keys(w);
  if (Object.keys(a.w).length !== keys.length) return false;
  for (const k of keys) if (a.w[k] !== w[k]) return false;
  return true;
}

export default function StatusBar({ store, hello, status, attempt, fps, moodNow = null }: StatusBarProps) {
  const tick = useTickSnapshot(store);
  const w = hello?.canvas.w ?? 800, h = hello?.canvas.h ?? 500;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const ghostRef = useRef<HTMLDivElement | null>(null);
  const [m, setM] = useState<Metrics>(NO_METRICS);

  // Read this pane's own width and every ruler cell's natural width. Runs before paint after each render (the content
  // changes at 4 Hz), and again whenever the window is resized without a re-render.
  const measure = useCallback(() => {
    const root = rootRef.current, ghost = ghostRef.current;
    if (!root || !ghost) return;
    const avail = Math.max(0, root.clientWidth - 2 * PAD);
    const next: Record<string, number> = {};
    for (let i = 0; i < ghost.children.length; i++) {
      const el = ghost.children[i] as HTMLElement;
      const key = el.dataset.cell;
      if (key) next[key] = Math.ceil(el.getBoundingClientRect().width) + SAFETY;
    }
    setM((prev) => (sameMetrics(prev, avail, next) ? prev : { avail, w: next }));
  }, []);

  useMeasureEffect(measure);
  useEffect(() => {
    const root = rootRef.current;
    if (!root || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(root);
    return () => ro.disconnect();
  }, [measure]);

  let help = HELP_TEXT;
  let variants: string[] = [HELP_TEXT, ...HELP_SHORT];
  const ev = store.events.length ? store.events[store.events.length - 1] : null;
  if (tick && ev && tick.t_ms - ev.t_ms <= EVENT_MS && tick.t_ms - ev.t_ms >= 0) {
    const side = typeof ev.data?.side === "string" ? ` ${ev.data.side}` : "";
    help = `event: ${ev.kind}${side}`;
    variants = [help, `event: ${ev.kind}`, ev.kind];
  }
  const helpVariants = variants.filter((v, i) => v.length > 0 && variants.indexOf(v) === i);

  // The mood_change frame (d.3) wins until a tick at or after the transition has landed.
  const fresh = moodNow !== null && (!tick || tick.t_ms <= moodNow.t_ms);
  const mood: Mood = fresh && moodNow ? moodNow.to : tick?.mood.state ?? "CRUISING";
  const mc = moodColor(mood, tick ? tick.t_ms / 1000 : 0);
  const mk = tick?.market;
  const source = mk?.source ?? hello?.market.mode;
  // The brand ticker, but never bare over a price that is not ours. Two different lies are possible here and both are
  // closed in the ticker label itself, because the SIM cell further right is too easy to miss:
  //   * a SIMULATED price under $SYNAPSE  -> "(sim)" rides with the number;
  //   * a REAL price from somebody else's pair under $SYNAPSE, which is what a DexScreener feed is until $SYNAPSE
  //     actually lists -> "not launched" rides with the number, in red, and outlives it (the alts below).
  // A missing `token_live` counts as "not launched" (brand.ts), so an old backend cannot silently unlock the bare
  // ticker. When the flag is true this cell is byte-for-byte what it has always been.
  const live = tokenLive(mk?.token_live, hello?.market.token_live);
  const symbol = live ? (source === "dexscreener" ? TOKEN : `${TOKEN} (sim)`) : TOKEN;
  const chg = mk ? mk.chg_m5 : 0;
  const qualifier = <span className="font-bold" style={{ color: "#a80000" }}>{NOT_LAUNCHED}</span>;
  const priceText = `$${fmtPrice(mk?.price_usd ?? null)}`;
  const chgText = <span style={{ color: chg >= 0 ? "#008000" : "#c00000" }}>{chg >= 0 ? "+" : ""}{chg.toFixed(1)}%</span>;
  const marketTitle = mk
    ? `price ${mk.price_usd ?? "-"} | m5 ${mk.chg_m5}% h1 ${mk.chg_h1}% h24 ${mk.chg_h24}%`
      + (live ? "" : ` | $${TOKEN} has not launched - this is ${feedSubject(mk.symbol, mk.dex, mk.chain)}, the brain's sensory input`)
    : "market";
  const dot = status === "open" ? "#00a800" : status === "connecting" ? "#e0c000" : "#d00000";
  const wsText = status === "open" ? "online" : status === "connecting" ? "connecting" : status === "closed" ? "closed" : `reconnecting (${attempt})`;

  // ------------------------------------------------------------------------------- the cells, in display order
  const cells: Cell[] = [
    { key: "pos", min: 58, title: "fly position (px)", body: tick ? `${Math.round(tick.fly.x)},${Math.round(tick.fly.y)}` : "-,-" },
    { key: "size", min: 62, title: "document size", body: `${w} x ${h}` },
    {
      key: "mood", min: 88, className: "font-bold",
      title: tick ? `mood since ${(tick.mood.since_ms / 1000).toFixed(0)} s (prev ${tick.mood.prev})` : "mood",
      style: {
        color: mood === "SLEEP" || mood === "CRUISING" ? "#000" : mc,
        background: mood === "CRUISING" ? undefined : mood === "SLEEP" ? "#dfdfdf" : undefined,
      },
      body: (
        <>
          <span className="mr-1 inline-block h-[8px] w-[8px] border border-black align-middle" style={{ background: mc }} />
          {mood}
        </>
      ),
    },
    {
      key: "market", min: live ? 196 : 268, className: "font-mono", title: marketTitle,
      body: live ? (
        <>
          {symbol} {priceText}{" "}
          {chgText}
        </>
      ) : (
        <>
          {symbol} — {qualifier} {priceText} {chgText}
        </>
      ),
      // Shed the change, then the price. There is deliberately no form that prints a number without the qualifier.
      alts: live ? undefined : [
        { min: 224, body: <>{symbol} — {qualifier} {priceText}</> },
        { min: 160, body: <>{symbol} — {qualifier}</> },
      ],
    },
    {
      key: "sim", min: 210, title: "realtime factor / brain speed / connectome size",
      body: (
        <>
          {tick ? `RTF ${tick.sim.rtf.toFixed(2)} · brain ${tick.sim.speed.toFixed(2)}x` : "RTF - · brain -"}
          {hello ? ` · ${hello.connectome.n} n / ${hello.connectome.e} e` : ""}
        </>
      ),
    },
    { key: "source", min: 40, className: "text-center font-bold", title: `market source: ${source ?? "sim"}`, body: sourceBadge(source) },
    ...(fps !== null ? [{ key: "fps" as const, min: 52, className: "font-mono", title: "frames per second", body: `${fps} fps` }] : []),
    {
      key: "ws", min: 62, title: `websocket ${status}`,
      body: (
        <>
          <span className="mr-1 inline-block h-[8px] w-[8px] rounded-full border border-black align-middle" style={{ background: dot }} />
          {wsText}
        </>
      ),
    },
  ];

  // ------------------------------------------------------------------------------- the fit
  // Keep a prefix of the priority order, then hand the stretchy help pane whatever is left. Stopping at the first cell
  // that does not fit (rather than skipping it in favour of a narrower one further down the list) is what makes the row
  // shed cells strictly from the bottom upward, so widening the window adds them back in exactly the same order.
  // `shown` also carries WHICH form each surviving cell prints: a cell with `alts` tries them longest-first and is
  // only dropped when even its shortest form does not fit.
  const shown = new Map<CellKey, number>();
  let used = 0;
  if (m.avail > 0) {
    for (const c of cells.slice().sort((a, b) => CELL_PRIO[a.key] - CELL_PRIO[b.key])) {
      const gap = shown.size ? GAP : 0;
      const forms = formsOf(c);
      let pick = -1;
      let need = 0;
      for (let i = 0; i < forms.length; i++) {
        const w = Math.max(forms[i].min, m.w[formKey(c.key, i)] ?? forms[i].min) + gap;
        if (used + w <= m.avail) { pick = i; need = w; break; }
      }
      if (pick < 0) break;
      used += need;
      shown.set(c.key, pick);
    }
  }
  const room = m.avail > 0 ? Math.max(0, m.avail - used - (shown.size ? GAP : 0)) : 0;
  const showHelp = room >= FILLER_MIN;
  let helpText = "";
  if (showHelp) {
    for (const v of helpVariants) {
      const vw = m.w[`help:${v}`];
      if (vw !== undefined && vw <= room) { helpText = v; break; }
    }
  }
  const visible = cells.filter((c) => shown.has(c.key));

  return (
    <div
      ref={rootRef}
      data-testid="status-bar"
      data-cells={[...(showHelp ? ["help"] : []), ...visible.map((c) => c.key)].join(",")}
      className="relative flex items-center gap-[2px] overflow-hidden bg-win-gray p-[2px] text-[11px]"
    >
      {showHelp ? <div data-cell="help" title={help} className={`${CELL_CLASS} min-w-0 flex-1`}>{helpText}</div> : null}
      {visible.map((c, i) => paneOf(c, !showHelp && i === visible.length - 1, shown.get(c.key) ?? 0))}
      {/* The ruler: every pane at its natural width, out of flow and invisible, read by `measure()` above. `w-max`
          keeps the shrink-to-fit box from squeezing these panes the way the visible row used to be squeezed, and the
          0 x 0 clipping wrapper keeps that over-wide row out of the status bar's own scroll width. */}
      <div aria-hidden className="pointer-events-none invisible absolute left-0 top-0 h-0 w-0 overflow-hidden">
        <div ref={ghostRef} className="flex w-max items-center gap-[2px]">
          {cells.flatMap((c) => formsOf(c).map((_f, i) => paneOf(c, false, i)))}
          {helpVariants.map((v) => (
            <div key={v} data-cell={`help:${v}`} className={`${CELL_CLASS} flex-none`}>{v}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
