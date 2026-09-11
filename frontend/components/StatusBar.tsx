"use client";
// SPEC section e.5 `StatusBar`: .bevel-in cells - help text (replaced by the last event kind for 2 s), fly position,
// document size, mood badge, price ticker, RTF / brain speed / n / e, market source badge, WS dot. Re-renders at 4 Hz
// through useTickSnapshot (never from a tick handler).
import type { ReactNode } from "react";
import { TOKEN } from "@/lib/brand";
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
const EVENT_MS = 2000;

function Cell({ children, className = "", title, style }: { children: ReactNode; className?: string; title?: string; style?: React.CSSProperties }) {
  return (
    <div className={"bevel-in h-[18px] overflow-hidden whitespace-nowrap bg-win-gray px-1 leading-[14px] " + className} title={title} style={style}>
      {children}
    </div>
  );
}

export function sourceBadge(source: string | undefined | null): string {
  if (source === "dexscreener") return "DEX";
  if (source === "sim(fallback)") return "SIM(fallback)";
  return "SIM";
}

export default function StatusBar({ store, hello, status, attempt, fps, moodNow = null }: StatusBarProps) {
  const tick = useTickSnapshot(store);
  const w = hello?.canvas.w ?? 800, h = hello?.canvas.h ?? 500;

  let help = HELP_TEXT;
  const ev = store.events.length ? store.events[store.events.length - 1] : null;
  if (tick && ev && tick.t_ms - ev.t_ms <= EVENT_MS && tick.t_ms - ev.t_ms >= 0) {
    const side = typeof ev.data?.side === "string" ? ` ${ev.data.side}` : "";
    help = `event: ${ev.kind}${side}`;
  }

  // The mood_change frame (d.3) wins until a tick at or after the transition has landed.
  const fresh = moodNow !== null && (!tick || tick.t_ms <= moodNow.t_ms);
  const mood: Mood = fresh && moodNow ? moodNow.to : tick?.mood.state ?? "CRUISING";
  const mc = moodColor(mood, tick ? tick.t_ms / 1000 : 0);
  const m = tick?.market;
  const symbol = TOKEN;
  const chg = m ? m.chg_m5 : 0;
  const dot = status === "open" ? "#00a800" : status === "connecting" ? "#e0c000" : "#d00000";
  const wsText = status === "open" ? "online" : status === "connecting" ? "connecting" : status === "closed" ? "closed" : `reconnecting (${attempt})`;

  return (
    <div className="flex items-center gap-[2px] bg-win-gray p-[2px] text-[11px]" data-testid="status-bar">
      <Cell className="min-w-0 flex-1" title={help}>{help}</Cell>
      <Cell className="w-[84px]" title="fly position (px)">{tick ? `${Math.round(tick.fly.x)},${Math.round(tick.fly.y)}` : "-,-"}</Cell>
      <Cell className="w-[70px]" title="document size">{`${w} x ${h}`}</Cell>
      <Cell className="w-[86px] font-bold" title={tick ? `mood since ${(tick.mood.since_ms / 1000).toFixed(0)} s (prev ${tick.mood.prev})` : "mood"}
        style={{ color: mood === "SLEEP" || mood === "CRUISING" ? "#000" : mc, background: mood === "CRUISING" ? undefined : mood === "SLEEP" ? "#dfdfdf" : undefined }}>
        <span className="mr-1 inline-block h-[8px] w-[8px] border border-black align-middle" style={{ background: mc }} />
        {mood}
      </Cell>
      <Cell className="w-[150px] font-mono" title={m ? `price ${m.price_usd ?? "-"} | m5 ${m.chg_m5}% h1 ${m.chg_h1}% h24 ${m.chg_h24}%` : "market"}>
        {symbol} ${fmtPrice(m?.price_usd ?? null)}{" "}
        <span style={{ color: chg >= 0 ? "#008000" : "#c00000" }}>{chg >= 0 ? "+" : ""}{chg.toFixed(1)}%</span>
      </Cell>
      <Cell className="w-[210px]" title="realtime factor / brain speed / connectome size">
        {tick ? `RTF ${tick.sim.rtf.toFixed(2)} · brain ${tick.sim.speed.toFixed(2)}x` : "RTF - · brain -"}
        {hello ? ` · ${hello.connectome.n} n / ${hello.connectome.e} e` : ""}
      </Cell>
      <Cell className="w-[74px] text-center font-bold" title={`market source: ${m?.source ?? hello?.market.mode ?? "sim"}`}>
        {sourceBadge(m?.source ?? hello?.market.mode)}
      </Cell>
      {fps !== null ? <Cell className="w-[52px] font-mono" title="frames per second">{fps} fps</Cell> : null}
      <Cell className="w-[110px]" title={`websocket ${status}`}>
        <span className="mr-1 inline-block h-[8px] w-[8px] rounded-full border border-black align-middle" style={{ background: dot }} />
        {wsText}
      </Cell>
    </div>
  );
}
