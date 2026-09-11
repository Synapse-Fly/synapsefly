"use client";
// SPEC section e.5 `PaintWindow`: "untitled - Paint ($SYNAPSE)" with the Paint icon; MenuBar row, then
// [ToolPalette | workspace], then ColorPalette, then StatusBar. The workspace is a win-gray area holding the white
// 800 x 500 document (hello.canvas) inside a .bevel-in frame with three inert 3 px resize handles. 880 x 680 at (16, 16).
import { useCallback, useEffect, useState, type RefObject } from "react";
import type { FlySocket } from "@/lib/ws";
import { useTickSnapshot } from "@/lib/store";
import type { InkStyle, Mood, PokeStim } from "@/lib/types";
import Win95Window, { type WinRect } from "./Win95Window";
import MenuBar, { type MenuAction, type MenuFlags } from "./MenuBar";
import ToolPalette from "./ToolPalette";
import ColorPalette from "./ColorPalette";
import StatusBar from "./StatusBar";
import FlyCanvas, { type FlyCanvasHandle } from "./FlyCanvas";

export interface PaintWindowProps {
  sock: FlySocket;
  onAction(a: MenuAction): void;
  canvasRef: RefObject<FlyCanvasHandle | null>;
  flags?: Partial<MenuFlags>;
  rect?: WinRect;
}

export const PAINT_RECT = { x: 16, y: 16, w: 880, h: 600 };
const DEFAULT_INK: InkStyle = { color: "#000000", width: 2, alpha: 1, style: "solid", stamp: null };

export default function PaintWindow({ sock, onAction, canvasRef, flags = {}, rect }: PaintWindowProps) {
  const tick = useTickSnapshot(sock.store);
  const [active, setActive] = useState<PokeStim>("sugar");
  const [lastPoke, setLastPoke] = useState<{ stim: PokeStim; at: number } | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  // SPEC d.3: the status bar mood badge repaints on the mood_change frame itself (one cheap state update per
  // transition - a few per minute, never per tick), not on the next 4 Hz snapshot.
  const [moodNow, setMoodNow] = useState<{ to: Mood; t_ms: number } | null>(null);
  const { on } = sock;

  useEffect(() => on("mood_change", (m) => setMoodNow({ to: m.to, t_ms: m.t_ms })), [on]);
  useEffect(() => on("hello", () => setMoodNow(null)), [on]);          // new session: trust the ticks again

  // Pokes from any client are echoed as `poke` events: flash the matching tool.
  useEffect(() => on("event", (m) => {
    if (m.kind !== "poke") return;
    const stim = m.data?.stim;
    if (typeof stim === "string") setLastPoke({ stim: stim as PokeStim, at: performance.now() });
  }), [on]);

  const onPoke = useCallback((stim: PokeStim) => {
    setActive(stim);
    setLastPoke({ stim, at: performance.now() });
    onAction({ kind: "poke", stim });
  }, [onAction]);

  const onFps = useCallback((f: number) => setFps((p) => (p === f ? p : f)), []);

  const ink = tick?.ink ?? DEFAULT_INK;
  const freshMood = moodNow !== null && (!tick || tick.t_ms <= moodNow.t_ms);
  const mood: Mood = freshMood && moodNow ? moodNow.to : tick?.mood.state ?? "CRUISING";
  const W = sock.hello?.canvas.w ?? 800, H = sock.hello?.canvas.h ?? 500;
  const zoom = !!flags.zoom;

  return (
    <Win95Window
      id="paint"
      title="untitled - Paint ($SYNAPSE)"
      icon={
        /* eslint-disable-next-line @next/next/no-img-element */
        <img src="/favicon-48.png" alt="" width={16} height={16} style={{ imageRendering: "pixelated" }} />
      }
      initial={rect ?? PAINT_RECT}
      menu={<MenuBar sock={sock} onAction={onAction} flags={flags} />}
      statusBar={<StatusBar store={sock.store} hello={sock.hello} status={sock.status} attempt={sock.attempt} fps={flags.fps ? fps : null} moodNow={moodNow} />}
      collapsible
    >
      <div className="flex h-full flex-col">
        <div className="flex min-h-0 flex-1">
          <ToolPalette active={active} onPoke={onPoke} lastPoke={lastPoke} inkWidth={ink.width} inkColor={ink.style === "rainbow" ? "#ff00ff" : ink.color} />
          <div className="relative min-w-0 flex-1 overflow-auto bg-win-gray" style={{ padding: "4px 4px 8px 6px" }}>
            <div className="relative inline-block" style={{ transform: zoom ? "scale(1.5)" : undefined, transformOrigin: "0 0", padding: "0 6px 6px 0" }}>
              <div className="bevel-in" style={{ width: W + 4, height: H + 4, lineHeight: 0 }}>
                <FlyCanvas ref={canvasRef} sock={sock} onFps={onFps} flip={!!flags.flip} pokeStim={active} />
              </div>
              {/* three inert resize handles (right, bottom, corner) */}
              <div className="absolute h-[3px] w-[3px] bg-win-blue" style={{ left: W + 6, top: (H + 4) / 2 - 1 }} aria-hidden />
              <div className="absolute h-[3px] w-[3px] bg-win-blue" style={{ left: (W + 4) / 2 - 1, top: H + 6 }} aria-hidden />
              <div className="absolute h-[3px] w-[3px] bg-win-blue" style={{ left: W + 6, top: H + 6 }} aria-hidden />
            </div>
          </div>
        </div>
        <ColorPalette ink={ink} mood={mood} tSec={tick ? tick.t_ms / 1000 : 0} />
      </div>
    </Win95Window>
  );
}
