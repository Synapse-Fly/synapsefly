"use client";
// SPEC section e.5 `PaintWindow`: "untitled - Paint ($SYNAPSE)" with the Paint icon; MenuBar row, then
// [ToolPalette | workspace], then ColorPalette, then StatusBar. The workspace is a win-gray area holding the white
// 800 x 500 document (hello.canvas) inside a .bevel-in frame with three inert 3 px resize handles.
//
// The document BITMAP stays hello.canvas (800 x 500: the protocol, the trail backfill and the snapshot pipeline all
// depend on it) - only its CSS box scales. A ResizeObserver measures the workspace and lib/layout.ts `fitDocBox`
// returns the largest aspect-correct box that fits, so the canvas fills the Paint window instead of sitting in a
// corner of a gray field (SPEC e.5 hard-coded it at W+4 x H+4). FlyCanvas maps pointer events back through
// getBoundingClientRect, so pokes stay accurate at any scale, and composite()/snapshots still use canvas.width.
//
// PaintWindow also hosts the two first-impression pieces, because it is the component that owns both the socket and
// the canvas handle: OfflineNotice (the Win95 box that explains a dead WebSocket instead of leaving an empty desktop)
// and ShareDialog (File > Share to X... / the Share button). Both render position:fixed OUTSIDE the window frame.
//
// The other half of the deal is in lib/layout.ts: it sizes the Paint WINDOW to an exact 8:5 document plus this
// component's chrome (PAINT_CHROME_W = 58 px ToolPalette + the 2 px borders, PAINT_CHROME_H = title bar + MenuBar +
// ColorPalette + StatusBar + borders, DOC_INSET = the gray pad, the .bevel-in frame and the resize handles). Change
// any of those rows here and those constants have to move with them, or the gray dead band comes back.
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type RefObject } from "react";
import type { FlySocket } from "@/lib/ws";
import { useTickSnapshot } from "@/lib/store";
import type { InkStyle, Mood, PokeStim } from "@/lib/types";
import { DESK_MARGIN, DOC_FRAME, DOC_INSET, DOC_PAD, fitDocBox, paintHeightFor, type WinRect } from "@/lib/layout";
import Win95Window, { type WinGeometry } from "./Win95Window";
import MenuBar, { type MenuAction, type MenuFlags } from "./MenuBar";
import ToolPalette from "./ToolPalette";
import ColorPalette from "./ColorPalette";
import StatusBar from "./StatusBar";
import FlyCanvas, { type FlyCanvasHandle } from "./FlyCanvas";
import OfflineNotice from "./OfflineNotice";
import ShareDialog, { SHARE_TICK_FRESH_MS } from "./ShareDialog";

export interface PaintWindowProps {
  sock: FlySocket;
  onAction(a: MenuAction): void;
  canvasRef: RefObject<FlyCanvasHandle | null>;
  flags?: Partial<MenuFlags>;
  /** Computed layout rect (lib/layout.ts). Falls back to the historical PAINT_RECT. */
  rect?: WinRect;
  bounds?: WinRect;
  onGeometry?: (id: string, g: WinGeometry) => void;
}

/** Fallback rect when page.tsx does not pass one: 880 px wide and exactly as tall as its document needs. */
export const PAINT_RECT = { x: DESK_MARGIN, y: DESK_MARGIN, w: 880, h: paintHeightFor(880) };
const DEFAULT_INK: InkStyle = { color: "#000000", width: 2, alpha: 1, style: "solid", stamp: null };
/** Room the resize handles need beside/below the document frame (DOC_INSET = 2*DOC_PAD + frame + this). */
const DOC_HANDLE = 8;

// --------------------------------------------------------------------------------------- prefers-reduced-motion
// The CSS half of this lives in app/globals.css (it stills every CSS animation and transition). This is the JS half,
// for motion that CSS cannot reach - here, the rainbow mood swatch, whose hue is recomputed from the tick clock four
// times a second. Under "reduce" the phase is pinned to 0 and the swatch is simply a still colour. The simulation
// itself is never touched: the fly keeps flying and the trail keeps growing, they just do not strobe the chrome.
const MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function subscribeMotion(cb: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => undefined;
  const mq = window.matchMedia(MOTION_QUERY);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}
function getMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(MOTION_QUERY).matches;
}
const getMotionServer = (): boolean => false;

/** True when the visitor asked their OS for reduced motion. SSR renders the normal (animated) page. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribeMotion, getMotion, getMotionServer);
}

/** True when the socket is open AND a tick really landed in the last SHARE_TICK_FRESH_MS - i.e. the mood and the
 *  market number the share text would quote are current rather than left over from before an outage. Read from the
 *  store's own arrival stamp (recvAt), so it is right even when the dialog is opened long after the ticks stopped. */
function ticksAreLive(sock: FlySocket): boolean {
  if (sock.status !== "open") return false;
  const latest = sock.store.latest();
  return latest !== null && performance.now() - latest.recvAt < SHARE_TICK_FRESH_MS;
}

export default function PaintWindow({ sock, onAction, canvasRef, flags = {}, rect, bounds, onGeometry }: PaintWindowProps) {
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

  // File > Share to X... (and the Share button in the menu bar) never leave this window: PaintWindow owns the canvas
  // handle, so it catches the action and opens the dialog itself. Every other action goes on to page.tsx untouched.
  const [sharing, setSharing] = useState(false);
  const handleAction = useCallback((a: MenuAction) => {
    if (a.kind === "share") { setSharing(true); return; }
    onAction(a);
  }, [onAction]);
  const closeShare = useCallback(() => setSharing(false), []);
  const composite = useCallback(() => canvasRef.current?.composite() ?? null, [canvasRef]);
  const reduceMotion = usePrefersReducedMotion();

  const ink = tick?.ink ?? DEFAULT_INK;
  const freshMood = moodNow !== null && (!tick || tick.t_ms <= moodNow.t_ms);
  const mood: Mood = freshMood && moodNow ? moodNow.to : tick?.mood.state ?? "CRUISING";
  const W = sock.hello?.canvas.w ?? 800, H = sock.hello?.canvas.h ?? 500;
  const zoom = !!flags.zoom;

  // Measure the workspace; the document box follows it.
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [area, setArea] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const read = () => {
      const w = el.clientWidth, h = el.clientHeight;
      setArea((p) => (p.w === w && p.h === h ? p : { w, h }));
    };
    read();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", read);
      return () => window.removeEventListener("resize", read);
    }
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const doc = useMemo(() => {
    if (area.w <= 0 || area.h <= 0) return { w: W, h: H, scale: 1 };       // pre-measure first paint
    const fit = fitDocBox(area.w - DOC_INSET, area.h - DOC_INSET, W, H);
    // View > Zoom 1.5x magnifies past the fit; the workspace then scrolls (classic Paint behaviour).
    return zoom ? { w: Math.round(fit.w * 1.5), h: Math.round(fit.h * 1.5), scale: fit.scale * 1.5 } : fit;
  }, [area.w, area.h, W, H, zoom]);

  const frameW = doc.w + DOC_FRAME, frameH = doc.h + DOC_FRAME;

  return (
    <>
    <Win95Window
      id="paint"
      title="untitled - Paint ($SYNAPSE)"
      icon={
        /* eslint-disable-next-line @next/next/no-img-element */
        <img src="/favicon-48.png" alt="" width={16} height={16} style={{ imageRendering: "pixelated" }} />
      }
      rect={rect ?? PAINT_RECT}
      bounds={bounds}
      onGeometry={onGeometry}
      menu={<MenuBar sock={sock} onAction={handleAction} flags={flags} />}
      statusBar={<StatusBar store={sock.store} hello={sock.hello} status={sock.status} attempt={sock.attempt} fps={flags.fps ? fps : null} moodNow={moodNow} />}
      collapsible
    >
      {/* data-reduced-motion is the DOM mirror of usePrefersReducedMotion: it makes the JS half of the
          prefers-reduced-motion handling inspectable (and stylable) instead of invisible. */}
      <div className="flex h-full flex-col" data-reduced-motion={reduceMotion ? "true" : "false"}>
        <div className="flex min-h-0 flex-1">
          <ToolPalette active={active} onPoke={onPoke} lastPoke={lastPoke} inkWidth={ink.width} inkColor={ink.style === "rainbow" ? "#ff00ff" : ink.color} />
          <div ref={hostRef} className="relative flex min-h-0 min-w-0 flex-1 overflow-auto bg-win-gray" style={{ padding: DOC_PAD }}>
            {/* margin:auto (not justify-center) so the document stays reachable when zoom makes it overflow */}
            <div className="relative shrink-0" style={{ width: frameW + DOC_HANDLE, height: frameH + DOC_HANDLE, margin: "auto" }}>
              <div className="bevel-in paint-doc" style={{ width: frameW, height: frameH, lineHeight: 0 }} data-doc-scale={doc.scale.toFixed(3)}>
                <FlyCanvas ref={canvasRef} sock={sock} onFps={onFps} flip={!!flags.flip} pokeStim={active} />
              </div>
              {/* three inert resize handles (right, bottom, corner), glued to the scaled document */}
              <div className="absolute h-[3px] w-[3px] bg-win-blue" style={{ left: frameW + 2, top: frameH / 2 - 1 }} aria-hidden />
              <div className="absolute h-[3px] w-[3px] bg-win-blue" style={{ left: frameW / 2 - 1, top: frameH + 2 }} aria-hidden />
              <div className="absolute h-[3px] w-[3px] bg-win-blue" style={{ left: frameW + 2, top: frameH + 2 }} aria-hidden />
            </div>
          </div>
        </div>
        <ColorPalette ink={ink} mood={mood} tSec={reduceMotion ? 0 : tick ? tick.t_ms / 1000 : 0} />
      </div>
    </Win95Window>
    {/* Both are position:fixed, so they are siblings of the window (a transformed ancestor - the shake animation -
        would otherwise become their containing block) and neither can be clipped by the window's overflow. */}
    <OfflineNotice sock={sock} />
    {sharing ? (
      <ShareDialog composite={composite} tick={tick} neurons={sock.hello?.connectome.n ?? null}
        live={ticksAreLive(sock)} onClose={closeShare} />
    ) : null}
    </>
  );
}
