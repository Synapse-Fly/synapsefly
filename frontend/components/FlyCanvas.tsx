"use client";
// SPEC section e.5 `FlyCanvas`: two stacked 800 x 500 canvases (trail: persistent, overlay: cleared every frame).
// One requestAnimationFrame loop reads the mutable tick store (no React state per tick): for every new tick it draws
// the trail segment from the last drawn pose to the interpolated pose with drawInk(tick.ink), applies drawStamp when
// ink.stamp is set and handles tick.events (jump -> streaks on the overlay for 400 ms, wall_bump -> bump mark); then
// draws the fly at interpolate(prev, latest, now, tick_ms). Trail drawing is skipped while the fly is jumping. The
// `clear` event frame clears the trail. Pointer down on the document sends a poke with the active poke tool, the side
// derived from the click position relative to the fly heading, rate-limited to one per 500 ms client-side.
// Imperative handle: clear(), composite() (trail + overlay + a 24 px caption bar, for snapshots / Save As), dirty().
import { useCallback, useEffect, useImperativeHandle, useRef, type PointerEvent as ReactPointerEvent, type Ref } from "react";
import { TOKEN } from "@/lib/brand";
import type { FlySocket } from "@/lib/ws";
import type { FlyState, InkStyle, Mood, PokeStim } from "@/lib/types";
import { interpolate, type FlyPose } from "@/lib/interp";
import { drawFly, drawHourglass } from "@/lib/flySprite";
import { drawInk, drawStamp, resetInkState } from "@/lib/ink";
import { fmtPrice } from "@/lib/api";
import { getState, postPoke } from "@/lib/api";

export interface FlyCanvasHandle { clear(): void; composite(): HTMLCanvasElement; dirty(): boolean }
export interface FlyCanvasProps {
  sock: FlySocket;
  onFps?(fps: number): void;
  flip: boolean;
  pokeStim?: PokeStim;
  ref?: Ref<FlyCanvasHandle>;
}

export const CAPTION_H = 24;
const DEFAULT_W = 800, DEFAULT_H = 500;
const STREAK_MS = 400;
const POKE_MIN_INTERVAL_MS = 500;
const DEFAULT_TICK_MS = 50;
// A trail segment longer than this is a jump/teleport, not a stroke; a seq gap wider than this means rAF was
// throttled (hidden tab) and the ring of 4 has rolled, so the trail is repainted from the server instead.
const MAX_SEGMENT_PX = 60;
const MAX_SEQ_GAP = 3;

interface Streak { x: number; y: number; heading: number; until: number }

function poseOf(f: FlyState): FlyPose {
  return {
    x: f.x, y: f.y, heading: f.heading, speed: f.speed, wing_hz: f.wing_hz, wing_amp: f.wing_amp, wing_ext: f.wing_ext,
    mode: f.mode, leg_phase: f.leg_phase, proboscis: f.proboscis, jump_t_ms: f.jump_t_ms,
  };
}

function drawStreaks(ctx: CanvasRenderingContext2D, streaks: Streak[], now: number): void {
  for (const s of streaks) {
    const left = (s.until - now) / STREAK_MS;
    if (left <= 0) continue;
    ctx.save();
    ctx.globalAlpha = Math.max(0.1, Math.min(1, left));
    ctx.translate(s.x, s.y);
    ctx.rotate(s.heading);
    ctx.strokeStyle = "#404040";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    for (const dy of [-6, 0, 6]) {
      ctx.beginPath();
      ctx.moveTo(-12, dy);
      ctx.lineTo(-34 - (dy === 0 ? 6 : 0), dy);
      ctx.stroke();
    }
    ctx.restore();
  }
}

export default function FlyCanvas({ sock, onFps, flip, pokeStim = "sugar", ref }: FlyCanvasProps) {
  const store = sock.store;
  const hello = sock.hello;
  const W = hello?.canvas.w ?? DEFAULT_W;
  const H = hello?.canvas.h ?? DEFAULT_H;

  const trailRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const dirtyRef = useRef(false);
  const poseRef = useRef<FlyPose | null>(null);
  const lastPokeAt = useRef(-Infinity);
  // Values the rAF loop reads without restarting: socket status, flip flag, tick length, fps callback.
  const live = useRef({ status: sock.status, flip, tickMs: hello?.tick_ms ?? DEFAULT_TICK_MS, onFps });
  const tickMs = hello?.tick_ms ?? DEFAULT_TICK_MS;
  const status = sock.status;
  const { on } = sock;
  useEffect(() => {
    live.current.status = status;
    live.current.flip = flip;
    live.current.tickMs = tickMs;
    live.current.onFps = onFps;
  }, [status, flip, tickMs, onFps]);

  const clear = useCallback(() => {
    const trail = trailRef.current;
    if (!trail) return;
    const ctx = trail.getContext("2d");
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, trail.width, trail.height);
    ctx.restore();
    resetInkState(ctx);
    dirtyRef.current = false;
  }, []);

  const composite = useCallback((): HTMLCanvasElement => {
    const trail = trailRef.current, overlay = overlayRef.current;
    const w = trail?.width ?? W, h = trail?.height ?? H;
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h + CAPTION_H;
    const ctx = c.getContext("2d");
    if (!ctx) return c;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    if (trail) ctx.drawImage(trail, 0, 0);
    if (overlay) ctx.drawImage(overlay, 0, 0);
    // Paint-style caption bar
    ctx.fillStyle = "#c0c0c0";
    ctx.fillRect(0, h, w, CAPTION_H);
    ctx.fillStyle = "#808080";
    ctx.fillRect(0, h, w, 1);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, h + 1, w, 1);
    const t = store.latest()?.tick ?? null;
    const mood: Mood = t?.mood.state ?? "CRUISING";
    const sym = TOKEN;
    const price = fmtPrice(t?.market.price_usd ?? null);
    const chg = t ? `${t.market.chg_m5 >= 0 ? "+" : ""}${t.market.chg_m5.toFixed(1)}%` : "-";
    const secs = t ? (t.t_ms / 1000).toFixed(1) : "0.0";
    ctx.fillStyle = "#000000";
    ctx.font = '12px "MS Sans Serif", "Microsoft Sans Serif", Tahoma, Arial, sans-serif';
    ctx.textBaseline = "middle";
    // SPEC e.6 caption, verbatim: "FlyBrain | <mood> | <symbol> <price> <chg_m5>% | t=<t_ms/1000>s" (no "$" before
    // the price - the symbol stands in for it).
    ctx.fillText(`FlyBrain | ${mood} | ${sym} ${price} ${chg} | t=${secs}s`, 6, h + CAPTION_H / 2 + 1);
    return c;
  }, [store, hello, W, H]);

  useImperativeHandle(ref, () => ({ clear, composite, dirty: () => dirtyRef.current }), [clear, composite]);

  // (Re)size both canvases when the document size is known; assigning width clears the bitmap, so paint white again.
  useEffect(() => {
    for (const c of [trailRef.current, overlayRef.current]) {
      if (!c) continue;
      if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    }
    clear();
  }, [W, H, clear]);

  // The out-of-band `clear` event (from this or any other client, or the REST API) clears the trail layer.
  useEffect(() => on("event", (m) => { if (m.kind === "clear") clear(); }), [on, clear]);

  // The drawing loop.
  useEffect(() => {
    const trail = trailRef.current, overlay = overlayRef.current;
    if (!trail || !overlay) return;
    const tctx = trail.getContext("2d");
    const octx = overlay.getContext("2d");
    if (!tctx || !octx) return;
    tctx.imageSmoothingEnabled = false;
    octx.imageSmoothingEnabled = false;

    let raf = 0;
    let lastSeq = -1;
    let lastDrawn: { x: number; y: number } | null = null;
    let streaks: Streak[] = [];
    let frames = 0;
    let fpsT0 = performance.now();
    let backfilling = false;
    let disposed = false;

    // The trail only ever grows inside this rAF loop, but a hidden tab throttles rAF to zero while the socket keeps
    // delivering ticks into the ring of 4 - so minutes of painting would be lost and replaced by one straight chord.
    // The server keeps the authoritative trail (GET /api/state, SPEC c.28), so on a seq gap, on becoming visible and
    // once on mount (which also restores the painting across an F5) we repaint the whole trail from it.
    const backfillTrail = async (reason: string) => {
      if (backfilling || disposed) return;
      backfilling = true;
      lastDrawn = null; // no segment is drawn from a stale pose while the repaint is in flight
      try {
        const pts = (await getState()).trail ?? [];
        if (disposed || pts.length === 0) return;
        tctx.clearRect(0, 0, trail.width, trail.height);
        resetInkState(tctx);
        const tSec = performance.now() / 1000;
        for (let i = 1; i < pts.length; i++) {
          const [x0, y0] = pts[i - 1];
          const [x1, y1, color, width] = pts[i];
          if (Math.hypot(x1 - x0, y1 - y0) > MAX_SEGMENT_PX) continue; // jump / wrap: not a painted stroke
          drawInk(tctx, x0, y0, x1, y1, { color, width, alpha: 1, style: "solid", stamp: null }, tSec);
        }
        const last = pts[pts.length - 1];
        lastDrawn = { x: last[0], y: last[1] };
        dirtyRef.current = true;
      } catch (e) {
        console.warn(`[canvas] trail backfill (${reason}) failed`, e);
      } finally {
        backfilling = false;
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") void backfillTrail("visible");
    };
    document.addEventListener("visibilitychange", onVisibility);
    void backfillTrail("mount");

    const handleNewTick = (seqPrev: number, now: number) => {
      const latest = store.latest();
      if (!latest) return;
      const prev = store.prev();
      const fresh: typeof latest[] = [];
      if (prev && prev.tick.seq > seqPrev && prev.tick.seq < latest.tick.seq) fresh.push(prev);
      fresh.push(latest);
      const tSec = now / 1000;
      for (const st of fresh) {
        const tick = st.tick;
        const fly = tick.fly;
        const ink: InkStyle = tick.ink;
        const pose = st === latest ? interpolate(prev, latest, now, live.current.tickMs) : poseOf(fly);
        const jumping = pose.mode === "jump" || fly.mode === "jump";
        if (lastDrawn && !jumping && ink.alpha > 0) {
          const before = dirtyRef.current;
          drawInk(tctx, lastDrawn.x, lastDrawn.y, pose.x, pose.y, ink, tSec);
          if (!before) dirtyRef.current = Math.hypot(pose.x - lastDrawn.x, pose.y - lastDrawn.y) > 0 && Math.hypot(pose.x - lastDrawn.x, pose.y - lastDrawn.y) <= 40;
        }
        if (ink.stamp && ink.stamp !== "dash") {
          drawStamp(tctx, pose.x, pose.y, ink, pose, tSec);
          dirtyRef.current = true;
        }
        for (const ev of tick.events ?? []) {
          if (ev.kind === "jump") {
            const hd = typeof ev.data?.heading_out === "number" ? ev.data.heading_out : fly.heading;
            streaks.push({ x: fly.x, y: fly.y, heading: hd, until: now + STREAK_MS });
          } else if (ev.kind === "wall_bump") {
            drawStamp(tctx, fly.x, fly.y, { ...ink, stamp: "bump" }, pose, tSec);
          }
        }
        lastDrawn = { x: pose.x, y: pose.y };
      }
    };

    const loop = (now: number) => {
      try {
        octx.clearRect(0, 0, overlay.width, overlay.height);
        const latest = store.latest();
        if (latest) {
          if (latest.tick.seq !== lastSeq) {
            const gap = lastSeq >= 0 ? latest.tick.seq - lastSeq : 0;
            if (gap > MAX_SEQ_GAP) void backfillTrail(`seq gap ${gap}`);
            else handleNewTick(lastSeq, now);
            lastSeq = latest.tick.seq;
          }
          const prev = store.prev();
          const pose = interpolate(prev, latest, now, live.current.tickMs);
          poseRef.current = pose;
          const tSec = now / 1000;
          const mood = latest.tick.mood.state;
          if (streaks.length) {
            drawStreaks(octx, streaks, now);
            streaks = streaks.filter((s) => s.until > now);
          }
          if (live.current.flip) {
            octx.save();
            octx.translate(pose.x, pose.y);
            octx.scale(1, -1);
            octx.translate(-pose.x, -pose.y);
            drawFly(octx, { ...pose, heading: -pose.heading }, tSec, mood, 2);
            octx.restore();
          } else {
            drawFly(octx, pose, tSec, mood, 2);
          }
          if (live.current.status !== "open") drawHourglass(octx, pose.x + 18, pose.y - 26);
        } else {
          octx.save();
          octx.fillStyle = "#808080";
          octx.font = '12px "MS Sans Serif", Tahoma, Arial, sans-serif';
          octx.fillText(live.current.status === "open" ? "waiting for the first tick..." : `brain offline - ${live.current.status}`, 8, 16);
          octx.restore();
          drawHourglass(octx, overlay.width / 2 - 5, overlay.height / 2 - 7);
        }
        frames += 1;
        if (now - fpsT0 >= 1000) {
          const fps = Math.round((frames * 1000) / (now - fpsT0));
          frames = 0;
          fpsT0 = now;
          live.current.onFps?.(fps);
        }
      } catch (e) {
        console.error("[canvas] frame failed", e);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      disposed = true;
      document.removeEventListener("visibilitychange", onVisibility);
      cancelAnimationFrame(raf);
    };
  }, [store]);

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const now = performance.now();
    if (now - lastPokeAt.current < POKE_MIN_INTERVAL_MS) return;
    lastPokeAt.current = now;
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const pose = poseRef.current;
    let side: "L" | "R" | "both" = "both";
    if (pose) {
      const hx = Math.cos(pose.heading), hy = Math.sin(pose.heading);
      const dx = x - pose.x, dy = y - pose.y;
      const cross = hx * dy - hy * dx;          // screen y points down, heading clockwise-positive: cross > 0 = right side
      side = cross > 0 ? "R" : "L";
    }
    const stim = pokeStim;
    if (!sock.send({ type: "poke", stim, strength: 1, side, duration_ms: 500 })) {
      postPoke(stim, 1, side, 500).catch((err) => console.warn("[canvas] poke fallback failed", err));
    }
  }, [sock, pokeStim]);

  return (
    <div className="relative bg-paint-white" style={{ width: W, height: H, lineHeight: 0 }} data-testid="fly-canvas">
      <canvas ref={trailRef} width={W} height={H} className="absolute left-0 top-0 block" style={{ width: "100%", height: "100%", imageRendering: "pixelated" }} aria-hidden />
      <canvas
        ref={overlayRef}
        width={W}
        height={H}
        className="absolute left-0 top-0 block"
        style={{ width: "100%", height: "100%", imageRendering: "pixelated", cursor: "crosshair", touchAction: "none" }}
        onPointerDown={onPointerDown}
        aria-label="FlyBrain canvas: click to poke the fly"
        role="img"
      />
    </div>
  );
}
