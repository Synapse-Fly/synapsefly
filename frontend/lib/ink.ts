// SPEC section e.4: ink drawing on the persistent trail layer + mood colours (MOOD_COLORS of section c.19).
import type { FlyPose } from "./interp";
import type { InkStyle, Mood } from "./types";

export const MOOD_COLORS: Record<Mood, string> = {
  SLEEP: "#6b6b6b", CRUISING: "#000000", FEEDING: "#ff8c00", EUPHORIA: "rainbow",
  ANXIOUS: "#ffd700", PANIC: "#ff0000", ESCAPE: "#ff3300", COURTSHIP: "#ff69b4",
};

/** MOOD_COLORS of section c.19; "rainbow" -> hsl((tSec*90) % 360, 100%, 45%). */
export function moodColor(mood: Mood, tSec: number): string {
  const c = MOOD_COLORS[mood] ?? "#000000";
  if (c === "rainbow") return `hsl(${((tSec * 90) % 360 + 360) % 360}, 100%, 45%)`;
  return c;
}

/** Zigzag phase and accumulated path length for the dotted / hearts cadence, kept PER trail layer (not per module,
 *  so a second canvas - e.g. a snapshot preview - cannot desync the cadence of the live one). */
interface InkState { zig: number; dotAcc: number; heartAcc: number }
const inkStates = new WeakMap<CanvasRenderingContext2D, InkState>();

function inkStateOf(ctx: CanvasRenderingContext2D): InkState {
  let s = inkStates.get(ctx);
  if (!s) { s = { zig: 1, dotAcc: 0, heartAcc: 0 }; inkStates.set(ctx, s); }
  return s;
}

/** Reset the cadence of one trail layer (called by FlyCanvas.clear()). */
export function resetInkState(ctx: CanvasRenderingContext2D): void {
  inkStates.set(ctx, { zig: 1, dotAcc: 0, heartAcc: 0 });
}

const HEART_PATTERN = [
  ".XX.XX.",
  "XXXXXXX",
  "XXXXXXX",
  ".XXXXX.",
  "..XXX..",
  "...X...",
];

/** Pixel heart of `size` px width (pattern 7 wide), centred on (cx, cy). */
export function drawPixelHeart(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number, color: string): void {
  const px = size / 7;
  const w = 7 * px, h = 6 * px;
  const x0 = Math.round(cx - w / 2), y0 = Math.round(cy - h / 2);
  ctx.fillStyle = color;
  for (let r = 0; r < HEART_PATTERN.length; r++) {
    const row = HEART_PATTERN[r];
    for (let c = 0; c < row.length; c++) {
      if (row[c] === "X") ctx.fillRect(Math.round(x0 + c * px), Math.round(y0 + r * px), Math.ceil(px), Math.ceil(px));
    }
  }
}

const MAX_SEGMENT_PX = 40;

export function drawInk(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, ink: InkStyle, tSec: number): void {
  // A malformed frame must never reach the canvas calls: NaN fails every comparison below, so guard explicitly
  // (Math.hypot(NaN, 0) is NaN and would pass both the `=== 0` and the `> MAX` test).
  if (!Number.isFinite(x0) || !Number.isFinite(y0) || !Number.isFinite(x1) || !Number.isFinite(y1)) return;
  if (!Number.isFinite(ink.width) || !Number.isFinite(ink.alpha)) return;
  const dx = x1 - x0, dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len === 0 || len > MAX_SEGMENT_PX) return;            // teleport / wrap / reconnect: skipped
  if (ink.alpha <= 0) return;
  const state = inkStateOf(ctx);
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = ink.width;
  ctx.globalAlpha = ink.alpha;
  const color = ink.style === "rainbow" ? moodColor("EUPHORIA", tSec) : ink.color;
  switch (ink.style) {
    case "solid":
    case "rainbow": {
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      break;
    }
    case "zigzag": {
      // two half-segments whose midpoint is offset +-3 px perpendicular, alternating each call
      const nx = -dy / len, ny = dx / len;
      const mx = (x0 + x1) / 2 + nx * 3 * state.zig;
      const my = (y0 + y1) / 2 + ny * 3 * state.zig;
      state.zig = -state.zig;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(mx, my);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      break;
    }
    case "dotted": {
      // 2 px dots every 6 px of accumulated path length
      ctx.fillStyle = color;
      let s = state.dotAcc;
      const ux = dx / len, uy = dy / len;
      let d = 6 - s;
      while (d <= len) {
        ctx.fillRect(Math.round(x0 + ux * d) - 1, Math.round(y0 + uy * d) - 1, 2, 2);
        d += 6;
      }
      s = (s + len) % 6;
      state.dotAcc = s;
      break;
    }
    case "hearts": {
      // 7 px pixel heart every 14 px of accumulated path length
      const ux = dx / len, uy = dy / len;
      let d = 14 - state.heartAcc;
      while (d <= len) {
        drawPixelHeart(ctx, x0 + ux * d, y0 + uy * d, 7, "#ff69b4");
        d += 14;
      }
      state.heartAcc = (state.heartAcc + len) % 14;
      break;
    }
  }
  ctx.restore();
}

export function drawStamp(ctx: CanvasRenderingContext2D, x: number, y: number, ink: InkStyle, pose: FlyPose, tSec: number): void {
  if (!ink.stamp) return;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;                 // malformed frame: draw nothing
  ctx.save();
  switch (ink.stamp) {
    case "blob": {
      ctx.globalAlpha = Math.max(0.2, ink.alpha);
      ctx.fillStyle = ink.color;
      ctx.beginPath();
      ctx.arc(x, y, 2 + 8 * (Number.isFinite(pose.proboscis) ? pose.proboscis : 0), 0, Math.PI * 2);
      ctx.fill();
      break;
    }
    case "heart": {
      drawPixelHeart(ctx, x, y - 10, 9, "#ff69b4");
      break;
    }
    case "zzz": {
      // "z" text 8 px rising 6 px per 0.5 s
      const rise = ((tSec * 2) % 1) * 6;
      ctx.font = "bold 8px Tahoma, Arial, sans-serif";
      ctx.fillStyle = ink.color;
      ctx.textBaseline = "alphabetic";
      ctx.fillText("z", Math.round(x + 8), Math.round(y - 8 - rise));
      break;
    }
    case "dash": {
      // nothing on the trail (the jump segment is skipped); the overlay draws the motion streaks
      break;
    }
    case "bump": {
      // 1 px, 6-px-long tick mark on the wall side nearest the fly
      const W = ctx.canvas.width, H = ctx.canvas.height;
      const dl = x, dr = W - x, dt = y, db = H - y;
      const m = Math.min(dl, dr, dt, db);
      ctx.strokeStyle = ink.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (m === dl) { ctx.moveTo(1.5, Math.round(y) - 3); ctx.lineTo(1.5, Math.round(y) + 3); }
      else if (m === dr) { ctx.moveTo(W - 1.5, Math.round(y) - 3); ctx.lineTo(W - 1.5, Math.round(y) + 3); }
      else if (m === dt) { ctx.moveTo(Math.round(x) - 3, 1.5); ctx.lineTo(Math.round(x) + 3, 1.5); }
      else { ctx.moveTo(Math.round(x) - 3, H - 1.5); ctx.lineTo(Math.round(x) + 3, H - 1.5); }
      ctx.stroke();
      break;
    }
  }
  ctx.restore();
}
