// SPEC section e.4: procedural pixel fly (Paint look: flat colours, no gradients), drawn at `scale` and rotated by
// pose.heading around the thorax. Body frame: +x = forward, +y = the fly's right side (screen y points down).
import type { FlyPose } from "./interp";
import type { Mood } from "./types";

const C_BODY = "#3b2a1a";
const C_STRIPE = "#6b4a2a";
const C_THORAX = "#4a3620";
const C_EYE = "#c02020";
const C_LEG = "#2a1a0a";
const C_WING = "rgba(200,220,255,0.55)";
const C_WING_BLUR = "rgba(200,220,255,0.28)";
const C_WING_EDGE = "rgba(120,140,180,0.6)";
const DEG = Math.PI / 180;
const TWO_PI = Math.PI * 2;

/** Frame counter + sparkle position kept PER canvas context, so a second canvas (e.g. a snapshot preview) cannot
 *  desync the EUPHORIA sparkle cadence of the live overlay. */
interface SpriteState { frame: number; bx: number; by: number }
const spriteStates = new WeakMap<CanvasRenderingContext2D, SpriteState>();

function spriteStateOf(ctx: CanvasRenderingContext2D): SpriteState {
  let s = spriteStates.get(ctx);
  if (!s) { s = { frame: 0, bx: 0, by: 0 }; spriteStates.set(ctx, s); }
  return s;
}

function ellipse(ctx: CanvasRenderingContext2D, cx: number, cy: number, rx: number, ry: number, rot: number, fill: string, stroke?: string): void {
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, rot, 0, TWO_PI);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 0.5; ctx.stroke(); }
}

function leg(ctx: CanvasRenderingContext2D, ax: number, ay: number, angle: number, l1: number, l2: number, bend: number): void {
  const kx = ax + Math.cos(angle) * l1, ky = ay + Math.sin(angle) * l1;
  const a2 = angle + bend;
  const fx = kx + Math.cos(a2) * l2, fy = ky + Math.sin(a2) * l2;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(kx, ky);
  ctx.lineTo(fx, fy);
  ctx.stroke();
}

/** One wing: attach point (ax, ay), `side` -1 left / +1 right, `thetaDeg` measured from the backward axis. */
function wing(ctx: CanvasRenderingContext2D, ax: number, ay: number, side: number, thetaDeg: number, fill: string): void {
  const dir = Math.PI - side * thetaDeg * DEG;
  const cx = ax + Math.cos(dir) * 6, cy = ay + Math.sin(dir) * 6;
  ellipse(ctx, cx, cy, 6, 2.5, dir, fill, C_WING_EDGE);
}

export function drawFly(ctx: CanvasRenderingContext2D, pose: FlyPose, tSec: number, mood: Mood, scale: number = 2): void {
  const st = spriteStateOf(ctx);
  st.frame++;
  ctx.save();
  ctx.imageSmoothingEnabled = false;

  let ox = 0, oy = 0;
  if (mood === "PANIC") {                        // shake +-1 px at 30 Hz
    const ph = TWO_PI * 30 * tSec;
    ox = Math.sin(ph) > 0 ? 1 : -1;
    oy = Math.cos(ph) > 0 ? 1 : -1;
  }

  // jump shadow (screen space, under the body)
  if (pose.mode === "jump") {
    const d = Math.min(8, (pose.jump_t_ms ?? 0) / 50);
    ctx.save();
    ctx.translate(pose.x + ox + d, pose.y + oy + d);
    ctx.rotate(pose.heading);
    ctx.scale(scale, scale);
    ellipse(ctx, -2, 0, 12, 4, 0, "rgba(0,0,0,0.25)");
    ctx.restore();
  }

  ctx.translate(pose.x + ox, pose.y + oy);
  ctx.rotate(pose.heading);
  ctx.scale(scale, scale);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // ESCAPE: three motion lines behind the fly
  if (mood === "ESCAPE") {
    ctx.strokeStyle = "#555555";
    ctx.lineWidth = 1;
    for (const y of [-4, 0, 4]) {
      ctx.beginPath(); ctx.moveTo(-15, y); ctx.lineTo(-22 - (y === 0 ? 3 : 0), y); ctx.stroke();
    }
  }

  // wings (drawn under the body)
  const tucked = pose.mode === "sleep" || pose.mode === "feed" || pose.mode === "freeze";
  const wingAttach = 1.5;
  if (pose.wing_hz > 0) {
    const ph = TWO_PI * pose.wing_hz * tSec;
    const t1 = 20 + 25 * pose.wing_amp * Math.sin(ph);
    const t2 = 20 + 25 * pose.wing_amp * Math.sin(ph + Math.PI);
    for (const s of [-1, 1]) {
      wing(ctx, -1, s * wingAttach, s, t1, C_WING_BLUR);
      wing(ctx, -1, s * wingAttach, s, t2, C_WING_BLUR);
    }
    ctx.beginPath(); ctx.arc(0, 0, 9, 0, TWO_PI); ctx.fillStyle = "rgba(200,220,255,0.12)"; ctx.fill();   // beat envelope
  } else {
    for (const s of [-1, 1]) {
      let theta = 20;
      if (pose.wing_ext !== 0 && pose.wing_ext === s) theta = 70 + 4 * Math.sin(TWO_PI * 25 * tSec);   // song: held out, jittered
      wing(ctx, -1, s * wingAttach, s, theta, C_WING);
    }
  }

  // legs: six 1.5 px two-segment polylines, tripod gait (legs 1,3,5 at phase, 2,4,6 at phase+0.5)
  ctx.strokeStyle = C_LEG;
  ctx.lineWidth = 1.5;
  const attach: [number, number, number][] = [[3, 2.5, 55], [0, 3.2, 95], [-3, 2.5, 135]];   // x, |y|, base angle (deg, right side)
  const walking = pose.mode === "walk" || pose.mode === "court" || pose.mode === "fly" || pose.mode === "jump";
  let idx = 0;
  for (const s of [-1, 1]) {
    for (let k = 0; k < 3; k++) {
      idx++;
      const [ax, ay, base] = attach[k];
      const phase = (idx % 2 === 1) ? pose.leg_phase : pose.leg_phase + 0.5;
      let ang = s * base * DEG;
      let l1 = 5, l2 = 5, bend = s * 35 * DEG;
      if (tucked) {
        l1 = 3; l2 = 2; bend = s * 120 * DEG; ang = s * (base + 10) * DEG;
      } else if (pose.mode === "groom" && k === 0) {
        ang = s * (20 + 25 * Math.sin(TWO_PI * 10 * tSec)) * DEG;                                // front pair rubbing at 10 Hz
      } else if (walking) {
        ang += (pose.mode === "fly" ? 10 : 25 * Math.sin(TWO_PI * phase)) * DEG * (k === 1 ? 1 : (k === 0 ? 1 : -1));
        if (pose.mode === "fly" || pose.mode === "jump") { l1 = 4; l2 = 4; bend = s * 60 * DEG; }
      }
      leg(ctx, ax, s * ay, ang, l1, l2, bend);
    }
  }

  // abdomen (12x6) with three lighter stripes
  ellipse(ctx, -9, 0, 6, 3, 0, C_BODY);
  ctx.strokeStyle = C_STRIPE;
  ctx.lineWidth = 1;
  for (const sx of [-7, -9.5, -12]) {
    const hh = Math.sqrt(Math.max(0, 1 - ((sx + 9) / 6) ** 2)) * 3;
    ctx.beginPath(); ctx.moveTo(sx, -hh + 0.4); ctx.lineTo(sx, hh - 0.4); ctx.stroke();
  }
  // thorax (10x7)
  ellipse(ctx, 0, 0, 5, 3.5, 0, C_THORAX);
  // head (r 4) + compound eyes (r 1.5)
  ellipse(ctx, 8, 0, 4, 4, 0, C_BODY);
  ellipse(ctx, 9.5, -2.3, 1.5, 1.5, 0, C_EYE);
  ellipse(ctx, 9.5, 2.3, 1.5, 1.5, 0, C_EYE);

  // proboscis
  if (pose.proboscis > 0) {
    const len = 8 * pose.proboscis;
    ctx.strokeStyle = C_BODY;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(12, 0); ctx.lineTo(12 + len, 0); ctx.stroke();
    if (pose.mode === "feed") {
      ellipse(ctx, 12 + len + 1, 0, 3, 3, 0, "#ffd700");
      ellipse(ctx, 12 + len, -1, 1, 1, 0, "#ffffff");
    }
  }

  // EUPHORIA: 4-point sparkle at a random point on the body, re-rolled every 5th frame
  if (mood === "EUPHORIA") {
    if (st.frame % 5 === 0) { st.bx = -14 + Math.random() * 26; st.by = -3 + Math.random() * 6; }
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    const { bx, by } = st;
    ctx.beginPath();
    ctx.moveTo(bx - 3, by); ctx.lineTo(bx + 3, by);
    ctx.moveTo(bx, by - 3); ctx.lineTo(bx, by + 3);
    ctx.stroke();
    ctx.fillStyle = "#ffff00";
    ctx.fillRect(bx - 0.5, by - 0.5, 1, 1);
  }
  ctx.restore();

  // SLEEP: three rising "z" glyphs, one every 0.5 s (screen space)
  if (pose.mode === "sleep") {
    ctx.save();
    ctx.font = `bold ${Math.round(5 * scale)}px Tahoma, Arial, sans-serif`;
    ctx.fillStyle = "#333333";
    for (let k = 0; k < 3; k++) {
      const age = ((tSec + k * 0.5) % 1.5 + 1.5) % 1.5;
      ctx.globalAlpha = Math.max(0, 1 - age / 1.5);
      ctx.fillText("z", Math.round(pose.x + 4 * scale + age * 8), Math.round(pose.y - 5 * scale - age * 12));
    }
    ctx.restore();
  }
}

const HOURGLASS = [
  "XXXXXXXXXXX",
  ".X.......X.",
  ".X.sssss.X.",
  ".X.sssss.X.",
  "..X.sss.X..",
  "...X.s.X...",
  "....X.X....",
  "....X.X....",
  "...X...X...",
  "..X.....X..",
  ".X...s...X.",
  ".X..sss..X.",
  ".X.sssss.X.",
  ".X.......X.",
  "XXXXXXXXXXX",
];

/** Win95 wait cursor (11x15 px) with its top-left corner at (x, y): drawn beside the fly while the socket is not open. */
export function drawHourglass(ctx: CanvasRenderingContext2D, x: number, y: number): void {
  ctx.save();
  const x0 = Math.round(x), y0 = Math.round(y);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(x0, y0, 11, 15);
  for (let r = 0; r < HOURGLASS.length; r++) {
    const row = HOURGLASS[r];
    for (let c = 0; c < row.length; c++) {
      const ch = row[c];
      if (ch === ".") continue;
      ctx.fillStyle = ch === "X" ? "#000000" : "#7cc0ff";
      ctx.fillRect(x0 + c, y0 + r, 1, 1);
    }
  }
  ctx.restore();
}
