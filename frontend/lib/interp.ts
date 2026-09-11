// SPEC section e.3: pose interpolation between the two most recent ticks (one tick of latency, smooth at 60 fps).
import type { StoredTick } from "./store";
import type { FlyMode, FlyState } from "./types";

export interface FlyPose { x: number; y: number; heading: number; speed: number; wing_hz: number; wing_amp: number; wing_ext: -1|0|1;
  mode: FlyMode; leg_phase: number; proboscis: number; jump_t_ms: number|null }

const TWO_PI = Math.PI * 2;

/** Shortest-arc angle interpolation (radians). */
export function lerpAngle(a: number, b: number, t: number): number {
  let d = (b - a) % TWO_PI;
  if (d > Math.PI) d -= TWO_PI;
  else if (d < -Math.PI) d += TWO_PI;
  return a + d * t;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function poseOf(f: FlyState): FlyPose {
  return {
    x: f.x, y: f.y, heading: f.heading, speed: f.speed,
    wing_hz: f.wing_hz, wing_amp: f.wing_amp, wing_ext: f.wing_ext,
    mode: f.mode, leg_phase: f.leg_phase, proboscis: f.proboscis, jump_t_ms: f.jump_t_ms,
  };
}

// SPEC e.3 lists exactly two snap conditions - `prev == null` and a gap > 4 * tickMs. No distance guard is added
// here on purpose: a wrap/teleport is already skipped by drawInk's 40 px segment limit (e.4), and an extra snap
// would be behaviour a reader of e.3 cannot predict.
export function interpolate(prev: StoredTick|null, latest: StoredTick, now: number, tickMs: number): FlyPose {
  const lf = latest.tick.fly;
  if (!prev || tickMs <= 0) return poseOf(lf);
  const gap = latest.recvAt - prev.recvAt;
  if (gap > 4 * tickMs || gap < 0) return poseOf(lf);           // reconnect / background tab: snap, no long segment
  const pf = prev.tick.fly;
  const dx = lf.x - pf.x, dy = lf.y - pf.y;
  const a = clamp01((now - latest.recvAt) / tickMs);
  const pose = poseOf(lf);
  pose.x = pf.x + dx * a;
  pose.y = pf.y + dy * a;
  pose.heading = lerpAngle(pf.heading, lf.heading, a);
  pose.speed = pf.speed + (lf.speed - pf.speed) * a;
  // e.3 enumerates exactly x/y/heading/speed as lerped; everything else (mode, wing_*, proboscis, jump_t_ms and
  // leg_phase) is taken from `latest` - `poseOf(lf)` above already copied leg_phase, so no interpolation here.
  return pose;
}
