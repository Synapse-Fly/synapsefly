// SPEC section e.4 `RasterPainter`: oscilloscope-style scrolling spike raster on a single 2D canvas.
// 8 lanes in REGIONS order (section 0.1), one 1 px row per sampled neuron (`hello.raster.rows`), star rows 2 px tall
// with a label in the 60 px left gutter, per-lane rate in the right gutter, black background. The bitmap itself is
// the state: `frame(now)` shifts it left by `elapsed * pxPerS` px with one drawImage self-copy and clears the
// revealed strip; `push(tick)` plots `tick.spikes` at the right edge. No React, no allocations per frame.
//
// Geometry (two declared deviations from section e.4, which sketches a fixed 480 x (8 x (per_region + 12)) bitmap):
//  1. The 60 px label gutter (spec) and the 44 px "x.x Hz" gutter are *carved out* of the plot area instead of
//     overlaying it, so the newest spike column and the oldest one are both visible. The spike x formula is therefore
//     `plotRight - 1 - (win_ms - dt*dt_ms)*pxPerS/1000` where the spec writes `W - 1 - ...`.
//  2. The bitmap width follows the host: `min(host, RASTER_GUTTER_L + windowS*pxPerS + RASTER_GUTTER_R)`, never below
//     `RASTER_MIN_W` (480, the spec number). `windowS` (8 s) is always honoured exactly - the effective scroll speed is
//     `plotWidth / windowS`, which is exactly the documented 60 px/s when the plot gets its full `windowS*pxPerS` px.
// The height is the full section e.4 height `8 * (per_region + 12)` and the canvas is sized in css px 1:1, so no raster
// row and no labelled star row is ever dropped; the section e.7 window is shorter and its host (overflow-auto) scrolls.
// Star rows are 2 px tall, so their second pixel lands on the next slot's row (on the next lane's header strip for the
// last slot of a lane) - that overlap is inherent in the `y = laneTop + slot % per_region` geometry of section e.4.
import type { HelloMsg, RasterRow, TickMsg } from "./types";

/** Lane colours in REGIONS order (section e.4). */
export const LANE_COLORS: readonly string[] = [
  "#7fdbff", // optic_lobe
  "#b10dc9", // antennal_lobe
  "#ffdc00", // mushroom_body
  "#2ecc40", // central_complex
  "#ff851b", // sez
  "#aaaaaa", // central_other
  "#ff4136", // descending_motor
  "#39cccc", // vnc
];

/** Short lane titles that fit the 60 px gutter at 8 px monospace (REGIONS order). */
export const LANE_TITLES: readonly string[] = [
  "optic lobe", "antennal", "mushroom", "central cx", "SEZ", "central", "descending", "VNC",
];

export const RASTER_MIN_W = 480;   // narrowest bitmap (section e.4: 480 x ...)
export const RASTER_GUTTER_L = 60; // left gutter: lane title + star labels
export const RASTER_GUTTER_R = 44; // right gutter: "x.x Hz" per lane
export const RASTER_HEADER = 12;   // px strip above each lane (8 lanes x 12 px)
const N_LANES = 8;
const FONT_SMALL = "8px 'Lucida Console', Consolas, monospace";
const FONT_HZ = "9px 'Lucida Console', Consolas, monospace";
const LABEL_MIN_GAP = 8;           // px between two star labels in the gutter
const MAX_GAP_S = 2;               // a rAF gap longer than this (tab throttled) is clamped

export interface RasterOpts { windowS?: number /* 8 */; pxPerS?: number /* 60 */ }

/** Bitmap height in px for `per_region` sampled rows per lane: `8 * (per_region + 12)` (section e.4). */
export function rasterHeight(perRegion: number): number {
  return N_LANES * (Math.max(1, perRegion | 0) + RASTER_HEADER);
}

export class RasterPainter {
  readonly canvas: HTMLCanvasElement;
  readonly rows: readonly RasterRow[];
  readonly per: number;
  readonly dtMs: number;
  /** Seconds of history the plot area shows (section e.4 `windowS`, default 8) - exact at every width. */
  readonly windowS: number;
  /** Reference scroll speed (section e.4 `pxPerS`, default 60): sets the widest useful bitmap. */
  readonly refPxPerS: number;
  private readonly ctx: CanvasRenderingContext2D;
  private W = 0;
  private H = 0;
  private scroll = 60;              // effective px/s = plot width / windowS
  private lastNow: number | null = null;
  private acc = 0;                  // fractional pixels not yet shifted
  private frozen = false;
  private labels = true;
  private lastSeq = -1;
  private regionHz: number[] = new Array<number>(N_LANES).fill(0);

  constructor(canvas: HTMLCanvasElement, hello: HelloMsg, opts?: RasterOpts) {
    this.canvas = canvas;
    this.rows = Array.isArray(hello?.raster?.rows) ? hello.raster.rows : [];
    this.per = Math.max(1, (hello?.raster?.per_region ?? 0) | 0);
    this.dtMs = hello?.dt_ms > 0 ? hello.dt_ms : 1;
    const ws = opts?.windowS;
    const px = opts?.pxPerS;
    this.windowS = ws !== undefined && ws > 0 ? ws : 8;
    this.refPxPerS = px !== undefined && px > 0 ? px : 60;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("RasterPainter: 2D context unavailable");
    this.ctx = ctx;
    this.resize();
  }

  /** Canvas height in px = 8 * (per_region + 12). */
  get height(): number { return this.H; }
  get width(): number { return this.W; }
  /** Effective scroll speed (px/s) for the current width: `plotWidth / windowS`. */
  get pxPerS(): number { return this.scroll; }
  /** Left edge of the scrolling plot area. */
  get plotLeft(): number { return RASTER_GUTTER_L; }
  /** Right edge (exclusive) of the scrolling plot area. */
  get plotRight(): number { return this.W - RASTER_GUTTER_R; }

  laneTop(lane: number): number { return lane * (this.per + RASTER_HEADER); }
  laneRow0(lane: number): number { return this.laneTop(lane) + RASTER_HEADER; }

  /**
   * Re-reads the host size (DPR 1). Width follows the host up to `windowS * refPxPerS` px of plot area, the height is
   * always the full `8 * (per_region + 12)`; the css size is pinned to the bitmap size so one raster row is one device
   * pixel and the host scrolls instead of the browser scaling the bitmap. Assigning width/height clears the bitmap, so
   * it only happens when the geometry actually changed.
   */
  resize(): void {
    const host = this.canvas.parentElement;
    const hostW = host ? Math.floor(host.clientWidth || 0) : 0;
    const prefW = RASTER_GUTTER_L + Math.round(this.windowS * this.refPxPerS) + RASTER_GUTTER_R;
    const W = Math.max(RASTER_MIN_W, Math.min(hostW > 0 ? hostW : prefW, prefW));
    const H = rasterHeight(this.per);
    this.scroll = (W - RASTER_GUTTER_L - RASTER_GUTTER_R) / this.windowS;
    if (W === this.W && H === this.H) return;
    this.W = W; this.H = H;
    this.canvas.width = W;
    this.canvas.height = H;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    this.acc = 0;
    this.paintAll();
  }

  setFrozen(f: boolean): void {
    this.frozen = f;
    if (!f) this.lastNow = null;                        // resume without a catch-up jump
  }

  setLabels(on: boolean): void {
    if (on === this.labels) return;
    this.labels = on;
    this.paintLeftGutter();
  }

  /**
   * Plots `tick.spikes` at the right edge (section e.4 formula) and refreshes the right gutter rates.
   * `ageOffsetMs` shifts every spike further left (brain ms): used when two ticks arrived between two frames, so the
   * older one lands one tick window behind the newer one instead of on top of it. Returns false when the tick was not
   * consumed (already plotted, or frozen), so the caller can re-offer it after unfreezing.
   * While frozen the bitmap is a screenshot: neither the spikes nor the Hz gutter move, so the image stays internally
   * consistent (section e.4 "frozen mode stops shifting (screenshots)").
   */
  push(tick: TickMsg, ageOffsetMs: number = 0): boolean {
    if (!tick || this.frozen || tick.seq === this.lastSeq) return false;
    this.lastSeq = tick.seq;
    const regions = tick.rates?.regions;
    if (Array.isArray(regions)) {
      for (let i = 0; i < N_LANES; i++) {
        const v = Number(regions[i]);
        this.regionHz[i] = Number.isFinite(v) ? v : 0;
      }
    }
    this.plotSpikes(tick, ageOffsetMs > 0 ? ageOffsetMs : 0);
    this.paintRightGutter();
    return true;
  }

  /** Shifts the plot area left by `elapsed * pxPerS` px (one drawImage self-copy) and clears the revealed strip. */
  frame(now: number): void {
    if (this.lastNow === null) { this.lastNow = now; return; }
    let elapsed = (now - this.lastNow) / 1000;
    this.lastNow = now;
    if (this.frozen) return;
    if (elapsed <= 0) return;
    if (elapsed > MAX_GAP_S) elapsed = MAX_GAP_S;
    this.acc += elapsed * this.scroll;
    const dx = Math.floor(this.acc);
    if (dx < 1) return;
    this.acc -= dx;
    const x0 = this.plotLeft, x1 = this.plotRight;
    const plotW = x1 - x0;
    const ctx = this.ctx;
    if (dx >= plotW) {
      ctx.fillStyle = "#000000";
      ctx.fillRect(x0, 0, plotW, this.H);
      this.paintSeparators(x0, plotW);
      return;
    }
    ctx.drawImage(this.canvas, x0 + dx, 0, plotW - dx, this.H, x0, 0, plotW - dx, this.H);
    ctx.fillStyle = "#000000";
    ctx.fillRect(x1 - dx, 0, dx, this.H);
    this.paintSeparators(x1 - dx, dx);
  }

  /**
   * Row under canvas-pixel `y` (hover tooltip); null in the header strips, outside the bitmap and on padding slots
   * (sections c.12 / d.1: a region smaller than `per_region` pads its lane with `neuron == -1`, `label == ""`).
   */
  rowAt(y: number): RasterRow | null {
    if (!(y >= 0) || y >= this.H) return null;
    const laneH = this.per + RASTER_HEADER;
    const lane = Math.floor(y / laneH);
    if (lane < 0 || lane >= N_LANES) return null;
    const yy = Math.floor(y) - lane * laneH - RASTER_HEADER;
    if (yy < 0 || yy >= this.per) return null;
    const slot = lane * this.per + yy;
    const row = this.rows[slot];
    if (row && row.region === lane && row.neuron >= 0) return row;
    // star rows are 2 px tall: the pixel below a star row still belongs to it
    const above = this.rows[slot - 1];
    if (yy > 0 && above && above.star && above.region === lane && above.neuron >= 0) return above;
    return null;
  }

  // ------------------------------------------------------------------ private painting helpers

  private paintAll(): void {
    const ctx = this.ctx;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, this.W, this.H);
    this.paintSeparators(0, this.W);
    this.paintLeftGutter();
    this.paintRightGutter();
  }

  /** Dim horizontal separator at the bottom of each header strip, restricted to [x, x + w). */
  private paintSeparators(x: number, w: number): void {
    const ctx = this.ctx;
    ctx.fillStyle = "#202020";
    for (let lane = 0; lane < N_LANES; lane++) {
      ctx.fillRect(x, this.laneRow0(lane) - 1, w, 1);
    }
  }

  private paintLeftGutter(): void {
    const ctx = this.ctx;
    const gw = this.plotLeft;
    ctx.save();
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, gw, this.H);
    ctx.beginPath();
    ctx.rect(0, 0, gw, this.H);
    ctx.clip();
    ctx.font = FONT_SMALL;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    for (let lane = 0; lane < N_LANES; lane++) {
      const top = this.laneTop(lane);
      ctx.fillStyle = LANE_COLORS[lane];
      ctx.fillText(LANE_TITLES[lane] ?? "", 2, top + RASTER_HEADER / 2);
      ctx.fillRect(0, this.laneRow0(lane) - 1, gw, 1);      // coloured separator in the gutter
      // star rows: tick mark always, label when enabled and there is room
      let lastLabelY = -Infinity;
      const s0 = lane * this.per, s1 = Math.min(this.rows.length, s0 + this.per);
      for (let slot = s0; slot < s1; slot++) {
        const row = this.rows[slot];
        if (!row || !row.star || row.region !== lane || row.neuron < 0) continue;
        const y = this.laneRow0(lane) + (row.slot % this.per);
        ctx.fillStyle = LANE_COLORS[lane];
        ctx.fillRect(gw - 4, y, 4, 2);
        if (!this.labels) continue;
        if (y - lastLabelY < LABEL_MIN_GAP) continue;
        lastLabelY = y;
        ctx.fillStyle = "#e0e0e0";
        ctx.fillText(rowLabel(row), 2, y + 1);
      }
    }
    ctx.restore();
  }

  private paintRightGutter(): void {
    const ctx = this.ctx;
    const x = this.plotRight;
    const w = this.W - x;
    ctx.save();
    ctx.fillStyle = "#000000";
    ctx.fillRect(x, 0, w, this.H);
    ctx.beginPath();
    ctx.rect(x, 0, w, this.H);
    ctx.clip();
    ctx.font = FONT_HZ;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (let lane = 0; lane < N_LANES; lane++) {
      ctx.fillStyle = LANE_COLORS[lane];
      ctx.fillRect(x, this.laneRow0(lane) - 1, w, 1);
      ctx.fillText(`${this.regionHz[lane].toFixed(1)} Hz`, this.W - 2, this.laneRow0(lane) + this.per / 2);
    }
    ctx.restore();
  }

  private plotSpikes(tick: TickMsg, ageOffsetMs: number): void {
    const sp = tick.spikes;
    if (!sp || !Array.isArray(sp.slots) || !Array.isArray(sp.dt)) return;
    const n = Math.min(sp.slots.length, sp.dt.length);
    if (n === 0) return;
    const ctx = this.ctx;
    const x1 = this.plotRight, x0 = this.plotLeft;
    const steps = Number(tick.sim?.steps) || 0;
    const winMs = sp.win_ms > 0 ? sp.win_ms : steps * this.dtMs;
    const scale = this.scroll / 1000;
    let curLane = -1;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, 0, x1 - x0, this.H);
    ctx.clip();
    for (let k = 0; k < n; k++) {
      const row = this.rows[Math.floor(Number(sp.slots[k]))];
      if (!row || row.neuron < 0) continue;
      const lane = row.region;
      if (lane < 0 || lane >= N_LANES) continue;
      const dtSteps = Number(sp.dt[k]);
      const ageMs = winMs - (Number.isFinite(dtSteps) ? dtSteps : 0) * this.dtMs + ageOffsetMs;
      let x = Math.floor(x1 - 1 - ageMs * scale);
      if (!Number.isFinite(x)) continue;
      if (x < x0) x = x0;
      if (x >= x1) x = x1 - 1;
      const y = this.laneRow0(lane) + (row.slot % this.per);
      if (lane !== curLane) { ctx.fillStyle = LANE_COLORS[lane]; curLane = lane; }
      ctx.fillRect(x, y, 1, row.star ? 2 : 1);
    }
    ctx.restore();
  }
}

/** Gutter label of a star row, e.g. "DNp01 R", "MN9", "PFL3 L" (section e.4). */
export function rowLabel(row: RasterRow): string {
  const label = typeof row.label === "string" ? row.label : "";
  const base = label.length > 9 ? label.slice(0, 9) : label;
  return row.side === "M" ? base : `${base} ${row.side}`;
}
