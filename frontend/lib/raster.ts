// SPEC section e.4 `RasterPainter`: oscilloscope-style scrolling spike raster on a single 2D canvas.
// 8 lanes in REGIONS order (section 0.1), one row per sampled neuron (`hello.raster.rows`), star rows a pixel thicker
// with a label in the 60 px left gutter, per-lane rate in the right gutter, black background. The bitmap itself is
// the state: `frame(now)` shifts it left by `elapsed * pxPerS` px with one drawImage self-copy and clears the
// revealed strip; `push(tick)` plots `tick.spikes` at the right edge. No React, no allocations per frame.
//
// Geometry (three declared deviations from section e.4, which sketches a fixed 480 x (8 x (per_region + 12)) bitmap):
//  1. The 60 px label gutter (spec) and the 44 px "x.x Hz" gutter are *carved out* of the plot area instead of
//     overlaying it, so the newest spike column and the oldest one are both visible. The spike x formula is therefore
//     `plotRight - 1 - (win_ms - dt*dt_ms)*pxPerS/1000` where the spec writes `W - 1 - ...`.
//  2. THE BITMAP FILLS ITS PANE ON BOTH AXES *DOWN TO ONE DEVICE ROW PER SAMPLED NEURON*, and no further. The fixed
//     8*(per_region+12) height left a ~95 px inert strip under the raster in a three-column (>= 2200 px) layout, so a
//     pane taller than that natural height now gets the extra pixels (thicker rows with black between them, pitch
//     `laneRowSpace / per_region` > 1). A pane SHORTER than it keeps the natural height and scrolls - the host pane is
//     `overflow-y: auto` - because squeezing 48 neurons into the 8 px a 160 px pane can spare (the 2-column layout at
//     1366x768) renders each lane as a solid block instead of a raster. Every sampled neuron keeps its own row at
//     every real size; only a sample too tall for RASTER_MAX_H falls back to sharing device rows
//     (`RasterPainter.dense`, which the header line discloses). Lane tops are `round(lane * H / 8)`, so the 8 bands
//     stay in REGIONS order and the last one ends exactly on the last pixel. The header strip is proportional as well
//     as capped (`laneH / 3`, clamped to [RASTER_HEADER_MIN, RASTER_HEADER]) so a short lane spends its pixels on
//     spikes rather than on its title.
//  3. The width is always exactly the pane's `clientWidth` (no RASTER_MIN_W floor): the pane clips on the x axis
//     rather than growing a horizontal scrollbar, so the one ResizeObserver tick it takes to follow the vertical
//     scrollbar appearing costs 10 px of plot for one frame instead of a scrollbar that steals 13 px of height.
//     `windowS` (8 s) is always honoured exactly - the effective scroll speed is `plotWidth / windowS`, which is the
//     documented 60 px/s when the plot area happens to be `windowS * refPxPerS` px wide.
// The canvas is sized 1:1 in css px (DPR 1), so one raster row is one device pixel and nothing is ever stretched.
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

export const RASTER_MIN_W = 480;   // fallback width when the host has not been laid out yet (section e.4: 480 x ...)
export const RASTER_GUTTER_L = 60; // left gutter: lane title + star labels
export const RASTER_GUTTER_R = 44; // right gutter: "x.x Hz" per lane
export const RASTER_HEADER = 12;   // px strip above each lane when the lane can spare it
export const RASTER_HEADER_MIN = 4; // ... and when it cannot (a very short pane)
/** Narrowest / shortest bitmap we will ever allocate (a pane smaller than this is clipped by its host). */
export const RASTER_FLOOR_W = RASTER_GUTTER_L + RASTER_GUTTER_R + 40;
export const RASTER_FLOOR_H = 8 * (RASTER_HEADER_MIN + 1);
/** Largest bitmap (px) - a sanity cap so a pathological pane (or per_region) cannot allocate an absurd bitmap. */
export const RASTER_MAX_W = 4096;
export const RASTER_MAX_H = 4096;
/** A spike row is never drawn thicker than this, however much room the pitch gives it. */
export const RASTER_MAX_ROW_H = 24;
const N_LANES = 8;
const FONT_SMALL = "8px 'Lucida Console', Consolas, monospace";
const FONT_HZ = "9px 'Lucida Console', Consolas, monospace";
const LABEL_MIN_GAP = 8;           // px between two star labels in the gutter
const MAX_GAP_S = 2;               // a rAF gap longer than this (tab throttled) is clamped

export interface RasterOpts { windowS?: number /* 8 */; pxPerS?: number /* 60 */ }

/**
 * "Natural" bitmap height for `per_region` rows per lane: `8 * (per_region + 12)` (section e.4) - the shortest bitmap
 * that still gives every sampled neuron its own device row. The painter uses it as the FLOOR of the height it takes
 * from the pane (a shorter pane scrolls) and as the fallback when the pane cannot be measured (SSR / display:none).
 */
export function rasterHeight(perRegion: number): number {
  return N_LANES * (Math.max(1, perRegion | 0) + RASTER_HEADER);
}

/** One lane band of the raster. `[top, end)` is its slice of the bitmap; rows live in `[row0, end)`. */
export interface LaneGeom {
  /** First pixel of the lane (its header strip). */
  top: number;
  /** First spike row of the lane (`top + headerH`). */
  row0: number;
  /** One past the last pixel of the lane (= the next lane's `top`; the bitmap height for lane 7). */
  end: number;
  /** Distance in px between two consecutive sampled rows; < 1 means neighbours share a device row. */
  pitch: number;
}

export interface RasterGeom {
  /** Bitmap height these lanes tile exactly: `lanes[0].top === 0` and `lanes[7].end === H`. */
  H: number;
  /** Height of each lane's header strip (lane title + separator). */
  headerH: number;
  /** Thickness of a plotted spike row (star rows get one more pixel). */
  rowH: number;
  /** Smallest `lanes[i].pitch` - the lane that shares device rows first (the lanes differ by rounding only). */
  minPitch: number;
  /** True when `minPitch` is below 1 px, i.e. some neurons of some lane share a device row. */
  dense: boolean;
  lanes: LaneGeom[];
}

/**
 * Tiles `height` px into the 8 lane bands and derives the row pitch for `perRegion` rows per lane. Pure (no DOM), so
 * the whole size sweep - a 160 px pane to a 1400 px one - is checkable without a browser.
 */
export function computeRasterGeom(height: number, perRegion: number): RasterGeom {
  const per = Math.max(1, perRegion | 0);
  const H = Math.max(RASTER_FLOOR_H, Math.floor(Number.isFinite(height) ? height : 0));
  const laneH = H / N_LANES;
  // The title strip is proportional (a third of the lane) as well as capped at RASTER_HEADER, so a short lane spends
  // its pixels on spikes instead of on its title; it never eats more than `laneH - 1` (one row always survives).
  const headerH = Math.max(
    RASTER_HEADER_MIN,
    Math.min(RASTER_HEADER, Math.floor(laneH / 3), Math.floor(laneH) - 1),
  );
  const lanes: LaneGeom[] = [];
  let minPitch = Infinity;
  for (let lane = 0; lane < N_LANES; lane++) {
    const top = Math.round(lane * laneH);
    const end = lane === N_LANES - 1 ? H : Math.round((lane + 1) * laneH);
    const row0 = Math.min(end - 1, top + headerH);
    const pitch = (end - row0) / per;                    // fills [row0, end) exactly
    if (pitch < minPitch) minPitch = pitch;
    lanes.push({ top, row0, end, pitch });
  }
  const rowH = minPitch < 2 ? 1 : Math.min(RASTER_MAX_ROW_H, Math.max(1, Math.floor(minPitch) - 1));
  return { H, headerH, rowH, minPitch, dense: minPitch < 1, lanes };
}

export class RasterPainter {
  readonly canvas: HTMLCanvasElement;
  readonly rows: readonly RasterRow[];
  readonly per: number;
  readonly dtMs: number;
  /** Seconds of history the plot area shows (section e.4 `windowS`, default 8) - exact at every width. */
  readonly windowS: number;
  /** Reference scroll speed (section e.4 `pxPerS`, default 60): the speed when the plot gets its reference width. */
  readonly refPxPerS: number;
  private readonly ctx: CanvasRenderingContext2D;
  private W = 0;
  private H = 0;
  private scroll = 60;              // effective px/s = plot width / windowS
  private geom: RasterGeom;
  private slotY: Int32Array;        // sampled row -> top y of its band (-1: padding slot)
  private slotH: Int32Array;        // ... and its thickness (clamped to the lane)
  private yToSlot: Int32Array;      // device row -> sampled row (-1: header strip / unclaimed)
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
    this.geom = computeRasterGeom(rasterHeight(this.per), this.per);
    this.slotY = new Int32Array(N_LANES * this.per).fill(-1);
    this.slotH = new Int32Array(N_LANES * this.per);
    this.yToSlot = new Int32Array(0);
    this.resize();
  }

  /** Canvas height in px = the host pane's height. */
  get height(): number { return this.H; }
  get width(): number { return this.W; }
  /** Effective scroll speed (px/s) for the current width: `plotWidth / windowS`. */
  get pxPerS(): number { return this.scroll; }
  /** Left edge of the scrolling plot area. */
  get plotLeft(): number { return RASTER_GUTTER_L; }
  /** Right edge (exclusive) of the scrolling plot area. */
  get plotRight(): number { return this.W - RASTER_GUTTER_R; }
  /** px between two consecutive sampled rows at the current height (lane 0; the lanes differ by < 1 %). */
  get rowPitch(): number { return this.geom.lanes[0].pitch; }
  /** ... and the smallest pitch across the 8 lanes: the one that shares device rows first. Pairs with `dense`. */
  get minRowPitch(): number { return this.geom.minPitch; }
  /** True when some lane cannot give every sampled neuron its own device row (only above RASTER_MAX_H). */
  get dense(): boolean { return this.geom.dense; }
  /** Shortest bitmap that still gives every sampled neuron its own device row - the floor of the height we take. */
  get naturalHeight(): number { return Math.min(RASTER_MAX_H, rasterHeight(this.per)); }

  laneTop(lane: number): number { return this.geom.lanes[clampLane(lane)].top; }
  laneRow0(lane: number): number { return this.geom.lanes[clampLane(lane)].row0; }
  laneEnd(lane: number): number { return this.geom.lanes[clampLane(lane)].end; }
  /** Top y of a sampled row's band, or -1 when the slot is padding (section c.12). */
  slotTop(slot: number): number {
    return slot >= 0 && slot < this.slotY.length ? this.slotY[slot] : -1;
  }

  /**
   * Re-reads the host size (DPR 1) and re-tiles the lanes: the bitmap takes the pane's width, and its height too as
   * long as that leaves every sampled neuron its own device row (`naturalHeight` is the floor - a shorter pane scrolls
   * instead of stacking neurons on one row). The css size is pinned to the bitmap size so one raster row is one device
   * pixel and nothing is scaled. Assigning width/height clears the bitmap, so it only happens when the geometry
   * actually changed - and a host that measures 0 (collapsed window / display:none) keeps the last good geometry
   * rather than falling back to a size nobody asked for and wiping the history on the way back.
   */
  resize(): void {
    const host = this.canvas.parentElement;
    const hostW = host ? Math.floor(host.clientWidth || 0) : 0;
    const hostH = host ? Math.floor(host.clientHeight || 0) : 0;
    if ((hostW <= 0 || hostH <= 0) && this.W > 0 && this.H > 0) return;
    const W = clamp(hostW > 0 ? hostW : RASTER_MIN_W, RASTER_FLOOR_W, RASTER_MAX_W);
    const H = clamp(Math.max(hostH, this.naturalHeight), RASTER_FLOOR_H, RASTER_MAX_H);
    const geom = computeRasterGeom(H, this.per);
    this.scroll = (W - RASTER_GUTTER_L - RASTER_GUTTER_R) / this.windowS;
    if (W === this.W && geom.H === this.H) return;
    this.W = W; this.H = geom.H; this.geom = geom;
    this.buildRowMaps();
    this.canvas.width = W;
    this.canvas.height = geom.H;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${geom.H}px`;
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
    if (plotW <= 0) return;
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
   * Resolved through the pixel -> slot map built by `buildRowMaps`, so it is exact at any pitch (including the thicker
   * star rows and the shared rows of a dense pane, where the topmost neuron of the pixel wins).
   */
  rowAt(y: number): RasterRow | null {
    const yy = Math.floor(y);
    if (!(yy >= 0) || yy >= this.yToSlot.length) return null;
    const slot = this.yToSlot[yy];
    if (slot < 0) return null;
    const row = this.rows[slot];
    return row && row.neuron >= 0 ? row : null;
  }

  // ------------------------------------------------------------------ private painting helpers

  /**
   * Rebuilds `slotY` / `slotH` (sampled row -> pixels) and `yToSlot` (pixel -> sampled row) for the current geometry.
   * Every one of the `8 * per_region` slots gets a y, so no neuron and no spike is ever dropped; when the pitch is
   * below 1 px consecutive slots land on the same pixel and the topmost one owns it for hover purposes.
   * TWO passes: every slot claims its own top pixel first, and only then does the row THICKNESS fill what is still
   * free - otherwise a star row (drawn one px thicker) would swallow the first pixel of the neuron below it and that
   * neuron would be unreachable by the hover tooltip, one per lane at a pitch of 1.
   */
  private buildRowMaps(): void {
    const { lanes, rowH, H } = this.geom;
    const n = N_LANES * this.per;
    if (this.slotY.length !== n) { this.slotY = new Int32Array(n); this.slotH = new Int32Array(n); }
    if (this.yToSlot.length !== H) this.yToSlot = new Int32Array(H);
    this.slotY.fill(-1);
    this.slotH.fill(0);
    this.yToSlot.fill(-1);
    for (let lane = 0; lane < N_LANES; lane++) {
      const { row0, end, pitch } = lanes[lane];
      for (let i = 0; i < this.per; i++) {
        const slot = lane * this.per + i;
        const row = this.rows[slot];
        if (!row || row.neuron < 0 || row.region !== lane) continue;   // padding slot: no pixels
        const y = Math.min(end - 1, row0 + Math.floor(i * pitch));
        const h = Math.max(1, Math.min(row.star ? rowH + 1 : rowH, end - y));
        this.slotY[slot] = y;
        this.slotH[slot] = h;
        if (y >= 0 && y < H && this.yToSlot[y] === -1) this.yToSlot[y] = slot;   // pass 1: own top pixel
      }
    }
    for (let slot = 0; slot < n; slot++) {                             // pass 2: the rest of the thickness
      const y = this.slotY[slot];
      if (y < 0) continue;
      for (let dy = 1; dy < this.slotH[slot]; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < H && this.yToSlot[yy] === -1) this.yToSlot[yy] = slot;
      }
    }
  }

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
      ctx.fillRect(x, this.geom.lanes[lane].row0 - 1, w, 1);
    }
  }

  private paintLeftGutter(): void {
    const ctx = this.ctx;
    const gw = this.plotLeft;
    const { headerH, lanes } = this.geom;
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
      const { top, row0 } = lanes[lane];
      ctx.fillStyle = LANE_COLORS[lane];
      if (headerH >= 7) ctx.fillText(LANE_TITLES[lane] ?? "", 2, top + headerH / 2);
      ctx.fillRect(0, row0 - 1, gw, 1);                     // coloured separator in the gutter
      // star rows: tick mark always, label when enabled and there is room
      let lastLabelY = -Infinity;
      const s0 = lane * this.per, s1 = Math.min(this.rows.length, s0 + this.per);
      for (let slot = s0; slot < s1; slot++) {
        const row = this.rows[slot];
        if (!row || !row.star) continue;
        const y = this.slotY[slot];
        if (y < 0) continue;
        ctx.fillStyle = LANE_COLORS[lane];
        ctx.fillRect(gw - 4, y, 4, Math.min(2, this.slotH[slot]));
        if (!this.labels) continue;
        if (y - lastLabelY < LABEL_MIN_GAP) continue;       // never let two labels collide
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
    const lanes = this.geom.lanes;
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
      const { top, row0, end } = lanes[lane];
      ctx.fillStyle = LANE_COLORS[lane];
      ctx.fillRect(x, row0 - 1, w, 1);
      // one readout per lane, centred on its rows - skipped only when the band is too short for 9 px of text
      if (end - top >= 11) ctx.fillText(`${this.regionHz[lane].toFixed(1)} Hz`, this.W - 2, (row0 + end) / 2);
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
      const slot = Math.floor(Number(sp.slots[k]));
      if (!(slot >= 0) || slot >= this.slotY.length) continue;
      const y = this.slotY[slot];
      if (y < 0) continue;                                  // padding slot (neuron == -1)
      const lane = this.rows[slot].region;
      if (lane < 0 || lane >= N_LANES) continue;
      const dtSteps = Number(sp.dt[k]);
      const ageMs = winMs - (Number.isFinite(dtSteps) ? dtSteps : 0) * this.dtMs + ageOffsetMs;
      let x = Math.floor(x1 - 1 - ageMs * scale);
      if (!Number.isFinite(x)) continue;
      if (x < x0) x = x0;
      if (x >= x1) x = x1 - 1;
      if (lane !== curLane) { ctx.fillStyle = LANE_COLORS[lane]; curLane = lane; }
      ctx.fillRect(x, y, 1, this.slotH[slot]);
    }
    ctx.restore();
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function clampLane(lane: number): number {
  return lane < 0 ? 0 : lane > N_LANES - 1 ? N_LANES - 1 : lane | 0;
}

/** Gutter label of a star row, e.g. "DNp01 R", "MN9", "PFL3 L" (section e.4). */
export function rowLabel(row: RasterRow): string {
  const label = typeof row.label === "string" ? row.label : "";
  const base = label.length > 9 ? label.slice(0, 9) : label;
  return row.side === "M" ? base : `${base} ${row.side}`;
}
