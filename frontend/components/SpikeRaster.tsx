"use client";
// SPEC section e.5 `SpikeRaster`: wraps lib/raster.ts `RasterPainter`. The rAF loop calls painter.frame(now) every
// frame and painter.push(tick) for every new tick read from the mutable store (no React state per tick); the header
// line re-renders at 4 Hz through useTickSnapshot; the hover tooltip comes from painter.rowAt(y).
//
// The canvas carries NO width/height attributes: the painter owns the bitmap and sizes it to this pane (measured with
// a ResizeObserver, DPR 1 so css px == device px). That is what removed the ~95 px white strip under the raster in a
// three-column layout - the bitmap used to be a fixed 8*(per_region+12) px tall regardless of the window. It still
// never goes BELOW that natural height (one device row per sampled neuron), so the pane stays `overflow-auto`: a short
// window scrolls the raster instead of squashing 6-8 neurons of a lane onto one row. The pane is painted black because
// `.bevel-in` is plain CSS (background:#fff) and beats Tailwind's layered `bg-black` utility: any pixel the canvas
// does not cover would flash white.
import { useCallback, useEffect, useRef, useState } from "react";
import type { FlySocket } from "@/lib/ws";
import type { RasterRow } from "@/lib/types";
import { useTickSnapshot } from "@/lib/store";
import { LANE_COLORS, LANE_TITLES, RasterPainter } from "@/lib/raster";

export interface SpikeRasterProps { sock: FlySocket; frozen: boolean; labels: boolean }

interface Hover { row: RasterRow; cssY: number }

const DEFAULT_PER = 48;

export function SpikeRaster({ sock, frozen, labels }: SpikeRasterProps) {
  const hello = sock.hello;
  const store = sock.store;
  const tick = useTickSnapshot(store);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const painterRef = useRef<RasterPainter | null>(null);
  const flagsRef = useRef({ frozen, labels });
  const [hover, setHover] = useState<Hover | null>(null);
  // Row density of the current bitmap, for the header readout. `pitch` is the painter's WORST lane (minRowPitch) and
  // `dense` its own flag, so the disclosure follows the lane that actually shares device rows - lane 0 can sit at
  // exactly 1.00 px while a rounding-shorter lane is below it.
  const [density, setDensity] = useState<{ pitch: number; dense: boolean }>({ pitch: 1, dense: false });

  // Same numbers => keep the previous object so React bails out instead of re-rendering on every resize tick.
  const readDensity = useCallback((p: RasterPainter) => {
    setDensity((d) => (d.pitch === p.minRowPitch && d.dense === p.dense ? d : { pitch: p.minRowPitch, dense: p.dense }));
  }, []);

  // Keep the painter flags in sync with the props (also applied when the painter is (re)created).
  useEffect(() => {
    flagsRef.current = { frozen, labels };
    const p = painterRef.current;
    if (p) { p.setFrozen(frozen); p.setLabels(labels); }
  }, [frozen, labels]);

  // One painter per hello (a new run_id / raster table means a new painter) + the rAF loop.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !hello) return;
    let painter: RasterPainter;
    try {
      painter = new RasterPainter(canvas, hello);
    } catch (e) {
      console.error("[raster] painter init failed", e);
      return;
    }
    painterRef.current = painter;
    painter.setFrozen(flagsRef.current.frozen);
    painter.setLabels(flagsRef.current.labels);
    readDensity(painter);

    let raf = 0;
    let lastSeq = -1;
    const loop = (now: number) => {
      try {
        painter.frame(now);
        const latest = store.latest();
        // While frozen the painter refuses ticks (the bitmap is a screenshot), so `lastSeq` must not advance either:
        // the ticks still in the store are plotted on the first frame after unfreezing instead of being lost.
        if (latest && latest.tick.seq !== lastSeq && !flagsRef.current.frozen) {
          const prev = store.prev();
          if (prev && prev.tick.seq > lastSeq && prev.tick.seq < latest.tick.seq) {
            // two ticks since the last frame: the older one goes one tick window further left
            painter.push(prev.tick, Math.max(0, latest.tick.t_ms - prev.tick.t_ms));
          }
          if (painter.push(latest.tick)) lastSeq = latest.tick.seq;
        }
      } catch (e) {
        console.error("[raster] frame failed", e);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    // The pane drives the bitmap: every window resize / drag re-tiles the 8 lanes to fill it (down to 1 px per row).
    const host = canvas.parentElement ?? canvas;
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => {
        try {
          painter.resize();
          readDensity(painter);
        } catch (e) { console.error("[raster] resize failed", e); }
      });
      ro.observe(host);
    }
    return () => {
      cancelAnimationFrame(raf);
      if (ro) ro.disconnect();
      if (painterRef.current === painter) painterRef.current = null;
    };
  }, [hello, store, readDensity]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const painter = painterRef.current;
    if (!canvas || !painter) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.height <= 0) return;
    const scaleY = painter.height / rect.height;
    const y = (e.clientY - rect.top) * scaleY;
    const row = painter.rowAt(y);
    setHover((h) => {
      if (!row) return h === null ? h : null;
      if (h && h.row.slot === row.slot) return h;
      const top = painter.slotTop(row.slot);
      return { row, cssY: (top < 0 ? y : top) / scaleY };
    });
  }, []);

  const onPointerLeave = useCallback(() => setHover(null), []);

  const per = hello?.raster.per_region ?? DEFAULT_PER;
  const { pitch, dense } = density;
  const spikesPerS = tick && tick.spikes.win_ms > 0 ? tick.spikes.total / (tick.spikes.win_ms / 1000) : null;
  const activePct = tick ? tick.sim.active_frac * 100 : null;

  return (
    <div className="flex h-full w-full flex-col bg-win-gray text-[11px]" data-testid="spike-raster">
      <div className="flex items-center gap-2 px-1 py-0.5 font-mono">
        <span>spikes/s {spikesPerS === null ? "-" : spikesPerS.toFixed(0)}</span>
        <span>&middot;</span>
        <span>active {activePct === null ? "-" : activePct.toFixed(2)}%</span>
        <span>&middot;</span>
        <span
          className="text-[#404040]"
          title={
            "sampled rows: 8 lanes x per_region (FLY_RASTER_PER_REGION); the raster fills the window down to one " +
            `screen row per neuron and no further (now ${pitch.toFixed(2)} px per row) - a window shorter than that ` +
            "scrolls instead of stacking neurons on one row"
          }
        >
          rows 8&times;{per}
        </span>
        {dense ? (
          <span
            className="text-[#404040]"
            title={
              "the sample is taller than the largest bitmap we allocate: about " + Math.round(1 / pitch) +
              " neighbouring neurons of a lane share one screen row (no neuron and no spike is dropped) - " +
              "lower FLY_RASTER_PER_REGION to give each its own row"
            }
          >
            @{pitch.toFixed(2)}px/row
          </span>
        ) : null}
        {tick?.spikes.capped ? (
          <span className="bg-[#ff4136] px-1 text-[9px] font-bold text-white" title="raster sample capped (uniformly subsampled)">
            &#9660; cap
          </span>
        ) : null}
        <span className="ml-auto text-[#404040]">
          {tick ? `t=${(tick.t_ms / 1000).toFixed(1)}s` : hello ? "waiting for ticks" : "no hello yet"}
          {frozen ? " | FROZEN" : ""}
        </span>
      </div>
      {/* overflow-x is HIDDEN, not auto: the bitmap is one ResizeObserver tick behind the pane's width whenever the
          vertical scrollbar appears, and a transient horizontal scrollbar would eat 13 px of height and re-trigger the
          observer. The painter always resizes back to clientWidth, so nothing stays clipped. */}
      <div className="bevel-in relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-0" style={{ background: "#000000" }}>
        <canvas
          ref={canvasRef}
          className="block"
          style={{ imageRendering: "pixelated" }}
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
          aria-label="spike raster oscilloscope"
        />
        {hover ? (
          <div
            className="pointer-events-none absolute left-16 z-10 border border-black bg-[#ffffe1] px-1 py-0.5 font-mono text-[10px] text-black"
            style={{ top: Math.max(0, hover.cssY - 18) }}
          >
            <span style={{ color: LANE_COLORS[hover.row.region] }}>&#9632;</span>{" "}
            {hover.row.label} {hover.row.side !== "M" ? hover.row.side : ""} &middot; #{hover.row.neuron} &middot;{" "}
            {LANE_TITLES[hover.row.region] ?? hello?.regions[hover.row.region] ?? "?"} &middot; slot {hover.row.slot}
            {hover.row.star ? " ★" : ""}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default SpikeRaster;
