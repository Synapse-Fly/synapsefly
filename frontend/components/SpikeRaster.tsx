"use client";
// SPEC section e.5 `SpikeRaster`: wraps lib/raster.ts `RasterPainter`. The rAF loop calls painter.frame(now) every
// frame and painter.push(tick) for every new tick read from the mutable store (no React state per tick); the header
// line re-renders at 4 Hz through useTickSnapshot; the hover tooltip comes from painter.rowAt(y).
import { useCallback, useEffect, useRef, useState } from "react";
import type { FlySocket } from "@/lib/ws";
import type { RasterRow } from "@/lib/types";
import { useTickSnapshot } from "@/lib/store";
import { LANE_COLORS, LANE_TITLES, RASTER_MIN_W, RasterPainter, rasterHeight } from "@/lib/raster";

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

    const host = canvas.parentElement ?? canvas;
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => { try { painter.resize(); } catch (e) { console.error("[raster] resize failed", e); } });
      ro.observe(host);
    }
    return () => {
      cancelAnimationFrame(raf);
      if (ro) ro.disconnect();
      if (painterRef.current === painter) painterRef.current = null;
    };
  }, [hello, store]);

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
      const rowY = (painter.laneRow0(row.region) + (row.slot % painter.per)) / scaleY;
      return { row, cssY: rowY };
    });
  }, []);

  const onPointerLeave = useCallback(() => setHover(null), []);

  const per = hello?.raster.per_region ?? DEFAULT_PER;
  const spikesPerS = tick && tick.spikes.win_ms > 0 ? tick.spikes.total / (tick.spikes.win_ms / 1000) : null;
  const activePct = tick ? tick.sim.active_frac * 100 : null;

  return (
    <div className="flex h-full w-full flex-col bg-win-gray text-[11px]" data-testid="spike-raster">
      <div className="flex items-center gap-2 px-1 py-0.5 font-mono">
        <span>spikes/s {spikesPerS === null ? "-" : spikesPerS.toFixed(0)}</span>
        <span>&middot;</span>
        <span>active {activePct === null ? "-" : activePct.toFixed(2)}%</span>
        <span>&middot;</span>
        <span className="text-[#404040]" title="sampled rows: 8 lanes x per_region (FLY_RASTER_PER_REGION); the canvas keeps every row and the window scrolls">
          rows 8&times;{per}
        </span>
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
      <div className="bevel-in relative min-h-0 flex-1 overflow-auto bg-black p-0">
        <canvas
          ref={canvasRef}
          width={RASTER_MIN_W}
          height={rasterHeight(per)}
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
