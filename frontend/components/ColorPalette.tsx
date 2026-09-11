"use client";
// SPEC section e.5 `ColorPalette`: the classic 2 x 14 Paint swatches; the swatch nearest ink.color (RGB distance) is
// outlined; the two-square current-colour box shows ink.color (fore) over moodColor(mood) (back).
import type { InkStyle, Mood } from "@/lib/types";
import { moodColor } from "@/lib/ink";

/** `tSec` = brain time in seconds (tick.t_ms / 1000) used for the rainbow phase; render stays pure. */
export interface ColorPaletteProps { ink: InkStyle; mood: Mood; tSec?: number }

export const PAINT_SWATCHES: readonly string[] = [
  "#000000", "#808080", "#800000", "#808000", "#008000", "#008080", "#000080", "#800080", "#808040", "#004040", "#0080ff", "#004080", "#8000ff", "#804000",
  "#ffffff", "#c0c0c0", "#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ffff80", "#00ff80", "#80ffff", "#8080ff", "#ff0080", "#ff8040",
];

function parseHex(c: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(c.trim());
  if (!m) return null;
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** Index of the swatch nearest `color` by RGB distance (0 when unparsable). */
export function nearestSwatch(color: string): number {
  const rgb = parseHex(color);
  if (!rgb) return 0;
  let best = 0, bestD = Infinity;
  PAINT_SWATCHES.forEach((s, i) => {
    const t = parseHex(s);
    if (!t) return;
    const d = (rgb[0] - t[0]) ** 2 + (rgb[1] - t[1]) ** 2 + (rgb[2] - t[2]) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

export default function ColorPalette({ ink, mood, tSec = 0 }: ColorPaletteProps) {
  const near = nearestSwatch(ink.color);
  const back = moodColor(mood, tSec);
  const fore = ink.color;                       // e.5: the fore square is ink.color verbatim (not the rainbow)
  return (
    <div className="flex items-center gap-2 bg-win-gray px-1 py-1" data-testid="color-palette">
      <div className="bevel-in relative h-[34px] w-[34px] bg-win-gray" title={`ink ${ink.color} ${ink.style} w${ink.width} a${ink.alpha} / mood ${mood}`}>
        <div className="absolute left-[12px] top-[12px] h-[14px] w-[14px] border border-black" style={{ background: back }} />
        <div className="absolute left-[4px] top-[4px] h-[14px] w-[14px] border border-black" style={{ background: fore }} />
      </div>
      <div className="grid grid-cols-14 grid-rows-2 gap-0">
        {PAINT_SWATCHES.map((c, i) => (
          <div
            key={c}
            className="bevel-in h-4 w-4"
            style={{ background: c, outline: i === near ? "2px solid #000" : undefined, outlineOffset: i === near ? 0 : undefined, zIndex: i === near ? 1 : undefined }}
            title={c}
          />
        ))}
      </div>
      <div className="ml-1 text-[11px] text-black">
        {ink.style}{ink.stamp ? ` + ${ink.stamp}` : ""} &middot; {ink.width}px &middot; a{ink.alpha.toFixed(1)}
      </div>
    </div>
  );
}
