"use client";
// SPEC section e.5 `ToolPalette`: the classic 2 x 8 Paint tool box (inline-SVG pixel glyphs, `pencil` pressed), five
// 24 px poke tools below it (sugar cube, red candle = loom, bitter leaf, water drop, dust) that call onPoke, the tool of
// `lastPoke` flashing pressed for 300 ms (local click or the `poke` event echoed from another client), and the
// "active tool" box showing the current ink width as a line sample.
import { useEffect, useState, type ReactNode } from "react";
import type { PokeStim } from "@/lib/types";

export interface ToolPaletteProps {
  active: string;
  onPoke(stim: PokeStim): void;
  lastPoke: { stim: PokeStim; at: number } | null;
  inkWidth?: number;
  inkColor?: string;
}

const FLASH_MS = 300;

function G({ children, title }: { children: ReactNode; title: string }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" shapeRendering="crispEdges" aria-hidden>
      <title>{title}</title>
      {children}
    </svg>
  );
}

/** The 16 Paint tools in the classic order (two columns, read row by row). */
const PAINT_TOOLS: { name: string; glyph: ReactNode }[] = [
  { name: "Free-Form Select", glyph: <G title="Free-Form Select"><path d="M3 9 L5 4 L9 3 L12 5 L12 9 L9 12 L5 12 Z" fill="none" stroke="#000" strokeDasharray="1 1" /></G> },
  { name: "Select", glyph: <G title="Select"><rect x="2.5" y="2.5" width="11" height="11" fill="none" stroke="#000" strokeDasharray="1 1" /></G> },
  { name: "Eraser/Color Eraser", glyph: <G title="Eraser"><rect x="3" y="6" width="9" height="6" fill="#fff" stroke="#000" /><rect x="3" y="6" width="9" height="2" fill="#ff80c0" /></G> },
  { name: "Fill With Color", glyph: <G title="Fill"><path d="M4 7 L9 3 L13 7 L8 12 Z" fill="#fff" stroke="#000" /><rect x="3" y="9" width="2" height="4" fill="#0000ff" /><rect x="8" y="1" width="1" height="3" fill="#000" /></G> },
  { name: "Pick Color", glyph: <G title="Pick Color"><path d="M3 13 L9 7" stroke="#000" strokeWidth="2" /><rect x="8" y="4" width="4" height="4" fill="#c0c0c0" stroke="#000" /><rect x="11" y="2" width="3" height="3" fill="#000" /></G> },
  { name: "Magnifier", glyph: <G title="Magnifier"><circle cx="6.5" cy="6.5" r="4" fill="#fff" stroke="#000" /><path d="M10 10 L14 14" stroke="#000" strokeWidth="2" /></G> },
  { name: "Pencil", glyph: <G title="Pencil"><path d="M3 13 L11 5" stroke="#000" strokeWidth="3" /><path d="M11 5 L13 3" stroke="#ffff00" strokeWidth="3" /><rect x="2" y="12" width="2" height="2" fill="#000" /></G> },
  { name: "Brush", glyph: <G title="Brush"><rect x="7" y="1" width="2" height="7" fill="#804000" /><rect x="5" y="8" width="6" height="2" fill="#c0c0c0" stroke="#000" /><rect x="5" y="10" width="6" height="4" fill="#000" /></G> },
  { name: "Airbrush", glyph: <G title="Airbrush"><rect x="2" y="9" width="5" height="5" fill="#c0c0c0" stroke="#000" /><rect x="6" y="7" width="3" height="3" fill="#000" />{[[10, 3], [12, 5], [11, 8], [13, 10], [9, 4], [13, 2], [10, 11], [14, 7]].map(([x, y]) => <rect key={`${x}-${y}`} x={x} y={y} width="1" height="1" fill="#000" />)}</G> },
  { name: "Text", glyph: <G title="Text"><rect x="3" y="2" width="10" height="2" fill="#000" /><rect x="7" y="4" width="2" height="9" fill="#000" /><rect x="5" y="12" width="6" height="1" fill="#000" /><rect x="3" y="4" width="1" height="2" fill="#000" /><rect x="12" y="4" width="1" height="2" fill="#000" /></G> },
  { name: "Line", glyph: <G title="Line"><path d="M2 13 L13 2" stroke="#000" strokeWidth="1" /></G> },
  { name: "Curve", glyph: <G title="Curve"><path d="M2 13 C 4 2, 11 14, 13 2" fill="none" stroke="#000" /></G> },
  { name: "Rectangle", glyph: <G title="Rectangle"><rect x="2.5" y="3.5" width="11" height="9" fill="none" stroke="#000" /></G> },
  { name: "Polygon", glyph: <G title="Polygon"><path d="M2 12 L6 3 L10 8 L14 4 L12 13 Z" fill="none" stroke="#000" /></G> },
  { name: "Ellipse", glyph: <G title="Ellipse"><ellipse cx="8" cy="8" rx="5.5" ry="4" fill="none" stroke="#000" /></G> },
  { name: "Rounded Rectangle", glyph: <G title="Rounded Rectangle"><rect x="2.5" y="3.5" width="11" height="9" rx="3" fill="none" stroke="#000" /></G> },
];

/** Poke tools: stim, tooltip, glyph (section e.5). */
export const POKE_TOOLS: { stim: PokeStim; name: string; key: string; glyph: ReactNode }[] = [
  { stim: "sugar", name: "Sugar (S) - poke a fake BUY: feeds the fly, it gets happy", key: "S", glyph: <G title="Sugar"><path d="M4 6 L7 3 L13 3 L13 9 L10 12 L4 12 Z" fill="#fff" stroke="#000" /><path d="M4 6 L10 6 L10 12" fill="none" stroke="#808080" /><path d="M10 6 L13 3" stroke="#808080" /></G> },
  { stim: "loom", name: "Sell wall (L) - poke a fake SELL: a looming threat, the fly panics", key: "L", glyph: <G title="Loom"><rect x="7" y="1" width="2" height="14" fill="#a80000" /><rect x="4" y="4" width="8" height="8" fill="#ff0000" stroke="#a80000" /></G> },
  { stim: "bitter", name: "Bitter (B) - an aversive taste, the fly recoils", key: "B", glyph: <G title="Bitter"><path d="M3 13 C 3 6, 8 2, 14 2 C 14 8, 9 13, 3 13 Z" fill="#008000" stroke="#004000" /><path d="M4 12 L12 4" stroke="#00c000" /></G> },
  { stim: "water", name: "Water (W) - a drink", key: "W", glyph: <G title="Water"><path d="M8 1 C 10 5, 13 8, 13 10.5 A 5 5 0 0 1 3 10.5 C 3 8, 6 5, 8 1 Z" fill="#0080ff" stroke="#004080" /><rect x="5" y="9" width="1" height="2" fill="#fff" /></G> },
  { stim: "dust", name: "Dust (D) - makes the fly stop and groom", key: "D", glyph: <G title="Dust">{[[3, 4], [6, 2], [9, 5], [12, 3], [4, 8], [8, 8], [11, 9], [2, 12], [6, 11], [10, 13], [13, 12], [7, 5]].map(([x, y], i) => <rect key={i} x={x} y={y} width={i % 3 === 0 ? 2 : 1} height={i % 3 === 0 ? 2 : 1} fill={i % 2 ? "#808040" : "#a08060"} />)}</G> },
];

export default function ToolPalette({ active, onPoke, lastPoke, inkWidth = 2, inkColor = "#000000" }: ToolPaletteProps) {
  // The flash of the last poke expires 300 ms after it started (a timeout marks that poke as expired).
  const [expiredAt, setExpiredAt] = useState(-Infinity);
  useEffect(() => {
    if (!lastPoke) return;
    const at = lastPoke.at;
    const t = setTimeout(() => setExpiredAt((e) => (e >= at ? e : at)), FLASH_MS);
    return () => clearTimeout(t);
  }, [lastPoke]);
  const flashing = lastPoke && lastPoke.at > expiredAt ? lastPoke.stim : null;

  return (
    <div className="flex w-[58px] shrink-0 flex-col items-center gap-1 bg-win-gray py-1" data-testid="tool-palette">
      <div className="grid grid-cols-2 gap-0">
        {PAINT_TOOLS.map((t) => (
          <button
            key={t.name}
            type="button"
            className="btn95 flex h-6 w-6 items-center justify-center"
            style={{ padding: 0 }}
            title={t.name}
            aria-label={t.name}
            aria-pressed={t.name === "Pencil"}
          >
            {t.glyph}
          </button>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-2 gap-0" title="Poke tools: click the document to poke the fly with the active tool">
        {POKE_TOOLS.map((t) => (
          <button
            key={t.stim}
            type="button"
            className={"btn95 flex h-6 w-6 items-center justify-center " + (flashing === t.stim ? "tool-flash" : "")}
            style={{ padding: 0 }}
            title={t.name}
            aria-label={t.name}
            aria-pressed={active === t.stim}
            data-stim={t.stim}
            onClick={() => onPoke(t.stim)}
          >
            {t.glyph}
          </button>
        ))}
      </div>
      <div className="bevel-in mt-1 flex h-[40px] w-[42px] items-center justify-center bg-white" title={`active tool: ${active} - ink width ${inkWidth} px`}>
        <svg width="34" height="30" viewBox="0 0 34 30" aria-hidden>
          <line x1="4" y1="24" x2="30" y2="6" stroke={inkColor} strokeWidth={Math.max(1, Math.min(8, inkWidth))} strokeLinecap="round" />
        </svg>
      </div>
    </div>
  );
}
