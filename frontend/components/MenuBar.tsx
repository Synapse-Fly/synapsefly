"use client";
// SPEC section e.5 `MenuBar`: Paint's File / Edit / View / Image / Options / Help menus in the Win95 look. Click to
// open, hover to switch, Esc / click-out to close. Inert items render but do nothing (retro feel). The root carries
// `data-menu-open` while a menu is open so the document-level keyboard shortcuts (page.tsx) can stand down.
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import type { FlySocket } from "@/lib/ws";
import type { HelloMsg, MarketMode, PokeStim, TickMsg } from "@/lib/types";
import { useTickSnapshot, type TickStore } from "@/lib/store";
import { MessageBox } from "./Win95Window";
import { XGlyph } from "./ShareDialog";
import { X_URL, X_HANDLE, GITHUB_URL } from "@/lib/brand";
const SITE_URL = "https://www.synapsefly.com";

export type MenuAction =
  | { kind: "clear" }
  | { kind: "save" }
  // Opens the Share to X dialog. PaintWindow intercepts it (it owns the canvas handle); page.tsx never sees it.
  | { kind: "share" }
  | { kind: "poke"; stim: PokeStim }
  | { kind: "market"; mode: MarketMode }
  | { kind: "tweet_test" }
  | { kind: "toggle"; what: "labels" | "freeze" | "zoom" | "fps" | "flip" }
  | { kind: "about" }
  | { kind: "link"; url: string }
  | { kind: "reset_layout" }
  | { kind: "exit" };

export interface MenuFlags { labels: boolean; frozen: boolean; zoom: boolean; fps: boolean; flip: boolean }
export interface MenuBarProps { sock: FlySocket; onAction(a: MenuAction): void; flags?: Partial<MenuFlags> }

interface Item {
  label: string;
  shortcut?: string;
  action?: MenuAction;
  inert?: boolean;
  checked?: boolean;
  sub?: Item[];
  sep?: boolean;
}
interface Menu { label: string; items: Item[] }

/** Poke submenu rows (label -> stim), section e.5 / e.7 order. */
export const POKE_MENU: readonly { label: string; stim: PokeStim; key: string }[] = [
  { label: "Sugar", stim: "sugar", key: "S" },
  { label: "Bitter", stim: "bitter", key: "B" },
  { label: "Sell wall (loom)", stim: "loom", key: "L" },
  { label: "Water", stim: "water", key: "W" },
  { label: "Dust", stim: "dust", key: "D" },
  { label: "Pheromone", stim: "pheromone", key: "C" },
  { label: "Sleep", stim: "sleep", key: "Z" },
  { label: "Reward", stim: "reward", key: "R" },
  { label: "Punish", stim: "punish", key: "P" },
];
export const REGIME_MENU: readonly { label: string; mode: MarketMode; key: string }[] = [
  { label: "CALM", mode: "CALM", key: "1" },
  { label: "PUMP", mode: "PUMP", key: "2" },
  { label: "DUMP", mode: "DUMP", key: "3" },
  { label: "CHOP", mode: "CHOP", key: "4" },
  { label: "RUG", mode: "RUG", key: "5" },
  { label: "DEAD", mode: "DEAD", key: "6" },
  { label: "sim", mode: "sim", key: "0" },
  { label: "dexscreener", mode: "dexscreener", key: "" },
];

function buildMenus(flags: Partial<MenuFlags>, hello: HelloMsg | null): Menu[] {
  const channels = hello?.channels ?? null;
  const modes = hello?.market_modes ?? null;
  return [
    {
      label: "File",
      items: [
        { label: "New", shortcut: "Ctrl+N", action: { kind: "clear" } },
        { label: "Open...", shortcut: "Ctrl+O", inert: true },
        { label: "Save", shortcut: "Ctrl+S", inert: true },
        { label: "Save As...", action: { kind: "save" } },
        { label: "Share to X...", action: { kind: "share" } },
        { sep: true, label: "" },
        { label: "Print Preview", inert: true },
        { label: "Page Setup...", inert: true },
        { label: "Print...", shortcut: "Ctrl+P", inert: true },
        { sep: true, label: "" },
        { label: "Send...", inert: true },
        { sep: true, label: "" },
        { label: "Exit", shortcut: "Alt+F4", action: { kind: "exit" } },
      ],
    },
    {
      label: "Edit",
      items: [
        { label: "Undo", shortcut: "Ctrl+Z", inert: true },
        { label: "Repeat", shortcut: "F4", inert: true },
        { sep: true, label: "" },
        { label: "Cut", shortcut: "Ctrl+X", inert: true },
        { label: "Copy", shortcut: "Ctrl+C", inert: true },
        { label: "Paste", shortcut: "Ctrl+V", inert: true },
        { label: "Clear Selection", shortcut: "Del", inert: true },
        { label: "Select All", shortcut: "Ctrl+A", inert: true },
        { sep: true, label: "" },
        { label: "Clear trail", shortcut: "N", action: { kind: "clear" } },
      ],
    },
    {
      label: "View",
      items: [
        { label: "Tool Box", shortcut: "Ctrl+T", inert: true, checked: true },
        { label: "Color Box", shortcut: "Ctrl+L", inert: true, checked: true },
        { label: "Status Bar", inert: true, checked: true },
        { sep: true, label: "" },
        { label: "Raster labels", action: { kind: "toggle", what: "labels" }, checked: flags.labels ?? true },
        { label: "Freeze raster", shortcut: "F", action: { kind: "toggle", what: "freeze" }, checked: flags.frozen ?? false },
        { label: "Zoom canvas 1.5x", action: { kind: "toggle", what: "zoom" }, checked: flags.zoom ?? false },
        { label: "Show FPS", action: { kind: "toggle", what: "fps" }, checked: flags.fps ?? false },
        { sep: true, label: "" },
        // Clears the stored window positions and re-applies the computed tiling (lib/layout.ts).
        { label: "Tile windows (fit screen)", shortcut: "G", action: { kind: "reset_layout" } },
      ],
    },
    {
      label: "Image",
      items: [
        { label: "Flip fly", action: { kind: "toggle", what: "flip" }, checked: flags.flip ?? false },
        { label: "Flip/Rotate...", shortcut: "Ctrl+R", inert: true },
        { label: "Stretch/Skew...", shortcut: "Ctrl+W", inert: true },
        { label: "Invert Colors", shortcut: "Ctrl+I", inert: true },
        { label: "Attributes...", shortcut: "Ctrl+E", inert: true },
        { sep: true, label: "" },
        { label: "Clear Image", shortcut: "Ctrl+Shft+N", action: { kind: "clear" } },
      ],
    },
    {
      label: "Options",
      items: [
        {
          label: "Poke",
          sub: POKE_MENU.map((p) => ({
            label: p.label, shortcut: p.key, action: { kind: "poke", stim: p.stim } as MenuAction,
            inert: channels ? !channels.includes(p.stim) : false,
          })),
        },
        {
          label: "Market regime",
          sub: REGIME_MENU.map((r) => ({
            label: r.label, shortcut: r.key || undefined, action: { kind: "market", mode: r.mode } as MenuAction,
            inert: modes ? !modes.includes(r.mode) : false,
          })),
        },
        { sep: true, label: "" },
        { label: "Test tweet", shortcut: "T", action: { kind: "tweet_test" } },
        { sep: true, label: "" },
        { label: "Edit Colors...", inert: true },
        { label: "Get Colors...", inert: true },
        { label: "Save Colors...", inert: true },
        { label: "Draw Opaque", inert: true, checked: true },
      ],
    },
    {
      label: "Help",
      items: [
        { label: "What is this?", shortcut: "?", action: { kind: "about" } },
        { sep: true, label: "" },
        { label: "SynapseFly.com", action: { kind: "link", url: SITE_URL } },
        { label: "X (@SynapseFly)", action: { kind: "link", url: X_URL } },
        { label: "GitHub (source)", action: { kind: "link", url: GITHUB_URL } },
        { sep: true, label: "" },
        { label: "About FlyBrain", action: { kind: "about" } },
      ],
    },
  ];
}

function Mnemonic({ label }: { label: string }) {
  return (<><u>{label.slice(0, 1)}</u>{label.slice(1)}</>);
}

export default function MenuBar({ sock, onAction, flags = {} }: MenuBarProps) {
  const [open, setOpen] = useState<number | null>(null);
  const [sub, setSub] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const menus = buildMenus(flags, sock.hello);

  const close = useCallback(() => { setOpen(null); setSub(null); }, []);

  useEffect(() => {
    if (open === null) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); close(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); setOpen((o) => (o === null ? 0 : (o + 1) % menus.length)); setSub(null); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); setOpen((o) => (o === null ? 0 : (o + menus.length - 1) % menus.length)); setSub(null); }
    };
    document.addEventListener("mousedown", onDown, true);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDown, true);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [open, close, menus.length]);

  const fire = useCallback((it: Item) => {
    if (it.inert || !it.action) { if (!it.sub) close(); return; }
    close();
    try { onAction(it.action); } catch (e) { console.error("[menu] action failed", e); }
  }, [onAction, close]);

  const renderRows = (items: Item[], depth: number): ReactNode => (
    <div className="menu-drop text-[13px]" role="menu">
      {items.map((it, i) => {
        if (it.sep) return <div key={"sep" + i} className="menu-sep" />;
        const isSubOpen = !!it.sub && sub === it.label;
        return (
          <div
            key={it.label + i}
            className="menu-row"
            role="menuitem"
            aria-disabled={it.inert ? "true" : undefined}
            data-active={isSubOpen ? "true" : undefined}
            onMouseEnter={() => { if (depth === 0) setSub(it.sub ? it.label : null); }}
            onClick={(e) => { e.stopPropagation(); if (it.sub) setSub(it.label); else fire(it); }}
          >
            {it.checked ? <span className="menu-check" aria-hidden>&#10003;</span> : null}
            <span className="flex-1"><Mnemonic label={it.label} /></span>
            {it.shortcut ? <span className="ml-6 text-right" style={{ minWidth: 40 }}>{it.shortcut}</span> : null}
            {it.sub ? <span className="menu-arrow" aria-hidden>&#9656;</span> : null}
            {it.sub && isSubOpen ? (
              <div className="absolute left-full top-[-3px] z-10">{renderRows(it.sub, depth + 1)}</div>
            ) : null}
          </div>
        );
      })}
    </div>
  );

  return (
    <div
      ref={rootRef}
      className="relative flex h-[24px] items-center bg-win-gray px-[2px] text-[13px]"
      data-menu-open={open !== null ? "true" : undefined}
      role="menubar"
    >
      {/* Under 420 px of viewport the row has no room for both this and the Share button; the button wins (the icon
          is decoration, the button is the one thing a visitor wants to do with the painting). */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/favicon-48.png" alt="SynapseFly" width={18} height={18} className="ml-[2px] mr-[3px] hidden min-[420px]:block" style={{ imageRendering: "pixelated" }} />
      {menus.map((m, i) => (
        <div key={m.label} className="relative">
          <div
            className="menu-item"
            data-open={open === i ? "true" : undefined}
            role="menuitem"
            aria-haspopup="menu"
            aria-expanded={open === i}
            onMouseDown={(e) => { e.preventDefault(); setSub(null); setOpen((o) => (o === i ? null : i)); }}
            onMouseEnter={() => { if (open !== null && open !== i) { setOpen(i); setSub(null); } }}
          >
            <Mnemonic label={m.label} />
          </div>
          {open === i ? <div className="absolute left-0 top-full z-[1000]">{renderRows(m.items, 0)}</div> : null}
        </div>
      ))}
      {/* The one thing a first-time visitor wants to do with the painting, one click from the canvas. Same action as
          File > Share to X...; PaintWindow catches it and opens ShareDialog.

          This row cannot scroll or wrap (it is 24 px of Win95 chrome and an open dropdown must be allowed to hang
          out of it), so the button has to FIT instead of overflowing the window frame. Measured against the six
          menu titles, which are ~323 px wide with the icon and ~300 px without it:
            >= 420 px  icon + "X Share"        (the six titles + 68 px still fit)
            360-419    no icon, glyph only     (300 + 29 px fits a 340 px row)
            < 360      no button at all        - File > Share to X... is the way in, and nothing sticks out. */}
      <button
        type="button"
        className="btn95 ml-auto mr-[2px] hidden h-[18px] shrink-0 items-center gap-1 text-[11px] min-[360px]:flex"
        onClick={() => { close(); onAction({ kind: "share" }); }}
        title={`Share this painting on X (${X_HANDLE}) - saves the PNG, opens a pre-filled composer`}
        aria-label="Share this painting on X"
        data-testid="share-button"
      >
        <XGlyph /> <span className="hidden min-[420px]:inline">Share</span>
      </button>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- About dialog

export interface AboutDialogProps { hello: HelloMsg | null; store: TickStore; onClose(): void }

/** Help > About FlyBrain: connectome provenance (the synthetic note in bold), run_id, backend, dt, RTF (4 Hz snapshot). */
export function AboutDialog({ hello, store, onClose }: AboutDialogProps) {
  const tick: TickMsg | null = useTickSnapshot(store);
  const c = hello?.connectome ?? null;
  const rows: [string, ReactNode][] = c
    ? [
        ["Connectome", c.name],
        ["Source", c.source],
        ["Neurons / edges", `${c.n.toLocaleString("en-US")} n / ${c.e.toLocaleString("en-US")} e (${Math.round(c.synapses).toLocaleString("en-US")} synapses)`],
        ["Gain", `${c.gain} (${c.weights_mode ?? "n/a"})`],
        ["License", c.license],
        ["Citation", c.citation ?? "none"],
        ["Note", <b key="note">{c.note}</b>],
        ["Run", hello ? `${hello.run_id} (backend ${hello.backend}, dt ${hello.dt_ms} ms, ${hello.tick_hz} Hz ticks, ${hello.realtime ? "realtime" : "no realtime"})` : "-"],
        ["RTF", tick ? `${tick.sim.rtf.toFixed(2)} (brain ${tick.sim.speed.toFixed(2)}x, ${tick.sim.step_ms.toFixed(2)} ms/step)` : "-"],
        ["Market", hello ? `${hello.market.symbol} via ${hello.market.mode} (${hello.market.chain})` : "-"],
        ["Agent", hello ? `llm ${hello.agent.llm}, x ${hello.agent.x}, ${hello.agent.tweets_per_day}/day, lang ${hello.agent.lang}` : "-"],
      ]
    : [["Status", "no hello frame received yet - is the backend running on port 4000?"]];
  return (
    <MessageBox title="About FlyBrain" width={520} buttons={[{ label: "OK", onClick: onClose, default: true }]}
      icon={<PaintIcon size={32} />}>
      <div className="mb-2 font-bold">SynapseFly / FlyBrain ($SYNAPSE)</div>
      <div className="mb-2">A spiking Drosophila male CNS shaped brain (MaleCNS v1.0 shape) wired to a token market, painting in a Win95 Paint window.</div>
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
        <a className="text-win-blue" href={SITE_URL} target="_blank" rel="noopener noreferrer">synapsefly.com</a>
        <a className="text-win-blue" href={X_URL} target="_blank" rel="noopener noreferrer">X {X_HANDLE}</a>
        <a className="text-win-blue" href={GITHUB_URL} target="_blank" rel="noopener noreferrer">GitHub (source)</a>
      </div>
      <div className="mb-2 text-[11px] text-win-dark">
        Tip: drag a title bar to move a window; press <b>G</b> (View &gt; Tile windows) to refit every window to the screen.
        The X and GitHub links also live in the taskbar tray, next to the clock.
      </div>
      <table className="w-full border-collapse text-[11px]">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="align-top">
              <td className="whitespace-nowrap pr-2 text-win-dark">{k}</td>
              <td className="break-words">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </MessageBox>
  );
}

/** Paint program icon (pixel palette + brush), reused by the window title and the About box. */
export function PaintIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" shapeRendering="crispEdges" aria-hidden>
      <rect x="1" y="3" width="11" height="10" fill="#fff" stroke="#000" />
      <rect x="2" y="4" width="3" height="3" fill="#ff0000" />
      <rect x="6" y="4" width="3" height="3" fill="#00ff00" />
      <rect x="2" y="8" width="3" height="3" fill="#0000ff" />
      <rect x="6" y="8" width="3" height="3" fill="#ffff00" />
      <rect x="11" y="1" width="2" height="9" fill="#804000" />
      <rect x="10" y="10" width="4" height="4" fill="#000" />
      <rect x="11" y="11" width="2" height="2" fill="#c0c0c0" />
    </svg>
  );
}
