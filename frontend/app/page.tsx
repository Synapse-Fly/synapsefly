"use client";
// SPEC section e.7 `page.tsx`: creates the socket, the canvas handle, the UI flags, the MenuAction dispatcher, the
// document-level keyboard shortcuts and the snapshot service (section e.6).
//
// This is now a REAL Windows 95 desktop, not five windows that are always open:
//   * big desktop icons down the left edge open the apps, and the X / GitHub icons are shortcuts that open the link
//     directly (the visitor could not find those links when they were 16 px buttons in the taskbar corner);
//   * a window opens from an icon, the Start menu or its taskbar button; [x] closes it, [_] minimizes it to the
//     taskbar, the middle button maximizes it over the desktop. The whole thing persists (Win95Window's store), so
//     a returning visitor gets their desktop back;
//   * a FIRST-TIME visitor gets exactly one window - "untitled - Paint" - because the painting fly IS the product;
//   * the tiling in lib/layout.ts fills the desktop with whatever subset is open (the taskbar's Tile button,
//     View > Tile windows, key G), and a window opened on its own cascades instead of landing on the rect of the
//     one it covers - the cascade is authentic, and the Tile button is the visible way back out of it;
//   * windows never tile over the icon column: the tiling box starts to the right of it, and the column never
//     spends a second column on a single leftover icon.
//
// OfflineNotice and ShareDialog are rendered by PaintWindow (it owns the canvas handle), not from here.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./desktop.css";
import { useFlySocket } from "@/lib/ws";
import { CA_FILE, CA_WINDOW_TITLE, TOKEN, X_URL, X_HANDLE, GITHUB_URL } from "@/lib/brand";
import { postMarketMode, postSnapshot, postTweetTest } from "@/lib/api";
import { postPoke } from "@/lib/api";
import type { ClientMsg, MarketMode, PokeStim } from "@/lib/types";
import {
  cascadeRect, computeDeskBox, computeLayout, computeStackedLayout, DESK_MARGIN, TASKBAR_H, WINDOW_IDS,
  type WinRect, type WindowId,
} from "@/lib/layout";
import PaintWindow from "@/components/PaintWindow";
import Win95Window, {
  MessageBox, desktop, initDesktop, tileWindows, useDesktop, useStacked, visibleWindows,
  focusedWindow, type WinGeometry,
} from "@/components/Win95Window";
import { AboutDialog, type MenuAction, type MenuFlags } from "@/components/MenuBar";
import type { FlyCanvasHandle } from "@/components/FlyCanvas";
import DesktopIcons, { DESKTOP_CELL_H, DESKTOP_CELL_W, type DesktopItem } from "@/components/DesktopIcons";
import StartMenu, { type StartProgram } from "@/components/StartMenu";
import { AppIcon, type IconName } from "@/lib/icons";
import SpikeRaster from "@/components/SpikeRaster";
import MoodPanel from "@/components/MoodPanel";
import MarketTicker from "@/components/MarketTicker";
import TweetNotepad from "@/components/TweetNotepad";
import IntroModal from "@/components/IntroModal";
import BrainView3D from "@/components/BrainView3D";
import ContractWindow, { CA_WINDOW_ID } from "@/components/ContractWindow";

type Dialog = null | { kind: "about" } | { kind: "confirm_clear"; run_id: string } | { kind: "bin" };

const SITE_URL = "https://www.synapsefly.com";
const KEY_POKES: Record<string, PokeStim> = { S: "sugar", B: "bitter", L: "loom", W: "water", D: "dust", C: "pheromone", Z: "sleep", R: "reward", P: "punish" };
const KEY_REGIMES: Record<string, MarketMode> = { "1": "CALM", "2": "PUMP", "3": "DUMP", "4": "CHOP", "5": "RUG", "6": "DEAD", "0": "sim" };
/** Free key (S B L W D C Z R P, 0-6, N F T and ? are taken): tile the open windows back over the viewport. */
const KEY_TILE = "G";
/** The explainer is an icon and a Start menu row, but it is a modal, not a window - so it has its own pseudo-id. */
const README_ID = "readme";
const BIN_ID = "bin";
/**
 * Fly Status is the one window whose body does NOT fill: MoodPanel is ~430 px of fixed rows and MarketTicker ~360,
 * so maximizing it to a 1030 px desktop left ~84 % of the window flat #c0c0c0. Maximize therefore fills the desktop
 * WIDTH and stops at the height the two panels can use (with slack, and they scroll their own overflow anyway).
 * Everything else - Paint, the raster, the notepad, the 3D view - fills its body and maximizes to the whole box.
 */
const STATUS_MAX_H = 520;
/**
 * CA.txt is a dialog, not a tiled app: it holds one address, two links and a warning. It is NOT in lib/layout.ts's
 * WINDOW_IDS (that file belongs to the tiler and the five apps that fill the desktop), so it never joins the tiling
 * and gets its own centred rect instead - see `caRect`. These are its size; the rect clamps them to the box.
 */
const CA_W = 436, CA_H = 258, CA_STACKED_H = 300;

/** `id` is a plain string, not a WindowId: CA.txt lives in the desktop store without being one of the tiled five. */
interface AppMeta {
  id: string;
  label: string;
  icon: IconName;
  title: string;
  /** Desktop icon label, when the window title is too long for an 80 px cell ("CA.txt - Notepad" -> "CA.txt"). */
  desk?: string;
}

/**
 * The apps, in the order they appear on the desktop, in the taskbar and in Start > Programs.
 *
 * CA.txt is deliberately SECOND, directly under Paint: from launch on, "where is the contract address" is the single
 * most common reason anyone opens this page, and the answer has to be above the fold on a laptop without scrolling,
 * hunting or a Start menu. Paint keeps the top cell because the painting fly is still the product.
 */
const APPS: readonly AppMeta[] = [
  { id: "paint", label: "untitled - Paint", icon: "paint", title: "The fly paints here. Double the product, half the pixels." },
  { id: CA_WINDOW_ID, label: CA_WINDOW_TITLE, desk: CA_FILE, icon: "contract", title: `The $${TOKEN} contract address - check it here, not in a reply or a DM.` },
  { id: "raster", label: "Oscilloscope", icon: "oscilloscope", title: "Spike raster: every dot is one neuron firing." },
  { id: "status", label: "Fly Status", icon: "status", title: "Mood, drives and the live market feed." },
  { id: "brain", label: "Fly Brain (3D)", icon: "brain", title: "The synthetic connectome, spiking in 3D." },
  { id: "tweets", label: "tweets.txt", icon: "notepad", title: "What the fly has posted." },
];

const DESKTOP_ITEMS: readonly DesktopItem[] = [
  ...APPS.map((a): DesktopItem => ({ id: a.id, label: a.desk ?? a.label, icon: a.icon, kind: "window", title: a.title })),
  { id: README_ID, label: "Read Me.txt", icon: "readme", kind: "window", title: "What is this? Start here." },
  { id: "link-x", label: `Follow ${X_HANDLE}`, icon: "x", kind: "link", href: X_URL, title: `SynapseFly on X (${X_HANDLE}) - opens in a new tab` },
  { id: "link-github", label: "GitHub", icon: "github", kind: "link", href: GITHUB_URL, title: "Source on GitHub (Synapse-Fly/synapsefly) - opens in a new tab" },
  { id: BIN_ID, label: "Recycle Bin", icon: "bin", kind: "window", title: "The fly has never deleted anything." },
];
/**
 * The same column without the joke item. When the viewport's column height leaves exactly ONE icon over for a second
 * column, that icon costs the tiling a whole 80 px of desktop width - so the Recycle Bin drops out instead: it is the
 * only icon on the desktop that opens nothing. (The original case was a 1366 x 768 laptop, which fits eight cells per
 * column; the rule is written against the live count, not against nine.)
 */
const DESKTOP_ITEMS_NO_BIN: readonly DesktopItem[] = DESKTOP_ITEMS.filter((it) => it.id !== BIN_ID);
/** DesktopIcons' own padding, both edges: the column's usable height is its box minus this. */
const ICON_PAD = 16;

const PROGRAMS: readonly StartProgram[] = [
  ...APPS.map((a): StartProgram => ({ id: a.id, label: a.label, icon: a.icon })),
  { id: README_ID, label: "Read Me.txt", icon: "readme" },
];

// The window manager needs to know the ids and what a first-time visitor sees BEFORE anything renders: exactly one
// window, Paint, because a visitor who lands on five windows does not know where to look.
//
// CA.txt is registered here alongside the five tiled windows - the desktop store keys on plain strings, so it gets
// open/minimized/maximized/z-order and a persisted state like any other window - but it stays CLOSED on a first
// visit. The launch post tells people to look for the icon, and an unasked-for window over the painting would be the
// one thing louder than the product. One click opens it.
initDesktop([...WINDOW_IDS, CA_WINDOW_ID], ["paint"]);

/** Live viewport box, read from documentElement (excludes scrollbars) and throttled with rAF. SSR uses a laptop. */
function useViewport(): { w: number; h: number } {
  const [vp, setVp] = useState<{ w: number; h: number }>({ w: 1440, h: 860 });
  useEffect(() => {
    let raf = 0;
    const read = () => {
      raf = 0;
      const d = document.documentElement;
      const w = d.clientWidth || window.innerWidth || 1440;
      const h = d.clientHeight || window.innerHeight || 860;
      setVp((p) => (p.w === w && p.h === h ? p : { w, h }));
    };
    const onResize = () => { if (!raf) raf = requestAnimationFrame(read); };
    read();
    window.addEventListener("resize", onResize);
    window.addEventListener("orientationchange", onResize);
    // A scrollbar appearing or disappearing changes clientWidth/Height WITHOUT a resize event; without this the
    // desktop would keep a ~10 px dead strip after the first layout removes the initial scrollbars.
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    ro?.observe(document.documentElement);
    return () => {
      if (raf) cancelAnimationFrame(raf);
      ro?.disconnect();
      window.removeEventListener("resize", onResize);
      window.removeEventListener("orientationchange", onResize);
    };
  }, []);
  return vp;
}

function isEditableTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable === true;
}

function Clock() {
  const [now, setNow] = useState<string>("");
  useEffect(() => {
    const fmt = () => {
      const d = new Date();
      const hh = d.getHours(), mm = d.getMinutes();
      const h12 = hh % 12 === 0 ? 12 : hh % 12;
      setNow(`${h12}:${mm < 10 ? "0" : ""}${mm} ${hh < 12 ? "AM" : "PM"}`);
    };
    fmt();
    const t = setInterval(fmt, 1000);
    return () => clearInterval(t);
  }, []);
  return <div className="bevel-in flex h-[22px] shrink-0 items-center bg-win-gray px-2" suppressHydrationWarning>{now}</div>;
}

// ------------------------------------------------------------------------------ tray icons (inline SVG, no deps)

function XGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M18.24 2.25h3.31l-7.23 8.26 8.5 11.24h-6.64l-5.2-6.8-5.95 6.8H1.72l7.48-8.55L1.05 2.25h6.8l4.71 6.23 5.68-6.23Zm-1.16 17.52h1.83L6.99 4.13H5.03l12.05 15.64Z" />
    </svg>
  );
}
/** Four tiled panes, each with its navy title bar: the taskbar Tile button's glyph. Hand-placed, crisp, no deps. */
function TileGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" shapeRendering="crispEdges" aria-hidden focusable="false" style={{ display: "block" }}>
      <rect x="0" y="0" width="12" height="12" fill="#000000" />
      <rect x="1" y="1" width="4" height="4" fill="#ffffff" />
      <rect x="7" y="1" width="4" height="4" fill="#ffffff" />
      <rect x="1" y="7" width="4" height="4" fill="#ffffff" />
      <rect x="7" y="7" width="4" height="4" fill="#ffffff" />
      <rect x="1" y="1" width="4" height="1" fill="#000080" />
      <rect x="7" y="1" width="4" height="1" fill="#000080" />
      <rect x="1" y="7" width="4" height="1" fill="#000080" />
      <rect x="7" y="7" width="4" height="1" fill="#000080" />
    </svg>
  );
}
function GitHubGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-2.91-.88-2.91-2.77 0-.82.28-1.48.74-2-.07-.2-.31-.93.07-1.92 0 0 .65-.2 2.15.76a5.5 5.5 0 0 1 1.48-.2c.5 0 1 .07 1.48.2 1.5-.97 2.15-.76 2.15-.76.38.99.14 1.72.07 1.92.46.52.74 1.17.74 2 0 1.9-1.14 2.57-2.92 2.77.3.26.56.76.56 1.54 0 1.11-.01 2-.01 2.27 0 .21.15.46.55.38A7.99 7.99 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

export default function Home() {
  const sock = useFlySocket();
  const stacked = useStacked();
  const vp = useViewport();
  const desk = useDesktop();

  // The icon column owns a strip on the left of the desktop, so no tiled window can ever bury it. It is one cell
  // wide, or two when the icons have to wrap on a short screen (DesktopIcons wraps into a second column itself) -
  // but never two for a single leftover icon, see `icons` below.
  const iconAreaH = useMemo(() => computeDeskBox(vp.w, vp.h, 0).h + DESK_MARGIN, [vp.w, vp.h]);
  const iconRows = useMemo(() => Math.max(1, Math.floor((iconAreaH - ICON_PAD) / DESKTOP_CELL_H)), [iconAreaH]);
  /**
   * The icons the column actually shows. A wrap that leaves exactly ONE icon in the last column is the bad case -
   * it buys the visitor one icon for a full 80 px of tiling width - so there the Recycle Bin drops out and the column
   * stays whole. Every other wrap point is balanced enough to keep, and a tall enough viewport shows every icon (a
   * 1080 px screen fits all ten in one column). CA.txt is second either way, so it is above the fold on anything. A
   * stacked phone lays the icons out as ROWS across the top, where the column height means nothing, so nothing is
   * ever dropped there.
   */
  const icons = useMemo(
    () => (!stacked && iconRows > 1 && DESKTOP_ITEMS.length % iconRows === 1 ? DESKTOP_ITEMS_NO_BIN : DESKTOP_ITEMS),
    [stacked, iconRows],
  );
  const gutter = useMemo(
    () => Math.ceil(icons.length / iconRows) * DESKTOP_CELL_W + ICON_PAD,
    [icons, iconRows],
  );
  /** Stacked (phone) desktop: the icons are a band of rows across the top instead of a column down the side. */
  const stackedIconsH = useMemo(() => {
    const cols = Math.max(1, Math.floor((vp.w - ICON_PAD) / DESKTOP_CELL_W));
    return Math.ceil(DESKTOP_ITEMS.length / cols) * DESKTOP_CELL_H + ICON_PAD;
  }, [vp.w]);

  const tiled = useMemo(
    () => WINDOW_IDS.filter((id) => { const w = desk.wins[id]; return !!w && w.open && !w.min && w.tiled; }),
    [desk],
  );
  const layout = useMemo(() => computeLayout(vp.w, vp.h, { open: tiled, gutter }), [vp.w, vp.h, tiled, gutter]);
  const stackedRects = useMemo(() => computeStackedLayout(vp.w), [vp.w]);
  const rects: Record<WindowId, WinRect> = useMemo(() => {
    if (stacked) return stackedRects;
    const out = {} as Record<WindowId, WinRect>;
    for (const id of WINDOW_IDS) {
      const w = desk.wins[id];
      out[id] = w && !w.tiled ? cascadeRect(layout.box, w.slot, id) : layout.win[id];
    }
    return out;
  }, [stacked, stackedRects, desk, layout]);
  /**
   * CA.txt's rect. It is not one of the tiled five, so it gets no share of the tiling: it opens CENTRED over the
   * desktop (a third of the way down, the classic dialog position), clamped to the box so it is whole at any size.
   * Stacked, it is one full-width row of the scrolling column like every other window. A visitor who drags it keeps
   * their position (Win95Window's stored x/y wins over this).
   */
  const caRect = useMemo<WinRect>(() => {
    if (stacked) return { x: 0, y: 0, w: stackedRects.paint.w, h: CA_STACKED_H };
    const box = layout.box;
    const w = Math.min(CA_W, box.w);
    const h = Math.min(CA_H, box.h);
    return {
      x: box.x + Math.max(0, Math.round((box.w - w) / 2)),
      y: box.y + Math.max(0, Math.round((box.h - h) / 3)),
      w, h,
    };
  }, [stacked, stackedRects, layout.box]);

  const canvasRef = useRef<FlyCanvasHandle | null>(null);
  const [flags, setFlags] = useState<MenuFlags>({ labels: true, frozen: false, zoom: false, fps: false, flip: false });
  const [dialog, setDialog] = useState<Dialog>(null);
  const dialogRef = useRef<Dialog>(null);
  useEffect(() => { dialogRef.current = dialog; }, [dialog]);
  const [showIntro, setShowIntro] = useState(false);
  const showIntroRef = useRef(false);
  useEffect(() => { showIntroRef.current = showIntro; }, [showIntro]);
  // First visit opens the intro. Deferred to the next frame so it is a callback from an external system (rAF) rather
  // than a synchronous setState inside the effect body (react-hooks/set-state-in-effect).
  useEffect(() => {
    let seen = true;
    try { seen = !!localStorage.getItem("synapsefly_intro_v1"); } catch { /* private mode */ }
    if (seen) return;
    const raf = requestAnimationFrame(() => setShowIntro(true));
    return () => cancelAnimationFrame(raf);
  }, []);
  const closeIntro = useCallback(() => { setShowIntro(false); try { localStorage.setItem("synapsefly_intro_v1", "1"); } catch { /* ignore */ } }, []);
  const openIntro = useCallback(() => setShowIntro(true), []);
  const lastRunId = useRef<string | null>(null);
  const { send, on } = sock;

  // The desktop grows to the furthest window edge so a window dragged out of the default tiling stays reachable by
  // scrolling; the default tiling fits the viewport, so nothing scrolls until the user moves one.
  const geom = useRef<Map<string, WinGeometry>>(new Map());
  const [extent, setExtent] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const onGeometry = useCallback((id: string, g: WinGeometry) => {
    const prev = geom.current.get(id);
    if (prev && prev.right === g.right && prev.bottom === g.bottom) return;
    geom.current.set(id, g);
    let w = 0, h = 0;
    for (const v of geom.current.values()) { w = Math.max(w, v.right); h = Math.max(h, v.bottom); }
    setExtent((p) => (p.w === w && p.h === h ? p : { w, h }));
  }, []);

  // ------------------------------------------------------------------------ opening things

  /** Desktop icon, Start menu row or taskbar button. The two modals have pseudo-ids; everything else is a window. */
  const openApp = useCallback((id: string) => {
    if (id === README_ID) { setShowIntro(true); return; }
    if (id === BIN_ID) { setDialog({ kind: "bin" }); return; }
    desktop.open(id);
  }, []);

  const onIconOpen = useCallback((item: DesktopItem) => {
    // A "link" item is a real anchor: it navigates itself, and window.open here would spawn a second tab.
    if (item.kind === "link") return;
    openApp(item.id);
  }, [openApp]);

  const openWindows = useMemo(() => APPS.filter((a) => desk.wins[a.id]?.open), [desk]);
  const focused = useMemo(() => focusedWindow(desk), [desk]);
  const openIds = useMemo(() => {
    const ids: string[] = visibleWindows(desk);
    if (showIntro) ids.push(README_ID);
    return ids;
  }, [desk, showIntro]);

  const sendOr = useCallback((m: ClientMsg, fallback: () => Promise<unknown>) => {
    if (send(m)) return;
    fallback().catch((e) => console.warn("[page] REST fallback failed", e));
  }, [send]);

  const dispatch = useCallback((a: MenuAction) => {
    try {
      switch (a.kind) {
        case "clear":
          canvasRef.current?.clear();
          sendOr({ type: "clear" }, async () => undefined);
          break;
        case "save": {
          const c = canvasRef.current?.composite();
          if (!c) return;
          const link = document.createElement("a");
          link.href = c.toDataURL("image/png");
          link.download = `flybrain-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
          document.body.appendChild(link);
          link.click();
          link.remove();
          break;
        }
        case "poke":
          sendOr({ type: "poke", stim: a.stim, strength: 1, side: "both", duration_ms: 500 }, () => postPoke(a.stim, 1, "both", 500));
          break;
        case "market":
          sendOr({ type: "set_market_mode", mode: a.mode }, () => postMarketMode(a.mode));
          break;
        case "tweet_test":
          sendOr({ type: "tweet_test" }, () => postTweetTest());
          break;
        case "toggle":
          setFlags((f) => {
            switch (a.what) {
              case "labels": return { ...f, labels: !f.labels };
              case "freeze": return { ...f, frozen: !f.frozen };
              case "zoom": return { ...f, zoom: !f.zoom };
              case "fps": return { ...f, fps: !f.fps };
              case "flip": return { ...f, flip: !f.flip };
              default: return f;
            }
          });
          break;
        case "about":
          setDialog({ kind: "about" });
          break;
        case "link":
          window.open(a.url, "_blank", "noopener,noreferrer");
          break;
        case "reset_layout":
          // Drop every stored window position, un-maximize, and re-tile whatever is open (no reload: the socket
          // stays open).
          tileWindows();
          // CA.txt is a dialog, not one of the tiled five (it is not in WINDOW_IDS), so the tiler never gives it a
          // slot: left open, tileWindows() would mark it `tiled` yet snap it back to its centred dialog rect,
          // floating over the freshly tiled grid. Dismiss it instead (close() also clears the bogus `tiled` flag);
          // one click on its icon, tray chip or Start entry brings it back, centred.
          desktop.close(CA_WINDOW_ID);
          break;
        case "exit":
          // File > Exit closes the Paint window, like an application. Its desktop icon brings it back.
          desktop.close("paint");
          break;
        default:
          break;
      }
    } catch (e) {
      console.error("[page] action failed", e);
    }
  }, [sendOr]);

  // Keyboard shortcuts (section e.7), ignored while a menu or dialog is open or an input is focused.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || isEditableTarget(e.target)) return;
      if (dialogRef.current || showIntroRef.current) return;
      if (document.querySelector("[data-menu-open]")) return;
      if (e.ctrlKey || e.metaKey) {
        if (e.key.toLowerCase() === "n" && !e.shiftKey && !e.altKey) { e.preventDefault(); dispatch({ kind: "clear" }); }
        return;
      }
      if (e.altKey) return;
      const k = e.key.length === 1 ? e.key.toUpperCase() : e.key;
      // "?" is Shift+/ on a US layout and Shift+, or a dead key elsewhere: accept the produced char or the code.
      if (k === "?" || (e.shiftKey && (e.key === "/" || e.code === "Slash"))) { e.preventDefault(); dispatch({ kind: "about" }); return; }
      if (KEY_POKES[k]) { e.preventDefault(); dispatch({ kind: "poke", stim: KEY_POKES[k] }); return; }
      if (KEY_REGIMES[k]) { e.preventDefault(); dispatch({ kind: "market", mode: KEY_REGIMES[k] }); return; }
      if (k === "N") { e.preventDefault(); dispatch({ kind: "clear" }); return; }
      if (k === "F") { e.preventDefault(); dispatch({ kind: "toggle", what: "freeze" }); return; }
      if (k === "T") { e.preventDefault(); dispatch({ kind: "tweet_test" }); return; }
      if (k === KEY_TILE) { e.preventDefault(); dispatch({ kind: "reset_layout" }); return; }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [dispatch]);

  // Snapshot service (section e.6): composite the canvases and answer within deadline_ms.
  useEffect(() => on("snapshot_request", (m) => {
    try {
      const c = canvasRef.current?.composite();
      if (!c) return;
      const png_b64 = c.toDataURL("image/png").split(",")[1] ?? "";
      if (!png_b64) return;
      if (!send({ type: "snapshot", id: m.id, png_b64 })) {
        postSnapshot(m.id, png_b64).catch((e) => console.warn("[page] snapshot REST fallback failed", e));
      }
    } catch (e) {
      console.error("[page] snapshot failed", e);
    }
  }), [on, send]);

  // A hello with a different run_id after a reconnect = a new session: clear the trail (ask first when non-empty).
  useEffect(() => on("hello", (h) => {
    const prev = lastRunId.current;
    lastRunId.current = h.run_id;
    if (prev && prev !== h.run_id) {
      if (canvasRef.current?.dirty()) setDialog({ kind: "confirm_clear", run_id: h.run_id });
      else canvasRef.current?.clear();
    }
  }), [on]);

  const desktopClass = stacked ? "flex flex-col gap-2 p-2" : "relative";
  const desktopStyle = stacked
    ? undefined
    : {
        width: "100%",
        // No DESK_MARGIN on the width: a window whose right edge sits a few px inside the box still measured wider
        // than clientWidth once the vertical scrollbar had taken 10 px, and the desktop then grew a HORIZONTAL
        // scrollbar for ~7 px of nothing. The margin still applies vertically, where the desktop is meant to grow.
        minWidth: Math.max(layout.deskW, extent.w),
        height: Math.max(layout.deskH, extent.h + DESK_MARGIN),
      };

  return (
    <div className="min-h-screen" style={{ paddingBottom: TASKBAR_H }} data-testid="desktop">
      <div className={desktopClass} style={desktopStyle} data-cols={stacked ? "stacked" : layout.cols}>
        {/* The icon column. On a phone the desktop is a single scrolling stack, so the icons become a band of rows
            above the windows instead of a column beside them. */}
        {stacked ? (
          <div className="shrink-0" style={{ height: stackedIconsH }} data-testid="desktop-icons">
            <DesktopIcons items={icons} openIds={openIds} onOpen={onIconOpen} />
          </div>
        ) : (
          <div className="icon-gutter" style={{ width: gutter, height: iconAreaH }} data-testid="desktop-icons">
            <DesktopIcons items={icons} openIds={openIds} onOpen={onIconOpen} />
          </div>
        )}

        <PaintWindow sock={sock} onAction={dispatch} canvasRef={canvasRef} flags={flags} rect={rects.paint} bounds={layout.box} onGeometry={onGeometry} />
        <Win95Window id="raster" title="Oscilloscope - spike raster" icon={<AppIcon name="oscilloscope" size={16} />} rect={rects.raster} bounds={layout.box} onGeometry={onGeometry}>
          <SpikeRaster sock={sock} frozen={flags.frozen} labels={flags.labels} />
        </Win95Window>
        <Win95Window id="status" title="Fly Status" icon={<AppIcon name="status" size={16} />} rect={rects.status} bounds={layout.box} maxH={STATUS_MAX_H} onGeometry={onGeometry}>
          <div className="grid h-full min-h-0 grid-cols-2 gap-[2px] overflow-hidden bg-win-gray">
            <div className="min-h-0 overflow-auto"><MoodPanel sock={sock} /></div>
            <div className="min-h-0 overflow-auto"><MarketTicker sock={sock} /></div>
          </div>
        </Win95Window>
        <Win95Window id="brain" title="Fly Brain (3D) - live connectome" icon={<AppIcon name="brain" size={16} />} rect={rects.brain} bounds={layout.box} onGeometry={onGeometry}>
          <BrainView3D sock={sock} />
        </Win95Window>
        <Win95Window id="tweets" title="tweets.txt - Notepad" icon={<AppIcon name="notepad" size={16} />} rect={rects.tweets} bounds={layout.box} onGeometry={onGeometry} collapsible>
          <TweetNotepad sock={sock} onTest={() => dispatch({ kind: "tweet_test" })} />
        </Win95Window>
        {/* CA.txt renders its own window chrome (like PaintWindow) so the title, the icon and the two states live in
            one file. Same store, same [_][#][x], same taskbar button - it is just not in the tiling. */}
        <ContractWindow sock={sock} rect={caRect} bounds={layout.box} onGeometry={onGeometry} />
      </div>

      <div className="bevel-out fixed bottom-0 left-0 right-0 z-[99999] flex h-[30px] items-center gap-[3px] px-1" role="toolbar" aria-label="taskbar" data-testid="taskbar">
        <StartMenu
          programs={PROGRAMS}
          onOpen={openApp}
          onIntro={openIntro}
          onAbout={() => setDialog({ kind: "about" })}
          onTile={() => dispatch({ kind: "reset_layout" })}
          siteUrl={SITE_URL}
        />
        {/* The ONLY on-screen affordance for the tiling. Opening five apps one by one cascades them, which is
            authentic, but nothing told a visitor that key G / View > Tile windows exists - so four windows stayed
            ~80 % buried. Same command, same keyboard shortcut, same menu row: this is just the visible handle.
            Deliberately NOT automatic - the visitor chose click-to-open, and re-arranging their windows behind
            their back would fight that. Hidden on a stacked phone, where there is nothing to tile. */}
        <button
          type="button"
          className="btn95 tile-btn"
          title="Tile windows (G)"
          aria-label="Tile windows"
          data-testid="tile-btn"
          onClick={() => dispatch({ kind: "reset_layout" })}
        >
          <TileGlyph /><span>Tile</span>
        </button>
        <div className="tray-sep" aria-hidden />
        {/* One button per OPEN window. The focused one is pressed in; clicking it minimizes it, clicking any other
            brings it up - exactly the Win95 taskbar. Under 768 px the label is hidden and the button is its 16 px
            icon (desktop.css), so four fit instead of one; aria-label keeps the window's name as the accessible
            name, and the strip scrolls horizontally - scrollbar hidden - to reach the rest. */}
        <div className="taskbar-apps flex min-w-0 flex-1 items-center gap-[3px]" data-testid="taskbar-apps">
          {openWindows.map((a) => {
            const w = desk.wins[a.id];
            const min = !!w?.min;
            return (
              <button
                key={a.id}
                type="button"
                className="btn95 task-btn"
                data-task={a.id}
                data-min={min ? "true" : "false"}
                aria-pressed={focused === a.id}
                aria-label={min ? `${a.label} (minimized)` : a.label}
                onClick={() => desktop.toggle(a.id)}
                title={min ? `${a.label} (minimized - click to restore)` : a.label}
              >
                <AppIcon name={a.icon} size={16} />
                <span>{a.label}</span>
              </button>
            );
          })}
        </div>
        <div className="tray-sep" aria-hidden />
        {/* System tray. The links live on the desktop and in the Start menu now; these stay as the always-visible
            shortcut they always were.
            The CA chip leads the tray because it is the one thing an arriving visitor is looking for, and unlike its
            neighbours it is a BUTTON, not a link: it opens CA.txt on this page (where the address comes off the live
            wire and carries its verify-this warning) instead of sending anyone to a third-party page to find it.
            Below md it is the icon alone, exactly like the X and GitHub chips, and like them it is flex: none - the
            tray can never push Start, the connection LED or the clock off a 390 px taskbar. */}
        <button type="button" className="btn95 ca-chip" onClick={() => openApp(CA_WINDOW_ID)}
          title={`${CA_WINDOW_TITLE} - check the $${TOKEN} contract address here`} aria-label={`Open ${CA_WINDOW_TITLE}`}
          data-testid="tray-ca">
          <AppIcon name="contract" size={16} /><span className="hidden md:inline">CA</span>
        </button>
        <a href={X_URL} target="_blank" rel="noopener noreferrer" title={`SynapseFly on X (${X_HANDLE})`}
          aria-label={`SynapseFly on X, ${X_HANDLE}`} className="btn95 flex h-[22px] shrink-0 items-center gap-1 text-[11px] no-underline">
          <XGlyph /><span className="hidden md:inline">{X_HANDLE}</span>
        </a>
        <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" title="Source on GitHub (Synapse-Fly/synapsefly)"
          aria-label="Source on GitHub" className="btn95 flex h-[22px] shrink-0 items-center gap-1 text-[11px] no-underline">
          <GitHubGlyph /><span className="hidden md:inline">GitHub</span>
        </a>
        <div className="tray-sep" aria-hidden />
        <div className="bevel-in flex h-[22px] shrink-0 items-center gap-1 bg-win-gray px-2 text-[11px]" title={`websocket ${sock.status}${sock.latencyMs !== null ? `, ${sock.latencyMs} ms` : ""}`}>
          <span className="inline-block h-[8px] w-[8px] rounded-full border border-black" style={{ background: sock.status === "open" ? "#00a800" : sock.status === "connecting" ? "#e0c000" : "#d00000" }} />
          <span className="hidden sm:inline">
            {sock.status === "open" ? (sock.latencyMs !== null ? `${sock.latencyMs} ms` : "online") : sock.status === "reconnecting" ? `reconnecting (${sock.attempt})` : sock.status}
          </span>
        </div>
        <Clock />
      </div>

      {showIntro ? <IntroModal sock={sock} onClose={closeIntro} /> : null}
      {dialog?.kind === "about" ? <AboutDialog hello={sock.hello} store={sock.store} onClose={() => setDialog(null)} /> : null}
      {dialog?.kind === "bin" ? (
        <MessageBox
          title="Recycle Bin"
          icon={<AppIcon name="bin" size={32} />}
          buttons={[{ label: "OK", default: true, onClick: () => setDialog(null) }]}
        >
          <div className="font-bold">Empty.</div>
          <div className="mt-1">
            The fly has never deleted anything. Every stroke it has ever painted is still on the canvas, and there is
            no undo - only File &gt; New, which the fly cannot reach.
          </div>
        </MessageBox>
      ) : null}
      {dialog?.kind === "confirm_clear" ? (
        <MessageBox
          title="Paint"
          icon={<span className="text-[28px] leading-none" aria-hidden>&#9432;</span>}
          buttons={[
            { label: "Yes", default: true, onClick: () => { canvasRef.current?.clear(); setDialog(null); } },
            { label: "No", onClick: () => setDialog(null) },
          ]}
        >
          <div>The brain restarted (new session {dialog.run_id}).</div>
          <div className="mt-1">Clear the trail from the previous session?</div>
        </MessageBox>
      ) : null}
    </div>
  );
}
