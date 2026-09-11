"use client";
// SPEC section e.5 `Win95Window`: absolutely positioned window with a .bevel-out frame and a .titlebar; the
// [_][#][x] buttons, title bar drag with pointer capture, position persisted to localStorage "fly.win.<id>" ({x,y}).
// Below 1280 px the desktop stacks the windows vertically (drag disabled).
//
// THIS FILE IS ALSO THE WINDOW MANAGER. The desktop is a real Win95 desktop now - windows open from an icon and
// close for good - so open/minimized/maximized/z-order live in ONE store here instead of in each window's useState:
//
//   * the store is a module-level snapshot read through `useSyncExternalStore`, so page.tsx (taskbar, Start menu,
//     desktop icons, tiling) and every window read exactly the same state without prop-drilling through components
//     this engineer does not own (PaintWindow renders its own Win95Window and cannot forward new props);
//   * it persists to localStorage under a VERSIONED key ("fly.desktop.v1") and validates every field on the way
//     back in - an unknown, stale or corrupt value falls back to the first-visit default instead of bricking
//     the page;
//   * `getServerSnapshot` returns the first-visit default, so SSR and hydration agree and the restored desktop is
//     applied in the commit right after hydration.
//
// A minimized or closed window is NOT unmounted: it is `display:none`. FlyCanvas holds the whole painting in its
// backing bitmap, and unmounting Paint (or the raster's ring buffer) would silently throw the session away - closing
// the window must lose the window, not the fly's work.
//
// Other deviations from the SPEC, driven by "everything must fill the screen":
//   * `rect` (computed by lib/layout.ts from the live viewport) drives BOTH position and size;
//   * a stored (dragged) position still wins over rect.x/y, but it is clamped so the title bar always stays inside
//     the desktop box (nothing can be stranded off-screen);
//   * the maximize button fills the desktop box instead of scaling the body by 1.5x;
//   * `fly.win.reset` (View > Tile windows) drops every stored position, un-maximizes and re-tiles;
//   * `onGeometry` reports the window extent to page.tsx, which grows the desktop so a window dragged below the fold
//     is reachable by scrolling.
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { CASCADE_MAX_SLOTS, type WinRect } from "@/lib/layout";

export type { WinRect };

export interface WinGeometry { right: number; bottom: number }

export interface Win95WindowProps {
  id: string;
  title: string;
  icon?: ReactNode;
  /** Computed layout rect (position AND size). A dragged window keeps its stored x/y, never its own size. */
  rect: WinRect;
  /** Usable desktop area: drag clamp + what "maximize" fills. Defaults to the rect itself. */
  bounds?: WinRect;
  /**
   * Tallest this window's content can actually use. When set, MAXIMIZE fills the desktop width but stops here
   * instead of stretching a fixed-height panel over a screenful of empty gray (Fly Status: its two panels are ~430 px
   * of rows, so a 1030 px maximized window was 84 % flat #c0c0c0). Windows whose body really fills - Paint, the
   * raster, the notepad, the 3D view - leave it unset and maximize to the whole box.
   */
  maxH?: number;
  menu?: ReactNode;
  statusBar?: ReactNode;
  collapsible?: boolean;
  /** Overrides the default [x] behaviour (which is: close the window on the desktop). */
  onClose?: () => void;
  onGeometry?: (id: string, g: WinGeometry) => void;
  children: ReactNode;
}

/**
 * Below this the desktop metaphor is dropped and the windows become one scrolling stack (no taskbar arrangement, no
 * drag, no tiling). It was 1280, which handed the phone layout to every non-maximized laptop window - and that is a
 * worse deal than it sounds: stacked, the icon band eats 104 px of HEIGHT off the top and the Paint window starts
 * below the fold, where the desktop layout fits the whole window on screen with room to spare. 1024 is the narrowest
 * viewport that still leaves a usable column beside the icon gutter (1024 - 16 - 176 = 832 px, and computeLayout
 * degrades to a single column by itself).
 */
export const STACK_BREAKPOINT = 1024;
export const WIN_FOCUS_EVENT = "fly.win.focus";
export const WIN_SHAKE_EVENT = "fly.win.shake";
/** View > Tile windows: every window drops its stored position and snaps back to the computed layout. */
export const WIN_RESET_EVENT = "fly.win.reset";
const TITLE_H = 20;
/** How much of a dragged window must stay inside the desktop box (so its title bar is always grabbable). */
const KEEP_VISIBLE_W = 160;

// ------------------------------------------------------------------------------------------- localStorage (guarded)

/** Same-tab listeners: the `storage` event only fires in OTHER tabs, so lsSet/lsRemove notify locally too. */
const lsListeners = new Set<() => void>();
function notifyLs(): void {
  for (const cb of Array.from(lsListeners)) { try { cb(); } catch { /* ignore */ } }
}

export function lsGet(key: string): string | null {
  try { return typeof window === "undefined" ? null : window.localStorage.getItem(key); } catch { return null; }
}
export function lsSet(key: string, value: string): void {
  try { if (typeof window !== "undefined") window.localStorage.setItem(key, value); } catch { /* private mode */ }
  notifyLs();
}
export function lsRemove(key: string): void {
  try { if (typeof window !== "undefined") window.localStorage.removeItem(key); } catch { /* private mode */ }
  notifyLs();
}

// ------------------------------------------------------------------------------------------- the desktop store

export interface WinState {
  /** On the desktop at all (an icon opens it, [x] closes it). */
  open: boolean;
  /** Minimized to the taskbar. */
  min: boolean;
  /** Maximized over the whole desktop box. */
  max: boolean;
  /** In the tiling (View > Tile windows / key G) instead of cascading on its own. */
  tiled: boolean;
  /** Cascade slot when it is not tiled: the classic down-and-right stagger. */
  slot: number;
}

export interface DesktopState {
  wins: Readonly<Record<string, WinState>>;
  /** Bottom-to-top z-order of the windows that have been opened. The last visible one has the focus. */
  order: readonly string[];
}

/** Versioned: a value written by an older (or a newer, broken) build is ignored rather than half-applied. */
const DESK_KEY = "fly.desktop.v1";
const DESK_VERSION = 1;
/** A window rendered without being registered by initDesktop() behaves like the old always-open one. */
const UNREGISTERED: WinState = { open: true, min: false, max: false, tiled: true, slot: 0 };

let deskDefault: DesktopState = { wins: {}, order: [] };
let deskState: DesktopState | null = null;
const deskSubs = new Set<() => void>();

/**
 * Declare the windows the desktop knows about and which of them a FIRST-TIME visitor sees. Call it once, at module
 * scope, before anything renders (page.tsx does). `deskDefault` is also the server snapshot, so it must be stable.
 */
export function initDesktop(ids: readonly string[], firstVisitOpen: readonly string[]): void {
  const wins: Record<string, WinState> = {};
  for (const id of ids) {
    const open = firstVisitOpen.includes(id);
    wins[id] = { open, min: false, max: false, tiled: open, slot: 0 };
  }
  deskDefault = { wins, order: ids.filter((id) => firstVisitOpen.includes(id)) };
  deskState = null;
}

function parseWin(v: unknown, fallback: WinState): WinState {
  if (!v || typeof v !== "object") return fallback;
  const o = v as Record<string, unknown>;
  const bool = (k: string, d: boolean): boolean => (typeof o[k] === "boolean" ? (o[k] as boolean) : d);
  const slot = typeof o.slot === "number" && Number.isFinite(o.slot) ? Math.max(0, Math.min(64, Math.round(o.slot))) : 0;
  return { open: bool("open", fallback.open), min: bool("min", false), max: bool("max", false), tiled: bool("tiled", fallback.tiled), slot };
}

/** Read the persisted desktop. ANY surprise (missing, unparsable, wrong version, wrong shape) -> the default. */
function loadDesk(): DesktopState {
  const raw = lsGet(DESK_KEY);
  if (!raw) return deskDefault;
  try {
    const p = JSON.parse(raw) as { v?: unknown; wins?: unknown; order?: unknown };
    if (!p || typeof p !== "object" || p.v !== DESK_VERSION) return deskDefault;
    if (!p.wins || typeof p.wins !== "object") return deskDefault;
    const stored = p.wins as Record<string, unknown>;
    const wins: Record<string, WinState> = {};
    for (const id of Object.keys(deskDefault.wins)) wins[id] = parseWin(stored[id], deskDefault.wins[id]);
    const seen = new Set<string>();
    const order: string[] = [];
    if (Array.isArray(p.order)) {
      for (const id of p.order) {
        if (typeof id === "string" && wins[id]?.open && !seen.has(id)) { seen.add(id); order.push(id); }
      }
    }
    for (const id of Object.keys(wins)) if (wins[id].open && !seen.has(id)) order.push(id);
    return { wins, order };
  } catch {
    return deskDefault;
  }
}

function getDesk(): DesktopState {
  if (!deskState) deskState = loadDesk();
  return deskState;
}
function getDeskServer(): DesktopState {
  return deskDefault;
}
function subscribeDesk(cb: () => void): () => void {
  deskSubs.add(cb);
  return () => { deskSubs.delete(cb); };
}
function setDesk(fn: (s: DesktopState) => DesktopState): void {
  const prev = getDesk();
  const next = fn(prev);
  if (next === prev) return;
  deskState = next;
  lsSet(DESK_KEY, JSON.stringify({ v: DESK_VERSION, wins: next.wins, order: next.order }));
  for (const cb of Array.from(deskSubs)) { try { cb(); } catch { /* ignore */ } }
}

const winOf = (s: DesktopState, id: string): WinState => s.wins[id] ?? UNREGISTERED;

function withWin(s: DesktopState, id: string, patch: Partial<WinState>): DesktopState {
  const cur = winOf(s, id);
  const next: WinState = { ...cur, ...patch };
  if (id in s.wins && cur.open === next.open && cur.min === next.min && cur.max === next.max && cur.tiled === next.tiled && cur.slot === next.slot) {
    return s;
  }
  return { wins: { ...s.wins, [id]: next }, order: s.order };
}
function toTop(s: DesktopState, id: string): DesktopState {
  if (s.order.length > 0 && s.order[s.order.length - 1] === id) return s;
  return { wins: s.wins, order: [...s.order.filter((x) => x !== id), id] };
}
function toBottom(s: DesktopState, id: string): DesktopState {
  return { wins: s.wins, order: [id, ...s.order.filter((x) => x !== id)] };
}

/** Open, not minimized: what is actually on the desktop, bottom to top. */
export function visibleWindows(s: DesktopState): string[] {
  return s.order.filter((id) => { const w = s.wins[id]; return !!w && w.open && !w.min; });
}
/** The window with the focus: the top visible one. */
export function focusedWindow(s: DesktopState): string | null {
  const v = visibleWindows(s);
  return v.length ? v[v.length - 1] : null;
}

/** The desktop commands. Every one is an event handler (never called during render). */
export const desktop = {
  /** Desktop icon / Start menu / taskbar: open it, restore it if minimized, or just focus it if it is already up. */
  open(id: string): void {
    setDesk((s) => {
      const w = winOf(s, id);
      if (w.open && !w.min) return toTop(s, id);
      if (w.open) return toTop(withWin(s, id, { min: false }), id);
      // First window on an empty desktop fills it (tiled); every later one cascades, one step down and right, so it
      // never lands on the corner of the window it just covered.
      //
      // The slot is the LOWEST one no other window on the desktop is holding - NOT `others.length`, which collided
      // with a live slot as soon as anything had been closed (open 4 windows, close the 2nd, reopen it: it came back
      // on slot 3, byte-identical to the 4th window's origin, and buried that window's title bar out of reach).
      const others = visibleWindows(s).filter((x) => x !== id);
      const taken = new Set<number>(others.map((x) => winOf(s, x).slot));
      let slot = 0;
      while (taken.has(slot) && slot < CASCADE_MAX_SLOTS) slot += 1;
      return toTop(withWin(s, id, { open: true, min: false, max: false, tiled: others.length === 0, slot }), id);
    });
  },
  close(id: string): void {
    setDesk((s) => {
      const next = withWin(s, id, { open: false, min: false, max: false, tiled: false });
      return { wins: next.wins, order: next.order.filter((x) => x !== id) };
    });
  },
  minimize(id: string): void {
    setDesk((s) => (winOf(s, id).open ? toBottom(withWin(s, id, { min: true }), id) : s));
  },
  toggleMax(id: string): void {
    setDesk((s) => toTop(withWin(s, id, { max: !winOf(s, id).max }), id));
  },
  focus(id: string): void {
    setDesk((s) => (winOf(s, id).min ? toTop(withWin(s, id, { min: false }), id) : toTop(s, id)));
  },
  /** Taskbar button: the focused window minimizes, anything else comes up. */
  toggle(id: string): void {
    setDesk((s) => {
      const w = winOf(s, id);
      if (w.open && !w.min && focusedWindow(s) === id) return toBottom(withWin(s, id, { min: true }), id);
      return toTop(withWin(s, id, { min: false }), id);
    });
  },
  /** View > Tile windows (key G): every visible window rejoins the tiling, un-maximized. */
  tileAll(): void {
    setDesk((s) => {
      let next = s;
      for (const id of visibleWindows(s)) next = withWin(next, id, { tiled: true, max: false });
      return next;
    });
  },
  snapshot(): DesktopState {
    return getDesk();
  },
};

/** The whole desktop state (page.tsx: taskbar, icons, tiling). */
export function useDesktop(): DesktopState {
  return useSyncExternalStore(subscribeDesk, getDesk, getDeskServer);
}

export interface WinView extends WinState { z: number; focused: boolean; visible: boolean }

/** One window's state, plus the derived z-index and focus. */
export function useWinView(id: string): WinView {
  const s = useDesktop();
  return useMemo(() => {
    const w = winOf(s, id);
    const i = s.order.indexOf(id);
    return { ...w, z: 10 + (i < 0 ? 0 : i), focused: focusedWindow(s) === id, visible: w.open && !w.min };
  }, [s, id]);
}

// ------------------------------------------------------------------------------------------- stored positions

const STACK_QUERY = `(max-width: ${STACK_BREAKPOINT - 1}px)`;
function subscribeStacked(cb: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => undefined;
  const mq = window.matchMedia(STACK_QUERY);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}
function getStacked(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(STACK_QUERY).matches;
}
const getStackedServer = (): boolean => false;

/** True when the viewport is narrower than 1280 px (windows stack, drag disabled). SSR renders the desktop layout. */
export function useStacked(): boolean {
  return useSyncExternalStore(subscribeStacked, getStacked, getStackedServer);
}

function subscribeStorage(cb: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  lsListeners.add(cb);
  window.addEventListener("storage", cb);
  return () => { lsListeners.delete(cb); window.removeEventListener("storage", cb); };
}
const getStorageServer = (): string | null => null;

/** The persisted {x,y} of a window (null on the server, when absent or corrupt). */
function useStoredPos(id: string): { x: number; y: number } | null {
  const get = useCallback(() => lsGet("fly.win." + id), [id]);
  const raw = useSyncExternalStore(subscribeStorage, get, getStorageServer);
  return useMemo(() => {
    if (!raw) return null;
    try {
      const p = JSON.parse(raw) as { x?: unknown; y?: unknown };
      if (typeof p.x === "number" && typeof p.y === "number" && Number.isFinite(p.x) && Number.isFinite(p.y)) {
        return { x: Math.max(0, p.x), y: Math.max(0, p.y) };
      }
    } catch { /* corrupt value: keep the computed rect */ }
    return null;
  }, [raw]);
}

/** Ask a window to come to the front (and un-collapse). */
export function focusWindow(id: string): void {
  desktop.focus(id);
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(WIN_FOCUS_EVENT, { detail: { id } }));
}

/** Shake a window (the retro "no, you cannot leave" gesture). */
export function shakeWindow(id: string): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(WIN_SHAKE_EVENT, { detail: { id } }));
}

/** View > Tile windows (key G): forget every stored position and re-tile the windows that are open. */
export function tileWindows(): void {
  if (typeof window === "undefined") return;
  try {
    for (let i = window.localStorage.length - 1; i >= 0; i--) {
      const k = window.localStorage.key(i);
      if (k && k.startsWith("fly.win.")) window.localStorage.removeItem(k);
    }
  } catch { /* private mode */ }
  notifyLs();
  desktop.tileAll();
  window.dispatchEvent(new CustomEvent(WIN_RESET_EVENT));
  try { window.scrollTo({ top: 0, left: 0 }); } catch { /* ignore */ }
}

// ------------------------------------------------------------------------------------------- window

export default function Win95Window({ id, title, icon, rect, bounds, maxH, menu, statusBar, collapsible = true, onClose, onGeometry, children }: Win95WindowProps) {
  const stacked = useStacked();
  const stored = useStoredPos(id);
  const view = useWinView(id);
  const [dragPos, setPos] = useState<{ x: number; y: number } | null>(null);
  const posRef = useRef<{ x: number; y: number } | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [shaking, setShaking] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ dx: number; dy: number; pointerId: number } | null>(null);

  const box: WinRect = bounds ?? rect;
  const maxed = view.max && !stacked;
  const hidden = !view.visible;
  /** Maximized height: the whole box, or the content's own ceiling when the window declared one (see `maxH`). */
  const maxedH = maxH && maxH > 0 ? Math.min(box.h, Math.round(maxH)) : box.h;

  const bringToFront = useCallback(() => { desktop.focus(id); }, [id]);

  useEffect(() => {
    const onFocus = (ev: Event) => {
      const d = (ev as CustomEvent<{ id?: string }>).detail;
      if (d && d.id === id) {
        setCollapsed(false);
        rootRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    };
    const onShake = (ev: Event) => {
      const d = (ev as CustomEvent<{ id?: string }>).detail;
      if (d && d.id === id) setShaking(true);
    };
    const onReset = () => { posRef.current = null; setPos(null); setCollapsed(false); };
    window.addEventListener(WIN_FOCUS_EVENT, onFocus);
    window.addEventListener(WIN_SHAKE_EVENT, onShake);
    window.addEventListener(WIN_RESET_EVENT, onReset);
    return () => {
      window.removeEventListener(WIN_FOCUS_EVENT, onFocus);
      window.removeEventListener(WIN_SHAKE_EVENT, onShake);
      window.removeEventListener(WIN_RESET_EVENT, onReset);
    };
  }, [id]);

  // Clamp a position so the window stays inside the desktop box horizontally and its title bar stays reachable.
  // A window may still hang past the bottom edge (the desktop then grows and scrolls - see page.tsx onGeometry).
  // Pure in props (no DOM reads: it also runs during render): KEEP_VISIBLE_W px of the title bar stay in the box.
  const clampPos = useCallback((x: number, y: number): { x: number; y: number } => {
    const keep = Math.min(Math.max(1, rect.w || KEEP_VISIBLE_W), KEEP_VISIBLE_W);
    const maxX = Math.max(0, box.x + box.w - keep);
    const maxY = Math.max(0, box.y + box.h - TITLE_H - 2);
    const cx = Number.isFinite(x) ? x : 0, cy = Number.isFinite(y) ? y : 0;
    return { x: Math.round(Math.min(Math.max(0, cx), maxX)), y: Math.round(Math.min(Math.max(0, cy), maxY)) };
  }, [box.x, box.y, box.w, box.h, rect.w]);

  const onTitlePointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    bringToFront();
    if (stacked || maxed) return;
    if ((e.target as HTMLElement).closest("button")) return;
    if (e.button !== 0) return;
    const el = rootRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    drag.current = { dx: e.clientX - r.left, dy: e.clientY - r.top, pointerId: e.pointerId };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    e.preventDefault();
  }, [bringToFront, stacked, maxed]);

  const onTitlePointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.pointerId !== e.pointerId) return;
    // The desktop scrolls, so translate viewport coords through the live parent rect instead of offsetLeft/Top.
    const parent = rootRef.current?.offsetParent as HTMLElement | null;
    const pr = parent ? parent.getBoundingClientRect() : null;
    const ox = pr ? pr.left : 0, oy = pr ? pr.top : 0;
    const next = clampPos(e.clientX - d.dx - ox, e.clientY - d.dy - oy);
    posRef.current = next;
    setPos(next);
  }, [clampPos]);

  const onTitlePointerUp = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.pointerId !== e.pointerId) return;
    drag.current = null;
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    // Persist from a ref, NOT from inside a setState updater: React runs updaters during the render phase, and the
    // lsSet notification would then re-render the other windows mid-render ("Cannot update a component ...").
    const p = posRef.current;
    if (p) lsSet("fly.win." + id, JSON.stringify({ x: Math.round(p.x), y: Math.round(p.y) }));
  }, [id]);

  const onCloseClick = useCallback(() => {
    if (onClose) { onClose(); return; }
    desktop.close(id);
  }, [onClose, id]);

  // Position: a dragged/stored position wins over the computed rect, but is clamped back into the box on every
  // resize so a window saved on a 4K screen is not stranded off a laptop viewport.
  const raw = dragPos ?? stored ?? { x: rect.x, y: rect.y };
  const pos = useMemo(() => clampPos(raw.x, raw.y), [clampPos, raw.x, raw.y]);

  const style: CSSProperties = stacked
    ? { position: "relative", width: "100%", height: collapsed ? "auto" : rect.h, zIndex: 1, display: hidden ? "none" : undefined }
    : {
        position: "absolute",
        left: maxed ? box.x : pos.x,
        top: maxed ? box.y : pos.y,
        width: Math.max(KEEP_VISIBLE_W, maxed ? box.w : rect.w),
        height: collapsed ? "auto" : Math.max(TITLE_H + 8, maxed ? maxedH : rect.h),
        zIndex: view.z,
        display: hidden ? "none" : undefined,
      };

  // Report the occupied extent so page.tsx can grow (and therefore scroll) the desktop.
  useEffect(() => {
    if (!onGeometry) return;
    const el = rootRef.current;
    if (!el) return;
    const report = () => {
      if (stacked || hidden) { onGeometry(id, { right: 0, bottom: 0 }); return; }
      onGeometry(id, { right: Math.round(el.offsetLeft + el.offsetWidth), bottom: Math.round(el.offsetTop + el.offsetHeight) });
    };
    report();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(report);
    ro.observe(el);
    return () => ro.disconnect();
  }, [onGeometry, id, stacked, hidden, pos.x, pos.y, rect.w, rect.h, maxed, collapsed]);

  return (
    <div
      ref={rootRef}
      className={"bevel-out flex flex-col select-none " + (shaking ? "win-shake" : "")}
      style={style}
      data-window={id}
      data-open={view.open ? "true" : "false"}
      data-min={view.min ? "true" : "false"}
      data-max={view.max ? "true" : "false"}
      data-focused={view.focused ? "true" : "false"}
      onPointerDownCapture={bringToFront}
      onAnimationEnd={() => setShaking(false)}
    >
      <div
        className={"titlebar flex h-[22px] items-center gap-1 text-[13px]" + (view.focused ? "" : " titlebar-idle")}
        style={{ cursor: stacked || maxed ? "default" : "move", padding: "0 2px 0 4px" }}
        onPointerDown={onTitlePointerDown}
        onPointerMove={onTitlePointerMove}
        onPointerUp={onTitlePointerUp}
        onPointerCancel={onTitlePointerUp}
        onDoubleClick={() => { if (collapsible) setCollapsed((c) => !c); }}
      >
        {icon ? <span className="flex h-4 w-4 items-center justify-center" aria-hidden>{icon}</span> : null}
        <span className="flex-1 truncate">{title}</span>
        <button type="button" className="btn95 h-[16px] w-[18px] text-[11px] leading-none font-bold" title="Minimize to the taskbar"
          style={{ padding: 0 }} onClick={() => desktop.minimize(id)} aria-label="minimize">
          <span className="relative top-[-3px]">_</span>
        </button>
        <button type="button" className="btn95 h-[16px] w-[18px] text-[11px] leading-none font-bold"
          title={maxed ? "Restore (back to the tiled layout)" : "Maximize (fill the desktop)"}
          style={{ padding: 0 }} onClick={() => desktop.toggleMax(id)} aria-pressed={view.max} aria-label="maximize">
          <span className="relative top-[-1px]">&#9633;</span>
        </button>
        <button type="button" className="btn95 ml-[2px] h-[16px] w-[18px] text-[11px] leading-none font-bold" title="Close"
          style={{ padding: 0 }} onClick={onCloseClick} aria-label="close">
          <span className="relative top-[-1px]">&times;</span>
        </button>
      </div>
      {menu ? <div hidden={collapsed}>{menu}</div> : null}
      <div className="relative min-h-0 flex-1 overflow-hidden bg-win-gray" hidden={collapsed}>
        <div className="h-full w-full overflow-hidden">{children}</div>
      </div>
      {statusBar ? <div hidden={collapsed}>{statusBar}</div> : null}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- message box

export interface MessageBoxButton { label: string; onClick: () => void; default?: boolean }
export interface MessageBoxProps { title: string; children: ReactNode; buttons: MessageBoxButton[]; icon?: ReactNode; width?: number }

/** Modal Win95 message box (About dialog, trail-clear confirmation, Shut Down). Esc triggers the last button. */
export function MessageBox({ title, children, buttons, icon, width = 420 }: MessageBoxProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); buttons[buttons.length - 1]?.onClick(); }
      else if (e.key === "Enter") { e.preventDefault(); (buttons.find((b) => b.default) ?? buttons[0])?.onClick(); }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [buttons]);
  return (
    <div className="fixed inset-0 z-[100000] flex items-center justify-center" style={{ background: "rgba(0,0,0,0.15)" }} role="dialog" aria-modal="true">
      <div className="bevel-out flex max-h-[90vh] flex-col overflow-auto" style={{ width, maxWidth: "94vw" }}>
        <div className="titlebar flex h-[22px] items-center text-[13px]" style={{ padding: "0 4px" }}>
          <span className="flex-1 truncate">{title}</span>
        </div>
        <div className="flex gap-3 p-3 text-[12px]">
          {icon ? <div className="shrink-0" aria-hidden>{icon}</div> : null}
          <div className="min-w-0 flex-1">{children}</div>
        </div>
        <div className="flex justify-center gap-2 px-3 pb-3">
          {buttons.map((b) => (
            <button key={b.label} type="button" className="btn95 h-[23px] min-w-[75px]" onClick={b.onClick}
              style={b.default ? { outline: "1px solid #000", outlineOffset: -3 } : undefined}>
              {b.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
