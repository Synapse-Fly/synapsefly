"use client";
// SPEC section e.5 `Win95Window`: absolutely positioned window with a .bevel-out frame and a .titlebar; `_` collapses
// to the title bar, the square toggles a 1.5x zoom of the body (CSS transform), `x` calls onClose or shakes the window
// 3 x 4 px. Title bar drag with pointer capture, clamped to the desktop; position persisted to localStorage
// "fly.win.<id>" ({x,y}); the focused window takes the highest z-index ("fly.win.z" counter). Below 1280 px the
// desktop stacks the windows vertically (drag disabled). A `fly.win.focus` CustomEvent (taskbar) brings a window up.
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

export interface WinRect { x: number; y: number; w: number; h: number }
export interface Win95WindowProps {
  id: string;
  title: string;
  icon?: ReactNode;
  initial: WinRect;
  menu?: ReactNode;
  statusBar?: ReactNode;
  collapsible?: boolean;
  onClose?: () => void;
  children: ReactNode;
}

export const STACK_BREAKPOINT = 1280;
export const WIN_FOCUS_EVENT = "fly.win.focus";
export const WIN_SHAKE_EVENT = "fly.win.shake";
const Z_KEY = "fly.win.z";
const TITLE_H = 20;

// ------------------------------------------------------------------------------------------- localStorage (guarded)

export function lsGet(key: string): string | null {
  try { return typeof window === "undefined" ? null : window.localStorage.getItem(key); } catch { return null; }
}
export function lsSet(key: string, value: string): void {
  try { if (typeof window !== "undefined") window.localStorage.setItem(key, value); } catch { /* private mode */ }
}

let zCounter = 0;
function nextZ(): number {
  if (zCounter === 0) zCounter = Math.max(10, Number(lsGet(Z_KEY)) || 10);
  zCounter += 1;
  lsSet(Z_KEY, String(zCounter));
  return zCounter;
}

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
  window.addEventListener("storage", cb);
  return () => window.removeEventListener("storage", cb);
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
    } catch { /* corrupt value: keep initial */ }
    return null;
  }, [raw]);
}

/** Ask a window to come to the front (and un-collapse); used by the taskbar buttons. */
export function focusWindow(id: string): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(WIN_FOCUS_EVENT, { detail: { id } }));
}

/** Shake a window (File > Exit: the retro "no, you cannot leave" gesture). */
export function shakeWindow(id: string): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(WIN_SHAKE_EVENT, { detail: { id } }));
}

// ------------------------------------------------------------------------------------------- window

export default function Win95Window({ id, title, icon, initial, menu, statusBar, collapsible = true, onClose, children }: Win95WindowProps) {
  const stacked = useStacked();
  const stored = useStoredPos(id);
  const [dragPos, setPos] = useState<{ x: number; y: number } | null>(null);
  const pos = dragPos ?? stored ?? { x: initial.x, y: initial.y };
  const [z, setZ] = useState(10);
  const [collapsed, setCollapsed] = useState(false);
  const [zoom, setZoom] = useState(false);
  const [shaking, setShaking] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ dx: number; dy: number; pointerId: number } | null>(null);

  const bringToFront = useCallback(() => { setZ(nextZ()); }, []);

  useEffect(() => {
    const onFocus = (ev: Event) => {
      const d = (ev as CustomEvent<{ id?: string }>).detail;
      if (d && d.id === id) {
        setCollapsed(false);
        bringToFront();
        rootRef.current?.scrollIntoView({ block: "nearest" });
      }
    };
    const onShake = (ev: Event) => {
      const d = (ev as CustomEvent<{ id?: string }>).detail;
      if (d && d.id === id) setShaking(true);
    };
    window.addEventListener(WIN_FOCUS_EVENT, onFocus);
    window.addEventListener(WIN_SHAKE_EVENT, onShake);
    return () => {
      window.removeEventListener(WIN_FOCUS_EVENT, onFocus);
      window.removeEventListener(WIN_SHAKE_EVENT, onShake);
    };
  }, [id, bringToFront]);

  const clampToDesktop = useCallback((x: number, y: number): { x: number; y: number } => {
    const el = rootRef.current;
    const parent = el?.parentElement;
    if (!el || !parent) return { x: Math.max(0, x), y: Math.max(0, y) };
    const pw = parent.clientWidth, ph = parent.clientHeight;
    const w = el.offsetWidth, h = Math.max(TITLE_H, el.offsetHeight);
    return {
      x: Math.min(Math.max(0, x), Math.max(0, pw - w)),
      y: Math.min(Math.max(0, y), Math.max(0, ph - h)),
    };
  }, []);

  const onTitlePointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    bringToFront();
    if (stacked) return;
    if ((e.target as HTMLElement).closest("button")) return;
    if (e.button !== 0) return;
    const el = rootRef.current;
    if (!el) return;
    drag.current = { dx: e.clientX - el.offsetLeft, dy: e.clientY - el.offsetTop, pointerId: e.pointerId };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    e.preventDefault();
  }, [bringToFront, stacked]);

  const onTitlePointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.pointerId !== e.pointerId) return;
    setPos(clampToDesktop(e.clientX - d.dx, e.clientY - d.dy));
  }, [clampToDesktop]);

  const onTitlePointerUp = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.pointerId !== e.pointerId) return;
    drag.current = null;
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    setPos((p) => {
      if (p) lsSet("fly.win." + id, JSON.stringify({ x: Math.round(p.x), y: Math.round(p.y) }));
      return p;
    });
  }, [id]);

  const onCloseClick = useCallback(() => {
    if (onClose) { onClose(); return; }
    setShaking(true);
  }, [onClose]);

  const style: CSSProperties = stacked
    ? { position: "relative", width: "100%", height: collapsed ? "auto" : zoom ? initial.h * 1.5 : initial.h, zIndex: 1 }
    : {
        position: "absolute", left: pos.x, top: pos.y,
        width: zoom ? Math.round(initial.w * 1.5) : initial.w,
        height: collapsed ? "auto" : zoom ? Math.round(initial.h * 1.5) : initial.h,
        zIndex: z,
      };

  const bodyInner: CSSProperties = zoom
    ? { transform: "scale(1.5)", transformOrigin: "0 0", width: `${100 / 1.5}%`, height: `${100 / 1.5}%` }
    : { width: "100%", height: "100%" };

  return (
    <div
      ref={rootRef}
      className={"bevel-out flex flex-col select-none " + (shaking ? "win-shake" : "")}
      style={style}
      data-window={id}
      onPointerDownCapture={bringToFront}
      onAnimationEnd={() => setShaking(false)}
    >
      <div
        className="titlebar flex h-[20px] items-center gap-1 text-[12px]"
        style={{ cursor: stacked ? "default" : "move", padding: "0 2px 0 4px" }}
        onPointerDown={onTitlePointerDown}
        onPointerMove={onTitlePointerMove}
        onPointerUp={onTitlePointerUp}
        onPointerCancel={onTitlePointerUp}
        onDoubleClick={() => { if (collapsible) setCollapsed((c) => !c); }}
      >
        {icon ? <span className="flex h-4 w-4 items-center justify-center" aria-hidden>{icon}</span> : null}
        <span className="flex-1 truncate">{title}</span>
        <button type="button" className="btn95 h-[14px] w-4 text-[10px] leading-none font-bold" title="Minimize"
          style={{ padding: 0 }} disabled={!collapsible} onClick={() => setCollapsed((c) => !c)} aria-label="collapse">
          <span className="relative top-[-3px]">_</span>
        </button>
        <button type="button" className="btn95 h-[14px] w-4 text-[10px] leading-none font-bold" title="Maximize (zoom 1.5x)"
          style={{ padding: 0 }} onClick={() => setZoom((v) => !v)} aria-pressed={zoom} aria-label="zoom">
          <span className="relative top-[-1px]">&#9633;</span>
        </button>
        <button type="button" className="btn95 ml-[2px] h-[14px] w-4 text-[10px] leading-none font-bold" title="Close"
          style={{ padding: 0 }} onClick={onCloseClick} aria-label="close">
          <span className="relative top-[-1px]">&times;</span>
        </button>
      </div>
      {menu ? <div hidden={collapsed}>{menu}</div> : null}
      <div className="relative min-h-0 flex-1 overflow-hidden bg-win-gray" hidden={collapsed}>
        <div style={bodyInner} className="overflow-hidden">{children}</div>
      </div>
      {statusBar ? <div hidden={collapsed}>{statusBar}</div> : null}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- message box

export interface MessageBoxButton { label: string; onClick: () => void; default?: boolean }
export interface MessageBoxProps { title: string; children: ReactNode; buttons: MessageBoxButton[]; icon?: ReactNode; width?: number }

/** Modal Win95 message box (About dialog, trail-clear confirmation). Esc triggers the last button. */
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
      <div className="bevel-out flex flex-col" style={{ width }}>
        <div className="titlebar flex h-[20px] items-center text-[12px]" style={{ padding: "0 4px" }}>
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
