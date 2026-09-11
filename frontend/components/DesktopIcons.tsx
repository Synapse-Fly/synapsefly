"use client";
// The Windows 95 desktop icon grid - the thing the visitor actually clicks to open an app.
//
// Behaviour is the real Win95 desktop, with one deliberate deviation the user asked for: a SINGLE click opens.
//   * icons flow top-to-bottom down the left edge and wrap into a second column when the desktop is too short
//     (a column-direction flex container with wrap, exactly how Explorer arranged them);
//   * clicking an icon selects it (navy label, 50 % navy-tinted art) and immediately calls onOpen. A double click
//     cannot open twice: `openLock` swallows a repeat of the same id inside OPEN_LOCK_MS, and for a link it also
//     preventDefault()s the second click so the browser does not open a second tab;
//   * the grid is a roving-tabindex listbox: arrows move the selection, Home/End jump, Enter/Space open, and the
//     focused cell shows the Win95 dotted focus rectangle;
//   * "link" items are real anchors, so the browser shows the target URL on hover and middle-click / ctrl-click
//     behave normally - onOpen still fires so the shell can log it;
//   * a pointerdown anywhere outside the grid clears the selection, like clicking bare desktop.
//
// Pure presentation: no sockets, no window rendering, no layout math beyond its own cells. The parent is responsible
// for giving it a box that already excludes the taskbar (it fills 100 % of its parent's height and never scrolls).
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type MouseEvent as ReactMouseEvent, type ReactElement } from "react";
import { AppIcon, type IconName } from "@/lib/icons";

export interface DesktopItem {
  /** Window id ("paint" | "raster" | "status" | "brain" | "tweets" | "readme") or a link id. */
  id: string;
  label: string;
  icon: IconName;
  kind: "window" | "link";
  /** Required when kind is "link". */
  href?: string;
  /** Tooltip; falls back to the label. */
  title?: string;
}

export interface DesktopIconsProps {
  items: readonly DesktopItem[];
  /** Windows currently open - their icon gets the subtle "open" treatment. */
  openIds: readonly string[];
  /** Single click, tap, or Enter/Space on the focused icon. */
  onOpen: (item: DesktopItem) => void;
  className?: string;
}

/** Cell geometry. Big, as asked: 48 px of art in an 80 x 88 cell, not the cramped 32 px original. */
export const DESKTOP_CELL_W = 80;
export const DESKTOP_CELL_H = 88;
export const DESKTOP_ICON_SIZE = 48;
/** Padding inside the grid, so the first icon is not welded to the screen corner. */
const PAD = 8;
/** Label line box; two of these is the label's hard ceiling. */
const LINE_H = 13;
/** A second click on the same id inside this window is the tail of a double click, not a second open. */
const OPEN_LOCK_MS = 400;

/** Hard 1 px black outline so white labels survive the teal desktop. */
const LABEL_SHADOW = "1px 1px 0 #000, -1px 1px 0 #000, 1px -1px 0 #000, -1px -1px 0 #000";

function clamp(n: number, lo: number, hi: number): number {
  return n < lo ? lo : n > hi ? hi : n;
}

export default function DesktopIcons({ items, openIds, onOpen, className }: DesktopIconsProps): ReactElement {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const cellRefs = useRef<Array<HTMLElement | null>>([]);
  const openLock = useRef<Map<string, number>>(new Map());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [rows, setRows] = useState(6);

  // Unique, CSS-url-safe id for the selection tint filter (useId emits colons, which url(#...) will not take).
  const tintId = `win95-tint-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;

  // How many cells fit in one column, so ArrowLeft / ArrowRight can step a whole column.
  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const usable = el.clientHeight - PAD * 2;
      setRows(Math.max(1, Math.floor(usable / DESKTOP_CELL_H)));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Bare desktop (or any window) clicked: drop the selection.
  useEffect(() => {
    const onDown = (e: PointerEvent) => {
      const root = rootRef.current;
      if (root && e.target instanceof Node && root.contains(e.target)) return;
      setSelectedId(null);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, []);

  const selectedIndex = items.findIndex((it) => it.id === selectedId);
  const activeIndex = selectedIndex >= 0 ? selectedIndex : 0;

  const activate = useCallback((e: ReactMouseEvent, item: DesktopItem) => {
    setSelectedId(item.id);
    const now = Date.now();
    const last = openLock.current.get(item.id) ?? 0;
    if (now - last < OPEN_LOCK_MS) {
      // Tail of a double click. Never open twice, and never let a link spawn a second tab.
      if (item.kind === "link") e.preventDefault();
      return;
    }
    openLock.current.set(item.id, now);
    onOpen(item);
  }, [onOpen]);

  const focusIndex = useCallback((next: number) => {
    if (items.length === 0) return;
    const i = clamp(next, 0, items.length - 1);
    setSelectedId(items[i].id);
    cellRefs.current[i]?.focus();
  }, [items]);

  const onKeyDown = useCallback((e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (items.length === 0) return;
    switch (e.key) {
      case "ArrowDown": e.preventDefault(); focusIndex(activeIndex + 1); return;
      case "ArrowUp": e.preventDefault(); focusIndex(activeIndex - 1); return;
      case "ArrowRight": e.preventDefault(); focusIndex(activeIndex + rows); return;
      case "ArrowLeft": e.preventDefault(); focusIndex(activeIndex - rows); return;
      case "Home": e.preventDefault(); focusIndex(0); return;
      case "End": e.preventDefault(); focusIndex(items.length - 1); return;
      case " ":
      case "Spacebar": {
        // A <button> opens on Space by itself; an <a> would scroll the page instead, so drive its click.
        if (items[activeIndex]?.kind === "link") {
          e.preventDefault();
          cellRefs.current[activeIndex]?.click();
        }
        return;
      }
      default:
    }
  }, [activeIndex, focusIndex, items, rows]);

  return (
    <div
      ref={rootRef}
      role="listbox"
      aria-label="Desktop icons"
      aria-orientation="vertical"
      tabIndex={-1}
      onKeyDown={onKeyDown}
      className={`flex h-full w-max max-w-full flex-col flex-wrap content-start items-start overflow-hidden select-none ${className ?? ""}`}
      style={{ padding: PAD, cursor: "default" }}
    >
      {/* Exact Win95 selection blend: every channel halved, blue lifted to navy. Alpha is untouched, so the tint
          follows the art instead of painting a rectangle. */}
      <svg width="0" height="0" aria-hidden="true" focusable="false" style={{ position: "absolute" }}>
        <filter id={tintId} colorInterpolationFilters="sRGB">
          <feColorMatrix
            type="matrix"
            values="0.5 0 0 0 0  0 0.5 0 0 0  0 0 0.5 0 0.25  0 0 0 1 0"
          />
        </filter>
      </svg>

      {items.map((item, i) => {
        const selected = item.id === selectedId;
        const open = openIds.includes(item.id);
        const common = {
          ref: (el: HTMLElement | null) => { cellRefs.current[i] = el; },
          role: "option",
          "aria-selected": selected,
          tabIndex: i === activeIndex ? 0 : -1,
          title: item.title ?? item.label,
          draggable: false,
          onClick: (e: ReactMouseEvent) => activate(e, item),
          // No hover state on purpose: Win95 desktop icons had none, and nothing here may be hover-only.
          className: "group flex shrink-0 flex-col items-center justify-start bg-transparent p-0 text-inherit no-underline outline-none",
          style: {
            width: DESKTOP_CELL_W,
            height: DESKTOP_CELL_H,
            cursor: "default",
            border: 0,
            font: "inherit",
          } as const,
        };

        const inner = (
          <>
            {/* 48 px of art in a 56 px box - the extra 8 px carries the "open" bevel without nudging the art. */}
            <span
              className="w95-art flex items-center justify-center"
              style={{
                width: DESKTOP_ICON_SIZE + 8,
                height: DESKTOP_ICON_SIZE + 8,
                borderWidth: 1,
                borderStyle: "solid",
                borderColor: open ? "#808080 #ffffff #ffffff #808080" : "transparent",
                background: open ? "rgba(255,255,255,0.14)" : "transparent",
                filter: selected ? `url(#${tintId})` : undefined,
              }}
            >
              <AppIcon name={item.icon} size={DESKTOP_ICON_SIZE} />
            </span>
            <span
              className="mt-[3px] inline-block text-center break-words group-focus-visible:outline-1 group-focus-visible:outline-dotted group-focus-visible:outline-white"
              style={{
                maxWidth: DESKTOP_CELL_W - 6,
                maxHeight: LINE_H * 2,
                overflow: "hidden",
                fontSize: 11,
                lineHeight: `${LINE_H}px`,
                padding: "0 2px",
                color: "#ffffff",
                background: selected ? "#000080" : "transparent",
                textShadow: selected ? "none" : LABEL_SHADOW,
                outlineOffset: -1,
              }}
            >
              {item.label}
            </span>
          </>
        );

        return item.kind === "link" && item.href ? (
          <a key={item.id} {...common} href={item.href} target="_blank" rel="noopener noreferrer">
            {inner}
          </a>
        ) : (
          <button key={item.id} {...common} type="button">
            {inner}
          </button>
        );
      })}
    </div>
  );
}
