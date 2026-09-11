"use client";
// The Windows 95 Start button and its menu.
//
// Why it exists: the visitor could not find the project's X account, because it was a 16 px button in the far
// bottom-right corner of the taskbar. Everything the desktop can do is now reachable from the bottom-LEFT corner
// every Windows user has clicked since 1995 - the apps, the links, the explainer, the About box.
//
// Behaviour is the real thing: the button toggles the menu, the menu closes on Escape, on a click anywhere outside
// it and on choosing an item; arrows move, Right/Left open and close a flyout, Enter activates. The navy strip with
// the rotated product name is the classic sidebar.
//
// "Shut Down..." is a joke and nothing more: it opens a dialog, the dialog says no, and the simulation never stops.
// Nothing in this file touches the socket, the canvas or the window state beyond the callbacks it is handed.
import { useCallback, useEffect, useRef, useState, type ReactElement } from "react";
import { AppIcon, type IconName } from "@/lib/icons";
import { MessageBox } from "./Win95Window";
import { GITHUB_URL, PROJECT, X_HANDLE, X_URL } from "@/lib/brand";
import { TASKBAR_H } from "@/lib/layout";

export interface StartProgram {
  /** Window id handed straight back to onOpen. */
  id: string;
  label: string;
  icon: IconName;
}

export interface StartMenuProps {
  /** Programs submenu, in order. */
  programs: readonly StartProgram[];
  /** A program row was chosen: open that window. */
  onOpen: (id: string) => void;
  /** Help > What is this? */
  onIntro: () => void;
  /** Help > About FlyBrain */
  onAbout: () => void;
  /** Help > Tile windows (the same command as View > Tile windows and key G). */
  onTile: () => void;
  /** synapsefly.com - the site link (brand.ts owns every other URL). */
  siteUrl: string;
}

type SubId = "programs" | "links" | "help";

/** Rows inside one list, in DOM order; used by the arrow keys. */
function rowsOf(el: Element | null): HTMLElement[] {
  const list = el?.closest("[data-list]");
  if (!list) return [];
  return Array.from(list.querySelectorAll<HTMLElement>(":scope > .start-item > [data-row], :scope > [data-row]"));
}

function Row(props: {
  icon?: IconName;
  iconShortcut?: boolean;
  label: string;
  title?: string;
  /** This row owns a flyout with this id (the arrow keys read it back off the DOM). */
  subFor?: SubId;
  expanded?: boolean;
  href?: string;
  onClick?: () => void;
  onEnter?: () => void;
}): ReactElement {
  const inner = (
    <>
      <span className="flex h-4 w-4 shrink-0 items-center justify-center" aria-hidden>
        {props.icon ? <AppIcon name={props.icon} size={16} shortcut={props.iconShortcut} /> : null}
      </span>
      <span className="start-label">{props.label}</span>
      {props.subFor ? <span className="start-arrow" aria-hidden>&#9654;</span> : null}
    </>
  );
  const common = {
    className: "start-row",
    "data-row": "true",
    "data-sub-for": props.subFor,
    title: props.title ?? props.label,
    onMouseEnter: props.onEnter,
  };
  if (props.href) {
    return (
      <a {...common} href={props.href} target="_blank" rel="noopener noreferrer" onClick={props.onClick}>
        {inner}
      </a>
    );
  }
  return (
    <button
      {...common}
      type="button"
      aria-haspopup={props.subFor ? "menu" : undefined}
      aria-expanded={props.subFor ? !!props.expanded : undefined}
      onClick={props.onClick}
    >
      {inner}
    </button>
  );
}

export default function StartMenu({ programs, onOpen, onIntro, onAbout, onTile, siteUrl }: StartMenuProps): ReactElement {
  const [open, setOpen] = useState(false);
  const [sub, setSub] = useState<SubId | null>(null);
  const [shutdown, setShutdown] = useState<0 | 1 | 2>(0);
  const btnRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  const close = useCallback((refocus = false) => {
    setOpen(false);
    setSub(null);
    if (refocus) btnRef.current?.focus();
  }, []);

  // Outside click (anywhere on the desktop, a window or the taskbar) closes the menu, exactly like Windows.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target;
      if (!(t instanceof Node)) return;
      if (panelRef.current?.contains(t) || btnRef.current?.contains(t)) return;
      setOpen(false);
      setSub(null);
    };
    document.addEventListener("pointerdown", onDown, true);
    return () => document.removeEventListener("pointerdown", onDown, true);
  }, [open]);

  // Focus the first row the moment the panel is in the DOM. This is a ref callback and not an effect + rAF on
  // purpose: it has to happen in the commit that mounts the panel, whatever the browser is doing with frames.
  const setPanel = useCallback((node: HTMLDivElement | null) => {
    panelRef.current = node;
    if (node) node.querySelector<HTMLElement>("[data-row]")?.focus();
  }, []);

  const choose = useCallback((fn: () => void) => {
    close();
    fn();
  }, [close]);

  const handleKey = useCallback((e: KeyboardEvent) => {
    let active = document.activeElement as HTMLElement | null;
    if (!active || !panelRef.current || !panelRef.current.contains(active)) {
      // Focus is outside (the Start button, or nowhere): the arrows start at the first row.
      active = panelRef.current?.querySelector<HTMLElement>("[data-row]") ?? null;
      if (active && (e.key === "ArrowDown" || e.key === "ArrowUp")) { e.preventDefault(); active.focus(); return; }
    }
    switch (e.key) {
      case "Escape":
        e.preventDefault();
        if (sub) { setSub(null); panelRef.current?.querySelector<HTMLElement>(`[data-sub-for="${sub}"]`)?.focus(); return; }
        close(true);
        return;
      case "ArrowDown":
      case "ArrowUp": {
        e.preventDefault();
        const rows = rowsOf(active);
        const i = rows.indexOf(active as HTMLElement);
        if (rows.length === 0) return;
        const next = e.key === "ArrowDown" ? (i + 1) % rows.length : (i - 1 + rows.length) % rows.length;
        rows[next]?.focus();
        return;
      }
      case "ArrowRight": {
        const id = active?.getAttribute("data-sub-for") as SubId | null;
        if (!id) return;
        e.preventDefault();
        setSub(id);
        // The flyout mounts on the next render; a timer (not rAF) so it still lands in a background tab.
        window.setTimeout(() => {
          panelRef.current?.querySelector<HTMLElement>(`[data-list="${id}"] [data-row]`)?.focus();
        }, 0);
        return;
      }
      case "ArrowLeft": {
        if (!sub) return;
        e.preventDefault();
        setSub(null);
        panelRef.current?.querySelector<HTMLElement>(`[data-sub-for="${sub}"]`)?.focus();
        return;
      }
      case "Enter":
      case " ": {
        // Activate the row ourselves rather than leaning on the browser's synthesized click: Space on an anchor
        // would scroll the page instead of following the link, and the two elements behave differently on Enter.
        if (active && panelRef.current?.contains(active)) { e.preventDefault(); active.click(); }
        return;
      }
      default:
    }
  }, [close, sub]);

  // Keys are handled on the DOCUMENT while the menu is open, not on the panel: focus can sit on the Start button
  // (which is not inside the panel) or be lost entirely, and Escape still has to close the menu.
  useEffect(() => {
    if (!open) return;
    document.addEventListener("keydown", handleKey, true);
    return () => document.removeEventListener("keydown", handleKey, true);
  }, [open, handleKey]);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className="btn95 start-btn"
        aria-expanded={open}
        aria-haspopup="menu"
        title="Start - programs, links and help"
        onClick={() => { setSub(null); setOpen((o) => !o); }}
      >
        <AppIcon name="status" size={16} />
        <span>Start</span>
      </button>

      {open ? (
        <div
          ref={setPanel}
          className="start-menu"
          style={{ bottom: TASKBAR_H }}
          role="menu"
          aria-label="Start"
          data-menu-open="start"
          data-testid="start-menu"
        >
          <div className="start-strip" aria-hidden>
            <span>Synapse<b>Fly</b></span>
          </div>

          <div className="start-items" data-list="root">
            <div className="start-item">
              <Row icon="paint" label="Programs" subFor="programs" expanded={sub === "programs"}
                onEnter={() => setSub("programs")} onClick={() => setSub("programs")} />
              {sub === "programs" ? (
                <div className="start-sub" data-list="programs" role="menu" aria-label="Programs">
                  {programs.map((p) => (
                    <Row key={p.id} icon={p.icon} label={p.label} onClick={() => choose(() => onOpen(p.id))} />
                  ))}
                </div>
              ) : null}
            </div>

            <div className="start-item">
              <Row icon="github" iconShortcut={false} label="Links" subFor="links" expanded={sub === "links"}
                onEnter={() => setSub("links")} onClick={() => setSub("links")} />
              {sub === "links" ? (
                <div className="start-sub" data-list="links" role="menu" aria-label="Links">
                  <Row icon="x" label={`Follow ${X_HANDLE}`} title={`${PROJECT} on X (${X_HANDLE})`} href={X_URL} onClick={() => close()} />
                  <Row icon="github" label="GitHub (source)" title="Synapse-Fly/synapsefly on GitHub" href={GITHUB_URL} onClick={() => close()} />
                  <Row icon="readme" label="synapsefly.com" title="The live site" href={siteUrl} onClick={() => close()} />
                </div>
              ) : null}
            </div>

            <div className="start-item">
              <Row icon="readme" label="Help" subFor="help" expanded={sub === "help"}
                onEnter={() => setSub("help")} onClick={() => setSub("help")} />
              {sub === "help" ? (
                <div className="start-sub" data-list="help" role="menu" aria-label="Help">
                  <Row icon="readme" label="What is this?" onClick={() => choose(onIntro)} />
                  <Row icon="oscilloscope" label="Tile windows" title="Fit every open window on the screen (G)" onClick={() => choose(onTile)} />
                  <Row icon="status" label="About FlyBrain..." onClick={() => choose(onAbout)} />
                </div>
              ) : null}
            </div>

            <div className="start-sep" role="separator" />

            <div className="start-item">
              <Row icon="bin" label="Shut Down..." title="It will not work." onEnter={() => setSub(null)}
                onClick={() => { close(); setShutdown(1); }} />
            </div>
          </div>
        </div>
      ) : null}

      {/* The joke. It opens a dialog, the dialog declines, and nothing anywhere stops running. */}
      {shutdown === 1 ? (
        <MessageBox
          title="Shut Down Windows"
          icon={<AppIcon name="status" size={32} />}
          width={430}
          buttons={[
            { label: "Yes", default: true, onClick: () => setShutdown(2) },
            { label: "Cancel", onClick: () => setShutdown(0) },
          ]}
        >
          <div className="font-bold">Are you sure you want to shut down {PROJECT}?</div>
          <div className="mt-2">The fly was not consulted.</div>
        </MessageBox>
      ) : null}
      {shutdown === 2 ? (
        <MessageBox
          title="Shut Down Windows"
          icon={<AppIcon name="status" size={32} />}
          width={430}
          buttons={[{ label: "OK", default: true, onClick: () => setShutdown(0) }]}
        >
          <div className="font-bold">No.</div>
          <div className="mt-2">
            The brain runs on a machine that is not yours, the canvas has no undo, and this button was never wired to
            anything. It is now safe to turn off your monitor - the fly will keep painting without you.
          </div>
        </MessageBox>
      ) : null}
      {shutdown !== 0 ? <span hidden data-menu-open="shutdown" /> : null}
    </>
  );
}
