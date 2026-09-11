"use client";
// SPEC section e.7 `page.tsx`: creates the socket, the canvas handle, the UI flags, the MenuAction dispatcher, the
// document-level keyboard shortcuts and the snapshot service (section e.6); renders the Win95 desktop with the four
// windows (Paint, Oscilloscope, Fly Status, tweets.txt) and a taskbar with a clock.
import { useCallback, useEffect, useRef, useState } from "react";
import { useFlySocket } from "@/lib/ws";
import { postMarketMode, postSnapshot, postTweetTest } from "@/lib/api";
import { postPoke } from "@/lib/api";
import type { ClientMsg, MarketMode, PokeStim } from "@/lib/types";
import PaintWindow from "@/components/PaintWindow";
import Win95Window, { MessageBox, focusWindow, shakeWindow, useStacked } from "@/components/Win95Window";
import { AboutDialog, PaintIcon, type MenuAction, type MenuFlags } from "@/components/MenuBar";
import type { FlyCanvasHandle } from "@/components/FlyCanvas";
import SpikeRaster from "@/components/SpikeRaster";
import MoodPanel from "@/components/MoodPanel";
import MarketTicker from "@/components/MarketTicker";
import TweetNotepad from "@/components/TweetNotepad";

type Dialog = null | { kind: "about" } | { kind: "confirm_clear"; run_id: string };

const KEY_POKES: Record<string, PokeStim> = { S: "sugar", B: "bitter", L: "loom", W: "water", D: "dust", C: "pheromone", Z: "sleep", R: "reward", P: "punish" };
const KEY_REGIMES: Record<string, MarketMode> = { "1": "CALM", "2": "PUMP", "3": "DUMP", "4": "CHOP", "5": "RUG", "6": "DEAD", "0": "sim" };

const WINDOWS = [
  { id: "paint", label: "untitled - Paint" },
  { id: "raster", label: "Oscilloscope" },
  { id: "status", label: "Fly Status" },
  { id: "tweets", label: "tweets.txt - Notepad" },
];

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
  return <div className="bevel-in flex h-[22px] items-center bg-win-gray px-2" suppressHydrationWarning>{now}</div>;
}

function OscilloscopeIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" shapeRendering="crispEdges" aria-hidden>
      <rect x="1" y="2" width="14" height="11" fill="#000" stroke="#c0c0c0" />
      <path d="M2 9 L4 9 L5 4 L6 12 L7 9 L10 9 L11 6 L12 9 L14 9" fill="none" stroke="#2ecc40" />
    </svg>
  );
}
function StatusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" shapeRendering="crispEdges" aria-hidden>
      <rect x="2" y="2" width="12" height="12" fill="#ffd700" stroke="#000" />
      <rect x="5" y="5" width="2" height="2" fill="#000" /><rect x="9" y="5" width="2" height="2" fill="#000" />
      <rect x="5" y="10" width="6" height="1" fill="#000" />
    </svg>
  );
}
function NotepadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" shapeRendering="crispEdges" aria-hidden>
      <rect x="3" y="1" width="10" height="14" fill="#fff" stroke="#000" />
      <rect x="5" y="4" width="6" height="1" fill="#000" /><rect x="5" y="7" width="6" height="1" fill="#000" /><rect x="5" y="10" width="4" height="1" fill="#000" />
    </svg>
  );
}

export default function Home() {
  const sock = useFlySocket();
  const stacked = useStacked();
  const canvasRef = useRef<FlyCanvasHandle | null>(null);
  const [flags, setFlags] = useState<MenuFlags>({ labels: true, frozen: false, zoom: false, fps: false, flip: false });
  const [dialog, setDialog] = useState<Dialog>(null);
  const dialogRef = useRef<Dialog>(null);
  useEffect(() => { dialogRef.current = dialog; }, [dialog]);
  const lastRunId = useRef<string | null>(null);
  const { send, on } = sock;

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
        case "exit":
          shakeWindow("paint");
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
      if (dialogRef.current) return;
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
  const desktopStyle = stacked ? undefined : { minHeight: 920, width: "100%" };

  return (
    <div className="min-h-screen pb-[34px]" data-testid="desktop">
      <div className={desktopClass} style={desktopStyle}>
        <PaintWindow sock={sock} onAction={dispatch} canvasRef={canvasRef} flags={flags} />
        <Win95Window id="raster" title="Oscilloscope - spike raster" icon={<OscilloscopeIcon />} initial={{ x: 912, y: 16, w: 560, h: 470 }}>
          <SpikeRaster sock={sock} frozen={flags.frozen} labels={flags.labels} />
        </Win95Window>
        <Win95Window id="status" title="Fly Status" icon={<StatusIcon />} initial={{ x: 912, y: 500, w: 560, h: 260 }}>
          <div className="grid h-full min-h-0 grid-cols-2 gap-[2px] overflow-hidden bg-win-gray">
            <div className="min-h-0 overflow-auto"><MoodPanel sock={sock} /></div>
            <div className="min-h-0 overflow-auto"><MarketTicker sock={sock} /></div>
          </div>
        </Win95Window>
        <Win95Window id="tweets" title="tweets.txt - Notepad" icon={<NotepadIcon />} initial={{ x: 16, y: 712, w: 880, h: 180 }} collapsible>
          <TweetNotepad sock={sock} onTest={() => dispatch({ kind: "tweet_test" })} />
        </Win95Window>
      </div>

      <div className="bevel-out fixed bottom-0 left-0 right-0 z-[99999] flex h-[30px] items-center gap-[3px] px-1" role="toolbar" aria-label="taskbar">
        <button type="button" className="btn95 flex h-[22px] items-center gap-1 font-bold" onClick={() => setDialog({ kind: "about" })} title="About FlyBrain">
          <PaintIcon /> FlyBrain
        </button>
        <div className="mx-1 h-[22px] w-[2px] border-l border-win-dark border-r-white" style={{ borderRightWidth: 1, borderRightStyle: "solid" }} />
        {WINDOWS.map((w) => (
          <button key={w.id} type="button" className="btn95 h-[22px] min-w-[120px] truncate text-left" onClick={() => focusWindow(w.id)} title={w.label}>
            {w.label}
          </button>
        ))}
        <div className="flex-1" />
        <div className="bevel-in flex h-[22px] items-center gap-1 bg-win-gray px-2 text-[11px]" title={`websocket ${sock.status}${sock.latencyMs !== null ? `, ${sock.latencyMs} ms` : ""}`}>
          <span className="inline-block h-[8px] w-[8px] rounded-full border border-black" style={{ background: sock.status === "open" ? "#00a800" : sock.status === "connecting" ? "#e0c000" : "#d00000" }} />
          {sock.status === "open" ? (sock.latencyMs !== null ? `${sock.latencyMs} ms` : "online") : sock.status === "reconnecting" ? `reconnecting (${sock.attempt})` : sock.status}
        </div>
        <Clock />
      </div>

      {dialog?.kind === "about" ? <AboutDialog hello={sock.hello} store={sock.store} onClose={() => setDialog(null)} /> : null}
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
