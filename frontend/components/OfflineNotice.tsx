"use client";
// First-impression gap #1: when the WebSocket cannot connect the desktop used to sit there in silence - the only hint
// was a 12 px "brain offline" line drawn inside the canvas. This is the Win95 message box that says it out loud.
//
// Rules it has to keep:
//   * it appears only after the socket has been away for OFFLINE_GRACE_MS (~4 s), so a normal page load and the
//     sub-second blips of a reconnect never flash a scary box;
//   * it disappears the moment `sock.status` is "open" again (and re-arms itself for the next outage);
//   * it is dismissible, non-modal (the desktop stays usable) and it never covers the taskbar - the wrapper reserves
//     TASKBAR_H + a margin at the bottom and is pointer-events:none, so only the box itself takes clicks;
//   * it invents NOTHING. No fake tick, no fake price: it reports `status` and `attempt` straight from lib/ws.ts and
//     points a visitor who arrived mid-outage at the X account and the source, which are up even when the brain is not.
import { useCallback, useEffect, useState } from "react";
import { GITHUB_URL, X_HANDLE, X_URL } from "@/lib/brand";
import { TASKBAR_H } from "@/lib/layout";
import type { FlySocket, WsStatus } from "@/lib/ws";

/** How long the socket has to be away before the notice appears (SPEC-free UX number: ~4 s). */
export const OFFLINE_GRACE_MS = 4000;

export interface OfflineNoticeProps {
  sock: FlySocket;
  /** Test/preview hook: shorten the grace period. */
  graceMs?: number;
}

/** One plain-English line for a non-open socket. `attempt` comes from the reconnect backoff in lib/ws.ts. */
export function offlineLine(status: WsStatus, attempt: number): string {
  if (status === "connecting") return "Still opening the first connection to the brain.";
  if (status === "reconnecting") return `Trying again automatically - reconnect attempt ${attempt}.`;
  return "The connection is closed. Reload the page to start a new one.";
}

/** The Win95 "information" icon: blue disc, white lowercase i. */
function InfoIcon({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="14" fill="#000080" stroke="#000" />
      <circle cx="16" cy="16" r="11" fill="#1084d0" />
      <rect x="14" y="7" width="4" height="4" fill="#fff" />
      <rect x="14" y="13" width="4" height="12" fill="#fff" />
      <rect x="11" y="22" width="10" height="3" fill="#fff" />
      <rect x="11" y="13" width="3" height="3" fill="#fff" />
    </svg>
  );
}

export default function OfflineNotice({ sock, graceMs = OFFLINE_GRACE_MS }: OfflineNoticeProps) {
  const { status, attempt } = sock;
  const [armed, setArmed] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  // Arm on a timer, disarm on "open". Every setState below runs from a timer / rAF callback (never synchronously in
  // the effect body) - the same rule the rest of the codebase follows for react-hooks/set-state-in-effect.
  useEffect(() => {
    if (status === "open") {
      const raf = requestAnimationFrame(() => { setArmed(false); setDismissed(false); });
      return () => cancelAnimationFrame(raf);
    }
    const t = setTimeout(() => setArmed(true), Math.max(0, graceMs));
    return () => clearTimeout(t);
  }, [status, graceMs]);

  const close = useCallback(() => setDismissed(true), []);
  const reload = useCallback(() => { try { window.location.reload(); } catch { /* ignore */ } }, []);

  if (status === "open" || !armed || dismissed) return null;

  return (
    <div
      className="pointer-events-none fixed inset-0 z-[99998] flex items-center justify-center p-3"
      style={{ paddingBottom: TASKBAR_H + 16 }}
      aria-live="polite"
      data-testid="offline-notice"
    >
      <div
        className="bevel-out pointer-events-auto flex w-full max-w-[460px] flex-col"
        style={{ boxShadow: "2px 2px 0 #000" }}
        role="dialog"
        aria-label="The brain is not answering"
      >
        <div className="titlebar flex h-[22px] items-center text-[13px]" style={{ padding: "0 2px 0 4px" }}>
          <span className="flex-1 truncate">FlyBrain - no connection</span>
          <button type="button" className="btn95 h-[16px] w-[18px] text-[11px] font-bold leading-none" style={{ padding: 0 }}
            onClick={close} title="Close" aria-label="close">
            <span className="relative top-[-1px]">&times;</span>
          </button>
        </div>

        <div className="flex gap-3 p-3 text-[12px]">
          <div className="shrink-0" aria-hidden><InfoIcon /></div>
          <div className="min-w-0 flex-1">
            <div className="mb-1 font-bold">The brain is not answering right now.</div>
            <div className="mb-1">
              SynapseFly&apos;s brain runs on a single machine. It is either restarting or down, so there are no ticks:
              the canvas, the oscilloscope and the ticker have nothing to draw. Nothing on this page is faked while it
              is quiet - what you see is a stopped simulation, not a replay.
            </div>
            <div className="mb-2" data-testid="offline-attempt">{offlineLine(status, attempt)}</div>
            <div className="bevel-in bg-win-gray px-2 py-1 text-[11px]">
              <div className="mb-1 text-win-dark">The project is still here while the brain is out:</div>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <a className="text-win-blue" href={X_URL} target="_blank" rel="noopener noreferrer">X {X_HANDLE}</a>
                <a className="text-win-blue" href={GITHUB_URL} target="_blank" rel="noopener noreferrer">GitHub (source)</a>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center gap-2 px-3 pb-3">
          <button type="button" className="btn95 h-[23px] min-w-[75px]" onClick={close}
            style={{ outline: "1px solid #000", outlineOffset: -3 }}>
            OK
          </button>
          <button type="button" className="btn95 h-[23px] min-w-[75px]" onClick={reload} title="Reload the page and open a new connection">
            Reload
          </button>
        </div>
      </div>
    </div>
  );
}
