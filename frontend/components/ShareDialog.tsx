"use client";
// First-impression gap #2: the painting is the whole point, but there was no way to get it out of the browser. This
// is the Win95 "Share to X" dialog behind Paint's File > Share to X... and the Share button in the menu bar.
//
// What it does (and deliberately does not do):
//   * PREVIEW - one still frame of FlyCanvas.composite() (trail + overlay + the Paint caption bar), taken when the
//     dialog opens and re-taken by "Refresh". The still is encoded to a PNG blob ONCE and the preview <img> points at
//     that blob's object URL, so the picture on screen is the file byte for byte - not a second, later composite that
//     merely looks similar. The fly never stops painting, so "the picture you approved" has to be a held bitmap.
//   * SAVE - <a download> pointing at that same object URL (a data: URL when the browser has no toBlob), then a
//     delayed revoke. This is the only way the image can reach X: an intent link cannot carry a picture.
//   * OPEN X - window.open of https://x.com/intent/post with `text` and `url` both encodeURIComponent'd. The intent
//     page is a PRE-FILLED COMPOSER; nothing is ever posted from here, and no account is touched.
//   * The text is composed from a CURRENT tick only - the mood and the 5-minute change the tick actually carries -
//     and it says "(simulated feed)" whenever the market source is not a real DEX feed. If the socket is down or the
//     newest tick is older than SHARE_TICK_FRESH_MS the mood and the number are dropped rather than exported stale
//     (PaintWindow measures that and passes `live`). No number is invented, and the whole post is kept inside the
//     280-character budget (the link costs a fixed 23 through t.co).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { TOKEN, X_HANDLE, tokenLive } from "@/lib/brand";
import type { Mood, TickMsg } from "@/lib/types";

/** The site the post links to (the same constant MenuBar/page.tsx use; brand.ts owns everything else). */
export const SITE_URL = "https://www.synapsefly.com";
/** X's current web-intent endpoint (the older /intent/tweet form redirects here). */
export const X_INTENT_URL = "https://x.com/intent/post";
export const MAX_TWEET = 280;
/** Every link counts as 23 characters on X, whatever its real length. */
export const URL_WEIGHT = 23;
/** What is left for the text once the site link and the space before it are paid for. */
export const TEXT_BUDGET = MAX_TWEET - URL_WEIGHT - 1;
/** Ticks arrive at 4 Hz; anything older than this is not "now" any more and is not quoted in the post. */
export const SHARE_TICK_FRESH_MS = 5000;
/** An object URL is kept this long after it stops being the preview, so an in-flight download still resolves. */
const REVOKE_MS = 60_000;

export interface ShareFacts {
  mood: Mood | null;
  /** tick.market.chg_m5 - null when no tick has arrived (then no number is written at all). */
  chgM5: number | null;
  /** tick.market.source: only "dexscreener" is a real feed, everything else is the simulator. */
  source: string | null;
  /** hello.connectome.n - the SYNTHETIC stand-in's neuron count. */
  neurons: number | null;
  /** tick.market.token_live: false/absent while $SYNAPSE has not launched, in which case the tracked pair is not ours
   *  and the post must not attribute its move to the ticker. */
  tokenLive?: boolean;
  /** false when the brain is not sending ticks: the mood and the market number are then from BEFORE the outage, so
   *  they are dropped instead of going to X as if they were current. Defaults to true. */
  live?: boolean;
}

/** The deadpan default post. Facts only: the mood, the tick's own 5-minute change, the synthetic neuron count. */
export function composeShareText(f: ShareFacts): string {
  const live = f.live !== false;
  const parts: string[] = [];
  parts.push(live && f.mood ? `The fly is ${f.mood}.` : "The fly is painting.");
  // The ticker is only attached to a number once the real pair is the one being polled; until then the move belongs
  // to a stand-in feed and is posted without the ticker.
  if (live && f.chgM5 !== null && Number.isFinite(f.chgM5)) {
    const sign = f.chgM5 >= 0 ? "+" : "";
    if (tokenLive(f.tokenLive)) {
      const feed = f.source === "dexscreener" ? "" : " (simulated feed)";
      parts.push(`$${TOKEN} ${sign}${f.chgM5.toFixed(1)}% on 5m${feed}.`);
    } else {
      const feed = f.source === "dexscreener" ? "stand-in feed" : "simulated feed";
      parts.push(`Its ${feed} did ${sign}${f.chgM5.toFixed(1)}% on 5m.`);
    }
  }
  parts.push(f.neurons !== null && Number.isFinite(f.neurons)
    ? `${Math.round(f.neurons).toLocaleString("en-US")} synthetic neurons, one Paint canvas, no undo.`
    : "A synthetic fly brain, one Paint canvas, no undo.");
  parts.push(X_HANDLE);
  const text = parts.join(" ");
  return text.length <= TEXT_BUDGET ? text : `${text.slice(0, TEXT_BUDGET - 1).trimEnd()}...`;
}

/** The intent URL. Both parameters are encoded; nothing is submitted - X just opens a composer with this in it. */
export function intentUrl(text: string, url: string = SITE_URL): string {
  return `${X_INTENT_URL}?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
}

export function shareFileName(now: Date = new Date()): string {
  return `flybrain-${now.toISOString().replace(/[:.]/g, "-")}.png`;
}

/** The X logo, inline (no new deps). The menu-bar Share button uses it too. */
export function XGlyph({ size = 11 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M18.24 2.25h3.31l-7.23 8.26 8.5 11.24h-6.64l-5.2-6.8-5.95 6.8H1.72l7.48-8.55L1.05 2.25h6.8l4.71 6.23 5.68-6.23Zm-1.16 17.52h1.83L6.99 4.13H5.03l12.05 15.64Z" />
    </svg>
  );
}

export interface ShareDialogProps {
  /** PaintWindow hands over FlyCanvas.composite() - the trail + overlay + caption bitmap (a fresh detached canvas). */
  composite(): HTMLCanvasElement | null;
  tick: TickMsg | null;
  neurons?: number | null;
  /** false when the socket is not open / the newest tick is stale (PaintWindow measures it). */
  live?: boolean;
  onClose(): void;
}

/** The held still: `url` is what the preview shows AND what "Save image" downloads - the same bytes, always. */
interface Shot { url: string; w: number; h: number; objectUrl: boolean }

function revokeLater(url: string | null): void {
  if (!url) return;
  setTimeout(() => { try { URL.revokeObjectURL(url); } catch { /* ignore */ } }, REVOKE_MS);
}

export default function ShareDialog({ composite, tick, neurons = null, live = true, onClose }: ShareDialogProps) {
  const mood = tick?.mood.state ?? null;
  const chgM5 = tick && Number.isFinite(tick.market.chg_m5) ? tick.market.chg_m5 : null;
  const source = tick?.market.source ?? null;
  const tokLive = tokenLive(tick?.market.token_live);
  const facts = useMemo<ShareFacts>(() => ({ mood, chgM5, source, neurons, live, tokenLive: tokLive }),
    [mood, chgM5, source, neurons, live, tokLive]);
  // Frozen at open time: the painting and the numbers keep moving, an edit box that rewrites itself under your
  // cursor does not. "Refresh" re-takes both.
  const [text, setText] = useState<string>(() => composeShareText(facts));
  const [shot, setShot] = useState<Shot | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The tick keeps arriving at 4 Hz; "Refresh" reads the newest facts from here instead of re-running on every one.
  const factsRef = useRef<ShareFacts>(facts);
  useEffect(() => { factsRef.current = facts; }, [facts]);
  // The object URL the preview currently points at. It stays alive until a Refresh replaces it - and then only after
  // REVOKE_MS, because "Save image" hands this very URL to the browser's downloader.
  const heldUrl = useRef<string | null>(null);

  const publish = useCallback((url: string, w: number, h: number, objectUrl: boolean) => {
    revokeLater(heldUrl.current);
    heldUrl.current = objectUrl ? url : null;
    setShot({ url, w, h, objectUrl });
    setError(null);
  }, []);

  // Take ONE still and encode it once. Everything downstream (the preview, "Save image") uses that single PNG.
  const snap = useCallback(() => {
    try {
      const c = composite();
      if (!c || c.width === 0) { setError("The canvas is not ready yet."); return; }
      const w = c.width, h = c.height;
      if (typeof c.toBlob === "function") {
        c.toBlob((blob) => {
          if (!blob) { publish(c.toDataURL("image/png"), w, h, false); return; }
          publish(URL.createObjectURL(blob), w, h, true);
        }, "image/png");
      } else {
        publish(c.toDataURL("image/png"), w, h, false);
      }
    } catch (e) {
      console.error("[share] preview failed", e);
      setError("Could not read the canvas.");
    }
  }, [composite, publish]);

  // The first preview is taken on the next frame (a rAF callback: no setState inside the effect body).
  useEffect(() => {
    const raf = requestAnimationFrame(() => snap());
    return () => cancelAnimationFrame(raf);
  }, [snap]);

  // Closing the dialog releases the still - late, so a download started a moment ago still completes.
  useEffect(() => () => { revokeLater(heldUrl.current); heldUrl.current = null; }, []);

  // While the dialog is open the desktop's single-key shortcuts (S = sugar, N = clear ...) stand down: this capture
  // listener stops the event before page.tsx's document listener sees it. Typing in the textarea is untouched.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation();
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const refresh = useCallback(() => {
    snap();
    setText(composeShareText(factsRef.current));
    setSaved(null);
  }, [snap]);

  const saveImage = useCallback(() => {
    if (!shot) { setError("The picture is not ready yet - give it a moment, then press Refresh."); return; }
    try {
      const name = shareFileName();
      const a = document.createElement("a");
      a.href = shot.url;                     // exactly the bytes the preview is showing
      a.download = name;
      a.rel = "noopener";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setSaved(name);
      setError(null);
    } catch (e) {
      console.error("[share] save failed", e);
      setError("Saving the PNG failed - the browser blocked the download.");
    }
  }, [shot]);

  const openX = useCallback(() => {
    try {
      window.open(intentUrl(text), "_blank", "noopener,noreferrer");
    } catch (e) {
      console.error("[share] intent failed", e);
      setError("Could not open X - allow pop-ups for this site.");
    }
  }, [text]);

  const left = TEXT_BUDGET - text.length;

  return (
    <div className="fixed inset-0 z-[100000] flex items-center justify-center p-3" style={{ background: "rgba(0,0,0,0.15)" }}
      role="dialog" aria-modal="true" aria-label="Share this painting on X" data-testid="share-dialog"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="bevel-out flex max-h-[92vh] w-full max-w-[520px] flex-col overflow-auto" style={{ boxShadow: "2px 2px 0 #000" }}>
        <div className="titlebar flex h-[22px] items-center text-[13px]" style={{ padding: "0 2px 0 4px" }}>
          <span className="flex-1 truncate">Share to X - untitled ($SYNAPSE)</span>
          <button type="button" className="btn95 h-[16px] w-[18px] text-[11px] font-bold leading-none" style={{ padding: 0 }}
            onClick={onClose} title="Close" aria-label="close">
            <span className="relative top-[-1px]">&times;</span>
          </button>
        </div>

        <div className="flex flex-col gap-2 p-3 text-[12px]">
          {/* the held still - the same PNG "Save image" hands to the browser */}
          <div className="bevel-in bg-paint-white p-1" style={{ lineHeight: 0 }}>
            {shot ? (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img src={shot.url} alt="The painting as it will be saved" width={shot.w} height={shot.h} data-testid="share-preview"
                style={{ width: "100%", height: "auto", imageRendering: "pixelated", display: "block" }} />
            ) : (
              <div className="p-3 text-center text-[11px] text-win-dark" style={{ lineHeight: "normal" }}>taking the picture...</div>
            )}
          </div>
          <div className="flex items-center gap-2 text-[11px] text-win-dark">
            <span className="flex-1">{shot
              ? `${shot.w} x ${shot.h} PNG. "Save image" writes this exact still; the fly has painted on since.`
              : "compositing the trail, the fly and the caption bar"}</span>
            <button type="button" className="btn95 h-[19px]" onClick={refresh} title="Take the picture again and rewrite the text from the current tick">
              Refresh
            </button>
          </div>

          <label className="mt-1 block" htmlFor="share-text">Post text</label>
          <textarea
            id="share-text"
            className="bevel-in w-full resize-none bg-paint-white p-1 font-mono text-[12px] leading-[15px] text-black"
            rows={4}
            value={text}
            maxLength={TEXT_BUDGET}
            onChange={(e) => setText(e.target.value)}
            data-testid="share-text"
            spellCheck={false}
          />
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-win-dark">X appends {SITE_URL} as a link (23 characters).</span>
            <span className="ml-auto font-mono" style={{ color: left < 0 ? "#c00000" : "#404040" }}>{left}</span>
          </div>
          {!live ? (
            <div className="text-[11px] text-win-dark" data-testid="share-stale">
              No tick is arriving right now, so the text carries no mood and no market number: the last ones are from
              before the brain went quiet, and a stale number is not worth posting.
            </div>
          ) : null}

          <div className="bevel-in bg-win-gray px-2 py-1 text-[11px]">
            <b>X cannot take the image through a link.</b> Save the PNG first, then attach it in the composer that
            opens - the text and the site link are already filled in. Nothing is posted from this page.
          </div>
          {saved ? <div className="text-[11px]" data-testid="share-saved">Saved <b>{saved}</b> to your downloads. Attach that file on X.</div> : null}
          {error ? <div className="text-[11px] font-bold text-[#c00000]" data-testid="share-error">{error}</div> : null}
        </div>

        <div className="flex flex-wrap justify-center gap-2 px-3 pb-3">
          <button type="button" className="btn95 h-[23px] min-w-[90px]" onClick={saveImage} style={{ outline: "1px solid #000", outlineOffset: -3 }}
            data-testid="share-save">
            Save image
          </button>
          <button type="button" className="btn95 flex h-[23px] min-w-[90px] items-center justify-center gap-1" onClick={openX}
            title={`Open the X composer with this text (${X_HANDLE})`} data-testid="share-open-x">
            <XGlyph /> Open X
          </button>
          <button type="button" className="btn95 h-[23px] min-w-[75px]" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
