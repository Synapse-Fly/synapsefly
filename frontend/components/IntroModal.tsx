"use client";
// First-visit explainer: a Win95 dialog that says, in plain language, what SynapseFly is and how to read the screen.
// Shown once per browser (localStorage) and reopenable from the taskbar "What is this?" button. Zero new deps.
import { useEffect, useRef, useState } from "react";
import type { FlySocket } from "@/lib/ws";
import { TOKEN, X_URL, GITHUB_URL, JANELIA_URL, NOTICE_URL, NOT_LAUNCHED_NOTE, feedSubject, tokenLive } from "@/lib/brand";
import { TASKBAR_H } from "@/lib/layout";

export interface IntroModalProps {
  sock: FlySocket;
  onClose(): void;
}

const PIXELATED: React.CSSProperties = { imageRendering: "pixelated" };

function Bullet({ icon, children }: { icon: string; children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2">
      <span className="mt-[1px] text-[15px] leading-none" aria-hidden>{icon}</span>
      <span>{children}</span>
    </li>
  );
}

function LegendRow({ win, what }: { win: string; what: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="whitespace-nowrap font-bold">{win}</span>
      <span className="text-[#404040]">{what}</span>
    </div>
  );
}

export default function IntroModal({ sock, onClose }: IntroModalProps) {
  const hello = sock.hello;
  // A missing flag counts as "not launched" (brand.ts), so an old backend cannot silently drop the notice.
  const live = tokenLive(hello?.market.token_live);
  const n = hello?.connectome.n ?? 20000;
  const chain = hello?.market.chain ?? "";
  const token = hello?.market.token ?? "";
  const chartUrl = chain && token ? `https://dexscreener.com/${chain}/${token}` : "https://dexscreener.com";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "Enter") { e.preventDefault(); onClose(); }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  // "There is more below". The body only scrolls on a viewport too short for the whole explainer, and on a phone the
  // Win95 scrollbar the CSS asks for is an OVERLAY scrollbar the browser hides at rest - so a clipped last line has
  // no affordance at all unless we draw one. Never a synchronous setState in the effect body
  // (react-hooks/set-state-in-effect): the first read comes from the ResizeObserver's own initial callback.
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [more, setMore] = useState(false);
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const read = () => setMore(el.scrollTop + el.clientHeight < el.scrollHeight - 1);
    el.addEventListener("scroll", read, { passive: true });
    window.addEventListener("resize", read);
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(read) : null;
    ro?.observe(el);
    return () => {
      el.removeEventListener("scroll", read);
      window.removeEventListener("resize", read);
      ro?.disconnect();
    };
  }, []);

  return (
    // The overlay still covers the whole screen (so nothing behind it is clickable), but its CONTENT box - which is
    // what `max-h-full` on the dialog measures against - stops above the taskbar. `92vh` measured against the
    // viewport instead, and on a 390 x 844 phone that left the body 53 px short with the last line of text sliced
    // through the middle and welded to the footer buttons.
    <div
      className="fixed inset-0 z-[100000] flex items-center justify-center bg-black/40 p-2 sm:p-3"
      style={{ paddingBottom: TASKBAR_H }}
      role="dialog"
      aria-modal="true"
      aria-label="What is SynapseFly"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bevel-out flex max-h-full w-full max-w-[580px] flex-col overflow-hidden" style={{ boxShadow: "2px 2px 0 #000" }}>
        {/* title bar */}
        <div className="titlebar flex shrink-0 items-center gap-2 text-[13px]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/favicon-48.png" alt="" width={16} height={16} style={PIXELATED} />
          <span className="flex-1">SynapseFly (${TOKEN}) — what is this?</span>
          <button type="button" className="btn95 h-[18px] w-[20px] text-[11px] leading-[12px]" onClick={onClose} aria-label="Close" title="Close">x</button>
        </div>

        {/* body. `win95-scroll` (app/desktop.css) gives it a real, visible Win95 scrollbar the moment it does
            overflow - a panel that scrolls with no scrollbar reads as a clipped layout, which is exactly how this
            looked on a phone - and the "more" chip below covers the phones whose scrollbar is an overlay one. */}
        <div className="relative flex min-h-0 flex-1 flex-col">
          <div ref={bodyRef} className="win95-scroll min-h-0 flex-1 overflow-auto overscroll-contain bg-win-gray p-3 text-[12px] leading-[1.5]">
            {/* Row, not a column, on a phone too: stacked, the 112 px logo alone cost 136 px of a 714 px body and
                pushed the last paragraph off the bottom. Beside the text at 64 px it costs nothing - the text block
                is taller than the logo either way - and the whole explainer fits a 390 x 844 screen with no scroll. */}
            <div className="flex flex-row gap-3">
              <div className="flex shrink-0 justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src="/logo-256.png" alt="SynapseFly pixel fly" width={112} height={112} style={PIXELATED} className="bevel-in h-16 w-16 bg-win-teal p-1 sm:h-28 sm:w-28" />
              </div>
              <div>
                <div className="text-[16px] font-bold leading-[1.2]">A fruit-fly brain, trading on vibes.</div>
                <p className="mt-1">
                  We rebuilt the wiring map of a fruit fly&apos;s brain — modeled on Janelia&apos;s
                  ~166,000-neuron MaleCNS connectome — as a synthetic MaleCNS-shaped brain of{" "}
                  {n.toLocaleString("en-US")} spiking neurons, and plugged the market into its senses. It runs live,
                  right now, on this page.
                </p>
              </div>
            </div>

            <div className="bevel-in mt-3 bg-white p-2">
              <div className="mb-1 font-bold">How it feels the market</div>
              <ul className="flex flex-col gap-1">
                <Bullet icon="🍬">A <b>buy</b> / price up feeds its <b>sugar</b> neurons — it gets happy, wanders and loops.</Bullet>
                <Bullet icon="📉">A <b>sell</b> / price down lights up its <b>looming-danger</b> and <b>escape</b> circuits — it panics, darts and hides.</Bullet>
                <Bullet icon="🖌️">Its <b>motor neurons</b> steer it across the canvas — the fly literally paints the price.</Bullet>
                <Bullet icon="🐦">On <b>euphoria</b> or <b>panic</b> it writes its own absurd tweet.</Bullet>
              </ul>
            </div>

            <div className="bevel-in mt-3 bg-white p-2">
              <div className="mb-1 font-bold">The windows</div>
              <div className="flex flex-col gap-[2px]">
                <LegendRow win="Paint" what="the fly and the trail it paints" />
                <LegendRow win="Oscilloscope" what="its neurons actually firing (spike raster)" />
                <LegendRow win="Fly Status" what="its live mood + the market it feels" />
                <LegendRow win="tweets.txt" what="what the fly has posted" />
              </div>
            </div>

            {/* Not fine print: the brain on screen is a synthetic stand-in, and a visitor must be told that in the
                same type size as everything else (docs/NOTICE.md section 1). */}
            <div className="bevel-in mt-3 bg-white p-2">
              <div className="mb-1 font-bold">What is actually running</div>
              <p>
                A <b>synthetic MaleCNS-shaped brain</b> of {n.toLocaleString("en-US")} spiking neurons: the real region
                taxonomy, cell-type names and published pathway weights, generated here — <b>not</b> a copy of the
                measured dataset. The real{" "}
                <a className="font-bold underline" href={JANELIA_URL} target="_blank" rel="noopener noreferrer">
                  Janelia MaleCNS connectome
                </a>{" "}
                (CC-BY 4.0, Berg et al. 2026) is one environment variable away.{" "}
                <a className="font-bold underline" href={NOTICE_URL} target="_blank" rel="noopener noreferrer">
                  Data &amp; citations ↗
                </a>
              </p>
              <p className="mt-1">
                <b>{NOT_LAUNCHED_NOTE}</b>{" "}
                {live
                  ? null
                  : hello?.market.mode === "sim"
                    ? "The price on screen is a simulation, fed to the brain as sensory input."
                    : `The price on screen belongs to ${feedSubject(hello?.market.symbol, undefined, hello?.market.chain)}, an unrelated pair the brain reads as sensory input.`}
              </p>
              <p className="mt-1">Not financial advice — it&apos;s a bug.</p>
            </div>
          </div>
          {more ? (
            <div
              className="btn95 pointer-events-none absolute bottom-[6px] right-[22px] flex h-[16px] items-center text-[10px] leading-[10px]"
              aria-hidden
            >
              ▼ more
            </div>
          ) : null}
        </div>

        {/* footer. shrink-0 so it is never squeezed out of a short viewport, and pt-2 so the buttons are not welded
            to the last line of the body (they were, which is half of why the clipped line read as a layout bug). */}
        <div className="flex shrink-0 flex-wrap items-center gap-2 bg-win-gray px-3 pt-2 pb-3">
          <a className="btn95 flex h-[24px] items-center text-[12px]" href={X_URL} target="_blank" rel="noopener noreferrer" title="Follow on X">
            𝕏 Follow
          </a>
          <a className="btn95 flex h-[24px] items-center text-[12px]" href={GITHUB_URL} target="_blank" rel="noopener noreferrer" title="Source on GitHub">
            GitHub ↗
          </a>
          {token ? (
            <a className="btn95 flex h-[24px] items-center text-[12px]" href={chartUrl} target="_blank" rel="noopener noreferrer" title="Live chart on DexScreener">
              Chart ↗
            </a>
          ) : null}
          <div className="min-w-[8px] flex-1" />
          <button type="button" className="btn95 h-[24px] px-3 text-[12px] font-bold" onClick={onClose} autoFocus>
            Enter the canvas →
          </button>
        </div>
      </div>
    </div>
  );
}
