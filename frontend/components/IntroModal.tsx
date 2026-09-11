"use client";
// First-visit explainer: a Win95 dialog that says, in plain language, what SynapseFly is and how to read the screen.
// Shown once per browser (localStorage) and reopenable from the taskbar "What is this?" button. Zero new deps.
import { useEffect } from "react";
import type { FlySocket } from "@/lib/ws";
import { TOKEN, X_URL, GITHUB_URL } from "@/lib/brand";

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

  return (
    <div
      className="fixed inset-0 z-[100000] flex items-center justify-center bg-black/40 p-3"
      role="dialog"
      aria-modal="true"
      aria-label="What is SynapseFly"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bevel-out flex max-h-[92vh] w-full max-w-[580px] flex-col overflow-hidden" style={{ boxShadow: "2px 2px 0 #000" }}>
        {/* title bar */}
        <div className="titlebar flex items-center gap-2 text-[13px]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/favicon-48.png" alt="" width={16} height={16} style={PIXELATED} />
          <span className="flex-1">SynapseFly (${TOKEN}) — what is this?</span>
          <button type="button" className="btn95 h-[18px] w-[20px] text-[11px] leading-[12px]" onClick={onClose} aria-label="Close" title="Close">x</button>
        </div>

        {/* body */}
        <div className="flex-1 overflow-auto bg-win-gray p-3 text-[12px] leading-[1.5]">
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="flex shrink-0 justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/logo-256.png" alt="SynapseFly pixel fly" width={112} height={112} style={PIXELATED} className="bevel-in bg-win-teal p-1" />
            </div>
            <div>
              <div className="text-[16px] font-bold leading-[1.2]">A real fruit-fly brain, trading on vibes.</div>
              <p className="mt-1">
                We took the wiring map of a fruit fly&apos;s brain — Janelia&apos;s ~166,000-neuron
                connectome — and plugged the market into its senses. It runs live, right now, on this page
                (a {n.toLocaleString()}-neuron spiking simulation).
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

          <p className="mt-2 text-[10px] text-[#404040]">
            Synthetic MaleCNS-shaped brain ({n.toLocaleString()} neurons) modeled on Janelia FlyEM&apos;s male
            Drosophila connectome (CC-BY). Not financial advice — it&apos;s a bug.
          </p>
        </div>

        {/* footer */}
        <div className="flex flex-wrap items-center gap-2 bg-win-gray px-3 pb-3">
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
