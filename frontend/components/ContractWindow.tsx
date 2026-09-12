"use client";
// "CA.txt - Notepad": the contract address, in full, one click from the desktop.
//
// Why this window exists at all: the launch post points people at this site to get the contract address, and a
// visitor who cannot find it in three seconds either leaves or buys whatever address a reply guy gave them. So the
// address gets a desktop icon directly under Paint, a taskbar chip, and this window - and all three read the same two
// fields off the wire (lib/brand.ts: `hello.market.token` plus `tokenLive(tick, hello)`), so they cannot disagree
// with each other or with the Fly Status price panel.
//
// The honesty rule is load-bearing here more than anywhere else on the page. Until the backend reports
// `token_live: true` AND a non-empty `FLY_TOKEN_ADDRESS`, this window shows no address at all - not a dash, not a
// zero address, not "TBA", nothing a screenshot could crop into something that looks like a contract address. A
// missing flag counts as not launched (brand.ts `tokenLive`), so a stale deploy or a socket that never opened fails
// to the safe state by construction.
import { useEffect, useRef, useState, type ReactNode } from "react";
import Win95Window, { type WinGeometry, type WinRect } from "@/components/Win95Window";
import { AppIcon } from "@/lib/icons";
import type { FlySocket } from "@/lib/ws";
import { useTickSnapshot } from "@/lib/store";
import {
  CA_NONE_NOTE, CA_VERIFY_NOTE, CA_WINDOW_TITLE, GITHUB_URL, NOT_LAUNCHED, NOT_LAUNCHED_NOTE, TOKEN, X_HANDLE,
  X_URL, contractAddress, copyText, dexscreenerUrl, explorerLink, selectText, tokenLive,
} from "@/lib/brand";

/** The window id in the desktop store (Win95Window's initDesktop / desktop.open). */
export const CA_WINDOW_ID = "ca";

/** How long the button reads "Copied." before going back to "Copy". */
const COPIED_MS = 1500;
/** The manual fallback has to be read, not glimpsed, so it stays up longer. */
const MANUAL_MS = 4000;

const RED = "#a80000";

/**
 * Anything with a `.current` holding an element - i.e. a `RefObject` of ANY element type. Declared readonly so a
 * `useRef<HTMLDivElement | null>` is assignable (a mutable `RefObject<T>` is invariant in T, a readonly one is not).
 */
export interface TextTarget { readonly current: HTMLElement | null }

export interface CopyButtonProps {
  /** The text to copy. Empty disables the button - there is nothing to put on a clipboard. */
  value: string;
  /** The element showing the same text. Selected for the visitor when the clipboard refuses us. */
  target?: TextTarget;
  className?: string;
  /** Idle label ("Copy"). */
  label?: string;
  title?: string;
  testId?: string;
}

/**
 * The one Copy button, used by BOTH CA surfaces (this window and the MarketTicker row) so the two cannot end up with
 * different clipboard behaviour. lib/brand.ts `copyText` owns the two copy paths; this owns the feedback:
 *
 *   "Copy"  ->  "Copied."       the clipboard took it (either path)
 *           ->  "Press Ctrl+C"  neither path worked - the address has been SELECTED for them instead, so the visitor
 *                               still leaves with it. A dead button on this particular control is not acceptable.
 */
export function CopyButton({ value, target, className, label = "Copy", title, testId }: CopyButtonProps) {
  const [state, setState] = useState<"idle" | "copied" | "manual">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timer.current !== null) clearTimeout(timer.current); }, []);

  // A plain handler, NOT useCallback: it reads `target.current`, and the React Compiler refuses to preserve a manual
  // memoization whose inferred deps include a ref read (react-hooks/preserve-manual-memoization). The compiler
  // memoizes this for us anyway.
  const onClick = () => {
    if (timer.current !== null) { clearTimeout(timer.current); timer.current = null; }
    // setState runs in the promise callback (an event, not a render or an effect body), which is what keeps this off
    // react-hooks/set-state-in-effect.
    void copyText(value).then((res) => {
      const manual = res === "manual";
      if (manual) selectText(target?.current ?? null);
      setState(manual ? "manual" : "copied");
      timer.current = setTimeout(() => { timer.current = null; setState("idle"); }, manual ? MANUAL_MS : COPIED_MS);
    });
  };

  const text = state === "copied" ? "Copied." : state === "manual" ? "Press Ctrl+C" : label;
  return (
    <button
      type="button"
      className={className ?? "btn95"}
      onClick={onClick}
      disabled={!value}
      data-state={state}
      data-testid={testId}
      title={title ?? (value ? `Copy ${value} to the clipboard` : "nothing to copy")}
    >
      <span aria-live="polite">{text}</span>
    </button>
  );
}

// ------------------------------------------------------------------------------------------------------- the window

export interface ContractWindowProps {
  sock: FlySocket;
  /** Computed layout rect (page.tsx centres this one in the desktop box; it is a dialog, not a tiled app). */
  rect: WinRect;
  bounds?: WinRect;
  onGeometry?: (id: string, g: WinGeometry) => void;
}

function Out({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      className="btn95 flex h-[23px] items-center text-[12px] no-underline"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  );
}

export default function ContractWindow({ sock, rect, bounds, onGeometry }: ContractWindowProps) {
  const hello = sock.hello;
  const tick = useTickSnapshot(sock.store);
  const addrRef = useRef<HTMLDivElement | null>(null);

  // The same two reads the ticker and the tray chip do. The tick is the freshest carrier of the flag, hello the
  // fallback, and an absent flag is NOT launched.
  const live = tokenLive(tick?.market.token_live, hello?.market.token_live);
  const address = contractAddress(hello?.market.token);
  // The chain for the CA surface comes from hello.market.chain - the CONFIGURED chain (FLY_CHAIN), which the
  // backend overlays onto hello from the first frame at launch. NEVER tick.market.chain: while the feed serves
  // the sim tape that field reads "sim", which would print "chain sim" and link to dexscreener.com/sim/<address>.
  const chain = (hello?.market.chain ?? "").trim();
  /** Three distinct states - the address body needs BOTH the launch flag and the string. */
  const hasAddress = live && address !== null;

  const dex = hasAddress ? dexscreenerUrl(chain, address) : null;
  const explorer = hasAddress ? explorerLink(chain, address) : null;

  return (
    <Win95Window
      id={CA_WINDOW_ID}
      title={CA_WINDOW_TITLE}
      icon={<AppIcon name="contract" size={16} />}
      rect={rect}
      bounds={bounds}
      onGeometry={onGeometry}
    >
      <div
        className="win95-scroll flex h-full flex-col gap-[6px] overflow-auto bg-win-gray p-2 text-[12px] leading-[15px]"
        data-testid="ca-window"
        data-live={hasAddress ? "true" : live ? "pending" : "false"}
      >
        {hasAddress ? (
          <>
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-[2px]">
              <span className="text-[15px] font-bold">${TOKEN}</span>
              <span className="text-[#404040]">contract address</span>
              {chain ? (
                <span className="ml-auto text-[11px] text-[#404040]">
                  chain <b className="font-mono text-black">{chain}</b>
                </span>
              ) : null}
            </div>

            {/* The address itself: monospace, selectable (the window frame sets `select-none`), and never truncated -
                this is the one string on the site a visitor is going to compare character by character. */}
            <div
              ref={addrRef}
              className="bevel-in select-text px-2 py-[5px] font-mono text-[13px] leading-[17px] break-all"
              data-testid="ca-address"
            >
              {address}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <CopyButton
                value={address}
                target={addrRef}
                className="btn95 h-[25px] min-w-[96px] text-[12px] font-bold"
                testId="ca-copy"
              />
              {dex ? <Out href={dex}>DexScreener</Out> : null}
              {/* No explorer link for a chain lib/brand.ts cannot name honestly - a guessed domain is worse than none. */}
              {explorer ? <Out href={explorer.url}>{explorer.name}</Out> : null}
            </div>

            <div className="bevel-in px-2 py-1 text-[11px] leading-[14px]" data-testid="ca-verify">
              <b style={{ color: RED }}>{CA_VERIFY_NOTE}</b>
            </div>
          </>
        ) : live ? (
          // Launched, but no address on THIS frame (a reconnect, or a client that opened while the feed was still
          // on the sim tape). With the backend overlay this is rare, but it must NEVER read as the opposite of the
          // truth: the token HAS launched, the address is on its way, and the pinned post carries it right now.
          <>
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-[2px]">
              <span className="text-[15px] font-bold">${TOKEN}</span>
              <span className="text-[#404040]">contract address</span>
            </div>

            <div className="bevel-in px-2 py-[5px]" data-testid="ca-pending">
              <div className="text-[13px] font-bold">${TOKEN} has launched - loading the address...</div>
              <div className="mt-1 text-[11px] leading-[14px]">
                Fetching it from the live feed; it will appear here in a moment. Until then, the pinned post on{" "}
                {X_HANDLE} has the address. Same ticker, different address is not us.
              </div>
            </div>

            <div className="mt-auto flex flex-wrap items-center gap-2 pt-[2px]">
              <Out href={X_URL}>Follow {X_HANDLE}</Out>
              <Out href={GITHUB_URL}>GitHub</Out>
            </div>
          </>
        ) : (
          <>
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-[2px]">
              <span className="text-[15px] font-bold">${TOKEN}</span>
              <span className="text-[15px] font-bold" style={{ color: RED }}>{NOT_LAUNCHED}</span>
            </div>

            {/* Where the address WILL be. Deliberately empty of anything address-shaped. */}
            <div className="bevel-in px-2 py-[5px]" data-testid="ca-none">
              <div className="text-[13px] font-bold" style={{ color: RED }}>{CA_NONE_NOTE}</div>
              <div className="mt-1 text-[11px] leading-[14px]">{NOT_LAUNCHED_NOTE}</div>
            </div>

            <div className="text-[11px] leading-[14px]">
              When it launches, the address shows up here, in the Fly Status price panel, and in the pinned post on{" "}
              {X_HANDLE}. Those three have to match. Anything else - a reply, a DM, a lookalike site - is not us.
            </div>

            <div className="mt-auto flex flex-wrap items-center gap-2 pt-[2px]">
              <Out href={X_URL}>Follow {X_HANDLE}</Out>
              <Out href={GITHUB_URL}>GitHub</Out>
            </div>
          </>
        )}
      </div>
    </Win95Window>
  );
}
