// Single source of brand truth. The market feed's symbol may be a placeholder ("FLY" in sim, or a test token)
// until the real $SYNAPSE token lists; the UI always shows the brand ticker below and links to our accounts.
export const TOKEN = "SYNAPSE";
export const PROJECT = "SynapseFly";
export const X_URL = "https://x.com/SynapseFly";
export const X_HANDLE = "@SynapseFly";
export const GITHUB_URL = "https://github.com/Synapse-Fly/synapsefly";
export const LOGO = "/favicon-48.png";
/** Provenance links the UI must be able to show (the CC-BY attribution is the credibility anchor, docs/NOTICE.md). */
export const JANELIA_URL = "https://male-cns.janelia.org/";
export const NOTICE_URL = `${GITHUB_URL}/blob/main/docs/NOTICE.md`;

// ------------------------------------------------------------------------------- token launch state (honesty rule)
// $SYNAPSE HAS NOT LAUNCHED. There is no $SYNAPSE pair, so whatever the brain is reading belongs to someone else:
// a generated feed under FLY_MARKET=sim, or - when the backend polls DexScreener - a REAL, actively traded
// third-party pair. Either way a price printed under a bare "$SYNAPSE" is a claim this project cannot make, so every
// surface that shows the number also names what the number is. The wire says which state we are in through
// `market.token_live` (hello.market and every tick's market block).

/** The tracked pair is only $SYNAPSE when the server says so.
 *
 *  Takes the flag from the most authoritative source first (the tick, then hello) and TREATS A MISSING FLAG AS
 *  FALSE: an old backend, a stale deploy or a socket that never opened must fail safe. The qualifiers below may
 *  disappear only when a server has positively said `token_live: true`. */
export function tokenLive(...flags: readonly (boolean | null | undefined)[]): boolean {
  for (const f of flags) if (f === true || f === false) return f;
  return false;
}

/** The qualifier that rides with the ticker itself, so no price is ever printed under a bare "$SYNAPSE". */
export const NOT_LAUNCHED = "not launched";
/** One plain sentence, for the surfaces with room for prose (intro modal, share dialog). */
export const NOT_LAUNCHED_NOTE = `$${TOKEN} has not launched: there is no ${TOKEN} token and no ${TOKEN} market yet.`;
/** FLY_MARKET=sim: the numbers are generated here. Wording kept verbatim from the original sim disclaimer. */
export const SIM_FEED_NOTE = `simulated feed \u2014 no $${TOKEN} market yet`;

function trimmed(s: string | null | undefined): string | null {
  const t = typeof s === "string" ? s.trim() : "";
  return t.length ? t : null;
}

/** "CASHCAT on uniswap/robinhood" - whose market this actually is, built from the live tick (never hardcoded). */
export function feedSubject(symbol?: string | null, dex?: string | null, chain?: string | null): string {
  const who = trimmed(symbol) ?? "an unrelated token";
  const where = [trimmed(dex), trimmed(chain)].filter((s): s is string => s !== null).join("/");
  return where ? `${who} on ${where}` : who;
}

/** The red line under the price when a REAL third-party pair is being tracked under this brand. */
export function standInFeedNote(symbol?: string | null, dex?: string | null, chain?: string | null): string {
  return `stand-in feed: ${feedSubject(symbol, dex, chain)} \u2014 this is NOT $${TOKEN}`;
}

// ---------------------------------------------------------------------------------------- contract address surface
// The launch post points visitors at this site to get the contract address, so three surfaces show it: the CA.txt
// window (components/ContractWindow.tsx), the compact row in the Fly Status price panel (components/MarketTicker.tsx)
// and the taskbar tray chip (app/page.tsx). All three read the SAME two wire fields -
//
//     address   = hello.market.token        (FLY_TOKEN_ADDRESS, the pair the backend polls; "" while unset)
//     launched  = tokenLive(tick?.market.token_live, hello?.market.token_live)
//
// - and every string, URL and truncation they print comes from this file, so they cannot drift apart. NOTHING here
// invents, pads or placeholders an address: with no address on the wire there is no address on the page.

/** Desktop icon label / window title for the CA surface (Win95 named its windows after the document). */
export const CA_FILE = "CA.txt";
export const CA_WINDOW_TITLE = `${CA_FILE} - Notepad`;
/** The anti-impersonation line. Scam tokens copy the ticker, never the address, so the address is the only check. */
export const CA_VERIFY_NOTE =
  `Check this address against this page and the pinned post on ${X_HANDLE} before you buy. `
  + `Same ticker, different address is NOT us.`;
/** Pre-launch, in place of an address - so an early visitor leaves knowing that every "CA" in their replies is fake. */
export const CA_NONE_NOTE = `no contract address yet \u2014 anything you see elsewhere is not us`;

/**
 * The address the UI is allowed to print, or null. `hello.market.token` is an empty string until the owner sets
 * FLY_TOKEN_ADDRESS, and a blank token means NO ADDRESS - never a dash, never "TBA", never a zero address.
 */
export function contractAddress(token?: string | null): string | null {
  return trimmed(token);
}

/**
 * `0x8f3C1b...1B2c` (with a real ellipsis) - first `head` / last `tail`, the truncation wallets and explorers use,
 * for the surfaces that cannot spend 42 monospace characters. Returns "" when there is no address (callers then
 * render nothing), and returns a
 * short address UNCHANGED rather than eliding something the reader could have had in full.
 */
export function shortAddress(addr?: string | null, head = 6, tail = 4): string {
  const s = trimmed(addr);
  if (s === null) return "";
  return s.length <= head + tail + 1 ? s : `${s.slice(0, head)}\u2026${s.slice(-tail)}`;
}

/**
 * DexScreener's page for this token: `https://dexscreener.com/{chain}/{address}` when the wire gave us a chain, and
 * its search otherwise (which resolves the address on whatever chain it really lives on - a search URL states no
 * claim about the chain, so it is honest where a guessed `/{chain}/` would not be). null without an address.
 */
export function dexscreenerUrl(chain?: string | null, address?: string | null): string | null {
  const a = trimmed(address);
  if (a === null) return null;
  const c = trimmed(chain);
  const addr = encodeURIComponent(a);
  return c === null
    ? `https://dexscreener.com/search?q=${addr}`
    : `https://dexscreener.com/${encodeURIComponent(c.toLowerCase())}/${addr}`;
}

export interface ChainLink { name: string; url: string }

/**
 * The block explorer for `chain`, or null.
 *
 * Deliberately a closed list. An explorer domain cannot be derived from a chain name, and a link to a domain this
 * project guessed is worse than no link at all - at best a 404, at worst somebody else's site under our address. An
 * unrecognised chain therefore gets NO explorer link and the caller omits the button; DexScreener (which takes the
 * chain id the backend is already polling with) still works everywhere.
 */
export function explorerLink(chain?: string | null, address?: string | null): ChainLink | null {
  const a = trimmed(address);
  const c = trimmed(chain);
  if (a === null || c === null) return null;
  const addr = encodeURIComponent(a);
  switch (c.toLowerCase().replace(/[^a-z0-9]/g, "")) {
    case "ethereum": case "eth": case "mainnet":
      return { name: "Etherscan", url: `https://etherscan.io/token/${addr}` };
    case "bsc": case "bnb": case "bnbchain": case "binancesmartchain":
      return { name: "BscScan", url: `https://bscscan.com/token/${addr}` };
    case "base":
      return { name: "BaseScan", url: `https://basescan.org/token/${addr}` };
    case "arbitrum": case "arb": case "arbitrumone":
      return { name: "Arbiscan", url: `https://arbiscan.io/token/${addr}` };
    case "polygon": case "matic":
      return { name: "PolygonScan", url: `https://polygonscan.com/token/${addr}` };
    case "optimism": case "op":
      return { name: "Optimistic Etherscan", url: `https://optimistic.etherscan.io/token/${addr}` };
    case "avalanche": case "avax":
      return { name: "Snowtrace", url: `https://snowtrace.io/token/${addr}` };
    case "solana": case "sol":
      return { name: "Solscan", url: `https://solscan.io/token/${addr}` };
    case "blast":
      return { name: "BlastScan", url: `https://blastscan.io/token/${addr}` };
    case "linea":
      return { name: "LineaScan", url: `https://lineascan.build/token/${addr}` };
    case "scroll":
      return { name: "ScrollScan", url: `https://scrollscan.com/token/${addr}` };
    case "fantom": case "ftm":
      return { name: "FtmScan", url: `https://ftmscan.com/token/${addr}` };
    case "cronos":
      return { name: "CronoScan", url: `https://cronoscan.com/token/${addr}` };
    case "gnosis": case "xdai":
      return { name: "GnosisScan", url: `https://gnosisscan.io/token/${addr}` };
    case "celo":
      return { name: "CeloScan", url: `https://celoscan.io/token/${addr}` };
    case "moonbeam":
      return { name: "Moonscan", url: `https://moonscan.io/token/${addr}` };
    default:
      return null;     // chain we cannot name honestly: no link (see the note above)
  }
}

/** What `copyText` managed. "manual" = nothing reached the clipboard; the caller must select the text instead. */
export type CopyResult = "clipboard" | "execCommand" | "manual";

/**
 * Put `text` on the clipboard, by whichever of the two paths this browser allows.
 *
 * 1. `navigator.clipboard.writeText` - the modern path, but it exists only in a SECURE context and can still be
 *    refused (permission policy, an iframe without clipboard-write, Safari outside a user gesture);
 * 2. `document.execCommand("copy")` over a hidden, off-screen textarea - deprecated, and the only thing that works
 *    over plain http:// (which is how this page is served in development).
 *
 * Returns "manual" when both fail, and the caller then selects the address and asks for Ctrl+C - a visitor who came
 * for the contract address must never be left with a dead button and no address they can take.
 */
export async function copyText(text: string): Promise<CopyResult> {
  const s = typeof text === "string" ? text : "";
  if (!s) return "manual";
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(s);
      return "clipboard";
    }
  } catch {
    /* insecure context or permission denied: fall through to the textarea */
  }
  if (typeof document === "undefined") return "manual";
  let ta: HTMLTextAreaElement | null = null;
  try {
    ta = document.createElement("textarea");
    ta.value = s;
    ta.readOnly = true;
    ta.setAttribute("aria-hidden", "true");
    // Off-screen rather than display:none / hidden: the selection APIs ignore an unrendered element, and a
    // position:fixed textarea cannot scroll the page the way a negative-margin one can.
    ta.style.cssText = "position:fixed;top:0;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
    document.body.appendChild(ta);
    ta.focus({ preventScroll: true });
    ta.select();
    ta.setSelectionRange(0, s.length);
    return document.execCommand("copy") ? "execCommand" : "manual";
  } catch {
    return "manual";
  } finally {
    ta?.remove();
  }
}

/** Select an element's text so the visitor can finish the copy with Ctrl+C when the clipboard refused us. */
export function selectText(el: HTMLElement | null | undefined): void {
  if (!el || typeof window === "undefined" || typeof document === "undefined") return;
  try {
    const sel = window.getSelection();
    if (!sel) return;
    const range = document.createRange();
    range.selectNodeContents(el);
    sel.removeAllRanges();
    sel.addRange(range);
  } catch {
    /* no selection in this context: the address is still on screen and selectable by hand */
  }
}
