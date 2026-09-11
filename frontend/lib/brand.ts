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
