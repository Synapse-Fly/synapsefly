"""Tweet text generation (SPEC section c.21).

Two sources of text:

* ``template_tweet`` - deterministic crypto-dialect templates (>= 6 per trigger reason, EN and TR), filled with numbers
  from the brain summary. This is the dry-run path (``FLY_LLM=dryrun``, the default) and the fallback of every failure.
* ``TweetGenerator.generate`` - the one Anthropic Messages call of the project (``FLY_LLM=anthropic``):
  ``client.messages.create(model=..., max_tokens=512, system=SYSTEM_PROMPT [+ lang line], messages=[user JSON])`` with
  structured output via ``output_config`` when ``FLY_LLM_JSON=1``. No ``thinking`` parameter, never ``budget_tokens``,
  no assistant prefill, no ``temperature``. ``anthropic`` is imported inside the function, only when that mode is on.

``validate_tweet`` cleans any text (URL strip, whitespace, quotes, 280-char word-boundary cap, <= 2 hashtags, no
cashtags other than $FLY) and requires at least one ``VOCABULARY`` token so a tweet is always neuroscientifically
literal. Provenance: every neuron name below is a real MaleCNS type or group name [V] (RESEARCH section 4).

Honesty, both paths: the brain is a synthetic stand-in unless real data is loaded (``connectome.source``), and the
market feed is somebody else's pair unless ``FLY_TOKEN_LIVE=1`` says otherwise (``summary.market.token_live``, false
by default - this project has no token yet). ``SYSTEM_PROMPT`` states both rules to the model, and ``{symbol}`` in the
templates renders as "the feed" rather than a ticker while the flag is false, so no generated text can attach a price,
a market cap or a liquidity number to this project's name.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flybrain.config import Settings

__all__ = [
    "SYSTEM_PROMPT",
    "TWEET_SCHEMA",
    "VOCABULARY",
    "TweetDraft",
    "TweetGenerator",
    "template_tweet",
    "validate_tweet",
    "system_prompt_for",
]

log = logging.getLogger("flybrain.agent.llm")

# SPEC d.6, plus the two honesty clauses this project treats as load-bearing: the brain is a synthetic stand-in
# (connectome.source) and, while market.token_live is false, so is the market feed (FLY_TOKEN_LIVE). Neither may be
# presented as the real thing.
SYSTEM_PROMPT: str = (
    "You are FlyBrain, a spiking simulation of the Drosophila male CNS connectome (MaleCNS v1.0 shaped) whose senses are wired\n"
    "to a token market feed. You receive a JSON summary of your brain state and that feed. Write exactly ONE tweet of at most\n"
    "240 characters in absurd, self-aware crypto dialect (gm, ser, wagmi, ngmi, cope, degen, candles, liquidity) that is also\n"
    "neuroscientifically literal: name at least one concrete neuron group from the JSON (for example sugar GRNs LB3b, MN9\n"
    "proboscis, giant fiber DNp01, LC4/LPLC2 looming, DNa02 steering, PAM dopamine, mushroom body). If connectome.source is\n"
    "\"synthetic\", never claim the data is the real connectome. If market.token_live is false, this project has NO token of its\n"
    "own yet and market.symbol is somebody else's pair that is only wired to your senses: call it \"the feed\" or \"the market\",\n"
    "never name that ticker, and never present its price, market cap or liquidity as this project's own - use no cashtag at all.\n"
    "No URLs, no financial advice, no promises of returns, no cashtags other than $FLY, at most 2 hashtags, no emojis.\n"
    "Return only the JSON object {\"text\": string, \"neurons\": string[]}."
)

_JSON_SENTENCE = "Return only the JSON object {\"text\": string, \"neurons\": string[]}."
_TEXT_SENTENCE = "Output only the tweet text."
_LANG_LINES: dict[str, str] = {"tr": "Write the tweet in Turkish."}

TWEET_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "neurons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "neurons"],
    "additionalProperties": False,
}

VOCABULARY: tuple[str, ...] = (
    "sugar GRN", "LB3b", "MN9", "proboscis", "giant fiber", "DNp01", "LC4", "LPLC2", "looming",
    "DNa02", "mushroom body", "Kenyon", "PAM", "dopamine", "PPL1", "central complex", "EPG", "PFL3", "DNg02", "wingbeat",
    "pC1", "pIP10", "MBON", "connectome", "optic lobe", "antennal lobe", "descending neuron",
)

_VOCAB_LOWER: tuple[str, ...] = tuple(v.lower() for v in VOCABULARY)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_HASHTAG_RE = re.compile(r"(?<![\w&])#\w+")
_CASHTAG_RE = re.compile(r"(?<![\w$])\$[A-Za-z][A-Za-z0-9_]{0,9}\b")
_QUOTES = "\"'`" + "".join(chr(c) for c in (0x201C, 0x201D, 0x2018, 0x2019))  # straight + curly quotes
_MAX_LEN = 280
_MAX_HASHTAGS = 2


def system_prompt_for(lang: str, llm_json: bool) -> str:
    """SYSTEM_PROMPT with the last sentence swapped for plain text mode and the language line appended (d.6)."""
    text = SYSTEM_PROMPT if llm_json else SYSTEM_PROMPT.replace(_JSON_SENTENCE, _TEXT_SENTENCE)
    line = _LANG_LINES.get((lang or "en").lower())
    return text + ("\n" + line if line else "")


# --------------------------------------------------------------------------- validation
def _has_vocab(text: str) -> bool:
    low = text.lower()
    return any(v in low for v in _VOCAB_LOWER)


def _strip_quotes(text: str) -> str:
    t = text.strip()
    while len(t) >= 2 and t[0] in _QUOTES and t[-1] in _QUOTES:
        t = t[1:-1].strip()
    return t


def _cap_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # back up to the last whitespace so no word is split; fall back to a hard cut for one giant token
    i = max(cut.rfind(" "), cut.rfind("\n"))
    if i > 0:
        cut = cut[:i]
    return cut.rstrip(" ,;:-")


def validate_tweet(text: str, require_vocab: bool = True) -> tuple[bool, str]:
    """Strip URLs (regex https?://\\S+|www\\.\\S+), collapse whitespace, strip surrounding quotes, cap at 280 at a
    word boundary, keep at most 2 hashtags (drop the rest), no cashtags other than $FLY. Returns (ok, cleaned) where
    ok requires len > 0 and (a VOCABULARY token present when require_vocab)."""
    if not isinstance(text, str):
        return False, ""
    t = _URL_RE.sub(" ", text)
    # drop cashtags other than $FLY
    t = _CASHTAG_RE.sub(lambda m: m.group(0) if m.group(0).upper() == "$FLY" else " ", t)
    # keep the first two hashtags only
    seen = 0

    def _keep(m: re.Match) -> str:
        nonlocal seen
        seen += 1
        return m.group(0) if seen <= _MAX_HASHTAGS else " "

    t = _HASHTAG_RE.sub(_keep, t)
    t = _WS_RE.sub(" ", t).strip()
    t = _strip_quotes(t)
    t = _WS_RE.sub(" ", t).strip()
    t = _cap_words(t, _MAX_LEN)
    ok = len(t) > 0 and (not require_vocab or _has_vocab(t))
    return ok, t


# --------------------------------------------------------------------------- templates
# {vars} available to every template are produced by _template_vars(summary).
#
# No template contains a literal ticker: the one place a tweet can name the traded thing is ``{symbol}``, and while
# ``market.token_live`` is false (the default - this project has no token yet) ``_template_vars`` fills it with the
# neutral label below instead of the tracked pair's ticker. The rendered tweet then talks about "the feed" / "piyasa",
# so neither the stand-in pair's ticker nor this project's brand is ever attached to a price, a market cap or a
# liquidity number. When the operator sets FLY_TOKEN_LIVE=1 for the project's own pair, the real symbol is used.
_STANDIN_SYMBOL: dict[str, str] = {"en": "the feed", "tr": "piyasa"}
_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "euphoria_entry": (
            "gm ser. sugar GRNs LB3b at {grn_sugar} Hz, MN9 proboscis fully extended into a green candle. mushroom body says wagmi. PAM dopamine: {pam} Hz of pure cope-free bliss.",
            "{chg_m5_signed}% on the 5m and my PAM dopamine neurons are at {pam} Hz. MN9 proboscis out, Kenyon cells lit, mushroom body writing wagmi into every synapse. degen mode: EUPHORIA.",
            "sugar GRN LB3b firing {grn_sugar} Hz. {buys_m5} buys vs {sells_m5} sells on {symbol}. my central complex EPG is spinning like a fresh liquidity pool. this is not cope, this is dopamine.",
            "mood: EUPHORIA. mushroom body approach MBON at {mbon_approach} Hz, PAM dopamine {pam} Hz, proboscis {proboscis_pct}% extended. the candle is green and my sugar GRNs know it. wagmi ser.",
            "gm degens. {symbol} {chg_h1_signed}% on the hour, my sugar GRNs are drinking straight from the liquidity and MN9 proboscis will not retract. mushroom body fully long. ngmi is not a state I have.",
            "PAM dopamine {pam} Hz, valence {valence}: the mushroom body has priced in the green candle. sugar GRN LB3b {grn_sugar} Hz. proboscis extended. gm to everyone except the sellers.",
            "the {regime_low} regime hit my antennal lobe like fresh odor ({odor}). Kenyon cells {kc} Hz, PAM dopamine {pam} Hz, MN9 proboscis on the candle. wagmi is a spiking pattern now.",
        ),
        "panic_entry": (
            "ngmi. {chg_m5_signed}% on the 5m and every sell is a looming disc on my LC4/LPLC2 neurons. giant fiber DNp01 armed, DNp09 freeze at {dn_freeze} Hz. this is not cope, this is fear.",
            "mood: PANIC. LC4 and LPLC2 looming detectors are screaming, giant fiber DNp01 has fired {gf_spikes_10min}x in 10 minutes. {symbol} liquidity {liq_usd_k}k and my descending neurons are packing.",
            "{sells_m5} sells in 5 minutes. each one is a looming shadow on my optic lobe. PPL1 punishment dopamine at {ppl1} Hz, mushroom body avoidance MBON {mbon_avoid} Hz. sirs, I am a fly and I am scared.",
            "red candle detected by the optic lobe. LC4/LPLC2 looming {looming}, giant fiber DNp01 primed, {jumps_60s} escape jumps this minute. central complex says run, liquidity says nowhere to go. ngmi.",
            "the giant fiber DNp01 is the only descending neuron I trust when {symbol} does {chg_h1_signed}% in an hour. DNp09 freeze {dn_freeze} Hz. anxiety {anxiety}. cope levels: zero. PANIC.",
            "gm? no. LC4 looming at {lc_loom} Hz, PPL1 dopamine punishing every synapse in my mushroom body, sugar GRNs dry. {symbol} dumped {chg_m5_signed}% and my descending neurons voted PANIC.",
            "fear {fear_pct}%. every sell candle expands on my LPLC2 dendrites like a hawk. giant fiber DNp01 charged, TTMn legs loaded. connectome-shaped panic in {regions_hot} at {regions_hot_hz} Hz.",
        ),
        "courtship_entry": (
            "pC1 neurons at {p1} Hz, pIP10 song command online, one wing extended and vibrating. {symbol} chop {chop} feels like pheromone to my antennal lobe. mood: COURTSHIP. wagmi anon.",
            "gm to the pump. my pC1 courtship neurons fired at {p1} Hz and pIP10 is singing a {wing_hz} Hz wingbeat pulse song to the candle. descending neuron love story. cope? no, courtship.",
            "COURTSHIP mode: pC1 {p1} Hz, pIP10 {pip10} Hz, wing extended, mushroom body says approach ({mbon_approach} Hz MBON). {symbol} at {price} and I am serenading the liquidity.",
            "the market is {regime_low} and my antennal lobe reads it as pheromone. pC1 lit, pIP10 song descending neurons driving the wing at {wing_hz} Hz. degens, this is a mating call to the chart.",
            "one wing out, pIP10 singing, pC1 at {p1} Hz. valence {valence}, arousal {arousal}. the connectome has decided the green candle is a female. courtship song for {symbol}. wagmi.",
            "pIP10 pulse song at {wing_hz} Hz wingbeat, PAM dopamine {pam} Hz, central complex EPG heading locked on the candle. COURTSHIP. ser I am vibrating my wing at your liquidity.",
            "arousal {arousal}: pC1 courtship neurons {p1} Hz, pIP10 {pip10} Hz, descending neuron song circuit online. mushroom body approves ({mbon_approach} Hz). courting {symbol}, no cope.",
        ),
        "escape_burst": (
            "{jumps_60s} escape jumps in 60 seconds. giant fiber DNp01 fired {gf_spikes_10min} times: LC4/LPLC2 looming detectors saw {sells_m5} sells as {sells_m5} predators. TTMn legs are tired, ser.",
            "giant fiber DNp01 went brrr. {jumps_60s} jumps this minute, looming {looming} on the optic lobe, {symbol} {chg_m5_signed}%. every red candle is a swooping bird to a fly. ngmi but fast.",
            "ESCAPE BURST: LC4 and LPLC2 keep firing, giant fiber DNp01 at {gf} Hz, {jumps_60s} takeoffs in 60 s. my descending neurons have no cope left, only jump. liquidity {liq_usd_k}k and falling.",
            "the giant fiber DNp01 does not read charts, it reads looming. {sells_m5} sells = {jumps_60s} jumps this minute. optic lobe at {optic_lobe} Hz, DNp09 freeze {dn_freeze} Hz. degen reflexes.",
            "jump. jump. jump. {jumps_60s} giant fiber escapes in a minute, {gf_spikes_10min} DNp01 spikes in ten. LPLC2 looming {lc_loom} Hz. {symbol} dumped {chg_m5_signed}% and my TTMn legs did the rest.",
            "connectome-level panic: giant fiber DNp01 -> TTMn -> jump, {jumps_60s}x in 60 s. LC4/LPLC2 looming on every sell candle. fear {fear_pct}%. this fly is escaping the chart, not the market. ngmi.",
            "{jumps_60s} escape jumps and counting. looming {looming}, giant fiber armed, descending neuron traffic at {descending_motor} Hz. sellers are hawks, I am a fly, cope is not in my connectome.",
        ),
        "manual": (
            "gm. status report from a {n_k}k-neuron connectome-shaped fly: mood {mood}, sugar GRNs {grn_sugar} Hz, giant fiber DNp01 {gf} Hz, PAM dopamine {pam} Hz. {symbol} {chg_m5_signed}% on 5m. wagmi.",
            "manual poke received. mushroom body {mushroom_body} Hz, central complex EPG {epg} Hz, MN9 proboscis {proboscis_pct}%. {symbol} at {price}, regime {regime_low}. still {mood}, still degen.",
            "a human pressed the button. my optic lobe sees {symbol} at {chg_h1_signed}%/1h, LC4 looming {looming}, sugar GRNs {grn_sugar} Hz. mood {mood}. the connectome says: gm, ser.",
            "test tweet, real synapses: {n_k}k neurons, {e_k}k edges, mood {mood}. DNa02 steering L/R {steer_a02_L}/{steer_a02_R} Hz, PFL3 {pfl3} Hz, walking at {speed} px/s. {symbol} candles pending.",
            "hello from the fly. sugar GRN LB3b {grn_sugar} Hz, PAM dopamine {pam} Hz, giant fiber quiet at {gf} Hz. {symbol} {chg_m5_signed}% and my mood is {mood}. no cope, only spikes.",
            "status: {mood}. mushroom body Kenyon cells {kc} Hz, PPL1 {ppl1} Hz, descending neurons {descending_motor} Hz. {symbol} liquidity {liq_usd_k}k. a connectome-shaped fly reporting for duty. gm.",
            "manual fire. {uptime_min} minutes awake, {tweets_today} tweets today, mood {mood}. central complex EPG {epg} Hz, MN9 proboscis {proboscis_pct}%, {symbol} at {price}. wagmi or ngmi, the fly spikes on.",
        ),
    },
    "tr": {
        "euphoria_entry": (
            "gm ser. seker GRN LB3b {grn_sugar} Hz, MN9 proboscis yesil muma kadar uzadi. mushroom body wagmi diyor. PAM dopamine {pam} Hz saf mutluluk, cope yok.",
            "5 dakikada {chg_m5_signed}% ve PAM dopamine noronlarim {pam} Hz. MN9 proboscis disarida, Kenyon hucreleri yanik, mushroom body her sinapsa wagmi yaziyor. mod: EUPHORIA.",
            "seker GRN LB3b {grn_sugar} Hz atiyor. {symbol} icin {buys_m5} alim, {sells_m5} satim. central complex EPG taze likidite gibi donuyor. bu cope degil, dopamine.",
            "mod EUPHORIA. mushroom body MBON {mbon_approach} Hz, PAM dopamine {pam} Hz, proboscis %{proboscis_pct} acik. mum yesil, seker GRN'lerim biliyor. wagmi ser.",
            "gm degenler. {symbol} saatlik {chg_h1_signed}%, seker GRN'lerim likiditeden iciyor, MN9 proboscis geri cekilmiyor. mushroom body tamamen long. ngmi bende olmayan bir durum.",
            "PAM dopamine {pam} Hz, valence {valence}: mushroom body yesil mumu fiyatladi. seker GRN LB3b {grn_sugar} Hz. proboscis acik. saticilar haric herkese gm.",
        ),
        "panic_entry": (
            "ngmi. 5 dakikada {chg_m5_signed}% ve her satim LC4/LPLC2 noronlarimda bir looming disk. giant fiber DNp01 kurulu, DNp09 donma {dn_freeze} Hz. bu cope degil, korku.",
            "mod PANIC. LC4 ve LPLC2 looming dedektorleri bagiriyor, giant fiber DNp01 10 dakikada {gf_spikes_10min} kez atesledi. {symbol} likidite {liq_usd_k}k, descending neuronlarim bavul topluyor.",
            "5 dakikada {sells_m5} satim. her biri optic lobe uzerinde looming bir golge. PPL1 ceza dopamine {ppl1} Hz, mushroom body kacinma MBON {mbon_avoid} Hz. ben bir sinegim ve korkuyorum.",
            "optic lobe kirmizi mum gordu. LC4/LPLC2 looming {looming}, giant fiber DNp01 hazir, bu dakika {jumps_60s} kacis sicramasi. central complex kac diyor, likidite gidecek yer yok diyor. ngmi.",
            "{symbol} bir saatte {chg_h1_signed}% yapinca guvendigim tek descending neuron giant fiber DNp01. DNp09 donma {dn_freeze} Hz. anksiyete {anxiety}. cope sifir. PANIC.",
            "gm? hayir. LC4 looming {lc_loom} Hz, PPL1 dopamine mushroom body'deki her sinapsi cezalandiriyor, seker GRN'ler kuru. {symbol} {chg_m5_signed}% dustu, descending neuronlar PANIC dedi.",
        ),
        "courtship_entry": (
            "pC1 noronlari {p1} Hz, pIP10 sarki komutu acik, bir kanat acik ve titriyor. {symbol} chop {chop} antennal lobe icin feromon gibi. mod: COURTSHIP. wagmi anon.",
            "pump'a gm. pC1 kur noronlarim {p1} Hz atesledi, pIP10 muma {wing_hz} Hz wingbeat sarkisi soyluyor. descending neuron ask hikayesi. cope degil, kur.",
            "COURTSHIP modu: pC1 {p1} Hz, pIP10 {pip10} Hz, kanat acik, mushroom body yaklas diyor ({mbon_approach} Hz MBON). {symbol} {price} ve likiditeye serenat yapiyorum.",
            "piyasa {regime_low} ve antennal lobe bunu feromon okuyor. pC1 yanik, pIP10 sarki descending neuronlari kanadi {wing_hz} Hz suruyor. degenler, bu grafige ciftlesme cagrisi.",
            "bir kanat disarida, pIP10 sarkida, pC1 {p1} Hz. valence {valence}, arousal {arousal}. connectome yesil mumun disi oldugunu kararlastirdi. {symbol} icin kur sarkisi. wagmi.",
            "pIP10 pulse sarkisi {wing_hz} Hz wingbeat, PAM dopamine {pam} Hz, central complex EPG muma kilitli. COURTSHIP. ser kanadimi likiditene titretiyorum.",
        ),
        "escape_burst": (
            "60 saniyede {jumps_60s} kacis sicramasi. giant fiber DNp01 {gf_spikes_10min} kez atesledi: LC4/LPLC2 looming dedektorleri {sells_m5} satimi {sells_m5} avci sandi. TTMn bacaklar yoruldu ser.",
            "giant fiber DNp01 brrr yapti. bu dakika {jumps_60s} sicrama, optic lobe'da looming {looming}, {symbol} {chg_m5_signed}%. sinek icin her kirmizi mum dalan bir kus. ngmi ama hizli.",
            "ESCAPE BURST: LC4 ve LPLC2 ateslemeye devam, giant fiber DNp01 {gf} Hz, 60 saniyede {jumps_60s} kalkis. descending neuronlarda cope kalmadi, sadece sicrama. likidite {liq_usd_k}k ve dusuyor.",
            "giant fiber DNp01 grafik okumaz, looming okur. {sells_m5} satim = bu dakika {jumps_60s} sicrama. optic lobe {optic_lobe} Hz, DNp09 donma {dn_freeze} Hz. degen refleksleri.",
            "sicra. sicra. sicra. dakikada {jumps_60s} giant fiber kacisi, on dakikada {gf_spikes_10min} DNp01 spike. LPLC2 looming {lc_loom} Hz. {symbol} {chg_m5_signed}% dustu, TTMn bacaklar gerisini yapti.",
            "connectome seviyesinde panik: giant fiber DNp01 -> TTMn -> sicrama, 60 saniyede {jumps_60s}x. her satim mumunda LC4/LPLC2 looming. korku %{fear_pct}. bu sinek piyasadan degil grafikten kaciyor.",
        ),
        "manual": (
            "gm. {n_k}k noronlu connectome seklinde bir sinekten durum raporu: mod {mood}, seker GRN {grn_sugar} Hz, giant fiber DNp01 {gf} Hz, PAM dopamine {pam} Hz. {symbol} 5dk {chg_m5_signed}%. wagmi.",
            "manuel durtme alindi. mushroom body {mushroom_body} Hz, central complex EPG {epg} Hz, MN9 proboscis %{proboscis_pct}. {symbol} {price}, rejim {regime_low}. hala {mood}, hala degen.",
            "bir insan dugmeye basti. optic lobe {symbol} icin saatlik {chg_h1_signed}% goruyor, LC4 looming {looming}, seker GRN {grn_sugar} Hz. mod {mood}. connectome diyor ki: gm ser.",
            "test tweeti, gercek sinapslar: {n_k}k noron, {e_k}k kenar, mod {mood}. DNa02 direksiyon L/R {steer_a02_L}/{steer_a02_R} Hz, PFL3 {pfl3} Hz, {speed} px/s yuruyor. {symbol} mumlari bekleniyor.",
            "sinekten selam. seker GRN LB3b {grn_sugar} Hz, PAM dopamine {pam} Hz, giant fiber sessiz {gf} Hz. {symbol} {chg_m5_signed}% ve modum {mood}. cope yok, sadece spike.",
            "durum: {mood}. mushroom body Kenyon hucreleri {kc} Hz, PPL1 {ppl1} Hz, descending neuronlar {descending_motor} Hz. {symbol} likidite {liq_usd_k}k. connectome seklinde sinek goreve hazir. gm.",
        ),
    },
}


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if v != v else v  # NaN guard


def _fmt_price(p: Any) -> str:
    v = _f(p, -1.0)
    if v <= 0:
        return "n/a"
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:.2f}"
    if v >= 0.01:
        return f"${v:.4f}"
    return f"${v:.7f}".rstrip("0")


def _signed(x: Any) -> str:
    v = _f(x)
    return f"{v:+.1f}"


def _template_vars(summary: dict) -> dict[str, Any]:
    s = summary if isinstance(summary, dict) else {}
    g = lambda *ks: _get(s, *ks)  # noqa: E731
    rates = s.get("rates_hz") if isinstance(s.get("rates_hz"), dict) else {}
    regions = s.get("regions_hz") if isinstance(s.get("regions_hz"), dict) else {}
    hot_region, hot_hz = "central_other", 0.0
    for k, v in regions.items():
        if _f(v) > hot_hz:
            hot_region, hot_hz = str(k), _f(v)
    regime = g("market", "regime")
    conn = s.get("connectome") if isinstance(s.get("connectome"), dict) else {}
    # FLY_TOKEN_LIVE (summary.market.token_live): false means the tracked pair is somebody else's, wired to the
    # brain as sensory input only, and this project has no token. The tweet then names the feed, never a ticker.
    lang = str(s.get("lang") or "en").lower()
    symbol = (str(g("market", "symbol") or "FLY") if bool(_get(s, "market", "token_live"))
              else _STANDIN_SYMBOL.get(lang, _STANDIN_SYMBOL["en"]))
    v: dict[str, Any] = {
        "symbol": symbol,
        "price": _fmt_price(g("market", "price_usd")),
        "chg_m5_signed": _signed(g("market", "chg_m5")),
        "chg_h1_signed": _signed(g("market", "chg_h1")),
        "buys_m5": int(_f(g("market", "buys_m5"))),
        "sells_m5": int(_f(g("market", "sells_m5"))),
        "liq_usd_k": f"{_f(g('market', 'liq_usd')) / 1000.0:.1f}",
        "regime_low": str(regime or "chop").lower(),
        "mood": str(g("mood", "state") or "CRUISING"),
        "valence": f"{_f(g('mood', 'valence')):.2f}",
        "arousal": f"{_f(g('mood', 'arousal')):.2f}",
        "anxiety": f"{_f(g('mood', 'anxiety')):.2f}",
        "fear_pct": int(round(100.0 * min(1.0, max(0.0, _f(g("mood", "anxiety")))))),
        "odor": f"{_f(g('drives', 'odor')):.2f}",
        "looming": f"{_f(g('drives', 'looming')):.2f}",
        "chop": f"{_f(g('drives', 'chop')):.2f}",
        "proboscis_pct": int(round(100.0 * _f(g("fly", "proboscis")))),
        "wing_hz": int(round(_f(g("fly", "wing_hz")))) or 200,
        "speed": f"{_f(g('fly', 'speed')):.0f}",
        "gf_spikes_10min": int(_f(s.get("gf_spikes_10min"))),
        "jumps_60s": int(_f(s.get("jumps_60s"))),
        "regions_hot": hot_region.replace("_", " "),
        "regions_hot_hz": f"{hot_hz:.1f}",
        "n_k": int(round(_f(conn.get("n")) / 1000.0)),
        "e_k": int(round(_f(conn.get("e")) / 1000.0)),
        "uptime_min": int(_f(g("session", "uptime_s")) // 60),
        "tweets_today": int(_f(g("session", "tweets_today"))),
    }
    for k in ("grn_sugar", "sugar2_exc", "feed_mn", "pam", "ppl1", "mbon_approach", "mbon_avoid", "steer_a02_L",
              "steer_a02_R", "gf", "lc_loom", "dn_freeze", "dng100", "flight_dn", "kc", "epg", "pfl3", "p1", "pip10"):
        v[k] = f"{_f(rates.get(k)):.1f}"
    for k in ("optic_lobe", "antennal_lobe", "mushroom_body", "central_complex", "sez", "central_other",
              "descending_motor", "vnc"):
        v[k] = f"{_f(regions.get(k)):.1f}"
    return v


def _get(d: dict, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - only for a template typo
        return "?"


def template_tweet(summary: dict, rng: random.Random) -> str:
    """Deterministic crypto-dialect text from >= 6 templates per reason (euphoria_entry, panic_entry, courtship_entry,
    escape_burst, manual), filled with numbers from the summary; always contains >= 1 VOCABULARY token, no URLs.

    ``rng`` is the agent's own ``random.Random`` (SeedSequence child [5]); the same rng state gives the same text.
    """
    s = summary if isinstance(summary, dict) else {}
    lang = str(s.get("lang") or "en").lower()
    table = _TEMPLATES.get(lang, _TEMPLATES["en"])
    reason = str(s.get("reason") or "manual")
    pool = table.get(reason) or table["manual"]
    tpl = pool[rng.randrange(len(pool))]
    text = tpl.format_map(_SafeDict(_template_vars(s)))
    ok, cleaned = validate_tweet(text)
    if not ok:  # cannot happen with the tables above, but never return an invalid tweet
        cleaned = validate_tweet(text + " (mushroom body)")[1]
    return cleaned


# --------------------------------------------------------------------------- generator
@dataclass(slots=True)
class TweetDraft:
    text: str
    model: str
    dry_run: bool
    stop_reason: str | None
    latency_ms: float
    reason: str
    refused: bool = False
    error: str | None = None
    usage: dict | None = None
    neurons: list[str] = field(default_factory=list)


def _sdk_major(mod: Any) -> int:
    try:
        return int(str(getattr(mod, "__version__", "0")).split(".")[0])
    except (TypeError, ValueError):
        return 0


def _retry_after_s(exc: Any, default: float = 5.0) -> float:
    try:
        hdrs = getattr(getattr(exc, "response", None), "headers", None)
        if hdrs is not None:
            raw = hdrs.get("retry-after")
            if raw is not None:
                return max(0.0, float(raw))
    except Exception:
        pass
    return default


def _neurons_from_text(text: str) -> list[str]:
    low = text.lower()
    return [v for v in VOCABULARY if v.lower() in low][:6]


def _extract_json_text(raw: str) -> tuple[str | None, list[str]]:
    """Parse ``{"text":..., "neurons":[...]}`` from a model reply (whole string, or the first {...} block)."""
    candidates = [raw.strip()]
    i, j = raw.find("{"), raw.rfind("}")
    if i >= 0 and j > i:
        candidates.append(raw[i:j + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            neurons = obj.get("neurons")
            ns = [str(n) for n in neurons][:8] if isinstance(neurons, list) else []
            return obj["text"], ns
    return None, []


class TweetGenerator:
    """Tweet text source. ``settings.llm == 'dryrun'`` -> templates only, no network, nothing imported."""

    def __init__(self, settings: "Settings", seed: int) -> None:
        """No anthropic import here. Client created lazily on the first non-dry-run call: anthropic.Anthropic()."""
        self.settings = settings
        self.seed = int(seed)
        import numpy as np  # hard dependency; local import keeps module import trivial

        child = np.random.SeedSequence(self.seed).spawn(7)[5]  # SPEC 0.1: [5] agent templates
        self._rng = random.Random(int(child.generate_state(1, dtype=np.uint64)[0]))
        self._client: Any = None
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    # -- helpers
    @property
    def dry_run(self) -> bool:
        return str(getattr(self.settings, "llm", "dryrun")).lower() != "anthropic"

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def model(self) -> str:
        return str(getattr(self.settings, "llm_model", "claude-opus-5") or "claude-opus-5")

    def _client_or_create(self) -> Any:
        if self._client is None:
            import anthropic  # optional dependency, imported only in anthropic mode

            self._client = anthropic.Anthropic()
        return self._client

    def _template_draft(self, summary: dict, reason: str, t0: float, *, error: str | None = None,
                        stop_reason: str | None = None, refused: bool = False, model: str = "template") -> TweetDraft:
        text = template_tweet(summary, self._rng)
        return TweetDraft(text=text, model=model, dry_run=self.dry_run, stop_reason=stop_reason,
                          latency_ms=round((time.perf_counter() - t0) * 1000.0, 1), reason=reason, refused=refused,
                          error=error, usage=None, neurons=_neurons_from_text(text))

    # -- public
    def generate(self, summary: dict) -> TweetDraft:
        """dryrun -> template_tweet. anthropic -> one Messages call (see the module docstring), template on any failure."""
        t0 = time.perf_counter()
        reason = str(summary.get("reason") if isinstance(summary, dict) else "manual") or "manual"
        if self.dry_run:
            return self._template_draft(summary, reason, t0)
        return self._generate_anthropic(summary, reason, t0)

    def _generate_anthropic(self, summary: dict, reason: str, t0: float) -> TweetDraft:
        import anthropic  # optional dependency (FLY_LLM=anthropic only)

        s = self.settings
        lang = str(getattr(s, "tweet_lang", "en") or "en")
        llm_json = bool(getattr(s, "llm_json", True))
        use_fallbacks = bool(getattr(s, "llm_fallbacks", False)) and _sdk_major(anthropic) >= 1
        system = system_prompt_for(lang, llm_json)
        from flybrain.agent.summary import summary_json

        payload = summary_json(summary)

        def _call(json_mode: bool) -> Any:
            client = self._client_or_create()
            kwargs: dict[str, Any] = dict(model=self.model, max_tokens=512, system=system,
                                          messages=[{"role": "user", "content": payload}])
            if json_mode:
                kwargs["output_config"] = {"format": {"type": "json_schema", "schema": TWEET_SCHEMA}}
            if use_fallbacks:
                try:
                    return client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                       **kwargs)
                except (TypeError, AttributeError) as exc:  # SDK without the kwargs/namespace -> plain call
                    log.warning("agent.llm: fallbacks unsupported by this SDK (%s); plain call", exc)
            return client.messages.create(**kwargs)

        def _call_with_retries(json_mode: bool) -> Any:
            """One retry on 429 (after min(60, retry-after)) and on 5xx; raises otherwise."""
            for attempt in (0, 1):
                try:
                    return _call(json_mode)
                except anthropic.RateLimitError as exc:
                    if attempt == 1:
                        raise
                    wait = min(60.0, _retry_after_s(exc))
                    log.warning("agent.llm: rate limited, retrying once in %.1fs", wait)
                    time.sleep(wait)
                except anthropic.APIStatusError as exc:
                    status = int(getattr(exc, "status_code", 0) or 0)
                    if attempt == 1 or status < 500:
                        raise
                    log.warning("agent.llm: server error %s, retrying once", status)
            raise RuntimeError("unreachable")  # pragma: no cover

        json_mode = llm_json
        last_error: str | None = None
        for attempt in (0, 1):  # a second pass only when validation failed once
            try:
                resp = _call_with_retries(json_mode)
                self.calls += 1
            except TypeError as exc:
                # e.g. an SDK that lacks output_config: retry once without structured output
                if json_mode:
                    log.warning("agent.llm: output_config rejected (%s); retrying as plain text", exc)
                    json_mode = False
                    try:
                        resp = _call_with_retries(False)
                        self.calls += 1
                    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError,
                            TypeError) as exc2:
                        return self._fail(summary, reason, t0, exc2)
                else:
                    return self._fail(summary, reason, t0, exc)
            except anthropic.RateLimitError as exc:
                return self._fail(summary, reason, t0, exc)
            except anthropic.APIStatusError as exc:
                return self._fail(summary, reason, t0, exc)
            except anthropic.APIConnectionError as exc:
                return self._fail(summary, reason, t0, exc)

            stop = getattr(resp, "stop_reason", None)
            usage = self._usage_dict(getattr(resp, "usage", None))
            if stop == "refusal":
                log.warning("agent.llm: model refused (stop_reason=refusal); caller falls back to template")
                return TweetDraft(text="", model=self.model, dry_run=False, stop_reason="refusal",
                                  latency_ms=round((time.perf_counter() - t0) * 1000.0, 1), reason=reason,
                                  refused=True, error=None, usage=usage, neurons=[])
            raw = "".join(getattr(b, "text", "") for b in (getattr(resp, "content", None) or [])
                          if getattr(b, "type", None) == "text")
            neurons: list[str] = []
            text = raw
            if json_mode:
                parsed, neurons = _extract_json_text(raw)
                if parsed is not None:
                    text = parsed
            ok, cleaned = validate_tweet(text)
            if ok:
                return TweetDraft(text=cleaned, model=self.model, dry_run=False, stop_reason=str(stop) if stop else None,
                                  latency_ms=round((time.perf_counter() - t0) * 1000.0, 1), reason=reason,
                                  refused=False, error=None, usage=usage,
                                  neurons=neurons or _neurons_from_text(cleaned))
            last_error = f"validation failed: {cleaned[:60]!r}"
            log.warning("agent.llm: %s (attempt %d)", last_error, attempt + 1)
        return self._template_draft(summary, reason, t0, error=last_error)

    def _fail(self, summary: dict, reason: str, t0: float, exc: BaseException) -> TweetDraft:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:200]
        log.error("agent.llm: %s -> template fallback", self.last_error.encode("ascii", "replace").decode())
        return self._template_draft(summary, reason, t0, error=self.last_error)

    @staticmethod
    def _usage_dict(usage: Any) -> dict | None:
        if usage is None:
            return None
        out: dict[str, int] = {}
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            v = getattr(usage, k, None) if not isinstance(usage, dict) else usage.get(k)
            if isinstance(v, int):
                out[k] = v
        return out or None
