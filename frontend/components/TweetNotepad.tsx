"use client";
// SPEC section e.5 `TweetNotepad`: the "tweets.txt - Notepad" log. Loads getTweets(20) on mount, prepends live `tweet`
// frames (section d.4, dry-run included), keeps <= 50 (section e.8). Each entry is
// `[HH:MM:SS] <mood> (<dry-run|posted>) <text>` in 12 px monospace with a DRY RUN stamp, the neuron chips, an
// `open on X` link when a url exists and the error in red. "Generate test tweet" calls onTest (the page sends
// `tweet_test`) and is disabled for 5 s. Tweet frames are rare, so plain React state is fine here (no per-tick work).
import { useCallback, useEffect, useRef, useState } from "react";
import type { FlySocket } from "@/lib/ws";
import type { TweetMsg, TweetRecord } from "@/lib/types";
import { getTweets } from "@/lib/api";

export interface TweetNotepadProps { sock: FlySocket; onTest(): void }

const MAX_TWEETS = 50;        // section e.8: tweet frames <= 50
const LOAD_N = 20;            // section e.5: getTweets(20) on mount
const TEST_LOCK_MS = 5000;    // button disabled for 5 s after use
const NOTEPAD_MENU = ["File", "Edit", "Search", "Help"] as const;

function fmtClock(wall: number): string {
  if (!Number.isFinite(wall)) return "--:--:--";
  const d = new Date(wall * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function toRecord(m: TweetMsg): TweetRecord {
  // TweetRecord = TweetMsg without `type`/`seq` (+ optional summary)
  const { type: _type, seq: _seq, ...rest } = m;
  void _type; void _seq;
  return rest;
}

/** Newest first, de-duplicated by id (a REST record and its live frame share the id), capped at MAX_TWEETS. */
export function mergeTweets(current: readonly TweetRecord[], incoming: readonly TweetRecord[]): TweetRecord[] {
  const byId = new Map<string, TweetRecord>();
  for (const t of current) if (t && typeof t.id === "string") byId.set(t.id, t);
  for (const t of incoming) if (t && typeof t.id === "string") byId.set(t.id, t);   // live frame wins
  return Array.from(byId.values())
    .sort((a, b) => (b.wall ?? 0) - (a.wall ?? 0))
    .slice(0, MAX_TWEETS);
}

function statusWord(t: TweetRecord): string {
  if (t.posted) return "posted";
  if (t.dry_run) return "dry-run";
  return t.error ? "failed" : "not posted";
}

function TweetEntry({ t }: { t: TweetRecord }) {
  return (
    <div className="border-b border-dotted border-[#c0c0c0] py-[2px]" data-testid="tweet-entry" data-tweet-id={t.id}>
      <div className="whitespace-pre-wrap break-words">
        <span className="text-[#404040]">[{fmtClock(t.wall)}]</span>{" "}
        <span className="font-bold">{t.mood}</span>{" "}
        <span className="text-[#404040]">({statusWord(t)})</span>{" "}
        {t.text}
      </div>
      <div className="mt-[1px] flex flex-wrap items-center gap-1 text-[10px]">
        {t.dry_run ? (
          <span className="rotate-[-3deg] border-2 border-[#a80000] px-1 font-bold tracking-widest text-[#a80000]" title="FLY_X=dryrun: nothing was posted" data-testid="dry-run-stamp">
            DRY RUN
          </span>
        ) : null}
        <span className="text-[#404040]" title="text source">{t.model}</span>
        <span className="text-[#404040]" title="trigger reason">{t.reason}</span>
        {Array.isArray(t.neurons) ? t.neurons.map((n, i) => (
          <span key={`${n}-${i}`} className="border border-[#808080] bg-[#dfdfdf] px-1 leading-[12px]" title="neuron group named in the tweet">{n}</span>
        )) : null}
        {t.snapshot_source ? <span className="text-[#404040]" title="canvas snapshot source">png:{t.snapshot_source}</span> : null}
        {typeof t.latency_ms === "number" ? <span className="text-[#404040]">{t.latency_ms.toFixed(0)} ms</span> : null}
        {t.url ? (
          <a href={t.url} target="_blank" rel="noopener noreferrer" className="text-[#000080] underline" data-testid="open-on-x">open on X</a>
        ) : null}
        {t.error ? <span className="font-bold text-[#ff0000]" data-testid="tweet-error">error: {t.error}</span> : null}
      </div>
    </div>
  );
}

export function TweetNotepad({ sock, onTest }: TweetNotepadProps) {
  const hello = sock.hello;
  const on = sock.on;
  const [tweets, setTweets] = useState<TweetRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [locked, setLocked] = useState(false);
  const lockTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // initial load from REST (async -> no synchronous setState inside the effect)
  useEffect(() => {
    let alive = true;
    getTweets(LOAD_N)
      .then((rows) => {
        if (!alive) return;
        const list = Array.isArray(rows) ? rows : [];
        setTweets((cur) => mergeTweets(cur, list));
        setLoadError(null);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        console.warn("[notepad] getTweets failed", e);
        setLoadError(e instanceof Error ? e.message : String(e));
      });
    return () => { alive = false; };
  }, []);

  // live tweet frames (dry-run included)
  useEffect(() => on("tweet", (m) => {
    try {
      setTweets((cur) => mergeTweets(cur, [toRecord(m)]));
    } catch (e) {
      console.error("[notepad] tweet frame failed", e);
    }
  }), [on]);

  useEffect(() => () => { if (lockTimer.current !== null) clearTimeout(lockTimer.current); }, []);

  const onClickTest = useCallback(() => {
    if (locked) return;
    setLocked(true);
    if (lockTimer.current !== null) clearTimeout(lockTimer.current);
    lockTimer.current = setTimeout(() => { lockTimer.current = null; setLocked(false); }, TEST_LOCK_MS);
    try { onTest(); } catch (e) { console.error("[notepad] onTest failed", e); }
  }, [locked, onTest]);

  const dryCount = tweets.filter((t) => t.dry_run).length;
  const postedCount = tweets.filter((t) => t.posted).length;

  return (
    <div className="flex h-full min-h-0 flex-col bg-win-gray text-[11px]" data-testid="tweet-notepad">
      {/* Notepad menu strip (inert, retro feel) */}
      <div className="flex items-center gap-3 px-1 py-[1px] select-none" role="menubar" aria-label="Notepad menu (inert)">
        {NOTEPAD_MENU.map((label) => (
          <span key={label} className="cursor-default" role="menuitem" aria-disabled="true"><u>{label[0]}</u>{label.slice(1)}</span>
        ))}
        <span className="ml-auto text-[#404040]">
          {hello ? `LLM ${hello.agent.llm} / X ${hello.agent.x} / ${hello.agent.tweets_per_day} per day` : "no hello yet"}
        </span>
      </div>

      {/* the "text file" */}
      <div className="bevel-in min-h-0 flex-1 overflow-auto bg-white p-1 font-mono text-[12px] leading-[15px]" data-testid="tweet-list">
        {tweets.length === 0 ? (
          <div className="text-[#404040]">
            {loadError ? `tweets.txt: could not load history (${loadError}); waiting for live tweet frames` : "tweets.txt is empty - the fly has not tweeted yet. Press T or the button below for a dry-run test tweet."}
          </div>
        ) : tweets.map((t) => <TweetEntry key={t.id} t={t} />)}
      </div>

      {/* footer */}
      <div className="flex items-center gap-2 px-1 py-[2px]">
        <button type="button" className="btn95 disabled:text-[#808080]" onClick={onClickTest} disabled={locked}
          title={locked ? "please wait 5 s" : "sends tweet_test (reason manual; never posts unless FLY_X=post)"} data-testid="generate-test-tweet">
          Generate test tweet
        </button>
        <span className="text-[#404040]">{locked ? "requested..." : ""}</span>
        <span className="ml-auto text-[#404040]">{tweets.length} entries · {dryCount} dry-run · {postedCount} posted</span>
      </div>
    </div>
  );
}

export default TweetNotepad;
