// SPEC section e.4: REST helpers (all relative to NEXT_PUBLIC_API_URL; JSON; throw on !ok) + wire formatting helpers
// shared by the status bar and the panels (fmtPrice, fmtCompact).
import type { HealthResponse, MarketMode, PokeStim, StateResponse, TweetRecord } from "./types";

export const API_BASE: string = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000").replace(/\/+$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const res = await fetch(API_BASE + path, {
    ...init,
    headers: { accept: "application/json", ...(init?.body ? { "content-type": "application/json" } : {}), ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`${method} ${path} -> HTTP ${res.status}`);
  return (await res.json()) as T;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });
}

export function getState(): Promise<StateResponse> {
  return request<StateResponse>("/api/state");
}
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}
export function getTweets(limit: number = 20): Promise<TweetRecord[]> {
  return request<TweetRecord[]>(`/api/tweets?limit=${encodeURIComponent(String(limit))}`);
}
export function postPoke(stim: PokeStim, strength: number, side: "L"|"R"|"both", duration_ms: number): Promise<{ok: boolean}> {
  return post<{ok: boolean}>("/api/poke", { stim, strength, side, duration_ms });
}
export function postMarketMode(mode: MarketMode): Promise<{ok: boolean; mode: string}> {
  return post<{ok: boolean; mode: string}>("/api/market/mode", { mode });
}
export function postTweetTest(): Promise<TweetRecord> {
  return post<TweetRecord>("/api/tweet/test", {});
}
export function postSnapshot(id: string, png_b64: string): Promise<{ok: boolean; bytes: number}> {
  return post<{ok: boolean; bytes: number}>("/api/snapshot", { id, png_b64 });
}

const SUB = "₀₁₂₃₄₅₆₇₈₉";
function subscript(n: number): string {
  return String(n).split("").map((d) => SUB[Number(d)] ?? d).join("");
}

/** Price with subscript-zero notation for tiny prices: 0.000001234 -> "0.0<sub>5</sub>1234" (4 significant digits). */
export function fmtPrice(p: number|null|undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "-";
  if (p === 0) return "0.00";
  const abs = Math.abs(p);
  const sign = p < 0 ? "-" : "";
  if (abs >= 1000) return sign + abs.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (abs >= 1) return sign + abs.toFixed(2);
  if (abs >= 0.01) return sign + abs.toFixed(4);
  if (abs >= 0.001) return sign + abs.toFixed(6);
  const exp = abs.toExponential(3);                       // "1.234e-6"
  const [mant, e] = exp.split("e");
  const zeros = -Number(e) - 1;
  const digits = mant.replace(".", "").slice(0, 4);
  return `${sign}0.0${subscript(zeros)}${digits}`;
}

/** Compact dollar amounts: 1234500 -> "$1.23M"; null -> "-". */
export function fmtCompact(v: number|null|undefined, prefix: string = "$"): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}${prefix}${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}${prefix}${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}${prefix}${(abs / 1e3).toFixed(1)}K`;
  return `${sign}${prefix}${abs.toFixed(abs >= 100 ? 0 : 2)}`;
}
