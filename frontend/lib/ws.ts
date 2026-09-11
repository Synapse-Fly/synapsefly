// SPEC section e.2: useFlySocket - native WebSocket with reconnect/backoff, heartbeat, a typed event bus and the
// mutable tick store. SSR-safe (nothing runs on the server). No React state update per tick.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createTickStore, type TickStore } from "./store";
import type { ClientMsg, HelloMsg, ServerMsg } from "./types";

export type WsStatus = "connecting"|"open"|"reconnecting"|"closed";
export interface FlySocket {
  status: WsStatus; attempt: number; latencyMs: number|null;
  hello: HelloMsg|null;                 // React state (changes rarely)
  store: TickStore;                     // mutable, read in rAF loops (e.3)
  send(m: ClientMsg): boolean;          // false when not open (message dropped, never queued)
  on<K extends ServerMsg["type"]>(type: K, h: (m: Extract<ServerMsg, {type: K}>) => void): () => void;   // event bus, unsubscribe fn
}

type AnyHandler = (m: ServerMsg) => void;
interface Meta { status: WsStatus; attempt: number; latencyMs: number|null; hello: HelloMsg|null }

const PING_MS = 10_000;
const DEAD_MS = 15_000;
const WATCHDOG_MS = 2_500;

export const DEFAULT_WS_URL: string = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:4000/ws";

export function useFlySocket(url: string = DEFAULT_WS_URL): FlySocket {
  const [store] = useState<TickStore>(createTickStore);
  const [bus] = useState(() => new Map<string, Set<AnyHandler>>());
  const [meta, setMeta] = useState<Meta>({ status: "connecting", attempt: 0, latencyMs: null, hello: null });
  const wsRef = useRef<WebSocket|null>(null);

  const send = useCallback((m: ClientMsg): boolean => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    try {
      ws.send(JSON.stringify(m));
      return true;
    } catch (e) {
      console.warn("[ws] send failed", e);
      return false;
    }
  }, []);

  const on = useCallback(<K extends ServerMsg["type"]>(type: K, h: (m: Extract<ServerMsg, {type: K}>) => void): (() => void) => {
    let set = bus.get(type);
    if (!set) { set = new Set(); bus.set(type, set); }
    const fn = h as unknown as AnyHandler;
    set.add(fn);
    return () => { set?.delete(fn); };
  }, [bus]);

  useEffect(() => {
    if (typeof window === "undefined" || typeof WebSocket === "undefined") {
      // No WebSocket in this runtime (SSR, or a degenerate environment). Effects only run on the client, so this
      // branch is defensive; defer the status update to a microtask so we never call setState synchronously inside
      // the effect body (react-hooks/set-state-in-effect).
      let cancelled = false;
      queueMicrotask(() => { if (!cancelled) setMeta((p) => (p.status === "closed" ? p : { ...p, status: "closed" })); });
      return () => { cancelled = true; };
    }
    let closed = false;
    let attempt = 0;
    let ws: WebSocket|null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>|null = null;
    let pingTimer: ReturnType<typeof setInterval>|null = null;
    let watchdog: ReturnType<typeof setInterval>|null = null;
    let lastActivity = performance.now();

    const clearTimers = () => {
      if (pingTimer !== null) { clearInterval(pingTimer); pingTimer = null; }
      if (watchdog !== null) { clearInterval(watchdog); watchdog = null; }
    };

    const emit = (m: ServerMsg) => {
      const set = bus.get(m.type);
      if (!set) return;
      for (const h of set) {
        try { h(m); } catch (e) { console.error(`[ws] handler for ${m.type} failed`, e); }
      }
    };

    const dispatch = (m: ServerMsg) => {
      switch (m.type) {
        case "hello":
          attempt = 0;
          lastActivity = performance.now();
          store.setHello(m);
          setMeta((p) => ({ ...p, status: "open", attempt: 0, hello: m }));
          break;
        case "tick":
          lastActivity = performance.now();                 // only `tick`/`pong`/`hello` prove the server is alive
          store.push(m, performance.now());
          break;
        case "mood_change":
          document.title = "FlyBrain - " + m.to;
          break;
        case "pong": {
          lastActivity = performance.now();
          const rtt = Math.max(0, Math.round(Date.now() - m.t * 1000));
          setMeta((p) => (p.latencyMs === rtt ? p : { ...p, latencyMs: rtt }));
          break;
        }
        case "error":
          console.warn("[ws] server error", m.code, m.msg);
          break;
        default:
          break;
      }
      emit(m);
    };

    const scheduleReconnect = () => {
      if (closed || reconnectTimer !== null) return;
      const delay = Math.min(8000, 500 * 2 ** attempt) + Math.random() * 250;
      attempt += 1;
      const n = attempt;
      setMeta((p) => ({ ...p, status: "reconnecting", attempt: n }));
      reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, delay);
    };

    const connect = () => {
      if (closed) return;
      let sock: WebSocket;
      try {
        sock = new WebSocket(url);
      } catch (e) {
        console.warn("[ws] cannot open", url, e);
        scheduleReconnect();
        return;
      }
      ws = sock;
      wsRef.current = sock;
      lastActivity = performance.now();
      sock.onopen = () => {
        if (closed) { sock.close(); return; }
        lastActivity = performance.now();
        setMeta((p) => ({ ...p, status: "open" }));
        clearTimers();
        pingTimer = setInterval(() => {
          if (sock.readyState === WebSocket.OPEN) {
            try { sock.send(JSON.stringify({ type: "ping", t: Date.now() / 1000 })); } catch { /* ignored */ }
          }
        }, PING_MS);
        watchdog = setInterval(() => {
          if (performance.now() - lastActivity > DEAD_MS) {
            console.warn("[ws] no pong/tick for 15 s, reconnecting");
            try { sock.close(); } catch { /* ignored */ }
          }
        }, WATCHDOG_MS);
      };
      sock.onmessage = (ev: MessageEvent) => {
        try {
          if (typeof ev.data !== "string") return;
          const m = JSON.parse(ev.data) as ServerMsg;
          if (!m || typeof m !== "object" || typeof m.type !== "string") throw new Error("frame without type");
          // NOTE: `lastActivity` is deliberately NOT refreshed here. SPEC e.2/d.10 arm the 15 s watchdog on
          // `pong` or `tick` only; a server that keeps dribbling low-rate out-of-band frames while its tick
          // loop is dead must still be detected (dispatch() stamps the three frames that count).
          dispatch(m);
        } catch (e) {
          console.error("[ws] bad frame", e);
        }
      };
      sock.onerror = (ev) => {
        if (closed) return;                       // StrictMode dev double-mount: the first socket is closed before it opens
        console.warn("[ws] error", ev.type, url);
      };
      sock.onclose = () => {
        clearTimers();
        if (wsRef.current === sock) wsRef.current = null;
        if (closed) return;
        scheduleReconnect();
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimers();
      if (reconnectTimer !== null) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      const s = ws;
      ws = null;
      wsRef.current = null;
      if (s) {
        try { s.close(); } catch { /* ignored */ }
      }
    };
  }, [url, store, bus]);

  return useMemo<FlySocket>(() => ({
    status: meta.status, attempt: meta.attempt, latencyMs: meta.latencyMs, hello: meta.hello,
    store, send, on,
  }), [meta, store, send, on]);
}
