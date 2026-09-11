// SPEC section e.3: mutable tick store. Ticks land here without any React state update; canvases read
// latest()/prev() inside requestAnimationFrame, panels subscribe through useSyncExternalStore at <= 4 Hz.
import { useSyncExternalStore } from "react";
import type { HelloMsg, TickEvent, TickMsg } from "./types";

export interface StoredTick { tick: TickMsg; recvAt: number }          // recvAt = performance.now()
export interface TickStore {
  hello: HelloMsg|null; setHello(h: HelloMsg): void;
  push(t: TickMsg, recvAt: number): void;                              // ring of 4; drops the oldest
  latest(): StoredTick|null; prev(): StoredTick|null;
  subscribe(fn: () => void): () => void;                               // notified at most every 250 ms (4 Hz), for useSyncExternalStore
  snapshot(): TickMsg|null;                                            // stable reference between notifications (useSyncExternalStore contract)
  events: TickEvent[];                                                  // last 200 in-tick events (for panels)
}

const RING = 4;
const NOTIFY_MS = 250;
const MAX_EVENTS = 200;

export function createTickStore(): TickStore {
  const ring: (StoredTick|null)[] = new Array<StoredTick|null>(RING).fill(null);
  let head = -1;              // index of the latest entry
  let count = 0;
  const listeners = new Set<() => void>();
  let snap: TickMsg|null = null;
  let lastNotify = -Infinity;
  let timer: ReturnType<typeof setTimeout>|null = null;

  const notify = () => {
    timer = null;
    lastNotify = performance.now();
    const l = head >= 0 ? ring[head] : null;
    snap = l ? l.tick : null;
    for (const fn of listeners) {
      try { fn(); } catch (e) { console.error("[store] listener failed", e); }
    }
  };
  const schedule = () => {
    if (timer !== null) return;
    const wait = Math.max(0, NOTIFY_MS - (performance.now() - lastNotify));
    timer = setTimeout(notify, wait);
  };

  const store: TickStore = {
    hello: null,
    events: [],
    setHello(h) {
      store.hello = h;
      schedule();
    },
    push(t, recvAt) {
      head = (head + 1) % RING;
      ring[head] = { tick: t, recvAt };
      if (count < RING) count++;
      if (t.events && t.events.length) {
        const ev = store.events.concat(t.events);
        store.events = ev.length > MAX_EVENTS ? ev.slice(ev.length - MAX_EVENTS) : ev;
      }
      schedule();
    },
    latest() {
      return head >= 0 ? ring[head] : null;
    },
    prev() {
      if (count < 2) return null;
      return ring[(head - 1 + RING) % RING];
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => { listeners.delete(fn); };
    },
    snapshot() {
      return snap;
    },
  };
  return store;
}

const serverSnapshot = (): TickMsg|null => null;

export function useTickSnapshot(store: TickStore): TickMsg|null {
  return useSyncExternalStore(store.subscribe, store.snapshot, serverSnapshot);
}
