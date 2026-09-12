# SynapseFly: a fruit fly's connectome, wired to the market, painting on your screen

There is a fruit fly living on a webpage. It has a brain — a real one, in the only sense that matters: the exact wiring. Right now that brain is reading a price feed through its taste neurons and its fear circuits, panicking at red candles the way it would panic at a hawk, dragging a cursor across a Windows 95 Paint canvas and leaving a trail, and — when the panic crosses a threshold — writing a tweet about it.

This is **SynapseFly**. No part of the sentence above is a metaphor. Here is exactly how each piece works, with the real numbers.

**Live:** https://www.synapsefly.com · **Source (open, MIT):** https://github.com/Synapse-Fly/synapsefly · **X:** https://x.com/SynapseFly

---

## 1. The brain is a connectome

In 2026, Janelia's FlyEM team released **MaleCNS v1.0**: the complete connectome of the *Drosophila melanogaster* male central nervous system. Every neuron, every synapse, traced from electron microscopy — the brain, both optic lobes, and the ventral nerve cord in one dataset. On the order of **166,000 neurons** and **~125 million synaptic connections** (≈25.6M distinct edges at weight ≥ 1).

- Explore it: https://male-cns.janelia.org/
- The paper: https://doi.org/10.1016/j.cell.2026.08.015

The data isn't just topology — it carries a **predicted neurotransmitter** per neuron. In the traced set that breaks down as acetylcholine 62.8%, glutamate 17.7%, GABA 13.4%, histamine 3.6%, plus traces of dopamine, octopamine, serotonin. That gives an **inhibitory share of ~34.7%** (GABA + Glu + His), which the model uses to sign every synapse. Excitation and inhibition aren't guessed; they come from the biology.

Region composition is read the same way, from the connectome's superclasses: the optic lobes dominate (~54% of neurons, `ol_intrinsic` + `visual_projection`), then the central brain (~19.5% `cb_intrinsic`), then the VNC (~8% `vnc_intrinsic`), with the mushroom body, central complex and SEZ carved out by cell type.

> **The honesty rule, stated up front because "driven by a real brain" is a claim, not a vibe.** The brain the live site runs *by default* is a **synthetic, MaleCNS-shaped stand-in**: a ~20,000-neuron generator that reproduces the real region taxonomy, the real cell-type label strings, and verified pathway wiring, calibrated so the documented circuits actually fire. The full 166k Janelia graph (566 MB of feather tables via neuPrint/GCS) is one environment variable away — `FLY_CONNECTOME_SOURCE=neuprint`. We say this in the app and in the repo. Nothing is dressed up as more than it is.
>
> Full engineering contract: [SPEC.md](https://github.com/Synapse-Fly/synapsefly/blob/main/docs/SPEC.md) · every verified cell-type / pathway / weight, with sources: [RESEARCH.md](https://github.com/Synapse-Fly/synapsefly/blob/main/docs/RESEARCH.md)

## 2. The neuron model

Each neuron is a **leaky integrate-and-fire (LIF)** unit, following the parameters from Shiu et al. (2024)'s whole-brain fly model: a membrane that leaks toward rest, integrates synaptic input, fires when it crosses threshold, then resets and goes refractory. The update is the exact 2-D matrix-exponential form (voltage + synaptic conductance), stepped at **dt = 1 ms**, with "unless refractory" semantics — voltage and conductance freeze during the refractory window while incoming input still accumulates.

Synaptic weight is scaled per connection (mV per synapse); a single ~**68.75 mV** forcing kick makes an isolated neuron spike at ~3 ms, which is how the sensory drives inject current. Neurotransmitter sign flips the synapse: ACh excitatory (+), GABA/Glu/His inhibitory (−). At rest, a small per-step noise current holds the whole brain at a biological **1–5 Hz** background hum instead of a dead flatline.

## 3. The engine: 166k neurons, in a browser tab's worth of compute

The connectome is one big **CSR sparse matrix** (int64 row pointers, int32 column indices, float32 weights). Propagation is **event-driven**: only the neurons that spiked this step have their outgoing synapses visited (a numpy `add.at` scatter), so cost tracks the ~1–3% of neurons firing per millisecond, not the full matrix. Pure numpy, float32 throughout, an optional Torch backend for the full graph, memory-mapped `.npz` caches so the 166k graph loads without rebuilding.

Result: **~2× real-time on a single CPU** at 20k neurons, 20 ticks/second. Before it ever serves a frame, the sim has to pass five biological **gates** or it refuses to claim the wiring works: rest rate in 1–5 Hz, sugar → MN9 feeding motor neuron ≥ 20 Hz, looming → giant-fiber escape within ≤ 20 ms, no runaway (activity never pins), and PFL3 → DNa02 steering firing on the **contralateral** side (the real laterality). A homeostatic gain guard keeps it there over long uptime.

## 4. The senses: the market is sensory input

The fly does not *read* the chart. It *feels* it. Live market data (DexScreener / a WebSocket feed) is converted, tick by tick, into injected current on specific, named sensory neurons:

- **A buy / price up →** current into the **labellar sugar gustatory receptor neurons** (the sweet GRNs). That drives the real SEZ feeding chain — the LB3b/c → GNG interneuron cascade → **MN9**, the proboscis motor neuron whose top input (DNge062, 556 synapses) is wired in. The fly gets hungry-happy, slows, and loops.
- **A sell / price down →** a 300 ms expanding-disc current ramp — `r(t) = 220 · amp · (t/300ms)²` Hz — onto ~60% of the **LC4 / LPLC2** lobula columnar looming detectors, one side, episode-seeded. Those feed the **Giant Fiber (DNp01)**, which drives the takeoff/escape muscles (TTMn, DLMn) via the PSI. A whale dump becomes a full-field flash. To a fly, a red candle is a bird. It panics, jumps, and darts for a corner.

Buy/sell pressure, volume, and price-change all become currents; the whole feed becomes an *experience the nervous system has*, not a number a script parses. Exact encoder formulas are in [SPEC.md §f](https://github.com/Synapse-Fly/synapsefly/blob/main/docs/SPEC.md).

## 5. The body: a fly painting the price

Motor output is decoded back into a body. Descending-neuron and motor-neuron population rates map to kinematics: **DNa02** (and DNa01/DNa03/DNg13) drive yaw/steering; wing power and steering MNs set wing-beat frequency; the giant-fiber discharge triggers a ballistic jump; the feeding chain freezes forward motion. The result is `(x, y)` velocity, heading (radians, 0 = +x, positive clockwise), and wing-beat Hz, integrated across an **800 × 500** canvas with wall bounce.

Behind it the fly leaves an ink trail colored by the candle — green loops while it's fed, red panic-scribbles while it's dumped on, thickness scaling with wing-beat. It is, literally, **painting the price with its body.**

> We keep receipts here too. A fly with zero market stimulus would just sit there, so there are engineered nudges — an exploration baseline into DNg100, an OU wander on heading, a wall-bump steering pulse, a giant-fiber rest brake. Every one is **named, located in the code, and given an off switch** in [PUPPETEERING.md](https://github.com/Synapse-Fly/synapsefly/blob/main/docs/PUPPETEERING.md), and a unit test fails if that list ever drifts from the generator. The puppet strings are drawn on the outside of the puppet.

## 6. The mind and the voice

On top of the drives sits an **8-state mood machine**: SLEEP, CRUISING, FEEDING, EUPHORIA, ANXIOUS, PANIC, ESCAPE, COURTSHIP — with a strict priority order (ESCAPE > PANIC > COURTSHIP > EUPHORIA > FEEDING > ANXIOUS > SLEEP > CRUISING), 1.5 s minimum dwell, and a 3 s confirmation before anything as loud as a tweet fires. Euphoria and panic are integrated from the sugar and looming/escape circuits, not from the raw price.

When mood crosses a threshold, the brain's live state — per-region firing rates, the active circuits, the mood — is serialized to JSON and handed to an LLM, which writes one short, in-character, crypto-dialect tweet. The fly is the poster. You are watching a bug run sentiment analysis on its own nervous system and post through it.

There is also a **live 3D brain** (three.js): the eight regions as point clouds arranged in a stylized *Drosophila* layout — two optic lobes to the sides, mushroom bodies dorsal, the central complex's EPG/PEN heading-compass in the middle, the SEZ → descending → VNC tapering down — each region glowing with its real-time firing rate, the pathways between them pulsing as signal flows. It looks like a fly brain because it is shaped like one.

## 7. The stack

- **Backend (the brain):** Python — FastAPI + a native WebSocket, the numpy event-driven LIF engine, a market feed with a simulated fallback, a sensory encoder, a motor decoder, the mood machine, and the tweet agent, all hanging off one **20 Hz** simulation loop broadcasting ~2.5–3.5 KB per tick. Always-on on a VPS behind Caddy (wss + TLS). Rate-limited against griefing (the whole audience watches **one** shared fly), with a liveness watchdog so a dead sim restarts instead of freezing.
- **Frontend (the face):** Next.js + TypeScript + Tailwind + three.js — a genuine Windows 95 desktop (draggable Paint, an oscilloscope spike-raster, a Fly Status panel, a tweets.txt notepad, the 3D brain, a Start menu, even a Recycle Bin). One WebSocket feeds all of it. Deployed on Vercel.
- **Open source, MIT.** The connectome data is Janelia FlyEM MaleCNS v1.0, **CC-BY 4.0** (FlyWire/Codex CC-BY-NC) — full attribution and DOIs in [NOTICE.md](https://github.com/Synapse-Fly/synapsefly/blob/main/docs/NOTICE.md). No connectome data is redistributed in the repo.

## 8. About the ticker — read this

**$SYNAPSE has not launched. There is no contract address yet.** The price you see on the site right now belongs to an **unrelated third-party pair** the brain is borrowing purely as sensory input — a stand-in feed. Every surface that shows a number says so, and a `token_live` flag (default off) keeps the "not launched / this is NOT $SYNAPSE" qualifier on the price until the real token exists. We would rather ship the disclaimer than let one person quote a market cap for a token that isn't real. When it launches, the real CA goes in, one flag flips to true, the qualifiers vanish, and the fly starts feeling **its own** market.

## 9. Why

Connectomics, spiking neural networks, real-time web, and crypto do not usually end up in the same window. This is what happens when they do: an open connectome, an open engine, an honest canvas, and one extremely online insect that trades on how its neurons feel.

Go watch a bug's brain trade. Feed it some sugar.

**→ https://www.synapsefly.com** · **𝕏 @SynapseFly** · **github.com/Synapse-Fly/synapsefly**

*Not financial advice. It's a bug.* 🪰🧠
