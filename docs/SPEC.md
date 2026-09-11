# SynapseFly / FlyBrain — Implementation Contract (SPEC v1.0, 2026-09-10)

> **TR özet.** Bu belge 8 mühendisin birbiriyle konuşmadan paralel uygulayabileceği kesin sözleşmedir: depo yerleşimi,
> env değişkenleri, her Python modülünün tam public API'si (imza + docstring + dataclass alanları), WebSocket JSON
> protokolü, frontend bileşen sözleşmesi, encoder/decoder/mood formülleri (kesin sayılar), sentetik konektom üreteci,
> test listesi ve çalıştırma talimatları. Tüm sayılar `docs/RESEARCH.md`'de doğrulanmış verilere dayanır. Kod ve yorumlar
> İngilizce, README Türkçe. **Bugün sıfır hesapla uçtan uca çalışmak zorunludur** (sentetik konektom + simüle piyasa +
> LLM/X dry-run); gerçek MaleCNS, DexScreener, Claude ve X yolları birer env değişkeni uzaklıktadır.

This document is normative. Where it conflicts with the three proposals, this document wins. Where it is silent, follow
the closest verified fact in `docs/RESEARCH.md`, then the winning proposal (Proposal 2). Section order is fixed by the
task brief: **a** repo layout, **b** config, **c** Python module contracts, **d** WebSocket protocol, **e** frontend
contract, **f** encoder/decoder/mood formulas, **g** synthetic connectome, **h** tests + smoke, **i** run instructions.

---

## 0. Architecture decisions (final synthesis)

**Spine = Proposal 2 (ship-today / demo-first)**, grafted with the judges' picks:

| Decision | Choice | Source |
|---|---|---|
| Neuron model | Shiu 2024 LIF, **exact 2-D matrix-exponential update**, Brian2 "unless refractory" semantics (v and g frozen while refractory, incoming input still accumulates), reset `v=v_reset, g=0` | P1 (integrator), verified numerics in RESEARCH §7 |
| Time step | `FLY_DT_MS=1.0` default (ref 2 steps, delay 2 steps); 0.5 / 0.1 supported for fidelity runs | task brief |
| Sensory drive | `Drive` dataclass with rate / recruit / side / episode; Poisson forcing `g += 68.75 mV` or current mode via `current_from_rate` | P1 |
| Rest activity | **1–5 Hz background required**: per-step noise `g += N(mu·dt, sigma·sqrt(dt))`, defaults mu 0.5 mV/ms, sigma 3.5 mV/sqrt(ms) → ~2.3 Hz isolated (RESEARCH §7); **no "silent brain" gate** | task brief (overrides P2/P3 gate) |
| Synthetic graph | MaleCNS-shaped structured generator with **real type strings and verified counts**, 8-region taxonomy, 112 projection rules; weights **calibrated** by the steady-state formula (default) or **literature** means (fidelity mode) | P2 formula + P1/P3 catalog + RESEARCH §4–5 |
| Verified pathway fixes | PFL3→DNa02/DNa03/DNb01 **contralateral**; LC9/LC31a→DNp09 (not LC6/LPLC1); histaminergic photoreceptors **inhibitory** with tonic lamina/medulla drive; real SEZ feeding chain (GNG215/GNG232/GNG132/GNG089 → GNG108/GNG120/GNG117/GNG234/DNge062/DNge080 → MN9 plus GNG042→GNG015 disinhibition); DNg02→MNwm36/tp2/ps1 + weak DLMn/DVMn; DNg60/DNg74 GABAergic; DNb01 glutamatergic | RESEARCH §5 |
| Gap junctions | patch table applied to every source: DNp01→TTMn, DNp01→PSI, DNp01↔DNp01, LC4/LPLC2→DNp01 dendro-dendritic proxies, recorded in `meta.patches_applied` | P1/P3 |
| Guaranteed motion | explore baseline into DNg100 (`FLY_EXPLORE_BASELINE=0.25` → ≥ 7.5 Hz), OU wander on heading, `wander_floor` (|v| < 5 px/s for 3 s → 40 px/s), EPG wedge proprioceptive feedback, wall-bump DNa02 pulse; all isolated in `encoder.py`/`decoder.py`, documented as puppeteering, dialable to zero | P2 + P3 |
| Scheduler | wall-clock 20 Hz ticks that are **never skipped**; brain time per tick = `50 ms × speed`, `speed ∈ [0.25, 1]` adapts to compute; `speed`/`rtf` broadcast; `FLY_REALTIME=0` = fixed 50 steps/tick, no sleep (tests, replay) | P3 (+P2 budget) |
| Sparse layout | one CSR (`indptr int64 + int32 shadow`, `indices int32`, `data float32`) from 20k to 166.7k; uncompressed `.npz` cache with `mmap_mode='r'`; `Propagator` Protocol with numpy (`np.add.at`) and torch (`index_add_`) backends, `auto` → torch when E > 5M | P3 + P2 |
| Homeostasis | active fraction > 5 % for 10 consecutive steps → `gain *= 0.9` (floor 0.3), event `homeostasis`; `calibrate_gain` bisection (cached per connectome key) for literature-weight graphs | P2 + P1/P3 |
| Market | `MarketSource` Protocol; `SimulatedMarket` (5-regime Markov GBM + clustered trades + whales); `DexScreenerSource` (`/tokens/v1`, 60 s poll, rate-matched surrogate trades between polls); `MarketFeed` thread with automatic dex→sim fallback and `market_source` event | P1 + P2 + P3 |
| Looming | per sell event: 300 ms expanding-disc ramp `r(t)=220·amp·(t/300 ms)^2` Hz on a random 60 % of LC4/LPLC2 (episode-seeded, one side); tonic component `12·relu((looming−0.3)/0.7)^2` Hz; whale dump = 300 ms full-field flash; slow sustained looming → LC9/LC31a (`lc_freeze`) | P3 + P1 |
| Mood | 8 states, priority ESCAPE > PANIC > COURTSHIP > EUPHORIA > FEEDING > ANXIOUS > SLEEP > CRUISING, timers accumulate only while the condition holds, 1.5 s min dwell, 3 s confirmation before a tweet; ESCAPE never tweets, `ESCAPE_BURST` (≥ 3 jumps / 60 s) does | P2 + P3 |
| Agent | exact Anthropic call from the brief (`claude-opus-5`, `max_tokens=512`, no thinking param, no prefill), structured output via `output_config` (works on installed 0.86.0), dry-run templates, `validate_tweet` (strip URLs, ≤ 280, ≤ 2 hashtags, vocabulary token required), `data/agent_state.json` + `data/tweets.jsonl`, snapshot = browser PNG over WS with numpy fallback, tweepy `create_tweet` + v2 media upload + v1.1 fallback + text-only | P1 + P2 + P3 |
| Frontend | the **existing** `frontend/` scaffold (Next 16.3.4 / React 19.2.8 / Tailwind 4.3.3), no new dependencies, four draggable Win95 windows, procedural fly, ink styles, Notepad tweet log, keyboard shortcuts, `document.title = mood` | P2 UI on P3's scaffold |
| Canvas | **800 × 500** logical px, y down, `FLY_WALLS=bounce` (default) or `wrap` | task brief |
| Provenance | `hello.connectome.source ∈ {synthetic, csv, neuprint}`, `license`, `citation`; About dialog, README, LLM summary all label synthetic data as "synthetic structured stand-in (MaleCNS-shaped)" | P1 |
| Replay | `SessionLog` JSONL of every tick's inputs (market digest, trades, pokes, wall time) and `scripts/replay.py` asserting identical spike counts | P1 |

### 0.1 Cross-cutting conventions (binding for every module)

* **Units**: brain time in ms (`t_ms: int`), wall time in seconds since epoch (`wall: float`); rates in Hz *per neuron*;
  voltages in mV; canvas positions in px; heading in radians, `0 = +x`, **positive = clockwise on screen** (y points down);
  speed in px/s; angular velocity in rad/s; wingHz in Hz.
* **Indices**: neurons are `0..n-1` (`int32`); `body_id` is the external id (`int64`; synthetic = `1_000_000 + i`).
* **Sides**: `side: int8` ∈ {−1 (L), +1 (R), 0 (midline/unknown)}. Sided group keys are `<name>_L` / `<name>_R`; the
  unsuffixed key is the union.
* **Regions**: `REGIONS = ("optic_lobe","antennal_lobe","mushroom_body","central_complex","sez","central_other","descending_motor","vnc")`,
  ids 0–7 in this order everywhere (arrays, JSON, raster lanes).
* **Randomness**: one `np.random.SeedSequence(FLY_SEED)` spawned in a fixed order: `[0] connectome build, [1] engine
  (Poisson + noise), [2] market sim, [3] decoder wander, [4] raster sampling, [5] agent templates, [6] encoder episodes`.
  Deterministic by contract on the numpy backend with `FLY_MARKET=sim`.
* **Threads**: `SimulationLoop` (thread) owns engine/encoder/decoder/mood/agent-trigger; `MarketFeed` (thread) owns HTTP;
  uvicorn's asyncio loop owns WebSockets; `TweetAgent` work runs on a 1-worker `ThreadPoolExecutor`. Cross-thread
  hand-off only through `StateBus` (`loop.call_soon_threadsafe`) and thread-safe queues. **Never block the sim thread on I/O.**
* **No optional imports at module import time**: `torch`, `anthropic`, `tweepy`, `pyarrow`, `pandas`, `neuprint` are
  imported inside functions and only when the corresponding mode is enabled. Runtime hard deps: numpy, fastapi, uvicorn,
  websockets, pydantic, httpx.
* **Console/log output is ASCII only** (Windows console is cp1254) and the launcher sets `PYTHONUTF8=1`.
* **JSON keys are snake_case** on the wire, mirrored 1:1 in `frontend/lib/types.ts` (the frontend does not rename keys).
* **Errors never stop the sim**: every external call is wrapped; failures produce an `event` frame and a log line.
* **Docstrings** must carry the provenance tag of every biological number: `[V]` verified (RESEARCH), `[L]` literature,
  `[E]` engineered (puppeteering / stand-in), so reviewers can audit fidelity claims.

### 0.2 Parallel workstreams (8 engineers, no talking required)

| # | Owner | Modules | Depends only on |
|---|---|---|---|
| E1 | Connectome | `connectome/{schema,csr,groups,patches,synthetic,loaders,cache}.py`, `scripts/{prepare_malecns,fetch_neuprint,export_synthetic}.py`, fixtures | this spec |
| E2 | Engine | `snn/{params,propagators,engine,monitor,calibrate}.py`, `scripts/bench.py` | `Connectome` fields (§c.2) |
| E3 | Market + encoder | `market/{base,sim,dexscreener,feed}.py`, `encoder.py`, `scripts/demo_market.py` | `Drive` (§c.10), group names (§c.4) |
| E4 | Body | `decoder.py`, `mood.py` | `StepStats`/rates keys (§c.10), `Features` (§c.17) |
| E5 | Agent | `agent/{summary,llm,x_client,snapshot,orchestrator}.py` | tick dict (§d), `StateBus` (§c.25) |
| E6 | Server | `config.py`, `log.py`, `server/{protocol,state,session_log,png,loop,app}.py`, `__main__.py`, `run.py`, `scripts/{smoke,selftest,replay}.py`, `.env.example` | all public APIs in §c |
| E7 | Frontend core | `frontend/lib/{types,ws,store,interp,flySprite,ink}.ts`, `components/{Win95Window,PaintWindow,FlyCanvas,ToolPalette,ColorPalette,MenuBar,StatusBar}.tsx`, `app/{layout,page}.tsx`, `globals.css` | §d, §e |
| E8 | Frontend panels + docs | `components/{SpikeRaster,MoodPanel,MarketTicker,TweetNotepad}.tsx`, `lib/raster.ts`, `README.md` (TR), `docs/NOTICE.md`, CI script | §d, §e |

---

## a. Repository layout (exact paths)

```
C:\Users\USER\fly\
  .gitignore                       # exists; ignores data/* except data/.gitkeep and data/README.md, frontend/node_modules, .env*
  .env.example                     # every variable of section b with its default and a one-line comment
  README.md                        # Turkish: kurulum, mimari, veri kaynaklari, lisans (MaleCNS CC-BY), "sentetik" uyarisi
  docs\RESEARCH.md                 # research digest (exists)
  docs\SPEC.md                     # this file
  docs\NOTICE.md                   # data provenance + citations (tracked; data/ is gitignored)
  backend\
    pyproject.toml                 # [project] name="flybrain", version="0.1.0", requires-python=">=3.12"; packages=["flybrain"]; [tool.pytest.ini_options] testpaths=["tests"]
    requirements.txt               # runtime pins (section i)
    requirements-optional.txt      # torch / anthropic / tweepy / pyarrow / pandas / neuprint-python
    run.py                         # `py -3 run.py` == `py -3 -m flybrain` (adds backend/ to sys.path, calls flybrain.__main__.main)
    flybrain\
      __init__.py                  # __version__ = "0.1.0"
      __main__.py                  # main(): load_settings -> create_app -> uvicorn.run
      config.py                    # Settings + load_settings + redacted
      log.py                       # setup_logging(level) -> ASCII console + ring buffer (last 200 lines) for /api/health
      connectome\__init__.py
      connectome\schema.py         # REGIONS, CSR, Connectome
      connectome\csr.py            # build_csr, remap_ids, transpose_csr, sum_duplicates
      connectome\groups.py         # GROUP_REGEX, SIDED, READOUTS, STAR_TYPES, resolve_groups, region_of, side_of, validate_groups
      connectome\patches.py        # GAP_JUNCTIONS, apply_patches
      connectome\synthetic.py      # Pop, Proj, POPULATIONS, PROJECTIONS, REGION_FRAC, scale_populations, calibrated_weight, build_synthetic, export_csv
      connectome\loaders.py        # Schema, detect_schema, nt_sign, load_csv_dir, load_connectome
      connectome\cache.py          # cache_key, load_cached, store_cached
      snn\__init__.py
      snn\params.py                # LIFParams, StepConstants, current_from_rate, rate_from_current
      snn\propagators.py           # Propagator Protocol, NumpyPropagator, TorchPropagator, make_propagator
      snn\engine.py                # Drive, Injection, StepStats, LIFEngine
      snn\monitor.py               # RasterRow, SpikeMonitor, RateEstimator
      snn\calibrate.py             # CalibTargets, GateReport, run_gates, calibrate_gain
      market\__init__.py
      market\base.py               # Trade, MarketSnapshot, MarketSource
      market\sim.py                # Regime, REGIMES, SimulatedMarket
      market\dexscreener.py        # fetch_pairs, parse_pair, DexScreenerSource, SurrogateTrades
      market\feed.py               # MarketFeed
      encoder.py                   # Features, FeatureExtractor, Poke, SensoryEncoder, hill
      decoder.py                   # Readouts, MotorCommand, Kinematics, InkStyle, Wander, MotorDecoder, FlyBody
      mood.py                      # Mood, MoodInputs, MoodState, MoodMachine, MOOD_COLORS
      agent\__init__.py
      agent\summary.py             # build_brain_summary
      agent\llm.py                 # SYSTEM_PROMPT, TweetDraft, TweetGenerator, template_tweet, validate_tweet
      agent\x_client.py            # PostResult, XClient
      agent\snapshot.py            # SnapshotBroker, render_snapshot
      agent\orchestrator.py        # AgentState, TweetAgent
      server\__init__.py
      server\protocol.py           # pydantic v2 models for every WS/REST message
      server\state.py              # StateBus, TickHistory
      server\session_log.py        # SessionLog, read_session
      server\png.py                # write_png
      server\loop.py               # SimulationLoop
      server\app.py                # create_app, routes, /ws
    tests\
      conftest.py                  # tiny_connectome fixture (N=400), settings fixture, tmp data dir
      fixtures\neuprint_small\neurons.csv, connections.csv        # 40 neurons / 120 edges, neuPrint schema
      fixtures\codex_small\classification.csv.gz, consolidated_cell_types.csv.gz, neurons.csv.gz, connections.csv.gz
      test_csr.py test_engine.py test_engine_backends.py test_synthetic.py test_groups.py test_loaders.py
      test_encoder.py test_decoder.py test_mood.py test_market.py test_protocol.py test_agent.py test_server.py test_calibrate.py
  scripts\
    dev.ps1                        # starts backend (:4000) and frontend (:3000) in two PowerShell jobs, sets PYTHONUTF8=1
    smoke.py                       # 2000 steps, asserts rest rate / sugar->MN9 / loom->GF (section h)
    selftest.py                    # the four boot gates + RTF, prints a table, exit code 0/1
    demo_market.py                 # drives the sim market PUMP -> RUG -> CALM via POST /api/market/mode
    replay.py                      # replays data/sessions/<run_id>.jsonl and asserts identical spike counts
    bench.py                       # ms/step and RTF for N in {20k,40k,80k} x dt in {1.0,0.5}
    prepare_malecns.py             # GCS feathers -> data/connectome/malecns/{neurons,connections}.csv (needs pyarrow)
    fetch_neuprint.py              # anonymous urllib or token neuprint-python -> data/connectome/neuprint/{neurons,connections}.csv
    export_synthetic.py            # synthetic -> neuPrint-schema CSV (loader parity proof)
  data\                            # gitignored except .gitkeep / README.md
    README.md                      # what lives here (tracked)
    cache\                         # <key>.npz + <key>.json connectome caches, <key>.calib.json
    connectome\{malecns,neuprint,codex}\   # user-supplied / downloaded CSVs
    sessions\<run_id>.jsonl        # session logs
    snapshots\<t_ms>_<mood>.png    # canvas snapshots
    tweets.jsonl                   # every generated tweet
    agent_state.json               # cooldowns / daily counts (survives restarts)
  out\tweets\<ts>.txt|.json|.png   # dry-run artifacts (gitignored: add `out/` to .gitignore)
  frontend\                        # EXISTING Next 16.3.4 scaffold - do not re-scaffold
    package.json next.config.ts tsconfig.json postcss.config.mjs eslint.config.mjs   # exist; add "typecheck": "tsc --noEmit" script
    .env.local.example             # NEXT_PUBLIC_WS_URL=ws://localhost:4000/ws  NEXT_PUBLIC_API_URL=http://localhost:4000
    app\layout.tsx app\page.tsx app\globals.css app\favicon.ico
    components\Win95Window.tsx PaintWindow.tsx MenuBar.tsx ToolPalette.tsx ColorPalette.tsx StatusBar.tsx
    components\FlyCanvas.tsx SpikeRaster.tsx MoodPanel.tsx MarketTicker.tsx TweetNotepad.tsx
    lib\types.ts ws.ts store.ts interp.ts flySprite.ts ink.ts raster.ts api.ts
```

Python import root is `backend/` (`flybrain` is a top-level package). All relative paths in code are resolved against
`Settings.data_dir` / `Settings.out_dir`, which default to `<repo>/data` and `<repo>/out` (computed from `__file__`,
never from the CWD).

---

## b. Configuration (`flybrain/config.py`) — every environment variable

`load_settings()` reads `<repo>/.env` (plain `KEY=VALUE` lines, `#` comments, no quoting rules beyond strip) then
`os.environ` (environ wins), validates, and returns a frozen `Settings`. Booleans accept `1/0/true/false/yes/no`.

| Variable | Default | Type / allowed | Meaning |
|---|---|---|---|
| `FLY_CONNECTOME_SOURCE` | `synthetic` | `synthetic` \| `csv` \| `neuprint` | `synthetic`: generator; `csv`: `FLY_CONNECTOME_DIR/{neurons,connections}.csv(.gz)` (neuPrint-export or Codex schema autodetected); `neuprint`: same files under `data/connectome/neuprint`, fetched by `scripts/fetch_neuprint.py` (never fetched at runtime) |
| `FLY_CONNECTOME_DIR` | `data/connectome/malecns` | path | directory for `csv` source |
| `FLY_CONNECTOME_NAME` | `""` | str | cache/display name override (default derived from source + args) |
| `FLY_N_NEURONS` | `20000` | int 4000..200000 | synthetic size; also the `core` subset cap for real data |
| `FLY_MEAN_OUTDEG` | `25` | int 5..200 | synthetic background out-degree |
| `FLY_SYNTH_WEIGHTS` | `calibrated` | `calibrated` \| `literature` | pathway weight mode (section g) |
| `FLY_SUBSET` | `core` | `core` \| `all` | real data: `core` keeps every functional-group neuron + random fill to `FLY_N_NEURONS`; `all` keeps everything |
| `FLY_MIN_WEIGHT` | `3` | int ≥ 1 | real data: prune edges with synapse count below this |
| `FLY_DT_MS` | `1.0` | float ∈ {1.0, 0.5, 0.2, 0.1} | LIF step |
| `FLY_SEED` | `1337` | int | root seed |
| `FLY_BACKEND` | `numpy` | `numpy` \| `torch` \| `auto` | propagator backend (`auto` = torch when E > 5,000,000 and torch imports) |
| `FLY_GAIN` | `auto` | `auto` \| float | synaptic gain multiplier; `auto` = 1.0 for `calibrated` synthetic, else `data/cache/<key>.calib.json` or `calibrate_gain` on first start |
| `FLY_NOISE_MU` | `0.5` | float mV/ms | background noise mean per ms (section c.10) |
| `FLY_NOISE_SIGMA` | `3.5` | float mV/sqrt(ms) | background noise std |
| `FLY_DRIVE_MODE` | `poisson` | `poisson` \| `current` | how `Drive.rate_hz` is applied |
| `FLY_TICK_HZ` | `20` | int 5..50 | wall tick rate; `tick_ms = 1000 / FLY_TICK_HZ` |
| `FLY_REALTIME` | `1` | bool | 1 = phase-locked wall ticks with adaptive `speed`; 0 = fixed `tick_ms/dt` steps per tick, no sleeping |
| `FLY_MARKET` | `sim` | `sim` \| `dexscreener` | market source (`dexscreener` needs `FLY_TOKEN_ADDRESS`; falls back to sim after 3 failed polls) |
| `FLY_TOKEN_ADDRESS` | `""` | str | token mint / contract address |
| `FLY_CHAIN` | `solana` | str | DexScreener `chainId` |
| `FLY_DEX_POLL_S` | `60` | int ≥ 15 | DexScreener poll period (edge cache is ~60 s) |
| `FLY_SIM_REGIME_S` | `90` | float | mean regime dwell of the simulated market (scales all regime dwells) |
| `FLY_LLM` | `dryrun` | `dryrun` \| `anthropic` | tweet text source |
| `ANTHROPIC_API_KEY` | unset | str | read by the SDK when `FLY_LLM=anthropic` |
| `FLY_LLM_MODEL` | `claude-opus-5` | str | model id (never append a date suffix) |
| `FLY_LLM_JSON` | `1` | bool | use `output_config` structured output; on failure fall back to plain text |
| `FLY_LLM_FALLBACKS` | `0` | bool | pass server-side refusal fallbacks (requires anthropic ≥ 1.x); ignored otherwise |
| `FLY_TWEET_LANG` | `en` | `en` \| `tr` | language line appended to the system prompt |
| `FLY_X` | `dryrun` | `dryrun` \| `post` | posting mode |
| `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | unset | str | OAuth1 user context (Read+Write app) |
| `FLY_TWEET_COOLDOWN_S` | `900` | int | global cooldown between posts |
| `FLY_TWEET_REASON_COOLDOWN_S` | `2700` | int | per-reason cooldown |
| `FLY_TWEETS_PER_DAY` | `12` | int | hard daily cap (UTC day) |
| `FLY_PORT` | `4000` | int | HTTP/WS port |
| `FLY_HOST` | `127.0.0.1` | str | bind host |
| `FLY_CORS_ORIGINS` | `http://localhost:3000` | comma list | CORS allow-list |
| `FLY_CANVAS_W` / `FLY_CANVAS_H` | `800` / `500` | int | canvas bounds |
| `FLY_WALLS` | `bounce` | `bounce` \| `wrap` | boundary rule |
| `FLY_RASTER_PER_REGION` | `48` | int 8..128 | raster rows per region |
| `FLY_RASTER_CAP` | `2000` | int | max raster events per tick |
| `FLY_MOOD_FEEDBACK` | `1` | bool | mood → brain feedback drives (section f.2, labelled puppeteering) |
| `FLY_EXPLORE_BASELINE` | `0.25` | float 0..1 | exploration drive floor (0 disables) |
| `FLY_WANDER_SIGMA` | `0.6` | float rad/s/sqrt(s) | OU wander strength (0 disables) |
| `FLY_EASTER_EGGS` | `1` | bool | 69/420 / 04:20 courtship triggers |
| `FLY_SESSION_LOG` | `1` | bool | write `data/sessions/<run_id>.jsonl` |
| `FLY_REPLAY` | `""` | path | replay a session log instead of live inputs |
| `FLY_DATA_DIR` / `FLY_OUT_DIR` | `<repo>/data` / `<repo>/out` | path | state / artifacts |
| `FLY_LOG_LEVEL` | `INFO` | str | logging level |
| `NEUPRINT_APPLICATION_CREDENTIALS` | unset | str | neuPrint token for `scripts/fetch_neuprint.py` (anonymous HTTP when unset) |
| Frontend: `NEXT_PUBLIC_WS_URL` | `ws://localhost:4000/ws` | str | WebSocket URL |
| Frontend: `NEXT_PUBLIC_API_URL` | `http://localhost:4000` | str | REST base |

Derived (computed in `load_settings`, read-only fields): `tick_ms = 1000/tick_hz`, `steps_per_tick = round(tick_ms/dt_ms)`,
`run_id = sha1(f"{source}|{n_neurons}|{seed}|{dt_ms}|{synth_weights}|{market}")[:8]`, `connectome_key` (section c.8).

---

## c. Python module contracts

Every signature below is the public API. Private helpers are free. Type hints are mandatory; `np.ndarray` dtypes are
stated in docstrings. All dataclasses are `@dataclass(slots=True)` unless `frozen=True` is stated.

### c.1 `flybrain/config.py`

```python
@dataclass(frozen=True)
class Settings:
    # connectome
    connectome_source: str = "synthetic"      # FLY_CONNECTOME_SOURCE: synthetic|csv|neuprint
    connectome_dir: Path = Path("data/connectome/malecns")
    connectome_name: str = ""
    n_neurons: int = 20_000
    mean_outdeg: int = 25
    synth_weights: str = "calibrated"         # calibrated|literature
    subset: str = "core"                      # core|all
    min_weight: int = 3
    # engine
    dt_ms: float = 1.0
    seed: int = 1337
    backend: str = "numpy"                    # numpy|torch|auto
    gain: str = "auto"                        # "auto" or a float literal, e.g. "0.65"
    noise_mu: float = 0.5
    noise_sigma: float = 3.5
    drive_mode: str = "poisson"               # poisson|current
    tick_hz: int = 20
    realtime: bool = True
    # market
    market: str = "sim"                       # sim|dexscreener
    token_address: str = ""
    chain: str = "solana"
    dex_poll_s: int = 60
    sim_regime_s: float = 90.0
    # agent
    llm: str = "dryrun"                       # dryrun|anthropic
    llm_model: str = "claude-opus-5"
    llm_json: bool = True
    llm_fallbacks: bool = False
    tweet_lang: str = "en"
    x_mode: str = "dryrun"                    # FLY_X: dryrun|post
    tweet_cooldown_s: int = 900
    tweet_reason_cooldown_s: int = 2700
    tweets_per_day: int = 12
    # server / body
    port: int = 4000
    host: str = "127.0.0.1"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    canvas_w: int = 800
    canvas_h: int = 500
    walls: str = "bounce"                     # bounce|wrap
    raster_per_region: int = 48
    raster_cap: int = 2000
    mood_feedback: bool = True
    explore_baseline: float = 0.25
    wander_sigma: float = 0.6
    easter_eggs: bool = True
    session_log: bool = True
    replay: str = ""
    data_dir: Path = Path("data")             # absolute after load_settings
    out_dir: Path = Path("out")
    log_level: str = "INFO"
    # derived (filled by load_settings)
    tick_ms: float = 50.0
    steps_per_tick: int = 50
    run_id: str = ""
    connectome_key: str = ""

def load_settings(env: Mapping[str, str] | None = None, dotenv: Path | None = None) -> Settings:
    """Parse .env (default <repo>/.env if it exists) then env (environ wins). Validate every field against the
    allowed values in section b; raise ValueError with the offending variable name. Resolve data_dir/out_dir to
    absolute paths relative to the repository root (three parents of this file), create them if missing.
    Fill derived fields."""

def gain_value(settings: Settings) -> float | None:
    """None when settings.gain == 'auto', else float(settings.gain)."""

def redacted(settings: Settings) -> dict:
    """asdict() with X_* / ANTHROPIC_* never included and paths as strings (for GET /api/config)."""
```

### c.2 `flybrain/connectome/schema.py`

```python
REGIONS: tuple[str, ...] = ("optic_lobe", "antennal_lobe", "mushroom_body", "central_complex",
                            "sez", "central_other", "descending_motor", "vnc")
REGION_ID: dict[str, int]          # name -> 0..7
SYNTHETIC_BODY_BASE: int = 1_000_000

@dataclass(frozen=True)
class CSR:
    indptr: np.ndarray             # int64[n+1]
    indices: np.ndarray            # int32[E]  postsynaptic index, sorted within each row
    data: np.ndarray               # float32[E] SIGNED synapse count = weight * sign[pre]  (no w_syn, no gain)
    indptr32: np.ndarray | None    # int32 shadow copy when E < 2**31 (used for fast gathers), else None
    @property
    def n(self) -> int: ...
    @property
    def e(self) -> int: ...

@dataclass
class Connectome:
    name: str                      # e.g. "synthetic-20000-s1337-cal"
    source: str                    # "synthetic" | "csv" | "neuprint"
    n: int
    pre: np.ndarray                # int32[E], non-decreasing (row-major CSR order); no self-loops; no duplicate (pre,post)
    post: np.ndarray               # int32[E]
    weight: np.ndarray             # float32[E] synapse count > 0 (UNSIGNED)
    sign: np.ndarray               # float32[n] +1.0 / -1.0 (presynaptic NT sign, see loaders.nt_sign)
    region: np.ndarray             # uint8[n] index into REGIONS
    side: np.ndarray               # int8[n] -1 L, +1 R, 0 unknown/midline
    types: list[str]               # unique type labels; "" = untyped
    type_idx: np.ndarray           # int32[n] index into types
    body_id: np.ndarray            # int64[n]
    nt: np.ndarray                 # object[n] normalised NT string (acetylcholine|gaba|glutamate|histamine|dopamine|octopamine|serotonin|unknown)
    groups: dict[str, np.ndarray]  # group name -> sorted int32 indices (may overlap between groups; sided keys included)
    meta: dict                     # see below
    # meta keys (all sources): {"e": int, "synapses": float, "license": str, "citation": str|None, "seed": int|None,
    #   "build_args": dict, "patches_applied": list[dict], "gain_default": float, "weights_mode": str|None,
    #   "engineered_edges": list[str], "region_counts": dict[str,int], "group_counts": dict[str,int], "created": iso8601}

    @property
    def e(self) -> int: ...
    def type_of(self, i: int) -> str: ...
    def where(self, type_regex: str, side: int | None = None) -> np.ndarray:
        """Sorted int32 indices whose type fullmatches regex (re.fullmatch), optionally filtered by side."""
    def csr(self) -> CSR:
        """Build once (cached on the instance) from pre/post/weight/sign; data = weight * sign[pre]."""
    def validate(self) -> None:
        """Assert: len(pre)==len(post)==len(weight); pre sorted; 0<=pre,post<n; no self loops; weight>0; sign in {+1,-1};
        region<8; side in {-1,0,1}; every groups[] array sorted, int32, within range; sum(region_counts)==n.
        Raise ValueError with a precise message."""
    def save(self, stem: Path) -> None:
        """Write <stem>.npz (np.savez, UNCOMPRESSED: pre, post, weight, sign, region, side, type_idx, body_id, nt)
        and <stem>.json ({name, source, n, types, groups: {name: [indices]}, meta}). Group index arrays are stored
        inside the npz as 'group__<name>' to keep the json small."""
    @classmethod
    def load(cls, stem: Path, mmap: bool = False) -> "Connectome":
        """Inverse of save; mmap=True passes mmap_mode='r' to np.load (full-graph mode)."""
    def subset(self, keep: np.ndarray, min_weight: float = 1.0) -> "Connectome":
        """Restrict to keep x keep (keep sorted int32), drop edges with weight < min_weight, remap indices,
        rebuild groups by intersection, update meta['e'], meta['region_counts']."""
```

### c.3 `flybrain/connectome/csr.py`

```python
def build_csr(pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int, sum_duplicates: bool = True
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO -> (indptr int64[n+1], indices int32[E], data float32[E]). Sort key = pre.astype(int64)*n + post using
    np.argsort(kind='quicksort') (never 'stable'); duplicates (pre,post) summed when sum_duplicates; indptr via
    np.bincount(pre, minlength=n).cumsum(). Self-loops are dropped. Returns arrays sorted by (pre, post)."""

def remap_ids(pre_ids: np.ndarray, post_ids: np.ndarray, body_ids_sorted: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """int64 external ids -> int32 indices via np.searchsorted on the sorted body_id array.
    Returns (pre, post, keep_mask) where keep_mask marks edges whose both endpoints exist."""

def transpose_csr(indptr, indices, data, n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Incoming view (used by calibrate/validate and tests); pure numpy."""

def in_degree(indptr, indices, n) -> np.ndarray:
    """int64[n] number of incoming edges (np.bincount(indices, minlength=n))."""
```

### c.4 `flybrain/connectome/groups.py`

`GROUP_REGEX` is the table of RESEARCH §4 (fullmatch on `type`). Exact keys (implementers copy verbatim):

```python
GROUP_REGEX: dict[str, str] = {
  # sensory
  "grn_sugar": r"(LB3b|LB3c|PhG1[abc]|LgLG3|LgLG4|WG2)",
  "grn_sugar_labellar": r"(LB3b|LB3c|PhG1[abc])",
  "grn_water": r"LB3a", "grn_salt": r"LB3d",
  "grn_bitter": r"(LB1[a-e]|LgAG1)",
  "grn_pher": r"(LgLG1a|LgLG1b|LgLG2)",                      # [?] identity unverified
  "jo_aud": r"JO-A.*", "jo_groom": r"JO-(C|E|F).*", "bm": r"BM",
  "photoreceptor": r"(R1-R6|R[78][dpy])", "lamina": r"L[1-5]",
  "motion_in": r"(Mi1|Tm3|Mi4|Mi9|Tm1|Tm2|Tm4|Tm9)", "t4t5": r"T[45][a-d]",
  "lc_loom": r"(LC4|LPLC2)", "lc4": r"LC4", "lplc2": r"LPLC2",
  "lc_loom2": r"(LC6|LPLC1|LPLC4)", "lc_freeze": r"(LC9|LC31a)",
  "orn": r"ORN_.*", "alpn": r"[A-Za-z0-9+]+_(l|ad|il|lv|v)?PN", "alln": r"(lLN|il3LN|v2LN|vLN|lvLN).*",
  # mushroom body
  "kc": r"KC.*", "apl": r"APL", "dpm": r"DPM",
  "mbon_avoid": r"MBON0[1-6]", "mbon_approach": r"MBON(09|11|12|13|14|18)",
  "pam": r"PAM[0-9]{2}", "ppl1": r"PPL10[1-8]",
  # central complex
  "epg": r"EPG", "pen": r"PEN_[ab][(]PEN[12][)]", "peg": r"PEG", "delta7": r"Delta7", "ring": r"ER[1-6].*",
  "pfl3": r"PFL3", "pfl2": r"PFL2", "pfl1": r"PFL1", "hdelta": r"hDelta[A-M]", "pfn": r"PFN.*", "exr": r"ExR[1-8]",
  "dfb_sleep": r"FB[67][A-Z].*",
  # central other
  "p1": r"pC1.*", "mal": r"mAL_.*", "aipg": r"aIPg[0-9]+", "avlp_aud": r"AVLP73[23]m",
  "lal_ps": r"(LAL083|LAL126|LAL179|PS049|PS059|VES051|VES052|AOTU015|AOTU019)",
  # descending
  "gf": r"DNp01", "escape_dn": r"DNp(02|04|11)", "dn_saccade": r"DNp03", "dn_land": r"DNp(07|10)",
  "dn_freeze": r"DNp09", "dn_fwd": r"(DNg100|DNge053|DNg97)", "dng100": r"DNg100", "dn_back": r"MDN",
  "dn_halt": r"(DNg60|DNg74_[ab])",
  "steer_a02": r"DNa02", "steer_a01": r"DNa01", "steer_a03": r"DNa03", "steer_b01": r"DNb01", "steer_g13": r"DNg13",
  "flight_dn": r"DNg02_[a-g]", "groom_dn": r"(DNg62|DNge078)", "feed_dn": r"(DNge062|DNge080|DNg67)",
  "song_dn": r"(pIP10|pMP2)", "pip10": r"pIP10",
  # VNC efferent / motor
  "escape_vnc": r"(TTMn|PSI|GFC2)", "ttmn": r"TTMn", "psi": r"PSI", "gfc2": r"GFC2",
  "wing_power": r"(DLMn a, b|DLMn c-f|DVMn 1a-c|DVMn 2a, b|DVMn 3a, b)",
  "wing_steer": r"((b[123]|i[12]|iii[134]|hg[1-4]|tp[12]|tpn|ps[12]) MN|MNwm3[56])",
  "b1": r"b1 MN", "i1": r"i1 MN", "hg1": r"hg1 MN",
  "song_mn": r"(hg1|hg3|hg4|b1) MN",
  "leg_mn": r"(Ti flexor|Acc[.] ti flexor|Ti extensor|Tr flexor|Acc[.] tr flexor|Tr extensor|Fe reductor|Ta depressor|Ta levator|Sternotrochanter|Sternal anterior rotator|Sternal posterior rotator|Tergotr[.]|Pleural remotor/abductor|ltm|ltm1-tibia|ltm2-femur) MN",
  # SEZ feeding
  "feed_mn": r"MN9", "feed_mn_other": r"(MN1|MN6)",
  "feed_pre_exc": r"(GNG108|GNG120|GNG117|GNG234)", "feed_pre_inh": r"(GNG015|GNG095|GNG130|GNG180|GNG184)",
  "sugar2_exc": r"(GNG215|GNG232|GNG132|GNG089|PRW046|PRW047)",
  "sugar2_inh": r"(GNG042|GNG038|AN13B002|AN05B023d|GNG551)",
  "bitter2": r"(GNG016|GNG087|GNG592)",
  # VNC interneurons / song
  "song_vnc": r"(dPR1|dMS2|vPR6|vPR9_[abc])", "dms2": r"dMS2",
  "an_steer": r"(AN03A008|AN04B003)", "leg_premotor": r"(IN13A001|IN08A002|IN19A016|IN07B010|IN03B015|IN12B003|IN19B043)",
  "sad093": r"SAD093", "gng458": r"GNG458", "an05b102a": r"AN05B102a",
}
SIDED: frozenset[str] = frozenset({"gf","steer_a02","steer_a01","steer_a03","steer_b01","steer_g13","dn_freeze","dng100",
  "dn_fwd","escape_dn","dn_saccade","lc4","lplc2","lc_loom","lc_freeze","epg","pfl3","ttmn","b1","i1","hg1","dms2",
  "feed_mn","p1","pip10","flight_dn","grn_sugar_labellar","grn_bitter"})

# READOUTS: a PARTITION (each neuron in at most one readout) used by SpikeMonitor for per-population rates.
# Order is the wire order of tick.rates.pops. Sided readouts contribute "<name>_L"/"<name>_R" AND "<name>".
READOUTS: tuple[str, ...] = ("gf","escape_dn","dn_saccade","dn_land","dn_freeze","dng100","dn_fwd","dn_back","dn_halt",
  "steer_a02","steer_a01","steer_a03","steer_b01","steer_g13","flight_dn","groom_dn","feed_dn","pip10","song_dn",
  "ttmn","psi","gfc2","wing_power","b1","i1","hg1","wing_steer","leg_mn","feed_mn","feed_pre_exc","feed_pre_inh","sugar2_exc","sugar2_inh",
  "bitter2","grn_sugar","grn_water","grn_bitter","grn_pher","jo_aud","jo_groom","photoreceptor","lamina","motion_in","t4t5",
  "lc4","lplc2","lc_loom2","lc_freeze","orn","alpn","alln","kc","apl","mbon_avoid","mbon_approach","pam","ppl1",
  "epg","pen","delta7","ring","pfl3","dfb_sleep","p1","mal","lal_ps","dms2","song_vnc","an_steer","leg_premotor")
# Overlap resolution for the partition: first match in READOUTS order wins ("grn_sugar" therefore excludes nothing
# because "grn_sugar_labellar" is not a readout; "lc_loom" is derived = lc4 + lplc2 in RateEstimator.derived()).
# Consequences of the order: "pip10" precedes "song_dn" (so song_dn = pMP2 only), "b1"/"i1"/"hg1" precede "wing_steer",
# "dms2" precedes "song_vnc", "dng100" precedes "dn_fwd" (dn_fwd = DNge053 + DNg97). 70 readouts, 26 of them SIDED ->
# 70 + 52 + 9 derived = 131 keys in tick.rates.pops (section d.2).

STAR_TYPES: tuple[str, ...] = ("DNp01","DNa02","DNa01","DNg13","DNp09","DNg100","DNg02_a","MN9","TTMn","PSI","PFL3","EPG",
  "MBON01","MBON11","PAM01","PPL101","LB3b","LC4","LPLC2","LC9","pC1_14a","pIP10","dMS2","hg1 MN","DLMn c-f","R1-R6","L1","Mi1","T4a","ORN_DM1","KCg-m")

def region_of(superclass: str | None, cls: str | None, type_: str | None) -> int:
    """The 8-rule cascade of RESEARCH section 6. Returns REGION_ID value."""

def side_of(soma_side: str | None, instance: str | None = None) -> int:
    """'L'->-1, 'R'->+1, else 0; fallback: instance endswith '_L'/'_R'."""

def resolve_groups(types: list[str], type_idx: np.ndarray, side: np.ndarray) -> dict[str, np.ndarray]:
    """fullmatch every GROUP_REGEX over the UNIQUE type list (fast), expand to sorted int32 index arrays, add
    '<name>_L'/'<name>_R' for SIDED names. Missing groups are present as empty arrays (never KeyError downstream)."""

def readout_partition(groups: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, list[str]]:
    """pop_id int16[n] (-1 = none) and the list of readout names in READOUTS order (sided ones expanded to
    name, name_L, name_R as separate ids so that name == name_L + name_R)."""

def validate_groups(conn: "Connectome", required: Iterable[str] = ("grn_sugar","lc_loom","gf","feed_mn","steer_a02",
        "dng100","flight_dn","epg","pfl3","p1","escape_vnc","wing_power")) -> dict[str, int]:
    """Return {group: size}; raise ValueError listing every required group that resolved to zero neurons."""
```

### c.5 `flybrain/connectome/patches.py`

```python
@dataclass(frozen=True)
class GapJunction:
    pre: str; post: str; weight: float; laterality: str   # laterality: ipsi|contra|both
GAP_JUNCTIONS: tuple[GapJunction, ...] = (
    GapJunction("DNp01", "TTMn", 300.0, "ipsi"),     # [E] electrical GF->TTMn proxy (chemical count is 45/side)
    GapJunction("DNp01", "PSI",  200.0, "ipsi"),     # [E] electrical GF->PSI proxy (chemical 4/side)
    GapJunction("DNp01", "DNp01", 80.0, "contra"),   # [E] GF<->GF coupling
    GapJunction("LC4",   "DNp01",  6.0, "ipsi"),     # [E] dendro-dendritic proxy, added on top of chemical counts
    GapJunction("LPLC2", "DNp01",  6.0, "ipsi"),
)
def apply_patches(conn: Connectome, table: Sequence[GapJunction] = GAP_JUNCTIONS) -> Connectome:
    """Add (or increase) UNSIGNED weight of the listed type-to-type edges with laterality, keeping pre sorted and no
    duplicates; sign of the patched edge follows sign[pre] (all listed pres are excitatory). Append
    {"pre","post","weight","laterality","edges_added"} to meta['patches_applied']. Idempotent (skips if already applied)."""
```

### c.6 `flybrain/connectome/synthetic.py` (full generator spec in section g)

```python
@dataclass(frozen=True)
class Pop:
    type: str; region: str; n: int; nt: str; superclass: str; cls: str = ""
    # n = both sides combined; sides split ceil(n/2) L, floor(n/2) R; midline types (n==1) get side 0

@dataclass(frozen=True)
class Proj:
    pid: str                      # "P001".."P112"
    src: str; dst: str            # group name, type, or '|'-separated type list; wildcards '*' allowed (fnmatch)
    k_in: int                     # presynaptic partners per postsynaptic cell ('all' topology ignores k_in)
    r_nom_hz: float               # nominal presynaptic rate for calibration
    target_mv: float              # steady-state g target at r_nom (calibrated mode)
    laterality: str               # ipsi|contra|both|"ipsi/contra per row"
    topology: str                 # random|all|retinotopic|glomerular|wedge|ring_shift
    w_lit: float                  # literature mean synapses per pair (0 = engineered edge, no literature value)
    provenance: str               # "V" | "D" | "E"
    w_cv: float = 0.5             # lognormal jitter of per-edge weight (both modes)

POPULATIONS: tuple[Pop, ...]             # 289 rows of section g.1
PROJECTIONS: tuple[Proj, ...]            # 112 rows of section g.3
REGION_FRAC: dict[str, float] = {"optic_lobe": 0.605, "antennal_lobe": 0.023, "mushroom_body": 0.027,
    "central_complex": 0.018, "sez": 0.024, "central_other": 0.160, "descending_motor": 0.013, "vnc": 0.130}
FILLER_NT: tuple[tuple[str, float], ...] = (("acetylcholine", .62), ("glutamate", .18), ("gaba", .13), ("histamine", .04), ("unknown", .03))
REGION_ADJACENCY: dict[str, tuple[str, ...]]   # section g.4

def calibrated_weight(k_in: int, r_nom_hz: float, target_mv: float, w_syn: float = 0.275, tau_s: float = 5.0,
                      gain: float = 1.0, w_max: int = 400) -> int:
    """w = clip(round(target_mv / (k_in * w_syn * gain * r_nom_hz/1000 * tau_s)), 1, w_max).
    Steady-state g at nominal presynaptic rate equals target_mv (RESEARCH section 7)."""

def scale_populations(n_target: int, pops: Sequence[Pop] = POPULATIONS) -> tuple[float, list[Pop]]:
    """s = clip((n_target - 2000)/18000, 0.25, 1.0). Populations with n >= 40 are scaled to max(4, round(n*s)) (kept
    even when scaled); populations with n < 40 are never scaled. At n_target >= 20000, s == 1 and the core is exactly
    the table (6,535 neurons). Returns (s, pops)."""

def region_plan(n_target: int) -> dict[str, tuple[int, int, int]]:
    """{region: (core, filler, total)} using REGION_FRAC on the remaining budget: filler_r = floor(FRAC_r * (N - core_total)),
    remainder to optic_lobe. Raises ValueError if N < core_total. Section g.2 lists the expected values for N=20000."""

def build_synthetic(n_neurons: int = 20_000, seed: int = 1337, mean_outdeg: int = 25,
                    weights: str = "calibrated", inhib_frac: float = 0.33) -> Connectome:
    """Deterministic (SeedSequence child [0]). Steps: scale_populations -> region_plan -> assign indices region by region
    in POPULATIONS order then filler -> types/sides/nt/sign/region -> PROJECTIONS (topology rules of section g.3, weight
    per mode) -> background edges (section g.4) -> apply_patches -> sum duplicates -> build arrays -> resolve_groups ->
    validate. meta: weights_mode, gain_default (1.0 calibrated / 0.65 literature), engineered_edges (pids with 'E'),
    projections (resolved w per pid), region_counts, license='synthetic (no data)', citation=None,
    note='synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data'."""

def export_csv(conn: Connectome, out_dir: Path) -> tuple[Path, Path]:
    """Write neurons.csv + connections.csv in the neuPrint-export schema of c.7 so that load_csv_dir round-trips
    (test_synthetic::test_roundtrip)."""
```

### c.7 `flybrain/connectome/loaders.py`

```python
class Schema(str, Enum):
    NEUPRINT = "neuprint"        # neurons.csv: bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt[,predictedNt]
                                 # connections.csv: bodyId_pre,bodyId_post,weight
    CODEX_FAFB = "codex_fafb"    # classification.csv.gz (root_id,flow,super_class,class,sub_class,hemilineage,side,nerve)
                                 # + consolidated_cell_types.csv.gz (root_id,primary_type,...) + neurons.csv.gz (root_id,...,nt_type,...)
                                 # + connections.csv.gz (pre_root_id,post_root_id,neuropil,syn_count,nt_type)
    CODEX_MCNS = "codex_mcns"    # 'Root ID','Super Class','Class','Soma side','Primary Cell Type','Predicted NT type' + connections_princeton.csv.gz

NT_ALIASES: dict[str, str] = {"ach": "acetylcholine", "acetylcholine": "acetylcholine", "glut": "glutamate", "glu": "glutamate",
    "glutamate": "glutamate", "gaba": "gaba", "his": "histamine", "histamine": "histamine", "da": "dopamine", "dopamine": "dopamine",
    "oct": "octopamine", "octopamine": "octopamine", "ser": "serotonin", "5ht": "serotonin", "serotonin": "serotonin",
    "unc": "unknown", "unclear": "unknown", "unknown": "unknown", "": "unknown", "none": "unknown", "nan": "unknown"}
NT_SIGN: dict[str, float] = {"acetylcholine": 1, "dopamine": 1, "octopamine": 1, "serotonin": 1, "unknown": 1,
                             "gaba": -1, "glutamate": -1, "histamine": -1}

def nt_normalise(s: str | None) -> str:
    """lower().strip(); map through NT_ALIASES; unrecognised -> 'unknown'."""
def nt_sign(s: str | None) -> float:
    """NT_SIGN[nt_normalise(s)]."""

def detect_schema(path: Path) -> Schema:
    """Sniff the header of neurons.csv(.gz) / classification.csv.gz in `path`; raise ValueError if none matches."""

def load_csv_dir(path: Path, *, min_weight: int = 3, subset: str = "core", n_max: int | None = 20_000,
                 seed: int = 0, name: str | None = None) -> Connectome:
    """Read the schema's files (.csv or .csv.gz; numeric columns via np.loadtxt(usecols, dtype=int64); strings via csv
    module). Keep rows with status in {'Traced', ''} when a status column exists. Sides via side_of(). NT: consensusNt
    > predictedNt > nt_type > 'Predicted NT type'; edges without nt use the presynaptic neuron's NT. Sum duplicate
    (pre,post) rows (Codex has one row per neuropil). subset='core': keep every neuron that matches any GROUP_REGEX
    or class in {gustatory, Kenyon_Cell, MBON, DAN, CX, ALPN, ALLN} or superclass in {descending_neuron, cb_motor,
    vnc_motor, vnc_efferent}, then random-fill to n_max (seeded); subset='all': everything. Prune |weight| < min_weight.
    source='csv'; meta['license'] = 'CC-BY 4.0 (MaleCNS)' for NEUPRINT/CODEX_MCNS, 'CC-BY-NC 4.0 (FlyWire)' for CODEX_FAFB."""

def load_connectome(settings: Settings) -> Connectome:
    """Dispatch: synthetic -> build_synthetic(...); csv -> load_csv_dir(settings.connectome_dir, ...);
    neuprint -> load_csv_dir(settings.data_dir/'connectome'/'neuprint', ...) (raise FileNotFoundError with the
    exact fetch command if missing). Always: cache.load_cached(key) first, else build + store_cached; then apply_patches
    (idempotent) and validate_groups. Never performs network I/O."""
```

### c.8 `flybrain/connectome/cache.py`

```python
def cache_key(settings: Settings, extra: Mapping[str, object] | None = None) -> str:
    """sha1 over sorted repr of (connectome_source, connectome_dir + mtimes of its files, n_neurons, mean_outdeg,
    synth_weights, subset, min_weight, seed, flybrain.__version__, GENERATOR_VERSION)[:12]."""
def load_cached(key: str, cache_dir: Path, mmap: bool = False) -> Connectome | None: ...
def store_cached(key: str, conn: Connectome, cache_dir: Path) -> Path: ...
```

### c.9 `flybrain/snn/params.py`

```python
@dataclass(frozen=True)
class LIFParams:
    """Shiu et al. 2024 (Nature 634:210) defaults [L]."""
    v_rest: float = -52.0; v_reset: float = -52.0; v_th: float = -45.0   # mV
    tau_m: float = 20.0; tau_s: float = 5.0; t_ref: float = 2.2; delay: float = 1.8   # ms
    w_syn: float = 0.275          # mV per synapse
    f_poi: float = 250.0          # Poisson forcing kick = f_poi * w_syn = 68.75 mV [D]
    def constants(self, dt_ms: float) -> "StepConstants": ...

@dataclass(frozen=True)
class StepConstants:
    dt_ms: float; a_m: float; a_s: float; b: float; ref_steps: int; delay_steps: int; kick_mv: float
    # a_m = exp(-dt/tau_m); a_s = exp(-dt/tau_s); b = tau_s/(tau_m - tau_s)*(a_m - a_s);
    # ref_steps = round(t_ref/dt); delay_steps = round(delay/dt); kick_mv = f_poi*w_syn
    # dt 1.0 -> 0.951229 / 0.818731 / 0.044166 / 2 / 2 ; dt 0.5 -> 0.975310 / 0.904837 / 0.023491 / 4 / 4 ;
    # dt 0.1 -> 0.995012 / 0.980199 / 0.004938 / 22 / 18   (RESEARCH section 7, verified)

def current_from_rate(rate_hz: float, p: LIFParams = LIFParams()) -> float:
    """I (mV) such that an isolated neuron with steady offset I fires at rate_hz:
    I = (v_th - v_rest) / (1 - exp(-((1000/rate_hz) - t_ref)/tau_m)); 100 Hz -> 21.68 mV. rate_hz <= 0 -> 0.0."""
def rate_from_current(i_mv: float, p: LIFParams = LIFParams()) -> float:
    """Inverse; 0.0 when i_mv <= v_th - v_rest."""
```

### c.10 `flybrain/snn/engine.py`

```python
@dataclass(frozen=True)
class Drive:
    group: str                    # key of Connectome.groups (sided keys allowed)
    rate_hz: float                # Poisson rate per recruited neuron (or the rate converted by current_from_rate)
    recruit: float = 1.0          # fraction of the group driven (0..1)
    side: int = 0                 # 0 both, -1 L, +1 R (filters by Connectome.side)
    episode: int = 0              # deterministic recruitment permutation id: rng = seeded from (seed, group, episode)
    mode: str = "poisson"         # poisson|current
    weights: np.ndarray | None = None   # optional per-neuron multiplier of rate_hz (len == group size), e.g. flicker phase

@dataclass(frozen=True)
class Injection:
    """A Drive with an expiry, produced by inject(); cleared automatically when t_ms >= until_ms."""
    drive: Drive; until_ms: int; tag: str

@dataclass
class StepStats:
    n_steps: int
    t0_ms: int; t1_ms: int
    spike_counts_by_group: dict[str, int]   # readout partition names (incl. sided) -> spikes in window
    region_counts: np.ndarray               # int64[8]
    total_spikes: int
    active_frac_max: float                  # max over steps of spikes/n
    spike_indices_sample: np.ndarray        # int32[k] raster slots (see SpikeMonitor) ...
    spike_dt_sample: np.ndarray             # int16[k] step offset within the window (parallel to slots), capped
    capped: bool
    rates: dict[str, float]                 # RateEstimator.snapshot() after the window (Hz per neuron)
    region_rates: np.ndarray                # float32[8] Hz per neuron per region (EMA)
    forced_events: int; edge_visits: int; step_ms_mean: float; gf_spikes: dict[str, int]   # {"L": n, "R": n}
    noise_on: bool; gain: float

class LIFEngine:
    def __init__(self, connectome: Connectome, params: LIFParams = LIFParams(), seed: int = 0,
                 backend: str = "numpy", dt_ms: float = 1.0, gain: float = 1.0,
                 noise_mu: float = 0.5, noise_sigma: float = 3.5, drive_mode: str = "poisson",
                 monitor: "SpikeMonitor | None" = None) -> None:
        """State: v float32[n] (=v_rest), g float32[n] (=0), ref int16[n] (=0), i_ext float32[n] (=0),
        tonic float32[n] (constant currents from the 'tonic' table below), ring float32[delay_steps+1, n], t_step int.
        data_mv = csr.data * w_syn * gain (float32, cached; recomputed by set_gain). rng = Generator from SeedSequence child [1].
        backend 'auto' -> torch if csr.e > 5_000_000 and torch imports else numpy."""
    # --- injection API ---
    def inject(self, target: str | np.ndarray, *, rate_hz: float | None = None, current_mv: float | None = None,
               duration_ms: float = 50.0, recruit: float = 1.0, side: int = 0, episode: int = 0,
               tag: str = "", weights: np.ndarray | None = None) -> Injection:
        """Group name or explicit int32 indices. Exactly one of rate_hz/current_mv. Replaces any active injection with
        the same tag (tags are how the encoder updates a channel every tick). Returns the Injection."""
    def clear_injections(self, tag: str | None = None) -> None: ...
    def set_tonic(self, target: str | np.ndarray, current_mv: float) -> None:
        """Permanent constant current (mV offset of the resting potential), e.g. lamina/medulla 7.3 mV [E]."""
    def set_gain(self, gain: float) -> None
    def set_noise(self, mu: float, sigma: float) -> None
    # --- stepping ---
    def step(self, n_steps: int = 1) -> StepStats:
        """Run n_steps of dt. Per step, in THIS order:
          1. slot = t_step % (D+1); g_in = ring[slot] (view); expire injections whose until_ms <= t_ms
          2. active = ref == 0
             v[active] = v_rest + I[active] + (v[active] - v_rest - I[active])*a_m + b*g[active]     # I = i_ext + tonic
             g[active] *= a_s ;  ref[~active] -= 1                                                  # refractory: v,g frozen
          3. fired = active & (v > v_th) ; spk = flatnonzero(fired)
          4. g += g_in + noise + forced kicks:  g += rng.normal(mu*dt, sigma*sqrt(dt), n) (if sigma > 0);
             for each poisson Injection: p = rate*dt/1000 per recruited neuron -> g[idx[hit]] += kick_mv;
             current injections set i_ext (rebuilt from active injections when they change)
          5. reset: v[spk] = v_reset ; g[spk] = 0 ; ref[spk] = ref_steps
          6. ring[slot] = 0 ; propagate(spk) -> ring[(t_step + D) % (D+1)] += data_mv over outgoing edges
          7. monitor.record(spk, step_i) ; t_step += 1
        Returns StepStats for the window (monitor.flush()). Latency contract: an isolated neuron receiving one
        68.75 mV kick at step k fires at 3.0 ms <= t <= 3.0 ms + dt (test_engine::test_forced_kick_latency)."""
    def propagate(self, spk: np.ndarray, out: np.ndarray) -> int:
        """Delegates to the Propagator; returns edge visits."""
    # --- state ---
    @property
    def t_ms(self) -> int: ...
    @property
    def n(self) -> int: ...
    def state_dict(self) -> dict[str, np.ndarray | int | float]: ...
    def load_state_dict(self, d: dict) -> None: ...
    def reseed(self, seed: int) -> None: ...
    def rates(self) -> dict[str, float]: ...          # monitor.rates.snapshot()
```

Tonic-current table (applied by `SimulationLoop` at start, all `[E]`): `lamina` 7.3 mV, `motion_in` 7.3 mV
(≈15 Hz baseline so histaminergic photoreceptor input can modulate them), `feed_pre_inh` 7.05 mV (≈10 Hz tonic
inhibition of MN9, released by the GNG042→GNG015 disinhibition), `mal` 7.05 mV (tonic inhibition of pC1).

### c.11 `flybrain/snn/propagators.py`

```python
class Propagator(Protocol):
    def __call__(self, spk: np.ndarray, out: np.ndarray) -> int:
        """out[post] += data_mv[edge] for every outgoing edge of every index in spk (int32, sorted). Returns edge visits."""
    def set_data(self, data_mv: np.ndarray) -> None: ...

class NumpyPropagator:
    def __init__(self, csr: CSR, data_mv: np.ndarray, concat_threshold: int = 4000) -> None: ...
    # len(spk) <= threshold: python slices + np.concatenate ; else: vectorised flat-slot gather
    # (ln = b-a; seg = cumsum(ln)-ln; idx = repeat(a-seg, ln) + arange(ln.sum())); scatter with np.add.at(out, tg, wv)

class TorchPropagator:
    def __init__(self, csr: CSR, data_mv: np.ndarray, threads: int | None = None, device: str = "cpu",
                 deterministic: bool = True) -> None:
        """torch imported here. indices/data as tensors (indices int64). gather via repeat_interleave, scatter via
        out_t.index_add_(0, tg, wv) then copy back into out (numpy). torch.set_num_threads(min(8, os.cpu_count()-2)).
        deterministic=True -> torch.use_deterministic_algorithms(True) (~15% slower)."""

def make_propagator(backend: str, csr: CSR, data_mv: np.ndarray) -> Propagator:
    """'numpy' | 'torch' | 'auto' (torch if csr.e > 5_000_000 and importable, else numpy; logs the choice)."""
```

### c.12 `flybrain/snn/monitor.py`

```python
@dataclass(frozen=True)
class RasterRow:
    region: int; slot: int; neuron: int; label: str; side: str; star: bool   # side 'L'|'R'|'M'

class SpikeMonitor:
    def __init__(self, conn: Connectome, per_region: int = 48, cap: int = 2000, seed: int = 0) -> None:
        """pop_id/readout names from groups.readout_partition; region uint8 from conn. Raster sample: per region,
        first every STAR_TYPES neuron present in that region (L then R), then a seeded random sample of the rest, up to
        per_region rows; slot_of_neuron int32[n] (-1 = unsampled); rows list of RasterRow with slot = region*per_region + j."""
    rows: list[RasterRow]
    slot_of_neuron: np.ndarray
    def begin(self, t0_ms: int) -> None: ...
    def record(self, spk: np.ndarray, step_i: int) -> None:
        """np.bincount(pop_id[spk][pop_id[spk] >= 0]) into pop_counts; region bincount; append raster (slot, step_i)
        for sampled spikes; count DNp01 spikes per side; track max active fraction."""
    def flush(self) -> StepStats:
        """Build StepStats (uniform subsample of raster events to cap, capped=True when truncated), update
        RateEstimator, reset counters."""
    rates: "RateEstimator"

class RateEstimator:
    TAU_S: dict[str, float]   # readout -> EMA time constant (s): gf/escape_dn/dn_saccade/ttmn/psi/gfc2 0.05;
                              # steering/dn_fwd/dng100/dn_halt/dn_back/flight_dn/dn_freeze/groom_dn/wing_* 0.15;
                              # feed_mn/feed_*/sugar2_*/mbon_*/pam/ppl1/p1/song_*/pip10/dms2 0.30; default 0.20; regions 0.25
    def __init__(self, names: list[str], sizes: np.ndarray, tick_s: float) -> None: ...
    def update(self, counts: np.ndarray, region_counts: np.ndarray, region_sizes: np.ndarray, window_s: float) -> None:
        """r <- r*exp(-window/tau) + (counts/(size*window))*(1-exp(-window/tau)); Hz per neuron; size 0 -> 0."""
    def snapshot(self) -> dict[str, float]:
        """All readouts + derived: lc_loom = size-weighted mean of lc4/lplc2; escape_vnc = mean(ttmn, psi, gfc2);
        steer_a02_diff = steer_a02_R - steer_a02_L (same for a01, g13, b1, i1, hg1); dn_mean = descending_motor region rate."""
    def regions(self) -> np.ndarray: ...     # float32[8]
```

### c.13 `flybrain/snn/calibrate.py`

```python
@dataclass(frozen=True)
class CalibTargets:
    rest_rate_min_hz: float = 1.0; rest_rate_max_hz: float = 5.0      # mean over all neurons, 1 s, noise on, no drives
    rest_active_frac_max: float = 0.02                                # max per-step spikes/n at rest
    sugar_rate_hz: float = 100.0; mn9_min_hz: float = 20.0; mn9_max_hz: float = 90.0   # grn_sugar_labellar 100 Hz for 1 s
    loom_rate_hz: float = 150.0; gf_latency_max_ms: float = 20.0      # lc_loom 150 Hz, recruit 1.0 -> first DNp01 spike
    runaway_active_frac: float = 0.05                                 # never exceeded for 10 consecutive steps
    pfl3_rate_hz: float = 60.0; a02_diff_min_hz: float = 10.0         # pfl3_L 60 Hz for 500 ms -> steer_a02_R - steer_a02_L

@dataclass
class GateReport:
    rest_rate_hz: float; rest_active_frac: float; mn9_hz: float; gf_latency_ms: float | None; a02_diff_hz: float
    runaway: bool; rtf: float; gain: float; passed: dict[str, bool]; notes: list[str]
    def ok(self) -> bool: ...

def run_gates(conn: Connectome, settings: Settings, gain: float, targets: CalibTargets = CalibTargets()) -> GateReport:
    """Fresh LIFEngine(seed=settings.seed). (1) rest 1000 ms; (2) reset state, sugar for 1000 ms, MN9 rate over the last
    500 ms; (3) reset, loom, run 200 ms, latency to the first DNp01 spike; (4) reset, pfl3_L Poisson pfl3_rate_hz for
    500 ms, a02_diff_hz = steer_a02_R - steer_a02_L over the last 250 ms; (5) runaway flag from all runs; (6) rtf =
    brain ms / wall ms over run (1). The five gates are section g.6. Deterministic."""

def calibrate_gain(conn: Connectome, settings: Settings, targets: CalibTargets = CalibTargets(),
                   lo: float = 0.05, hi: float = 1.5, iters: int = 10) -> dict:
    """Bisection on gain (log scale): raise gain while MN9 < min and no runaway; lower while runaway or MN9 > max.
    Writes data/cache/<connectome_key>.calib.json {gain, report, targets, created}. Returns that dict. Literature-weight
    mode and real data only (calibrated synthetic uses gain 1.0 and merely runs run_gates in selftest)."""
```

### c.14 `flybrain/market/base.py`

```python
@dataclass(frozen=True)
class Trade:
    ts: float                     # wall seconds
    kind: str                     # "buy" | "sell"
    usd: float
    price: float                  # price_usd at the trade
    surrogate: bool = False       # True when synthesised between DexScreener polls

@dataclass
class MarketSnapshot:
    ts: float; source: str        # "sim" | "dexscreener" | "sim(fallback)"
    chain: str; dex: str; pair: str; symbol: str
    price_usd: float | None; price_native: float | None
    buys_m5: int; sells_m5: int; buys_h1: int; sells_h1: int
    chg_m5: float; chg_h1: float; chg_h6: float; chg_h24: float          # percent
    vol_m5: float; vol_h1: float
    liq_usd: float | None; fdv: float | None; mcap: float | None
    regime: str | None = None     # sim only: CALM|PUMP|DUMP|CHOP|RUG|DEAD
    seq: int = 0                  # increments on every new snapshot
    def to_wire(self) -> dict: ...  # the "market" WS message body (section d)

class MarketSource(Protocol):
    name: str
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def poll(self, now: float, dt_s: float) -> tuple[MarketSnapshot | None, list[Trade]]:
        """Non-blocking. Returns (new snapshot or None if unchanged, trades since the last call)."""
    def set_regime(self, name: str, seconds: float) -> bool:
        """Sim only; other sources return False."""
```

### c.15 `flybrain/market/sim.py`

```python
@dataclass(frozen=True)
class Regime:
    name: str; mu: float; sigma: float; lam0: float; p_buy: float; dwell_s: float
    # per-second drift/vol of ln P, Hawkes baseline trades/s, buy probability, mean dwell
REGIMES: tuple[Regime, ...] = (
    Regime("CALM", 0.0,     0.002,  0.30, 0.50, 90.0),
    Regime("PUMP", +0.0015, 0.006,  1.50, 0.82, 40.0),
    Regime("DUMP", -0.0020, 0.008,  1.50, 0.18, 30.0),
    Regime("CHOP", 0.0,     0.012,  0.80, 0.50, 45.0),
    Regime("RUG",  -0.0060, 0.015,  2.00, 0.10, 20.0),   # liquidity also drains 2%/s
    Regime("DEAD", 0.0,     0.0005, 0.02, 0.50, 120.0),
)
class SimulatedMarket:
    name = "sim"
    def __init__(self, seed: int, price0: float = 0.001, supply: float = 1e9, liq0: float = 50_000.0,
                 regimes: Sequence[Regime] = REGIMES, dwell_scale: float = 1.0,
                 hawkes_alpha: float = 0.4, hawkes_beta: float = 0.2, whale_p: float = 0.01, whale_mult: float = 30.0,
                 snapshot_period_s: float = 1.0) -> None: ...
    def poll(self, now, dt_s):
        """Advance by dt_s: regime switch with P = dt/(dwell*dwell_scale) (uniform over the other regimes);
        ln P += mu*dt + sigma*sqrt(dt)*Z + impact; trades ~ Poisson(lam(t)*dt), lam = lam0 + sum alpha*exp(-beta*(t-t_i))
        over the last 60 s; size ~ LogNormal(ln 80, 1.2) USD (x whale_mult with prob whale_p); impact = +-2e-4*sqrt(usd/100)
        on ln P; RUG: liq *= (1 - 0.02*dt). A new snapshot every snapshot_period_s from the 1-h trade ring (m5/h1 windows)
        and price ring (chg_* from prices 5 min/1 h/6 h/24 h ago or the earliest available); symbol 'FLY', chain 'sim'."""
    def set_regime(self, name, seconds) -> bool: ...   # forces the regime for `seconds`, then resumes Markov switching
```

### c.16 `flybrain/market/dexscreener.py` and `flybrain/market/feed.py`

```python
DEX_BASE = "https://api.dexscreener.com"
def fetch_pairs(chain: str, token: str, timeout_s: float = 10.0) -> list[dict]:
    """GET {DEX_BASE}/tokens/v1/{chain}/{token} via httpx with User-Agent 'synapsefly/0.1'. Returns the bare list
    ([] for unknown tokens is a valid answer). Raises httpx.HTTPError / ValueError on transport/parse failure."""
def parse_pair(p: dict, now: float, seq: int) -> MarketSnapshot:
    """Tolerant: priceUsd/priceNative are strings -> float or None; txns/volume/priceChange sub-keys default 0;
    liquidity.usd/fdv/marketCap absent -> None. source='dexscreener', dex=p['dexId'], pair=p['pairAddress'],
    symbol=p['baseToken']['symbol'], chain=p['chainId']."""
def choose_pair(pairs: list[dict]) -> dict | None:
    """Max liquidity.usd (missing -> 0); None for []."""

class SurrogateTrades:
    def __init__(self, rng: np.random.Generator) -> None: ...
    def step(self, snap: MarketSnapshot, prev: MarketSnapshot | None, dt_s: float, now: float) -> list[Trade]:
        """Rate-matched Poisson surrogate between polls: lam_buy = buys_m5/300 per s, lam_sell = sells_m5/300;
        usd ~ LogNormal around vol_m5/max(1, buys_m5+sells_m5); price = snap.price_usd; surrogate=True.
        When a fresh snapshot arrives, emit max(0, delta buys/sells since prev) real-delta trades first (surrogate=False)."""

class DexScreenerSource:
    name = "dexscreener"
    def __init__(self, chain: str, token: str, poll_s: int = 60, seed: int = 0) -> None: ...
    def start(self) -> None: ...   # background thread: fetch every poll_s, store latest, count consecutive failures
    def stop(self) -> None: ...
    def poll(self, now, dt_s): ...  # returns (latest snapshot if new seq else None, surrogate.step(...))
    @property
    def failures(self) -> int: ...
    def set_regime(self, name, seconds) -> bool: return False

class MarketFeed:
    """Owns the active source with fallback. Not a thread itself (DexScreenerSource has one)."""
    def __init__(self, settings: Settings, seed: int) -> None:
        """market='sim' -> SimulatedMarket(seed, dwell_scale=sim_regime_s/90). market='dexscreener' -> DexScreenerSource
        + a SimulatedMarket fallback; after 3 consecutive failures serve the fallback (source='sim(fallback)') and emit
        {'kind':'market_source','source':'sim(fallback)','reason':...}; resume on the next success."""
    def start(self) -> None; def stop(self) -> None
    def poll(self, now: float, dt_s: float) -> tuple[MarketSnapshot, list[Trade], list[dict]]:
        """Always returns a snapshot (the last known one, never None after start) plus new trades plus events."""
    @property
    def mode(self) -> str: ...     # "sim" | "dexscreener" | "sim(fallback)"
    def set_mode(self, mode: str) -> bool:
        """'sim' | 'dexscreener' | a regime name (CALM|PUMP|DUMP|CHOP|RUG|DEAD) which forces the sim regime for 60 s."""
```

### c.17 `flybrain/encoder.py`

```python
def hill(x: float, k: float, n: float = 1.5) -> float:
    """x^n/(k^n + x^n) for x >= 0, else 0."""

@dataclass
class Features:                   # all in [0,1] unless stated; computed every tick
    t_ms: int
    bp: float; mom: float; tick: float        # buy pressure -1..1, tanh(chg_m5/2), tanh(100*(P/P_prev-1))
    val: float; activity: float; rug: float   # EMAs
    up: float; down: float                    # sat(val), sat(-val)
    sugar: float; bitter: float; water: float; looming: float; loom_side: int; flash: float
    loom_episode: int; loom_t_ms: int | None  # current episode id, ms since the episode's ramp start
    sustained_loom: bool
    vol_norm: float; odor: float; chop: float
    courtship: float                          # C
    sleep_pressure: float
    hunger: float; sweet_gain: float
    explore: float
    candle: str                               # "up" | "down" | "flat"
    since_last_trade_s: float
    whale: Trade | None
    usd_ref: float

class FeatureExtractor:
    def __init__(self, settings: Settings, seed: int) -> None: ...
    def update(self, snap: MarketSnapshot, trades: list[Trade], now: float, dt_s: float, mood: str,
               feeding: bool, mn9_hz: float) -> Features:
        """Formulas in section f.1. Keeps: EMAs, trade impulses, looming episodes, price ring (90 s), session ATH,
        usd_ref running median (floor 20), hunger."""
    def easter_egg(self, snap: MarketSnapshot, now: float) -> str | None:
        """FLY_EASTER_EGGS: buys_m5 in {69, 420}, leading significant digits of price '420'/'69', local time 04:20/16:20
        (once per minute) -> reason string, else None."""

@dataclass(frozen=True)
class Poke:
    stim: str                     # sugar|bitter|loom|water|dust|pheromone|sleep|reward|punish|explore
    strength: float               # 0..1
    side: int                     # -1|0|+1
    until_ms: int

class SensoryEncoder:
    def __init__(self, conn: Connectome, settings: Settings, seed: int) -> None:
        """Precomputes EPG wedge index per EPG neuron (wedge = floor(16 * rank_within_side / n_side)), photoreceptor
        phases phi_i ~ U(0, 2pi) (seeded), and the six food glomeruli ORN groups (DM1, DM4, DM2, VM2, VA2, DL1)."""
    def encode(self, f: Features, heading: float, mood: str, pokes: list[Poke], t_ms: int) -> list[Drive]:
        """Section f.2. Returns the complete list of Drives for the next tick (the loop calls engine.inject for each
        with duration = tick brain-ms, tag = drive.group + ':' + channel). Mood feedback only when
        settings.mood_feedback. Explore baseline only when settings.explore_baseline > 0."""
    POKE_TABLE: dict[str, tuple[str, float]]   # stim -> (group, rate_hz at strength 1): sugar (grn_sugar,150), bitter (grn_bitter,120),
                                               # loom (lc_loom,150), water (grn_water,60), dust (jo_groom,80), pheromone (grn_pher,100),
                                               # sleep (dfb_sleep,20), reward (pam,80), punish (ppl1,80), explore (dng100,40)
```

### c.18 `flybrain/decoder.py`

```python
@dataclass(frozen=True)
class Readouts:                   # Hz per neuron, EMA; taken from StepStats.rates
    fwd: float; halt: float; back: float; freeze: float
    a02_l: float; a02_r: float; a01_l: float; a01_r: float; g13_l: float; g13_r: float
    flight: float; wing_power: float; b1_l: float; b1_r: float; i1_l: float; i1_r: float; hg1_l: float; hg1_r: float
    saccade_l: float; saccade_r: float; land: float
    mn9: float; groom: float; p1: float; pip10: float; song_mn: float; dms2_l: float; dms2_r: float
    gf_spikes_l: int; gf_spikes_r: int; dnp11_spikes: int
    lc_loom: float; dn_mean: float
    @classmethod
    def from_stats(cls, s: StepStats) -> "Readouts": ...

@dataclass
class MotorCommand:
    mode: str                     # walk|fly|jump|feed|freeze|groom|court|sleep
    v_target: float; omega: float # px/s, rad/s
    wing_hz: float; wing_amp: float; wing_ext: int   # wing_ext: -1|0|+1 extended wing (song)
    proboscis: float              # 0..1
    jump: dict | None             # {"heading": rad, "impulse": px/s, "side": "L"|"R"|"both", "forward": bool}
    halt_reason: str | None       # "freeze"|"feed"|"groom"|"halt_dn"|"sleep"|None
    events: list[dict]            # kind: jump|takeoff|landing|freeze|unfreeze|feed_start|feed_stop|groom|song|saccade|wander_floor|sleep|wake

@dataclass
class Kinematics:
    x: float; y: float; vx: float; vy: float; heading: float; speed: float; omega: float
    wing_hz: float; wing_amp: float; wing_ext: int; mode: str; leg_phase: float; proboscis: float
    jump_t_ms: int | None         # ms since the jump started, None if not jumping
    def to_wire(self) -> dict: ...

@dataclass(frozen=True)
class InkStyle:
    color: str; width: float; alpha: float; style: str; stamp: str | None
    # style: solid|rainbow|zigzag|dotted|hearts ; stamp: None|blob|heart|zzz|dash|bump

class Wander:
    def __init__(self, seed: int, sigma: float = 0.6, tau_s: float = 1.0) -> None: ...
    def sample(self, dt_s: float) -> float:   # OU angular-velocity noise (rad/s): eta <- eta*exp(-dt/tau) + sigma*sqrt(dt)*N(0,1)

class MotorDecoder:
    def __init__(self, settings: Settings, seed: int) -> None: ...
    def update(self, r: Readouts, f: Features, mood: str, t_ms: int, dt_s: float) -> MotorCommand:
        """Section f.3. Holds timers: last_jump_ms, feed_enter_timer, groom_until, court_until, stall_timer (wander floor),
        saccade_until, dnp11_recent_ms."""

class FlyBody:
    def __init__(self, w: int, h: int, walls: str, seed: int) -> None:
        """Starts at (w/2, h/2), heading 0, mode walk."""
    def integrate(self, cmd: MotorCommand, dt_s: float, t_ms: int) -> tuple[Kinematics, list[dict]]:
        """Section f.4. Returns kinematics and events (wall_bump {side}, wrap)."""
    def ink(self, kin: Kinematics, f: Features, mood: str, t_ms: int) -> InkStyle:   # section f.5
    @property
    def kin(self) -> Kinematics: ...
    def teleport(self, x: float, y: float) -> None: ...
```

### c.19 `flybrain/mood.py`

```python
class Mood(str, Enum):
    SLEEP = "SLEEP"; CRUISING = "CRUISING"; FEEDING = "FEEDING"; EUPHORIA = "EUPHORIA"
    ANXIOUS = "ANXIOUS"; PANIC = "PANIC"; ESCAPE = "ESCAPE"; COURTSHIP = "COURTSHIP"
MOOD_PRIORITY: tuple[Mood, ...] = (ESCAPE, PANIC, COURTSHIP, EUPHORIA, FEEDING, ANXIOUS, SLEEP, CRUISING)
MOOD_COLORS: dict[str, str] = {"SLEEP": "#6b6b6b", "CRUISING": "#000000", "FEEDING": "#ff8c00", "EUPHORIA": "rainbow",
    "ANXIOUS": "#ffd700", "PANIC": "#ff0000", "ESCAPE": "#ff3300", "COURTSHIP": "#ff69b4"}

@dataclass(frozen=True)
class MoodInputs:
    t_ms: int; sugar: float; looming: float; bitter: float; activity: float; sleep_pressure: float
    mn9: float; dnp09: float; lc_loom: float; pam: float; ppl1: float; mbon_approach: float; mbon_avoid: float
    p1: float; pip10: float; dn_mean: float; feeding: bool; jumped: bool; gf_spikes_10s: int
    any_drive_max: float; poked: bool; forced: str | None

@dataclass
class MoodState:
    state: Mood; prev: Mood; since_ms: int
    euphoria: float; anxiety: float; arousal: float; valence: float; fear: float; hunger: float; sleep: float
    dwell_left_ms: int; timers: dict[str, int]
    def to_wire(self) -> dict: ...

@dataclass(frozen=True)
class Transition:
    t_ms: int; src: Mood; dst: Mood; reason: str

class MoodMachine:
    def __init__(self, tick_s: float, tau_score_s: float = 2.0) -> None: ...
    def update(self, x: MoodInputs) -> tuple[MoodState, Transition | None]:
        """Section f.6. Exactly one transition per tick at most."""
    def confirmed(self, hold_ms: int = 3000) -> Transition | None:
        """The last transition once the new state has been held for hold_ms (returned once) - the agent trigger."""
```

### c.20 `flybrain/agent/summary.py`

```python
def build_brain_summary(tick: dict, history: "TickHistory", reason: str, conn_meta: dict, session: dict,
                        market: dict, lang: str) -> dict:
    """The ONLY thing the LLM sees (<= 1.5 KB). Exact shape in section d.6."""
```

### c.21 `flybrain/agent/llm.py`

```python
SYSTEM_PROMPT: str   # section d.6 verbatim
TWEET_SCHEMA: dict   # {"type":"object","properties":{"text":{"type":"string"},"neurons":{"type":"array","items":{"type":"string"}}},"required":["text","neurons"],"additionalProperties":False}
VOCABULARY: tuple[str, ...] = ("sugar GRN", "LB3b", "MN9", "proboscis", "giant fiber", "DNp01", "LC4", "LPLC2", "looming",
    "DNa02", "mushroom body", "Kenyon", "PAM", "dopamine", "PPL1", "central complex", "EPG", "PFL3", "DNg02", "wingbeat",
    "pC1", "pIP10", "MBON", "connectome", "optic lobe", "antennal lobe", "descending neuron")

@dataclass
class TweetDraft:
    text: str; model: str; dry_run: bool; stop_reason: str | None; latency_ms: float; reason: str
    refused: bool = False; error: str | None = None; usage: dict | None = None; neurons: list[str] = field(default_factory=list)

class TweetGenerator:
    def __init__(self, settings: Settings, seed: int) -> None:
        """No anthropic import here. Client created lazily on the first non-dry-run call: anthropic.Anthropic()."""
    def generate(self, summary: dict) -> TweetDraft:
        """dryrun -> template_tweet. anthropic ->
           kwargs = dict(model=settings.llm_model, max_tokens=512, system=SYSTEM_PROMPT + lang_line,
                         messages=[{"role": "user", "content": json.dumps(summary, ensure_ascii=False)}])
           if settings.llm_json: kwargs["output_config"] = {"format": {"type": "json_schema", "schema": TWEET_SCHEMA}}
           if settings.llm_fallbacks and sdk_major >= 1: (use client.beta.messages.create with
                betas=["server-side-fallback-2026-07-01"], fallbacks="default") else client.messages.create(**kwargs)
           NO thinking parameter, NEVER budget_tokens, no assistant prefill, no temperature.
           text = "".join(b.text for b in resp.content if b.type == "text")
           if resp.stop_reason == "refusal": TweetDraft(refused=True) -> caller uses template (no cooldown consumed)
           if llm_json: json.loads(text)["text"] else text; on JSONDecodeError fall back to raw text.
           except anthropic.RateLimitError: sleep min(60, retry-after) once, retry once, else template
           except anthropic.APIStatusError as e: status >= 500 -> retry once else template (error=str(e))
           except anthropic.APIConnectionError: template
           Always validate_tweet(); if validation fails once, regenerate once, then template."""

def template_tweet(summary: dict, rng: random.Random) -> str:
    """Deterministic crypto-dialect text from >= 6 templates per reason (euphoria_entry, panic_entry, courtship_entry,
    escape_burst, manual), filled with numbers from the summary; always contains >= 1 VOCABULARY token, no URLs."""

def validate_tweet(text: str, require_vocab: bool = True) -> tuple[bool, str]:
    """Strip URLs (regex https?://\\S+|www\\.\\S+), collapse whitespace, strip surrounding quotes, cap at 280 at a
    word boundary, keep at most 2 hashtags (drop the rest), no cashtags other than $FLY. Returns (ok, cleaned) where
    ok requires len > 0 and (a VOCABULARY token present when require_vocab)."""
```

### c.22 `flybrain/agent/x_client.py`

```python
@dataclass
class PostResult:
    posted: bool; dry_run: bool; id: str | None; url: str | None; error: str | None; media_id: str | None

class XClient:
    def __init__(self, settings: Settings) -> None:
        """tweepy imported lazily. dry_run = settings.x_mode != 'post'. Missing credentials with x_mode='post' ->
        log ERROR and behave as dry-run (never crash)."""
    def post(self, text: str, png: bytes | None = None) -> PostResult:
        """dry-run: return PostResult(posted=False, dry_run=True) after logging '[X DRY RUN] <text> (+image N bytes)'.
        live: media_id = upload_media_v2(png) or upload_media_v1(png) or None;
        resp = tweepy.Client(consumer_key=X_API_KEY, consumer_secret=X_API_SECRET, access_token=X_ACCESS_TOKEN,
               access_token_secret=X_ACCESS_TOKEN_SECRET, wait_on_rate_limit=False).create_tweet(text=text[:280],
               media_ids=[media_id] if media_id else None)
        id = str(resp.data["id"]); url = f"https://x.com/i/web/status/{id}".
        tweepy.TooManyRequests -> disabled_until = reset_time or now+900; tweepy.Forbidden/Unauthorized -> disabled for the
        session (health shows it); other -> error string."""
    def upload_media_v2(self, png: bytes) -> str | None:
        """POST https://api.x.com/2/media/upload multipart files={'media': png}, data={'media_category':'tweet_image'},
        auth=requests_oauthlib.OAuth1(...). Returns data['id'] or None."""
    def upload_media_v1(self, png: bytes) -> str | None:
        """tweepy.API(tweepy.OAuth1UserHandler(...)).media_upload(filename='fly.png', file=io.BytesIO(png)).media_id_string"""
    @property
    def disabled_reason(self) -> str | None: ...
```

### c.23 `flybrain/agent/snapshot.py`

```python
def render_snapshot(trail: list[tuple[float, float, str, float]], kin: Kinematics, mood: str, market: dict,
                    w: int = 800, h: int = 500) -> bytes:
    """numpy RGB raster: white canvas with a 3 px #c0c0c0 Paint frame, a 20 px title bar '#000080' and a 20 px status
    strip with a 5x7 bitmap-font caption 'FLYBRAIN | <mood> | <symbol> <price> <chg_m5>%'; Bresenham segments for the
    trail (x, y, '#rrggbb', width); fly sprite = 9x9 glyph rotated to 8 headings, scaled 2x; encoded with server.png.write_png."""

class SnapshotBroker:
    def __init__(self, bus: "StateBus", body: FlyBody, settings: Settings) -> None: ...
    def record_trail(self, x: float, y: float, color: str, width: float) -> None:   # ring of the last 4000 points (sim thread)
    def request(self, timeout_s: float = 3.0) -> tuple[bytes, str]:
        """Ask the most recently active browser over the bus ({'type':'snapshot_request','id'}) and wait for
        {'type':'snapshot','id','png_b64'} (max 2 MB); on timeout/no client render_snapshot(). Returns (png, 'browser'|'server').
        Saves to data/snapshots/<t_ms>_<mood>.png."""
    def deliver(self, id: str, png_b64: str) -> None:   # called by the WS handler (asyncio thread), thread-safe
```

### c.24 `flybrain/agent/orchestrator.py`

```python
TRIGGER_REASONS: tuple[str, ...] = ("euphoria_entry", "panic_entry", "courtship_entry", "escape_burst", "manual")

@dataclass
class AgentState:                 # persisted as data/agent_state.json
    last_fire_wall: float; last_fire_by_reason: dict[str, float]; day: str; fires_today: int
    recent_hashes: list[str]; disabled_reason: str | None
    def load(path: Path) -> "AgentState"; def save(self, path: Path) -> None

class TweetAgent:
    def __init__(self, settings: Settings, generator: TweetGenerator, poster: XClient, snapshots: SnapshotBroker,
                 bus: "StateBus", history: "TickHistory", conn_meta: dict, seed: int) -> None: ...
    def on_tick(self, tick: dict, transition: Transition | None, confirmed: Transition | None, jumps_60s: int) -> None:
        """Sim thread; never blocks. Trigger when confirmed.dst in {EUPHORIA, PANIC, COURTSHIP} (reason '<mood>_entry')
        or jumps_60s >= 3 (reason 'escape_burst', once per 60 s). allowed(reason) -> submit fire() to the 1-worker
        executor; otherwise log the suppression reason."""
    def allowed(self, reason: str, now: float) -> tuple[bool, str]:
        """(a) now - last_fire >= tweet_cooldown_s; (b) now - last_fire_by_reason[reason] >= tweet_reason_cooldown_s;
        (c) fires_today < tweets_per_day (UTC day); (d) not in flight; (e) not disabled. reason == 'manual' bypasses a-c."""
    def fire(self, reason: str, tick: dict) -> dict:
        """summary -> generator.generate -> validate -> snapshot -> poster.post -> record:
        append {id, t_ms, wall, reason, text, model, dry_run, posted, url, error, snapshot_source, neurons, mood, latency_ms,
        summary} (the section d.4 record plus summary) to data/tweets.jsonl,
        write out/tweets/<YYYYmmdd-HHMMSS>.{txt,json,png}, update+save AgentState, bus.publish_event('tweet', ...).
        Dedupe: sha1(text) in recent_hashes[-20:] -> regenerate once via template then skip. Hard 30 s timeout per job."""
    def request_manual(self) -> None: ...
    def history(self, limit: int = 20) -> list[dict]: ...
    def status(self) -> dict: ...  # for /api/health
```

### c.25 `flybrain/server/state.py`, `session_log.py`, `png.py`

```python
class StateBus:
    """Sim thread -> asyncio bridge. Created in app lifespan with the running loop."""
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None: ...
    def publish_tick(self, tick: dict) -> None          # call_soon_threadsafe; keeps latest
    def publish_event(self, msg: dict) -> None          # any non-tick server->client frame
    def latest_tick(self) -> dict | None
    def hello(self) -> dict; def set_hello(self, hello: dict) -> None
    def subscribe(self, maxsize: int = 4) -> asyncio.Queue   # drop-oldest when full
    def unsubscribe(self, q: asyncio.Queue) -> None
    def client_count(self) -> int
    # client -> sim direction (thread-safe queues consumed by SimulationLoop once per tick):
    def push_poke(self, poke: Poke) -> None
    def push_command(self, cmd: dict) -> None           # {"name": "set_market_mode"|"clear"|"tweet_test", ...}
    def drain_pokes(self) -> list[Poke]; def drain_commands(self) -> list[dict]
    # snapshot round trip
    def request_snapshot(self, id: str) -> None; def wait_snapshot(self, id: str, timeout_s: float) -> bytes | None
    def deliver_snapshot(self, id: str, png: bytes) -> None

class TickHistory:
    def __init__(self, seconds: float = 10.0, tick_s: float = 0.05) -> None: ...
    def push(self, tick: dict) -> None; def last(self, seconds: float) -> list[dict]
    def events(self, seconds: float) -> list[dict]; def gf_spikes(self, seconds: float) -> int; def jumps(self, seconds: float) -> int

class SessionLog:
    def __init__(self, path: Path, settings: Settings) -> None:
        """First line: {"kind":"header","settings":redacted,"run_id","connectome_key","created"}."""
    def write_tick(self, seq: int, t_ms: int, wall: float, snap: MarketSnapshot | None, trades: list[Trade],
                   pokes: list[Poke], commands: list[dict], total_spikes: int) -> None
    def close(self) -> None
def read_session(path: Path) -> Iterator[dict]: ...

def write_png(rgb: np.ndarray) -> bytes:
    """uint8[H,W,3] -> PNG bytes (filter 0, zlib level 6, struct chunks; no PIL)."""
```

### c.26 `flybrain/server/protocol.py` (pydantic v2, `extra='forbid'` on client messages)

Every wire message of section d has a model: `HelloMsg, TickMsg, MoodChangeMsg, TweetMsg, MarketMsg, EventMsg,
SnapshotRequestMsg, PongMsg, ErrorMsg` (server → client) and `PokeMsg, SetMarketModeMsg, PingMsg, SnapshotMsg,
TweetTestMsg, ClearMsg` (client → server) plus REST bodies `PokeRequest, MarketModeRequest, SnapshotUpload` and responses
`HealthResponse, StateResponse`. `parse_client(raw: str) -> ClientMsg` validates with a discriminated union on `type`
and raises `ValueError` (the handler answers `{"type":"error","code":"bad_message"}`). `tick_to_json(tick: dict) -> str`
serialises once per tick (`json.dumps(separators=(',',':'), allow_nan=False)`; floats rounded: positions 1 dp,
rates 2 dp, drives 3 dp).

### c.27 `flybrain/server/loop.py`

```python
class SimulationLoop(threading.Thread):
    def __init__(self, settings: Settings, conn: Connectome, engine: LIFEngine, feed: MarketFeed,
                 features: FeatureExtractor, encoder: SensoryEncoder, decoder: MotorDecoder, body: FlyBody,
                 mood: MoodMachine, agent: TweetAgent | None, bus: StateBus, history: TickHistory,
                 session_log: SessionLog | None, replay: Iterator[dict] | None = None) -> None: ...
    speed: float                  # sim-ms per wall-ms in [0.25, 1.0]
    def run(self) -> None:
        """Startup: apply tonic table; publish hello. Loop while not stopped, one iteration per wall tick (tick_ms):
          1. now = time.time(); pokes = bus.drain_pokes(); cmds = bus.drain_commands() (apply: set_market_mode -> feed.set_mode;
             clear -> reset trail ring + event 'clear'; tweet_test -> agent.request_manual())
          2. snap, trades, mkt_events = feed.poll(now, tick_s)   (replay: taken from the log instead)
          3. f = features.update(snap, trades, now, tick_s, mood.state, decoder.feeding, r.mn9)
          4. drives = encoder.encode(f, body.kin.heading, mood.state, pokes, t_ms); engine.inject each (duration = brain_ms)
          5. steps = steps_per_tick if not realtime else max(5, round(steps_per_tick * speed)); stats = engine.step(steps)
          6. r = Readouts.from_stats(stats); cmd = decoder.update(r, f, mood.state, t_ms, brain_s); kin, wall_events = body.integrate(cmd, brain_s, t_ms)
             (wall_bump -> engine.inject('steer_a02_<side away from wall>', rate_hz=40, duration_ms=100, tag='wall'))
          7. ink = body.ink(kin, f, mood.state, t_ms); agent.snapshots.record_trail(kin.x, kin.y, ink.color, ink.width)
             (TweetAgent is always constructed, dry-run included, so the trail ring for /api/state and snapshots always exists)
          8. ms, transition = mood.update(MoodInputs(...)); confirmed = mood.confirmed()
          9. homeostasis: if stats.active_frac_max > 0.05 for 10 consecutive steps -> engine.set_gain(gain*0.9) (floor 0.3), event
         10. tick = build_tick(...); bus.publish_tick(tick); history.push(tick); events -> bus.publish_event
             (mood_change frame on transition; market frame when snap.seq changed)
         11. agent.on_tick(tick, transition, confirmed, history.jumps(60)) ; session_log.write_tick(...)
         12. pacing (realtime): compute_ms = elapsed; if compute_ms > 0.8*tick_ms: speed = max(0.25, speed*0.85);
             elif compute_ms < 0.45*tick_ms for 40 consecutive ticks: speed = min(1.0, speed*1.1);
             sleep until next_deadline (perf_counter based, never skip a tick; if late, deadline += tick_ms and continue).
             rtf = (steps*dt) / max(compute_ms, 1e-3) reported in tick.sim; FLY_REALTIME=0: no sleep, speed fixed 1.0."""
    def stop(self, timeout_s: float = 2.0) -> None
    def latest(self) -> dict | None
    def build_tick(self, ...) -> dict          # section d.2; single source of truth for the tick shape
    def build_hello(self) -> dict              # section d.1
```

### c.28 `flybrain/server/app.py`, `__main__.py`, `run.py`

```python
def create_app(settings: Settings | None = None) -> FastAPI:
    """Lifespan: setup_logging -> load_connectome -> LIFEngine(+SpikeMonitor) -> MarketFeed.start() -> Feature/Encoder/
    Decoder/Body/Mood -> StateBus(loop) -> SnapshotBroker -> TweetAgent -> SimulationLoop.start(); on shutdown: loop.stop(),
    feed.stop(), session_log.close(), agent executor shutdown(wait=False). A module-level lock prevents double start
    (uvicorn --reload spawns two processes: dev.ps1 never uses --reload). CORS from settings."""
# REST (all JSON):
#  GET  /api/health   -> HealthResponse {ok, uptime_s, run_id, rtf, speed, seq, t_ms, backend, connectome:{name,source,n,e,gain,license},
#                        market:{mode,ok,last_poll,failures}, agent:{llm,x,tweets_today,last_tweet_wall,disabled_reason}, clients, log_tail:[...]}
#  GET  /api/state    -> {hello, tick (latest), trail: [[x,y,color,width],... <=4000], mood_history: [[t_ms,state],...], events: [...last 50]}
#  GET  /api/config   -> redacted(settings)
#  GET  /api/connectome/groups -> {group: {n, types:[...<=20], sided: bool}}
#  GET  /api/neurons?group=gf&limit=100 -> [{id, body_id, type, region, side}]
#  GET  /api/tweets?limit=20 -> agent.history(limit)
#  POST /api/poke     body PokeRequest {stim, strength=1.0, side="both", duration_ms=500} -> {ok:true} (rate limit 2/s per client IP)
#  POST /api/market/mode body {mode: "sim"|"dexscreener"|"CALM"|"PUMP"|"DUMP"|"CHOP"|"RUG"|"DEAD"} -> {ok, mode}
#  POST /api/snapshot body {id, png_b64} -> {ok, bytes}     (REST alternative to the WS snapshot reply)
#  POST /api/tweet/test -> fires the agent once with reason 'manual' (honours dry-run flags, bypasses cooldowns) -> tweet record
#  WS   /ws  -> hello once, then tick frames + event frames; accepts client frames of section d.7; ping/pong; per-client
#              bounded queue (drop-oldest, size 4); a send that stalls > 1 s drops the client.

# flybrain/__main__.py
def main(argv: list[str] | None = None) -> int:
    """argparse: --port, --host, --seed, --n, --source, --market, --realtime/--no-realtime (override env);
    os.environ.setdefault('PYTHONUTF8','1'); settings = load_settings(); uvicorn.run(create_app(settings), host, port,
    log_level='info', ws='websockets'). Returns 0."""
# backend/run.py: sys.path.insert(0, dirname(__file__)); from flybrain.__main__ import main; raise SystemExit(main())
```

### c.29 `flybrain/log.py`

```python
class RingHandler(logging.Handler):
    """Keeps the last `capacity` formatted lines in a deque (thread-safe); tail(n) -> list[str] (for /api/health.log_tail)."""
    def __init__(self, capacity: int = 200) -> None: ...
    def tail(self, n: int = 50) -> list[str]: ...

def setup_logging(level: str = "INFO") -> RingHandler:
    """Root logger: one StreamHandler on stdout with format '%(asctime)s %(levelname)s %(name)s: %(message)s' (ASCII only:
    non-ASCII characters are replaced by '?' by an errors='replace' stream wrapper so the cp1254 console never raises),
    plus the RingHandler; uvicorn/fastapi loggers are routed through the same handlers; idempotent (a second call only
    updates the level). Returns the RingHandler."""
```

---

## d. WebSocket protocol (`ws://127.0.0.1:4000/ws`, protocol version 1)

Binding rules: JSON **text** frames, exactly one message per frame, every message has a string `type`; all keys are
`snake_case`; brain time is `t_ms` (int, ms since sim start), wall time is `wall` (float, unix seconds); the first frame
after accept is always `hello`; after that the server sends one `tick` per wall tick (20 Hz by default) plus the
out-of-band frames below; the client may send frames of §d.7 at any time; unknown `type` → `error` frame `bad_message`,
the connection stays open. `server/protocol.py` (§c.26) holds one pydantic model per frame; `frontend/lib/types.ts`
(§e.1) mirrors every field name and type 1:1. Floats are rounded before serialisation: positions/speeds 1 dp, angles 3 dp,
rates 2 dp, drives/mood scores 3 dp, prices full precision (`repr`).

### d.1 `hello` (server → client, once per connection; also `GET /api/state.hello`)

```json
{"type":"hello","v":1,"run_id":"9f3a2c1b","server_wall":1789051563.120,
 "dt_ms":1.0,"tick_ms":50.0,"steps_per_tick":50,"tick_hz":20,"realtime":true,"backend":"numpy",
 "canvas":{"w":800,"h":500,"walls":"bounce"},
 "connectome":{"name":"synthetic-20000-s1337-cal","source":"synthetic","n":20000,"e":548120,"synapses":2691003.0,
               "gain":1.0,"weights_mode":"calibrated","license":"synthetic (no data)","citation":null,
               "note":"synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data",
               "region_counts":{"optic_lobe":11126,"antennal_lobe":705,"mushroom_body":1523,"central_complex":623,
                                "sez":520,"central_other":2505,"descending_motor":404,"vnc":2594}},
 "regions":["optic_lobe","antennal_lobe","mushroom_body","central_complex","sez","central_other","descending_motor","vnc"],
 "raster":{"per_region":48,"cap":2000,
           "rows":[{"region":0,"slot":0,"neuron":17,"label":"R1-R6","side":"L","star":true},
                   {"region":0,"slot":1,"neuron":19,"label":"R1-R6","side":"R","star":true},
                   {"region":6,"slot":288,"neuron":6001,"label":"DNp01","side":"L","star":true},
                   {"region":6,"slot":289,"neuron":6002,"label":"DNp01","side":"R","star":true},
                   "... exactly 8 * per_region rows, slot == index in this list"]},
 "pops":["gf","gf_L","gf_R","escape_dn","escape_dn_L","escape_dn_R","dn_saccade","dn_saccade_L","dn_saccade_R","dn_land",
         "dn_freeze","dn_freeze_L","dn_freeze_R","dng100","dng100_L","dng100_R","dn_fwd","dn_fwd_L","dn_fwd_R","dn_back","dn_halt",
         "steer_a02","steer_a02_L","steer_a02_R","... every key of RateEstimator.snapshot() in wire order (§d.2)"],
 "mood_states":["SLEEP","CRUISING","FEEDING","EUPHORIA","ANXIOUS","PANIC","ESCAPE","COURTSHIP"],
 "channels":["sugar","bitter","loom","water","dust","pheromone","sleep","reward","punish","explore"],
 "market_modes":["sim","dexscreener","CALM","PUMP","DUMP","CHOP","RUG","DEAD"],
 "market":{"mode":"sim","chain":"sim","symbol":"FLY","token":"","poll_s":1},
 "agent":{"llm":"dryrun","x":"dryrun","cooldown_s":900,"reason_cooldown_s":2700,"tweets_per_day":12,"lang":"en"},
 "features":{"explore_baseline":0.25,"wander_sigma":0.6,"mood_feedback":true,"easter_eggs":true,"drive_mode":"poisson",
             "noise_mu":0.5,"noise_sigma":3.5}}
```

* `raster.rows[i].slot == i`; `region` is the index into `regions`; `neuron` is the engine index (for `GET /api/neurons`);
  `side` ∈ `"L"|"R"|"M"`; `star` marks §c.4 `STAR_TYPES` rows (drawn 2 px tall with a gutter label).
* `pops` is the exact key order of `tick.rates.pops` (§d.2); the frontend never assumes it.
* `channels` = valid `poke.stim` values (= `SensoryEncoder.POKE_TABLE` keys), `market_modes` = valid `set_market_mode.mode`.
* A new `hello` after a reconnect with a different `run_id` means a new session: the client clears its trail (after
  confirming with a Win95 message box if the trail is non-empty).

### d.2 `tick` (server → client, every wall tick)

Full example with realistic numbers (calibrated synthetic 20k, sim market in `PUMP`, fly walking after a meal):

```json
{"type":"tick","seq":12345,"t_ms":617350,"wall":1789051563.120,
 "sim":{"rtf":2.31,"speed":1.0,"steps":50,"step_ms":0.19,"spikes":812,"active_frac":0.0081,"gain":1.0,
        "edge_visits":31240,"forced":214,"noise":true,"backend":"numpy"},
 "fly":{"x":412.3,"y":233.9,"vx":-38.2,"vy":12.1,"heading":2.834,"speed":40.1,"omega":-0.41,
        "wing_hz":0.0,"wing_amp":0.0,"wing_ext":0,"mode":"walk","leg_phase":0.31,"proboscis":0.0,"jump_t_ms":null},
 "ink":{"color":"#00a800","width":2.0,"alpha":1.0,"style":"solid","stamp":null},
 "mood":{"state":"CRUISING","prev":"FEEDING","since_ms":17250,"euphoria":0.12,"anxiety":0.05,"arousal":0.30,
         "valence":0.07,"fear":0.05,"hunger":0.41,"sleep":0.0,"dwell_left_ms":0},
 "market":{"source":"sim","mode":"sim","ts":1789051563.0,"seq":617,"chain":"sim","dex":"sim","pair":"SIM","symbol":"FLY","token_live":false,
           "price_usd":0.0012345,"price_native":0.0012345,"buys_m5":41,"sells_m5":27,"buys_h1":402,"sells_h1":377,
           "chg_m5":1.2,"chg_h1":-3.4,"chg_h6":5.1,"chg_h24":12.7,"vol_m5":5120.5,"vol_h1":61230.0,
           "liq_usd":50210.0,"fdv":1234500.0,"mcap":1234500.0,"regime":"PUMP",
           "last_trade":{"kind":"buy","usd":123.4,"ts":1789051562.4,"surrogate":false}},
 "drives":{"sugar":0.61,"bitter":0.02,"water":0.0,"looming":0.0,"loom_side":0,"flash":0.0,"odor":0.30,"chop":0.05,
           "courtship":0.0,"sleep_pressure":0.0,"explore":0.35,"up":0.55,"down":0.03,"activity":0.40,"hunger":0.41,
           "candle":"up","any_max":0.61},
 "rates":{"regions":[3.21,1.12,0.42,2.03,5.48,1.71,4.12,1.88],
          "pops":{"gf":0.0,"gf_L":0.0,"gf_R":0.0,"escape_dn":0.0,"escape_dn_L":0.0,"escape_dn_R":0.0,
                  "dn_saccade":0.0,"dn_saccade_L":0.0,"dn_saccade_R":0.0,"dn_land":0.0,
                  "dn_freeze":2.1,"dn_freeze_L":2.0,"dn_freeze_R":2.2,"dng100":9.5,"dng100_L":9.0,"dng100_R":10.0,
                  "dn_fwd":3.1,"dn_fwd_L":3.0,"dn_fwd_R":3.2,"dn_back":0.4,"dn_halt":1.2,
                  "steer_a02":8.0,"steer_a02_L":4.0,"steer_a02_R":12.0,"steer_a01":3.0,"steer_a01_L":2.5,"steer_a01_R":3.5,
                  "steer_a03":2.0,"steer_a03_L":2.0,"steer_a03_R":2.0,"steer_b01":1.0,"steer_b01_L":1.0,"steer_b01_R":1.0,
                  "steer_g13":2.2,"steer_g13_L":1.8,"steer_g13_R":2.6,"flight_dn":0.3,"flight_dn_L":0.3,"flight_dn_R":0.3,
                  "groom_dn":0.5,"feed_dn":6.0,"pip10":0.0,"pip10_L":0.0,"pip10_R":0.0,"song_dn":0.0,
                  "ttmn":0.0,"ttmn_L":0.0,"ttmn_R":0.0,"psi":0.0,"gfc2":0.0,"wing_power":0.0,
                  "b1":0.0,"b1_L":0.0,"b1_R":0.0,"i1":0.0,"i1_L":0.0,"i1_R":0.0,"hg1":0.0,"hg1_L":0.0,"hg1_R":0.0,
                  "wing_steer":0.2,"leg_mn":6.4,"feed_mn":18.0,"feed_mn_L":17.0,"feed_mn_R":19.0,
                  "feed_pre_exc":22.0,"feed_pre_inh":9.5,"sugar2_exc":48.0,"sugar2_inh":35.0,"bitter2":1.1,
                  "grn_sugar":72.0,"grn_water":0.0,"grn_bitter":2.0,"grn_bitter_L":2.0,"grn_bitter_R":2.0,"grn_pher":0.0,
                  "jo_aud":2.1,"jo_groom":3.9,"photoreceptor":21.0,"lamina":14.0,"motion_in":15.5,"t4t5":4.0,
                  "lc4":0.4,"lc4_L":0.4,"lc4_R":0.4,"lplc2":0.5,"lplc2_L":0.5,"lplc2_R":0.5,"lc_loom2":0.6,
                  "lc_freeze":0.3,"lc_freeze_L":0.3,"lc_freeze_R":0.3,"orn":18.0,"alpn":11.0,"alln":7.0,"kc":1.4,"apl":9.0,
                  "mbon_avoid":1.0,"mbon_approach":6.0,"pam":18.0,"ppl1":0.5,
                  "epg":9.1,"epg_L":9.0,"epg_R":9.2,"pen":6.0,"delta7":4.0,"ring":2.0,"pfl3":7.5,"pfl3_L":6.0,"pfl3_R":9.0,
                  "dfb_sleep":0.2,"p1":0.1,"p1_L":0.1,"p1_R":0.1,"mal":10.0,"lal_ps":3.0,
                  "dms2":0.0,"dms2_L":0.0,"dms2_R":0.0,"song_vnc":0.0,"an_steer":2.0,"leg_premotor":5.2,
                  "lc_loom":0.46,"escape_vnc":0.0,"steer_a02_diff":8.0,"steer_a01_diff":1.0,"steer_g13_diff":0.8,
                  "b1_diff":0.0,"i1_diff":0.0,"hg1_diff":0.0,"dn_mean":4.12}},
 "spikes":{"t0_ms":617300,"win_ms":50,"total":812,"capped":false,"slots":[3,3,17,41,288],"dt":[0,12,5,33,49]},
 "events":[{"kind":"feed_stop","t_ms":617300,"data":{"meal_ms":4200}}]}
```

Field semantics (normative):

| Field | Meaning |
|---|---|
| `seq` | tick counter from 1; `t_ms` = brain time at the END of the tick; `wall` = time the tick was published |
| `sim.rtf` | `steps*dt_ms / compute_ms` of this tick (how much faster than realtime the engine ran); `sim.speed` = brain-ms per wall-ms in `[0.25, 1]` (§c.27); `steps` = LIF steps executed this tick; `step_ms` = mean ms per step; `spikes` = `StepStats.total_spikes`; `active_frac` = `StepStats.active_frac_max`; `forced` = Poisson kicks; `noise` = background noise on |
| `fly.*` | `Kinematics.to_wire()`: px, px/s, rad (0 = +x, clockwise positive), rad/s; `mode ∈ walk\|fly\|jump\|feed\|freeze\|groom\|court\|sleep`; `wing_hz` 0 or 150–250; `wing_amp` 0–1; `wing_ext ∈ -1\|0\|1`; `leg_phase` 0–1; `proboscis` 0–1; `jump_t_ms` ms since jump start or `null` |
| `ink.*` | `InkStyle`: `color` `#rrggbb`, `width` px, `alpha` 0–1, `style ∈ solid\|rainbow\|zigzag\|dotted\|hearts`, `stamp ∈ null\|blob\|heart\|zzz\|dash\|bump` |
| `mood.*` | `MoodState.to_wire()`: scores 0–1 except `valence` −1..1; `since_ms` brain ms in the current state; `dwell_left_ms` ms before a non-ESCAPE transition is allowed |
| `market.*` | `MarketSnapshot.to_wire()` + `mode` (`MarketFeed.mode`) + `last_trade` (`null` when none this session); `regime` null unless the sim is the active source |
| `drives.*` | the `Features` channels that drive the brain (§f.1); `loom_side ∈ -1\|0\|1`; `candle ∈ up\|down\|flat`; `any_max` = max of sugar/bitter/water/looming/flash/odor/chop/courtship/sleep_pressure |
| `rates.regions` | Hz per neuron per region, `REGIONS` order (EMA τ 0.25 s) |
| `rates.pops` | `RateEstimator.snapshot()` (§c.12): every readout of `READOUTS`, sided ones followed by `_L`/`_R`, then the derived keys `lc_loom, escape_vnc, steer_a02_diff, steer_a01_diff, steer_g13_diff, b1_diff, i1_diff, hg1_diff, dn_mean` — the key order is `hello.pops` (131 keys for the synthetic table) |
| `spikes` | raster sample of this tick: `slots[k]` indexes `hello.raster.rows`, `dt[k]` = LIF steps after `t0_ms` (`0..steps-1`), parallel arrays sorted by `dt`; uniformly subsampled to `raster.cap` with `capped=true` |
| `events` | in-pipeline events of this tick (decoder/body/mood/loop): `{kind, t_ms, data}`; kinds in §d.8 |

### d.3 `mood_change` (server → client, on every transition; not gated by dwell confirmation)

```json
{"type":"mood_change","seq":12300,"t_ms":615100,"wall":1789051560.87,
 "from":"FEEDING","to":"CRUISING","reason":"mn9 < 15 Hz for 1.0 s","mood":{"state":"CRUISING","prev":"FEEDING","since_ms":0,
 "euphoria":0.33,"anxiety":0.05,"arousal":0.41,"valence":0.28,"fear":0.05,"hunger":0.20,"sleep":0.0,"dwell_left_ms":1500}}
```

`mood` is the same object as `tick.mood`. The frontend sets `document.title = "FlyBrain - <to>"` and repaints the mood
badge immediately (does not wait for the next tick).

### d.4 `tweet` (server → client, after every agent fire, dry-run included)

```json
{"type":"tweet","seq":12480,"t_ms":624000,"wall":1789051569.4,"id":"20260910-181209","reason":"euphoria_entry",
 "text":"gm ser. sugar GRNs LB3b at 131 Hz, MN9 proboscis fully extended into a green candle. mushroom body says wagmi. PAM dopamine: 33 Hz of pure cope-free bliss.",
 "model":"template","dry_run":true,"posted":false,"url":null,"error":null,"snapshot_source":"browser",
 "neurons":["LB3b","MN9","PAM"],"mood":"EUPHORIA","latency_ms":3.1}
```

`model` is `"template"` in dry-run, else the model id. `id` is the `out/tweets/<id>.{txt,json,png}` stem. `snapshot_source ∈ browser|server`.
The same record (plus `summary`) is what `GET /api/tweets` returns.

### d.5 `market` (server → client, whenever `MarketSnapshot.seq` changes: ~1 Hz sim, ~1/60 Hz DexScreener)

```json
{"type":"market","seq":12345,"t_ms":617350,"wall":1789051563.12,"mode":"sim",
 "market":{"source":"sim","ts":1789051563.0,"seq":617,"chain":"sim","dex":"sim","pair":"SIM","symbol":"FLY","token_live":false,"price_usd":0.0012345,
           "price_native":0.0012345,"buys_m5":41,"sells_m5":27,"buys_h1":402,"sells_h1":377,"chg_m5":1.2,"chg_h1":-3.4,
           "chg_h6":5.1,"chg_h24":12.7,"vol_m5":5120.5,"vol_h1":61230.0,"liq_usd":50210.0,"fdv":1234500.0,"mcap":1234500.0,
           "regime":"PUMP"},
 "trades":[{"kind":"buy","usd":123.4,"ts":1789051562.4,"surrogate":false},{"kind":"sell","usd":40.0,"ts":1789051562.9,"surrogate":false}]}
```

`trades` = the trades consumed since the previous `market` frame (≤ 200, oldest dropped). The ticker sparkline and the
buys/sells bars are built client-side from these frames; `tick.market` is only a convenience copy of the last one.

### d.6 Agent JSON: brain summary and system prompt (the ONLY things the LLM sees)

`build_brain_summary()` (§c.20) returns exactly this shape (≤ 1.5 KB; numbers rounded as shown; every key always present):

```json
{"schema":"flybrain.summary.v1","reason":"euphoria_entry","lang":"en",
 "mood":{"state":"EUPHORIA","prev":"FEEDING","since_s":3.0,"euphoria":0.78,"anxiety":0.03,"valence":0.62,"arousal":0.71,"hunger":0.2},
 "market":{"source":"sim","symbol":"FLY","price_usd":0.0012345,"chg_m5":4.2,"chg_h1":11.0,"chg_h24":40.5,"buys_m5":88,"sells_m5":21,
           "vol_m5":21000.0,"liq_usd":52000.0,"mcap":1450000.0,"regime":"PUMP","last_trade":{"kind":"buy","usd":940.0}},
 "drives":{"sugar":0.83,"bitter":0.02,"looming":0.0,"odor":0.55,"chop":0.0,"courtship":0.0,"sleep_pressure":0.0,"explore":0.46},
 "rates_hz":{"grn_sugar":131.0,"sugar2_exc":40.0,"feed_mn":52.0,"pam":33.0,"ppl1":0.0,"mbon_approach":18.0,"mbon_avoid":3.0,
             "steer_a02_L":6.0,"steer_a02_R":19.0,"gf":0.0,"lc_loom":0.0,"dn_freeze":1.2,"dng100":12.0,"flight_dn":0.0,
             "kc":2.1,"epg":9.0,"pfl3":12.0,"p1":0.0,"pip10":0.0},
 "regions_hz":{"optic_lobe":4.1,"antennal_lobe":6.0,"mushroom_body":1.9,"central_complex":3.3,"sez":9.8,"central_other":2.2,
               "descending_motor":7.2,"vnc":2.4},
 "top_types":[["LB3b",131.0],["LB3c",128.0],["GNG215",61.0],["MN9",52.0],["PAM01",40.0]],
 "events_10s":["feed_start","mood:FEEDING->EUPHORIA"],"gf_spikes_10min":0,"jumps_60s":0,
 "fly":{"mode":"feed","x":412,"y":233,"speed":0.0,"wing_hz":0.0,"proboscis":0.6,"trail_px":18240},
 "connectome":{"source":"synthetic","name":"synthetic-20000-s1337-cal","n":20000,"e":548120,
               "note":"synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data"},
 "session":{"uptime_s":3612.0,"tweets_today":2,"last_tweet_reason":"panic_entry"},
 "vocabulary_hint":["sugar GRNs (LB3b/LB3c)","MN9 proboscis","PAM dopamine","mushroom body","DNa02 steering",
                    "giant fiber DNp01","LC4/LPLC2 looming","central complex EPG/PFL3"]}
```

`rates_hz` keys are fixed (the 19 listed) and come from `tick.rates.pops`; `top_types` = the 5 highest per-neuron rates
among `STAR_TYPES` present; `connectome.note` is copied from `meta['note']` (real data: `"MaleCNS v1.0, CC-BY 4.0"`).

`SYSTEM_PROMPT` (§c.21), verbatim (a final line `"Write the tweet in Turkish."` is appended when `FLY_TWEET_LANG=tr`):

```
You are FlyBrain, a spiking simulation of the Drosophila male CNS connectome (MaleCNS v1.0 shaped) whose senses are wired
to the $FLY token market. You receive a JSON summary of your brain state and the market. Write exactly ONE tweet of at most
240 characters in absurd, self-aware crypto dialect (gm, ser, wagmi, ngmi, cope, degen, candles, liquidity) that is also
neuroscientifically literal: name at least one concrete neuron group from the JSON (for example sugar GRNs LB3b, MN9
proboscis, giant fiber DNp01, LC4/LPLC2 looming, DNa02 steering, PAM dopamine, mushroom body). If connectome.source is
"synthetic", never claim the data is the real connectome. No URLs, no financial advice, no promises of returns, no cashtags
other than $FLY, at most 2 hashtags, no emojis. Return only the JSON object {"text": string, "neurons": string[]}.
```

When `FLY_LLM_JSON=0` the last sentence is replaced by `Output only the tweet text.`

### d.7 Client → server frames

| Frame | JSON | Effect |
|---|---|---|
| `poke` | `{"type":"poke","stim":"sugar","strength":1.0,"side":"both","duration_ms":500}` | `stim ∈ hello.channels`; `strength` 0–1; `side ∈ "L"\|"R"\|"both"` → `Poke.side` −1/+1/0; `duration_ms` 50–5000 → `bus.push_poke(Poke(stim, strength, side, until_ms = t_ms + duration_ms))`; rate limit 2 per second per client (excess → `error rate_limited`) |
| `set_market_mode` | `{"type":"set_market_mode","mode":"PUMP"}` | `mode ∈ hello.market_modes` → `bus.push_command({"name":"set_market_mode","mode":...})` → `MarketFeed.set_mode`; a regime name forces the sim regime for 60 s; `dexscreener` is refused with `error forbidden` when `FLY_TOKEN_ADDRESS` is empty |
| `clear` | `{"type":"clear"}` | `push_command({"name":"clear"})`: server trail ring reset + out-of-band event `clear` to all clients (every client clears its trail layer) |
| `tweet_test` | `{"type":"tweet_test"}` | `push_command({"name":"tweet_test"})` → `TweetAgent.request_manual()` (reason `manual`, never posts unless `FLY_X=post`; bypasses cooldowns, not the daily cap) |
| `ping` | `{"type":"ping","t":1789051563.1}` | answered immediately with `pong` |
| `snapshot` | `{"type":"snapshot","id":"snap-17","png_b64":"iVBORw0..."}` | reply to `snapshot_request`; ≤ 2 MB decoded, PNG magic checked → `bus.deliver_snapshot(id, png)`; unknown/expired id is ignored |

Validation is `parse_client()` (§c.26, `extra='forbid'`); any failure → `{"type":"error","code":"bad_message","msg":"<pydantic message>"}`.

### d.8 `event` (server → client, out-of-band) and in-tick event kinds

Out-of-band frame (published with `bus.publish_event`, not repeated inside `tick.events`):

```json
{"type":"event","seq":12346,"t_ms":617400,"wall":1789051563.17,"kind":"market_source","data":{"mode":"sim(fallback)","reason":"3 consecutive DexScreener failures: ConnectTimeout"}}
```

| kind | where | `data` |
|---|---|---|
| `market_source` | event frame | `{mode, reason}` |
| `homeostasis` | event frame | `{gain_before, gain_after, active_frac}` |
| `clear` | event frame | `{by: "client"\|"api"}` |
| `poke` | event frame | `{stim, strength, side, duration_ms}` (echo to every client so all UIs flash the tool) |
| `easter_egg` | event frame | `{reason: "buys_m5 == 420"\|"price 0.000420"\|"clock 04:20"}` |
| `whale` | event frame | `{kind, usd, ratio}` (trade ≥ 10 × `usd_ref`) |
| `calibration` | event frame | `{gain, report}` (first-start `calibrate_gain` result, literature/real modes only) |
| `jump` | `tick.events` | `{heading_out, impulse, side, forward}` |
| `takeoff`, `landing` | `tick.events` | `{wing_hz}` / `{reason}` |
| `freeze`, `unfreeze` | `tick.events` | `{dnp09_hz, looming}` |
| `feed_start`, `feed_stop` | `tick.events` | `{mn9_hz}` / `{meal_ms}` |
| `groom`, `song`, `saccade` | `tick.events` | `{side}` |
| `wander_floor` | `tick.events` | `{stalled_ms}` (the guaranteed-motion fallback engaged, §f.3) |
| `sleep`, `wake` | `tick.events` | `{}` / `{reason}` |
| `wall_bump`, `wrap` | `tick.events` | `{side: "L"\|"R"}` / `{edge}` |
| `gf_spike` | `tick.events` | `{side, count}` (one per tick with DNp01 spikes) |
| `mood` | `tick.events` | `{from, to, reason}` (duplicate of the `mood_change` frame, for rAF consumers) |

### d.9 `snapshot_request`, `pong`, `error`

```json
{"type":"snapshot_request","id":"snap-17","deadline_ms":3000}
{"type":"pong","t":1789051563.1,"server_wall":1789051563.4,"seq":12345}
{"type":"error","code":"bad_message","msg":"poke.stim: Input should be one of ..."}      // codes: bad_message | rate_limited | forbidden
```

`snapshot_request` goes to the most recently active client only (last frame received); the client composites its
canvases (§e.6) and answers within `deadline_ms`; otherwise the server renders its own PNG (§c.23).

### d.10 Framing, ordering and size budget

* Per-client `asyncio.Queue(maxsize=4)`, drop-oldest: a slow browser loses old ticks, never stalls the sim. `hello`,
  `mood_change`, `tweet`, `market`, `event`, `snapshot_request` use the same queue (they are small) — ordering per client is
  therefore publish order. The tick JSON is serialised **once** per tick (`tick_to_json`) and the same string is sent to
  every client.
* Sizes (measured shape, minified): `hello` ≈ 30 KB once (384 raster rows); `tick` ≈ 2.2 KB fixed part (`pops` 131 keys ≈
  1.8 KB of it) + raster ≈ 9 bytes/event. At the required 1–5 Hz rest activity the 384 sampled rows produce 20–100 events
  per 50 ms tick → **2.5–3.5 KB/tick ≈ 50–70 KB/s at 20 Hz**; a PUMP+RUG burst at 5 % active fraction hits the 2000-event cap
  → ≈ 20 KB/tick ≈ 400 KB/s, still fine on localhost. Clients that only want state may send nothing and ignore `spikes`.
* Heartbeat: the client pings every 10 s; if no `pong` or `tick` for 15 s it closes and reconnects (§e.2). The server relies
  on the websockets library ping (20 s) and drops a client whose send stalls > 1 s.
* The frontend must never parse floats by position: everything is keyed.

---

## e. Frontend contract (`frontend/`, existing Next 16.3.4 / React 19.2.8 / Tailwind 4.3.3 scaffold)

### e.0 Stack rules (binding)

* **Use the scaffold that exists.** Never run `create-next-app` again. `package.json` dependencies stay exactly as they
  are (`next 16.3.4`, `react 19.2.8`, `react-dom 19.2.8`; dev: `@tailwindcss/postcss ^4`, `tailwindcss ^4`, `typescript ^5`,
  `eslint ^9`, `eslint-config-next 16.3.4`, `@types/*`). **No new npm dependencies** (no socket.io, no chart/canvas/UI libs,
  no state libs). The only allowed `package.json` change is adding `"typecheck": "tsc --noEmit"` to `scripts`.
* Next 16 conventions (from `node_modules/next/dist/docs/`, verified in RESEARCH §11): App Router only; Turbopack is the
  default dev bundler; **no `middleware.ts` / `proxy.ts`**; no route handlers, no server actions (the Python server is the
  only backend); browser-visible env vars are `NEXT_PUBLIC_*`; `PageProps`/`LayoutProps` are global helpers (no import);
  request APIs are async (we use none). `app/layout.tsx` is the only server component; **every file under `components/`
  and `app/page.tsx` starts with `"use client"`** (they use hooks, canvases, `window`, `localStorage`).
* Offline-first: remove the `next/font/google` (Geist) import from the scaffold's `layout.tsx`; font stack is
  `"MS Sans Serif", "Microsoft Sans Serif", Tahoma, Arial, system-ui, sans-serif`, body 12 px; no external assets, no CDNs;
  favicon = a 16 px pixel fly as an inline `data:` SVG in `layout.tsx` metadata (`icons`), the scaffold's `favicon.ico`
  may stay.
* Tailwind v4: `app/globals.css` keeps `@import "tailwindcss";` and defines the Win95 palette as `@theme` tokens
  (`--color-win-gray: #c0c0c0; --color-win-dark: #808080; --color-win-light: #dfdfdf; --color-win-blue: #000080;
  --color-win-blue2: #1084d0; --color-win-teal: #008080; --color-paint-white: #ffffff`) plus four plain CSS utility classes
  (`.bevel-out`, `.bevel-in`, `.titlebar`, `.btn95`) — there is **no `tailwind.config.*`**. Light look only: the Paint chrome
  ignores `prefers-color-scheme` (delete the scaffold's dark-mode block).
* Rendering rule: **no React state update per tick.** Ticks land in a mutable store (`lib/store.ts`); canvases read the
  store inside `requestAnimationFrame`; panels subscribe through `useSyncExternalStore` with a 4 Hz throttled snapshot.
* Env (`frontend/.env.local.example`, copied to `.env.local`): `NEXT_PUBLIC_WS_URL=ws://localhost:4000/ws`,
  `NEXT_PUBLIC_API_URL=http://localhost:4000`.
* Quality gates: `npm run typecheck` (strict TS, no `any` except the pydantic-error `msg`), `npm run lint`, `npm run build` all pass.

### e.1 `lib/types.ts` — 1:1 mirror of §d (normative; implementers copy verbatim)

```ts
export type Region = "optic_lobe"|"antennal_lobe"|"mushroom_body"|"central_complex"|"sez"|"central_other"|"descending_motor"|"vnc";
export const REGIONS: readonly Region[] = ["optic_lobe","antennal_lobe","mushroom_body","central_complex","sez","central_other","descending_motor","vnc"];
export type Mood = "SLEEP"|"CRUISING"|"FEEDING"|"EUPHORIA"|"ANXIOUS"|"PANIC"|"ESCAPE"|"COURTSHIP";
export type FlyMode = "walk"|"fly"|"jump"|"feed"|"freeze"|"groom"|"court"|"sleep";
export type InkStyleName = "solid"|"rainbow"|"zigzag"|"dotted"|"hearts";
export type Stamp = null|"blob"|"heart"|"zzz"|"dash"|"bump";
export type Side = "L"|"R"|"M";
export type PokeStim = "sugar"|"bitter"|"loom"|"water"|"dust"|"pheromone"|"sleep"|"reward"|"punish"|"explore";
export type MarketMode = "sim"|"dexscreener"|"CALM"|"PUMP"|"DUMP"|"CHOP"|"RUG"|"DEAD";
export type Candle = "up"|"down"|"flat";

export interface RasterRow { region: number; slot: number; neuron: number; label: string; side: Side; star: boolean }

export interface HelloMsg {
  type: "hello"; v: 1; run_id: string; server_wall: number;
  dt_ms: number; tick_ms: number; steps_per_tick: number; tick_hz: number; realtime: boolean; backend: string;
  canvas: { w: number; h: number; walls: "bounce"|"wrap" };
  connectome: { name: string; source: "synthetic"|"csv"|"neuprint"; n: number; e: number; synapses: number; gain: number;
                weights_mode: string|null; license: string; citation: string|null; note: string; region_counts: Record<Region, number> };
  regions: Region[];
  raster: { per_region: number; cap: number; rows: RasterRow[] };
  pops: string[];
  mood_states: Mood[];
  channels: PokeStim[];
  market_modes: MarketMode[];
  market: { mode: "sim"|"dexscreener"|"sim(fallback)"; chain: string; symbol: string; token: string; poll_s: number };
  agent: { llm: "dryrun"|"anthropic"; x: "dryrun"|"post"; cooldown_s: number; reason_cooldown_s: number; tweets_per_day: number; lang: "en"|"tr" };
  features: { explore_baseline: number; wander_sigma: number; mood_feedback: boolean; easter_eggs: boolean;
              drive_mode: "poisson"|"current"; noise_mu: number; noise_sigma: number };
}

export interface SimStats { rtf: number; speed: number; steps: number; step_ms: number; spikes: number; active_frac: number;
  gain: number; edge_visits: number; forced: number; noise: boolean; backend: string }
export interface FlyState { x: number; y: number; vx: number; vy: number; heading: number; speed: number; omega: number;
  wing_hz: number; wing_amp: number; wing_ext: -1|0|1; mode: FlyMode; leg_phase: number; proboscis: number; jump_t_ms: number|null }
export interface InkStyle { color: string; width: number; alpha: number; style: InkStyleName; stamp: Stamp }
export interface MoodState { state: Mood; prev: Mood; since_ms: number; euphoria: number; anxiety: number; arousal: number;
  valence: number; fear: number; hunger: number; sleep: number; dwell_left_ms: number }
export interface Trade { kind: "buy"|"sell"; usd: number; ts: number; surrogate: boolean }
export interface MarketSnapshot { source: "sim"|"dexscreener"|"sim(fallback)"; ts: number; seq: number; chain: string; dex: string;
  pair: string; symbol: string; price_usd: number|null; price_native: number|null; buys_m5: number; sells_m5: number;
  buys_h1: number; sells_h1: number; chg_m5: number; chg_h1: number; chg_h6: number; chg_h24: number; vol_m5: number; vol_h1: number;
  liq_usd: number|null; fdv: number|null; mcap: number|null; regime: string|null }
export interface TickMarket extends MarketSnapshot { mode: "sim"|"dexscreener"|"sim(fallback)"; last_trade: Trade|null }
export interface Drives { sugar: number; bitter: number; water: number; looming: number; loom_side: -1|0|1; flash: number; odor: number;
  chop: number; courtship: number; sleep_pressure: number; explore: number; up: number; down: number; activity: number;
  hunger: number; candle: Candle; any_max: number }
export interface Spikes { t0_ms: number; win_ms: number; total: number; capped: boolean; slots: number[]; dt: number[] }
export interface TickEvent { kind: TickEventKind; t_ms: number; data: Record<string, unknown> }
export type TickEventKind = "jump"|"takeoff"|"landing"|"freeze"|"unfreeze"|"feed_start"|"feed_stop"|"groom"|"song"|"saccade"
  |"wander_floor"|"sleep"|"wake"|"wall_bump"|"wrap"|"gf_spike"|"mood";
export interface TickMsg {
  type: "tick"; seq: number; t_ms: number; wall: number;
  sim: SimStats; fly: FlyState; ink: InkStyle; mood: MoodState; market: TickMarket; drives: Drives;
  rates: { regions: number[]; pops: Record<string, number> };
  spikes: Spikes; events: TickEvent[];
}
export interface MoodChangeMsg { type: "mood_change"; seq: number; t_ms: number; wall: number; from: Mood; to: Mood; reason: string; mood: MoodState }
export interface TweetMsg { type: "tweet"; seq: number; t_ms: number; wall: number; id: string; reason: string; text: string; model: string;
  dry_run: boolean; posted: boolean; url: string|null; error: string|null; snapshot_source: "browser"|"server"; neurons: string[];
  mood: Mood; latency_ms: number }
export interface MarketMsg { type: "market"; seq: number; t_ms: number; wall: number; mode: "sim"|"dexscreener"|"sim(fallback)";
  market: MarketSnapshot; trades: Trade[] }
export type OobEventKind = "market_source"|"homeostasis"|"clear"|"poke"|"easter_egg"|"whale"|"calibration";
export interface EventMsg { type: "event"; seq: number; t_ms: number; wall: number; kind: OobEventKind; data: Record<string, unknown> }
export interface SnapshotRequestMsg { type: "snapshot_request"; id: string; deadline_ms: number }
export interface PongMsg { type: "pong"; t: number; server_wall: number; seq: number }
export interface ErrorMsg { type: "error"; code: "bad_message"|"rate_limited"|"forbidden"; msg: string }
export type ServerMsg = HelloMsg|TickMsg|MoodChangeMsg|TweetMsg|MarketMsg|EventMsg|SnapshotRequestMsg|PongMsg|ErrorMsg;

export type ClientMsg =
  | { type: "poke"; stim: PokeStim; strength: number; side: "L"|"R"|"both"; duration_ms: number }
  | { type: "set_market_mode"; mode: MarketMode }
  | { type: "clear" }
  | { type: "tweet_test" }
  | { type: "ping"; t: number }
  | { type: "snapshot"; id: string; png_b64: string };

// REST shapes (lib/api.ts)
export interface TweetRecord extends Omit<TweetMsg, "type"|"seq"> { summary?: Record<string, unknown> }
export interface StateResponse { hello: HelloMsg; tick: TickMsg|null; trail: [number, number, string, number][];
  mood_history: [number, Mood][]; events: (TickEvent|EventMsg)[] }
export interface HealthResponse { ok: boolean; uptime_s: number; run_id: string; rtf: number; speed: number; seq: number; t_ms: number;
  backend: string; connectome: { name: string; source: string; n: number; e: number; gain: number; license: string };
  market: { mode: string; ok: boolean; last_poll: number|null; failures: number };
  agent: { llm: string; x: string; tweets_today: number; last_tweet_wall: number|null; disabled_reason: string|null };
  clients: number; log_tail: string[] }
```

A build-time guard: `tests/test_protocol.py::test_types_ts_mirror` (§h) parses this file's `interface` field names and
asserts they equal the pydantic model fields, so the two sides cannot drift silently.

### e.2 `lib/ws.ts` — `useFlySocket`

```ts
export type WsStatus = "connecting"|"open"|"reconnecting"|"closed";
export interface FlySocket {
  status: WsStatus; attempt: number; latencyMs: number|null;
  hello: HelloMsg|null;                 // React state (changes rarely)
  store: TickStore;                     // mutable, read in rAF loops (e.3)
  send(m: ClientMsg): boolean;          // false when not open (message dropped, never queued)
  on<K extends ServerMsg["type"]>(type: K, h: (m: Extract<ServerMsg, {type: K}>) => void): () => void;   // event bus, unsubscribe fn
}
export function useFlySocket(url: string = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:4000/ws"): FlySocket;
```

Behaviour: opens a native `WebSocket(url)` in an effect (SSR-safe: nothing runs on the server); `onmessage` →
`JSON.parse` → dispatch by `type`: `hello` → React state + `store.setHello` (+ trail-clear prompt when `run_id` changed);
`tick` → `store.push(tick, performance.now())` (no React update); `mood_change` → `document.title = "FlyBrain - " + to` and
bus; `tweet`/`market`/`event`/`error` → bus; `pong` → `latencyMs = now - t`; `snapshot_request` → the snapshot service
(§e.6) answers with `send({type:"snapshot", ...})`. Reconnect: on close/error set `reconnecting`, wait
`min(8000, 500 * 2 ** attempt) + random(0..250)` ms, retry; `attempt` resets to 0 after a `hello`. Heartbeat: `ping` every
10 s; if neither `pong` nor `tick` arrived for 15 s → `close()` (which triggers the reconnect path). While not `open` the
canvases keep the last frame, the status bar dot turns red with `reconnecting (n)`, and the fly overlay draws a Win95
hourglass cursor beside the fly. Unmount closes the socket and cancels timers.

### e.3 `lib/store.ts` and `lib/interp.ts`

```ts
export interface StoredTick { tick: TickMsg; recvAt: number }          // recvAt = performance.now()
export interface TickStore {
  hello: HelloMsg|null; setHello(h: HelloMsg): void;
  push(t: TickMsg, recvAt: number): void;                              // ring of 4; drops the oldest
  latest(): StoredTick|null; prev(): StoredTick|null;
  subscribe(fn: () => void): () => void;                               // notified at most every 250 ms (4 Hz), for useSyncExternalStore
  snapshot(): TickMsg|null;                                            // stable reference between notifications (useSyncExternalStore contract)
  events: TickEvent[];                                                  // last 200 in-tick events (for panels)
}
export function createTickStore(): TickStore;
export function useTickSnapshot(store: TickStore): TickMsg|null;       // useSyncExternalStore(store.subscribe, store.snapshot)

// lib/interp.ts
export interface FlyPose { x: number; y: number; heading: number; speed: number; wing_hz: number; wing_amp: number; wing_ext: -1|0|1;
  mode: FlyMode; leg_phase: number; proboscis: number; jump_t_ms: number|null }
export function lerpAngle(a: number, b: number, t: number): number;                  // shortest arc
export function interpolate(prev: StoredTick|null, latest: StoredTick, now: number, tickMs: number): FlyPose;
```

`interpolate`: `alpha = clamp((now - latest.recvAt) / tickMs, 0, 1)`; position/heading/speed are lerped from
`prev.fly` to `latest.fly` (angle-lerp for heading) — one tick (50 ms) of latency, perfectly smooth at 60 fps; if `prev`
is null or the two ticks are > 4 × `tickMs` apart (reconnect, tab throttling) return `latest.fly` unchanged (no long
straight segment); `mode`, `wing_*`, `proboscis`, `jump_t_ms` are taken from `latest`. Wall bounces are visible as a
heading jump because the server already reflected the heading.

### e.4 Drawing libraries

```ts
// lib/flySprite.ts
export function drawFly(ctx: CanvasRenderingContext2D, pose: FlyPose, tSec: number, mood: Mood, scale?: number /* default 2 */): void;
// lib/ink.ts
export function drawInk(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, ink: InkStyle, tSec: number): void;
export function drawStamp(ctx: CanvasRenderingContext2D, x: number, y: number, ink: InkStyle, pose: FlyPose, tSec: number): void;
export function moodColor(mood: Mood, tSec: number): string;          // MOOD_COLORS of §c.19; "rainbow" -> hsl((tSec*90) % 360, 100%, 45%)
// lib/raster.ts
export class RasterPainter {
  constructor(canvas: HTMLCanvasElement, hello: HelloMsg, opts?: { windowS?: number /* 8 */; pxPerS?: number /* 60 */ });
  push(tick: TickMsg): void;                                          // plots tick.spikes at the right edge (called from rAF when a new tick arrived)
  frame(now: number): void;                                           // shifts the bitmap left by elapsed*pxPerS (drawImage self-copy)
  setFrozen(f: boolean): void; setLabels(on: boolean): void; resize(): void; rowAt(y: number): RasterRow|null;
}
// lib/api.ts (all relative to NEXT_PUBLIC_API_URL; JSON; throw on !ok)
export function getState(): Promise<StateResponse>;
export function getHealth(): Promise<HealthResponse>;
export function getTweets(limit?: number): Promise<TweetRecord[]>;
export function postPoke(stim: PokeStim, strength: number, side: "L"|"R"|"both", duration_ms: number): Promise<{ok: boolean}>;
export function postMarketMode(mode: MarketMode): Promise<{ok: boolean; mode: string}>;
export function postTweetTest(): Promise<TweetRecord>;
export function postSnapshot(id: string, png_b64: string): Promise<{ok: boolean; bytes: number}>;
```

`drawFly` (procedural, `imageSmoothingEnabled = false`, drawn at `scale` then rotated by `pose.heading` around the
thorax): abdomen = dark brown (#3b2a1a) ellipse 12×6 with three lighter (#6b4a2a) stripes; thorax = ellipse 10×7; head =
circle r 4 with two red (#c02020) compound eyes r 1.5; six 1.5 px legs as two-segment polylines in tripod gait (legs 1,3,5
at `phase = leg_phase`, legs 2,4,6 at `phase + 0.5`, swing amplitude 25°) when `mode ∈ walk|court`, tucked when
`sleep|feed|freeze`, front pair rubbing (10 Hz) when `groom`; wings = two translucent ellipses 12×5
(`rgba(200,220,255,0.55)`): when `wing_hz > 0` each wing is drawn twice per frame at angles `±(20 + 25·wing_amp·sin(2π·wing_hz·tSec))°`
and `±(20 + 25·wing_amp·sin(2π·wing_hz·tSec + π))°` with half alpha (stroboscopic blur) plus a faint circle r 9 alpha 0.12
(beat envelope); when `wing_ext ≠ 0` that wing is held out at 70° and jittered ±4° at 25 Hz (song); `proboscis > 0` draws a
1.5 px line of length `8·proboscis` forward with a yellow sugar-drop (r 3, white highlight) at its tip when `mode == feed`;
`mode == jump` draws a 2 px shadow offset by `min(8, jump_t_ms/50)` px; `mode == sleep` draws three rising "z" glyphs
(one every 0.5 s); mood `PANIC` shakes the sprite ±1 px at 30 Hz; `EUPHORIA` draws a 4-point sparkle every 5th frame at a
random point on the body; `ESCAPE` draws three motion lines behind the fly. All colours flat (Paint look), no gradients.

`drawInk` on the persistent trail layer with `lineCap = "round"`, `lineWidth = ink.width`, `globalAlpha = ink.alpha`:
`solid` → one segment in `ink.color`; `rainbow` → `moodColor("EUPHORIA", tSec)`; `zigzag` → the segment is drawn as two
half-segments offset ±3 px perpendicular, alternating each call; `dotted` → 2 px dots every 6 px along the segment;
`hearts` → a 7 px pixel heart (#ff69b4) every 14 px of accumulated path length. Segments longer than 40 px (teleport,
wrap, reconnect) are skipped. `drawStamp`: `blob` → filled circle r `2 + 8·proboscis` in `ink.color`; `heart` → 9 px heart;
`zzz` → "z" text 8 px rising 6 px per 0.5 s; `dash` → nothing on the trail (the jump segment is skipped) but three
dashed motion streaks on the overlay; `bump` → a 1 px 6-px-long tick mark on the wall side.

`RasterPainter`: canvas 480 × (8 lanes × `per_region` px + 8 × 12 px gutter) at DPR 1, black background (oscilloscope),
lanes in **`REGIONS` order top to bottom**, lane colours `#7fdbff` optic_lobe, `#b10dc9` antennal_lobe, `#ffdc00`
mushroom_body, `#2ecc40` central_complex, `#ff851b` sez, `#aaaaaa` central_other, `#ff4136` descending_motor, `#39cccc`
vnc; each spike is a 1×1 px rect at `x = W - 1 - (win_ms - dt·dt_ms)·pxPerS/1000`, `y = laneTop + slot % per_region`;
`star` rows are 2 px tall and get their `label` (e.g. `DNp01 R`, `MN9`, `PFL3 L`) drawn in the 60 px left gutter; `frame()`
shifts the whole bitmap left by `elapsed·pxPerS` px via `drawImage(canvas, -dx, 0)` and clears the revealed strip; the
right gutter shows `tick.rates.regions[i].toFixed(1) + " Hz"` per lane in the lane colour; `rowAt(y)` supports the hover
tooltip (`label`, `side`, `neuron`). Frozen mode stops shifting (screenshots).

### e.5 Components (props + behaviour)

All components are `"use client"`, receive the socket/store through props (no context library), and never call `setState`
from a tick handler.

| Component | Props | Behaviour |
|---|---|---|
| `Win95Window` | `{ id: string; title: string; icon?: ReactNode; initial: {x,y,w,h}; menu?: ReactNode; statusBar?: ReactNode; collapsible?: boolean; onClose?: () => void; children }` | Absolutely positioned window with `.bevel-out` frame, `.titlebar` (gradient `win-blue → win-blue2`, white bold 12 px title, three 16×14 `.btn95` glyphs `_ □ ×`: `_` collapses to the title bar, `□` toggles 1.5× zoom of the body via CSS transform, `×` calls `onClose` or shakes the window 3 × 4 px). Title bar drag with pointer events (capture, clamp to the desktop); position persisted to `localStorage["fly.win." + id]` (JSON `{x,y}`), restored on mount; focused window gets the highest z-index (`fly.win.z` counter). |
| `PaintWindow` | `{ sock: FlySocket; onAction(a: MenuAction): void; canvasRef: RefObject<FlyCanvasHandle> }` | Title `untitled - Paint ($FLY)` with the Paint icon; layout grid: `MenuBar` row, then `[ToolPalette \| workspace]`, then `ColorPalette`, then `StatusBar`. Workspace = `win-gray` area with the white 800×500 document (`hello.canvas`) inside a `.bevel-in` frame with three 3 px resize handles drawn (inert). Default size 880×680 at (16, 16). |
| `MenuBar` | `{ sock: FlySocket; onAction(a: MenuAction): void }` | Menus (click to open, hover to switch, Esc/click-out to close, Win95 look): **File** → New (Ctrl+N: `clear` + local clear), Save As... (download the composited PNG, allowed client-side), Exit (shake); **Edit** → Undo (inert), Clear trail (= New); **View** → Raster labels (toggle), Freeze raster (toggle), Zoom 1.5× (toggle), Show FPS; **Image** → Flip fly (mirror sprite), Clear Image (= New); **Options** → Poke ▸ Sugar / Bitter / Sell wall (loom) / Water / Dust / Pheromone / Sleep / Reward / Punish; Market regime ▸ CALM / PUMP / DUMP / CHOP / RUG / DEAD / sim / dexscreener; Test tweet; **Help** → About FlyBrain (dialog with `hello.connectome` name/source/n/e/gain/license/citation/**note**, run_id, backend, dt, RTF; the synthetic note is shown in bold). Items that are inert render but do nothing (retro feel). `MenuAction = {kind:"clear"}\|{kind:"save"}\|{kind:"poke"; stim: PokeStim}\|{kind:"market"; mode: MarketMode}\|{kind:"tweet_test"}\|{kind:"toggle"; what:"labels"\|"freeze"\|"zoom"\|"fps"\|"flip"}\|{kind:"about"}`. |
| `ToolPalette` | `{ active: string; onPoke(stim: PokeStim): void; lastPoke: {stim: PokeStim; at: number}\|null }` | 2×8 grid of 24 px `.btn95` buttons with inline-SVG pixel glyphs (free-form select, rectangle select, eraser, fill, pick colour, magnifier, pencil, brush, airbrush, text, line, curve, rectangle, polygon, ellipse, rounded rectangle); `pencil` is rendered pressed. Below: five 24 px **poke tools** (sugar cube, red candle = loom, bitter leaf, water drop, dust) that call `onPoke`; the tool of `lastPoke` flashes pressed for 300 ms (also when the poke came from another client via the `poke` event). Below that, the Paint "active tool" box shows the current `ink.width` as a line sample. |
| `ColorPalette` | `{ ink: InkStyle; mood: Mood }` | The classic 2×14 Paint swatches (`#000000 #808080 #800000 #808000 #008000 #008080 #000080 #800080 #808040 #004040 #0080ff #004080 #8000ff #804000` / `#ffffff #c0c0c0 #ff0000 #ffff00 #00ff00 #00ffff #0000ff #ff00ff #ffff80 #00ff80 #80ffff #8080ff #ff0080 #ff8040`); the swatch nearest `ink.color` (RGB distance) is outlined; the two-square current-colour box shows `ink.color` (fore) over `moodColor(mood)` (back). |
| `StatusBar` | `{ store: TickStore; hello: HelloMsg\|null; status: WsStatus; attempt: number; fps: number\|null }` | `.bevel-in` cells: help text `For Help, click Help Topics on the Help Menu.` (replaced by the last event kind for 2 s); `${round(x)},${round(y)}`; `800 x 500`; mood badge in `moodColor`; `${symbol} $${fmtPrice(price)} ${chg_m5 >= 0 ? "+" : ""}${chg_m5}%`; `RTF ${rtf.toFixed(2)} · brain ${speed.toFixed(2)}x · ${n} n / ${e} e`; `SIM`/`DEX`/`SIM(fallback)` badge; WS dot (green open / yellow connecting / red reconnecting (n)). Re-renders at 4 Hz via `useTickSnapshot`. `fmtPrice` uses subscript-zero notation for tiny prices (`0.0₅1234`). |
| `FlyCanvas` | `{ sock: FlySocket; onFps?(fps: number): void; flip: boolean }`; imperative handle `FlyCanvasHandle { clear(): void; composite(): HTMLCanvasElement; }` via `useImperativeHandle` | Two stacked `<canvas width=800 height=500>` (CSS `width:100%`, `image-rendering: pixelated`): **trail** (never cleared except `clear()`), **overlay** (cleared every frame). rAF loop per frame: `latest = store.latest()`; if `latest.tick.seq` changed since the last frame → for each new tick draw the trail segment from the last drawn pose to the interpolated pose with `drawInk(tick.ink)`, apply `drawStamp` when `ink.stamp` is set, and handle `tick.events` (`jump` → three streaks on the overlay for 400 ms; `wall_bump` → `bump` mark); then `pose = interpolate(prev, latest, now, hello.tick_ms)`, `drawFly(overlay, pose, now/1000, mood)`; skip trail drawing while `pose.mode == "jump"`. On `event` frame `clear` → `clear()`. The trail layer also accepts `snapshot` compositing (§e.6). Pointer down on the document sends `poke` with the active poke tool (default `sugar`), `side` from the click x relative to the fly (`"L"` if left of the fly heading, else `"R"`), `strength = 1`, `duration_ms = 500`, rate-limited to 1 per 500 ms client-side. |
| `SpikeRaster` | `{ sock: FlySocket; frozen: boolean; labels: boolean }` | Wraps `RasterPainter`; rAF: `painter.frame(now)`, and `painter.push(tick)` for each new tick; hover tooltip from `rowAt`; a header line `spikes/s ${(spikes / (win_ms/1000)).toFixed(0)} · active ${(active_frac*100).toFixed(2)}%`; `capped` shows a small `▼` badge. Lane order = `hello.regions` (= `REGIONS`). |
| `MoodPanel` | `{ sock: FlySocket }` | 4 Hz snapshot. Row 1: mood label in `moodColor` with a 32×32 pixel face per mood (SLEEP closed eyes + z, CRUISING neutral, FEEDING proboscis, EUPHORIA spiral eyes, ANXIOUS sweat drop, PANIC wide eyes, ESCAPE motion lines, COURTSHIP heart eyes) and `since ${(since_ms/1000).toFixed(0)}s`. Row 2: Win95 segmented progress bars for `euphoria`, `anxiety`, `arousal`, `hunger`, `sleep` (0–1) and `valence` (−1..1 centred), with threshold ticks at the §f.6 values (euphoria 0.70/0.40, anxiety 0.35/0.45/0.70). Row 3: drive meters `sugar bitter looming(L/R arrow by loom_side) flash odor chop courtship sleep_pressure explore`. Row 4: a 300 × 12 px mood timeline (last 5 min, one column per 4 ticks, coloured by mood). Row 5: population table for `gf_L gf_R dn_freeze dng100 steer_a02_L steer_a02_R feed_mn flight_dn p1 pam ppl1` with 8 s sparklines (client ring) and `Hz` values. Footer: `run_id`, connectome name/source, `n/e`, gain, dt, backend, `rtf`, `speed`. |
| `MarketTicker` | `{ sock: FlySocket }` | Subscribes to `market` frames (bus) + 4 Hz snapshot: symbol + source badge (`SIM` / `DEX` / `SIM(fallback)`) + regime chip; price (`fmtPrice`, monospace); `chg_m5 / chg_h1 / chg_h24` coloured green/red with ▲▼ glyphs; buys vs sells m5 as a two-colour bar; `vol_m5`, `liq_usd`, `mcap` (compact `$1.23M`); last trade (`kind`, usd, age, `~` prefix when surrogate); a 5-minute price sparkline and a trade tape (last 12 trades) built from `market` frames; Options ▸ regime buttons duplicated here as `.btn95` chips. |
| `TweetNotepad` | `{ sock: FlySocket; onTest(): void }` | Title `tweets.txt - Notepad`; loads `getTweets(20)` on mount, prepends live `tweet` frames; each entry `[HH:MM:SS] <mood> (<dry-run\|posted>) <text>` in monospace 12 px, a `DRY RUN` stamp when `dry_run`, an `open on X` link when `url`; the neurons list as small chips; `error` in red; "Generate test tweet" `.btn95` calls `onTest` (sends `tweet_test`), disabled for 5 s after use. |

### e.6 Snapshot service (`snapshot_request` → `snapshot`)

Lives in `app/page.tsx` (owns the `FlyCanvasHandle`). On `snapshot_request {id, deadline_ms}`: `c = canvasRef.current.composite()`
→ an offscreen 800×524 canvas = trail layer + overlay + a 24 px Paint-style caption bar at the bottom
(`FlyBrain | <mood> | <symbol> <price> <chg_m5>% | t=<t_ms/1000>s`, 12 px, `win-gray` background); `png_b64 =
c.toDataURL("image/png").split(",")[1]`; `send({type:"snapshot", id, png_b64})` (and, if the socket is not open,
`postSnapshot(id, png_b64)`). Must answer within `deadline_ms` (the composite takes < 20 ms). The same `composite()` feeds
File ▸ Save As.

### e.7 `app/layout.tsx`, `app/page.tsx`, `globals.css`, keyboard, desktop

* `layout.tsx` (server component): `<html lang="en"><body className="bg-win-teal">`, `metadata = { title: "FlyBrain - CRUISING",
  description: "SynapseFly ($FLY): a spiking fly brain painting the market", icons: { icon: "data:image/svg+xml,..." } }`; no fonts import.
* `page.tsx` (`"use client"`): creates the socket (`useFlySocket()`), `canvasRef`, UI flags (`labels`, `frozen`, `zoom`, `flip`,
  `fps`), the `MenuAction` dispatcher, keyboard shortcuts, the snapshot service; renders the desktop with four windows:
  `PaintWindow` (16,16, 880×680), `Win95Window id="raster"` "Oscilloscope - spike raster" (912,16, 560×470) containing `SpikeRaster`,
  `Win95Window id="status"` "Fly Status" (912,500, 560×260) containing `MoodPanel` + `MarketTicker` (two columns), and
  `Win95Window id="tweets"` "tweets.txt - Notepad" (16,712, 880×180, collapsible) containing `TweetNotepad`. Below 1280 px width
  the windows stack vertically (CSS grid fallback, drag disabled). A Win95 taskbar strip at the bottom shows the four window
  buttons and a clock.
* Keyboard (document-level, ignored while a menu is open or an input is focused): `S` sugar poke, `B` bitter, `L` loom
  (sell wall), `W` water, `D` dust, `C` pheromone (courtship), `Z` sleep, `R` reward, `P` punish; `N` new canvas (clear);
  `F` freeze raster; `T` test tweet; `1..6` market regime CALM/PUMP/DUMP/CHOP/RUG/DEAD; `0` back to `sim`; `?` opens About.
* `globals.css`: Tailwind import, `@theme` tokens (§e.0), `.bevel-out { border: 2px solid; border-color: #fff #808080 #808080 #fff; background: #c0c0c0 }`,
  `.bevel-in { border: 2px solid; border-color: #808080 #fff #fff #808080; background: #fff }`,
  `.titlebar { background: linear-gradient(90deg, #000080, #1084d0); color: #fff; font-weight: bold; padding: 2px 4px; user-select: none }`,
  `.btn95 { @apply bevel-out; padding: 0 6px } .btn95:active, .btn95[aria-pressed="true"] { border-color: #808080 #fff #fff #808080 }`,
  `canvas { image-rendering: pixelated }`, `body { font: 12px "MS Sans Serif", "Microsoft Sans Serif", Tahoma, Arial, system-ui, sans-serif; margin: 0 }`.

### e.8 Performance and robustness rules

* Canvases: trail layer is persistent (drawn incrementally), overlay is cleared per frame; the raster is a single
  `drawImage` self-shift per frame plus ≤ 2000 `fillRect`s per tick; panels re-render at 4 Hz. Target: 60 fps on an
  integrated GPU laptop with all four windows open.
* Background tabs: rAF pauses; on resume, `interpolate` detects the gap (> 4 ticks) and snaps (no long segments).
* `market` frames are buffered (≤ 600 trades / 5 min) for the sparkline; `tweet` frames ≤ 50; in-tick events ≤ 200.
* Every WS handler is wrapped in `try/catch` and logs to the console; a malformed frame never breaks the loop.
* `localStorage` access is wrapped (private mode); missing values fall back to the `initial` window positions.

---

## f. Encoder, decoder and mood formulas (exact numbers)

Notation: `sat(x) = clip(x, 0, 1)`, `relu(x) = max(0, x)`, `hill(x; k, n) = x^n / (k^n + x^n)` (§c.17), `EMA_τ` = first-order
low-pass `y ← y + (x − y)·(1 − exp(−dt/τ))`. `dt` is the wall tick in seconds (`tick_ms/1000`, 0.05) for market features
and mood timers, and the **brain** seconds of the tick (`steps·dt_ms/1000`, i.e. `0.05·speed`) for decoder/body
integration — so the fly slows down with the brain when `speed < 1`. Every number below tagged `[E]` is engineered
(puppeteering or stand-in); `[V]`/`[L]` are anchored in RESEARCH.

### f.1 Market → `Features` (`FeatureExtractor.update`, §c.17)

State kept between ticks: `val, activity, rug, vol_norm, water_imp, imp_sugar, imp_loom, flash_until, hunger, courtship,
z (sleep), odor_rise_t, ath, liq_start, liq_prev, price_prev, ema_vol_1h, trade_sizes (deque 200), last_trade_wall,
loom_episode, loom_ramp_start_ms, loom_last_sell_ms, loom_side, sustained_timer, consecutive_bull_snaps`.

**On every new snapshot** (`snap.seq` changed; `P = price_usd or price_native`, `P_prev` = previous snapshot price):

```
n        = buys_m5 + sells_m5
bp       = (buys_m5 - sells_m5) / (n + 1)                                  # -1..1
mom      = tanh(chg_m5 / 2.0)
tick     = tanh(100 * (P / P_prev - 1))  if P and P_prev else 0.0
val_snap = clip(0.5*bp + 0.3*mom + 0.2*tick, -1, 1)
act_snap = sat(log1p(n) / log1p(500))
rug_snap = 1.0 if (chg_m5 <= -15) or (liq_usd and liq_start and liq_usd < 0.7*liq_start) else 0.0
ema_vol_1h <- EMA_3600(vol_m5)  (initialised to vol_m5) ;  vol_snap = sat(vol_m5 / (3*ema_vol_1h + 1))
water_imp += sat((liq_usd/liq_prev - 1) / 0.05)  if liq_usd and liq_prev and liq_usd > liq_prev
candle   = "up" if P > 1.0005*P_prev else "down" if P < 0.9995*P_prev else "flat"
ath      = max(ath, P) ; consecutive_bull_snaps += 1 if (buys_m5 >= 3*sells_m5 and chg_m5 > 5) else reset to 0
```

**Every tick** (wall `dt`):

```
val      <- EMA_3(val_snap)        activity <- EMA_10(act_snap)      vol_norm <- EMA_5(vol_snap)
rug      <- max(rug_snap, rug*exp(-dt/20))                            water_imp <- water_imp*exp(-dt/10)
for each Trade tr (in ts order):
    trade_sizes.append(tr.usd) ; usd_ref = max(20, median(trade_sizes)) ; last_trade_wall = tr.ts
    s = min(1.5, tr.usd / usd_ref)
    buy : imp_sugar += 0.15*s
    sell: imp_loom  += 0.20*s ; loom_last_sell_ms = t_ms ; loom_ramp_start_ms = t_ms          # new expanding disc
          if no episode active (t_ms - loom_last_sell_ms > 2000 before this trade): loom_episode += 1 ;
              loom_side = rng.choice([-1, +1]) (sim) / +1 if int(tr.ts*1000) % 2 else -1 (dex)
    whale = tr if tr.usd >= 10*usd_ref else whale ; a whale SELL sets flash_until = t_ms + 300
imp_sugar <- imp_sugar*exp(-dt/1.5) ; imp_loom <- imp_loom*exp(-dt/1.5)
up       = sat(val)                       down = sat(-val)
sugar    = sat(relu(val) + imp_sugar)                                                   # 0..1
bitter   = sat(0.5*relu(-val) + 0.5*rug)
looming  = sat(1.2*relu(-val) + imp_loom)
flash    = 1.0 if t_ms < flash_until else 0.0
loom_t_ms = t_ms - loom_ramp_start_ms if (loom_ramp_start_ms is not None and t_ms - loom_ramp_start_ms < 300) else None
episode active while t_ms - loom_last_sell_ms <= 2000 ; when inactive loom_side = 0
sustained_timer  += dt if looming >= 0.20 else reset ;  sustained_loom = sustained_timer >= 3.0
water    = sat(water_imp)
odor     = sat(0.4*activity + 0.6*vol_norm) * (0.4 + 0.6*exp(-(t - odor_rise_t)/2))   # olfactory adaptation tau 2 s [L]
           (odor_rise_t <- t whenever the raw sat(...) rises by > 0.1 within one tick)
chop     = sat(activity * (1 - min(1, abs(chg_m5)/2)))
courtship: set to 1.0 when P >= 1.02*previous ath (new session high by 2 %), or consecutive_bull_snaps >= 3, or
           easter_egg() != None (event easter_egg), or an active 'pheromone' poke ; then courtship <- courtship*exp(-dt/8)
z        += dt/90 if activity < 0.35 else -dt/10 ; z = clip(z, 0, 1) ; sleep_pressure = z
hunger   += dt/600 ; if feeding: hunger -= 0.06*sat(mn9_hz/60)*dt ; hunger = clip(hunger, 0, 1) (initial 0.5)
sweet_gain = 0.6 + 0.4*hunger                                                            # hunger raises sweet sensitivity [L]
explore  = 0.0 if mood == "SLEEP" else settings.explore_baseline + 0.25*activity          # [E] guaranteed-motion floor
since_last_trade_s = now - last_trade_wall (1e9 if none)
```

`easter_egg()` (only when `settings.easter_eggs`): `buys_m5 in (69, 420)`; the first three significant digits of `P`
are `420` or the first two are `69`; local time `HH:MM` is `04:20` or `16:20` (fires once per minute). Returns the reason
string of §d.8.

### f.2 `Features` → `list[Drive]` (`SensoryEncoder.encode`, §c.17)

The encoder emits the following Drives every tick (skipping any with `rate_hz < 1`), each applied by the loop with
`engine.inject(group, rate_hz=…, duration_ms=brain_ms, recruit, side, episode, tag=group+":"+channel, weights)`. In
`FLY_DRIVE_MODE=current` the same list is emitted with `mode="current"` and `current_mv = current_from_rate(rate_hz)`.
`t = t_ms/1000`.

| # | channel | group | rate_hz | recruit | side | episode | weights | prov |
|---|---|---|---|---|---|---|---|---|
| 1 | sugar | `grn_sugar` | `150·hill(sweet_gain·sugar; 0.25, 1.5)` (sugar 1 → 133 Hz, 0.5 → 111, 0.2 → 62, 0.05 → 12) | `0.4 + 0.6·sugar` | 0 | 0 | — | [L] Shiu: sugar GRNs 10–200 Hz |
| 2 | water | `grn_water` | `60·water` | 1 | 0 | 0 | — | [E] |
| 3 | bitter | `grn_bitter` | `120·hill(bitter; 0.25, 1.5)` | `0.4 + 0.6·bitter` | 0 | 0 | — | [L] |
| 4 | loom_ramp | `lc_loom` | `220·looming·(loom_t_ms/300)²` while `loom_t_ms` is not None (expanding disc, 300 ms) | 0.6 | `loom_side` | `loom_episode` | — | [L] loom-tuned LC response, [E] numbers |
| 5 | loom_tonic | `lc_loom` | `12·relu((looming − 0.3)/0.7)²` | 1 | `loom_side` | 0 | — | [E] primes the GF (5 mV g_ss at 12 Hz) |
| 6 | loom_aux | `lc_loom2` | `0.5 × (row 4 + row 5)` | 0.6 | `loom_side` | `loom_episode` | — | [E] |
| 7 | flash | `lc_loom` | `220` while `flash > 0` (whale dump = instantaneous full-field expansion) | 1 | 0 | `loom_episode` | — | [E] |
| 8 | freeze | `lc_freeze` | `80·hill(looming; 0.3, 2)` only while `sustained_loom` (slow sustained threat → LC9/LC31a → DNp09) | 1 | `loom_side` | 0 | — | [V] pathway, [E] numbers |
| 9 | light | `photoreceptor` | `15 + 45·activity` | 1 | 0 | 0 | `w_i = 1 + 0.5·sin(2π·f_flick·t + φ_i)`, `f_flick = 2 + 6·activity` Hz, `φ_i ~ U(0,2π)` fixed per cell | [E] moving flicker → T4/T5 |
| 10 | odor | `orn` | `80·hill(odor; 0.3, 1.5)` | 1 | 0 | 0 | `w_i = 1.0` for the six food glomeruli `ORN_(DM1\|DM4\|DM2\|VM2\|VA2\|DL1)`, `0.25` otherwise | [E] volume = smell of food |
| 11 | reward | `pam` | `40·up` | 1 | 0 | 0 | — | [E] |
| 12 | punish | `ppl1` | `40·down` | 1 | 0 | 0 | — | [E] |
| 13 | dust | `jo_groom` | `60·hill(chop; 0.3, 1.5)` | 1 | 0 | 0 | — | [E] chop = dust on the antenna |
| 14 | pheromone | `grn_pher` | `100·hill(courtship; 0.3, 1.5)` | 1 | 0 | 0 | — | [?] identity |
| 15 | song_in | `jo_aud` | `100·hill(courtship; 0.3, 1.5)` | 1 | 0 | 0 | — | [D] |
| 16 | sleep | `dfb_sleep` | `10·sleep_pressure` | 1 | 0 | 0 | — | [L] dFB |
| 17 | explore | `dng100` | `30·explore + 20·sugar·(1 − feeding)` (baseline ≥ 7.5 Hz when `explore_baseline = 0.25`) | 1 | 0 | 0 | — | [E] **guaranteed motion**; 0 when `explore_baseline = 0` |
| 18 | compass | `epg` | `40` | 1 | 0 | 0 | wedge `k = floor(((heading mod 2π)/2π)·16)`; `w = 1.0` for EPG cells of wedge k, `0.375` for wedges k±1 (→ 15 Hz), else 0 (wedge of an EPG cell = `floor(16·rank_within_side/n_side)`) | [E] proprioceptive heading feedback |
| 19 | mood_fb | see below | only when `settings.mood_feedback` | | | | | [E] circular by design |
| 20 | poke | `POKE_TABLE[stim].group` | `POKE_TABLE[stim].rate × strength` for every active `Poke` (`until_ms > t_ms`) | 1 | `poke.side` | 0 | — | user |

Mood feedback (row 19, all `[E]`, labelled `channel="mood"`): `EUPHORIA` → `pam` 60 Hz and `flight_dn` `40·euphoria` Hz;
`PANIC` → `ppl1` 60 Hz and `dn_freeze` 20 Hz; `ANXIOUS` → `ppl1` 20 Hz; a `courtship` upward crossing of 0.5 (or a
`pheromone` poke) → `p1` 80 Hz for 6 s (tag `p1:court`, tracked by the encoder); `SLEEP` → every other row's rate × 0.3
(raised arousal threshold). Wake-up "grogginess": for 2 s after leaving `SLEEP` all rates × 0.5.

`POKE_TABLE` (§c.17): `sugar (grn_sugar, 150)`, `bitter (grn_bitter, 120)`, `loom (lc_loom, 150)`, `water (grn_water, 60)`,
`dust (jo_groom, 80)`, `pheromone (grn_pher, 100)`, `sleep (dfb_sleep, 20)`, `reward (pam, 80)`, `punish (ppl1, 80)`,
`explore (dng100, 40)`. A `loom` poke additionally sets `loom_side = poke.side` and starts a ramp episode in the
`FeatureExtractor` (so pokes look exactly like a sell wall). The wall bump pulse (`steer_a02_<side>` 40 Hz / 100 ms) is
injected by the loop, not the encoder (§c.27 step 6).

Sanity of the numbers with the calibrated synthetic graph (§g): sugar 0.5 → labellar GRNs 111 Hz → `GNG215/232` g_ss ≈
15 mV → `GNG108` ≈ 14 mV → MN9 net ≈ 14 − 3 (tonic GABA) = 11 mV → `rate_from_current(11) ≈ 45 Hz` (feeding threshold
30 Hz) ✓. A sell wall (`looming` 0.7, ramp on 60 % of LC4/LPLC2): rate reaches 220·0.7 = 154 Hz at 300 ms; the GF's
`g_ss` at 33 Hz population rate is already 14 mV, so DNp01 fires ≈ 100–150 ms into the ramp (§g.6 gate: ≤ 20 ms at a
150 Hz full-recruitment step).

### f.3 Population rates → `MotorCommand` (`MotorDecoder.update`, §c.18)

`Readouts.from_stats(s)` maps `s.rates` (§c.12 keys) to fields: `fwd = max(dng100, dn_fwd)`, `halt = dn_halt`,
`back = dn_back`, `freeze = dn_freeze`, `a02_l/r = steer_a02_L/R`, `a01_l/r = steer_a01_L/R`, `g13_l/r = steer_g13_L/R`,
`flight = flight_dn`, `wing_power = wing_power`, `b1_l/r, i1_l/r, hg1_l/r = b1_L/R, i1_L/R, hg1_L/R`, `saccade_l/r =
dn_saccade_L/R`, `land = dn_land`, `mn9 = feed_mn`, `groom = groom_dn`, `p1 = p1`, `pip10 = pip10`, `song_mn = 0.5·(hg1 + b1)`,
`dms2_l/r = dms2_L/R`, `gf_spikes_l/r = s.gf_spikes["L"/"R"]`, `dnp11_spikes = s.spike_counts_by_group["escape_dn"]`
(pooled DNp02/04/11 count, used as the forward-takeoff bias [E]), `lc_loom = lc_loom`, `dn_mean = dn_mean`.

Derived signals (all `[L]` unless tagged):

```
turn       = clip((a02_r - a02_l)/40 + 0.5*(g13_r - g13_l)/60 + 0.3*(a01_r - a01_l)/60, -1, 1)   # Yang 2024: ipsiversive; + = clockwise
halt       = sat(r.halt / 30)          back = sat(r.back / 20)
wing_power = sat(sat(r.flight/50) + 0.3*sat(r.wing_power/30))
feeding    : enter when r.mn9 >= 30 Hz for >= 200 ms ; exit when r.mn9 < 15 Hz for >= 1000 ms ; proboscis = sat((r.mn9 - 15)/45) while feeding
freeze     : enter when r.freeze >= 40 Hz and (f.looming >= 0.2 or f.sustained_loom) ; min 800 ms ; exit when r.freeze < 20 Hz or f.looming < 0.1 for 500 ms   # Zacarias 2018
groom      : enter when r.groom >= 40 Hz ; min 3000 ms ; exit when r.groom < 15 Hz
court      : while mood == COURTSHIP or r.p1 >= 15 Hz
song       : r.pip10 >= 20 Hz or r.song_mn >= 20 Hz -> wing_ext = +1 if dms2_r >= dms2_l else -1 (event song{side}) ; else 0
jump       : (gf_spikes_l + gf_spikes_r) > 0 and t_ms - last_jump_ms >= 1500                          # GF refractory [E]
fly        : enter when wing_power >= 0.3 for 200 ms, or at the end of a jump when f.looming >= 0.5 (event takeoff)
             exit (event landing) when r.land >= 20 Hz or wing_power < 0.15 for 2000 ms
saccade    : in fly mode, r.saccade_l >= 40 Hz or r.saccade_r >= 40 Hz -> heading += (+pi/2 if R else -pi/2) over 100 ms (once per 500 ms)
```

Mode priority (first true wins): `jump` (400 ms after a GF spike) > `sleep` (mood SLEEP) > `court` > `feed` > `freeze` >
`groom` > `fly` > `walk`.

Jump (event `jump`): `side = "L"|"R"|"both"` from which GF fired; `forward = dnp11_spikes > 0`;
`heading_out = heading + U(−π/6, π/6)` if forward else `heading + π + U(−π/4, π/4) + (−π/6 if side == "L" else +π/6 if side == "R" else 0)`
(flee away from the threatened side; `+` is clockwise = toward the fly's right); `impulse = 600` px/s; `jump = {heading,
impulse, side, forward}`; during the 400 ms jump `wing_hz = 250`, no trail; then `fly` if `f.looming >= 0.5` else `walk`.

Per-mode command:

| mode | `v_target` (px/s) | `omega` (rad/s) | `wing_hz` | `wing_amp` | `halt_reason` |
|---|---|---|---|---|---|
| walk | `120·sat(fwd/30)·(1 − halt)`; if `back > 0.5`: `−40·back` | `4.0·turn + η` | 0 | 0 | `halt_dn` if halt ≥ 0.8 |
| fly | `180 + 220·wing_power` | `7.0·turn + 0.02·((b1_r + i1_r) − (b1_l + i1_l)) + η` | `150 + 100·wing_power` | `0.6 + 0.4·sat(r.wing_power/15)` | — |
| jump | impulse (body handles decay) | 0 | 250 | 1 | — |
| court | 60 | `+3.0` (circling, clockwise) | 0 | 0 | — |
| feed | 0 | 0 | 0 | 0 | `feed` |
| freeze | 0 (body adds ±1.5 px tremor) | 0 | 0 | 0 | `freeze` |
| groom | 0 | 0 | 0 | 0 | `groom` |
| sleep | 0 | 0 | 0 | 0 | `sleep` |

`η = Wander.sample(dt)` (§c.18: OU, `σ = settings.wander_sigma` rad/s/√s, τ 1 s) in `walk`/`fly` only `[E]`.
**Wander floor `[E]`** (guaranteed motion; active only when `settings.explore_baseline > 0`): in `walk` mode, if
`|kin.speed| < 5` px/s for more than 3000 ms (`stall_timer`), set `v_target = 40` px/s until `fwd > 5` Hz; emit
`wander_floor {stalled_ms}` once per engagement. The explore baseline (`dng100` ≥ 7.5 Hz → `fwd/30 ≥ 0.25` → 30 px/s)
normally prevents it from ever engaging; the event count is reported in `selftest.py` so puppeteering is measurable.

### f.4 Kinematics integration (`FlyBody.integrate`, §c.18)

```
tau_v = 0.15 (walk) | 0.30 (fly) | 0.05 (court/feed/freeze/groom/sleep)
jump : v = impulse * exp(-jump_t/0.15)                  (jump_t in s since the jump started; mode 'jump' lasts 400 ms)
else : v <- v + (v_target - v) * (1 - exp(-dt/tau_v))
heading <- wrap(heading + omega*dt)  to (-pi, pi]          # jump: heading = jump.heading at jump start; saccade adds its ramp
vx = v*cos(heading) ; vy = v*sin(heading) ; x += vx*dt ; y += vy*dt
walls (margin m = 8 px):
  bounce: x < m -> x = m, heading = pi - heading, event wall_bump{side}  (x > w-m mirrored) ; y < m -> y = m, heading = -heading ; (y > h-m mirrored)
          side = "L" if the wall lies to the fly's left before reflection (sign of cross(heading_vec, wall_normal) < 0) else "R"
  wrap  : x < 0 -> x += w (event wrap{edge:"left"}), etc. (trail pen is lifted by the > 40 px segment rule on the client)
leg_phase <- (leg_phase + |v|/20 * dt) mod 1               # one gait cycle per body length (20 px)
speed = v (signed: negative while backing) ; omega as commanded ; jump_t_ms = ms since jump start while mode == jump else None
freeze tremor: the wire pose gets x,y + U(-1.5, 1.5) px each tick (the true position is unchanged)
```

Start pose: `(w/2, h/2)`, heading 0, mode walk. `teleport()` lifts the pen (next segment > 40 px is skipped client-side).

### f.5 Ink (`FlyBody.ink`, §c.18 `InkStyle`)

```
color : "#00a800" if f.candle == "up" else "#a80000" if "down" else "#000000"
        overridden by mood: FEEDING "#ff8c00", COURTSHIP "#ff69b4", PANIC "#ff0000", ESCAPE "#ff3300", SLEEP "#6b6b6b", EUPHORIA "#ff00ff" (style rainbow overrides)
width : 1 + round(3 * sat(abs(chg_m5) / 3))              # 1..4 px
alpha : 1.0 walk/feed/freeze/groom/sleep ; 0.4 fly ; 0.7 court ; 0.0 jump (segment skipped anyway)
style : "rainbow" EUPHORIA | "zigzag" PANIC/ESCAPE | "dotted" FEEDING | "hearts" COURTSHIP | "solid" otherwise
stamp : "blob" every 250 ms while feeding | "heart" every 500 ms in court | "zzz" every 500 ms in sleep | "dash" during jump | "bump" on a wall_bump tick | None
```

### f.6 Mood machine (`MoodMachine.update`, §c.19)

Scores (EMA `τ = tau_score_s = 2 s`, inputs from `MoodInputs`):

```
E_raw   = 0.45*sugar + 0.35*sat(mn9/60) + 0.20*sat(pam/40)
A_raw   = 0.50*looming + 0.30*sat(dnp09/40) + 0.20*sat(lc_loom/60)
euphoria <- EMA_2(E_raw) ; anxiety <- EMA_2(A_raw) ; fear = anxiety (wire alias)
arousal  <- EMA_2(0.5*sat(dn_mean/5) + 0.5*any_drive_max)
valence  = clip(euphoria - anxiety + 0.2*(mbon_approach - mbon_avoid)/(mbon_approach + mbon_avoid + 5), -1, 1)
hunger   = x.hunger (passthrough from Features via the loop) ; sleep = sleep_pressure
```

Transition table (evaluated once per tick; "held T" = the condition has been continuously true for T ms — timers
accumulate only while the condition holds and reset otherwise; **minimum dwell 1500 ms** in every state before a
non-ESCAPE transition (`dwell_left_ms`); at most one transition per tick):

| state | enter when (priority order ESCAPE > PANIC > COURTSHIP > EUPHORIA > FEEDING > ANXIOUS > SLEEP) | exit when → target |
|---|---|---|
| ESCAPE | `jumped` (immediate, ignores dwell) | after 1500 ms → PANIC if `anxiety ≥ 0.45`, ANXIOUS if `≥ 0.20`, else CRUISING |
| PANIC | `anxiety ≥ 0.70` held 2000, or `gf_spikes_10s ≥ 3` | `anxiety < 0.45` held 3000 → ANXIOUS |
| COURTSHIP | (`p1 ≥ 15` or `pip10 ≥ 20`) held 500, or `forced == "COURTSHIP"` | dwell 8000 ms, then `p1 < 5 and pip10 < 5` held 2000 → CRUISING |
| EUPHORIA | `euphoria ≥ 0.70` held 3000 and `anxiety < 0.35` | `euphoria < 0.40` held 2000 → FEEDING if `feeding` else CRUISING |
| FEEDING | `feeding` (decoder flag) | `not feeding` → CRUISING |
| ANXIOUS | `anxiety ≥ 0.35` held 1000, or `bitter ≥ 0.4` held 2000 | `anxiety < 0.20 and bitter < 0.2` held 5000 → CRUISING |
| SLEEP | `sleep_pressure ≥ 0.95 and activity < 0.35 and any_drive_max < 0.10 and dn_mean < 2` held 20000 | `any_drive_max > 0.10 or poked or jumped or activity ≥ 0.35` → CRUISING (event `wake`) |
| CRUISING | default | — |

Algorithm per tick: (1) update scores and timers; (2) if `jumped` → ESCAPE (from any state, resets dwell); (3) else if
`dwell_left_ms > 0` → no transition; (4) else scan the enter conditions of the states above the current one in priority
order (a state never re-enters itself) → first hit wins; (5) else if the current state's exit condition holds → its
target. `Transition.reason` is the human string of the fired rule, e.g. `"euphoria 0.74 >= 0.70 for 3.0 s"`,
`"mn9 < 15 Hz for 1.0 s"`. `MoodMachine.confirmed(hold_ms=3000)` returns the last transition exactly once after its
destination has been held for 3 s (never for ESCAPE) — that is the agent trigger (§c.24), so flicker never tweets.
`MOOD_COLORS` of §c.19 are the UI colours; `MoodState.to_wire()` is the `tick.mood` object of §d.2.

---

## g. Synthetic connectome generator (`connectome/synthetic.py`, §c.6)

The generator is a **structured stand-in shaped like MaleCNS v1.0**: every core neuron carries a real MaleCNS `type`
string (so every `GROUP_REGEX` of §c.4 resolves on it, `test_synthetic::test_all_groups_resolve`), the 8-region
taxonomy of §c.2, a real `superclass`/`class` (so `region_of()` reproduces the table's region column,
`test_synthetic::test_region_of_matches_table`), a neurotransmitter and therefore a sign, and 112 explicit projection
rules carrying the verified MaleCNS synapse counts (RESEARCH §5) with correct laterality. Everything else is a
subcritical random background. It is **not data**: `meta['note']` says so and every consumer (hello, About dialog,
LLM summary, README) repeats it.

### g.0 Rule DSL (how `POPULATIONS` and `PROJECTIONS` are interpreted)

* `Pop(type, region, n, nt, superclass, cls)` — `n` is both sides combined; sides are assigned `ceil(n/2)` L then
  `floor(n/2)` R (indices interleaved L,R,L,R,… so that `rank_within_side` is well defined); `n == 1` → side 0.
  `sign = NT_SIGN[nt]` (§c.7). Neuron indices are allocated **region by region in `REGIONS` order**, and inside a region
  in `POPULATIONS` order, core populations first, then that region's filler; `body_id = 1_000_000 + index`.
* `Proj(pid, src, dst, k_in, r_nom_hz, target_mv, laterality, topology, w_lit, provenance, w_cv)` — `src`/`dst` are
  resolved in this order: (1) exact key of `GROUP_REGEX` → that group; (2) otherwise a `'|'`-separated list of type
  patterns matched with `fnmatch` against the unique type list (`*` allowed, e.g. `ORN_*`, `*PN`, `KC*`, `PAM*`, `PPL10*`).
  A resolution that yields zero neurons on either side is a build error.
* `laterality`: `ipsi` = source and target on the same side (side-0 cells count as both sides), `contra` = opposite sides,
  `both` = sides ignored.
* `topology` (per **target** cell, `k = k_in`):
  * `random` — `k` distinct sources drawn uniformly (without replacement) from the allowed-side source set (fewer if the set is smaller);
  * `all` — every allowed-side source (`k_in` is informative; the calibrated weight is still computed from the table's `k_in`,
    which equals the per-side source count, so the two agree by construction);
  * `retinotopic` — sources and targets are ranked within side; the target at normalised rank `ρ_t ∈ [0,1)` takes the `k`
    sources whose normalised rank is nearest to `ρ_t` (columnar wiring; used for the optic lobe);
  * `glomerular` — the glomerulus token is the text before the first `_` of an `ORN_<G>` source and of a `<G>_…PN` target;
    connect all allowed-side sources with the same token (`k_in` = 20 = ORNs per glomerulus per side);
  * `wedge` — source and target both get a wedge index `floor(16·rank_within_side/n_side)`; connect equal wedges
    (`k_in` informative);
  * `ring_shift` — as `wedge` but the target wedge is `(source_wedge + 1) mod 16` for R-side sources and `(source_wedge − 1) mod 16`
    for L-side sources (PEN → EPG shift). For `Delta7 → EPG` (`all`) the implementation additionally skips the source's own wedge.
* Weight per resolved edge: `w_base = calibrated_weight(k_in, r_nom_hz, target_mv)` in `calibrated` mode (§c.6 formula,
  `gain = 1`) or `w_lit` in `literature` mode (rows with `w_lit == 0` — all `E` rows — use the calibrated value in both
  modes); then `w = max(1, round(w_base · LogNormal(0, w_cv)))` with `w_cv = 0.5` (per-edge jitter, seeded), capped at 400.
  The signed CSR value is `w · sign[src]`.
* `provenance`: `V` = verified pathway (RESEARCH §5), `D` = literature/digest, `E` = engineered stand-in (listed in
  `meta['engineered_edges']`).

### g.1 `POPULATIONS` — 289 rows, core = 6,535 neurons at `s = 1` (N ≥ 20,000)

Counts follow RESEARCH §4 where the real population is small (every DN/MN/GRN/pC1/MBON/PAM/CX type is at its true
count) and are scaled down for the large sensory/relay populations (R1-R6 3,377 → 400, KC 4,064 → 800, T4/T5 13,580 → 480,
ORN 2,635 → 320). Types marked `?` in RESEARCH (pheromone `LgLG1a/b`, `LgLG2`) are included at true count but only
reachable through `E` edges.

| # | type | region | n | nt | superclass | class |
|---|---|---|---:|---|---|---|
| 1 | `R1-R6` | optic_lobe | 400 | histamine | ol_sensory | visual |
| 2 | `R7p` | optic_lobe | 30 | histamine | ol_sensory | visual |
| 3 | `R7y` | optic_lobe | 30 | histamine | ol_sensory | visual |
| 4 | `R8p` | optic_lobe | 30 | histamine | ol_sensory | visual |
| 5 | `R8y` | optic_lobe | 30 | histamine | ol_sensory | visual |
| 6 | `L1` | optic_lobe | 120 | glutamate | ol_intrinsic | - |
| 7 | `L2` | optic_lobe | 120 | acetylcholine | ol_intrinsic | - |
| 8 | `L3` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 9 | `L4` | optic_lobe | 40 | acetylcholine | ol_intrinsic | - |
| 10 | `L5` | optic_lobe | 40 | acetylcholine | ol_intrinsic | - |
| 11 | `C2` | optic_lobe | 40 | gaba | ol_intrinsic | - |
| 12 | `C3` | optic_lobe | 40 | gaba | ol_intrinsic | - |
| 13 | `Mi1` | optic_lobe | 120 | acetylcholine | ol_intrinsic | - |
| 14 | `Tm3` | optic_lobe | 120 | acetylcholine | ol_intrinsic | - |
| 15 | `Mi4` | optic_lobe | 60 | gaba | ol_intrinsic | - |
| 16 | `Mi9` | optic_lobe | 60 | glutamate | ol_intrinsic | - |
| 17 | `Tm1` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 18 | `Tm2` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 19 | `Tm4` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 20 | `Tm9` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 21 | `T4a` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 22 | `T4b` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 23 | `T4c` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 24 | `T4d` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 25 | `T5a` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 26 | `T5b` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 27 | `T5c` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 28 | `T5d` | optic_lobe | 60 | acetylcholine | ol_intrinsic | - |
| 29 | `LC4` | optic_lobe | 126 | acetylcholine | visual_projection | - |
| 30 | `LPLC2` | optic_lobe | 185 | acetylcholine | visual_projection | - |
| 31 | `LC6` | optic_lobe | 124 | acetylcholine | visual_projection | - |
| 32 | `LPLC1` | optic_lobe | 134 | acetylcholine | visual_projection | - |
| 33 | `LPLC4` | optic_lobe | 97 | acetylcholine | visual_projection | - |
| 34 | `LC9` | optic_lobe | 219 | acetylcholine | visual_projection | - |
| 35 | `LC31a` | optic_lobe | 32 | acetylcholine | visual_projection | - |
| 36 | `ORN_DM1` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 37 | `ORN_DM4` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 38 | `ORN_DM2` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 39 | `ORN_VM2` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 40 | `ORN_VA2` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 41 | `ORN_DL1` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 42 | `ORN_DA1` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 43 | `ORN_DL3` | antennal_lobe | 40 | acetylcholine | cb_sensory | olfactory |
| 44 | `DM1_lPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 45 | `DM4_adPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 46 | `DM2_lPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 47 | `VM2_adPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 48 | `VA2_adPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 49 | `DL1_adPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 50 | `DA1_lPN` | antennal_lobe | 8 | acetylcholine | cb_intrinsic | ALPN |
| 51 | `DL3_lPN` | antennal_lobe | 4 | acetylcholine | cb_intrinsic | ALPN |
| 52 | `lLN1_a` | antennal_lobe | 20 | gaba | cb_intrinsic | ALLN |
| 53 | `lLN2T_a` | antennal_lobe | 20 | gaba | cb_intrinsic | ALLN |
| 54 | `KCg-m` | mushroom_body | 340 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 55 | `KCg-d` | mushroom_body | 40 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 56 | `KCab-c` | mushroom_body | 130 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 57 | `KCab-m` | mushroom_body | 90 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 58 | `KCab-s` | mushroom_body | 60 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 59 | `KCab-p` | mushroom_body | 20 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 60 | `KCa'b'-ap1` | mushroom_body | 40 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 61 | `KCa'b'-ap2` | mushroom_body | 40 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 62 | `KCa'b'-m` | mushroom_body | 40 | acetylcholine | cb_intrinsic | Kenyon_Cell |
| 63 | `APL` | mushroom_body | 2 | gaba | cb_intrinsic | - |
| 64 | `DPM` | mushroom_body | 2 | dopamine | cb_intrinsic | - |
| 65 | `MBON01` | mushroom_body | 2 | glutamate | cb_intrinsic | MBON |
| 66 | `MBON03` | mushroom_body | 2 | glutamate | cb_intrinsic | MBON |
| 67 | `MBON04` | mushroom_body | 2 | glutamate | cb_intrinsic | MBON |
| 68 | `MBON05` | mushroom_body | 2 | glutamate | cb_intrinsic | MBON |
| 69 | `MBON06` | mushroom_body | 2 | glutamate | cb_intrinsic | MBON |
| 70 | `MBON09` | mushroom_body | 2 | gaba | cb_intrinsic | MBON |
| 71 | `MBON11` | mushroom_body | 2 | gaba | cb_intrinsic | MBON |
| 72 | `MBON12` | mushroom_body | 4 | acetylcholine | cb_intrinsic | MBON |
| 73 | `MBON13` | mushroom_body | 2 | acetylcholine | cb_intrinsic | MBON |
| 74 | `MBON14` | mushroom_body | 2 | acetylcholine | cb_intrinsic | MBON |
| 75 | `MBON18` | mushroom_body | 2 | acetylcholine | cb_intrinsic | MBON |
| 76 | `PAM01` | mushroom_body | 44 | dopamine | cb_intrinsic | DAN |
| 77 | `PAM02` | mushroom_body | 26 | dopamine | cb_intrinsic | DAN |
| 78 | `PAM03` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 79 | `PAM04` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 80 | `PAM05` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 81 | `PAM06` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 82 | `PAM07` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 83 | `PAM08` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 84 | `PAM09` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 85 | `PAM10` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 86 | `PAM11` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 87 | `PAM12` | mushroom_body | 20 | dopamine | cb_intrinsic | DAN |
| 88 | `PAM13` | mushroom_body | 16 | dopamine | cb_intrinsic | DAN |
| 89 | `PAM14` | mushroom_body | 16 | dopamine | cb_intrinsic | DAN |
| 90 | `PAM15` | mushroom_body | 14 | dopamine | cb_intrinsic | DAN |
| 91 | `PPL101` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 92 | `PPL102` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 93 | `PPL103` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 94 | `PPL104` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 95 | `PPL105` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 96 | `PPL106` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 97 | `PPL107` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 98 | `PPL108` | mushroom_body | 2 | dopamine | cb_intrinsic | DAN |
| 99 | `EPG` | central_complex | 46 | acetylcholine | cb_intrinsic | CX |
| 100 | `PEN_a(PEN1)` | central_complex | 20 | acetylcholine | cb_intrinsic | CX |
| 101 | `PEN_b(PEN2)` | central_complex | 22 | acetylcholine | cb_intrinsic | CX |
| 102 | `PEG` | central_complex | 18 | acetylcholine | cb_intrinsic | CX |
| 103 | `Delta7` | central_complex | 42 | glutamate | cb_intrinsic | CX |
| 104 | `ER4d` | central_complex | 26 | gaba | cb_intrinsic | CX |
| 105 | `ER4m` | central_complex | 18 | gaba | cb_intrinsic | CX |
| 106 | `ER2_a` | central_complex | 12 | gaba | cb_intrinsic | CX |
| 107 | `PFL3` | central_complex | 24 | acetylcholine | cb_intrinsic | CX |
| 108 | `PFL2` | central_complex | 12 | acetylcholine | cb_intrinsic | CX |
| 109 | `PFL1` | central_complex | 14 | acetylcholine | cb_intrinsic | CX |
| 110 | `hDeltaB` | central_complex | 19 | acetylcholine | cb_intrinsic | CX |
| 111 | `PFNd` | central_complex | 40 | acetylcholine | cb_intrinsic | CX |
| 112 | `PFNv` | central_complex | 20 | acetylcholine | cb_intrinsic | CX |
| 113 | `ExR1` | central_complex | 4 | gaba | cb_intrinsic | CX |
| 114 | `ExR2` | central_complex | 4 | gaba | cb_intrinsic | CX |
| 115 | `FB6A` | central_complex | 10 | glutamate | cb_intrinsic | CX |
| 116 | `FB6H` | central_complex | 10 | glutamate | cb_intrinsic | CX |
| 117 | `FB7A` | central_complex | 10 | glutamate | cb_intrinsic | CX |
| 118 | `FB7B` | central_complex | 10 | glutamate | cb_intrinsic | CX |
| 119 | `LB3b` | sez | 11 | acetylcholine | cb_sensory | gustatory |
| 120 | `LB3c` | sez | 23 | acetylcholine | cb_sensory | gustatory |
| 121 | `PhG1a` | sez | 2 | acetylcholine | cb_sensory | gustatory |
| 122 | `PhG1b` | sez | 2 | acetylcholine | cb_sensory | gustatory |
| 123 | `PhG1c` | sez | 4 | acetylcholine | cb_sensory | gustatory |
| 124 | `LB3a` | sez | 17 | acetylcholine | cb_sensory | gustatory |
| 125 | `LB3d` | sez | 26 | acetylcholine | cb_sensory | gustatory |
| 126 | `LB1a` | sez | 11 | acetylcholine | cb_sensory | gustatory |
| 127 | `LB1b` | sez | 6 | acetylcholine | cb_sensory | gustatory |
| 128 | `LB1c` | sez | 16 | acetylcholine | cb_sensory | gustatory |
| 129 | `LB1d` | sez | 5 | acetylcholine | cb_sensory | gustatory |
| 130 | `LB1e` | sez | 19 | acetylcholine | cb_sensory | gustatory |
| 131 | `GNG215` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 132 | `GNG232` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 133 | `GNG132` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 134 | `GNG089` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 135 | `PRW046` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 136 | `PRW047` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 137 | `GNG042` | sez | 2 | gaba | cb_intrinsic | - |
| 138 | `GNG038` | sez | 2 | gaba | cb_intrinsic | - |
| 139 | `GNG551` | sez | 2 | gaba | cb_intrinsic | - |
| 140 | `GNG108` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 141 | `GNG120` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 142 | `GNG117` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 143 | `GNG234` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 144 | `GNG015` | sez | 2 | gaba | cb_intrinsic | - |
| 145 | `GNG095` | sez | 2 | gaba | cb_intrinsic | - |
| 146 | `GNG130` | sez | 2 | gaba | cb_intrinsic | - |
| 147 | `GNG180` | sez | 2 | gaba | cb_intrinsic | - |
| 148 | `GNG184` | sez | 2 | gaba | cb_intrinsic | - |
| 149 | `GNG016` | sez | 2 | unknown | cb_intrinsic | - |
| 150 | `GNG087` | sez | 3 | glutamate | cb_intrinsic | - |
| 151 | `GNG592` | sez | 2 | glutamate | cb_intrinsic | - |
| 152 | `MN9` | sez | 2 | acetylcholine | cb_motor | - |
| 153 | `MN1` | sez | 2 | acetylcholine | cb_motor | - |
| 154 | `MN6` | sez | 2 | acetylcholine | cb_motor | - |
| 155 | `SAD093` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 156 | `GNG458` | sez | 2 | gaba | cb_intrinsic | - |
| 157 | `GNG532` | sez | 2 | acetylcholine | cb_intrinsic | - |
| 158 | `pC1_1a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 159 | `pC1_2a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 160 | `pC1_4a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 161 | `pC1_7b` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 162 | `pC1_10a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 163 | `pC1_12a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 164 | `pC1_14a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 165 | `pC1_16a` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 166 | `pC1_18b` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 167 | `pC1_19` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 168 | `mAL_m8` | central_other | 16 | gaba | cb_intrinsic | - |
| 169 | `mAL_m1` | central_other | 12 | gaba | cb_intrinsic | - |
| 170 | `aIPg7` | central_other | 6 | acetylcholine | cb_intrinsic | - |
| 171 | `AVLP732m` | central_other | 4 | acetylcholine | cb_intrinsic | - |
| 172 | `AVLP733m` | central_other | 4 | acetylcholine | cb_intrinsic | - |
| 173 | `JO-A1` | central_other | 60 | acetylcholine | cb_sensory | mechanosensory |
| 174 | `JO-A2` | central_other | 40 | acetylcholine | cb_sensory | mechanosensory |
| 175 | `JO-CM` | central_other | 40 | acetylcholine | cb_sensory | mechanosensory |
| 176 | `JO-EV1` | central_other | 30 | acetylcholine | cb_sensory | mechanosensory |
| 177 | `JO-FV` | central_other | 30 | acetylcholine | cb_sensory | mechanosensory |
| 178 | `BM` | central_other | 9 | acetylcholine | cb_sensory | mechanosensory_tactile |
| 179 | `LAL083` | central_other | 4 | glutamate | cb_intrinsic | - |
| 180 | `LAL126` | central_other | 4 | glutamate | cb_intrinsic | - |
| 181 | `LAL179` | central_other | 4 | acetylcholine | cb_intrinsic | - |
| 182 | `PS049` | central_other | 4 | gaba | cb_intrinsic | - |
| 183 | `PS059` | central_other | 4 | gaba | cb_intrinsic | - |
| 184 | `VES051` | central_other | 4 | glutamate | cb_intrinsic | - |
| 185 | `VES052` | central_other | 4 | glutamate | cb_intrinsic | - |
| 186 | `AOTU015` | central_other | 4 | acetylcholine | cb_intrinsic | - |
| 187 | `AOTU019` | central_other | 4 | gaba | cb_intrinsic | - |
| 188 | `PVLP020` | central_other | 4 | gaba | cb_intrinsic | - |
| 189 | `DNp01` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 190 | `DNp02` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 191 | `DNp04` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 192 | `DNp11` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 193 | `DNp03` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 194 | `DNp07` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 195 | `DNp10` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 196 | `DNp09` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 197 | `DNg100` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 198 | `DNge053` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 199 | `DNg97` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 200 | `MDN` | descending_motor | 4 | acetylcholine | descending_neuron | - |
| 201 | `DNg60` | descending_motor | 2 | gaba | descending_neuron | - |
| 202 | `DNg74_a` | descending_motor | 2 | gaba | descending_neuron | - |
| 203 | `DNg74_b` | descending_motor | 2 | gaba | descending_neuron | - |
| 204 | `DNa01` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 205 | `DNa02` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 206 | `DNa03` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 207 | `DNb01` | descending_motor | 2 | glutamate | descending_neuron | - |
| 208 | `DNg13` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 209 | `DNg02_a` | descending_motor | 10 | acetylcholine | descending_neuron | - |
| 210 | `DNg02_b` | descending_motor | 5 | acetylcholine | descending_neuron | - |
| 211 | `DNg02_c` | descending_motor | 4 | acetylcholine | descending_neuron | - |
| 212 | `DNg02_d` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 213 | `DNg02_e` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 214 | `DNg02_f` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 215 | `DNg02_g` | descending_motor | 4 | acetylcholine | descending_neuron | - |
| 216 | `DNg62` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 217 | `DNge078` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 218 | `DNge062` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 219 | `DNge080` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 220 | `DNg67` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 221 | `pIP10` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 222 | `pMP2` | descending_motor | 2 | acetylcholine | descending_neuron | - |
| 223 | `DNge129` | descending_motor | 2 | gaba | descending_neuron | - |
| 224 | `TTMn` | descending_motor | 2 | glutamate | vnc_motor | - |
| 225 | `PSI` | descending_motor | 2 | unknown | vnc_efferent | - |
| 226 | `DLMn a, b` | descending_motor | 2 | glutamate | vnc_motor | - |
| 227 | `DLMn c-f` | descending_motor | 8 | glutamate | vnc_motor | - |
| 228 | `DVMn 1a-c` | descending_motor | 6 | glutamate | vnc_motor | - |
| 229 | `DVMn 2a, b` | descending_motor | 4 | glutamate | vnc_motor | - |
| 230 | `DVMn 3a, b` | descending_motor | 4 | glutamate | vnc_motor | - |
| 231 | `b1 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 232 | `b2 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 233 | `b3 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 234 | `i1 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 235 | `i2 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 236 | `iii1 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 237 | `iii3 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 238 | `hg1 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 239 | `hg2 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 240 | `hg3 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 241 | `hg4 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 242 | `tp1 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 243 | `tp2 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 244 | `tpn MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 245 | `ps1 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 246 | `ps2 MN` | descending_motor | 2 | glutamate | vnc_motor | - |
| 247 | `MNwm35` | descending_motor | 2 | glutamate | vnc_motor | - |
| 248 | `MNwm36` | descending_motor | 2 | glutamate | vnc_motor | - |
| 249 | `Ti flexor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 250 | `Ti extensor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 251 | `Tr flexor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 252 | `Tr extensor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 253 | `Fe reductor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 254 | `Ta depressor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 255 | `Ta levator MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 256 | `Sternotrochanter MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 257 | `Sternal anterior rotator MN` | descending_motor | 12 | glutamate | vnc_motor | - |
| 258 | `Sternal posterior rotator MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 259 | `Pleural remotor/abductor MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 260 | `ltm MN` | descending_motor | 6 | glutamate | vnc_motor | - |
| 261 | `GFC2` | vnc | 10 | acetylcholine | vnc_intrinsic | - |
| 262 | `dPR1` | vnc | 2 | acetylcholine | vnc_intrinsic | - |
| 263 | `dMS2` | vnc | 20 | acetylcholine | vnc_intrinsic | - |
| 264 | `vPR6` | vnc | 8 | acetylcholine | vnc_intrinsic | - |
| 265 | `vPR9_a` | vnc | 3 | gaba | vnc_intrinsic | - |
| 266 | `vPR9_b` | vnc | 3 | gaba | vnc_intrinsic | - |
| 267 | `vPR9_c` | vnc | 3 | gaba | vnc_intrinsic | - |
| 268 | `AN13B002` | vnc | 2 | gaba | ascending_neuron | - |
| 269 | `AN05B023d` | vnc | 2 | gaba | ascending_neuron | - |
| 270 | `AN03A008` | vnc | 2 | acetylcholine | ascending_neuron | - |
| 271 | `AN04B003` | vnc | 2 | acetylcholine | ascending_neuron | - |
| 272 | `AN08B020` | vnc | 2 | acetylcholine | ascending_neuron | - |
| 273 | `AN05B102a` | vnc | 4 | acetylcholine | ascending_neuron | - |
| 274 | `IN13A001` | vnc | 6 | gaba | vnc_intrinsic | - |
| 275 | `IN13A022` | vnc | 6 | gaba | vnc_intrinsic | - |
| 276 | `IN08A002` | vnc | 6 | glutamate | vnc_intrinsic | - |
| 277 | `IN19A016` | vnc | 6 | gaba | vnc_intrinsic | - |
| 278 | `IN07B010` | vnc | 6 | acetylcholine | vnc_intrinsic | - |
| 279 | `IN19B043` | vnc | 6 | acetylcholine | vnc_intrinsic | - |
| 280 | `IN03B015` | vnc | 6 | gaba | vnc_intrinsic | - |
| 281 | `IN12B003` | vnc | 6 | gaba | vnc_intrinsic | - |
| 282 | `IN21A026` | vnc | 6 | glutamate | vnc_intrinsic | - |
| 283 | `LgLG3` | vnc | 162 | acetylcholine | vnc_sensory | gustatory |
| 284 | `LgLG4` | vnc | 43 | acetylcholine | vnc_sensory | gustatory |
| 285 | `WG2` | vnc | 97 | acetylcholine | vnc_sensory | gustatory |
| 286 | `LgAG1` | vnc | 25 | acetylcholine | vnc_sensory | gustatory |
| 287 | `LgLG1a` | vnc | 136 | acetylcholine | vnc_sensory | gustatory |
| 288 | `LgLG1b` | vnc | 134 | acetylcholine | vnc_sensory | gustatory |
| 289 | `LgLG2` | vnc | 130 | acetylcholine | vnc_sensory | gustatory |

Region subtotals of the core: optic_lobe 2,977 (35 rows) · antennal_lobe 396 (18) · mushroom_body 1,160 (45) ·
central_complex 381 (20) · sez 197 (39) · central_other 351 (31) · descending_motor 229 (72) · vnc 844 (29).
Core inhibitory share (GABA+Glu+His by count) = 20.4 %; with the background NT mix the whole graph lands at 31–33 %
(`test_synthetic::test_inhibitory_fraction` asserts 0.28 ≤ f ≤ 0.36).

### g.2 Region plan and group sizes

`scale_populations`: `s = clip((N − 2000)/18000, 0.25, 1.0)`; rows with `n ≥ 40` become `max(4, round(n·s))`; rows with
`n < 40` are never scaled. `region_plan`: `filler_r = floor(REGION_FRAC[r] · (N − core))`, the rounding remainder goes to
`optic_lobe`. Expected values (`test_synthetic::test_region_plan`):

| N | s | core | filler | optic_lobe | antennal_lobe | mushroom_body | central_complex | sez | central_other | descending_motor | vnc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | 0.25 | 2,830 | 1,170 | 859+710 | 156+26 | 541+31 | 285+21 | 197+28 | 246+187 | 229+15 | 317+152 |
| 8,000 | 0.333 | 3,234 | 4,766 | 1,092+2,888 | 180+109 | 609+128 | 295+85 | 197+114 | 257+762 | 229+61 | 375+619 |
| **20,000** | 1.0 | **6,535** | 13,465 | 2,977+8,149 = **11,126** | 396+309 = **705** | 1,160+363 = **1,523** | 381+242 = **623** | 197+323 = **520** | 351+2,154 = **2,505** | 229+175 = **404** | 844+1,750 = **2,594** |
| **166,700** | 1.0 | **6,535** | 160,165 | 2,977+96,904 = **99,881** | 396+3,683 = **4,079** | 1,160+4,324 = **5,484** | 381+2,882 = **3,263** | 197+3,843 = **4,040** | 351+25,626 = **25,977** | 229+2,082 = **2,311** | 844+20,821 = **21,665** |

(Rows are core+filler per region; `N=166,700` is an engine stress size — the core is never
up-sampled, so it is not a fidelity claim.) Functional group sizes at `s = 1` (both sides; sided groups split ⌈n/2⌉/⌊n/2⌋),
`test_synthetic::test_group_sizes` asserts these exactly:

| group | n (N>=20000) | group | n | group | n |
|---|---:|---|---:|---|---:|
| `grn_sugar` | 344 | `grn_sugar_labellar` | 42 | `grn_water` | 17 |
| `grn_salt` | 26 | `grn_bitter` | 82 | `grn_pher` | 400 |
| `jo_aud` | 100 | `jo_groom` | 100 | `bm` | 9 |
| `photoreceptor` | 520 | `lamina` | 380 | `motion_in` | 600 |
| `t4t5` | 480 | `lc_loom` | 311 | `lc4` | 126 |
| `lplc2` | 185 | `lc_loom2` | 355 | `lc_freeze` | 251 |
| `orn` | 320 | `alpn` | 36 | `alln` | 40 |
| `kc` | 800 | `apl` | 2 | `dpm` | 2 |
| `mbon_avoid` | 10 | `mbon_approach` | 14 | `pam` | 316 |
| `ppl1` | 16 | `epg` | 46 | `pen` | 42 |
| `peg` | 18 | `delta7` | 42 | `ring` | 56 |
| `pfl3` | 24 | `pfl2` | 12 | `pfl1` | 14 |
| `hdelta` | 19 | `pfn` | 60 | `exr` | 8 |
| `dfb_sleep` | 40 | `p1` | 60 | `mal` | 28 |
| `aipg` | 6 | `avlp_aud` | 8 | `lal_ps` | 36 |
| `gf` | 2 | `escape_dn` | 6 | `dn_saccade` | 2 |
| `dn_land` | 4 | `dn_freeze` | 2 | `dn_fwd` | 6 |
| `dng100` | 2 | `dn_back` | 4 | `dn_halt` | 6 |
| `steer_a02` | 2 | `steer_a01` | 2 | `steer_a03` | 2 |
| `steer_b01` | 2 | `steer_g13` | 2 | `flight_dn` | 29 |
| `groom_dn` | 4 | `feed_dn` | 6 | `song_dn` | 4 |
| `pip10` | 2 | `escape_vnc` | 14 | `ttmn` | 2 |
| `psi` | 2 | `gfc2` | 10 | `wing_power` | 24 |
| `wing_steer` | 36 | `b1` | 2 | `i1` | 2 |
| `hg1` | 2 | `song_mn` | 8 | `leg_mn` | 78 |
| `feed_mn` | 2 | `feed_mn_other` | 4 | `feed_pre_exc` | 8 |
| `feed_pre_inh` | 10 | `sugar2_exc` | 12 | `sugar2_inh` | 10 |
| `bitter2` | 7 | `song_vnc` | 39 | `dms2` | 20 |
| `an_steer` | 4 | `leg_premotor` | 42 | `sad093` | 2 |
| `gng458` | 2 | `an05b102a` | 4 |  |  |

### g.3 `PROJECTIONS` — the 112 rules

Columns: `k_in` presynaptic partners per target cell (per allowed side); `r_nom` the nominal presynaptic rate used for
calibration; `target mV` the steady-state `g` at `r_nom` (14 mV = feed-forward "fires by construction" [2× the 7 mV
threshold gap], 6–10 mV = strong modulatory, 3–4 mV = weak/tonic); `w_cal` = `calibrated_weight(k_in, r_nom, target)`
(what `calibrated` mode uses); `w_lit` = literature mean synapses per pair (`literature` mode; 0 = none, use `w_cal`).
Inhibitory rows (GABA/Glu/His sources) are the same formula; the sign comes from the source. Rows are grouped by circuit.

| pid | src | dst | k_in | topology | lat | r_nom Hz | target mV | w_cal | w_lit | prov | note |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| P001 | `R1-R6` | `L1` | 6 | retinotopic | ipsi | 30 | 8 | 32 | 27 | V | histaminergic, inhibitory on L1 |
| P002 | `R1-R6` | `L2\|L3` | 6 | retinotopic | ipsi | 30 | 8 | 32 | 27 | V | L3 real mean is 5 |
| P003 | `L1` | `Mi1\|Tm3` | 2 | retinotopic | ipsi | 15 | 8 | 194 | 65 | V | Tm3 real mean is 12 |
| P004 | `L1` | `L5\|C2\|C3` | 2 | retinotopic | ipsi | 15 | 6 | 145 | 45 | V | L1->L5 54.5, C3 44.7, C2 18.4 |
| P005 | `L2\|L3` | `Tm1\|Tm2\|Tm4\|Tm9` | 2 | retinotopic | ipsi | 15 | 8 | 194 | 30 | D | OFF pathway relay |
| P006 | `Mi1\|Tm3` | `T4a\|T4b\|T4c\|T4d` | 7 | retinotopic | ipsi | 15 | 8 | 55 | 10 | V | Mi1->T4a 10.1 (7 Mi1 per T4a) |
| P007 | `Mi4\|Mi9` | `T4a\|T4b\|T4c\|T4d` | 4 | retinotopic | ipsi | 15 | 3 | 36 | 4.5 | V | inhibitory flanks (GABA / Glu) |
| P008 | `Tm1\|Tm2\|Tm4\|Tm9` | `T5a\|T5b\|T5c\|T5d` | 6 | retinotopic | ipsi | 15 | 8 | 65 | 5 | V | Tm9->T5a 4.8 |
| P009 | `T4a\|T4b\|T4c\|T4d` | `LPLC2` | 8 | retinotopic | ipsi | 20 | 6 | 27 | 3.2 | V | 1,483 T4a -> 184 LPLC2 |
| P010 | `T5a\|T5b\|T5c\|T5d` | `LPLC2\|LC4` | 8 | retinotopic | ipsi | 20 | 6 | 27 | 3.5 | V | T5a->LC4 1.5 |
| P011 | `T4a\|T4b\|T4c\|T4d\|T5a\|T5b\|T5c\|T5d` | `LC6\|LPLC1\|LPLC4\|LC9\|LC31a` | 8 | retinotopic | ipsi | 20 | 6 | 27 | 3 | D | auxiliary + freezing LCs |
| P012 | `LC4` | `DNp01` | 63 | all | ipsi | 33 | 7 | 2 | 50.5 | V | every ipsi LC4 contacts the GF; 7 mV = half the 14 mV GF budget |
| P013 | `LPLC2` | `DNp01` | 92 | all | ipsi | 33 | 7 | 2 | 26.3 | V | other half of the GF budget |
| P014 | `LC4` | `DNp04` | 63 | all | ipsi | 33 | 14 | 5 | 92 | V | strongest LC4 target |
| P015 | `LC4` | `DNp02\|DNp11` | 63 | all | ipsi | 33 | 14 | 5 | 32 | V | 33.7 / 30.0 |
| P016 | `LPLC2` | `DNp04` | 92 | all | ipsi | 33 | 6 | 1 | 18.4 | V |  |
| P017 | `DNp11` | `DNp01` | 1 | all | contra | 50 | 6 | 87 | 108 | V | contralateral, 118 syn total |
| P018 | `DNp01` | `GFC2` | 1 | all | ipsi | 50 | 14 | 204 | 12 | V | GFC2 = 10 cells (5 per side) |
| P019 | `GFC2` | `TTMn` | 5 | all | ipsi | 50 | 14 | 41 | 40 | V |  |
| P020 | `DNp01` | `TTMn\|PSI` | 1 | all | ipsi | 50 | 14 | 204 | 25 | V | chemical only (45 / 4); electrical proxies added by patches.py |
| P021 | `PSI` | `DLMn c-f` | 1 | all | contra | 50 | 14 | 204 | 50.8 | V | PSI crosses the midline |
| P022 | `PSI` | `DLMn a, b\|DVMn 3a, b` | 1 | all | both | 50 | 10 | 145 | 17 | V | 21.5 ipsi / 12.8 contra |
| P023 | `LC9\|LC31a` | `DNp09` | 125 | all | ipsi | 40 | 14 | 2 | 17 | V | DNp09's #1/#2 inputs (16.2 / 23.8) |
| P024 | `PVLP020` | `DNp09` | 2 | all | ipsi | 10 | 3 | 109 | 148 | V | GABA brake on DNp09 |
| P025 | `JO-A1\|JO-A2` | `DNp01` | 11 | random | both | 30 | 6 | 13 | 48 | V | auditory input to the GF (531 syn) |
| P026 | `DNp02\|DNp04\|DNp11` | `TTMn` | 3 | all | ipsi | 50 | 14 | 68 | 0 | E | takeoff DNs -> jump MN (no direct chemical path verified) |
| P027 | `IN13A022\|IN21A026` | `TTMn` | 6 | all | ipsi | 5 | 3 | 73 | 100 | V | tonic inhibition of TTMn (987 GABA / 507 Glu) |
| P028 | `DNp07\|DNp10` | `Ti extensor MN\|Tr extensor MN` | 2 | random | ipsi | 40 | 6 | 55 | 0 | E | landing leg extension stand-in |
| P029 | `PFL3` | `DNa02` | 12 | all | contra | 20 | 10 | 30 | 30.7 | V | L->R 380, R->L 356; CONTRALATERAL |
| P030 | `PFL3` | `DNa03\|DNb01` | 12 | all | contra | 20 | 8 | 24 | 25 | V | 19.5 / 31.4, contralateral |
| P031 | `DNa03` | `DNa02` | 1 | all | ipsi | 30 | 6 | 145 | 276 | V |  |
| P032 | `DNa02` | `Sternal anterior rotator MN` | 1 | all | ipsi | 30 | 14 | 339 | 64.7 | V | 776 syn ipsi |
| P033 | `AN03A008\|AN04B003` | `DNa02` | 2 | all | ipsi | 10 | 6 | 218 | 555 | V | ascending steering input |
| P034 | `PS049\|PS059` | `DNa02` | 4 | all | ipsi | 10 | 4 | 73 | 252 | V | GABA |
| P035 | `LAL083\|LAL126\|VES051\|VES052` | `DNa02` | 8 | all | both | 10 | 4 | 36 | 158 | V | Glu; LAL contra, VES ipsi |
| P036 | `LAL179\|AOTU015\|AOTU019` | `DNa02` | 6 | all | both | 10 | 4 | 48 | 145 | V | ACh contra / ACh ipsi / GABA contra |
| P037 | `DNg97\|GNG532\|LAL083` | `DNg13` | 3 | all | both | 20 | 6 | 73 | 125 | V | 236 contra / 313 / 200 |
| P038 | `GNG458\|DNge129` | `DNg100` | 2 | all | ipsi | 5 | 3 | 218 | 988 | V | GABA brake; no verified excitatory brain input to DNg100 |
| P039 | `DNg100` | `IN13A001` | 1 | all | contra | 30 | 6 | 145 | 36 | V |  |
| P040 | `DNg100\|DNge053\|DNg97` | `IN07B010\|IN19B043` | 3 | random | ipsi | 30 | 14 | 113 | 0 | E | excitatory walking premotor pool |
| P041 | `IN07B010\|IN19B043` | `Ti flexor MN\|Ti extensor MN\|Tr flexor MN\|Tr extensor MN\|Fe reductor MN\|Ta depressor MN\|Ta levator MN\|Sternotrochanter MN\|ltm MN` | 6 | random | ipsi | 40 | 14 | 42 | 0 | E | walking premotor -> leg MNs |
| P042 | `IN13A001\|IN03B015\|IN12B003\|IN19A016` | `Ti flexor MN\|Ti extensor MN\|Tr flexor MN\|Tr extensor MN\|Fe reductor MN\|Ta depressor MN\|Ta levator MN\|Sternotrochanter MN\|ltm MN` | 6 | random | ipsi | 10 | 3 | 36 | 0 | E | inhibitory leg premotor |
| P043 | `MDN` | `IN07B010\|IN03B015\|IN12B003` | 2 | random | ipsi | 30 | 10 | 121 | 466 | V | backward walking (476 / 462 / 461) |
| P044 | `DNg60\|DNg74_a\|DNg74_b` | `Sternal posterior rotator MN\|Pleural remotor/abductor MN\|Tr flexor MN\|Sternotrochanter MN` | 3 | all | ipsi | 30 | 8 | 65 | 280 | V | GABAergic halt DNs inhibit leg MNs directly |
| P045 | `GNG232` | `DNg60` | 1 | all | ipsi | 40 | 8 | 145 | 0 | E | feeding halts walking |
| P046 | `MBON12\|MBON13\|MBON14\|MBON18` | `AOTU015\|LAL179` | 4 | random | both | 20 | 8 | 73 | 0 | E | approach valence -> steering excitation |
| P047 | `MBON01\|MBON03\|MBON05\|MBON06` | `DNg100\|DNge053` | 4 | random | both | 20 | 6 | 55 | 0 | E | avoid valence (Glu) suppresses forward DNs |
| P048 | `EPG` | `PFL3\|PFL1\|PFL2` | 2 | wedge | ipsi | 20 | 8 | 145 | 25 | V | 968 ipsi of 1,182 |
| P049 | `Delta7` | `PFL3` | 8 | random | both | 10 | 4 | 36 | 29 | D | Glu; 5,617 total |
| P050 | `PFNd\|PFNv` | `hDeltaB` | 6 | random | both | 15 | 8 | 65 | 15 | D |  |
| P051 | `hDeltaB` | `PFL3\|PFL2\|PFL1` | 4 | random | both | 15 | 6 | 73 | 10 | D |  |
| P052 | `DNg02_a\|DNg02_b\|DNg02_c\|DNg02_d\|DNg02_e\|DNg02_f\|DNg02_g` | `MNwm36` | 14 | all | ipsi | 30 | 14 | 24 | 84 | V | 2,349 pooled |
| P053 | `DNg02_a\|DNg02_b\|DNg02_c\|DNg02_d\|DNg02_e\|DNg02_f\|DNg02_g` | `tp2 MN\|ps1 MN` | 14 | all | ipsi | 30 | 10 | 17 | 38 | V | 1,362 / 759 |
| P054 | `DNg02_a\|DNg02_b\|DNg02_c\|DNg02_d\|DNg02_e\|DNg02_f\|DNg02_g` | `DVMn 1a-c\|DVMn 3a, b` | 14 | all | both | 30 | 8 | 14 | 12 | V | 436 / 326 |
| P055 | `DNg02_a\|DNg02_b\|DNg02_c\|DNg02_d\|DNg02_e\|DNg02_f\|DNg02_g` | `DLMn c-f\|DLMn a, b` | 14 | all | both | 30 | 4 | 7 | 2.8 | V | weak direct power-MN drive |
| P056 | `DNg02_a\|DNg02_b\|DNg02_c\|DNg02_d\|DNg02_e\|DNg02_f\|DNg02_g` | `b1 MN\|i1 MN\|b2 MN\|i2 MN` | 14 | all | ipsi | 30 | 6 | 10 | 0 | E | flight context for steering MNs |
| P057 | `DNp03` | `i1 MN` | 1 | all | contra | 40 | 10 | 182 | 37 | V | 75 syn contra |
| P058 | `DNp03` | `b1 MN\|hg1 MN` | 1 | all | contra | 40 | 8 | 145 | 0 | E | saccade stand-in (DNp03 -> b1 MN absent in v1.0) |
| P059 | `LB3b\|LB3c\|PhG1a\|PhG1b\|PhG1c` | `GNG215\|GNG232\|GNG132\|GNG089` | 21 | all | ipsi | 100 | 14 | 5 | 40 | V | 741 / 495 / 426 / 196 pooled over ~16 pairs |
| P060 | `LB3b\|LB3c\|PhG1a\|PhG1b\|PhG1c` | `GNG038\|GNG042` | 21 | all | ipsi | 100 | 14 | 5 | 54 | V | GABA 2nd order (1,361 / 902) |
| P061 | `LB3b\|LB3c\|PhG1a\|PhG1b\|PhG1c` | `AN13B002\|DNg67\|DNge062` | 21 | all | ipsi | 100 | 8 | 3 | 22 | V | AN13B002 464, DNg67 195; DNge062 hop is [E] |
| P062 | `LgLG3\|LgLG4\|WG2` | `AN13B002\|AN05B023d` | 151 | all | ipsi | 60 | 14 | 1 | 58 | V | 8,770 + 6,936 + 7,964 |
| P063 | `LgLG3\|WG2` | `PRW046\|PRW047` | 130 | all | ipsi | 60 | 10 | 1 | 11 | V | 1,495 / 1,522 |
| P064 | `GNG215\|GNG232` | `GNG108\|DNge080` | 2 | all | ipsi | 60 | 14 | 85 | 435 | V | 588 / 283 / 283 |
| P065 | `GNG089` | `GNG108\|GNG120` | 1 | all | ipsi | 60 | 8 | 97 | 80 | V | 112 / 49 |
| P066 | `GNG215\|GNG232\|PRW046\|PRW047` | `GNG117\|GNG234` | 4 | all | both | 60 | 14 | 42 | 0 | E | closes the loop onto the two remaining verified MN9 exciters |
| P067 | `GNG108\|GNG120\|GNG117\|GNG234\|DNge080\|DNge062` | `MN9` | 6 | all | ipsi | 60 | 14 | 28 | 376 | V | 373 / 359 / 410 / 362 / 197 / 556 |
| P068 | `GNG042` | `GNG015` | 1 | all | ipsi | 60 | 14 | 170 | 595 | V | disinhibition hop 1 |
| P069 | `GNG015\|GNG095\|GNG130\|GNG180\|GNG184` | `MN9` | 5 | all | ipsi | 10 | 3 | 44 | 372 | V | tonic GABA onto MN9 (10 Hz tonic table) |
| P070 | `GNG132` | `GNG130` | 1 | all | ipsi | 60 | 8 | 97 | 368 | V | inhibitory branch |
| P071 | `GNG038` | `GNG095\|GNG180` | 1 | all | ipsi | 60 | 8 | 97 | 0 | E | second disinhibition branch |
| P072 | `LB1a\|LB1b\|LB1c\|LB1d\|LB1e\|LgAG1` | `GNG016` | 41 | all | ipsi | 100 | 14 | 2 | 122 | V | 4,990 syn |
| P073 | `LB1a\|LB1b\|LB1c\|LB1d\|LB1e` | `GNG087\|GNG592` | 28 | all | ipsi | 100 | 14 | 4 | 63 | V | Glu 2nd order (2,148 / 1,365) |
| P074 | `GNG016` | `LB1c\|LB1e` | 1 | all | ipsi | 40 | 3 | 55 | 48 | V | feedback onto bitter GRNs (NT unclear -> +1) |
| P075 | `GNG087\|GNG592` | `GNG108\|GNG120\|GNG117\|GNG234` | 2 | all | both | 40 | 8 | 73 | 0 | E | bitter suppression of feeding premotor (no 2-hop bitter->MN9 path in v1.0) |
| P076 | `GNG016` | `PPL101\|PPL102\|PPL103` | 1 | all | both | 40 | 8 | 145 | 0 | E | bitter -> punishment dopamine |
| P077 | `GNG215\|GNG232\|PRW046\|PRW047` | `PAM01\|PAM02\|PAM03` | 4 | random | both | 60 | 8 | 24 | 0 | E | sugar -> reward dopamine |
| P078 | `LB3a` | `GNG215\|GNG132` | 8 | all | ipsi | 60 | 6 | 9 | 0 | E | water GRNs share the sweet 2nd-order layer (stand-in) |
| P079 | `ORN_*` | `*PN` | 20 | glomerular | ipsi | 20 | 14 | 25 | 92 | V | ORN_DM1 -> DM1_lPN: 73 -> 2 cells, 13,384 syn |
| P080 | `ORN_*` | `lLN1_a\|lLN2T_a` | 20 | random | ipsi | 20 | 8 | 15 | 5 | D |  |
| P081 | `lLN1_a\|lLN2T_a` | `*PN` | 10 | random | ipsi | 10 | 3 | 22 | 10 | D | GABA gain control |
| P082 | `*PN` | `KC*` | 6 | random | both | 20 | 14 | 85 | 21 | V | DM1_lPN -> KC mean 21.1; 6 PNs per KC |
| P083 | `KC*` | `APL` | 800 | all | both | 2 | 14 | 6 | 59 | V | 79,271 / 1,342 KCs |
| P084 | `APL` | `KC*` | 2 | all | both | 20 | 6 | 109 | 59 | V | GABA, all-to-all |
| P085 | `KCg-m\|KCg-d` | `MBON01\|MBON11\|MBON09` | 200 | random | both | 2 | 14 | 25 | 22.6 | V | KCg-m -> MBON01 30,660 |
| P086 | `KCab-c\|KCab-m\|KCab-s\|KCab-p` | `MBON06\|MBON14\|MBON18\|MBON12\|MBON13` | 200 | random | both | 2 | 14 | 25 | 18 | V | KCab -> MBON06 12,634; MBON12/13 hops are D |
| P087 | `KCa'b'-ap1\|KCa'b'-ap2\|KCa'b'-m` | `MBON03\|MBON04\|MBON05` | 100 | random | both | 2 | 14 | 51 | 8 | D |  |
| P088 | `PAM*` | `KC*` | 40 | random | both | 20 | 3 | 3 | 1.3 | V | PAM01 -> KCg-m 20,477 over 15,263 pairs |
| P089 | `PPL10*` | `KC*` | 4 | random | both | 20 | 3 | 27 | 5.4 | V | PPL101 -> KCg-m 7,224 |
| P090 | `FB6A\|FB6H\|FB7A\|FB7B` | `ExR1\|ExR2` | 10 | random | both | 10 | 6 | 44 | 20 | D | dFB sleep -> ExR (Glu) |
| P091 | `ExR1\|ExR2` | `PFL3\|hDeltaB\|PFNd` | 4 | random | both | 10 | 3 | 55 | 0 | E | GABA arousal gate (sleep mode) |
| P092 | `EPG` | `PEN_a(PEN1)\|PEN_b(PEN2)` | 2 | wedge | ipsi | 20 | 14 | 255 | 66 | V | 6,100 syn |
| P093 | `PEN_a(PEN1)` | `EPG` | 2 | ring_shift | ipsi | 20 | 14 | 255 | 156 | V | 14,398 syn; shift +1 on R, -1 on L |
| P094 | `PEN_b(PEN2)` | `EPG` | 2 | ring_shift | ipsi | 20 | 8 | 145 | 60 | D |  |
| P095 | `EPG` | `PEG` | 2 | wedge | ipsi | 20 | 8 | 145 | 40 | D |  |
| P096 | `EPG` | `Delta7` | 46 | all | both | 20 | 14 | 11 | 10 | V | 19,896 syn |
| P097 | `Delta7` | `EPG` | 42 | all | both | 20 | 6 | 5 | 2.2 | V | Glu; 4,294 syn; implementation skips the source's own wedge |
| P098 | `ER4d\|ER4m\|ER2_a` | `EPG` | 28 | all | both | 10 | 6 | 16 | 10 | V | GABA ring input (12,335 for ER4d) |
| P099 | `LgLG1a\|LgLG1b\|LgLG2` | `AN05B102a` | 200 | all | ipsi | 40 | 14 | 1 | 20 | D | pheromone GRN target [?] |
| P100 | `AN05B102a\|AVLP732m\|AVLP733m` | `pC1_*` | 6 | random | both | 30 | 10 | 40 | 0 | E | engineered pheromone/auditory relay onto P1 |
| P101 | `JO-A1\|JO-A2` | `AVLP732m\|AVLP733m` | 20 | random | ipsi | 30 | 14 | 17 | 20 | D | candidate auditory relay |
| P102 | `mAL_m8\|mAL_m1` | `pC1_*` | 14 | all | contra | 10 | 4 | 21 | 8 | V | GABA; top pC1 inputs (2,617 / 2,468) |
| P103 | `pC1_14a\|pC1_7b\|pC1_10a\|aIPg7` | `pIP10` | 6 | random | both | 30 | 14 | 57 | 120 | V | pC1_14a 565 (6 -> 2), aIPg7 1,280 |
| P104 | `pIP10\|pMP2` | `dPR1` | 2 | all | both | 30 | 14 | 170 | 278 | V | 1,112 / 1,750 |
| P105 | `dPR1` | `dMS2` | 1 | all | contra | 30 | 14 | 339 | 41 | V | 1,646 mostly contra |
| P106 | `dMS2` | `hg1 MN` | 10 | all | ipsi | 40 | 14 | 25 | 108 | V | 3,657 syn |
| P107 | `dMS2` | `b1 MN\|hg3 MN\|hg4 MN` | 10 | all | ipsi | 40 | 6 | 11 | 4.6 | V | 155 to b1; hg3/hg4 D |
| P108 | `vPR9_a\|vPR9_c` | `pIP10` | 3 | all | contra | 10 | 3 | 73 | 20 | D | GABA |
| P109 | `AN08B020` | `pC1_*` | 1 | all | both | 20 | 4 | 145 | 23 | V | 1,381 syn ascending |
| P110 | `JO-CM\|JO-EV1\|JO-FV` | `SAD093` | 20 | random | ipsi | 30 | 14 | 17 | 14 | V | 275 syn |
| P111 | `SAD093\|BM` | `DNg62\|DNge078` | 5 | random | both | 30 | 14 | 68 | 100 | V | 712 / 617 |
| P112 | `DNg62\|DNge078` | `Ti flexor MN\|Tr flexor MN` | 2 | random | ipsi | 40 | 8 | 73 | 0 | E | front-leg grooming stand-in |

Provenance count: 79 `V`, 14 `D`, 19 `E`. The `E` rows are exactly the puppeteering the README must list: takeoff DN → TTMn
(P026), landing (P028), walking premotor pools (P040–P042), feeding → halt (P045), valence → steering/walking (P046–P047),
flight steering context (P056), saccade stand-in (P058), GNG117/234 closure (P066), second disinhibition branch (P071),
bitter suppression / bitter → PPL1 (P075–P076), sugar → PAM (P077), water (P078), ExR arousal gate (P091),
pheromone/auditory relay onto pC1 (P100), grooming legs (P112).

### g.4 Background graph (`REGION_ADJACENCY`, degrees, weights, NT)

Every neuron (core and filler) receives background out-edges so the raster is alive and the regions look like a brain:

* Out-degree per neuron `d ~ LogNormal(ln(FLY_MEAN_OUTDEG) − 0.18, 0.6)` (mean = `FLY_MEAN_OUTDEG` = 25), clipped to
  `[3, 200]`; core neurons get `d/2` background edges (their projections already give them fan-out).
* Target choice: with probability 0.85 a random neuron of the **same region**, otherwise a random neuron of a region
  drawn from `REGION_ADJACENCY[region]` (uniform over the listed regions), never itself, duplicates summed:

```python
REGION_ADJACENCY = {
  "optic_lobe":       ("central_other", "descending_motor"),               # VPN highways only
  "antennal_lobe":    ("mushroom_body", "central_other"),
  "mushroom_body":    ("central_other", "central_complex"),
  "central_complex":  ("central_other", "descending_motor"),
  "sez":              ("descending_motor", "central_other"),
  "central_other":    ("central_complex", "descending_motor", "sez", "mushroom_body"),
  "descending_motor": ("vnc",),                                            # brain -> VNC only via DNs
  "vnc":              ("descending_motor", "central_other"),               # VNC -> brain via ANs
}
```

* Weight per background edge `~ Geometric(p = 0.206)` (mean 4.85 synapses = the real MaleCNS mean per connection [D]),
  capped at 40.
* Filler NT: `FILLER_NT` shares (§c.6) rescaled so that GABA+Glu+His sum to `inhib_frac` (0.33 default) and ACh/unknown
  absorb the rest; filler `type = ""`, `superclass` = the region's dominant superclass (`ol_intrinsic`, `cb_intrinsic`,
  `vnc_intrinsic`, …), `cls = ""`, side alternating L/R.
* Subcriticality (why the brain does not run away): at the required 1–5 Hz rest (noise-driven, §c.10) a neuron with
  in-degree 25, mean weight 4.85, net sign `+0.34` receives `g_ss ≈ 25·4.85·0.275·0.34·(0.0025)·5 ≈ 0.14 mV` of recurrent
  drive — 50× below the 7 mV gap; a synchronous volley of all 25 inputs adds 33 mV·0.34 ≈ 11 mV only if they coincide
  within τ_s, which the Poisson background never does. `test_synthetic::test_background_subcritical` asserts the mean
  signed in-degree weight ≤ 60 synapses.

### g.5 Build order, patches, meta, expected size

`build_synthetic(n_neurons, seed, mean_outdeg, weights, inhib_frac)`: (1) `scale_populations` → (2) `region_plan` →
(3) allocate indices (g.0) and fill `types/type_idx/side/nt/sign/region/body_id` → (4) resolve every `Proj` (g.0/g.3) into
COO edges with the topology rules, per-edge weight jitter, laterality → (5) background edges (g.4) → (6) concatenate,
drop self-loops, `csr.build_csr(sum_duplicates=True)` → `pre/post/weight` → (7) `apply_patches` (§c.5; the DNp01 → TTMn/PSI
chemical rows P020 get +300/+200 added, `DNp01 ↔ DNp01` 80 contra and the LC4/LPLC2 → DNp01 +6 proxies) → (8)
`resolve_groups`, `validate`, `meta`. RNG: `SeedSequence(seed).spawn(7)[0]` further split into three streams
(populations/sides, projections, background) so that changing `mean_outdeg` does not alter the core wiring.

Expected size at N = 20,000, `mean_outdeg = 25`: core projection edges ≈ 95 k (dominated by `KC* → APL`/`APL → KC*`
1.6 k, `PN → KC` 4.8 k, `LC → DN` 1.2 k, `T4/T5 → LC` 7.4 k, optic columns ≈ 9 k, `ORN → PN` 0.6 k, background ≈ 460 k)
→ **E ≈ 550 k ± 10 %**, Σ synapses ≈ 2.7 M, build time < 2 s, cache file ≈ 5 MB. `meta` keys are those of §c.2 with
`weights_mode`, `gain_default` (1.0 calibrated / 0.65 literature), `engineered_edges` (the 19 `E` pids),
`projections` (`{pid: {edges, w_mean}}`), `license = "synthetic (no data)"`, `citation = None`,
`note = "synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data"`.

### g.6 Calibration and REQUIRED behaviours

The synthetic default is `weights = "calibrated"` with `gain = 1.0` — pathways fire by construction, no first-start
bisection. `literature` mode (and every real-data source) starts from `gain_default = 0.65` and runs `calibrate_gain`
(§c.13) on first start (cached per `connectome_key`). Both modes must pass the gates of `run_gates`, which are the
product's non-negotiable behaviours (`scripts/selftest.py`, §h.3):

1. **Rest activity 1–5 Hz.** No drives, noise on (`mu 0.5, sigma 3.5`), 1000 ms: mean rate over all neurons in
   `[1.0, 5.0]` Hz (expected ≈ 2.3 Hz, RESEARCH §7), `active_frac_max ≤ 0.02`, no region silent (every region ≥ 0.3 Hz)
   and no region above 12 Hz. `test_synthetic::test_rest_rate`.
2. **Sugar drive raises the feeding chain and MN9.** `grn_sugar_labellar` Poisson 100 Hz, recruit 1.0, 1000 ms: over the last
   500 ms `feed_mn ≥ 20 Hz` (expected 40–60 Hz: labellar GRNs 21/side × w 5 at 100 Hz → GNG215/232/132/089 g_ss 14 mV
   (~50 Hz) → GNG108/DNge080 g_ss 14 mV → MN9 net ≈ 14 − 3 mV ≈ 45 Hz), `sugar2_exc ≥ 20 Hz`, `feed_pre_exc ≥ 15 Hz`,
   and `feed_mn` at least 5× its rest rate. `test_synthetic::test_sugar_to_mn9`.
3. **Escape drive → DNp01/GF burst within 20 ms.** `lc_loom` Poisson 150 Hz, recruit 1.0, both sides, from rest: the first
   DNp01 spike occurs ≤ 20 ms after the drive starts (expected 8–12 ms: LC kick → LC spike at ~3 ms → 2 ms delay →
   155 inputs × w 2 × 0.275 mV at 150 Hz = 12.8 mV/ms into the GF → threshold in ≈ 1–2 ms), followed by `ttmn` and
   `psi` spikes within a further 10 ms (patched electrical proxies) and `wing_power` (DLMn) within 25 ms.
   `test_synthetic::test_loom_to_gf_latency`.
4. **No runaway.** During gates 1–3 `active_frac` never exceeds 0.05 for 10 consecutive steps; after the drives stop the
   mean rate returns below 5 Hz within 500 ms. `test_synthetic::test_no_runaway`.
5. **Steering asymmetry.** `pfl3_L` Poisson 60 Hz for 500 ms → `steer_a02_R − steer_a02_L ≥ 10 Hz` (contralateral PFL3 → DNa02),
   `test_synthetic::test_pfl3_contralateral`.

Gate arithmetic is fixed by the table, so an engineer changing a `target_mv` re-runs `pytest tests/test_synthetic.py`
and `scripts/selftest.py` to prove nothing broke. In `literature` mode the GF budget is 5,601 synapses per side (RESEARCH
§7): a single LC4 spike gives a 13.9 mV·gain kick, so `calibrate_gain` lands around 0.2–0.4 and the looming rates of §f.2
are automatically modest; the gates are the same.

### g.7 Export parity

`export_csv(conn, out_dir)` writes `neurons.csv` (`bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt`
with `instance = f"{type}_{side}"`, `status = "Traced"`) and `connections.csv` (`bodyId_pre,bodyId_post,weight`, unsigned
weights, patches included) so that `load_csv_dir(out_dir, min_weight=1, subset="all")` returns a `Connectome` with identical
`n`, `e`, `sign`, `region`, `side`, groups and CSR (`test_synthetic::test_roundtrip`). `scripts/export_synthetic.py` is the
CLI (§i.5).

---

## h. Tests and scripts

All tests are offline, deterministic (`FLY_SEED`-derived), run with `py -3 -m pytest backend/tests -q` from the repo root
(`backend/pyproject.toml` sets `testpaths = ["tests"]`, `pythonpath = ["."]`), and finish in < 90 s on the target CPU
(`-m "not slow"` < 25 s). Nothing imports `torch`, `anthropic`, `tweepy`, `pyarrow` unless the test is marked `optional`
and the package imports (`pytest.importorskip`). Every test name below is normative: implementers may add tests, not
rename these.

### h.1 `tests/conftest.py` and fixtures

| Fixture | Scope | Contents |
|---|---|---|
| `settings_tmp(tmp_path)` | function | `load_settings(env={"FLY_DATA_DIR": tmp/data, "FLY_OUT_DIR": tmp/out, "FLY_SESSION_LOG": "0", "FLY_REALTIME": "0", "FLY_MARKET": "sim", "FLY_LLM": "dryrun", "FLY_X": "dryrun"}, dotenv=None)` |
| `tiny_connectome` | session | hand-built `Connectome` with `n = 400`: 40 typed neurons (`DNp01`×2, `TTMn`×2, `PSI`×2, `MN9`×2, `GNG215`×2, `GNG108`×2, `LB3b`×6, `LC4`×10, `LPLC2`×10, `DNa02`×2, `PFL3`×2) with correct regions/sides/NT, 360 filler, 4,000 random edges (`weight ~ Geometric(0.25)`, 30 % inhibitory) plus explicit edges `LB3b→GNG215 (w 5)`, `GNG215→GNG108 (w 85)`, `GNG108→MN9 (w 60)`, `LC4/LPLC2→DNp01 (w 4)`, `DNp01→TTMn (w 300)`; `validate()` passes |
| `small_synthetic` | session | `build_synthetic(n_neurons=4000, seed=1)` (core 2,830 + 1,170 filler, < 1 s) |
| `synthetic_20k` | session, `slow` | `build_synthetic(20_000, seed=1337)` cached under `tmp_path_factory` |
| `engine(tiny_connectome)` | function | `LIFEngine(tiny_connectome, seed=0, dt_ms=1.0, noise_sigma=0.0)` (noise off unless the test turns it on) |
| `sim_snapshot(seq)` | helper | builds a `MarketSnapshot` with sane sim values; `dex_pair_json` = the RESEARCH §8 sample as a dict |
| `fixtures/neuprint_small/` | files | `neurons.csv` (40 rows, neuPrint schema incl. `consensusNt` spellings `ACH`, `glut`, `GABA`, `unc`, ``) + `connections.csv` (120 rows incl. 3 duplicate pairs and 2 edges to unknown bodies) |
| `fixtures/codex_small/` | files | `classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz`, `connections.csv.gz` (same 40 neurons in the Codex FAFB schema, 130 rows with per-neuropil duplicates) |

### h.2 Test files and test names (what each asserts)

**`test_csr.py`** — `test_build_csr_matches_bruteforce` (random COO with duplicates vs a dict-of-dicts reference: indptr,
indices sorted within rows, summed data) · `test_build_csr_drops_self_loops` · `test_remap_ids_drops_unknown_bodies`
(keep_mask false exactly for the 2 foreign edges) · `test_transpose_roundtrip` (transpose twice == identity) ·
`test_in_degree` · `test_indptr32_shadow_when_small` (`CSR.indptr32` is int32 and equal to indptr).

**`test_engine.py`** (`LIFEngine`, numpy) — `test_step_constants_table` (`LIFParams().constants(dt)` reproduces the §c.9
table for dt 1.0/0.5/0.1 to 6 dp, `ref_steps`/`delay_steps` exact) · `test_forced_kick_latency` (isolated neuron, one
68.75 mV kick via `inject(indices, rate_hz=…)` forced at step k: first spike at `3.0 ≤ t ≤ 3.0 + dt` ms for dt ∈ {1.0, 0.5, 0.1}) ·
`test_current_from_rate_inversion` (`set_tonic` with `current_from_rate(r)` for r ∈ {20, 50, 100} → measured rate within
±5 % over 2 s at dt 1.0; `rate_from_current(current_from_rate(r)) == r`) · `test_unitary_epsp` (0.275 mV kick → peak v−v_rest
0.0433 ± 0.002 mV at 9–10 ms) · `test_refractory_freezes_v_and_g` (during `ref_steps` steps after a spike `v == v_reset`
and `g` does not decay; incoming input still accumulates and is visible on the first non-refractory step) ·
`test_delay_ring` (spike at step k arrives at the post at step k + delay_steps exactly) · `test_sign_convention`
(GABA pre lowers post `g`, ACh raises it) · `test_noise_rest_rate` (400 isolated neurons, `mu 0.5, sigma 3.5`, 2 s →
mean rate in [1.5, 3.5] Hz; `sigma 0` → 0 spikes) · `test_determinism` (two engines, same seed, 500 steps with drives →
identical spike counts per step and bit-identical `v`) · `test_reseed_changes_noise` · `test_inject_tag_replaces`
(re-injecting the same tag replaces, `clear_injections(tag)` removes, expiry at `until_ms`) · `test_recruit_and_side`
(`recruit=0.5, side=-1` drives exactly ⌈0.5·n_L⌉ left cells; same `episode` → same subset, different episode → different) ·
`test_current_mode_equivalent` (`drive_mode='current'` with `current_from_rate(100)` on a group gives 100 ± 10 Hz) ·
`test_set_gain_rescales` · `test_state_dict_roundtrip` · `test_step_stats_shape` (all §c.10 fields present, `region_counts`
length 8, raster arrays parallel and sorted).

**`test_engine_backends.py`** — `test_numpy_propagator_bruteforce` (random spikes → `out` equals a Python loop) ·
`test_concat_vs_vectorised_gather` (both branches of `NumpyPropagator` identical) · `test_torch_propagator_matches_numpy`
(`optional`; same `out` to 1e-5) · `test_make_propagator_auto` (`auto` → numpy for `tiny`; monkeypatched `e` → torch when
importable) · `test_event_driven_equals_dense` (tiny net, 200 steps: engine spike trains identical to a dense
`W @ spikes` reference implementation written in the test).

**`test_synthetic.py`** — `test_build_deterministic` (two builds same seed → identical arrays; different seed → different) ·
`test_counts_sum_to_n` (`n == 4000/20000`, `region_counts` sums, filler/core split of §g.2 `test_region_plan`) ·
`test_region_plan` (§g.2 table for N ∈ {4000, 8000, 20000, 166700} — the last via `region_plan()` only) ·
`test_group_sizes` (every value of the §g.2 group-size table, `slow` on 20k) · `test_all_groups_resolve` (every
`GROUP_REGEX` key non-empty; every `STAR_TYPES` type present) · `test_region_of_matches_table` (`region_of(superclass,
cls, type)` equals the `Pop.region` column for all 289 rows) · `test_sides_balanced` (|n_L − n_R| ≤ 1 per type; `n == 1` → side 0) ·
`test_projection_rules_resolve` (every `Proj` resolves ≥ 1 source and ≥ 1 target; `meta['projections'][pid]['edges'] > 0`) ·
`test_calibrated_weight_examples` (`calibrated_weight(12,100,14) == 8`, `(4,60,14) == 42`, `(155,33,14) == 2`, `(1,30,14) == 339`) ·
`test_laterality` (PFL3 → DNa02 edges are 100 % contralateral; LC4 → DNp01 100 % ipsilateral; DNp11 → DNp01 contralateral) ·
`test_patches_applied` (`meta['patches_applied']` has 5 entries; DNp01 → TTMn weight ≥ 300; idempotent on a second call) ·
`test_inhibitory_fraction` (0.28 ≤ fraction of neurons with sign −1 ≤ 0.36) · `test_background_subcritical` (§g.4) ·
`test_no_self_loops_no_duplicates` · `test_rest_rate` (§g.6 gate 1, `slow`) · `test_sugar_to_mn9` (gate 2, `slow`) ·
`test_loom_to_gf_latency` (gate 3) · `test_no_runaway` (gate 4, `slow`) · `test_pfl3_contralateral` (gate 5) ·
`test_literature_mode_builds` (`weights='literature'`, `gain_default == 0.65`, `w_lit` rows carry literature means) ·
`test_roundtrip` (`export_csv` → `load_csv_dir(subset='all', min_weight=1)` → identical `n, e, sign, region, side`, groups, CSR).

**`test_groups.py`** — `test_regex_table_compiles` · `test_region_of_cascade` (one case per rule of RESEARCH §6, incl.
`AN13B002 → vnc`, `GNG016 → sez`, `JO-A1 → central_other`, `MN9 → sez`, `TTMn → descending_motor`, null superclass →
central_other) · `test_side_of` (`L/R/M/None`, instance fallback `_L`) · `test_resolve_groups_sided_keys` ·
`test_readout_partition_is_partition` (each neuron in ≤ 1 readout id; `name == name_L + name_R` sizes) ·
`test_validate_groups_raises_listing_missing`.

**`test_loaders.py`** — `test_nt_normalise_aliases` (every `NT_ALIASES` key incl. `''`, `'unc'`, `'GLUT'`, `'Glu'`,
`'His'` → sign) · `test_detect_schema` (three fixture dirs → the right `Schema`; empty dir → ValueError) ·
`test_load_neuprint_small` (40 neurons, duplicate pairs summed, foreign edges dropped, `min_weight=3` pruning, groups
resolve `gf`/`feed_mn`, `meta['license']` CC-BY) · `test_load_codex_small` (same neurons via the Codex schema; per-neuropil
rows summed; `license` CC-BY-NC) · `test_subset_core_keeps_functional` (`n_max=30` keeps every functional neuron and
random-fills deterministically) · `test_load_connectome_dispatch` (`synthetic` → build; `csv` → loader; `neuprint` missing
dir → `FileNotFoundError` whose message contains `fetch_neuprint.py`) · `test_cache_roundtrip` (`store_cached`/`load_cached`
equal; `cache_key` changes with `n_neurons`, `seed`, file mtime).

**`test_encoder.py`** — `test_hill` (`hill(0.25, 0.25, 1.5) == 0.5`, monotone, `hill(0) == 0`) · `test_features_ranges`
(random snapshots/trades for 600 ticks: every `[0,1]` field stays in range, `bp ∈ [-1,1]`, `candle` valid) ·
`test_features_buy_raises_sugar_sell_raises_looming` · `test_loom_episode_and_side` (two sells 100 ms apart → same
episode, one `loom_side ≠ 0`; 3 s gap → new episode; sim side from rng, dex side from `ts` parity) ·
`test_flash_on_whale_sell` (usd ≥ 10 × usd_ref → `flash == 1` for 300 ms then 0) · `test_sustained_loom` ·
`test_sleep_pressure_rises_when_quiet` (activity < 0.35 for 90 s → ≥ 0.95; a trade burst decays it) ·
`test_hunger_and_sweet_gain` · `test_easter_egg` (`buys_m5 == 420`, price `0.000420…`, clock 04:20 via monkeypatched
`time.localtime`; disabled by `FLY_EASTER_EGGS=0`) · `test_encode_drive_table` (for a `Features` with sugar 0.5, looming 0.7
at `loom_t_ms` 150, activity 0.4: the Drive list contains `grn_sugar` at 111 ± 1 Hz recruit 0.7, `lc_loom` ramp at
`220·0.7·0.25 = 38.5` Hz recruit 0.6 side `loom_side`, `dng100` ≥ 7.5 Hz, `epg` with exactly 3 non-zero wedge weights) ·
`test_explore_baseline_zero_disables` · `test_mood_feedback_flag` · `test_sleep_scales_rates` · `test_pokes_become_drives`
(every `POKE_TABLE` stim → its group at `rate × strength`, side honoured, expired pokes ignored) · `test_current_mode`
(every drive has `mode='current'` and `current_mv == current_from_rate(rate_hz)`) · `test_all_drive_groups_exist`
(every `Drive.group` the encoder can emit is a key of `Connectome.groups` of `small_synthetic`).

**`test_decoder.py`** — `test_readouts_from_stats` (mapping of §f.3 incl. `song_mn`, `dnp11_spikes`) · `test_turn_sign`
(`steer_a02_R > _L` → `omega > 0` (clockwise), mirrored → `< 0`; clipping ±1) · `test_walk_speed_from_fwd` (`fwd 30` →
`v_target 120`, `halt 1` → 0, `back 0.8` → −32) · `test_feed_hysteresis` (mn9 30 Hz for 200 ms → `feed`, `feed_start`;
< 15 Hz for 1 s → `feed_stop` with `meal_ms`) · `test_freeze` · `test_groom_min_duration` · `test_jump_on_gf_spike`
(one `gf_spikes_r` → `mode jump`, `jump.side == "R"`, heading_out within `heading + π + π/6 ± π/4`; second GF spike
within 1.5 s ignored; `dnp11_spikes > 0` → forward) · `test_fly_enter_exit` · `test_saccade` · `test_court_and_song`
(`p1 15 Hz` → court; `pip10 20 Hz`, `dms2_r > dms2_l` → `wing_ext == +1`, event `song`) · `test_mode_priority` ·
`test_wander_floor` (fwd 0, 3.1 s of stall → `v_target 40`, one `wander_floor` event; disabled when `explore_baseline == 0`) ·
`test_wander_ou_stats` (10⁴ samples: mean ≈ 0, std ≈ σ·√(τ/2) ± 15 %) · `test_body_bounce` (heading reflected at each of
the four walls, `wall_bump.side` correct, position clamped inside the 8 px margin) · `test_body_wrap` · `test_body_jump_decay`
(`v = 600·e^{−t/0.15}`, `jump_t_ms` counts up then `None`) · `test_leg_phase_advances_one_cycle_per_20px` ·
`test_ink_style_table` (every mood → §f.5 colour/style; candle colours; width 1..4; stamps cadence) · `test_kinematics_to_wire_keys`
(exactly the `tick.fly` keys of §d.2).

**`test_mood.py`** — `test_scores` (E/A/valence/arousal formulas on fixed inputs) · one test per row of the §f.6 table:
`test_escape_on_jump` (immediate from any state, exits after 1.5 s to PANIC/ANXIOUS/CRUISING by anxiety) ·
`test_panic_requires_hold` (anxiety 0.75 for 1.9 s → no transition; 2.0 s → PANIC) · `test_panic_on_gf_burst` ·
`test_courtship_dwell_8s` · `test_euphoria_needs_low_anxiety` · `test_feeding_follows_flag` · `test_anxious_bitter_path` ·
`test_sleep_20s_and_wake` · `test_min_dwell_1500ms` · `test_priority_order` (ESCAPE > PANIC > COURTSHIP > EUPHORIA when
all conditions hold) · `test_timers_reset_when_condition_breaks` · `test_confirmed_once_after_3s` (returned exactly once,
never for ESCAPE) · `test_reason_strings` · `test_to_wire_keys` (§d.2 `mood` keys).

**`test_market.py`** — `test_sim_regime_dwell_means` (10⁴ s of sim: mean dwell of each regime within ±30 % of `dwell_s`)
· `test_sim_buy_probability_per_regime` (PUMP buys share ≈ 0.82 ± 0.05, DUMP ≈ 0.18) · `test_sim_snapshot_windows`
(`buys_m5` equals the trades of the last 300 s, `chg_m5` from the price ring) · `test_sim_rug_drains_liquidity` ·
`test_sim_set_regime_forces_then_resumes` · `test_sim_deterministic` · `test_dex_parse_pair_sample` (RESEARCH §8 JSON →
`price_usd 100.09`, `buys_m5 648`, `chg_m5 0.46`, `liq_usd 25108908.33`, `fdv is None`, `mcap is None`) ·
`test_dex_parse_pair_missing_fields` (no `priceChange`, `liquidity: null`, `priceUsd` absent → zeros/None, no exception) ·
`test_choose_pair_max_liquidity` · `test_surrogate_trades_rate_matched` (buys_m5 300 → ≈ 1 buy/s over 100 s ± 20 %,
`surrogate=True`; a fresh snapshot with +5 buys emits exactly 5 real-delta trades first) · `test_feed_fallback_after_3_failures`
(monkeypatched `fetch_pairs` raising → `mode == "sim(fallback)"`, one `market_source` event, snapshot never None) ·
`test_feed_set_mode` (`"PUMP"` → sim regime forced 60 s; `"dexscreener"` refused without token) · `test_snapshot_to_wire_keys`.

**`test_protocol.py`** — `test_hello_model_validates_example` (the §d.1 JSON, with `pops`/`rows` filled programmatically) ·
`test_tick_model_validates_example` (§d.2 JSON verbatim) · `test_mood_change_tweet_market_event_examples` (§d.3–d.5, d.8, d.9) ·
`test_parse_client_all_types` (every §d.7 frame parses; wrong `stim`, `strength 2`, extra key → `ValueError`) ·
`test_tick_to_json_rounding` (positions 1 dp, rates 2 dp, drives 3 dp; `allow_nan` → `ValueError` on NaN) ·
`test_tick_json_size_budget` (a synthetic tick with 100 raster events < 4 KB; with 2000 events < 24 KB) ·
`test_types_ts_mirror` (parses `frontend/lib/types.ts` with a regex over `interface X { … }` blocks and asserts the
field-name sets of `HelloMsg, TickMsg, SimStats, FlyState, InkStyle, MoodState, MarketSnapshot, TickMarket, Drives, Spikes,
MoodChangeMsg, TweetMsg, MarketMsg, EventMsg, SnapshotRequestMsg, PongMsg, ErrorMsg` equal the pydantic models' fields;
skipped with a warning if `frontend/lib/types.ts` is absent) · `test_rest_models` (`PokeRequest` defaults, `HealthResponse`).

**`test_agent.py`** (all dry-run, no network) — `test_build_brain_summary_shape` (exact §d.6 keys, ≤ 1500 bytes, the 19
`rates_hz` keys) · `test_template_tweet_every_reason` (each `TRIGGER_REASONS` × 6 seeds: ≤ 280 chars, ≥ 1 `VOCABULARY`
token, no URL, ≤ 2 hashtags, deterministic per seed) · `test_validate_tweet` (strips `https://…` and `www.…`, caps at 280 on a
word boundary, keeps 2 of 4 hashtags, drops `$DOGE` keeps `$FLY`, rejects vocabulary-free text) ·
`test_generator_dryrun_never_imports_anthropic` (`sys.modules` unchanged) · `test_generator_anthropic_mocked`
(a fake `anthropic` module injected into `sys.modules`: `Messages.create` receives `model, max_tokens=512, system, messages,
output_config` and **no** `thinking`/`temperature`/`budget_tokens`; JSON text block parsed; `stop_reason == "refusal"` →
`refused=True`; `RateLimitError` → one retry then template; `APIConnectionError` → template; malformed JSON → raw text) ·
`test_xclient_dryrun` (`posted=False, dry_run=True`, log line) · `test_xclient_missing_creds_post_mode_degrades` ·
`test_xclient_mocked_tweepy` (fake `tweepy` module: `create_tweet(text, media_ids)` → id/url; `Forbidden` → `disabled_reason`) ·
`test_agent_state_persistence` (`AgentState.save/load` roundtrip; restart does not reset `fires_today` on the same UTC day) ·
`test_agent_allowed_rules` (cooldown, per-reason cooldown, daily cap, in-flight, disabled; `manual` bypasses a–c) ·
`test_agent_fire_writes_artifacts` (`data/tweets.jsonl` line, `out/tweets/<id>.txt/.json/.png`, `tweet` frame published on
a stub bus, PNG magic bytes) · `test_agent_dedupe` · `test_render_snapshot_png` (valid PNG of 800×500+chrome, decodable by
a minimal zlib reader; trail pixels present) · `test_snapshot_broker_timeout_falls_back` (no client → `"server"`; delivered
bytes → `"browser"`).

**`test_server.py`** (`fastapi.testclient`, `small_synthetic` via monkeypatched `load_connectome`, `FLY_REALTIME=0`) —
`test_health` (keys of §c.28; `ok true`) · `test_state_has_hello_and_tick` · `test_config_redacted` (no `X_`/`ANTHROPIC` keys) ·
`test_groups_endpoint` · `test_neurons_endpoint` (`?group=gf` → 2 rows with `type == "DNp01"`) · `test_poke_rest_and_rate_limit`
(3 pokes in 100 ms → third is 429) · `test_market_mode_rest` · `test_snapshot_rest` · `test_tweet_test_rest` (dry-run
record) · `test_ws_hello_then_tick` (first frame `hello`, a `tick` within 1 s, both validate against the models) ·
`test_ws_client_frames` (`ping`→`pong`, bad frame → `error bad_message`, `poke` → `poke` event frame, `clear` → `clear`
event, `set_market_mode PUMP` → next `market` frame has `regime == "PUMP"`) · `test_ws_snapshot_roundtrip`
(`SnapshotBroker.request` from a thread → client receives `snapshot_request`, answers → `"browser"`) ·
`test_double_start_guard` (second `create_app` in the same process raises / returns the same app) · `test_session_log_written`
(`FLY_SESSION_LOG=1` → header line + one line per tick).

**`test_calibrate.py`** — `test_run_gates_calibrated_synthetic` (`small_synthetic`: `GateReport.ok()`; `rest_rate_hz` in
[1, 5]; `mn9_hz ≥ 20`; `gf_latency_ms ≤ 20`; `runaway False`) · `test_calibrate_gain_bisection_converges` (literature-mode
`small_synthetic`, `iters=6`: returns gain in `[0.05, 1.5]`, writes `<key>.calib.json`, second call reads the cache) ·
`test_gate_report_serialisable`.

**`scripts/replay.py` CI check** (`test_server::test_replay_bit_exact`, `slow`): run 200 ticks with `FLY_SESSION_LOG=1`,
then replay the log with `FLY_REPLAY=<file>` and assert identical `total_spikes` per tick and identical final `fly.x/y`.

### h.3 `scripts/smoke.py` and `scripts/selftest.py`

**`scripts/smoke.py`** (`py -3 scripts/smoke.py [--n 20000] [--seed 1337] [--dt 1.0] [--weights calibrated]`, exit 0/1,
< 30 s): builds (or loads from cache) the synthetic connectome, creates `LIFEngine` with noise on, and runs **2000 steps
of dt** in four phases, printing one table row per phase (`phase | steps | mean Hz | active_frac_max | MN9 Hz | DNp01
spikes | first GF ms | ms/step`):

1. `rest` — steps 0–999, no drives: asserts `1.0 ≤ mean_rate ≤ 5.0` Hz, `active_frac_max ≤ 0.02`, 0 DNp01 spikes.
2. `sugar` — steps 1000–1499, `inject("grn_sugar_labellar", rate_hz=100, duration_ms=500)`: asserts `feed_mn` rate over the
   last 250 ms ≥ 20 Hz and ≥ 5 × the rest value; `sugar2_exc ≥ 20 Hz`.
3. `loom` — steps 1500–1699, `inject("lc_loom", rate_hz=150, duration_ms=200)`: asserts first DNp01 spike ≤ 20 ms after step
   1500 and ≥ 1 `ttmn` spike within 30 ms.
4. `recover` — steps 1700–1999, no drives: asserts mean rate back below 5 Hz over the last 100 ms and no runaway
   (`active_frac ≤ 0.05` at every step of the run).

Also prints `RTF = 2000·dt / wall_ms` and fails if RTF < 1.5 at N = 20k (a performance regression guard; use `--no-rtf-check`
on slow machines). Every failed assertion prints the measured value and the group sizes involved.

**`scripts/selftest.py`** (`py -3 scripts/selftest.py [--n] [--seed] [--source synthetic|csv] [--dir]`, exit 0/1, ≈ 20 s):
loads the connectome exactly as the server would (`load_connectome(settings)`), runs `snn.calibrate.run_gates` and prints a
Win95-style ASCII box with the five gates of §g.6 (`rest 1-5 Hz`, `sugar -> MN9 >= 20 Hz`, `loom -> GF <= 20 ms`,
`no runaway`, `PFL3 contralateral`), each `PASS`/`FAIL` with the number, plus: `RTF >= 2.0` (20k synthetic on the target
CPU; 1.0 is the hard floor), the gain used (`auto` → `1.0` or the calib json), `wander_floor` engagements during a 20-s
headless loop run with the sim market in `CALM` (must be 0 — the explore baseline keeps the fly moving), the mean fly
speed over that run (must be ≥ 15 px/s), and whether the fly visited at least 3 of the 4 canvas quadrants. Also runs a
dry-run tweet through the full agent path (`reason='manual'`) and prints it. Exit code 1 if any gate fails. Meant to be the
first command a new machine runs (§i.6).

### h.4 Other scripts (contracts)

| Script | Usage | Behaviour |
|---|---|---|
| `scripts/dev.ps1` | `powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 [-NoFrontend]` | sets `$env:PYTHONUTF8=1`; starts `py -3 backend\run.py` as a PowerShell job (logs to `out\backend.log`) and `npm run dev` in `frontend\` as a second job; waits for `GET /api/health` to be ok (≤ 60 s), opens `http://localhost:3000`, then tails both logs; `Ctrl+C` stops both jobs. Never uses `--reload`. |
| `scripts/demo_market.py` | `py -3 scripts/demo_market.py [--url http://127.0.0.1:4000] [--loop]` | stage demo: `POST /api/market/mode` `PUMP` (60 s) → `RUG` (40 s) → `CALM` (60 s) → `DEAD` (120 s) → `CHOP` (45 s), printing the mood from `/api/state` every 5 s; expects to observe FEEDING/EUPHORIA, then ESCAPE/PANIC, then CRUISING, then SLEEP; `--loop` repeats. |
| `scripts/replay.py` | `py -3 scripts/replay.py data\sessions\<run_id>.jsonl [--assert] [--out replay.jsonl]` | reads the header's settings (forcing `FLY_REALTIME=0`, `FLY_MARKET=sim`, `FLY_SESSION_LOG=0`), rebuilds the identical connectome (asserts `connectome_key`), runs `SimulationLoop` with `replay=read_session(path)` (market/trades/pokes/commands taken from the log per tick), and with `--assert` compares `total_spikes` per tick and final `fly` pose to the log; prints `REPLAY OK <n ticks>` or the first diverging tick. |
| `scripts/bench.py` | `py -3 scripts/bench.py [--n 20000,40000,80000] [--dt 1.0,0.5] [--active 0.005,0.02]` | for each combination builds the synthetic graph, drives a random subset to hit the target active fraction, runs 500 steps ×3, prints `n, e, dt, active_frac, ms/step (median), RTF, backend`; `--torch` adds the torch backend column when importable. |
| `scripts/export_synthetic.py` | `py -3 scripts/export_synthetic.py --out data\connectome\synthetic_export [--n 20000] [--seed 1337]` | `build_synthetic` → `export_csv` (§g.7) and then `load_csv_dir` on the result, printing `n/e` equality (loader parity proof). |
| `scripts/prepare_malecns.py` | `py -3 scripts/prepare_malecns.py [--out data\connectome\malecns] [--min-weight 1] [--skip-download]` | needs `pyarrow` (prepare-time only): downloads the three GCS feathers of RESEARCH §2.1 (566 MB, anonymous HTTPS, resumable, size-checked) into `data\connectome\malecns\raw\`, then writes `neurons.csv` (neuPrint schema: `bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt,predictedNt`, `Traced` only) and `connections.csv` (`bodyId_pre,bodyId_post,weight`, traced-only weights, `weight ≥ min-weight`) — the files `FLY_CONNECTOME_SOURCE=csv` loads. Prints row counts and the expected `FLY_*` env lines. |
| `scripts/fetch_neuprint.py` | `py -3 scripts/fetch_neuprint.py [--out data\connectome\neuprint] [--n-random 25000] [--min-weight 3] [--chunk 2000] [--token ...]` | token (arg or `NEUPRINT_APPLICATION_CREDENTIALS`) → `neuprint-python` `Client`; no token → stdlib `urllib` POST to `/api/custom/custom` **without** an `Authorization` header (RESEARCH §2.2); Cypher templates of RESEARCH §2.2: functional neurons = types matching the union of `GROUP_REGEX` (expanded to the literal type list via a first query of distinct types) + classes {gustatory, Kenyon_Cell, MBON, DAN, CX, ALPN, ALLN} + superclasses {descending_neuron, cb_motor, vnc_motor, vnc_efferent}, plus `rand() < p` random fill to `--n-random`; edges chunked by ≤ 2000 pre bodies, ≤ 3 concurrent requests, ≤ 8 type-pairs per query; writes the same two CSVs as `prepare_malecns.py`; resumable per chunk. |

Every script prints ASCII only, exits non-zero on failure, and accepts `-h`.

---

## i. Run instructions, pins, environment files, first five minutes

### i.1 Prerequisites (already on this machine, verified in RESEARCH §11)

Windows 11, Python 3.14.3 as `py -3`, Node 24.11 / npm 11.6, the Next 16.3.4 scaffold in `frontend/` with `node_modules`
installed. No accounts, no network, no GPU are needed for the default path. Everything runs from the repo root
`C:\Users\USER\fly` in PowerShell; commands are shown for PowerShell (`$env:X="y"` to set a variable for the session).

### i.2 Backend pins

`backend/requirements.txt` (runtime, all already installed — `pip install` is a no-op check):

```
numpy==2.4.3
fastapi==0.135.2
uvicorn[standard]==0.42.0
websockets==16.0
pydantic==2.12.5
httpx==0.28.1
```

`backend/requirements-optional.txt` (each line enables one env switch; none is imported at runtime unless enabled):

```
anthropic==0.86.0        # FLY_LLM=anthropic   (installed; 1.x also works, see RESEARCH section 9)
tweepy==4.17.0           # FLY_X=post          (installed)
torch==2.11.0            # FLY_BACKEND=torch|auto (installed, CPU)
pyarrow==25.0.1          # scripts/prepare_malecns.py only (installed)
pandas==3.0.5            # scripts/prepare_malecns.py convenience only (installed)
neuprint-python==0.6.3   # scripts/fetch_neuprint.py with a token only (installed)
pytest==9.1.1            # tests (installed)
```

`backend/pyproject.toml`:

```toml
[project]
name = "flybrain"
version = "0.1.0"
description = "SynapseFly / FlyBrain: a spiking MaleCNS-shaped fly brain wired to a token market, painting in a Win95 Paint window"
requires-python = ">=3.12"
dependencies = ["numpy>=2.4,<3", "fastapi>=0.135,<1", "uvicorn[standard]>=0.42,<1", "websockets>=16,<17", "pydantic>=2.12,<3", "httpx>=0.28,<1"]
[project.optional-dependencies]
llm = ["anthropic>=0.86,<2"]
x = ["tweepy>=4.17,<5"]
torch = ["torch>=2.11"]
data = ["pyarrow>=25", "pandas>=3", "neuprint-python>=0.6.3"]
dev = ["pytest>=9.1"]
[project.scripts]
flybrain = "flybrain.__main__:main"
[tool.setuptools]
packages = ["flybrain", "flybrain.connectome", "flybrain.snn", "flybrain.market", "flybrain.agent", "flybrain.server"]
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["slow: > 5 s", "optional: needs an optional package"]
```

Installation is optional: `py -3 backend\run.py` and `py -3 -m pytest backend\tests` work without `pip install -e`
because `run.py` and `pyproject`'s `pythonpath` add `backend/` to `sys.path`. `py -3 -m pip install -e backend` gives the
`flybrain` console script and `py -3 -m flybrain` from any directory.

### i.3 Frontend `package.json` (unchanged scaffold + one script)

```json
{
  "name": "frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": { "dev": "next dev", "build": "next build", "start": "next start", "lint": "eslint", "typecheck": "tsc --noEmit" },
  "dependencies": { "next": "16.3.4", "react": "19.2.8", "react-dom": "19.2.8" },
  "devDependencies": { "@tailwindcss/postcss": "^4", "@types/node": "^20", "@types/react": "^19", "@types/react-dom": "^19",
                       "eslint": "^9", "eslint-config-next": "16.3.4", "tailwindcss": "^4", "typescript": "^5" }
}
```

`frontend/.env.local.example`:

```
NEXT_PUBLIC_WS_URL=ws://localhost:4000/ws
NEXT_PUBLIC_API_URL=http://localhost:4000
```

### i.4 `.env.example` (repo root; copy to `.env`; every value is the offline/dry-run default of section b)

```
# ---- SynapseFly / FlyBrain ----  (all defaults run offline with zero accounts)
# connectome
FLY_CONNECTOME_SOURCE=synthetic        # synthetic | csv | neuprint
FLY_CONNECTOME_DIR=data/connectome/malecns   # for csv: neurons.csv(.gz) + connections.csv(.gz) (neuPrint or Codex schema)
FLY_CONNECTOME_NAME=
FLY_N_NEURONS=20000                    # 4000..200000 (8000 on a weak laptop)
FLY_MEAN_OUTDEG=25
FLY_SYNTH_WEIGHTS=calibrated           # calibrated | literature
FLY_SUBSET=core                        # real data: core | all
FLY_MIN_WEIGHT=3
# engine
FLY_DT_MS=1.0                          # 1.0 | 0.5 | 0.2 | 0.1
FLY_SEED=1337
FLY_BACKEND=numpy                      # numpy | torch | auto
FLY_GAIN=auto                          # auto | float
FLY_NOISE_MU=0.5
FLY_NOISE_SIGMA=3.5
FLY_DRIVE_MODE=poisson                 # poisson | current
FLY_TICK_HZ=20
FLY_REALTIME=1
# market
FLY_MARKET=sim                         # sim | dexscreener
FLY_TOKEN_ADDRESS=                     # e.g. a Solana mint; required for dexscreener
FLY_CHAIN=solana
FLY_DEX_POLL_S=60
FLY_SIM_REGIME_S=90
# agent (LLM)
FLY_LLM=dryrun                         # dryrun | anthropic
ANTHROPIC_API_KEY=
FLY_LLM_MODEL=claude-opus-5
FLY_LLM_JSON=1
FLY_LLM_FALLBACKS=0
FLY_TWEET_LANG=en                      # en | tr
# agent (X)
FLY_X=dryrun                           # dryrun | post
X_API_KEY=
X_API_SECRET=
X_ACCESS_TOKEN=
X_ACCESS_TOKEN_SECRET=
FLY_TWEET_COOLDOWN_S=900
FLY_TWEET_REASON_COOLDOWN_S=2700
FLY_TWEETS_PER_DAY=12
# server / body
FLY_PORT=4000
FLY_HOST=127.0.0.1
FLY_CORS_ORIGINS=http://localhost:3000
FLY_CANVAS_W=800
FLY_CANVAS_H=500
FLY_WALLS=bounce                       # bounce | wrap
FLY_RASTER_PER_REGION=48
FLY_RASTER_CAP=2000
FLY_MOOD_FEEDBACK=1                    # mood -> brain feedback drives (puppeteering; 0 to disable)
FLY_EXPLORE_BASELINE=0.25              # exploration drive floor (0 = pure connectome-driven walking)
FLY_WANDER_SIGMA=0.6
FLY_EASTER_EGGS=1
FLY_SESSION_LOG=1
FLY_REPLAY=
FLY_DATA_DIR=
FLY_OUT_DIR=
FLY_LOG_LEVEL=INFO
# data scripts only
NEUPRINT_APPLICATION_CREDENTIALS=
```

### i.5 Run recipes (PowerShell)

Zero-account default (synthetic 20k brain, simulated market, dry-run LLM and X):

```powershell
cd C:\Users\USER\fly
Copy-Item .env.example .env                       # once
Copy-Item frontend\.env.local.example frontend\.env.local   # once
$env:PYTHONUTF8 = "1"
py -3 backend\run.py                              # -> http://127.0.0.1:4000  (first start builds + caches the connectome, ~2 s; boot < 5 s)
# second terminal:
cd C:\Users\USER\fly\frontend ; npm run dev       # -> http://localhost:3000  (fly moves within 2 s of connecting)
# or both at once:
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```

Checks: `curl http://127.0.0.1:4000/api/health` (expect `"ok":true`, `rtf > 1`), `py -3 scripts\selftest.py`,
`py -3 scripts\smoke.py`, `py -3 -m pytest backend\tests -q`, `cd frontend; npm run typecheck; npm run lint`.
Content: `py -3 scripts\demo_market.py` (PUMP → RUG → CALM → DEAD → CHOP), keyboard `S L B C N F T 1-6 0` in the browser,
`Options ▸ Poke` and `Options ▸ Market regime` menus, `Help ▸ About` for provenance.

Overrides: `py -3 backend\run.py --n 8000 --seed 7 --port 4001 --no-realtime` (CLI flags of §c.28 override env).

Real MaleCNS v1.0 data (CC-BY 4.0; zero account via GCS; ≈ 570 MB download, 2–5 min prepare):

```powershell
py -3 scripts\prepare_malecns.py --out data\connectome\malecns --min-weight 1
$env:FLY_CONNECTOME_SOURCE = "csv" ; $env:FLY_CONNECTOME_DIR = "data\connectome\malecns" ; $env:FLY_SUBSET = "core" ; $env:FLY_N_NEURONS = "20000"
py -3 scripts\selftest.py --source csv --dir data\connectome\malecns     # runs calibrate_gain on first start (cached), prints the gates
py -3 backend\run.py
# full graph (166.7k / ~25.6M edges, memmap, torch auto): $env:FLY_SUBSET="all"; $env:FLY_BACKEND="auto"   (RTF 0.3-0.9x on this CPU; the UI shows "brain 0.5x")
```

neuPrint route (token from https://neuprint.janelia.org/account; anonymous works today):

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "<token>"     # optional
py -3 scripts\fetch_neuprint.py --out data\connectome\neuprint --n-random 25000 --min-weight 3
$env:FLY_CONNECTOME_SOURCE = "neuprint" ; py -3 backend\run.py
```

Real market (DexScreener, no key; edge cache 60 s; falls back to sim after 3 failures and says so in the UI):

```powershell
$env:FLY_MARKET = "dexscreener" ; $env:FLY_CHAIN = "solana" ; $env:FLY_TOKEN_ADDRESS = "<mint or 0x address>" ; py -3 backend\run.py
```

Real Claude (paid; ≈ $0.01 per tweet at claude-opus-5 prices):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..." ; $env:FLY_LLM = "anthropic" ; py -3 backend\run.py
curl -X POST http://127.0.0.1:4000/api/tweet/test          # generates (never posts while FLY_X=dryrun)
```

Real X (pay-per-use; app must be Read+Write with tokens regenerated after enabling write; verify pricing at developer.x.com first):

```powershell
$env:X_API_KEY="..." ; $env:X_API_SECRET="..." ; $env:X_ACCESS_TOKEN="..." ; $env:X_ACCESS_TOKEN_SECRET="..." ; $env:FLY_X = "post"
```

Replay a session bit-exactly: `py -3 scripts\replay.py data\sessions\<run_id>.jsonl --assert`.
Export the synthetic graph as CSV (loader parity): `py -3 scripts\export_synthetic.py --out data\connectome\synthetic_export`.
Benchmarks: `py -3 scripts\bench.py --n 20000,40000,80000 --dt 1.0,0.5`.

Ports: backend 4000 (127.0.0.1 only), frontend 3000. `Ctrl+C` stops uvicorn; the lifespan stops the sim/market threads
within one tick and flushes `data/sessions/<run_id>.jsonl` and `data/agent_state.json`. Logs: console (ASCII) and
`/api/health.log_tail`.

### i.6 First five minutes checklist (new machine, offline)

1. `cd C:\Users\USER\fly; Copy-Item .env.example .env; Copy-Item frontend\.env.local.example frontend\.env.local` — 10 s.
2. `py -3 scripts\selftest.py` — builds the 20k synthetic brain (≈ 2 s), prints the five gates (`rest 1-5 Hz`, `sugar -> MN9`,
   `loom -> GF <= 20 ms`, `no runaway`, `PFL3 contralateral`), `RTF`, the 20-s headless motion check (`wander_floor = 0`,
   mean speed ≥ 15 px/s) and one dry-run tweet. All `PASS` → 30 s.
3. `py -3 backend\run.py` in terminal 1; `curl http://127.0.0.1:4000/api/health` shows `"ok":true` — 10 s.
4. `cd frontend; npm run dev` in terminal 2; open `http://localhost:3000`: the Paint window shows a walking fly leaving a
   black/green/red trail within 2 s, the Oscilloscope shows 8 lanes ticking, Fly Status shows `CRUISING`, the ticker shows
   `FLY` / `SIM` — 40 s.
5. Press `S` (sugar): the fly stops, the proboscis extends, ink turns orange and dotted, `feed_mn` climbs in the status
   table; `FEEDING` appears in the title bar. Press `L` (sell wall): a red-flash ramp on the raster's optic-lobe lane, a
   `DNp01` star row fires, the fly jumps (zigzag trail), `ESCAPE` → `PANIC`/`ANXIOUS`. Press `1`..`6` to force market regimes,
   `T` to see a dry-run tweet in the Notepad window — 60 s.
6. `py -3 scripts\demo_market.py` in terminal 3 and watch `PUMP → RUG → CALM → DEAD` produce `EUPHORIA`, `PANIC`, `CRUISING`,
   `SLEEP` and two dry-run tweets in `out\tweets\` — 5 min.
7. Optional: `py -3 -m pytest backend\tests -q -m "not slow"` (< 25 s) and `cd frontend; npm run typecheck`.

### i.7 Tracked documentation files (E8 owns)

* `README.md` (Turkish): what it is (Meta-Meme / DeSci), 5-minute quick start (= §i.6), architecture diagram (sim thread →
  StateBus → WS → Paint), the env table of section b, data sources and licences (MaleCNS **CC-BY 4.0** with the citation
  string of RESEARCH §1; FlyWire/Codex FAFB **CC-BY-NC 4.0** research-only), the **"sentetik" uyarısı** (the default brain is
  a synthetic structured stand-in, not real connectome data; how to switch to real data), the **puppeteering list**
  (explore baseline, wander, wander floor, mood feedback, EPG feedback, wall bump, the 19 `E` projection rows) with the
  env flags that disable each, cost notes (Claude ≈ $0.01/tweet; X pay-per-use — verify), and the science references of
  RESEARCH §7/§13.
* `docs/NOTICE.md`: data provenance + citations (Berg et al. 2026 Cell doi:10.1016/j.cell.2026.08.015; Shiu et al. 2024
  Nature doi:10.1038/s41586-024-07763-9; Dorkenwald/Schlegel 2024 for FlyWire; MANC Takemura 2024), licence texts, the
  statement that no MaleCNS data is redistributed by this repository (users download it themselves).
* `data/README.md` (tracked; everything else under `data/` is gitignored): the directory map of section a and what is safe
  to delete (`cache/`, `snapshots/`, `sessions/`) versus what carries state (`agent_state.json`, `tweets.jsonl`).
* `.gitignore` additions: `out/`, `backend/**/__pycache__/`.






