# data/ — runtime state and downloaded data (gitignored)

Only `.gitkeep` and this `README.md` are tracked; `.gitignore` excludes everything else under `data/`. Synthetic runs
(the default) need nothing here: the first start creates what it needs. Every path below is relative to
`Settings.data_dir` (default `<repo>/data`, override with `FLY_DATA_DIR`); dry-run tweet artifacts go to
`<repo>/out/tweets/` (`FLY_OUT_DIR`), also gitignored.

## Directory map (SPEC section a)

| Path | What | Safe to delete? |
|---|---|---|
| `cache/<key>.npz` + `cache/<key>.json` | connectome caches (uncompressed npz, loaded with `mmap_mode='r'`; `<key>` = `cache_key(...)` of the source/size/seed/weights/patches, SPEC section c.8) | **yes** — rebuilt on the next start (synthetic: ~2 s for 20k; real data: re-parsed from the CSVs) |
| `cache/<key>.calib.json` | `calibrate_gain` result for literature-weight / real graphs (`FLY_GAIN=auto`) | yes — recomputed on the next start (~20 s) |
| `connectome/malecns/` | `neurons.csv`, `connections.csv` written by `scripts/prepare_malecns.py` (+ `raw/*.feather`, the 566 MB GCS downloads) | yes, but you will download again; keep `raw/` to re-prepare with another `--min-weight` |
| `connectome/neuprint/` | the same two CSVs written by `scripts/fetch_neuprint.py` (resumable per chunk) | yes (re-fetch) |
| `connectome/codex/` | user-supplied FlyWire/Codex CSVs (CC-BY-NC 4.0, research only) | your own copy |
| `sessions/<run_id>.jsonl` | session logs: one header line (settings, connectome key) + one line per tick with every input (market digest, trades, pokes, commands, wall time) — the input of `scripts/replay.py --assert` | yes — only replay needs them; they grow ~1 MB per 10 min at 20 Hz |
| `snapshots/<t_ms>_<mood>.png` | canvas snapshots taken for tweets (browser composite or the server's numpy fallback) | yes |
| `tweets.jsonl` | **every** generated tweet (dry-run included), one JSON record per line — the source of `GET /api/tweets` history | **carries state**: delete only if you want to forget the tweet history |
| `agent_state.json` | agent cooldowns, per-reason cooldowns, `fires_today` (UTC day), last reason, `disabled_reason` — survives restarts so the daily cap cannot be reset by restarting | **carries state**: deleting it resets cooldowns and the daily counter |

## Licences of what may land here

* MaleCNS v1.0 (`connectome/malecns`, `connectome/neuprint`): **CC-BY 4.0**, Janelia FlyEM — Berg et al. (2026) Cell,
  doi:10.1016/j.cell.2026.08.015. Downloaded by you; never redistributed by this repository (see `docs/NOTICE.md`).
* FlyWire/Codex (`connectome/codex`): **CC-BY-NC 4.0**, research only.
* Everything else here is generated locally by the simulation.

## Quick recipes

```powershell
# wipe every cache (safe): 
Remove-Item -Recurse -Force data\cache, data\snapshots, data\sessions -ErrorAction SilentlyContinue
# reset the tweet agent (cooldowns + daily counter) but keep the tweet history:
Remove-Item data\agent_state.json
# replay a logged session bit-exactly:
py -3 scripts\replay.py data\sessions\<run_id>.jsonl --assert
```
