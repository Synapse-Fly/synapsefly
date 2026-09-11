# PUPPETEERING — the engineered (`[E]`) list

Every biological number in this repository carries a provenance tag in its docstring: `[V]` verified against the
dataset or by simulation, `[L]` literature, `[E]` **engineered** — a puppeteering / stand-in choice that exists so the
fly always moves and the demo always looks alive, not because a fly does that.

This file is the audit list of the `[E]` items. It used to be section 8 of a (Turkish) engineering README; that README
was replaced by a short public one, so the disclosure lives here and `docs/NOTICE.md` section 3 points at it.
`backend/tests/test_docs_e8.py` fails if this file and the shipped generator disagree.

## 1. Why this list exists

A connectome-driven demo has an obvious failure mode: the honest simulation sits still, so the author nudges it, and
the nudges quietly become the product. The rule here is that every nudge is named, located in the code, and given an
off switch. With all of them off, the remaining motion is pure connectome-driven output — quieter, and occasionally
motionless.

## 2. The engineered mechanisms

| # | Mechanism | Where | Off switch |
|---|---|---|---|
| 1 | **Exploration baseline** — `30 * explore` Hz drive into the `dng100` group (`explore = FLY_EXPLORE_BASELINE + 0.25 * activity`, >= 7.5 Hz) | `encoder.py` row 17 | `FLY_EXPLORE_BASELINE=0` |
| 2 | **OU wander** — Ornstein-Uhlenbeck noise added to the heading while walking / flying (`sigma`, tau 1 s) | `decoder.py` `Wander` | `FLY_WANDER_SIGMA=0` |
| 3 | **Wander floor** — if speed < 5 px/s for 3 s, `v_target = 40 px/s` (emits the `wander_floor` event that `selftest.py` counts) | `decoder.py` | closes together with `FLY_EXPLORE_BASELINE=0` |
| 4 | **Mood feedback** — EUPHORIA -> PAM 60 Hz + `flight_dn`; PANIC -> PPL1 60 Hz + `dn_freeze` 20 Hz; ANXIOUS -> PPL1 20 Hz; courtship -> P1 80 Hz for 6 s; SLEEP scales every row *except* the mood-feedback rows by 0.3 (EUPHORIA/PANIC rows cannot coexist with SLEEP; the `p1` courtship row is scaled by 0.3 as well); on waking, all rows x 0.5 for 2 s | `encoder.py` row 19 | `FLY_MOOD_FEEDBACK=0` |
| 5 | **EPG proprioceptive feedback** — 16 wedges relative to the current heading; 40 Hz into wedge k, 15 Hz into its neighbours | `encoder.py` row 18 | no switch (a single row inside `encoder.py`) |
| 6 | **Wall bump** — a 40 Hz / 100 ms pulse into `steer_a02_<side>` on a `wall_bump` tick | `server/loop.py` step 6 | `FLY_WALLS=wrap` (no bumps happen) |
| 7 | Other `[E]` encoder rows: water (2), loom tonic/aux/flash (5-7), light flicker (9), odour (10), reward/punishment (11-12), dust (13) — the market-to-sense mapping is itself a design decision | `encoder.py` section f.2 | zero without market input |
| 8 | GF jump refractory 1.5 s; DNp02/04/11 forward-takeoff bias | `decoder.py` | — |
| 9 | Gap-junction patch table (DNp01->TTMn 300, DNp01->PSI 200, DNp01<->DNp01 80, LC4/LPLC2->DNp01 6) | `connectome/patches.py` | audited through `meta.patches_applied` |
| 10 | Tonic current table (lamina / motion_in 7.3 mV ...) | `server/loop.py` startup | — |
| 11 | **GF rest brake** (P113 + P114) — P113: ~32 mV steady-state inhibition of DNp01 from 159 cells (depth; scales with the noise floor); P114: ~8 mV from inhibitory cells the tonic table holds near threshold (`mal`, Mi4/Mi9) — continuity, since on a cold start they are the only population that fires inside the first 10 ms | `connectome/synthetic.py` `REST_BRAKES` | `FLY_NOISE_SIGMA=0` (with no noise, P113 is zero too); audited through `meta['rest_brakes']` |

## 3. The 21 engineered projection rows

The shipped synthetic generator emits **114** projection rules: **79** `V` (verified), **14** `D` (derived) and
**21** `E` (engineered). The audit is a one-liner — it always reads the code, never this table:

```powershell
py -3 -c "import sys; sys.path.insert(0,'backend'); from flybrain.connectome.synthetic import PROJECTIONS; print(len(PROJECTIONS), [p.pid for p in PROJECTIONS if p.provenance=='E'])"
```

The `E` rows: takeoff DN -> TTMn (P026), landing (P028), walking premotor pools (P040-P042), feeding -> stop (P045),
valence -> steering / walking (P046-P047), flight steering context (P056), saccade proxy (P058), GNG117/234 closure
(P066), the second disinhibition branch (P071), pain suppression / pain -> PPL1 (P075-P076), sugar -> PAM (P077),
water (P078), ExR arousal gate (P091), pheromone / auditory relay -> pC1 (P100), grooming legs (P112) and the
**GF rest brake (P113 + P114)**: ring neurons (ER4d/ER4m/ER2_a), PVLP020/AOTU019/PS049/PS059/LAL083/LAL126/VES051/
VES052, VNC inhibitory interneurons (IN13A022/IN21A026/IN08A002), vPR9_a-c, GNG458/DNge129 and dFB
(FB6A/FB6H/FB7A/FB7B) -> DNp01 (P113), and mAL_m8/mAL_m1 + Mi4/Mi9 -> DNp01 (P114, the continuity brake driven by the
tonic table).

Without that brake the giant fiber (DNp01) fires at 10-20 Hz under the mandatory noise with no stimulus at all — and
every GF spike is a jump, while `docs/SPEC.md` section h.3 requires **0** DNp01 spikes at rest. The cells that do the
braking have no input in the table and no encoder drive, which is why the brake scales with the noise floor. None of
these rows is used once real data is loaded (only the gap-junction patch is applied).

## 4. 112 design rules vs 114 shipped rules

`docs/SPEC.md` section 0 and section g.3 count this table as **112** rules / 19 `E`; the shipped generator contains
**114** rules / 21 `E`. The two extra rows are exactly P113 and P114 (the GF rest brake, added after the SPEC was
frozen because the rest-rate gate in section h.3 could not pass without them). This document describes the shipped
code; when the two numbers disagree, the one-liner in section 3 is the answer.
