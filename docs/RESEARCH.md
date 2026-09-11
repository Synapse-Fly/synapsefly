# SynapseFly / FlyBrain — Research Digest (verified 2026-09-10)

> **TR özet.** Bu belge, MaleCNS v1.0 konektomu, Shiu 2024 LIF parametreleri, DexScreener/Anthropic/X API'leri ve
> yerel makinedeki paket durumu için **bugün doğrulanmış** referans bilgileri toplar. Her satırın sağında güven
> seviyesi var: **[V]** = bu oturumda canlı doğrulandı (HTTP isteği, paket introspeksiyonu veya sayısal simülasyon),
> **[D]** = üç teklifin araştırma özetinden alındı, doğrudan doğrulanmadı, **[L]** = literatür bilgisi (DOI Crossref ile
> doğrulandı ama içerik doğrulanmadı), **[?]** = düşük güven / tahmin. SPEC.md bu belgeye dayanır.

Confidence legend used throughout: **[V] verified live in this session** (HTTP call, package introspection, or numeric
simulation on this machine), **[D] taken from the proposals' research digest, not re-verified**, **[L] literature claim
whose DOI resolves on Crossref (content not re-read)**, **[?] low confidence / guess — treat as a TODO**.

---

## 1. The dataset: Janelia FlyEM "MaleCNS v1.0"

| Item | Value | Conf. |
|---|---|---|
| Name / neuPrint dataset id | `male-cns:v1.0` (also `male-cns:v0.9` still served) | [V] |
| Description (neuPrint) | "The complete MaleCNS connectome from the Janelia FlyEM Team Project, the Cambridge Drosophila Connectomics Group, and Google Connectomics. Covers the brain and nerve cord; 167k neurons." | [V] |
| Neurons with `status = 'Traced'` | **165,122** (Orphan 6,464; null 4,214; Anchor 611; Assign 11) | [V] |
| neuPrint `Meta.totalPreCount` / `totalPostCount` | 45,656,140 presynaptic sites / 311,833,243 postsynaptic sites (`postHPThreshold` 0.7) | [V] |
| "~125M synapses" (task statement) | Not reproduced from neuPrint meta; probably the count of connections between traced neurons at min-confidence 0.5 | [?] |
| Edge count of the traced-only weights table (weight>=1) | ~25.6M edges (from proposals' digest) | [D] |
| Paper | Berg S., Beckett I.R., Costa M., Schlegel P., Januszewski M., Marin E.C., Nern A., … *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, **Cell 2026**, doi:10.1016/j.cell.2026.08.015 (DOI resolves to `S0092-8674(26)00942-6`) | [V] |
| Preprint | bioRxiv 10.1101/2025.10.09.680999 (v2, 2025-10-30) | [V] |
| License | **CC-BY** (project page and download page both state "The FlyEM Male CNS dataset is licensed under CC-BY") | [V] |
| Explorer | https://male-cns.janelia.org/ (HTTP 200) | [V] |
| Project page | https://www.janelia.org/project-team/flyem/male-cns-connectome (HTTP 200) | [V] |
| Google Research blog | https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/ (HTTP 200) | [V] |
| Download docs | https://janelia-flyem.github.io/male-cns/download/ | [V] |
| Cell Type Explorer | https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/ | [V] (link only) |
| neuPrint UI | https://neuprint.janelia.org/?dataset=male-cns%3Av1.0&qt=findneurons | [V] |
| R / navis | `natverse/malecns`, `navis` + `flybrains` (transforms between MaleCNS and other templates) | [V] (links) |

Citation string to put in `hello.connectome.citation`, README and `data/NOTICE.md`:
`Berg et al. (2026) Sexual dimorphism in the complete Drosophila male central nervous system connectome. Cell. doi:10.1016/j.cell.2026.08.015 (data: FlyEM MaleCNS v1.0, CC-BY 4.0)`.

---

## 2. Data access — three real routes, all verified

### 2.1 Public GCS flat files (zero account, anonymous HTTPS) [V]

Bucket `gs://flyem-male-cns` (public). Directory prefix URLs 404 (expected for GCS); individual objects and the JSON
listing API work anonymously:

```
GET https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o?prefix=v1.0/connectome-data/&fields=items(name,size,updated)
```

Objects under `v1.0/connectome-data/flat-connectome/` (verified listing, sizes in MB, updated 2026-06-03/08):

| File | Size | Columns (verified by reading the Arrow IPC header with an HTTP `Range: bytes=0-262143` request) |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.5 | `bodyId:int64, type, instance, somaSide, superclass, class, subclass, status, statusLabel(dict<int8>), flywireType, hemibrainType, mancType, mancBodyid, group, itoleeHl, trumanHl, supertype, birthtime, synonyms, rootSide, somaNeuromere, dimorphism, matchingNotes, entryNerve, exitNerve, receptorType, fruDsx, mancSerial, mcnsSerial, serialMotif, assignedOlHex1/2:double, somaLocation:list<int64>, tosomaLocation:list<int64>, vfbId` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.3 | `body:int64, cell_type, total_nt_predictions:int32, predicted_nt_confidence:double, predicted_nt, ground_truth, celltype_total_nt_predictions:int32, celltype_predicted_nt, celltype_predicted_nt_confidence:double, consensus_nt` |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 778.1 | per-segment synapse counts (not needed) |
| `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | **508.0** | `body_pre:int64, body_post:int64, weight:int64, type_pre, type_post` |
| `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` | 502.2 | same 5 columns |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1051.2 | `body_pre, body_post, weight` (all segments) |
| `syn-partners-…-traced-only.feather` / `…-significant-only` / full | 2965 / 2966 / 6777 | per-synapse partner table (not needed) |
| `syn-points-…feather` | 13061 | per-synapse points (not needed) |
| `tbar-neurotransmitters-male-cns-v1.0.feather` | 2651.7 | per-T-bar NT probabilities (not needed) |

Base URL for direct downloads: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<file>`.
The three files the project needs total **566 MB** (annotations 14.5 + NT 43.3 + traced-only weights 508.0).
Reading requires `pyarrow` (`pyarrow.feather.read_table`), which is installed on this machine (25.0.1, cp314) but is a
*prepare-time* dependency only — the runtime must never import it. Other v1.0 prefixes (verified): `database/`
(neo4j dump + neuprint-inputs CSVs per the download page), `segmentation/`, `nblasts/`, `supervoxels/`, …

### 2.2 neuPrint HTTP API — anonymous reads work today [V]

* `GET https://neuprint.janelia.org/api/dbmeta/datasets` **without any Authorization header** → 200, lists
  `male-cns:v1.0` (and `male-cns:v0.9`, `manc:v1.2.3`, `optic-lobe:v1.1`, `hemibrain:v1.2.1`, …).
* `POST https://neuprint.janelia.org/api/custom/custom` with body `{"cypher": "...", "dataset": "male-cns:v1.0"}` and
  header `Content-Type: application/json`, **no Authorization header** → 200 with `{"columns": [...], "data": [[...]]}`.
  Every count and weight in sections 4–5 below was obtained this way.
* An **empty or bogus bearer token is rejected**: `Authorization: Bearer <anything invalid>` → 401
  `{"message":"invalid or expired token — neuPrint has moved to a new authorization system; log in at https://neuprint.janelia.org/account to obtain a new token"}`.
  Consequence: anonymous mode must *omit the header*, not send a dummy one.
* `neuprint-python` 0.6.3 (installed) **cannot be used anonymously**: `Client(server, dataset)` without a token raises
  `RuntimeError: No token provided. Please provide one or set NEUPRINT_APPLICATION_CREDENTIALS`, and a dummy token
  produces HTTP 401. So `scripts/fetch_neuprint.py` uses stdlib `urllib` for the anonymous path and `neuprint-python`
  only when `NEUPRINT_APPLICATION_CREDENTIALS` is set (token from https://neuprint.janelia.org/account).
* Anonymous access may be withdrawn at any time; treat it as a convenience, GCS flat files as the durable route. [?]
* Server behaviour: indexed matches `MATCH (n:Neuron {type: 'LC4'})` return in <1 s; a `UNWIND` of 70 type pairs with
  un-indexed `WHERE a.type = p[0]` timed out (HTTP 504 after ~180 s). Keep ≤ 8 pairs per request, use property-map
  matching, ≤ 3 concurrent requests, and avoid joining the 3,377 `R1-R6` × 1,776 `L1` neurons in one query
  (use `WITH a LIMIT 150` sampling).
* Node (`:Neuron`) properties [V]: `bodyId, type, instance, somaSide ('L'|'R'|'M'?), superclass, class, subclass, status,
  statusLabel, consensusNt, predictedNt, predictedNtConfidence, totalNtPredictions, celltypePredictedNt,
  celltypeTotalNtPredictions, group, flywireType, hemibrainType, mancType, mancBodyid, mancGroup, itoleeHl,
  birthtime, synonyms, somaLocation, vfbId, pre, post, upstream, downstream, synweight`.
* Edge (`:ConnectsTo`) properties [V]: `weight` (use this), `weightHP` (high-precision subset), `weightHR`, `roiInfo`.
* `instance` format [V]: `"<type>_<side>"`, e.g. `DNp01(GF)_R`, `DNa02_L`, `MN9_R`, `TTMn_L`.
* Verified body IDs (handy for tests against real data) [V]: DNp01 R=10001 L=10010; DNa02 R=10360 L=523769;
  DNg100 L=10045 R=10056; DNp09 L=10783 R=11177; MN9 L=10331 R=16949; TTMn L=804642 R=800146.

Cypher templates used (copy into `scripts/fetch_neuprint.py`):

```cypher
// neuron table for a set of types (indexed)
MATCH (n:Neuron) WHERE n.status = 'Traced' AND n.type IN $types
RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.superclass AS superclass, n.class AS class,
       n.somaSide AS somaSide, n.consensusNt AS consensusNt, n.predictedNt AS predictedNt
// random fill of traced neurons (p = fraction)
MATCH (n:Neuron) WHERE n.status = 'Traced' AND rand() < $p RETURN n.bodyId, n.type, n.superclass, n.class, n.somaSide, n.consensusNt
// edges among a body-id set, chunked by pre (<= 2000 ids per chunk)
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN $chunk AND b.bodyId IN $all AND c.weight >= $minw
RETURN a.bodyId AS bodyId_pre, b.bodyId AS bodyId_post, c.weight AS weight
```

### 2.3 neo4j dump / neuPrint input CSVs [D]

Listed on the download page: `gs://flyem-male-cns/v1.0/database/neo4j` (neo4j 4.4.16 dump built with
`flyem-snapshot`) and `gs://flyem-male-cns/v1.0/database/neuprint-inputs` (the CSVs used to build it). Prefix verified to
exist; individual file names not verified. A "neuPrint export" CSV in this project means the two-file schema
`neurons.csv (bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt[,predictedNt])` +
`connections.csv (bodyId_pre,bodyId_post,weight)` that our own fetch script writes.

### 2.4 FlyWire / Codex (female FAFB) CSVs [D]

`classification.csv.gz (root_id,flow,super_class,class,sub_class,hemilineage,side,nerve)`,
`consolidated_cell_types.csv.gz (root_id,primary_type,…)`, `neurons.csv.gz (root_id,…,nt_type,…)`,
`connections.csv.gz (pre_root_id,post_root_id,neuropil,syn_count,nt_type)`. Licence **CC-BY-NC 4.0** — research/schema
parity only; not for a commercial token context. Codex also mirrors MaleCNS with `Root ID`/`Super Class`/… headers [?].

---

## 3. Verified population statistics (Traced neurons, `male-cns:v1.0`) [V]

### 3.1 Superclass (basis of the synthetic generator's region fractions)

| superclass | n | share | → region |
|---|---:|---:|---|
| ol_intrinsic | 89,390 | 54.1% | optic_lobe |
| cb_intrinsic | 32,160 | 19.5% | central_other (minus KC/MBON/DAN/CX/GNG/PRW which go to MB/CX/SEZ) |
| vnc_intrinsic | 13,151 | 8.0% | vnc |
| visual_projection | 9,201 | 5.6% | optic_lobe |
| vnc_sensory | 6,365 | 3.9% | vnc |
| cb_sensory | 4,868 | 2.9% | sez (gustatory), central_other (JO/mechanosensory), antennal_lobe (olfactory) |
| ol_sensory | 4,114 | 2.5% | optic_lobe |
| ascending_neuron | 1,846 | 1.1% | vnc |
| descending_neuron | 1,314 | 0.8% | descending_motor |
| vnc_motor | 708 | 0.4% | descending_motor |
| visual_centrifugal | 563 | 0.3% | optic_lobe |
| sensory_ascending | 537 | 0.3% | vnc |
| (null) | 516 | 0.3% | central_other |
| cb_motor | 107 | 0.06% | sez |
| vnc_efferent | 94 | 0.06% | descending_motor |
| cb_endocrine 72, ENS 47, vnc_endocrine 22, sensory_descending 12, efferent_ascending 8, others ≤ 5 | | | misc |

Resulting region fractions used by the generator (sum = 1.000): optic_lobe 0.605, central_other 0.160, vnc 0.130,
mushroom_body 0.027, antennal_lobe 0.023, sez 0.024, central_complex 0.018, descending_motor 0.013.

### 3.2 Class (Traced)

visual 4,107 · Kenyon_Cell **4,064** · CX 2,950 · olfactory 2,639 · mechanosensory_tactile 2,558 · mechanosensory 1,733 ·
unknown_sensory 1,707 · mechanosensory_proprioceptive 1,454 · gustatory **1,428** · ALPN 686 · ALLN 420 · DAN 340 ·
ol_bilateral 116 · MBON 97 · hygrosensory 66 · chemosensory 58 · SEZPN 27 · thermosensory 25 · ALIN 24 · ALON 14.

### 3.3 `consensusNt` (Traced) — sign convention input

acetylcholine 103,718 (62.8%) · glutamate 29,296 (17.7%) · gaba 22,055 (13.4%) · histamine 5,910 (3.6%) · unclear 3,100 ·
null 502 · dopamine 392 · octopamine 101 · serotonin 48. **Inhibitory share (GABA+Glu+His) = 34.7%** — the "~30%
inhibitory" requirement for the synthetic graph is therefore 0.30–0.35.

---

## 4. Cell-type table: group → regex → region → verified count → role

Regex is Python `re.fullmatch` on the MaleCNS `type` string. Counts are **both sides combined** from neuPrint [V] unless
marked. "Sided" groups also get `_L`/`_R` entries (by `somaSide`).

| group | regex (fullmatch) | region | count [V] | role in FlyBrain | conf. |
|---|---|---|---:|---|---|
| `grn_sugar` | `(LB3b\|LB3c\|PhG1[abc]\|LgLG3\|LgLG4\|WG2)` | sez (LB/PhG) / vnc (LgLG, WG) | LB3b 11, LB3c 23, PhG1a 2, PhG1b 2, PhG1c 4, LgLG3 162, LgLG4 43, WG2 97 = **344** | buys / price-up → Poisson drive | identity of LB3b/c as sweet: Engert 2022, Shiu 2022 [L]; PhG1 sweet [D] |
| `grn_sugar_labellar` | `(LB3b\|LB3c\|PhG1[abc])` | sez | 42 | drives the verified SEZ feeding chain | [V] counts |
| `grn_water` | `LB3a` | sez | 17 | liquidity adds | [D] |
| `grn_salt` | `LB3d` | sez | 26 | unused (reserved) | [D] |
| `grn_bitter` | `(LB1[a-e]\|LgAG1)` | sez / vnc | LB1a 11, b 6, c 16, d 5, e 19, LgAG1 25 = **82** | sells / rug → bitter | [D] |
| `grn_pher` | `(LgLG1a\|LgLG1b\|LgLG2)` | vnc | 136 / 134 / 130 | courtship stimulus (pheromone) | **[?]** identity unverified — outputs go to AN05B102a/c, IN05B002, AN05B023b/c, not to pC1 |
| `jo_aud` | `JO-A.*` | central_other | JO-A1, JO-A2, JO-A-unclear (34 JO types, 672 cells total) | song/auditory → GF (JO-A/B → DNp01 531 syn [V]) and pC1 | [V] types |
| `jo_groom` | `JO-(C\|E\|F).*` | central_other | JO-CM, JO-EV1, JO-EV2, JO-FV, JO-ED2_a… | dust → grooming: JO-C/E/F → SAD093 → DNg62/DNge078 [V] | [V] |
| `bm` | `BM` | central_other | 9 | bristle mechanosensory → aDN (617 syn [V]) | [V] |
| `photoreceptor` | `(R1-R6\|R[78][dpy])` | optic_lobe | R1-R6 **3,377** (histamine), R7/R8 subtypes 2,713 | light / flicker input; **inhibitory (histaminergic)** | [V] |
| `lamina` | `L[1-5]` | optic_lobe | L1 1,776 (Glu), L2 1,779 (ACh), L3–L5 | sign-inverting relay | [V] |
| `motion_in` | `(Mi1\|Tm3\|Mi4\|Mi9\|Tm1\|Tm2\|Tm4\|Tm9)` | optic_lobe | Mi1 1,773, Tm3 2,054, … | T4/T5 inputs | [V] |
| `t4t5` | `T[45][a-d]` | optic_lobe | 8 types, 13,580 cells | motion → LPLC2 | [V] |
| `lc_loom` | `(LC4\|LPLC2)` | optic_lobe | LC4 **126**, LPLC2 **185** | looming → GF | [V] |
| `lc_loom2` | `(LC6\|LPLC1\|LPLC4)` | optic_lobe | 124 / 134 / 97 | auxiliary looming | [V] |
| `lc_freeze` | `(LC9\|LC31a)` | optic_lobe | 219 / 32 | slow looming → DNp09 freezing (top two DNp09 inputs [V]) | [V] |
| `orn` | `ORN_.*` | antennal_lobe | 53 glomerular types, 2,635 cells | volume → odour | [V] |
| `alpn` | `[A-Za-z0-9+]+_(l\|ad\|il\|lv\|v)?PN` | antennal_lobe | 76 types, 305 cells (note `VP2+Z_lvPN` contains `+`) | | [V] |
| `alln` | class `ALLN` (regex fallback `(lLN\|il3LN\|v2LN\|vLN\|lvLN).*`) | antennal_lobe | 420 | | [V] class |
| `kc` | `KC.*` | mushroom_body | 15 types, **4,064** (KCg-m 1,342, KCab-c 488, …) | | [V] |
| `apl` / `dpm` | `APL` / `DPM` | mushroom_body | 2 / 2 (APL gaba; DPM consensusNt = dopamine in v1.0) | | [V] |
| `mbon_avoid` | `MBON0[1-6]` | mushroom_body | MBON01 2 (Glu), 03 2 (Glu), … | negative valence | Aso 2014 [L] |
| `mbon_approach` | `MBON(09\|11\|12\|13\|14)` | mushroom_body | MBON11 2 (GABA), MBON12 4 (ACh) | positive valence | Aso 2014 [L] |
| `pam` / `ppl1` | `PAM[0-9]{2}` / `PPL10[1-8]` | mushroom_body | 15 PAM types 316 (PAM01 44) / 8 PPL1 types 16 | reward / punishment dopamine | [V] |
| `epg`, `pen`, `peg`, `delta7`, `ring` | `EPG`, `PEN_[ab][(]PEN[12][)]`, `PEG`, `Delta7`, `ER[1-6].*` | central_complex | EPG 46, PEN_a(PEN1) 20, PEN_b(PEN2) 22, PEG 18, Delta7 42 (Glu), ER4d 26 (GABA), 26 ER types 282 | heading compass | [V] |
| `pfl3`, `pfl2`, `pfl1`, `hdelta`, `pfn`, `exr`, `dfb_sleep` | `PFL3`, `PFL2`, `PFL1`, `hDelta[A-M]`, `PFN.*`, `ExR[1-8]`, `FB[67][A-Z].*` | central_complex | 24 / 12 / 14 / 189 (13) / 456 (10) / 26 (8) / 140 (43 types) | goal steering; sleep (dFB) | [V] counts; dFB role Donlea 2014 [L] |
| `p1` | `pC1.*` | central_other | 49 types, 156 cells (pC1_14a → pIP10 [V]) | courtship | [V] |
| `mal` | `mAL_.*` | central_other | mAL_m8 16, mAL_m1 12 (GABA; top pC1 inputs) | | [V] |
| `gf` (sided) | `DNp01` | descending_motor | 2 | giant fiber → escape jump | [V] |
| `escape_dn` | `DNp(02\|04\|11)` | descending_motor | 2 each | takeoff bias | [V] |
| `dn_saccade` / `dn_land` | `DNp03` / `DNp(07\|10)` | descending_motor | 2 / 2+2 | flight saccade (DNp03 → i1 MN 75 contra [V]) / landing | [V] counts |
| `dn_freeze` | `DNp09` | descending_motor | 2 | freezing (Zacarias 2018) / walking (Bidaye 2020) | [L] |
| `dn_fwd` | `(DNg100\|DNge053\|DNg97)` | descending_motor | 2 / 2 / 2 | forward walking readout | [D] roles |
| `dn_back` | `MDN` | descending_motor | 4 | backward walking (Bidaye 2014) | [L] |
| `dn_halt` | `(DNg60\|DNg74_[ab])` | descending_motor | 2 / 2 / 2 — **all GABAergic** [V]; outputs to leg MNs directly | halt | [V] |
| `steer_a02` (sided) | `DNa02` | descending_motor | 2 (ACh) | yaw (Yang 2024) | [V]/[L] |
| `steer_a01`, `steer_a03`, `steer_g13` (sided) | `DNa01`, `DNa03`, `DNg13` | descending_motor | 2 each (ACh) | yaw | [V] |
| `steer_b01` | `DNb01` | descending_motor | 2 — **glutamatergic** [V] | PFL3 target (753 contra) | [V] |
| `flight_dn` | `DNg02_[a-g]` | descending_motor | **29** (a 10, b 5, c–g 2–4) | wingbeat power (Namiki 2022) | [V] |
| `groom_dn` | `(DNg62\|DNge078)` | descending_motor | 2 / 2 (aDN1/aDN2 stand-ins; MaleCNS has no `aDN*` type) | grooming | [V] existence, [D] identity |
| `feed_dn` | `(DNge062\|DNge080\|DNg67)` | descending_motor | 2 each; DNge062 is MN9's top input (556) | feeding | [V] |
| `song_dn` | `(pIP10\|pMP2)` | descending_motor | 2 / 2 | song command | [V] |
| `escape_vnc` | `(TTMn\|PSI\|GFC2)` | descending_motor | 2 (Glu) / 2 (`vnc_efferent`, NT unclear) / **10** (`vnc_intrinsic`) | jump / flight init | [V] |
| `wing_power` | `(DLMn a, b\|DLMn c-f\|DVMn 1a-c\|DVMn 2a, b\|DVMn 3a, b)` | descending_motor | 2 / 8 / 6 / 4 / 4 | wingbeat amplitude readout | [V] |
| `wing_steer` | `((b[123]\|i[12]\|iii[134]\|hg[1-4]\|tp[12]\|tpn\|ps[12]) MN\|MNwm3[56])` | descending_motor | 16 matched types, 32 cells; DNg02 → MNwm36 2,349, tp2 MN 1,362, ps1 MN 759 [V] | flight steering | [V] |
| `song_mn` | `(hg1\|hg3\|hg4\|b1) MN` | descending_motor | 2 each; dMS2 → hg1 MN 3,657 [V] | wing-extension song | [V] |
| `leg_mn` | `(Ti flexor\|Acc[.] ti flexor\|Ti extensor\|Tr flexor\|Acc[.] tr flexor\|Tr extensor\|Fe reductor\|Ta depressor\|Ta levator\|Sternotrochanter\|Sternal anterior rotator\|Sternal posterior rotator\|Tergotr[.]\|Pleural remotor/abductor\|ltm\|ltm1-tibia\|ltm2-femur) MN` | descending_motor | 232 for the flexor/extensor/… subset; `Sternal anterior rotator MN` 12 (DNa02 → 776 syn ipsi) | leg readout | [V] |
| `feed_mn` | `MN9` | sez | 2 (`cb_motor`) | proboscis extension | [V] |
| `feed_pre_exc` | `(GNG108\|GNG120\|GNG117\|GNG234)` | sez | 2 each (ACh; MN9 inputs 380/442/410/362) | | [V] |
| `feed_pre_inh` | `(GNG015\|GNG095\|GNG130\|GNG180\|GNG184)` | sez | 2 each (GABA; MN9 inputs 478/432/407/287/272) | | [V] |
| `sugar2_exc` | `(GNG215\|GNG232\|GNG132\|GNG089\|PRW046\|PRW047)` | sez | 2 each (ACh; direct LB3b/c targets) | 2nd-order sugar | [V] |
| `sugar2_inh` | `(GNG042\|GNG038\|AN13B002\|AN05B023d\|GNG551)` | sez / vnc | GABA; AN13B002 = top target of all sugar GRNs (16,395) | disinhibition / gating | [V] |
| `bitter2` | `(GNG016\|GNG087\|GNG592)` | sez | GNG016 2 (NT unclear; feeds back onto LB1 GRNs), GNG087 3 (Glu) | | [V] |
| `song_vnc` | `(dPR1\|dMS2\|vPR6\|vPR9_[abc])` | vnc | 2 / 20 / 8 / 3+3+3 (vPR9 GABA) | | [V] |
| `lal_ps` | `(LAL083\|LAL126\|LAL179\|PS049\|PS059\|VES051\|VES052\|AOTU015\|AOTU019)` | central_other | DNa02 premotor inputs (see 5.3) | | [V] |
| `an_steer` | `(AN03A008\|AN04B003)` | vnc | DNa02 inputs 1,458 / 764 (ACh, ipsi) | | [V] |

Types that the proposals used but that **do not exist** in v1.0 [V]: `tectIN*`, `aDN1/aDN2`, `helicon`, `vAB3`, `PPN1`,
`iii4 MN`, `hi1 MN`, `tpN MN`, `MNhm*`, `MN8`, `MN11` (real: `MN11D`), `PEN_a`/`PEN_b` without the parenthesised suffix,
`vPR9` without `_a/_b/_c`. MN types present: `MN1, MN2Da, MN2Db, MN2V, MN3L, MN3M, MN4a, MN4b, MN5, MN6, MN9, MN10, MN11D, MN12D` (+4 more).

---

## 5. Verified pathway weights (synapse counts, `weight` property, both sides pooled) [V]

Columns: edges = number of (pre,post) neuron pairs, total = sum of synapses, mean = per pair, ipsi/contra = split by
`somaSide` equality. "—" = query returned no rows at weight ≥ 1 (the pathway **does not exist** in v1.0).

### 5.1 Looming → escape

| pre → post | edges | total | mean | ipsi | contra | note |
|---|---:|---:|---:|---:|---:|---|
| LC4 → DNp01 | 126 | 6,362 | 50.5 | 6,362 | 0 | every LC4 cell contacts the ipsilateral GF |
| LPLC2 → DNp01 | 185 | 4,862 | 26.3 | 4,862 | 0 | |
| LC4 → DNp04 | 126 | 11,597 | 92.0 | ipsi | 0 | strongest LC4 target |
| LC4 → DNp02 | 125 | 4,209 | 33.7 | ipsi | 0 | |
| LC4 → DNp11 | 122 | 3,666 | 30.0 | ipsi | 0 | |
| LPLC2 → DNp04 | 185 | 3,398 | 18.4 | ipsi | 0 | |
| LPLC2 → DNp02 | 4 | 5 | 1.3 | | | negligible |
| DNp11 → DNp01 | 3 | 118 | 39.3 | 10 | 108 | contralateral |
| DNp01 → GFC2 | 12 | 144 | 12.0 | 118 | 26 | GFC2 = 10 cells |
| GFC2 → TTMn | 12 | 481 | 40.1 | 458 | 23 | |
| DNp01 → TTMn | 2 | 90 | 45 | 90 | 0 | chemical only; the real GF→TTMn synapse is mostly **electrical** → patch |
| DNp01 → PSI | 4 | 16 | 4 | 11 | 5 | chemical only; electrical in reality → patch |
| PSI → DLMn c-f | 8 | 406 | 50.8 | 0 | **406** | PSI crosses the midline |
| PSI → DLMn a, b | 2 | 43 | 21.5 | 43 | 0 | |
| PSI → DVMn 3a, b | 4 | 51 | 12.8 | 0 | 51 | |
| JO-A/B → DNp01 | 11 | 531 | | | | auditory input to the GF |
| LC6 → DNp09, LPLC1 → DNp09 | — | | | | | **absent** (proposal 2 assumed it) |
| LC9 → DNp09 | 122 | 1,977 | 16.2 | ipsi | 0 | DNp09's #1 input (top-22 list: LC9 1,760 @w≥10, LC31a 753, PVLP020 592 GABA, …) |
| LC31a → DNp09 | 32 | 763 | 23.8 | ipsi | 0 | |
| TTMn top inputs (w≥5) | | | | | | IN13A022 987 (GABA), IN21A026 507 (Glu), GFC2 471, IN13A032 325, … DNp01 90 |

### 5.2 Optic lobe (sampled: 150 presynaptic cells)

R1-R6 → L1 26.7 syn/pair, → L2 27.3, → L3 5.3 (histaminergic, inhibitory) · L1 → Mi1 64.7, → L5 54.5, → C3 44.7,
→ C2 18.4, → Tm3 11.7 (≈7 Tm3 per L1) · Mi1 → T4a 10.1 (≈7 Mi1 per T4a), Tm3 → T4a 5.1, Mi9 → T4a 5.1, Mi4 → T4a 3.9 ·
Tm9 → T5a 4.8 · T4a → LPLC2 3.2 (1,483 T4a → 184 LPLC2), T5a → LPLC2 3.5, T5a → LC4 1.5, T4a → LC4 —.

### 5.3 Steering / walking

| pre → post | edges | total | mean | ipsi | contra |
|---|---:|---:|---:|---:|---:|
| PFL3 → DNa02 | 24 | 736 | 30.7 | 0 | **736** (L→R 380, R→L 356) |
| PFL3 → DNa03 | 24 | 468 | 19.5 | 0 | 468 |
| PFL3 → DNb01 | 24 | 753 | 31.4 | 0 | 753 |
| DNa03 → DNa02 | 2 | 552 | 276 | 552 | 0 |
| DNa02 → Sternal anterior rotator MN | 12 | 776 | 64.7 | 776 | 0 |
| DNg100 → IN13A001 | 6 | 217 | 36.2 | 0 | 217 |
| PFL2 → DNg13 | — | | | | (proposal 2 assumed it) |
| MBON01/MBON11 → DNg100 | — | | | | (proposals 2/3 assumed it) |
| DNa02 top inputs (w≥10) | AN03A008 1,458 (ACh, ipsi) · PS049 1,020 (GABA, ipsi) · PS059 998 (GABA, ipsi) · AN04B003 764 · PFL3 736 (contra) · LAL083 667 (Glu, contra) · LAL126 634 (Glu, contra) · VES052 618 / VES051 607 (Glu, ipsi) · LAL179 588 (ACh, contra) · AOTU019 586 (GABA, contra) · AOTU015 572 (ACh, ipsi) · DNa03 552 | | | | |
| DNg13 top inputs | VES200m 361 (Glu) · LAL073 327 (Glu, contra) · CB0244 324 · GNG532 313 · DNg97 236 (contra) · LAL083 200 | | | | |
| DNg100 top inputs | GNG458 2,070 (GABA) · DNge129 1,879 (GABA) · AN02A002 1,569 (Glu) · PLP300m 1,363 (ACh) · PVLP137 1,194 (ACh) — mostly inhibitory; there is **no verified excitatory brain pathway** from our sensory groups to DNg100, so the exploration baseline is injected (documented) | | | | |
| DNg60/DNg74 outputs | Sternal posterior rotator MN 1,037 · DNge050 1,016 · Pleural remotor/abductor MN 822 · Tr flexor MN 788 · Sternotrochanter MN 719 — GABAergic DNs inhibiting leg MNs directly | | | | |
| MDN outputs | IN07B010 476 · LBL40 463 · IN03B015 462 (GABA) · IN12B003 461 (GABA) · … DNa13 336 | | | | |
| DNa02 out-degree | 1,283 edges, 14,455 synapses; 495 edges ≥ 5 synapses carry 13,076 (90.5%) | | | | |

### 5.4 Flight

DNg02_a → DLMn c-f 101 (36 pairs, mean 2.8; 58 ipsi / 43 contra) · DNg02_a → DLMn a, b 24 · pooled DNg02_a–g outputs
(w≥5): MNwm36 2,349 · tp2 MN 1,362 · AN27X004 853 (His) · IN19B043 836 · AN19B019 797 · ps1 MN 759 · DVMn 1a-c 436 ·
DLMn c-f 423 · DVMn 3a, b 326 · DNp31 327 · dMS10 325. DNp03 → i1 MN 75 (contra), DNp03 → b1 MN —.

### 5.5 Feeding (SEZ) — the real chain differs from all three proposals

Direct outputs of labellar sugar GRNs LB3b/LB3c (w≥5): GNG038 1,361 (GABA) · GNG042 902 (GABA) · ANXXX462a 757 (ACh) ·
GNG215 741 (ACh) · GNG175 580 (GABA) · GNG232 495 (ACh) · AN13B002 464 (GABA) · GNG132 426 (ACh) · ANXXX462b 351 ·
GNG452 332 · GNG229 306 · GNG197 274 · DNg103 226 · GNG228 212 · GNG254 209 · GNG089 196 · DNg67 195 · DNpe030 140.
Pooled sugar GRNs (incl. leg/wing): AN13B002 **16,395** · AN05B023d 7,964 · ANXXX013 3,920 · DNge153 3,405 · IN05B011a 2,956
· DNpe029 1,984 · PRW047 1,522 · PRW046 1,495 · PRW070 1,488 · AN04A001 1,454. LgLG3 → AN13B002 8,770; WG2 → AN13B002 6,936.

MN9 inputs (w≥5): DNge062 556 (ACh) · GNG015 478 (GABA) · GNG120 442 (ACh) · GNG095 432 (GABA) · GNG117 410 (ACh) ·
GNG130 407 (GABA) · GNG108 380 (ACh) · GNG234 362 (ACh) · GNG180 287 (GABA) · GNG184 272 (GABA) · DNge051 231 (GABA) ·
DNge080 219 (ACh).

**No 2-hop sugar → X → MN9 path exists at w ≥ 10 on both hops.** Verified 3-hop chains (w ≥ 20 per hop, weights
w1/w2/w3): GNG215 (ACh) → GNG108 (ACh) → MN9 (588/56/373); GNG232 (ACh) → GNG108 → MN9 (283/96/373);
GNG232 → DNge080 (ACh) → MN9 (283/119/197); GNG089 → GNG108 → MN9 (112/59/373); GNG089 → GNG120 → MN9 (49/35/359);
ANXXX462b → DNge062 → MN9 (136/27/464); disinhibition: GNG042 (GABA) → GNG015 (GABA) → MN9 (595/329/443);
inhibitory branch: GNG132 (ACh) → GNG130 (GABA) → MN9 (368/144/391). Proposals' `GNG540/GNG550/GNG056 → GNG232 → MN9`
and `DNg67 → MN9`, `GNG055 → MN9` are **absent** (0 edges).

Bitter: LB1a-e → GNG016 4,990 (NT unclear; GNG016 outputs go back to LB1e 924, LB1c 746, LgAG3, LB1b, PhG13/14/16 —
i.e. feedback onto GRNs), → AN05B023a 3,445 (GABA), → GNG087 2,148 (Glu), → GNG592 1,365 (Glu). **No 2-hop bitter → MN9
path** at w ≥ 10; bitter suppression of feeding must be an engineered edge in the synthetic graph (labelled).

### 5.6 Mushroom body / antennal lobe / central complex

KCg-m → MBON01 30,660 (1,340 KCs, mean 22.6) · KCg-m → MBON11 24,213 · KCg-m ↔ APL 79,271 / 79,151 · KCab-* → APL 71,045,
→ DPM 21,728, → MBON06 12,634, → MBON14 1,795, → MBON18 1,382, → MBON11 661 (KCab-c → MBON12 —) · PAM01 → KCg-m 20,477
(15,263 pairs, mean 1.3) · PPL101 → KCg-m 7,224 · ORN_DM1 → DM1_lPN 13,384 (73 ORNs → 2 PNs) · DM1_lPN → KC 853 pairs,
mean 21.1 · EPG → PEN_a(PEN1) 6,100 · PEN_a → EPG 14,398 · EPG → Delta7 19,896 · Delta7 → EPG 4,294 · ER4d → EPG 12,335
(GABA) · EPG → PFL3 1,182 (968 ipsi) · Delta7 → PFL3 5,617.

### 5.7 Courtship / song / grooming

pC1_14a → pIP10 565 (6 → 2) · pIP10 → dPR1 1,112 (mean 278, both sides) · pMP2 → dPR1 1,750 · dPR1 → dMS2 1,646 (mostly contra)
· dMS2 → hg1 MN 3,657 (mean 107.6) · dMS2 → b1 MN 155 · vPR9_a/c → pIP10 (GABA, contra) · pIP10 top inputs: aIPg7 1,280
(ACh) · ICL008m 916 (GABA) · AVLP710m/717m/718m · pC1_14a 565 · pC1 top inputs: mAL_m8 2,617 (GABA), SMP702m, mAL_m1 2,468
(GABA), oviIN, pC1_18b, AN08B020 1,381 (ACh ascending; its own inputs are DNs, not leg GRNs), AVLP732m/733m (ACh; candidate
auditory relay [?]). Grooming: JO-C/E/F → SAD093 (ACh) → DNg62/DNge078 (275 → 712) · BM → aDNs 617 · aDN inputs also DNg98 948 (GABA).

---

## 6. Region taxonomy (8 regions, fixed order)

`REGIONS = ("optic_lobe", "antennal_lobe", "mushroom_body", "central_complex", "sez", "central_other", "descending_motor", "vnc")`

`region_of(superclass, cls, type)` — first rule that matches wins:
1. superclass ∈ {ol_intrinsic, ol_sensory, visual_projection, visual_centrifugal, visual_projection_tbc} or cls == "visual" → `optic_lobe`
2. cls ∈ {olfactory, ALPN, ALLN, ALIN, ALON} or type matches `ORN_.*` → `antennal_lobe`
3. cls ∈ {Kenyon_Cell, MBON, DAN} or type ∈ {APL, DPM} → `mushroom_body`
4. cls == "CX" → `central_complex`
5. superclass == "cb_motor", or (cls == "gustatory" and superclass == "cb_sensory"), or cls == "SEZPN", or type matches `(GNG[0-9]+|PhG.*|LB[0-9].*|PRW[0-9]+|SAD[0-9]+|FLA[0-9]+|MN[0-9]+[A-Za-z]*)` → `sez`
6. superclass ∈ {descending_neuron, vnc_motor, vnc_efferent, efferent_descending, sensory_descending} → `descending_motor`
7. superclass starts with `vnc` or ∈ {ascending_neuron, sensory_ascending, efferent_ascending, sensory_ascending_tbc} → `vnc`
8. else (remaining cb_intrinsic, non-gustatory cb_sensory such as JO/BM, cb_endocrine, ENS, null) → `central_other`

Side: `somaSide` `L`/`R` → -1/+1; anything else (M, null, '') → 0. Fallback: `instance` suffix `_L`/`_R`.

---

## 7. Neuron model: Shiu et al. 2024 LIF, with numerically verified constants

Source: Shiu P.K., Sterne G.R., Spiller N., et al. *A Drosophila computational brain model reveals sensorimotor
processing*, **Nature 634:210–219 (2024)**, doi:10.1038/s41586-024-07763-9 [L, DOI verified]. Parameter values below are
the ones all three proposals agree on and the Brian2 `model.py` of that paper [D]:

| parameter | value | note |
|---|---|---|
| v_rest = v_reset | −52 mV | |
| v_th | −45 mV | threshold gap 7 mV |
| tau_m | 20 ms | membrane |
| tau_s | 5 ms | synaptic (current-based exponential, in mV) |
| t_ref | 2.2 ms | absolute refractory ("unless refractory" freezes v and g) |
| delay | 1.8 ms | axonal/synaptic delay |
| w_syn | 0.275 mV per synapse | Shiu's single free parameter; times synapse count, times presynaptic sign |
| sign | ACh/DA/OA/5-HT +1; GABA/Glu/His −1; unknown/unclear +1 | Shiu convention; histamine inhibitory added for the optic lobe (ort chloride channel) [L: Lappalainen 2024] |
| Poisson forcing | per driven neuron, per step, with p = rate·dt/1000: `g += f_poi·w_syn = 250 × 0.275 = 68.75 mV` | from proposal 1's reading of Shiu's code [D]; produces one spike per event (verified below) |
| Reset | v = v_reset, g = 0, refractory | g-zeroing is the proposals' reading of Shiu's reset [D] |

Equations: `dv/dt = (v0 − v + g + I_ext)/tau_m`, `dg/dt = −g/tau_s`, `on_pre: g += w` after `delay`.

**Exact 2-D update** (derived and simulated in this session [V]): with `A_m = exp(−dt/tau_m)`, `A_s = exp(−dt/tau_s)`,
`B = tau_s/(tau_m − tau_s)·(A_m − A_s)`, and `g_old` the value before this step's decay,
`v ← v0 + I_ext + (v − v0 − I_ext)·A_m + B·g_old`, `g ← g_old·A_s`, then threshold, then add arriving input to g, then reset.

| dt (ms) | A_m | A_s | B | ref_steps | delay_steps |
|---|---|---|---|---|---|
| 1.0 | 0.951229 | 0.818731 | 0.044166 | 2 | 2 |
| 0.5 | 0.975310 | 0.904837 | 0.023491 | 4 | 4 |
| 0.2 | 0.990050 | 0.960789 | 0.009753 | 11 | 9 |
| 0.1 | 0.995012 | 0.980199 | 0.004938 | 22 | 18 |

Numerically verified consequences [V]:
* Forced kick of 68.75 mV into g at t=0 → first spike at **3.1 ms (dt 0.1), 3.5 ms (dt 0.5), 4.0 ms (dt 1.0)** with the
  ordering above (input added after the update, so the latency is 3 ms + one step of quantisation). Proposal 1's claim of
  "3.0 ms at any dt" holds only if the kick is applied *before* the update in the same step; the SPEC fixes the ordering and
  the test asserts `3.0 ≤ t_spike ≤ 3.0 + dt`.
* Unitary EPSP (0.275 mV kick): peak **0.0433 mV at 9.3 ms** (dt 0.1) / 10 ms (dt 1.0) → ≈ **162 coincident unitary
  synapses** to reach threshold from rest.
* Constant-current inversion `I = (v_th − v0)/(1 − exp(−((1000/r) − t_ref)/tau_m))`: 10 Hz → 7.05 mV, 20 → 7.71, 50 → 11.88,
  **100 → 21.68**, 150 → 34.97 mV; simulated rates at dt 1.0: 10.0 / 20.0 / 50.0 / 100.0 / 143 Hz (dt 0.1: 148.9 at 150).
* Steady-state synaptic drive: `g_ss = k_in · w · w_syn · gain · (r/1000) · tau_s` (mV) and `v_ss = v0 + g_ss`
  (proposal 2's `calibrated_weight`). Examples: k_in 12, r 100 Hz, target 14 mV → w = 8; k_in 4, r 60 → w = 42; k_in 155, r 33 → w = 2.
* With **real** LC4/LPLC2 → GF counts (5,601 synapses per side) and gain 1.0, the GF steady-state input crosses the 7 mV gap
  at ≈ **0.9 Hz** of population rate, i.e. one LC4 spike (50 synapses) gives a 13.9 mV g-kick ≈ 2.2 mV peak in v, and ≈ 4
  near-coincident LC4/LPLC2 spikes fire the GF. Literature-weight graphs therefore need `calibrate_gain` and modest looming
  rates; calibrated-weight synthetic graphs are safe by construction.
* Background "noise current" for the required **1–5 Hz rest activity** [V]: adding `g += N(mu·dt, sigma·sqrt(dt))` mV per
  step to every neuron gives, for isolated neurons at dt 1.0: (mu 0.5, sigma 3.0) → 1.33 Hz, **(mu 0.5, sigma 3.5) → ≈ 2.3 Hz**,
  (0.5, 4.0) → 3.45 Hz, (0.8, 3.0) → 4.3 Hz, (1.0, 2.0) → 4.1 Hz; the same values at dt 0.5 reproduce the dt 1.0 rates
  (0.5/3.0 → 1.32 Hz, 0.5/4.0 → 3.44 Hz). Pure current-mode noise (OU on I_ext, sigma ≤ 5 mV) yields < 0.7 Hz — noise must
  enter through g. Recurrent background at in-degree 25, w 4 and 2 Hz adds only g_ss = 0.275 mV (subcritical).

Related modelling references (DOIs verified on Crossref): Lappalainen et al. 2024 Nature 10.1038/s41586-024-07939-3
(connectome-constrained vision model); Dorkenwald et al. 2024 Nature 10.1038/s41586-024-07558-y and Schlegel et al. 2024
Nature 10.1038/s41586-024-07686-5 (FlyWire); Eckstein et al. 2024 Cell 10.1016/j.cell.2024.03.016 (NT prediction);
Takemura et al. 2024 eLife 10.7554/eLife.97769 (MANC); Cheong et al. eLife 10.7554/eLife.96084 (DN → motor circuits).

---

## 8. DexScreener API [V live sample 2026-09-10]

* Endpoint used: `GET https://api.dexscreener.com/tokens/v1/{chainId}/{tokenAddress[,addr2,...]}` → **bare JSON array** of
  pair objects (verified). Unknown token → `[]` with HTTP **200** (verified) — never a 404.
* Alternative: `GET https://api.dexscreener.com/latest/dex/tokens/{tokenAddress}` → `{"schemaVersion":"1.0.0","pairs":[…]}` (verified).
* Rate limits (docs): 300 requests/min for `/tokens/v1`, `/latest/dex/*`, `/token-pairs/v1`; 60/min for `/token-profiles`
  and `/token-boosts`. [D/medium] Responses are edge-cached ~60 s [D] → poll every 60 s; 30 s polling returns duplicates.
* No trade-level endpoint exists; buy/sell "events" are deltas of the 5-minute counters between polls. [V by schema]
* Verified pair object shape (Wrapped SOL / USDC on Raydium):

```json
{"chainId":"solana","dexId":"raydium","url":"https://dexscreener.com/solana/58oq...","pairAddress":"58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
 "baseToken":{"address":"So111...112","name":"Wrapped SOL","symbol":"SOL"},
 "quoteToken":{"address":"EPjF...Dt1v","name":"USD Coin","symbol":"USDC"},
 "priceNative":"100.09019","priceUsd":"100.090",
 "txns":{"m5":{"buys":648,"sells":656},"h1":{"buys":26649,"sells":27601},"h6":{"buys":96602,"sells":97927},"h24":{"buys":482322,"sells":486935}},
 "volume":{"h24":82556292.09,"h6":12671063.94,"h1":3444547.17,"m5":83071.16},
 "priceChange":{"m5":0.46,"h1":-0.05,"h6":-0.73,"h24":-3.39},
 "liquidity":{"usd":25108908.33,"base":125588,"quote":12538748},
 "pairCreatedAt":1669602450000,
 "info":{"imageUrl":"…","header":"…","openGraph":"…","websites":[{"url":"https://solana.com","label":"Website"}],"socials":[{"url":"https://x.com/solana","type":"twitter"}]}}
```

Parsing rules derived from the sample: `priceUsd` and `priceNative` are **strings** (parse with `float`); `fdv` and
`marketCap` were **absent** on this pair (optional keys — use `.get`); `priceChange.m5` may be absent on illiquid pairs
[D]; `liquidity` may be null [D]; choose the pair with the largest `liquidity.usd`; `chainId` examples: `solana`, `ethereum`,
`base`, `bsc`. Client: stdlib `urllib.request` or `httpx` (0.28.1 installed) with `User-Agent: synapsefly/0.1`.

---

## 9. Anthropic Python SDK — installed 0.86.0, verified surface [V]

* `anthropic.__version__ == "0.86.0"`. `anthropic.resources.messages.Messages.create` **accepts `output_config`**
  (structured outputs), `thinking`, and there is `Messages.parse`; **no** `betas`/`fallbacks` on the non-beta client;
  `output_format` is not a parameter. `Message.stop_reason` literal = `end_turn|max_tokens|stop_sequence|tool_use|pause_turn|refusal`.
  Exceptions present: `RateLimitError`, `APIStatusError`, `APIConnectionError`, `AuthenticationError`, `BadRequestError`.
  → The structured-output call in the SPEC works on the installed version; the judges' "needs 1.x" concern only applies to
  `betas=[...]`/`fallbacks`/`httpx2`.
* Authoritative call (from the task, matches the bundled claude-api skill): `client = anthropic.Anthropic()` (reads
  `ANTHROPIC_API_KEY`), `client.messages.create(model="claude-opus-5", max_tokens=512, system=..., messages=[{"role":"user","content": ...}])`,
  no `thinking` parameter (adaptive is the default on claude-opus-5), **never** `budget_tokens`, no assistant prefill,
  no `temperature`; iterate `response.content` and use blocks with `block.type == "text"`; check `response.stop_reason`
  (`"refusal"` → fallback template; `"max_tokens"` → truncate/validate). Optional JSON: `output_config={"format": {"type":
  "json_schema", "schema": {...}}}` (works on 0.86.0), or `client.messages.parse(..., output_format=PydanticModel)`.
* Pricing (skill table, cached 2026-06-24): claude-opus-5 **$5 / $25 per MTok** (input/output). A tweet call ≈ 1.2k input +
  ≤ 200 output tokens ≈ $0.01.
* Upgrade to 1.x (optional, `pip install -U "anthropic>=1.4,<2"`): HTTP layer moves to `httpx2`, Python ≥ 3.10, Text
  Completions and `temperature/top_p/top_k` removed, raw-dict `output_format=` removed (use `output_config`). None of these
  affect the SPEC's call. Server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"], fallbacks="default"`)
  require 1.x and are optional (`FLY_LLM_FALLBACKS=1`).

---

## 10. X (Twitter) posting — tweepy 4.17.0 verified surface [V]

* `tweepy.Client.create_tweet(text=..., media_ids=[...], ...)` exists (params verified: `text, media_ids, quote_tweet_id,
  in_reply_to_tweet_id, reply_settings, poll_*, place_id, community_id, user_auth`). Response: `resp.data["id"]`.
* `tweepy.Client` has **no** `media_upload`/`upload_media`/`create_media` (verified). v1.1 `tweepy.API(auth).media_upload(filename, file=BytesIO)`
  exists (verified) but v1.1 media endpoints may be sunset [D]. Primary image path therefore = raw
  `POST https://api.x.com/2/media/upload` (multipart `media`, `media_category=tweet_image`) signed with
  `requests_oauthlib.OAuth1` (2.0.0 installed, tweepy dependency) → `data.id` [D for the endpoint shape]; fallback v1.1;
  final fallback text-only.
* Exceptions verified: `tweepy.TooManyRequests` (429; has `reset_time` in 4.17 [D]), `tweepy.Forbidden` (403),
  `tweepy.Unauthorized`, `tweepy.HTTPException`, `tweepy.TweepyException`; `tweepy.OAuth1UserHandler` exists.
* Credentials: `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` (app must have Read+Write; tokens
  regenerated after enabling write). [D]
* Pricing: proposals state pay-per-use ≈ $0.015 per post and $0.20 per post containing a URL, no free write tier. **[?]
  not verified here** — check https://developer.x.com before setting `FLY_X=post`. The tweet validator strips URLs regardless.

---

## 11. Local environment (verified 2026-09-10)

| package | version | role |
|---|---|---|
| Python | 3.14.3 (`py -3`, MSC v.1944 64-bit) | |
| numpy | 2.4.3 | runtime |
| torch | 2.11.0+cpu (10 threads default; `os.cpu_count()` = 16) | optional backend |
| fastapi / starlette / uvicorn / websockets | 0.135.2 / 1.0.0 / 0.42.0 / 16.0 | runtime |
| pydantic | 2.12.5 | runtime (protocol models) |
| httpx | 0.28.1 | runtime (DexScreener) |
| anthropic | 0.86.0 | optional (LLM) |
| tweepy / requests_oauthlib | 4.17.0 / 2.0.0 | optional (X) |
| pytest | 9.1.1 | dev |
| pyarrow / pandas / scipy / neuprint-python | 25.0.1 / 3.0.5 / 1.18.1 / 0.6.3 | **present in the global site-packages although the task brief says they are not** — treat as prepare-time only; the runtime must not import them |
| orjson | missing | not used |
| Node / npm | 24.11.1 / 11.6.2 | frontend |

Frontend scaffold **already exists** at `C:\Users\USER\fly\frontend` [V]: Next **16.3.4**, React 19.2.8, TypeScript 5.9.3,
Tailwind **4.3.3** via `@tailwindcss/postcss` (no `tailwind.config.*`; `globals.css` uses `@import "tailwindcss"` and
`@theme inline`), ESLint 9 flat config, `node_modules` installed, `app/{layout.tsx,page.tsx,globals.css,favicon.ico}`,
`public/*.svg`, `tsconfig` paths `@/* → ./*`, no `src/` dir. `AGENTS.md`/`CLAUDE.md` (auto-added by `next dev`) say: read
`node_modules/next/dist/docs/` before writing code. Relevant Next 16 facts from those docs [V]: Turbopack is the default
bundler (`next dev --webpack` to opt out); `LayoutProps<'/'>`/`PageProps` are global type helpers (no import); request APIs
(`params`, `searchParams`, `cookies()`, …) are async; `middleware.ts` is now `proxy.ts`; browser-visible env vars must be
prefixed `NEXT_PUBLIC_`; `next/image` defaults changed (we do not use it). The scaffold's `layout.tsx` imports Google fonts
via `next/font/google` — **remove** for offline-first operation.

Repository root also contains a `.gitignore` [V] that ignores `data/*` except `data/.gitkeep` and `data/README.md` — any
persistent file we want tracked must live outside `data/` (e.g. `docs/NOTICE.md`), runtime state in `data/` stays untracked.

---

## 12. Prior art

| project | what it is | relevance | conf. |
|---|---|---|---|
| Shiu et al. 2024 LIF model (Brian2, FlyWire v630) | whole-brain LIF with the parameters above; sugar GRN activation → MN9; `w_syn` fit | our neuron model and calibration targets (sugar 100 Hz → ~80% max MN9) | [L] |
| FlyWire / Codex (Dorkenwald 2024, Schlegel 2024) | female whole-brain connectome, CSV downloads, CC-BY-NC | loader schema parity | [L] |
| MANC (Takemura 2024), Cheong et al. | male VNC connectome; DN → MN circuits, hemilineage naming (`IN13A001`, `AN08B020`, …) reused in MaleCNS | VNC type strings | [L]/[V] |
| Lappalainen et al. 2024 | connectome-constrained optic-lobe model; histamine sign | optic lobe wiring sanity | [L] |
| neuprint-python 0.6.3 / navis / natverse malecns | official access tooling | fetch scripts | [V] |
| "fly-brain-minecraft" (MaleCNS ≥5-synapse graph in a game, gain 0.65, KC input gain 0.25) | cited by proposals as prior port | gain defaults for literature-weight mode | **[?] existence/values not verified here** |
| Brian2 | reference integrator; "unless refractory" semantics | test oracle (not a dependency; not installed) | [D] |

---

## 13. Behavioural literature used by the decoder (DOIs verified on Crossref) [L]

von Reyn 2014 Nat Neurosci 10.1038/nn.3741 (GF spike timing → jump); Ache 2019 Curr Biol 10.1016/j.cub.2019.01.079
(LC4/LPLC2 → GF looming size/velocity); Klapoetke 2017 Nature 10.1038/nature24626 (LPLC2); Zacarias 2018 Nat Commun
10.1038/s41467-018-05875-1 (DNp09 freezing); Bidaye 2014 Science 10.1126/science.1249964 (MDN backward walking); Bidaye 2020
Neuron 10.1016/j.neuron.2020.07.032 (DNp09/BPN forward walking); Namiki 2018 eLife 10.7554/eLife.34272 (DN atlas); Namiki 2022
Curr Biol 10.1016/j.cub.2022.01.008 (DNg02 flight power); Yang 2024 Cell 10.1016/j.cell.2024.08.033 (DNa02/DNa01/DNg13 steering);
Westeinde 2024 Nature 10.1038/s41586-024-07039-2 and Mussells Pires 2024 Nature 10.1038/s41586-023-07006-3 (PFL3 → DNa02 goal
steering); Seelig & Jayaraman 2015 Nature 10.1038/nature14446 and Green 2017 Nature 10.1038/nature22343 (EPG/PEN compass);
Aso 2014 eLife 10.7554/eLife.04577 (MBON valence); Sterne 2021 eLife 10.7554/eLife.71679, Shiu 2022 eLife 10.7554/eLife.79887,
Engert 2022 eLife 10.7554/eLife.78110 (taste circuits, LB3b/c sweet, LB1 bitter); von Philipsborn 2011 Neuron
10.1016/j.neuron.2011.01.011 (pIP10/dPR1/vPR6 song); Hampel 2015 eLife 10.7554/eLife.08758 (aDN grooming); Donlea 2014 Neuron
10.1016/j.neuron.2013.12.013 (dFB sleep).

---

## 14. Open items / not verified

1. Exact Shiu Brian2 reset semantics (`g = 0` on spike) and the Poisson-forcing weight (68.75 mV) — from proposals only. If
   fidelity runs disagree with Shiu's MN9 curves, revisit `snn/params.py` first.
2. Pheromone GRN identity (`LgLG1a/1b/2`) and the pC1 auditory relay (`AVLP732m/733m`) — engineered hops, labelled in code.
3. DexScreener 60-s edge cache and per-endpoint limits — from docs/proposals, not measured.
4. X API pricing and v2 media-upload payload shape.
5. The "~125M synapses" figure vs neuPrint's 45.7M pre / 311.8M post counts.
6. The full-graph edge count (25.6M at weight ≥ 1) and the 205 MB CSR size — from the proposals' benchmarks on this machine.
7. Codex MCNS mirror header names.
