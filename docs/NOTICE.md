# NOTICE — data provenance, citations and licences

SynapseFly / FlyBrain (`flybrain` 0.1.0). This file is tracked; everything under `data/` is gitignored and is
**never** part of the repository.

## 1. No connectome data is redistributed by this repository

* The default configuration (`FLY_CONNECTOME_SOURCE=synthetic`) runs on a **synthetic structured stand-in shaped like
  MaleCNS v1.0**. It is generated on the user's machine from a population/projection table (`docs/SPEC.md` section g)
  that uses real cell-type names and published/verified counts, but **no synapse table, body id, coordinate or any
  other record of the MaleCNS dataset is copied**. It is not real connectome data and is labelled as such everywhere
  (`hello.connectome.note`, Help > About, the LLM summary, the README). Its licence field is `"synthetic (no data)"` and
  its citation field is `null`.
* Real data is downloaded **by the user**, on demand, with `scripts/prepare_malecns.py` (Google Cloud Storage flat
  files, anonymous HTTPS) or `scripts/fetch_neuprint.py` (neuPrint API, optional token) into `data/connectome/…`, which
  `.gitignore` excludes. The runtime never fetches data from the network.
* The test fixtures under `backend/tests/fixtures/` (`neuprint_small`, `codex_small`: 40 neurons / ~120 edges) are
  **hand-written schema examples** with the column layouts of the two real formats; the numbers in them are invented.

## 2. Datasets

### 2.1 Janelia FlyEM MaleCNS v1.0 — CC-BY 4.0

* Dataset: `male-cns:v1.0` (neuPrint), "The complete MaleCNS connectome from the Janelia FlyEM Team Project, the
  Cambridge Drosophila Connectomics Group, and Google Connectomics. Covers the brain and nerve cord; 167k neurons."
  165,122 neurons with status `Traced`.
* Licence: **Creative Commons Attribution 4.0 International (CC-BY 4.0)** — stated on the project page and the download
  page ("The FlyEM Male CNS dataset is licensed under CC-BY"). https://creativecommons.org/licenses/by/4.0/
* Attribution required when real data is used (this is what `hello.connectome.citation` carries):

  > Berg et al. (2026) Sexual dimorphism in the complete Drosophila male central nervous system connectome. Cell.
  > doi:10.1016/j.cell.2026.08.015 (data: FlyEM MaleCNS v1.0, CC-BY 4.0)

* Paper: Berg S., Beckett I.R., Costa M., Schlegel P., Januszewski M., Marin E.C., Nern A., et al. *Sexual dimorphism in
  the complete Drosophila male central nervous system connectome.* **Cell**, 2026. doi:10.1016/j.cell.2026.08.015.
  Preprint: bioRxiv doi:10.1101/2025.10.09.680999.
* Sources used by the scripts: `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (public bucket;
  `body-annotations-male-cns-v1.0-minconf-0.5.feather`, `body-neurotransmitters-male-cns-v1.0.feather`,
  `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather`) and `https://neuprint.janelia.org`
  (`POST /api/custom/custom`, dataset `male-cns:v1.0`).
* Links: https://male-cns.janelia.org/ · https://www.janelia.org/project-team/flyem/male-cns-connectome ·
  https://janelia-flyem.github.io/male-cns/download/ · https://neuprint.janelia.org/?dataset=male-cns%3Av1.0
* The cell-type names, superclass/class taxonomy, `consensusNt` conventions and the verified population counts and
  pathway synapse counts recorded in `docs/RESEARCH.md` (sections 3–5) were obtained from this dataset through the
  public neuPrint API and are used to *shape* the synthetic generator. They are facts about the dataset, reproduced here
  under CC-BY with the attribution above.

### 2.2 FlyWire / Codex (FAFB, female) — CC-BY-NC 4.0

* Used only for **loader schema parity** (`classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz`,
  `connections.csv.gz`) and research comparisons. Licence: **CC-BY-NC 4.0** (non-commercial).
  https://creativecommons.org/licenses/by-nc/4.0/ — do not use FlyWire data in a commercial or token-marketing context;
  the loader sets `license = "CC-BY-NC 4.0"` so the UI shows it.
* Citations: Dorkenwald S. et al. (2024) *Neuronal wiring diagram of an adult brain.* **Nature**,
  doi:10.1038/s41586-024-07558-y; Schlegel P. et al. (2024) *Whole-brain annotation and multi-connectome cell typing of
  Drosophila.* **Nature**, doi:10.1038/s41586-024-07686-5. Neurotransmitter predictions: Eckstein N. et al. (2024) **Cell**,
  doi:10.1016/j.cell.2024.03.016.

### 2.3 MANC (male adult nerve cord) — CC-BY 4.0

* VNC type-name conventions (`IN13A001`, `AN08B020`, hemilineage naming) reused by MaleCNS and by the synthetic table.
  Takemura S. et al. (2024) *A Connectome of the Male Drosophila Ventral Nerve Cord.* **eLife**, doi:10.7554/eLife.97769;
  Cheong H.S.J. et al. *Transforming descending input into behavior: The organization of premotor circuits in the
  Drosophila Male Adult Nerve Cord connectome.* **eLife**, doi:10.7554/eLife.96084.

## 3. Model and parameter provenance

* Neuron model and constants (`v_rest = v_reset = -52 mV`, `v_th = -45 mV`, `tau_m = 20 ms`, `tau_s = 5 ms`,
  `t_ref = 2.2 ms`, delay `1.8 ms`, `w_syn = 0.275 mV`, sign convention, Poisson forcing `68.75 mV`):
  Shiu P.K., Sterne G.R., Spiller N., et al. (2024) *A Drosophila computational brain model reveals sensorimotor
  processing.* **Nature** 634:210–219, doi:10.1038/s41586-024-07763-9. The exact 2-D matrix-exponential update, the
  step-constant table and the background-noise rest-rate calibration were derived and numerically verified in
  `docs/RESEARCH.md` section 7; they are not copied from the paper's code.
* Optic-lobe histamine sign (photoreceptors inhibitory): Lappalainen J.K. et al. (2024) **Nature**,
  doi:10.1038/s41586-024-07939-3.
* Behavioural rules of the decoder (`docs/RESEARCH.md` section 13, DOIs verified on Crossref): von Reyn 2014
  (10.1038/nn.3741); Ache 2019 (10.1016/j.cub.2019.01.079); Klapoetke 2017 (10.1038/nature24626); Zacarias 2018
  (10.1038/s41467-018-05875-1); Bidaye 2014 (10.1126/science.1249964); Bidaye 2020 (10.1016/j.neuron.2020.07.032);
  Namiki 2018 (10.7554/eLife.34272); Namiki 2022 (10.1016/j.cub.2022.01.008); Yang 2024 (10.1016/j.cell.2024.08.033);
  Westeinde 2024 (10.1038/s41586-024-07039-2); Mussells Pires 2024 (10.1038/s41586-023-07006-3); Seelig & Jayaraman
  2015 (10.1038/nature14446); Green 2017 (10.1038/nature22343); Aso 2014 (10.7554/eLife.04577); Sterne 2021
  (10.7554/eLife.71679); Shiu 2022 (10.7554/eLife.79887); Engert 2022 (10.7554/eLife.78110); von Philipsborn 2011
  (10.1016/j.neuron.2011.01.011); Hampel 2015 (10.7554/eLife.08758); Donlea 2014 (10.1016/j.neuron.2013.12.013).
* Every biological number in the code carries a provenance tag in its docstring: `[V]` verified against the dataset or
  by simulation, `[L]` literature, `[E]` engineered (puppeteering / stand-in). The `[E]` items are listed in the README
  (section 8) so that fidelity claims can be audited; the engineered projection rows of the synthetic generator can be
  re-derived from the code itself with the one-liner printed there (`PROJECTIONS` filtered on `provenance == "E"`), and
  `backend/tests/test_docs_e8.py` fails if the README list and the generator disagree.

## 4. Third-party services (optional, off by default)

* **DexScreener** public API (`https://api.dexscreener.com/tokens/v1/…`, no key) — enabled by `FLY_MARKET=dexscreener`;
  subject to DexScreener's terms of use and rate limits. Data is displayed and used as a sensory input only; nothing is
  stored beyond the session log.
* **Anthropic API** (`FLY_LLM=anthropic`) — the model receives only the brain-summary JSON of `docs/SPEC.md` section
  d.6; usage is billed to the user's key.
* **X (Twitter) API** (`FLY_X=post`, tweepy) — posts are made from the user's own account and are the user's
  responsibility. The system prompt forbids financial advice, promises of returns and URLs.

## 5. Software licences of dependencies

numpy (BSD-3), fastapi (MIT), uvicorn (BSD-3), websockets (BSD-3), pydantic (MIT), httpx (BSD-3); optional: torch
(BSD-3), anthropic (MIT), tweepy (MIT), pyarrow (Apache-2.0), pandas (BSD-3), neuprint-python (BSD-3); frontend: next
(MIT), react / react-dom (MIT), tailwindcss (MIT), typescript (Apache-2.0), eslint (MIT). See each package for its full
text.

## 6. Disclaimer

This project is an art / meme / DeSci experiment. It is **not** investment advice; the simulated fly and its tweets
make no claims about any token's value. The default brain is a synthetic stand-in, not a measured nervous system; even
with real MaleCNS data loaded, the LIF model, the market-to-sense mapping and the motor decoding are simplifications
documented in `docs/SPEC.md` and `docs/RESEARCH.md`.
