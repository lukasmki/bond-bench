# bond-bench

A benchmark dataset and evaluation harness for bond order assignment methods

## Features

- a comprehensive dataset for bond order assignment
- an efficient and unified harness for evaluation of bond order assignment methods

## Getting the dataset

The prebuilt dataset is published as a GitHub release (tag `dataset-v<schema version>`). Pull it instead of rebuilding:

```bash
uv sync
uv run bond-bench pull                 # every stratum, reports and index.parquet into dataset/
uv run bond-bench pull A1 B5           # selected strata only (index.parquet is regenerated)
uv run bond-bench validate
```

Every file is checked against the release's `SHA256SUMS`. Files that are already up to date are skipped. Local files that differ from the release, such as your own rebuilds, are kept unless you pass `--force`. While the repository is private, `pull` needs a token: it uses `GITHUB_TOKEN`, `GH_TOKEN` or your `gh auth login`. Use `--tag` to pull another release.

To publish a new dataset release after a full build, run `uv run python scripts/publish_dataset.py`. It stages the assets in `build/release/` and prints the `gh release create` command. It refuses dev builds made with `--n`.

The dataset is derived from third-party sources with their own licenses; see [LICENSE.md](LICENSE.md).

## Getting published results

Benchmark results are published as a separate release (tag `results-v<schema version>`), so you can score and plot them without rerunning every method:

```bash
uv run bond-bench pull --results       # predictions, scores, provenance and plots into results/
uv run bond-bench score
```

The archive is checked against the release's `SHA256SUMS`. As with the dataset, local files that differ from the release are kept unless you pass `--force`, and results for methods not in the release are left alone. `results/summary.parquet` is regenerated afterwards, so it covers both. Each run's `provenance.json` records the SHA-256 of the stratum file it used, which matches the dataset release's `SHA256SUMS`.

## Building the dataset

The dataset is defined by `assets/benchmark_strata_manifest.csv`. Each stratum is built by its own builder into `dataset/<stratum_id>.h5`:

```bash
uv sync
uv run bond-bench info                 # strata, and whether each is implemented/built
uv run bond-bench build A1 B1 C1       # full size (n_structures from the manifest)
uv run bond-bench build A1 --n 30      # quick development build
uv run bond-bench validate             # schema, offsets, label chemistry, index
```

Raw downloads are cached in `data/raw/`. You can change the location with `--cache-dir` or `BOND_BENCH_CACHE`. Builds are deterministic for a given `--seed` and cache. Each build writes a funnel report of counts and rejection reasons to `dataset/reports/<stratum_id>.json`.

Implemented strata:

| id | source | notes |
| --- | --- | --- |
| A1 | PDB instances + wwPDB CCD | X-ray, three resolution bins. Platinum-style density filter (RSCC ≥ 0.9, RSR ≤ 0.3, complete, occupancy 1, no altlocs, not covalent). At most 3 instances per CCD id. Hydrogens are stripped; per-atom H counts are part of the label. |
| B1 | QMugs (GFN2-xTB) | The graph must match ChEMBL 27 by stereo-free InChIKey. Molecules present in GEOM-DRUGS are excluded by InChIKey connectivity layer. The first build streams GEOM's ~50 GB `rdkit_folder` once to collect its SMILES. |
| A2 | COD + curated SMILES | Entries whose curated SMILES Merkys et al. (2023) found isomorphic with the coordinate-derived structure; agreement with the deposited name is preferred. Single-crystal X-ray, R ≤ 0.05, no disorder, experimental H. CIFs come via rsync. Each record is one formula unit (salts stay charge-balanced). |
| B2 | GEOM-DRUGS | Lowest-energy conformer. Kept only if the conformer graph matches the GEOM SMILES and its charge. |
| B3 | QCML + GEOM-QM9 | Half QCML GFN0-equilibrium structures (≤ 8 heavy atoms, main-group elements), stratified by hetero-element signature. Half GEOM-QM9. Kept only if the geometry's connectivity matches the source graph. |
| B5 | Tautobase + Dhaked–Nicklaus | Tautomer groups (connected components over FixedH InChIs observed together). Each structure is ETKDG-embedded and then GFN2-xTB relaxed (tblite). A group is dropped if any member changes connectivity. Per-structure `pref_label`/`population` comes from one reference measurement (aqueous and quantitative preferred). |
| C1 | derived from B1 (+ B4) | Isotropic Gaussian noise with σ ∈ {0, 0.05, 0.10, 0.20} Å on every parent. Until B4 is built, all parents come from B1. |
| C5 | QCML + Transition1x | QCML normal-mode samples. Transition1x NEB images adjacent to an endpoint, labeled with the Grambow endpoint graph (matched by geometric connectivity), at most 1.5 eV above the endpoint and at least 0.5 eV below the TS. |

Every label goes through RDKit sanitization, a charge-sum check and a valence-plausibility check. The plausibility check rejects undeclared radicals and hypervalent S/P without multiple bonds, which catches a number of erroneous CCD definitions.

## Dataset format

Each `dataset/<stratum_id>.h5` stores ragged arrays:

- `/molecules/*` holds one row per structure: `structure_id`, `source_id`, `smiles`, `inchikey`, `total_charge`, `multiplicity`, `n_atoms`, `n_bonds`, `atom_offset`, `bond_offset`, and stratum-specific `meta_<key>` columns.
- `/atoms/*` holds the per-atom arrays:
  - `atomic_numbers`
  - `positions` (Å)
  - `formal_charges`
  - `implicit_hydrogens`: -1 where H are explicit atoms; otherwise the labeled H count
  - `radical_electrons`
- `/bonds/*` holds `bonds` (atom indices within the molecule), `bond_orders` (Kekulé 1/2/3; 9 is reserved for coordination bonds) and `aromatic`.
- The root attributes hold the manifest row, the schema version, the seed, build parameters and provenance.

`dataset/index.parquet` lists every structure across all strata.

```python
from bond_bench.data.store import StratumReader

reader = StratumReader("dataset/B1.h5")
record = reader[0]                          # MoleculeRecord with explicit H
for rec in reader.iter(strip_h=True):       # H-stripped variant, derived on the fly
    ...
```

## Evaluating methods

Each method maps a `Query` (elements, coordinates, total charge, multiplicity and the observation condition, but never the label) to a `Prediction`. The harness runs methods over the built strata, checks every output against the valid set 𝒢(z, Q), and scores it against the reference label.

```bash
uv run bond-bench methods                                  # registered methods and availability
uv run bond-bench run canonical-O --strata B1 A1 --n 200 --workers 8
uv run bond-bench run canonical-R --strata C1 --noise 0.05,0.1 --reps 3 --equivariance 50
uv run bond-bench run openbabel+project --strata A1       # wrappers are appended with +
uv run bond-bench run canonical-O --param preset=kim2015 --param c_max=2
uv run bond-bench score                                    # summary across all results
uv run bond-bench score --all-subgroups --rescore          # per res_bin / sigma, recomputed
uv sync --extra plot && uv run bond-bench plot             # matplotlib figures in results/plots/
uv run bond-bench plot --metric smiles_match taut_pref --format pdf
```

`bond-bench plot` writes:

- `strata_<metric>.png`: per-stratum bars for each method, with 95% CIs
- `noise_c1_<metric>.png`: robustness curves over the C1 σ ladder, plus `noise_onthefly_*` when `--noise` was used
- `equivariance.png`: violation rates per method

`--obs auto` (the default) follows the manifest's `hydrogens_default`: `Stripped` strata run H-free, `Present` strata run with explicit H, and `Both`/`Present + stripped` strata run both. With `--n`, records are sampled by group, so C1 noise ladders and B5 tautomer groups stay whole.

Implemented methods:

| name | family | notes |
| --- | --- | --- |
| `canonical-O` | optimization | Algorithm O: contact map A(X), then a lexicographic prior solved exactly with CP-SAT. The `kim2015` preset maximizes ΣB, then minimizes Σ\|q\|. The `atom_penalty` preset (`--param preset=atom_penalty`) minimizes the Wang et al. (2006) atomic penalty scores in `eval/tables/atom_penalties.csv`, then Σ\|q\|. It covers H, C, N, O, F, Si, P, S, Cl, Br and I; molecules with other elements are reported as unsupported. Returns every co-optimal structure; a `prior_tied` flag marks resonance or tautomer ambiguity. |
| `canonical-R` | rule | Algorithm R: bond-length windows and angle-based hybridization feed a greedy cascade in certainty order. Windows are centred on MMFF94 reference lengths (`scripts/mmff_windows.py`), so no stratum is used; `--param windows=b2` selects windows fitted on B2 instead. Hamming-nearest projection onto 𝒢 follows. |
| `rdkit-determinebonds` | optimization | RDKit's xyz2mol (Kim & Kim 2015). Requires explicit H. |
| `openbabel` | external | Open Babel `ConnectTheDots` + `PerceiveBondOrders` (GPL-2.0). Adapter `openbabel`. |
| `antechamber` | optimization | AmberTools antechamber `-j 4` (bondtype; Wang et al. 2006). Finds its own connectivity. Explicit H only; assigns no formal charges, so charged structures need `+project`. Adapter `antechamber` (pixi). |
| `indigo` | optimization | indigo-bondorder electron assignment (Welsh & Allison 2019, MIT) on the harness contact map. Explicit H only; connected graphs only; H C N O F P S Cl Br. `--param algorithm=fpt\|astar\|local_optimisation`. Adapter `indigo` (pixi, builds C++). |
| `ball` | optimization | BALL `assign_bond_orders` (AssignBondOrderProcessor, Dehof et al. 2011, LGPL-2.1) on the harness contact map, BALL penalty table. Explicit H only; connected graphs only; no formal charges. `--param strategy=fpt\|a_star\|ilp`. Adapter `ball` (pixi, builds C++, Linux). |
| `mambax` | ml | Loschen's random forest (`clf.p`, BSD-3) over pairs within 3 Å. Reports `edge_probs`; no formal charges or H counts. Adapter `mambax`. |
| `yuelbond` | ml | YuelBond edge GNN on heavy atoms; `--param checkpoint=geom_kekulized_3d` (default) or `geom_kekulized_cdg` (trained with 0.2 Å noise). Reports `edge_probs`; no formal charges or H counts. **Upstream has no licence**: code and weights are fetched by `setup.sh`, never shipped. Adapter `yuelbond`. |
| `bondnet` | ml | BondNet v4 joint PaiNN models (MIT; weights CC BY 4.0): explicit-H or heavy-only model chosen by the observation; `--param seed=42\|43\|44`. Reports `edge_probs`; reads no charge, no formal charges or H counts. Adapter `bondnet`. |
| `mperformer` | ml | MPerformer (GPL-3.0) on heavy atoms, upstream's thresholds on its continuous heads; predicts formal charges and H counts. Uses the checkpoint's trained charge head (`total_data`): upstream's `predict.py` asks for a head the checkpoint lacks and runs it untrained. Adapter `mperformer` (Uni-Core). |
| `external` | external | Any program that speaks the JSONL protocol in `eval/methods/external.py`: `--param command='prog {input} {output}'`. Queries carry the contact map as `contacts` when the method sets `send_contacts`. |
| `+project` | wrapper | Maps an invalid output to the nearest valid structure on the same support. |
| `+hybrid` | wrapper | Algorithm H: constrained MAP over 𝒢 from the inner method's `edge_probs` (for ML models). |

The shared primitives P1–P6 of `assets/boa_canonical_algorithms.md` are in `src/bond_bench/eval/core/`:

- `elements.py`: state tables Λ
- `geometry.py`: the contact map
- `validity.py`: the validity oracle
- `ilp.py`: the CP-SAT decoder, with lexicographic stages, solution pools and ⊥ diagnosis
- `canon.py`: equivariant tie-breaking and resonance classes

To add a method, subclass `Method` (in `eval/methods/base.py`), declare its `Capabilities`, implement `predict`, and decorate the class with `@register`. A method with heavy or conflicting dependencies should instead get its own environment under `adapters/<name>/` and subclass `ExternalMethod`.

### Adapters

Published methods with GPL code, PyTorch/Uni-Core models or conda-forge builds run in their own environment under `adapters/<name>/` (uv, or [pixi](https://pixi.sh) for conda-forge). Each `run.py` speaks the JSONL protocol of `eval/methods/external.py` through the helpers in `adapters/_shared/bbadapter.py`. `setup.sh` fetches the pinned upstream commit and checksummed weights into `adapters/<name>/vendor/` (not committed) and builds what needs building; the commands are under [Running the complete benchmark](#running-the-complete-benchmark).

The ML adapters run on CUDA when it is available. Adapters that report `edge_probs` (`mambax`, `yuelbond`, `bondnet`) can be decoded with `+hybrid`. Where a model has a 5-class output with an aromatic class, P(aromatic) is split equally between orders 1 and 2. `indigo` and `ball` assign bond orders to a given graph, so they receive the contact map A(X) that `canonical-O` uses (`send_contacts`), and their connectivity errors are those of `canonical-O`. Methods that do not predict formal charges or H counts report none, so their raw outputs are invalid for charged or H-stripped structures; `+project` and `+hybrid` repair them.

Known issues are documented with each adapter: `adapters/README.md` covers the shared conventions and the `+hybrid` charge-separation issue, and `adapters/<name>/README.md` covers upstream bugs, version pins, limits and test status.

Not wrapped: closed or licence-gated tools (IDATM/ChimeraX, SAMSON, Knodle, NAOMI/UNICON, Protoss, MOE, CSD), methods without public code, and cod-tools, which helped select stratum A2. See `assets/bond_order_methods_availability.md`.

Results go to `results/<method_tag>/<stratum>__<obs>.parquet` (predictions) and `.scores.parquet` (per-structure metrics), with a `provenance.json` alongside. `bond-bench score` aggregates them into `results/summary.parquet`, with Wilson 95% intervals for rates and a bootstrap interval for F1.

| metric | meaning |
| --- | --- |
| `valid` | output ∈ 𝒢(z, Q) under the broad Lewis table (the harness checks this; methods don't self-report it) |
| `exact` | B, q and h equal to the label, index by index |
| `resonance_match` | valid, with the same support and the same h as the label, i.e. in the label's resonance class |
| `smiles_match` | same RDKit canonical SMILES (resonance-invariant only up to aromaticity) |
| `f1_*`, `conn_*` | per-pair bond-order F1 and connectivity precision/recall over heavy-atom pairs |
| `taut_accept`, `taut_pref` | B5 only: the prediction is any member of the tautomer group, or the preferred member |
| `coverage` | fraction of structures the method supports; unsupported structures are excluded from the rates, while ⊥, errors and timeouts count as wrong |

Robustness is measured on C1 (`sigma=` subgroups) or by on-the-fly noise (`--noise`). Equivariance (`--equivariance N`) compares Φ((σ,g)·x) with σ·Φ(x). Inputs with exact geometric symmetry are reported separately, because strict equivariance is impossible for them.

## Running the complete benchmark

These steps run every registered method on every built stratum and then score and plot the results. They assume a Linux machine with an NVIDIA GPU. The four ML adapters use CUDA when it is available and fall back to CPU, which is much slower.

### Prerequisites

- `uv`, `git` and `curl` on `PATH`.
- [pixi](https://pixi.sh) for the `antechamber`, `indigo` and `ball` adapters. pixi is a single binary and installs without root:

  ```bash
  curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash   # installs to ~/.pixi/bin
  export PATH="$HOME/.pixi/bin:$PATH"
  ```

  Use a recent release, because the committed `pixi.lock` files were written by one. Put the `PATH` line in your job scripts as well, since compute nodes may not read `~/.bashrc`. Without pixi, skip those three adapters; the others need only uv.
- Disk space. The adapter environments and weights take several GB: the BALL build environment alone holds Qt6, Boost and compilers, and the ML weights total about 1 GB. If your home quota is small, move the caches to scratch:

  ```bash
  export PIXI_CACHE_DIR=/scratch/$USER/pixi-cache
  export UV_CACHE_DIR=/scratch/$USER/uv-cache
  ```

  The environments themselves live inside the clone (`adapters/<name>/.venv` and `.pixi`), so clone the repository where there is room too.

### 1. Get the code and the dataset

```bash
git clone <repository url> bond-bench && cd bond-bench
uv sync
uv run bond-bench pull            # needs GITHUB_TOKEN, GH_TOKEN or `gh auth login` while the repository is private
uv run bond-bench validate
```

### 2. Set up the adapters (once)

```bash
adapters/mambax/setup.sh        # uv; clf.p is in the upstream repo
adapters/yuelbond/setup.sh      # uv; 4 checkpoints from Zenodo 15353365
adapters/bondnet/setup.sh       # uv; 310 MB release archive from Zenodo 23080342
adapters/mperformer/setup.sh    # uv + Uni-Core; 584 MB checkpoint from Google Drive (gdown)
adapters/indigo/setup.sh        # pixi; builds the pyindigox module
adapters/ball/setup.sh          # pixi; builds BALL's assign_bond_orders from source (long)
(cd adapters/openbabel && uv sync)
pixi install --manifest-path adapters/antechamber/pixi.toml

uv run bond-bench methods       # every row should say "ok"; otherwise it names what is missing
```

On a machine with little memory, limit the C++ build jobs, e.g. `BUILD_JOBS=4 adapters/ball/setup.sh`. Google Drive can refuse the MPerformer download when its quota is exceeded; rerun `adapters/mperformer/setup.sh` later. Each `setup.sh` skips what is already in place.

### 3. Run every method

```bash
# built-in methods (in-process; --workers parallelizes over structures)
for m in canonical-O canonical-R rdkit-determinebonds; do
  uv run bond-bench run $m --workers 16 --equivariance 30
done
uv run bond-bench run canonical-O --param preset=atom_penalty --workers 16 --equivariance 30

# wrapped published methods (one adapter process per batch of 256 structures)
for m in openbabel antechamber indigo ball mambax yuelbond bondnet mperformer; do
  uv run bond-bench run $m --equivariance 30
done

# valid-set wrappers: +project maps any invalid output to the nearest valid structure,
# +hybrid decodes the edge probabilities of the models that report them
for m in openbabel rdkit-determinebonds antechamber indigo ball mambax yuelbond bondnet mperformer; do
  uv run bond-bench run $m+project
done
for m in mambax yuelbond bondnet; do
  uv run bond-bench run $m+hybrid
done
```

Each method writes to `results/<method_tag>/`. Rerunning a method replaces its results for the strata it covers.

### 4. Score and plot

```bash
uv run bond-bench score --all-subgroups
uv sync --extra plot && uv run bond-bench plot
```

### 5. Publish

```bash
uv run python scripts/publish_results.py
```

This regenerates `results/summary.parquet`, packs `results/` into `build/results-release/` as `results.tar.gz`, `summary.parquet` and `SHA256SUMS`, and prints the `gh release create` command (or `gh release upload --clobber` to replace the assets of an existing release). Local paths in `provenance.json` are made relative to the repository. Runs made with `--n` are refused unless you pass `--allow-partial`.

### Notes

- **Sample size.** Without `--n`, every structure of every stratum is used. Add `--n 200 --seed 0` for a quick, deterministic subset; the same `--n` and `--seed` give the same structures for every method.
- **Parallelism and timeouts for adapters.** `--workers` and the per-structure `--timeout` apply only to the built-in methods. Adapters run one process per batch of 256 structures, and a batch times out after one hour. `antechamber` and `ball` additionally stop any single molecule after 600 s; `indigo` has no per-molecule limit.
- **Optional variants** are tagged separately in `results/`:
  - `yuelbond --param checkpoint=geom_kekulized_cdg`, trained with 0.2 Å noise (relevant to C1);
  - `ball --param strategy=a_star` or `strategy=ilp`;
  - `indigo --param algorithm=astar`;
  - `bondnet --param seed=43` or `seed=44`, other training seeds.
- **Known issues.** `+hybrid` returns charge-separated aromatic rings for `mambax`, `yuelbond` and `bondnet`, which lowers their scores (see `adapters/README.md`). Method-specific issues are in `adapters/<name>/README.md`.
