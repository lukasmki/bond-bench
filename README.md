# bond-bench

A benchmark dataset and evaluation harness for bond order assignment methods

## Features

- a comprehensive dataset for bond order assignment
- an efficient and unified harness for evaluation of bond order assignment methods

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
| `openbabel` | external | Open Babel `ConnectTheDots` + `PerceiveBondOrders`. Runs in its own GPL environment in `adapters/openbabel` (`cd adapters/openbabel && uv sync`). |
| `external` | external | Any program that speaks the JSONL protocol in `eval/methods/external.py`: `--param command='prog {input} {output}'`. |
| `+project` | wrapper | Maps an invalid output to the nearest valid structure on the same support. |
| `+hybrid` | wrapper | Algorithm H: constrained MAP over 𝒢 from the inner method's `edge_probs` (for ML models). |

The shared primitives P1–P6 of `assets/boa_canonical_algorithms.md` are in `src/bond_bench/eval/core/`:

- `elements.py`: state tables Λ
- `geometry.py`: the contact map
- `validity.py`: the validity oracle
- `ilp.py`: the CP-SAT decoder, with lexicographic stages, solution pools and ⊥ diagnosis
- `canon.py`: equivariant tie-breaking and resonance classes

To add a method, subclass `Method` (in `eval/methods/base.py`), declare its `Capabilities`, implement `predict`, and decorate the class with `@register`. A method with heavy or conflicting dependencies should instead get its own environment under `adapters/<name>/` and subclass `ExternalMethod`.

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
