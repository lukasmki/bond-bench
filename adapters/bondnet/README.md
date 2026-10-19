# bondnet adapter

Wraps BondNet (Zhang 2026), from github.com/zxc8848/BondNet (MIT), pinned at commit `0a7de99`. It uses the v4 joint checkpoints from Zenodo 10.5281/zenodo.23080342 (CC BY 4.0). Method name: `bondnet`. The observation picks the model: `joint_explicit_h` for explicit-H queries, `joint_heavy_only` for stripped ones. Choose a training seed with `--param seed=42|43|44`.

## Known issues

- **No upstream predictor.** BondNet ships only evaluation scripts. `run.py` follows the release's inference path (`evaluate.py`, `scripts/export_per_molecule_stats.py`):
  - the model is loaded with upstream's `evaluate.load_model`;
  - candidate pairs are built with a 3.0 Å envelope and masked to ≤ 2.5 Å;
  - a pair is bonded if either direction's connectivity sigmoid is ≥ 0.5;
  - bond types are the argmax of the direction-symmetrized logits.

  Upstream's optional post-processing flags (`--valence_decode`, ring rules, and so on) are off, as in the release results.
- **H–H pairs are never candidates.** H₂ cannot be represented.
- **One H can bond to two heavy atoms.** Connectivity is decided per pair, and the adapter does not repair this; the harness reports such outputs as invalid.
- **Trained on neutral molecules.** BondNet reads no total charge and predicts no formal charges or H counts. Use `+project` or `+hybrid` for charged or stripped structures.
- **Untrained elements are accepted silently.** The element embedding covers every atomic number, but only H, B, C, N, O, F, Si, P, S, Cl, As, Br, I, Hg and Bi were seen in training. Capabilities restrict queries to those.
- **torch is pinned to 2.5.1, as the authors used.** `load_model` calls `torch.load` with its default arguments. On torch ≥ 2.6 that default becomes `weights_only=True`.
- **New upstream repository.** It was created on 2026-09-30, so the adapter pins a commit and the release archive's checksum.
- **`+hybrid` returns charge-separated rings.** On 20 B2 structures, `bondnet+hybrid` dropped F1 from 0.95 to 0.67; see [the shared issue](../README.md#issue-shared-by-the-probability-reporting-adapters).

## Verified

- Tested on macOS arm64, CPU.
- With explicit H, bond orders were in the label's resonance class for all six test molecules.
- Ran through `bond-bench run` on 20 B2 structures: F1 0.95, valid 0.30. The invalid outputs lack formal charges.
- Not yet run on Linux or CUDA.
