# yuelbond adapter

Wraps YuelBond (Wang & Dokholyan 2026), from github.com/hust220/yuel_bond, pinned at commit `5737825`. The checkpoints come from Zenodo record 15353365. Method name: `yuelbond`. Choose a checkpoint with `--param checkpoint=`:

- `geom_kekulized_3d` (default): trained on clean geometries.
- `geom_kekulized_cdg`: trained with 0.2 Å coordinate noise.
- `geom_3d` and `geom_cdg`: trained on aromatic labels.

## Known issues

- **No licence.** The upstream repository has no licence file. `setup.sh` fetches the code and weights, and bond-bench redistributes neither. Clear any published results with the authors.
- **Heavy atoms only.** H is not in the model's element vocabulary. The adapter strips H before calling the model. It then bonds each H to its nearest heavy atom and reports that pair with P(single) = 1, which `+hybrid` needs on explicit-H queries.
- **Unknown elements are silently encoded as Cl.** Upstream maps any element outside C, N, O, F, P, S, Cl, Br, I to Cl. The method's capabilities therefore restrict queries to those elements plus H.
- **At most 127 heavy atoms.** Upstream stores edge indices as int8 and drops molecules with more than 150 atoms. Larger queries are reported as unsupported.
- **One molecule per forward pass.** Upstream's batching offsets edge indices by real atom counts while padding the node features, so batches of mixed size read the wrong atoms.
- **Fixed 3 Å candidate cutoff.** Upstream's `--distance_cutoff` option is parsed but never used.
- **Exotic classes are dropped.** Dative, ionic, hydrogen, three-centre and unspecified predictions are removed and flagged `exotic_bond_dropped`; upstream writes them as bond code 0.
- **Odd ring structures on unrelaxed geometries.** On RDKit ETKDG embeddings, rings sometimes come out chemically impossible, e.g. pyridine as `[C]1=[C][N][C]=C=[C]1`. The adapter's output matches upstream's own CLI bond for bond on these inputs, so this is how the model behaves.
- **No formal charges or H counts.** Use `+project` or `+hybrid`.
- **`+hybrid` returns charge-separated rings.** See [the shared issue](../README.md#issue-shared-by-the-probability-reporting-adapters).
- **Checkpoints are unpickled.** They are loaded with `torch.load(weights_only=False)`, after a checksum check.

## Verified

- Tested on macOS arm64, CPU.
- Output matches upstream's CLI on two molecules.
- With `+hybrid` on explicit H, the result was in the label's resonance class for all six test molecules.
- Ran through `bond-bench run` on 20 B2 structures.
- Not yet run on Linux or CUDA.
