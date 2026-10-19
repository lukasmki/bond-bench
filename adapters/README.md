# Adapters

Each directory wraps one published bond-order method behind the JSONL protocol of `src/bond_bench/eval/methods/external.py`. The matching method classes are in `src/bond_bench/eval/methods/adapters.py`. Setup commands are in the main README. Each adapter's README lists its known issues and how far it has been tested.

| adapter | method | environment | upstream licence |
| --- | --- | --- | --- |
| [`openbabel`](openbabel/) | Open Babel bond perception | uv | GPL-2.0 |
| [`antechamber`](antechamber/README.md) | AmberTools antechamber / bondtype | pixi | GPL/LGPL mix |
| [`indigo`](indigo/README.md) | indigo-bondorder (Welsh & Allison) | pixi, C++ build | MIT |
| [`ball`](ball/README.md) | BALL AssignBondOrderProcessor (Dehof et al.) | pixi, C++ build, Linux | LGPL-2.1 |
| [`mambax`](mambax/README.md) | mambax random forest (Loschen) | uv | BSD-3 |
| [`yuelbond`](yuelbond/README.md) | YuelBond (Wang & Dokholyan) | uv, PyTorch | **none** |
| [`bondnet`](bondnet/README.md) | BondNet (Zhang & Zeng) | uv, PyTorch | MIT (weights CC BY 4.0) |
| [`mperformer`](mperformer/README.md) | MPerformer (Wang et al.) | uv, Uni-Core | GPL-3.0 |

## Conventions shared by all adapters

These are implemented in `_shared/bbadapter.py` and `_shared/fetch.sh`.

- **Upstream code and weights are fetched, not committed.** `setup.sh` checks out a pinned commit and downloads weights with SHA-256 checks into `vendor/` (gitignored).
- **Atom order is preserved.** Adapters whose model sees only heavy atoms (YuelBond, MPerformer, heavy-only BondNet) map indices back to the input order. On explicit-H queries they bond each H to its nearest heavy atom with a single bond.
- **Aromatic output is Kekulized with RDKit.** If Kekulization fails, aromatic bonds are reported as single and the prediction carries the flag `kekulize_failed`. `+project` can then repair it.
- **5-class outputs are mapped to 4 classes.** Where a model predicts {none, 1, 2, 3, aromatic}, `edge_probs` split P(aromatic) equally between orders 1 and 2.
- **No formal charges or H counts are invented.** An adapter reports them only when the wrapped method predicts them. Otherwise the raw output is invalid for charged or H-stripped structures, and `+project` or `+hybrid` must supply them.
- **Method parameters are forwarded as arguments.** For example, `--param checkpoint=...` reaches `run.py` as `key=value` after the input and output paths.

## Issue shared by the probability-reporting adapters

`mambax`, `yuelbond` and `bondnet` report `edge_probs` but no per-atom charge scores. Under `+hybrid`, with its default `charge_penalty=0`, charge-separated structures cost nothing. The constrained MAP then often returns charge-separated forms of aromatic rings, such as `[cH+]…[n-]`. The split P(aromatic) makes those forms tie with the neutral Kekulé structures.

On 20 B2 structures (explicit H), `bondnet+hybrid` was valid on every structure, but its per-bond F1 fell from 0.95 to 0.67 and its charge accuracy from 0.95 to 0.56. A nonzero `charge_penalty` on the `Hybrid` wrapper addresses this. However, `--param` currently reaches only the base method, so the penalty cannot yet be set from the command line.
