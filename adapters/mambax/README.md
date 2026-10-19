# mambax adapter

Wraps the random-forest bond classifier of Loschen (2018), from github.com/CHLoschen/mamba (BSD-3), pinned at commit `b649737`. It uses the pretrained `clf.p` that ships in that repository. Method name: `mambax`.

## Known issues

- **Old scikit-learn pin.** `clf.p` was pickled by scikit-learn 0.23.2. scikit-learn ≥ 1.3 changed the tree node layout and refuses to load it. The environment is therefore pinned to Python 3.11, scikit-learn 1.2.2 and numpy < 2, which load it with version warnings. The adapter suppresses those warnings.
- **The authors' file path is used, not an in-process call.** Each query is written to a temporary xyz file and passed to `mambax.ml.predict_bonds`, as `mambax --predict` does. Calling `extract_features` with an RDKit Mol instead made the model predict "no bond" for every pair. The cause was not found, so that path is not used.
- **Stripped input is out of distribution.** The shipped model was trained with explicit H; its trees split on "partner is H". Stripped queries use the same model, as upstream's `--noH` option does.
- **No formal charges or H counts.** Raw output is invalid for charged structures and for every stripped query. Use `+project` or `+hybrid`.
- **No valence repair.** Each pair within 3.0 Å, including H–H pairs, gets the argmax class.
- **Elements are those of the shipped training set:** H, B, C, N, O, F, Si, P, S, Cl, Br, I. The model accepts any atomic number as a feature, but other elements are untrained.
- **Slow.** Featurization is an O(n²) Python loop. The median was about 1.5 s per B2 structure on CPU.
- **`+hybrid` returns charge-separated rings.** See [the shared issue](../README.md#issue-shared-by-the-probability-reporting-adapters).

## Verified

- Tested on macOS arm64.
- On six test molecules with explicit H, the bond orders were in the label's resonance class for all six.
- Ran through `bond-bench run` on 20 B2 structures.
- Not yet run on Linux.
