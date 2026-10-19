# mperformer adapter

Wraps MPerformer (Wang et al. 2023), from github.com/FanmengWang/MPerformer (GPL-3.0), pinned at commit `0e8d6ae`. The checkpoint is the authors' Google Drive file linked from their README (584 MB, fetched with `gdown`). Method name: `mperformer`.

## Known issues

- **Upstream charge-head bug.** `predict.py` asks for a classification head named `cache`, but the checkpoint's trained head is `total_data`. Because `infer.py` loads with `strict=False`, upstream's own pipeline silently runs an untrained formal-charge head. The adapter uses `total_data` and raises an error if any weight is missing or unexpected. As a result its formal charges differ from what `predict.py` would produce.
- **Heavy atoms only.** MPerformer always removes H. On explicit-H queries the adapter bonds each H to its nearest heavy atom.
- **H counts are reported.** Upstream computes them with its thresholds (0–3) but discards them. Here they become the implicit H counts of stripped queries. Counts above 3 cannot be represented.
- **Formal charges are limited to {−1, 0, +1}.** They use upstream's thresholds on the continuous charge head.
- **Total charge is unused.** Upstream parses `CHARGE=` from the xyz title line but never passes it to the model.
- **At most 256 heavy atoms.** Upstream randomly subsamples atoms above that, so larger queries are reported as unsupported.
- **No edge probabilities.** Decoding uses upstream's thresholds on continuous outputs; values between 1.25 and 1.75 mean aromatic. `+hybrid` therefore does not apply.
- **Precision depends on the device.** On CUDA the model runs in fp16, as upstream does. On CPU it runs in fp32, because upstream's forced `--fp16` would halve the model on CPU too.
- **Uni-Core is built without its fused CUDA kernels.** That is Uni-Core's default and it falls back to plain PyTorch. Building with `--enable-cuda-ext` on the cluster may be faster but has not been tried.
- **The checkpoint comes from Google Drive.** `gdown` can hit Drive's download quota. The file's SHA-256 is pinned in `setup.sh`.

## Verified

- Tested on macOS arm64, CPU.
- Every checkpoint weight loads.
- Bond orders were in the label's resonance class for all six test molecules with explicit H, and for five of six when stripped.
- Ran through `bond-bench run` on 20 B2 structures.
- Not yet run on Linux or CUDA.
