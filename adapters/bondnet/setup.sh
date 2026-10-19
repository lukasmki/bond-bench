#!/usr/bin/env bash
# Fetch BondNet (github.com/zxc8848/BondNet, MIT) at a pinned commit and the joint
# checkpoints of the v4 release (Zenodo 10.5281/zenodo.23080342, CC BY 4.0).
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../_shared/fetch.sh"
fetch_git https://github.com/zxc8848/BondNet.git 0a7de994010f5c93d14d4dcc9d29d19f57db06c7 "$here/vendor/BondNet"
zip=$here/vendor/checkpoints_v4_weights.zip
fetch_file https://zenodo.org/records/23080342/files/checkpoints_v4_weights.zip "$zip" \
    eff9595a9aa115c0fc5f8c01c72a0c8063a5082921b6fe5b5b5fcb455922b798
unzip -q -o "$zip" 'checkpoints/joint_explicit_h/*' 'checkpoints/joint_heavy_only/*' 'checkpoints/CHECKPOINT_HASHES.tsv' -d "$here/vendor"
cd "$here" && uv sync --quiet
