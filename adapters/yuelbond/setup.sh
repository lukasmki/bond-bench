#!/usr/bin/env bash
# Fetch YuelBond (github.com/hust220/yuel_bond) at a pinned commit and its checkpoints
# (Zenodo 15353365). The repository has no licence file: nothing is redistributed by
# bond-bench, and results that depend on it should be cleared with the authors.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../_shared/fetch.sh"
fetch_git https://github.com/hust220/yuel_bond.git 5737825ae2ed8573ae740277c06c9f40d4bcdda6 "$here/vendor/yuel_bond"
zenodo=https://zenodo.org/records/15353365/files
models=$here/vendor/models
fetch_file $zenodo/geom_kekulized_3d.ckpt $models/geom_kekulized_3d.ckpt 9b091b8dc96b906c5e4d5076e1ae78449b5b4435bda54026f11e3de85607270a
fetch_file $zenodo/geom_kekulized_cdg.ckpt $models/geom_kekulized_cdg.ckpt af3cfd6ab7aae6db65a6a661fd519f05a50fa18f66f15b51308e62a512b683d1
fetch_file $zenodo/geom_3d.ckpt $models/geom_3d.ckpt 0bdca52eebc234d6908ed7a6a5150ce16f3bd5ec2bf13549592bba346d700cdc
fetch_file $zenodo/geom_cdg.ckpt $models/geom_cdg.ckpt 9f4c9ab7ce953345d148180236a38fabe3a9c726c63a9a079334c38f96ab1ebc
cd "$here" && uv sync --quiet
