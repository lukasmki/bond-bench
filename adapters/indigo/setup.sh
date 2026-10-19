#!/usr/bin/env bash
# Fetch indigo-bondorder (github.com/allison-group/indigo-bondorder, MIT) at a pinned commit
# and build its Python module in the pixi environment.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../_shared/fetch.sh"
fetch_git https://github.com/allison-group/indigo-bondorder.git 77b157487fa4f30e60fbb47b2860ddc46e6c5f3d "$here/vendor/indigo-bondorder"
pixi run --manifest-path "$here/pixi.toml" build
