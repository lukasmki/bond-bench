#!/usr/bin/env bash
# Fetch BALL (github.com/BALL-Project/ball, LGPL-2.1) at a pinned commit and build
# its assign_bond_orders utility in the pixi environment (Linux).
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../_shared/fetch.sh"
fetch_git https://github.com/BALL-Project/ball.git 54e7895fc6b64e50a71cadc2a18b915755f17bbf "$here/vendor/ball"
pixi run --manifest-path "$here/pixi.toml" build
