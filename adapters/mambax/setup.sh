#!/usr/bin/env bash
# Fetch mambax (CHLoschen/mamba, BSD-3) at a pinned commit; the pretrained clf.p is in the repo.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../_shared/fetch.sh"
fetch_git https://github.com/CHLoschen/mamba.git b6497378a918d031602a3960710b684b0231b7c7 "$here/vendor/mamba"
cd "$here" && uv sync --quiet
