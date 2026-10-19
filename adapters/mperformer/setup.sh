#!/usr/bin/env bash
# Fetch MPerformer (github.com/FanmengWang/MPerformer, GPL-3.0) at its only commit and the
# authors' checkpoint (Google Drive file linked from the README, 584 MB).
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../_shared/fetch.sh"
fetch_git https://github.com/FanmengWang/MPerformer.git 0e8d6ae61edc8eb0f342b968f160e8168043fdb6 "$here/vendor/MPerformer"
ckpt=$here/vendor/weight/checkpoint.pt
if [ ! -s "$ckpt" ]; then
    mkdir -p "$(dirname "$ckpt")"
    uvx --quiet gdown 1sHWm1xOy0I8_R50dPANfMUXoRQkoPCBJ -O "$ckpt"
fi
fetch_file unused "$ckpt" 7be90a592d747795a701f5f7533872a83247c16975d55068e78f410f251ebbd8
cd "$here" && uv sync --quiet
