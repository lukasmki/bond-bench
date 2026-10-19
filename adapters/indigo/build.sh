#!/usr/bin/env bash
# Build the pyindigox module into vendor/build (run inside the pixi env: `pixi run build`).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/vendor/indigo-bondorder
cmake -S "$src" -B "$here/vendor/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=OFF \
    -DPYTHON_EXECUTABLE="$(which python)" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_AR="$(command -v "${AR:-ar}")" -DCMAKE_RANLIB="$(command -v "${RANLIB:-ranlib}")"
cmake --build "$here/vendor/build" --target pyindigox -j "${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
