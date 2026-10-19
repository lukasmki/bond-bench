#!/usr/bin/env bash
# Build BALL's assign_bond_orders utility into vendor/build (run inside the pixi env: `pixi run build`).
# GUI, Python bindings and optional components are switched off; Qt6 Core/Network/Xml are still required.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cmake -S "$here/vendor/ball" -B "$here/vendor/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_AR="$(command -v "${AR:-ar}")" -DCMAKE_RANLIB="$(command -v "${RANLIB:-ranlib}")" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBALL_HAS_VIEW=OFF -DBALL_PYTHON_SUPPORT=OFF -DUSE_QTWEBENGINE=OFF -DUSE_RTFACT=OFF \
    -DBALL_BUILD_BALLAXY=OFF -DBALL_BUILD_CWL=OFF -DUSE_TBB=OFF -DUSE_LIBSVM=OFF \
    -DBISON_EXECUTABLE="$CONDA_PREFIX/bin/bison" -DFLEX_EXECUTABLE="$CONDA_PREFIX/bin/flex"
cmake --build "$here/vendor/build" --target assign_bond_orders -j "${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
