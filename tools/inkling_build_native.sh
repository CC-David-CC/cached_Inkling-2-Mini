#!/usr/bin/env bash
set -euo pipefail

runtime=${1:?Usage: inkling_build_native.sh RUNTIME_DIRECTORY}
source_dir="$runtime/strata-source"
revision=b28ca95e3e6847f2d5e317994f253b66716deeaa

if [[ $(git -C "$source_dir" rev-parse HEAD) != "$revision" ]]; then
    printf 'Expected Strata source revision %s\n' "$revision" >&2
    exit 1
fi

cmake -S "$source_dir/engine/src/inkling2" -B "$runtime/native-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_INK_CUDA_ARCHS="${INKLING_CUDA_ARCHS:-120}" \
    -DCMAKE_CUDA_COMPILER="${INKLING_NVCC:-/usr/local/cuda/bin/nvcc}"
cmake --build "$runtime/native-build" --target strata-inkling \
    --parallel "${INKLING_BUILD_JOBS:-6}"
