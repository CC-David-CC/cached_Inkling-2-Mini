#!/usr/bin/env bash
set -euo pipefail

inkling_dir=${INKLING_DIR:?Set INKLING_DIR to the engine and checkpoint directory}
exec "$inkling_dir/strata-inkling" --bench \
    --native "$inkling_dir/Inkling-2-Mini-SQ-20.strata" \
    --gpus "${INKLING_GPUS:-1}" \
    --max-context 8192 \
    --kv fp16 \
    --draft off \
    --prompt-len 512 \
    --gen 128
