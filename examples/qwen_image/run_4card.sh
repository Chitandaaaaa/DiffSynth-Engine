#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'

cd "$SCRIPT_DIR"
torchrun --nproc_per_node=4 run_4card.py "$@"
