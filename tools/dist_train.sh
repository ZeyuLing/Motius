#!/usr/bin/env bash
# dist_train.sh — Launch distributed training with accelerate
# Usage: bash tools/dist_train.sh CONFIG [GPUS] [extra args]
# Example: bash tools/dist_train.sh configs/classification/vit_base_demo.py 8

CONFIG=$1
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    PYTHON_BIN=
fi

if [ -n "$PYTHON_BIN" ]; then
    DEFAULT_GPUS=$($PYTHON_BIN -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
else
    DEFAULT_GPUS=1
fi

GPUS=${2:-$DEFAULT_GPUS}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching with $((NNODES * GPUS)) processes ($NNODES nodes x $GPUS GPUs)"
echo "Config: $CONFIG"

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_machines=$NNODES \
    --num_processes=$((NNODES * GPUS)) \
    --machine_rank=$NODE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    "$(dirname "$0")/train.py" \
    "$CONFIG" \
    "${@:3}"
