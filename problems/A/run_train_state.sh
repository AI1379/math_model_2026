#!/bin/bash

cd "$(dirname "$0")"

GPU=${1:-0}
EPOCHS=${2:-25}

export CUDA_VISIBLE_DEVICES=$GPU

python3 -m lcdtc_system.train_state \
  --epochs "$EPOCHS" \
  --batch 96 \
  --workers 4 \
  --image-size 320 \
  --backbone resnet18 \
  --lr 3e-4 \
  --amp \
  --output-dir ./output_lcdtc_state
