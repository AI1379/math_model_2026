#!/bin/bash

cd "$(dirname "$0")"

GPU=${1:-0}
EPOCHS=${2:-20}

export CUDA_VISIBLE_DEVICES=$GPU

python3 -m lcdtc_system.train_detector \
  --epochs "$EPOCHS" \
  --batch 6 \
  --workers 4 \
  --image-size 800 \
  --lr 2e-4 \
  --amp \
  --output-dir ./output_lcdtc_detector
