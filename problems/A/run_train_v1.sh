#!/bin/bash

cd "$(dirname "$0")"

GPU=${1:-0}
EPOCHS=${2:-40}

export CUDA_VISIBLE_DEVICES=$GPU

echo "=== Bottle Liquid V1 Training ==="
echo "GPU=$GPU  Epochs=$EPOCHS"
echo "Output: ./output_liquid_v1/"
echo ""

python3 -m liquid_v1.train \
  --epochs "$EPOCHS" \
  --workers 4 \
  --lcd-batch 48 \
  --seg-batch 8 \
  --lcd-image-size 320 \
  --seg-image-size 384 \
  --backbone resnet34 \
  --decoder-dim 192 \
  --lr 3e-4 \
  --amp \
  --output-dir ./output_liquid_v1
