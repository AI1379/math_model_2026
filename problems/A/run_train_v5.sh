#!/bin/bash

cd "$(dirname "$0")"

GPU=${1:-0}
EPOCHS=${2:-35}
BACKBONE=${3:-resnet34}
BATCH=${4:-96}
IMAGE_SIZE=${5:-384}

export CUDA_VISIBLE_DEVICES=$GPU

echo "=== Liquid V5 ROI-first Training ==="
echo "GPU=$GPU  Epochs=$EPOCHS  Backbone=$BACKBONE  Batch=$BATCH  ImageSize=$IMAGE_SIZE"
echo "Output: ./output_liquid_v5/"
echo ""

python3 -m liquid_v5.train \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers 6 \
  --image-size "$IMAGE_SIZE" \
  --context 0.15 \
  --backbone "$BACKBONE" \
  --lr 2e-4 \
  --warmup-epochs 3 \
  --label-smoothing 0.03 \
  --neighbor-smoothing 0.08 \
  --half-class-boost 1.5 \
  --half-sampler-boost 1.5 \
  --output-dir ./output_liquid_v5

