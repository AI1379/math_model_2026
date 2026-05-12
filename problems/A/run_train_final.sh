#!/bin/bash
# LCD-YOLOX Training Script (Final Version)
#
# v1 → v2 → final 的关键修复:
#   1. NMS: 评估时按内容类别做 NMS（v1 缺失 → 大量假阳性）
#   2. Mosaic + flip + color 增强（v1 无增强 → 过拟合）
#   3. 5 epoch 线性 warmup + cosine schedule
#   4. GIoU 回归损失（v1 用 L1）
#   5. EMA 指数移动平均
#   6. conf_thresh 从 0.05 降到 0.001
#
# 论文目标 (LCD-YOLOX, CrossFormer-S):
#   mAPct=0.548, APc@0.5=0.809, APt@0.5=0.607
#
# 依赖: pip install torch torchvision opencv-python numpy
#
# 用法:
#   bash run_train_final.sh          # 默认参数
#   bash run_train_final.sh 8 0.01  # 指定 batch_size 和 lr

cd "$(dirname "$0")"

# 参数
BATCH=${1:-16}
LR=${2:-0.01}
EPOCHS=${3:-300}
GPU=${4:-0}

export CUDA_VISIBLE_DEVICES=$GPU

echo "=== LCD-YOLOX Final Training ==="
echo "GPU=$GPU  Batch=$BATCH  LR=$LR  Epochs=$EPOCHS"
echo "Output: ./output_lcdtc_final/"
echo ""

python3 -m lcdtc_reproduce.train_final \
    --epochs $EPOCHS \
    --batch $BATCH \
    --lr $LR \
    --workers 4 \
    --img-size 640 \
    --eval-interval 10 \
    --print-interval 5 \
    --output-dir ./output_lcdtc_final
