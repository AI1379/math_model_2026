#!/bin/bash
cd ~/math_model_2026/problems/A
mkdir -p output_lcdtc
export CUDA_VISIBLE_DEVICES=3
python3 -m lcdtc_reproduce.train \
    --epochs 150 --batch 4 --lr 0.001 \
    --workers 2 --img-size 640 \
    --eval-interval 10 \
    --output-dir ./output_lcdtc
