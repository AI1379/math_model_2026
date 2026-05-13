#!/bin/bash

cd "$(dirname "$0")"

INPUT=${1:-./datasets/LCDTC/images/val2017}

python3 -m lcdtc_system.infer \
  --input "$INPUT" \
  --detector-ckpt ./output_lcdtc_detector/best.pth \
  --state-ckpt ./output_lcdtc_state/best.pth \
  --output-dir ./output_lcdtc_infer
