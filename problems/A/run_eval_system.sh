#!/bin/bash

cd "$(dirname "$0")"

python3 -m lcdtc_system.eval_pipeline \
  --detector-ckpt ./output_lcdtc_detector/best.pth \
  --state-ckpt ./output_lcdtc_state/best.pth
