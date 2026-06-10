#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-train_configs/full_finetune_text_baseline.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_config_i2i.yaml}"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  train_flux_text_full_finetune.py \
  --config "$CONFIG"
