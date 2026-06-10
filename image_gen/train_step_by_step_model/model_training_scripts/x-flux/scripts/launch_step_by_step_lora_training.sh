#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-train_configs/step_by_step_lora_channel_concat.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_config_i2i.yaml}"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  train_step_by_step_flux_lora.py \
  --config "$CONFIG"
