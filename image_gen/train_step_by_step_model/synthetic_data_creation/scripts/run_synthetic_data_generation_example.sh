#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-outputs/synthetic_flux_qwen}"
BATCH_NAME="${BATCH_NAME:-dataset_512_sbatch0}"
SOURCE_DIR="${DATA_ROOT}/${BATCH_NAME}"
COLLATED_DIR="${DATA_ROOT}/collated_train_pairs"
QWEN_EDIT_BASE_URL="${QWEN_EDIT_BASE_URL:-http://localhost:8092/v1}"

mkdir -p "$DATA_ROOT"

python generate_flux_images_and_detect_objects.py \
  --cuda 0 \
  --dataset_folder_name "$SOURCE_DIR" \
  --max_num_rels 4 \
  --num_samples_to_generate_per_k 10 \
  --verbose False

python filter_and_relabel_with_vqa.py \
  --input_dir "$SOURCE_DIR" \
  --verbose False \
  --skip_if_item_not_detected True \
  --use_spatial_relations_if_norelabel False

python remove_objects_with_qwen_edit.py \
  --source_folder "$SOURCE_DIR" \
  --qwen-edit-base-url "$QWEN_EDIT_BASE_URL" \
  --max_items_to_select 2

python collate_step_by_step_training_pairs.py \
  --input_primary_dir "$DATA_ROOT" \
  --output_dir "$COLLATED_DIR"

echo "Collated training pairs written to: $COLLATED_DIR"
