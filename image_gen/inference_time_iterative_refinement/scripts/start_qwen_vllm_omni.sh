#!/usr/bin/env bash
set -euo pipefail

QWEN_IMAGE_PORT="${QWEN_IMAGE_PORT:-8091}"
QWEN_EDIT_PORT="${QWEN_EDIT_PORT:-8092}"

echo "Start these in separate terminals, or adapt this script for your scheduler:"
echo "vllm serve Qwen/Qwen-Image --omni --port ${QWEN_IMAGE_PORT}"
echo "vllm serve Qwen/Qwen-Image-Edit --omni --port ${QWEN_EDIT_PORT}"
echo
echo "Then export:"
echo "export QWEN_IMAGE_BASE_URL=http://localhost:${QWEN_IMAGE_PORT}/v1"
echo "export QWEN_IMAGE_EDIT_BASE_URL=http://localhost:${QWEN_EDIT_PORT}/v1"

