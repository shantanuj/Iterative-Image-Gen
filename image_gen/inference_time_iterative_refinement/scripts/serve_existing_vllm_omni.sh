#!/usr/bin/env bash
set -euo pipefail

# Launch from an existing vLLM-Omni environment. The iterative runner can stay
# in the lightweight iterative-image-gen env.

MODEL="${1:-Qwen/Qwen-Image}"
PORT="${2:-8091}"

if [[ -n "${VLLM_OMNI_BIN:-}" ]]; then
  VLLM_OMNI_CMD="$VLLM_OMNI_BIN"
elif command -v vllm-omni >/dev/null 2>&1; then
  VLLM_OMNI_CMD="vllm-omni"
elif [[ -n "${VLLM_OMNI_ENV_DIR:-}" && -x "${VLLM_OMNI_ENV_DIR}/bin/vllm-omni" ]]; then
  VLLM_OMNI_CMD="${VLLM_OMNI_ENV_DIR}/bin/vllm-omni"
  export LD_LIBRARY_PATH="${VLLM_OMNI_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
else
  echo "Could not find vllm-omni. Activate a vLLM-Omni environment, put vllm-omni on PATH, or set VLLM_OMNI_BIN/VLLM_OMNI_ENV_DIR." >&2
  exit 1
fi

PYTHONPATH_ENTRIES=()
if [[ -n "${VLLM_SRC:-}" ]]; then
  PYTHONPATH_ENTRIES+=("$VLLM_SRC")
fi
if [[ -n "${VLLM_OMNI_SRC:-}" ]]; then
  PYTHONPATH_ENTRIES+=("$VLLM_OMNI_SRC")
fi
if [[ ${#PYTHONPATH_ENTRIES[@]} -gt 0 ]]; then
  IFS=:
  export PYTHONPATH="${PYTHONPATH_ENTRIES[*]}:${PYTHONPATH:-}"
  unset IFS
fi

exec "$VLLM_OMNI_CMD" serve "${MODEL}" --omni --port "${PORT}"
