#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-iter-refine-data-gen}"
MICROMAMBA_BIN="${MAMBA_EXE:-micromamba}"

if ! command -v "$MICROMAMBA_BIN" >/dev/null 2>&1; then
  echo "micromamba not found. Set MAMBA_EXE or install micromamba first." >&2
  exit 1
fi

if "$MICROMAMBA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Using existing micromamba environment: $ENV_NAME"
else
  "$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
fi

eval "$("$MICROMAMBA_BIN" shell hook -s bash)"
micromamba activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -d Grounded_SAM2 ]]; then
  echo "Grounded_SAM2 was not found in this directory."
  echo "Clone https://github.com/IDEA-Research/Grounded-SAM-2 as Grounded_SAM2 and download its SAM2/GroundingDINO checkpoints before running Step 1."
fi

echo "Environment ready: $ENV_NAME"
echo "Activate with: micromamba activate $ENV_NAME"
