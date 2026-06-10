#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-vllm-omni-image}"
MICROMAMBA_BIN="${MAMBA_EXE:-micromamba}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-${RELEASE_DIR}/external/vllm-omni}"
VLLM_VERSION="${VLLM_VERSION:-0.14.0}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_31_x86_64.whl}"
BUILD_VLLM_FROM_SOURCE="${BUILD_VLLM_FROM_SOURCE:-0}"
MAX_JOBS="${MAX_JOBS:-16}"

if ! command -v "$MICROMAMBA_BIN" >/dev/null 2>&1; then
  echo "micromamba not found. Set MAMBA_EXE or install micromamba first." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv into the base user environment with pip."
  python -m pip install --user uv
fi

if "$MICROMAMBA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Using existing micromamba environment: $ENV_NAME"
else
  "$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -c conda-forge python=3.12 pip
fi

eval "$("$MICROMAMBA_BIN" shell hook -s bash)"
micromamba activate "$ENV_NAME"

python -m pip install --upgrade pip uv
uv pip install numpy

if [[ "$BUILD_VLLM_FROM_SOURCE" == "1" ]]; then
  export MAX_JOBS
  export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
  uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
else
  if uv pip install "$VLLM_WHEEL_URL" --torch-backend=auto; then
    :
  else
    cat >&2 <<EOF

Could not install the prebuilt vLLM wheel.

On older Linux distributions, uv may report that manylinux_2_31 is incompatible
with the host platform, for example manylinux_2_28. vLLM 0.14.0 does not publish
a manylinux_2_28 x86_64 CUDA wheel.

Use one of:
  1. Docker/Apptainer with vllm/vllm-omni:v0.14.0rc1
  2. Build vLLM from source with controlled parallelism:

     BUILD_VLLM_FROM_SOURCE=1 MAX_JOBS=16 bash scripts/setup_vllm_omni.sh

If source build fails again, lower MAX_JOBS to 8 or 4.
EOF
    exit 1
  fi
fi

if [[ -d "$VLLM_OMNI_DIR" ]]; then
  uv pip install -e "$VLLM_OMNI_DIR"
else
  git clone https://github.com/vllm-project/vllm-omni.git "$VLLM_OMNI_DIR"
  uv pip install -e "$VLLM_OMNI_DIR"
fi

echo "vLLM-Omni environment ready: $ENV_NAME"
echo "Activate with: micromamba activate $ENV_NAME"
echo "Check with: vllm --help"
