#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-mpp-meta}"
DEVICE="${DEVICE:-cuda}"
PIPELINE_NAME="${PIPELINE_NAME:-server_quickstart}"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Standard location already validated on the target server.
  source /root/miniconda3/etc/profile.d/conda.sh
else
  echo "[server_quickstart] missing /root/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

conda activate "${ENV_NAME}"
cd "${ROOT_DIR}"

echo "[server_quickstart] root=${ROOT_DIR}"
echo "[server_quickstart] env=${ENV_NAME}"
echo "[server_quickstart] device=${DEVICE}"

python scripts/check_environment.py
python scripts/meta_model_cli.py smoke-pipeline \
  --device "${DEVICE}" \
  --sample_size 4 \
  --max_architectures 2 \
  --smoke_steps 2 \
  --family_steps 2 \
  --batch_size 2 \
  --val_batch_size 2 \
  --log_every 1 \
  --val_every 1 \
  --pipeline_name "${PIPELINE_NAME}"

