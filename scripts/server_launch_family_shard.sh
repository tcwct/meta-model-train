#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-mpp-meta}"
DEVICE="${DEVICE:-cuda}"
ARCH_CSV="${ARCH_CSV:-artifacts/full328_seed25/sampled_architectures_L6_n328_seed25.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/toy_diffusion/meta_model_family}"
FAMILY_NAME="${FAMILY_NAME:-server_v1_full328}"
ARCH_ID_MIN="${ARCH_ID_MIN:-0}"
ARCH_ID_MAX="${ARCH_ID_MAX:-163}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
VAL_EVERY="${VAL_EVERY:-50}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
else
  echo "[server_launch_family_shard] missing /root/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

cd "${ROOT_DIR}"
conda activate "${ENV_NAME}"

echo "[server_launch_family_shard] root=${ROOT_DIR}"
echo "[server_launch_family_shard] env=${ENV_NAME}"
echo "[server_launch_family_shard] device=${DEVICE}"
echo "[server_launch_family_shard] architecture_csv=${ARCH_CSV}"
echo "[server_launch_family_shard] family_name=${FAMILY_NAME}"
echo "[server_launch_family_shard] shard=${ARCH_ID_MIN}-${ARCH_ID_MAX}"
echo "[server_launch_family_shard] batch_size=${BATCH_SIZE}"
echo "[server_launch_family_shard] val_batch_size=${VAL_BATCH_SIZE}"
echo "[server_launch_family_shard] max_steps=${MAX_STEPS}"

python scripts/train_minimal_arch_family.py \
  --architecture_csv "${ARCH_CSV}" \
  --architecture_id_min "${ARCH_ID_MIN}" \
  --architecture_id_max "${ARCH_ID_MAX}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  --family_name "${FAMILY_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --log_every "${LOG_EVERY}" \
  --val_every "${VAL_EVERY}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --skip_existing \
  ${EXTRA_ARGS}
