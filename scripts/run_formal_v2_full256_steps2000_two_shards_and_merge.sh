#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-mpp-meta}"
DEVICE="${DEVICE:-cuda}"
ARCH_ARTIFACT_DIR="${ARCH_ARTIFACT_DIR:-artifacts/formal_v2_256}"
ARCH_DEPTH="${ARCH_DEPTH:-8}"
ARCH_SAMPLE_SIZE="${ARCH_SAMPLE_SIZE:-256}"
ARCH_SAMPLE_SEED="${ARCH_SAMPLE_SEED:-25}"
ARCH_CSV="${ARCH_CSV:-${ARCH_ARTIFACT_DIR}/sampled_architectures_L${ARCH_DEPTH}_n${ARCH_SAMPLE_SIZE}_seed${ARCH_SAMPLE_SEED}.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/toy_diffusion/meta_model_family}"
BASE_FAMILY_NAME="${BASE_FAMILY_NAME:-formal_v2_full256_steps2000}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
LOG_EVERY="${LOG_EVERY:-20}"
VAL_EVERY="${VAL_EVERY:-50}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LOG_DIR="${LOG_DIR:-logs/formal_v2_full256_steps2000}"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
else
  echo "[run_formal_v2_full256_steps2000_two_shards_and_merge] missing /root/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

cd "${ROOT_DIR}"
conda activate "${ENV_NAME}"

if [[ ! -f "${ARCH_CSV}" ]]; then
  echo "[launcher] architecture csv missing, generating full formal_v2 sample"
  python scripts/enumerate_architectures.py \
    --depth "${ARCH_DEPTH}" \
    --sample_size "${ARCH_SAMPLE_SIZE}" \
    --sample_seed "${ARCH_SAMPLE_SEED}" \
    --output_dir "${ARCH_ARTIFACT_DIR}"
fi

shard_a_name="${BASE_FAMILY_NAME}_a"
shard_b_name="${BASE_FAMILY_NAME}_b"
shard_a_min=0
shard_a_max=127
shard_b_min=128
shard_b_max=255
family_a_dir="${OUTPUT_DIR}/${shard_a_name}"
family_b_dir="${OUTPUT_DIR}/${shard_b_name}"
merged_csv="${OUTPUT_DIR}/${BASE_FAMILY_NAME}_meta_dataset.csv"

mkdir -p "${LOG_DIR}"
pids=()

launch_shard() {
  local shard_label="$1"
  local arch_min="$2"
  local arch_max="$3"
  local family_name="$4"
  local log_file="${LOG_DIR}/${shard_label}.log"

  echo "[launcher] start ${shard_label} arch=${arch_min}-${arch_max} family=${family_name}" >&2
  (
    ARCH_ID_MIN="${arch_min}" \
    ARCH_ID_MAX="${arch_max}" \
    FAMILY_NAME="${family_name}" \
    ARCH_CSV="${ARCH_CSV}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    MAX_STEPS="${MAX_STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    LOG_EVERY="${LOG_EVERY}" \
    VAL_EVERY="${VAL_EVERY}" \
    LR="${LR}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
      bash scripts/server_launch_family_shard.sh
  ) >"${log_file}" 2>&1 &
  LAUNCHED_PID=$!
}

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap cleanup INT TERM

launch_shard shard_a "${shard_a_min}" "${shard_a_max}" "${shard_a_name}"
pids+=("${LAUNCHED_PID}")
launch_shard shard_b "${shard_b_min}" "${shard_b_max}" "${shard_b_name}"
pids+=("${LAUNCHED_PID}")

echo "[launcher] shard_a_pid=${pids[0]} shard_b_pid=${pids[1]}"
echo "[launcher] logs=${LOG_DIR}"
echo "[launcher] arch_csv=${ARCH_CSV}"
echo "[launcher] merged_csv=${merged_csv}"

set +e
wait "${pids[0]}"
status_a=$?
wait "${pids[1]}"
status_b=$?
set -e

if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
  echo "[launcher] one or more shards failed: shard_a=${status_a} shard_b=${status_b}"
  echo "[launcher] inspect logs under ${LOG_DIR}"
  exit 1
fi

echo "[launcher] both shards finished, merging dataset"
python scripts/build_meta_dataset.py \
  --family_dir \
  "${family_a_dir}" \
  "${family_b_dir}" \
  --family_name "${BASE_FAMILY_NAME}" \
  --output_csv "${merged_csv}"

wc -l "${family_a_dir}/family_summary.csv"
wc -l "${family_b_dir}/family_summary.csv"
wc -l "${merged_csv}"

echo "[launcher] done merged_csv=${merged_csv}"
