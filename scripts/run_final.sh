#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY="${PY:-python}"
LOG_DIR="artifacts/run_logs"
mkdir -p "${LOG_DIR}" artifacts neural_artifacts

log_step() {
  echo "$(date -Is) $*" | tee -a "${LOG_DIR}/run_final.log"
}

bash scripts/check_inputs.sh

log_step "prepare sequence shards"
"${PY}" neural.py prepare --max-len 64 --partitions 32 2>&1 | tee "${LOG_DIR}/prepare_sequences.log"

log_step "train alfa gru seed777"
"${PY}" alfabank_gru.py all \
  --artifact-dir neural_artifacts \
  --run-name alfa_gru_seed777 \
  --seed 777 \
  2>&1 | tee "${LOG_DIR}/alfa_gru_seed777.log"

log_step "train id target cnn"
"${PY}" id_target_cnn.py all \
  --run-name id_target_cnn_local_seed111 \
  --seed 111 \
  2>&1 | tee "${LOG_DIR}/id_target_cnn.log"

cat <<'NOTE'

The full final submission also uses additional GRU/TCN/pretrained runs and
their validation predictions. See HISTORY.md and README.md for the exact final
stacking composition.

After all required intermediate predictions exist in artifacts/, run:

  python final_stacking.py
  python conservative_blends.py

Then copy:

  artifacts/submission_conservative_70_30.csv -> submission.csv

NOTE

log_step "documented pipeline finished"
