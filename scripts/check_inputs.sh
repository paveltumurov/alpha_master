#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

required=(
  "train_data.parquet"
  "test_data.parquet"
  "train_target.csv"
  "sample_submission (1).csv"
)

missing=0
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing: ${path}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Put the competition files in the repository root and rerun this script." >&2
  exit 1
fi

echo "All required input files are present."
