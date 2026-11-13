#!/usr/bin/env bash
set -euo pipefail

TARGET_REV=${1:-"data-v1.0"}
OUT_DIR=".cache/dataset_checkouts/${TARGET_REV}"
DATA_FILE="data/dataset.csv"
REPORT_FILE="reports/evidence/dvc_reproduce.txt"

mkdir -p "$OUT_DIR"
mkdir -p "reports/evidence"

echo "=== DVC Reproducible Dataset Checkout Demonstration ===" | tee "$REPORT_FILE"
echo "Target Revision: ${TARGET_REV}" | tee -a "$REPORT_FILE"
echo "Target File: ${DATA_FILE}" | tee -a "$REPORT_FILE"
echo "Checkout Directory: ${OUT_DIR}" | tee -a "$REPORT_FILE"
echo "Timestamp: $(date -u)" | tee -a "$REPORT_FILE"
echo "--------------------------------------------------------" | tee -a "$REPORT_FILE"

# 1. Fetch file using dvc get at specific Git revision
echo "[Step 1] Running: dvc get . ${DATA_FILE} --rev ${TARGET_REV} -o ${OUT_DIR}/dataset.csv" | tee -a "$REPORT_FILE"
dvc get . "${DATA_FILE}" --rev "${TARGET_REV}" -o "${OUT_DIR}/dataset.csv" --force

# 2. Calculate MD5 hash
if command -v md5 >/dev/null 2>&1; then
    ACTUAL_MD5=$(md5 -q "${OUT_DIR}/dataset.csv")
else
    ACTUAL_MD5=$(md5sum "${OUT_DIR}/dataset.csv" | awk '{print $1}')
fi
echo "[Step 2] Materialized File MD5: ${ACTUAL_MD5}" | tee -a "$REPORT_FILE"

# 3. Read expected MD5 from git revision .dvc file
EXPECTED_MD5=$(git show "${TARGET_REV}:${DATA_FILE}.dvc" | grep -A 2 "outs:" | grep "md5:" | awk '{print $NF}')
echo "[Step 3] Git revision ${TARGET_REV} recorded MD5: ${EXPECTED_MD5}" | tee -a "$REPORT_FILE"

# 4. Compare
echo "--------------------------------------------------------" | tee -a "$REPORT_FILE"
if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ]; then
    echo "SUCCESS: Checksums match perfectly! Verified reproducible dataset checkout." | tee -a "$REPORT_FILE"
else
    echo "FAILURE: Checksums do not match! Expected: ${EXPECTED_MD5}, Got: ${ACTUAL_MD5}" | tee -a "$REPORT_FILE"
    exit 1
fi
