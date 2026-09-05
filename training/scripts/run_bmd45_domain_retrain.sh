#!/usr/bin/env bash
set -euo pipefail

VCAS_ROOT="${VCAS_ROOT:-/root/autodl-tmp/vcas}"
CODE_ROOT="${CODE_ROOT:-${VCAS_ROOT}/code-20260811-domain-v1}"
DATASET_ROOT="${DATASET_ROOT:-${VCAS_ROOT}/datasets/dataset-domain-bmd45-v1/detection}"
RUNS_ROOT="${RUNS_ROOT:-${VCAS_ROOT}/runs/detection}"
EVAL_ROOT="${EVAL_ROOT:-${VCAS_ROOT}/manifests/domain-retrain-v1/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASELINE_MODEL="${BASELINE_MODEL:-${RUNS_ROOT}/DET-LARGE-ONTOLOGY-R3-FIX1/weights/best.pt}"
WARMUP_NAME="DET-DOMAIN-BMD45-WARMUP-R1"
FINETUNE_NAME="DET-DOMAIN-BMD45-FINETUNE-R1"
DATASET_VERSION="dataset-domain-bmd45-v1"
CODE_REVISION="${CODE_REVISION:-domain-bmd45-v1}"

mkdir -p "${RUNS_ROOT}" "${EVAL_ROOT}"
cd "${CODE_ROOT}"

evaluate_if_missing() {
  local model="$1"
  local name="$2"
  local output="$3"
  if [[ -s "${output}" ]]; then
    echo "SKIP: evaluation exists: ${output}"
    return
  fi
  "${PYTHON_BIN}" training/scripts/evaluate_detector_domain.py \
    --model "${model}" \
    --data "${DATASET_ROOT}/bmd45_val.yaml" \
    --split test \
    --imgsz 960 \
    --batch 16 \
    --device 0 \
    --project "${RUNS_ROOT}" \
    --name "${name}" \
    --output "${output}"
}

run_warmup() {
  local run_dir="${RUNS_ROOT}/${WARMUP_NAME}"
  if [[ -s "${run_dir}/run_summary.json" ]]; then
    echo "SKIP: warmup complete"
  elif [[ -s "${run_dir}/weights/last.pt" ]]; then
    "${PYTHON_BIN}" training/scripts/train_detector.py \
      --data "${DATASET_ROOT}/vehicle_det_v1.yaml" \
      --project "${RUNS_ROOT}" --name "${WARMUP_NAME}" \
      --resume "${run_dir}/weights/last.pt" \
      --imgsz 960 --epochs 5 --batch -1 --device 0 --workers 8 \
      --run-kind formal --dataset-version "${DATASET_VERSION}" \
      --code-revision "${CODE_REVISION}"
  else
    "${PYTHON_BIN}" training/scripts/train_detector.py \
      --data "${DATASET_ROOT}/vehicle_det_v1.yaml" \
      --model "${BASELINE_MODEL}" \
      --imgsz 960 --epochs 5 --batch -1 --device 0 --workers 8 \
      --patience 5 --optimizer AdamW --lr0 0.001 --lrf 0.10 \
      --weight-decay 0.0005 --warmup-epochs 1 --freeze 10 \
      --close-mosaic 2 --cos-lr --seed 20260811 \
      --project "${RUNS_ROOT}" --name "${WARMUP_NAME}" --save-period 1 \
      --run-kind formal --dataset-version "${DATASET_VERSION}" \
      --code-revision "${CODE_REVISION}"
  fi
}

run_finetune() {
  local run_dir="${RUNS_ROOT}/${FINETUNE_NAME}"
  local warmup_best="${RUNS_ROOT}/${WARMUP_NAME}/weights/best.pt"
  if [[ -s "${run_dir}/run_summary.json" ]]; then
    echo "SKIP: fine-tune complete"
  elif [[ -s "${run_dir}/weights/last.pt" ]]; then
    "${PYTHON_BIN}" training/scripts/train_detector.py \
      --data "${DATASET_ROOT}/vehicle_det_v1.yaml" \
      --project "${RUNS_ROOT}" --name "${FINETUNE_NAME}" \
      --resume "${run_dir}/weights/last.pt" \
      --imgsz 960 --epochs 40 --batch -1 --device 0 --workers 8 \
      --run-kind formal --dataset-version "${DATASET_VERSION}" \
      --code-revision "${CODE_REVISION}"
  else
    "${PYTHON_BIN}" training/scripts/train_detector.py \
      --data "${DATASET_ROOT}/vehicle_det_v1.yaml" \
      --model "${warmup_best}" \
      --imgsz 960 --epochs 40 --batch -1 --device 0 --workers 8 \
      --patience 10 --optimizer AdamW --lr0 0.001 --lrf 0.05 \
      --weight-decay 0.0005 --warmup-epochs 2 --close-mosaic 8 \
      --cos-lr --seed 20260811 \
      --project "${RUNS_ROOT}" --name "${FINETUNE_NAME}" --save-period 2 \
      --run-kind formal --dataset-version "${DATASET_VERSION}" \
      --code-revision "${CODE_REVISION}"
  fi
}

evaluate_if_missing \
  "${BASELINE_MODEL}" \
  "DET-DOMAIN-BMD45-BASELINE-EVAL" \
  "${EVAL_ROOT}/baseline-bmd45.json"
run_warmup
run_finetune
evaluate_if_missing \
  "${RUNS_ROOT}/${FINETUNE_NAME}/weights/best.pt" \
  "${FINETUNE_NAME}-BMD45-EVAL" \
  "${EVAL_ROOT}/finetune-bmd45.json"

echo "PASS: BMD-45 domain retraining and held-out evaluation completed"
