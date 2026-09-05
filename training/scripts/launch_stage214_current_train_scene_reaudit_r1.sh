#!/usr/bin/env bash
set -euo pipefail

union_script=/root/autodl-tmp/vcas/scripts/build_stage214_current_train_scene_union.py
audit_script=/root/autodl-tmp/vcas/scripts/audit_stage214_current_train_scenes.py
supervised=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage211-inatrc-toll-hardclass-r2/attribute_manifest.stage211-inatrc-toll-hardclass.csv
unlabeled=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage211-inatrc-toll-hardclass-r2/attribute_manifest.stage211-unlabeled.csv
labels=/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json
union_root=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage214-current-train-scene-union-r1
audit_root=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE214-CURRENT-TRAIN-SCENE-REAUDIT-R1
minimum_free_bytes=$((12 * 1024 * 1024 * 1024))

[[ -f "${union_script}" && ! -L "${union_script}" ]]
[[ -f "${audit_script}" && ! -L "${audit_script}" ]]
[[ -f "${supervised}" && ! -L "${supervised}" ]]
[[ -f "${unlabeled}" && ! -L "${unlabeled}" ]]
[[ -f "${labels}" && ! -L "${labels}" ]]
[[ ! -e "${union_root}" ]]
[[ ! -e "${audit_root}" ]]

available_bytes=$(df -PB1 /root/autodl-tmp | awk 'NR == 2 {print $4}')
[[ "${available_bytes}" =~ ^[0-9]+$ ]]
if (( available_bytes < minimum_free_bytes )); then
  echo "insufficient disk space for Stage214: available=${available_bytes}, required=${minimum_free_bytes}" >&2
  exit 20
fi

/root/miniconda3/bin/python "${union_script}" \
  --supervised-manifest "${supervised}" \
  --expected-supervised-sha256 a6344c55b2a6ce98533e0234f7cfd0699224ae2c1d44609067eced907dc3b69f \
  --unlabeled-manifest "${unlabeled}" \
  --expected-unlabeled-sha256 ffd93dafa0898e9e8652a6bb90fa8a68fd4287c55ce6877455009f06d5db20b6 \
  --allowed-root /root/autodl-tmp/vcas \
  --output-root "${union_root}"

union_manifest="${union_root}/attribute_manifest.stage214-current-train-scene-union.csv"
union_sha=$(sha256sum "${union_manifest}" | awk '{print $1}')
labels_sha=$(sha256sum "${labels}" | awk '{print $1}')

/root/miniconda3/bin/python "${audit_script}" \
  --manifest "${union_manifest}" \
  --expected-manifest-sha256 "${union_sha}" \
  --labels "${labels}" \
  --expected-labels-sha256 "${labels_sha}" \
  --allowed-root /root/autodl-tmp/vcas \
  --output-root "${audit_root}" \
  --workers 10 \
  --batch-size 4096 \
  --near-duplicate-hamming 4 \
  --max-effective-frames-per-track 5 \
  --include-all-train
