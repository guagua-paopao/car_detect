#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE_ROOT=/root/autodl-tmp/vcas/code
SCRIPT="$CODE_ROOT/training/scripts/build_stage73_uadetrac_color_pseudolabels_v2.py"
SOURCE=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2/attribute_manifest.training-only.csv
DATASET_ROOT=/root/autodl-tmp/vcas/datasets
LABELS="$CODE_ROOT/config/vehicle_labels.v1.json"
RULE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE73-COLOR-RULE-VALIDATION-V1/report.json
PRODUCTION=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
STAGE71_COLOR=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/best.pt
OUTPUT_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE73-UA-COLOR-CONSENSUS-V3
MANIFEST="$OUTPUT_ROOT/attribute_manifest.color-pseudo.train-only.csv"
REPORT="$OUTPUT_ROOT/report.json"
STATE="$OUTPUT_ROOT/state.json"
LOG="$OUTPUT_ROOT/run.log"

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print toupper($1)}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  fi
}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite train-only mining evidence: $OUTPUT_ROOT" >&2
  exit 65
fi

assert_sha256 "$SCRIPT" 13AF3F9809893C4901E12892A28CDFB7D1A06777CDE2184E0A06D4F4204DFAAA
assert_sha256 "$SOURCE" AB1F20AB1EF2E9DA0B26B48D49EFD0D4B9D324A22EC58DCDB2E102B371EBC369
assert_sha256 "$LABELS" C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F
assert_sha256 "$RULE" BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F
assert_sha256 "$PRODUCTION" 6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383
assert_sha256 "$STAGE71_COLOR" 14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121

mkdir -p "$OUTPUT_ROOT"
set +e
"$PYTHON" "$SCRIPT" \
  --manifest "$SOURCE" \
  --dataset-root "$DATASET_ROOT" \
  --labels "$LABELS" \
  --expected-labels-sha256 C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F \
  --validation-rule-report "$RULE" \
  --expected-validation-rule-sha256 BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F \
  --checkpoint "production=$PRODUCTION" \
  --checkpoint "stage71_color=$STAGE71_COLOR" \
  --output-manifest "$MANIFEST" \
  --output-report "$REPORT" \
  --minimum-track-frames 3 \
  --audit-frames-per-track 5 \
  --track-decision-agreement 0.60 \
  --output-frames-per-track 3 \
  --maximum-per-class 6000 \
  --minimum-total 500 \
  --minimum-classes 5 \
  --minimum-rows-per-counted-class 30 \
  --batch-size 128 \
  --workers 8 \
  --device cuda >"$LOG" 2>&1
rc=$?
set -e

report_sha=""
manifest_sha=""
if [[ -f "$REPORT" ]]; then
  report_sha="$(sha256sum "$REPORT" | awk '{print toupper($1)}')"
  printf '%s  %s\n' "$report_sha" "$(basename "$REPORT")" >"$REPORT.sha256"
fi
if [[ -f "$MANIFEST" ]]; then
  manifest_sha="$(sha256sum "$MANIFEST" | awk '{print toupper($1)}')"
  printf '%s  %s\n' "$manifest_sha" "$(basename "$MANIFEST")" >"$MANIFEST.sha256"
fi

"$PYTHON" -c 'import json,sys; from pathlib import Path; rc=int(sys.argv[2]); status="pass_train_only_manifest_available" if rc == 0 else ("fail_closed_insufficient_train_only_evidence" if rc == 2 else "failed_runtime"); Path(sys.argv[1]).write_text(json.dumps({"status":status,"exit_code":rc,"report":sys.argv[3],"report_sha256":sys.argv[4],"manifest":sys.argv[5],"manifest_sha256":sys.argv[6],"validation_images_or_truth_used_for_train_inference":False,"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"deployment_performed":False},indent=2)+"\n",encoding="utf-8")' "$STATE" "$rc" "$REPORT" "$report_sha" "$MANIFEST" "$manifest_sha"
exit "$rc"
