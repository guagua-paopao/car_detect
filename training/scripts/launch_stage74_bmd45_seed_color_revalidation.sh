#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE_ROOT=/root/autodl-tmp/vcas/code
SCRIPT="$CODE_ROOT/training/scripts/build_stage74_bmd45_seed_color_revalidation.py"
DATASET_ROOT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2
SOURCE="$DATASET_ROOT/bmd45-stage52-independent-color-v1/attribute_supplement.csv"
LABELS="$CODE_ROOT/config/vehicle_labels.v1.json"
RULE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE73-COLOR-RULE-VALIDATION-V1/report.json
PRODUCTION=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
STAGE71_COLOR=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/best.pt
OUTPUT_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-BMD45-SEED-COLOR-REVALIDATION-V1
MANIFEST="$OUTPUT_ROOT/attribute_manifest.bmd45-revalidated.train-only.csv"
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
  echo "Refusing to overwrite Stage74 evidence: $OUTPUT_ROOT" >&2
  exit 65
fi

assert_sha256 "$SCRIPT" 8B526E12802EB284FB6B9FEA81CF743FD7B9FE4C151DE06697A49DC5CEBCAA2A
assert_sha256 "$SOURCE" 6A9402A4010954D590CA474513C2007583AE07E430C152C74F0CDB6477AB573A
assert_sha256 "$LABELS" C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F
assert_sha256 "$RULE" BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F
assert_sha256 "$PRODUCTION" 6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383
assert_sha256 "$STAGE71_COLOR" 14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121

mkdir -p "$OUTPUT_ROOT"
set +e
"$PYTHON" "$SCRIPT" \
  --manifest "$SOURCE" \
  --expected-manifest-sha256 6A9402A4010954D590CA474513C2007583AE07E430C152C74F0CDB6477AB573A \
  --dataset-root "$DATASET_ROOT" \
  --labels "$LABELS" \
  --expected-labels-sha256 C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F \
  --validation-rule-report "$RULE" \
  --expected-validation-rule-sha256 BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F \
  --checkpoint "production=$PRODUCTION" \
  --checkpoint "stage71_color=$STAGE71_COLOR" \
  --output-manifest "$MANIFEST" \
  --output-report "$REPORT" \
  --near-duplicate-hamming 4 \
  --minimum-total 300 \
  --minimum-classes 5 \
  --minimum-rows-per-counted-class 20 \
  --maximum-per-class 2000 \
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

"$PYTHON" -c 'import json,sys; from pathlib import Path; rc=int(sys.argv[2]); status="pass_revalidated_auxiliary_available" if rc == 0 else ("fail_closed_insufficient_revalidated_auxiliary" if rc == 2 else "failed_runtime"); Path(sys.argv[1]).write_text(json.dumps({"status":status,"exit_code":rc,"report":sys.argv[3],"report_sha256":sys.argv[4],"manifest":sys.argv[5],"manifest_sha256":sys.argv[6],"old_pseudo_color_used_for_acceptance":False,"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"deployment_performed":False},indent=2)+"\n",encoding="utf-8")' "$STATE" "$rc" "$REPORT" "$report_sha" "$MANIFEST" "$manifest_sha"
exit "$rc"
