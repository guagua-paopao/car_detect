#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE_ROOT=/root/autodl-tmp/vcas/code
DATA_ROOT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2
SCRIPT="$CODE_ROOT/training/scripts/build_stage74_joint_color_manifest.py"
BASE="$DATA_ROOT/stage71-teacher-manifests-v5/attribute_manifest.stage71-color-teacher.csv"
UA_ROOT="$DATA_ROOT/stage70-uadetrac-training-pool-v2"
UA_MANIFEST=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE73-UA-COLOR-CONSENSUS-V4/attribute_manifest.color-pseudo.train-only.csv
UA_REPORT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE73-UA-COLOR-CONSENSUS-V4/report.json
BMD_ROOT="$DATA_ROOT"
BMD_MANIFEST=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-BMD45-SEED-COLOR-REVALIDATION-V1/attribute_manifest.bmd45-revalidated.train-only.csv
BMD_REPORT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE74-BMD45-SEED-COLOR-REVALIDATION-V1/report.json
OUTPUT_ROOT="$DATA_ROOT/stage74-joint-color-v2"
MANIFEST="$OUTPUT_ROOT/attribute_manifest.stage74-joint-color.csv"
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
  echo "Refusing to overwrite Stage74 V2 joint evidence: $OUTPUT_ROOT" >&2
  exit 65
fi

assert_sha256 "$SCRIPT" BEA0E347948F8A8E7610E970F2C105AE47092D8AEEFC8402AC8D9A766B698D26
assert_sha256 "$BASE" F4576A85019C99A3E287692AEEBC1F470E21113E26A58DB264A2703F502D8A63
assert_sha256 "$UA_MANIFEST" B89DDB21C8649A70D085ECAD1811CF4206618B41A453C51ED138B833F6C74E58
assert_sha256 "$UA_REPORT" 9184ADA81444F7737FC305E9F4CCD9FDE04FE83C25B2F6603A8EDDD59CB0190D
assert_sha256 "$BMD_MANIFEST" CFF3AD598A7AB78155A06CCF7BB87408E97347A971A6A4BF07CA93760721ED6A
assert_sha256 "$BMD_REPORT" EF3EC1BA288F5023F9CCF40A69A0F615A5B4CEE7863FE9545FD3670D8BA1E7B5

mkdir -p "$OUTPUT_ROOT"
set +e
"$PYTHON" "$SCRIPT" \
  --base-manifest "$BASE" \
  --expected-base-sha256 F4576A85019C99A3E287692AEEBC1F470E21113E26A58DB264A2703F502D8A63 \
  --ua-manifest "$UA_MANIFEST" \
  --expected-ua-manifest-sha256 B89DDB21C8649A70D085ECAD1811CF4206618B41A453C51ED138B833F6C74E58 \
  --ua-report "$UA_REPORT" \
  --expected-ua-report-sha256 9184ADA81444F7737FC305E9F4CCD9FDE04FE83C25B2F6603A8EDDD59CB0190D \
  --ua-root "$UA_ROOT" \
  --bmd-manifest "$BMD_MANIFEST" \
  --expected-bmd-manifest-sha256 CFF3AD598A7AB78155A06CCF7BB87408E97347A971A6A4BF07CA93760721ED6A \
  --bmd-report "$BMD_REPORT" \
  --expected-bmd-report-sha256 EF3EC1BA288F5023F9CCF40A69A0F615A5B4CEE7863FE9545FD3670D8BA1E7B5 \
  --bmd-root "$BMD_ROOT" \
  --expected-validation-rule-sha256 BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F \
  --output-manifest "$MANIFEST" \
  --output-report "$REPORT" \
  --near-duplicate-hamming 2 \
  --minimum-new-rows 1000 \
  --minimum-new-classes 6 \
  --minimum-new-rows-per-class 30 >"$LOG" 2>&1
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

"$PYTHON" -c 'import json,sys; from pathlib import Path; rc=int(sys.argv[2]); status="pass_joint_research_manifest_available" if rc == 0 else ("fail_closed_joint_support_insufficient" if rc == 2 else "failed_runtime"); Path(sys.argv[1]).write_text(json.dumps({"status":status,"exit_code":rc,"report":sys.argv[3],"report_sha256":sys.argv[4],"manifest":sys.argv[5],"manifest_sha256":sys.argv[6],"new_rows_explicitly_weighted_as_cctv":True,"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"deployment_performed":False},indent=2)+"\n",encoding="utf-8")' "$STATE" "$rc" "$REPORT" "$report_sha" "$MANIFEST" "$manifest_sha"
exit "$rc"
