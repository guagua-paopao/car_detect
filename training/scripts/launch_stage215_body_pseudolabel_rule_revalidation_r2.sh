#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SCRIPT="$BASE/code/training/scripts/evaluate_stage215_body_pseudolabel_rule.py"
TEST="$BASE/code/training/tests/test_evaluate_stage215_body_pseudolabel_rule.py"
DATASET_ROOT="$BASE/datasets/attribute-domain-v2"
MANIFEST="$DATASET_ROOT/vfg7-eval-v1/attribute_manifest.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
PRODUCTION="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
EFFICIENT="$BASE/runs/stage7/ATTR-V5-EFFV2S-256-UVH26-BODY/best.pt"
RESNET="$BASE/runs/stage14/ATTR-V12-RESNET50-256-HARD/best.pt"
STAGE167="$BASE/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1/best.pt"
STAGE211="$BASE/runs/attributes/ATTR-STAGE211-INATRC-TOLL-HARDCLASS-R1/ATTR-STAGE211-BODY-CONVNEXT-256-INATRC-HARDCLASS-R1/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE215-BODY-PSEUDOLABEL-RULE-REVALIDATION-R2"
STATE="$OUTPUT_ROOT.state.json"
SESSION=VCAS-STAGE215-BODY-RULE-REVALIDATION-R2

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

preflight() {
  for required in \
    "$PY" "$SCRIPT" "$TEST" "$MANIFEST" "$LABELS" "$PRODUCTION" \
    "$EFFICIENT" "$RESNET" "$STAGE167" "$STAGE211"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$SCRIPT" 1491ec0426af2b3f8846b0deba580b88d20e697ffcfaed33f3541bbaf340ebad
  assert_sha256 "$TEST" 4f7a7f87e26ea33496d0e9ec4a2a22ffc4426a2b2a1da38246a432aa016649c5
  assert_sha256 "$MANIFEST" 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3
  assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
  assert_sha256 "$PRODUCTION" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
  assert_sha256 "$EFFICIENT" f48b27a83b722b087329feca17c5ffb17958b2fed063e69248616d8d44ddceaf
  assert_sha256 "$RESNET" b15dc188e3928a33a85988d896828f4f5c9fd5e9f9cafef5b806785f2cb6edc0
  assert_sha256 "$STAGE167" c6dea9d50a01fd6395e403606083e955d8880f9f9e77048d0d5f28c487b11f0b
  assert_sha256 "$STAGE211" d3513fecbb3feb2b0207a80832123adb770cbb4ab3497cd52af8350084d4dd1a
  "$PY" "$TEST"
}

evaluate_variant() {
  local name="$1"
  local candidate="$2"
  local report="$OUTPUT_ROOT/$name/report.json"
  local log="$OUTPUT_ROOT/$name/run.log"
  mkdir -p "$OUTPUT_ROOT/$name"
  set +e
  "$PY" "$SCRIPT" \
    --manifest "$MANIFEST" --dataset-root "$DATASET_ROOT" \
    --labels "$LABELS" \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --checkpoint "production=$PRODUCTION" \
    --checkpoint "efficient=$EFFICIENT" \
    --checkpoint "resnet=$RESNET" \
    --checkpoint "candidate=$candidate" \
    --output "$report" --batch-size 128 --workers 6 --device cuda >"$log" 2>&1
  local rc=$?
  set -e
  if [[ -f "$report" ]]; then
    sha256sum "$report" >"$report.sha256"
  fi
  printf '%s\n' "$rc"
}

worker() {
  preflight
  mkdir -p "$OUTPUT_ROOT"
  local rc167 rc211
  rc167="$(evaluate_variant stage167 "$STAGE167")"
  rc211="$(evaluate_variant stage211 "$STAGE211")"
  "$PY" - "$STATE" "$OUTPUT_ROOT" "$rc167" "$rc211" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
root = Path(sys.argv[2])
return_codes = {"stage167": int(sys.argv[3]), "stage211": int(sys.argv[4])}
variants = []
for name, return_code in return_codes.items():
    report_path = root / name / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path.is_file() else ""
    variants.append({
        "name": name,
        "return_code": return_code,
        "status": report.get("status") if report else "missing_report",
        "selected": report.get("selected") if report else None,
        "report": str(report_path),
        "report_sha256": report_sha,
    })
passing = [item for item in variants if item["return_code"] == 0 and item["selected"]]
selected = max(
    passing,
    key=lambda item: (
        float(item["selected"]["coverage"]),
        float(item["selected"]["precision"]),
        int(item["selected"]["selected"]),
    ),
    default=None,
)
state = {
    "schema_version": "stage215-body-pseudolabel-rule-revalidation-state-v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "pass_validation_rule_available" if selected else "fail_closed_no_rule_passed",
    "selected_variant": selected["name"] if selected else None,
    "variants": variants,
    "split": "validation",
    "test_accessed": False,
    "frozen_video_used": False,
    "production_model_modified": False,
    "training_started": False,
    "deployment_performed": False,
}
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  sha256sum "$STATE" >"$STATE.sha256"
  if [[ "$rc167" -ne 0 && "$rc211" -ne 0 ]]; then
    exit 2
  fi
}

if [[ "${STAGE215_WORKER:-0}" == "1" ]]; then
  worker
  exit $?
fi

preflight
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
tmux new-session -d -s "$SESSION" "STAGE215_WORKER=1 bash '$0'"
echo "started tmux:$SESSION"
echo "state=$STATE"
