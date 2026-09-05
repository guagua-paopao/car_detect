#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code
RUNS=$BASE/runs/attributes
VALIDATION_STATE=$RUNS/ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1.state.json
VALIDATION_REPORT=$RUNS/ATTR-STAGE71-TEACHER-PAIR-VALIDATION-ONLY-V1/pair-validation-screen-report.json
VALIDATION_SESSION=VCAS-VALIDATE-STAGE71-TEACHERS
TEMPLATE=$CODE/training/stage71-student-matrix-template-v1.json
MATERIALIZER=$CODE/training/scripts/prepare_stage71_student_matrix_v2.py
RUNNER=$CODE/training/scripts/run_stage71_student_matrix.py
MATRIX=$CODE/training/stage71-student-matrix-v1.json
PREFLIGHT_LOG=$RUNS/ATTR-STAGE71-STUDENTS-PREFLIGHT-V1.log
OUT=$RUNS/ATTR-STAGE71-STUDENTS-V1
STATE=$RUNS/ATTR-STAGE71-STUDENTS-V1.state.json

[[ "$(sha256sum "$TEMPLATE" | awk '{print $1}')" == "2d52a2274422bc90f3f691a27979a971b7744f6844ccd599259fea486add39a5" ]]
[[ "$(sha256sum "$MATERIALIZER" | awk '{print $1}')" == "f0e65ab00c0a2e359e2f54088e138e3fcc1ef742ed0059ec1a409e7cf4bdcaba" ]]
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "4d1e5a4a0a371e9dada0b8182c43f79e1b555859912f771f03b11765abcf23d9" ]]
for path in "$MATRIX" "$PREFLIGHT_LOG" "$OUT" "$STATE"; do
  test ! -e "$path"
done

deadline=$((SECONDS + 43200))
while (( SECONDS < deadline )); do
  if [[ -f "$VALIDATION_STATE" ]]; then
    status="$($PY - "$VALIDATION_STATE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "missing"))
PY
)"
    if [[ "$status" == "complete" ]]; then
      break
    fi
    if [[ "$status" != "waiting_for_training" && "$status" != "running" ]]; then
      echo "teacher validation failed closed with status=$status" >&2
      exit 2
    fi
  fi
  if ! tmux has-session -t "=$VALIDATION_SESSION" 2>/dev/null; then
    echo "teacher validation session ended without complete evidence" >&2
    exit 2
  fi
  sleep 30
done
(( SECONDS < deadline )) || { echo "timed out waiting for teacher validation" >&2; exit 2; }

$PY "$MATERIALIZER" \
  --template "$TEMPLATE" \
  --expected-template-sha256 2d52a2274422bc90f3f691a27979a971b7744f6844ccd599259fea486add39a5 \
  --validation-state "$VALIDATION_STATE" \
  --validation-report "$VALIDATION_REPORT" \
  --output "$MATRIX"

$PY "$RUNNER" \
  --matrix "$MATRIX" \
  --output-root "$RUNS/ATTR-STAGE71-STUDENTS-PREFLIGHT-V1" \
  --state "$RUNS/ATTR-STAGE71-STUDENTS-PREFLIGHT-V1.state.json" \
  --device cuda \
  --code-revision stage71-specialist-students-research-v1 \
  --preflight-only >"$PREFLIGHT_LOG" 2>&1

exec $PY "$RUNNER" \
  --matrix "$MATRIX" \
  --output-root "$OUT" \
  --state "$STATE" \
  --device cuda \
  --code-revision stage71-specialist-students-research-v1
