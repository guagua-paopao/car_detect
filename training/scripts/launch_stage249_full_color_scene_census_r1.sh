#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SCRIPT="$BASE/code/training/scripts/finalize_stage249_full_color_scene_census.py"
STAGE248_ROOT="$BASE/runs/attributes/ATTR-STAGE248-FULL-COLOR-TRAIN-SCENE-REAUDIT-R1"
STAGE248_STATE="$STAGE248_ROOT/state.json"
STAGE248_SESSION=VCAS-STAGE248-FULL-COLOR-TRAIN-SCENE-REAUDIT-R1
STAGE223="$BASE/code/stage249-full-color-scene-census-r1/inputs/stage223-color-night-agent-visual-overlay.csv"
STAGE239="$BASE/code/stage249-full-color-scene-census-r1/inputs/stage239-unknown-night-agent-review-overlay.csv"
ROOT="$BASE/runs/attributes/ATTR-STAGE249-FULL-COLOR-SCENE-CENSUS-R1"
LOG="$BASE/logs/stage249-full-color-scene-census-r1.log"
SESSION=VCAS-STAGE249-FULL-COLOR-SCENE-CENSUS-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$STAGE223" a2a97bb3b3282a636cb699389035fd7672d6d22b68cdf9c72193ce64fcfc5b7e
assert_sha256 "$STAGE239" b44511db4f387b99f960c58708b96e6d7616a423d8ed15f350346ec0e902bd58
"$PY" -m py_compile "$SCRIPT"

if [[ "${STAGE249_WORKER:-0}" != "1" ]]; then
  test ! -e "$ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION" >&2
    exit 67
  fi
  tmux new-session -d -s "$SESSION" \
    "STAGE249_WORKER=1 bash '$BASE/code/stage249-full-color-scene-census-r1/launch_stage249_full_color_scene_census_r1.sh'"
  echo "started tmux:$SESSION"
  echo "root=$ROOT"
  echo "log=$LOG"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "Stage249 waiting for Stage248: $(date -Is)"
while tmux has-session -t "$STAGE248_SESSION" 2>/dev/null || \
      pgrep -af '[a]udit_stage177_full_train_scenes.py.*ATTR-STAGE248' >/dev/null 2>&1; do
  sleep 20
done

test -f "$STAGE248_STATE"
mapfile -t inputs < <("$PY" - "$STAGE248_STATE" <<'PY'
import json, sys
from pathlib import Path
state=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert state['status']=='complete'
report=Path(state['report'])
payload=json.loads(report.read_text(encoding='utf-8'))
assert payload['scope']['validation_or_test_images_opened']==0
assert payload['policy']['frozen_video_used'] is False
print(report)
print(state['report_sha256'])
print(payload['outputs']['enriched_manifest'])
print(payload['outputs']['enriched_manifest_sha256'])
PY
)
test "${#inputs[@]}" -eq 4
REPORT248="${inputs[0]}"
REPORT248_SHA="${inputs[1]}"
MANIFEST="${inputs[2]}"
MANIFEST_SHA="${inputs[3]}"
assert_sha256 "$REPORT248" "$REPORT248_SHA"
assert_sha256 "$MANIFEST" "$MANIFEST_SHA"

"$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --stage248-report "$REPORT248" \
  --expected-stage248-report-sha256 "$REPORT248_SHA" \
  --stage223-overlay "$STAGE223" \
  --expected-stage223-overlay-sha256 a2a97bb3b3282a636cb699389035fd7672d6d22b68cdf9c72193ce64fcfc5b7e \
  --stage239-overlay "$STAGE239" \
  --expected-stage239-overlay-sha256 b44511db4f387b99f960c58708b96e6d7616a423d8ed15f350346ec0e902bd58 \
  --output-root "$ROOT"
echo "Stage249 complete: $(date -Is)"
