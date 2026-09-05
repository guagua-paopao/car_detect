#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SCRIPT="$BASE/code/training/scripts/audit_stage177_full_train_scenes.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage242-nightowls-color-repair-r1/attribute_manifest.stage242-nightowls-color-repair.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
ROOT="$BASE/runs/attributes/ATTR-STAGE248-FULL-COLOR-TRAIN-SCENE-REAUDIT-R1"
LOG="$BASE/logs/stage248-full-color-train-scene-reaudit-r1.log"
SESSION=VCAS-STAGE248-FULL-COLOR-TRAIN-SCENE-REAUDIT-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$MANIFEST" 445babeb94649f32b208db080e2c2085451c26ca25ad9187b37ce6c2dd9112f8
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
"$PY" -m py_compile "$SCRIPT"

if [[ "${STAGE248_WORKER:-0}" != "1" ]]; then
  test ! -e "$ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION" >&2
    exit 67
  fi
  tmux new-session -d -s "$SESSION" \
    "STAGE248_WORKER=1 bash '$BASE/code/stage248-full-color-train-scene-reaudit-r1/launch_stage248_full_color_train_scene_reaudit_r1.sh'"
  echo "started tmux:$SESSION"
  echo "root=$ROOT"
  echo "log=$LOG"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "Stage248 full train-only color-scene re-audit started: $(date -Is)"
"$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 445babeb94649f32b208db080e2c2085451c26ca25ad9187b37ce6c2dd9112f8 \
  --labels "$LABELS" \
  --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --allowed-root "$BASE" \
  --output-root "$ROOT" \
  --workers 12 \
  --batch-size 4096 \
  --near-duplicate-hamming 4 \
  --max-effective-frames-per-track 5 \
  --include-all-train \
  --include-coarse-color
echo "Stage248 complete: $(date -Is)"
