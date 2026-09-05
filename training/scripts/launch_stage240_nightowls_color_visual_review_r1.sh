#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python3
CODE="$BASE/code/stage240-nightowls-color-visual-r1"
SCRIPT="$CODE/build_stage240_nightowls_color_visual_review.py"
MANIFEST="$BASE/runs/attributes/ATTR-STAGE234-NIGHTOWLS-VEHICLE-TRACKS-R1/stage234-nightowls-vehicle-tracks.csv"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE240-NIGHTOWLS-COLOR-VISUAL-R1"
LOG="$BASE/logs/stage240-nightowls-color-visual-r1.log"
SESSION=VCAS-STAGE240-NIGHTOWLS-COLOR-VISUAL-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$SCRIPT" f503dca67009e3494f84a609f768a751c59a99bb6564a704e0437cdc7160024e
assert_sha256 "$MANIFEST" 7b653ae0266ecb613dd40df70ab2f8f8075534004c58d789321dba52ad9387f6
"$PY" -m py_compile "$SCRIPT"
test ! -e "$OUTPUT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$SCRIPT' \
    --manifest '$MANIFEST' \
    --expected-manifest-sha256 7b653ae0266ecb613dd40df70ab2f8f8075534004c58d789321dba52ad9387f6 \
    --output-dir '$OUTPUT' \
    --minimum-track-length 3 \
    --minimum-independent-frames 3 \
    --minimum-average-quality 0.04 \
    --minimum-average-luma 40 \
    --minimum-average-saturation 25 \
    --minimum-width 48 \
    --minimum-height 32 \
    --tracks-per-page 20 >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "output=$OUTPUT"
echo "log=$LOG"
