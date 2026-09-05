#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training"
BUILDER="$CODE/scripts/build_stage98_two_axle_cctv_specialist_manifest.py"
TEST="$CODE/tests/test_build_stage98_two_axle_cctv_specialist_manifest.py"
STAGE93_BUILDER="$CODE/scripts/build_stage93_truck_subtype_manifest.py"
STAGE93="$BASE/datasets/attribute-domain-v2/stage93-truck-subtype-r2/attribute_manifest.stage93-truck-subtype.csv"
STAGE91="$BASE/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/attribute_manifest.stage91-all.csv"
STAGE91_REPORT="$BASE/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/stage91-inatrc-train-crops-report.json"
EVIDENCE="$BASE/sources/inatrc-stage91/inatrc-source-evidence-v1.json"
LABELS="$CODE/config/vehicle_labels.truck-subtype.v1.json"
OUTPUT="$BASE/datasets/attribute-domain-v2/stage98-two-axle-cctv-r1"
LOG="$BASE/runs/attributes/ATTR-STAGE98-TWO-AXLE-CCTV-MANIFEST-R1.log"
SESSION=VCAS-STAGE98-TWO-AXLE-R1

test ! -e "$OUTPUT"
test ! -e "$LOG"
echo "2bbd23f09c155e3b71d7b6c431610158e0fc2ca6f0945521026f479a1e2db0df  $BUILDER" | sha256sum -c -
echo "87405df4aeeb48a26e3253be81817082e64b6114fe31a8e5d1b3e07f2f103699  $TEST" | sha256sum -c -
echo "c148c625d35899d55ba5fed028f6479dab69f8538955082e81651d0cd99ef0b6  $STAGE93_BUILDER" | sha256sum -c -
echo "d8582417f7826ac76c2e290595c754a54df0838dc6bef324531cd7167d49256a  $STAGE93" | sha256sum -c -
echo "d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1  $STAGE91" | sha256sum -c -
echo "45585cc56e6d481145e59049113634d6fcf7179f694944f538baaeb0fdb2bfb0  $STAGE91_REPORT" | sha256sum -c -
echo "dfbddf87d61f5d3cf37e7e801fc8d9dcc32ee3396613b2af1cef4666cedd5145  $EVIDENCE" | sha256sum -c -
echo "2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46  $LABELS" | sha256sum -c -

cd "$CODE"
"$PY" -m unittest discover -s tests -p 'test_build_stage98_two_axle_cctv_specialist_manifest.py'
mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" \
  "mkdir -p '$OUTPUT' && '$PY' '$BUILDER' \
    --stage93-manifest '$STAGE93' \
    --expected-stage93-sha256 d8582417f7826ac76c2e290595c754a54df0838dc6bef324531cd7167d49256a \
    --stage91-manifest '$STAGE91' \
    --expected-stage91-sha256 d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1 \
    --stage91-report '$STAGE91_REPORT' \
    --expected-stage91-report-sha256 45585cc56e6d481145e59049113634d6fcf7179f694944f538baaeb0fdb2bfb0 \
    --source-evidence '$EVIDENCE' \
    --expected-source-evidence-sha256 dfbddf87d61f5d3cf37e7e801fc8d9dcc32ee3396613b2af1cef4666cedd5145 \
    --labels '$LABELS' \
    --expected-labels-sha256 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46 \
    --safety-root '$BASE/datasets' \
    --near-duplicate-hamming 4 \
    --minimum-train-per-class 10000 \
    --minimum-validation-per-class 300 \
    --minimum-two-axle-rows 4000 \
    --output-manifest '$OUTPUT/attribute_manifest.stage98-two-axle-cctv.csv' \
    --output-report '$OUTPUT/stage98-two-axle-cctv-report.json' >'$LOG' 2>&1; \
   rc=\$?; if [[ \$rc -eq 0 ]]; then sha256sum '$OUTPUT/attribute_manifest.stage98-two-axle-cctv.csv' > '$OUTPUT/attribute_manifest.stage98-two-axle-cctv.csv.sha256'; fi; \
   sha256sum '$OUTPUT/stage98-two-axle-cctv-report.json' > '$OUTPUT/stage98-two-axle-cctv-report.json.sha256'; exit \$rc"

echo "started tmux:$SESSION"
echo "output=$OUTPUT"
echo "log=$LOG"
