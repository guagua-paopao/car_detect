#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
AUDIT="$CODE/scripts/build_stage125_bmd_generic_truck_consensus.py"
TEST="$CODE/tests/test_build_stage125_bmd_generic_truck_consensus.py"
SOURCE="$BASE/datasets/attribute-domain-v2/stage113-domain-balanced-specialist-r1/attribute_manifest.stage113-domain-balanced-specialist.csv"
BASE_MANIFEST="$BASE/datasets/attribute-domain-v2/stage120-axle-semantic-clean-r1/attribute_manifest.stage120-axle-semantic-clean.csv"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
STAGE99="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
STAGE117="$BASE/runs/attributes/ATTR-STAGE117-SINGLE-BALANCE-SPECIALIST-R1/ATTR-STAGE117-TRUCK-SUBTYPE-CONVNEXT-256-SINGLE-BALANCE-R1/best.pt"
STAGE121="$BASE/runs/attributes/ATTR-STAGE121-AXLE-SEMANTIC-CLEAN-R1/ATTR-STAGE121-TRUCK-SUBTYPE-CONVNEXT-256-AXLE-CLEAN-R1/best.pt"
OUT="$BASE/datasets/attribute-domain-v2/stage125-bmd-generic-truck-consensus-r1"
MANIFEST="$OUT/attribute_manifest.stage125-bmd-generic-truck-consensus.csv"
REPORT="$OUT/stage125-bmd-generic-truck-consensus-report.json"
LOG="$BASE/runs/attributes/ATTR-STAGE125-BMD-GENERIC-CONSENSUS-R1.log"
SESSION=VCAS-STAGE125-BMD-GENERIC-CONSENSUS-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

for required in "$PY" "$AUDIT" "$TEST" "$SOURCE" "$BASE_MANIFEST" "$LABELS" "$STAGE99" "$STAGE117" "$STAGE121"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$AUDIT" 6e05faeb5f68f3d2d31504e1ded458e54601889d2ea1287236b7ae8819939857
assert_sha256 "$TEST" a8012c86079417acca50eb6cf7334abca31a2d668ac1fb8a8f2fea54e3653b7a
assert_sha256 "$SOURCE" 70100165c058f72cbb7c9fe35a81fd4e6336c3dcc32de0fdd02dc93ce7969cb0
assert_sha256 "$BASE_MANIFEST" ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$STAGE99" 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae
assert_sha256 "$STAGE117" adfb9d3a3ffa9d11164e7349cbdf5e142fb77598b5abfc0b9fee0b68cafff673
assert_sha256 "$STAGE121" fa80144ffc84886188d8268935f5cdadb5d74cda3af170b44cf80a4d3f060545
"$PY" -m py_compile "$AUDIT" "$TEST"
cd "$CODE"
"$PY" -m unittest discover -s tests -p 'test_build_stage125_bmd_generic_truck_consensus.py'
[[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage125" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2; exit 68
fi
tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$AUDIT' \
    --source-manifest '$SOURCE' --expected-source-manifest-sha256 70100165c058f72cbb7c9fe35a81fd4e6336c3dcc32de0fdd02dc93ce7969cb0 \
    --base-manifest '$BASE_MANIFEST' --expected-base-manifest-sha256 ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67 \
    --labels '$LABELS' --expected-labels-sha256 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46 \
    --checkpoint stage99='$STAGE99' --checkpoint stage117='$STAGE117' --checkpoint stage121='$STAGE121' \
    --dataset-root '$BASE/datasets' --confidence 0.70 --batch-size 64 --workers 8 --device cuda \
    --output-manifest '$MANIFEST' --output-report '$REPORT' >'$LOG' 2>&1; code=\$?; [[ \$code == 0 || \$code == 2 ]]"
echo "started tmux:$SESSION"
echo "report=$REPORT"
