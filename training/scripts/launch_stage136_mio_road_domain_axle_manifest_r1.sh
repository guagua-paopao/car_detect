#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
BUILDER="$CODE/scripts/build_stage136_mio_road_domain_axle_manifest.py"
TEST="$CODE/tests/test_build_stage136_mio_road_domain_axle_manifest.py"
BASE_DATA="$BASE/datasets/attribute-domain-v2/stage120-axle-semantic-clean-r1"
BASE_MANIFEST="$BASE_DATA/attribute_manifest.stage120-axle-semantic-clean.csv"
BASE_REPORT="$BASE_DATA/stage120-axle-semantic-clean-report.json"
MIO_DATA="$BASE/datasets/attribute-domain-v2/stage89-mio-balanced-expansion-r4"
MIO_MANIFEST="$MIO_DATA/attribute_manifest.stage89-mio-balanced.csv"
MIO_REPORT="$MIO_DATA/stage89-mio-balanced-report.json"
ROOT="$BASE/datasets/attribute-domain-v2/stage136-mio-road-domain-axle-r1"
OUTPUT_MANIFEST="$ROOT/attribute_manifest.stage136-mio-road-domain-axle.csv"
OUTPUT_REPORT="$ROOT/stage136-mio-road-domain-axle-report.json"
LOG="$BASE/runs/attributes/ATTR-STAGE136-MIO-ROAD-DOMAIN-AXLE-MANIFEST-R1.log"
SESSION=VCAS-STAGE136-MIO-ROAD-DOMAIN-AXLE-MANIFEST-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

worker() {
  for required in "$PY" "$BUILDER" "$TEST" "$BASE_MANIFEST" "$BASE_REPORT" "$MIO_MANIFEST" "$MIO_REPORT"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$BUILDER" f223d26591cc08d8b13c44f0e6ef6c5ba88c3f9d197e7f5a40d5c2fc00bd93e4
  assert_sha256 "$TEST" 0fec8a8f788030cb5be0041d4e822006b5437ed364a93959adec62ef03da7cfb
  assert_sha256 "$BASE_MANIFEST" ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67
  assert_sha256 "$BASE_REPORT" 30a4a8e3a2b3209da6ff5c336490a6c74021812d949d8f7eb77ee1f77ed7228d
  assert_sha256 "$MIO_MANIFEST" acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7
  assert_sha256 "$MIO_REPORT" 0de9acea113df457f2840b99e6b5fcf7da67a79f2193cc7ae94a3a60bcdc571d
  "$PY" -m py_compile "$BUILDER" "$TEST"
  cd "$CODE"
  PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_build_stage136_mio_road_domain_axle_manifest.py' -v
  [[ ! -e "$ROOT" ]] || { echo "refusing to overwrite Stage136" >&2; exit 66; }
  "$PY" "$BUILDER" \
    --base-manifest "$BASE_MANIFEST" --expected-base-manifest-sha256 ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67 \
    --base-report "$BASE_REPORT" --expected-base-report-sha256 30a4a8e3a2b3209da6ff5c336490a6c74021812d949d8f7eb77ee1f77ed7228d \
    --mio-manifest "$MIO_MANIFEST" --expected-mio-manifest-sha256 acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7 \
    --mio-report "$MIO_REPORT" --expected-mio-report-sha256 0de9acea113df457f2840b99e6b5fcf7da67a79f2193cc7ae94a3a60bcdc571d \
    --output-manifest "$OUTPUT_MANIFEST" --output-report "$OUTPUT_REPORT" \
    --near-duplicate-hamming 4
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage136" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "manifest=$OUTPUT_MANIFEST"
echo "report=$OUTPUT_REPORT"
