#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
EVAL="$CODE/scripts/evaluate_stage109_color_class_thresholds.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
SPECIALIST="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
COLOR_ROOT="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE109-COLOR-CLASS-THRESHOLD-VALIDATION-R1"
STATE="$ROOT.state.json"
LOG="$ROOT.log"
SESSION=VCAS-STAGE109-COLOR-CLASS-THRESHOLD-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

worker() {
  for required in "$PY" "$EVAL" "$LABELS" "$MANIFEST" "$BODY" "$SPECIALIST" "$BASELINE" "$COLOR_ROOT/best.pt" "$COLOR_ROOT/gate-best.pt"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$EVAL" 5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
  assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
  assert_sha256 "$MANIFEST" 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
  assert_sha256 "$BODY" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
  assert_sha256 "$SPECIALIST" 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae
  assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
  assert_sha256 "$COLOR_ROOT/best.pt" 3620a794e314370315f1c363b96b2aabebe14330d714ef9274d62d5a709b7fb2
  assert_sha256 "$COLOR_ROOT/gate-best.pt" 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2
  [[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage109 output" >&2; exit 66; }
  mkdir -p "$ROOT"
  declare -A color_sha=(
    [best]=3620a794e314370315f1c363b96b2aabebe14330d714ef9274d62d5a709b7fb2
    [gate-best]=811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2
  )
  for variant in best gate-best; do
    set +e
    "$PY" "$EVAL" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae \
      --body-specialist-subtype-threshold 0.50 \
      --color-checkpoint "$COLOR_ROOT/$variant.pt" --expected-color-checkpoint-sha256 "${color_sha[$variant]}" \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --datasets-safety-root "$BASE/datasets" --precision-target 0.935 \
      --device cuda --batch-size 64 --workers 8 --output "$ROOT/$variant/report.json"
    code=$?
    set -e
    [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
  done
  "$PY" - "$ROOT" "$STATE" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root,state=map(Path,sys.argv[1:])
variants=[]
for name in ('best','gate-best'):
    path=root/name/'report.json'
    data=json.load(open(path,encoding='utf-8'))
    assert data['policy']['split']=='validation'
    assert data['policy']['test_accessed'] is False
    assert data['policy']['frozen_video_used'] is False
    variants.append({
        'variant':name,
        'qualified':bool(data['gates']['all_pass']),
        'static':data['static']['candidate'],
        'unknown_reduction':data['comparison']['color_complex_static_unknown_relative_reduction'],
        'track_final':data['track_fusion']['candidate']['track_final'],
        'stability':data['track_fusion']['candidate']['stability'],
        'thresholds':data['selection']['thresholds'],
        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    })
qualified=[item['variant'] for item in variants if item['qualified']]
diagnostic=max(variants,key=lambda item=(0,): (
    int(item['static']['precision']>=.93)+int(item['static']['coverage']>=.25)+
    int(item['unknown_reduction']>=.20)+int(item['track_final']['precision']>=.93)+
    int(item['track_final']['coverage']>=.25)+int(item['stability']['transition_stability']>=.95),
    item['unknown_reduction'],item['static']['coverage'],item['track_final']['coverage']))
out={
  'status':'complete_validation_only',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'decision':'color_threshold_policy_qualified_pending_integrated_gate' if qualified else 'color_threshold_policy_rejected_fail_closed',
  'qualified_variants':qualified,
  'selected_diagnostic':diagnostic['variant'],
  'variants':variants,
  'test_accessed':False,'frozen_video_used':False,'production_model_modified':False,
  'backend_gates_run':False,'deployment_performed':False,
}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('decision','qualified_variants','selected_diagnostic')},ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit
fi

[[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage109 output" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
