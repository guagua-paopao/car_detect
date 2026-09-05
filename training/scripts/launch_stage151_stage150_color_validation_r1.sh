#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
EVAL="$CODE/scripts/evaluate_stage109_color_class_thresholds.py"
EVAL_BASE="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
SPECIALIST="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
TRAIN_STATE="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-DOMAIN-R1.state.json"
TRAIN_SESSION=VCAS-STAGE150-COLOR-DOMAIN-R1
ROOT="$BASE/runs/attributes/ATTR-STAGE151-STAGE150-COLOR-VALIDATION-R1"
STATE="$ROOT.state.json"
LOG="$ROOT.log"
SESSION=VCAS-STAGE151-STAGE150-COLOR-VALIDATION-R1

EVAL_SHA=5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
EVAL_BASE_SHA=35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
LABELS_SHA=22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
MANIFEST_SHA=70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
BODY_SHA=e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
SPECIALIST_SHA=0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae
BASELINE_SHA=6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

worker() {
  for _ in $(seq 1 240); do
    if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
      break
    fi
    sleep 30
  done
  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    echo "Stage150 training did not finish within two hours" >&2
    exit 2
  fi
  for required in "$PY" "$EVAL" "$EVAL_BASE" "$LABELS" "$MANIFEST" "$BODY" "$SPECIALIST" "$BASELINE" "$TRAIN_STATE"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$EVAL" "$EVAL_SHA"
  assert_sha256 "$EVAL_BASE" "$EVAL_BASE_SHA"
  assert_sha256 "$LABELS" "$LABELS_SHA"
  assert_sha256 "$MANIFEST" "$MANIFEST_SHA"
  assert_sha256 "$BODY" "$BODY_SHA"
  assert_sha256 "$SPECIALIST" "$SPECIALIST_SHA"
  assert_sha256 "$BASELINE" "$BASELINE_SHA"
  "$PY" -m py_compile "$EVAL" "$EVAL_BASE"
  [[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage151 output" >&2; exit 66; }
  mkdir -p "$ROOT"

  mapfile -t candidates < <("$PY" - "$TRAIN_STATE" <<'PY'
import hashlib,json,sys
from pathlib import Path
state_path=Path(sys.argv[1]); state=json.loads(state_path.read_text(encoding='utf-8'))
if state.get('status')!='training_complete_pending_stage151_validation':
    raise SystemExit(f"Stage150 training state is not validation-ready: {state.get('status')}")
policy=state.get('policy',{})
if policy.get('test_accessed') is not False or policy.get('stage148_test_reused') is not False or policy.get('frozen_video_used') is not False:
    raise SystemExit('Stage150 forbidden-data policy failed')
artifacts=state.get('artifacts',{})
for filename in ('best.pt','last.pt','gate-best.pt'):
    meta=artifacts.get(filename)
    if not meta:
        continue
    path=Path(meta['path'])
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=meta['sha256']:
        raise SystemExit(f'{filename} SHA256 mismatch')
    print(f"{filename[:-3]}\t{path}\t{actual}")
PY
  )
  [[ "${#candidates[@]}" -ge 2 ]] || { echo "fewer than two Stage150 checkpoints available" >&2; exit 67; }

  for entry in "${candidates[@]}"; do
    IFS=$'\t' read -r variant checkpoint checkpoint_sha <<<"$entry"
    set +e
    "$PY" "$EVAL" \
      --manifest "$MANIFEST" --expected-manifest-sha256 "$MANIFEST_SHA" \
      --labels "$LABELS" --expected-labels-sha256 "$LABELS_SHA" \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 "$BODY_SHA" \
      --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 "$SPECIALIST_SHA" \
      --body-specialist-subtype-threshold 0.50 \
      --color-checkpoint "$checkpoint" --expected-color-checkpoint-sha256 "$checkpoint_sha" \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 "$BASELINE_SHA" \
      --datasets-safety-root "$BASE/datasets" --precision-target 0.935 \
      --device cuda --batch-size 64 --workers 8 --output "$ROOT/$variant/report.json"
    code=$?
    set -e
    [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
  done

  "$PY" - "$ROOT" "$STATE" "$TRAIN_STATE" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root,state,training_state=map(Path,sys.argv[1:])
variants=[]
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    path=directory/'report.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['policy']['split']=='validation'
    assert data['policy']['test_accessed'] is False
    assert data['policy']['frozen_video_used'] is False
    variants.append({
      'variant':directory.name,
      'checkpoint_sha256':data['inputs']['color_checkpoint_sha256'],
      'qualified':bool(data['gates']['all_pass']),
      'static':data['static']['candidate'],
      'unknown_reduction':data['comparison']['color_complex_static_unknown_relative_reduction'],
      'track_final':data['track_fusion']['candidate']['track_final'],
      'stability':data['track_fusion']['candidate']['stability'],
      'thresholds':data['selection']['thresholds'],
      'per_class':data['per_class'],
      'stratified':data['stratified'],
      'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    })
qualified=[item for item in variants if item['qualified']]
diagnostic=max(variants,key=lambda item: (
    int(item['static']['precision']>=.93)+int(item['static']['coverage']>=.25)+
    int(item['unknown_reduction']>=.20)+int(item['track_final']['precision']>=.93)+
    int(item['track_final']['coverage']>=.25)+int(item['stability']['transition_stability']>=.95),
    item['unknown_reduction'],item['static']['coverage'],item['track_final']['coverage']))
selected=max(qualified,key=lambda item:(item['static']['coverage'],item['track_final']['coverage'],item['unknown_reduction'])) if qualified else None
out={
  'schema_version':'stage151-stage150-color-validation-state-v1',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'complete_validation_only',
  'decision':'color_component_qualified_pending_body_repair_and_integrated_gate' if selected else 'color_component_rejected_fail_closed',
  'qualified_variants':[item['variant'] for item in qualified],
  'selected_variant':selected['variant'] if selected else None,
  'selected_checkpoint_sha256':selected['checkpoint_sha256'] if selected else None,
  'selected_diagnostic':diagnostic['variant'],
  'training_state_sha256':hashlib.sha256(training_state.read_bytes()).hexdigest(),
  'variants':variants,
  'test_accessed':False,'stage148_test_reused':False,'frozen_video_used':False,
  'production_model_modified':False,'backend_gates_run':False,'deployment_performed':False,
}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('decision','qualified_variants','selected_variant','selected_diagnostic')},ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit
fi

[[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage151 output" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
