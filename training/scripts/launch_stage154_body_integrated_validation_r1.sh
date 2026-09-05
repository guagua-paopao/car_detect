#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
EVAL="$CODE/scripts/evaluate_stage110_body_class_thresholds.py"
EVAL_BASE="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
TRAIN_STATE="$BASE/runs/attributes/ATTR-STAGE153-BODY-NIGHT-SMALL-REPAIR-R2.state.json"
TRAIN_SESSION=VCAS-STAGE153-BODY-REPAIR-R2
SPECIALIST="$BASE/runs/attributes/ATTR-STAGE145-AXLE-DOMAIN-LETTERBOX-MULTIRES-R1/ATTR-STAGE145-TRUCK-SUBTYPE-CONVNEXT-288-AXLE-LETTERBOX-R1/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-DOMAIN-R1/ATTR-STAGE150-COLOR-CONVNEXT-256-VEHICLE-REAR-DOMAIN-R1/best.pt"
COLOR_STATE="$BASE/runs/attributes/ATTR-STAGE151-STAGE150-COLOR-VALIDATION-R2.state.json"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE154-BODY-NIGHT-SMALL-VALIDATION-R1"
BODY_STATE="$ROOT.state.json"
INTEGRATED_STATE="$BASE/runs/attributes/ATTR-STAGE154-INTEGRATED-COMPONENT-GATE-R1.state.json"

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

for ignored in {1..480}; do
  if ! tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "Stage153 training did not finish within four hours" >&2
  exit 2
fi

[[ ! -e "$ROOT" && ! -e "$BODY_STATE" && ! -e "$INTEGRATED_STATE" ]] || { echo "refusing to overwrite Stage154 evidence" >&2; exit 66; }
assert_sha256 "$EVAL" 0c7d105a54101ed2c8e8b9f803339f10ae7c46644479a36f40107b0a0a269f17
assert_sha256 "$EVAL_BASE" 447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
assert_sha256 "$SPECIALIST" 0cd4a23d8388d160cc652120aa80f64df1f051a9efd99ae97ba1f9057d317dde
assert_sha256 "$COLOR" fbbb9d78361e0b87800b1c4784a31df2c01fd107e0c4668d95317502e7671af4
assert_sha256 "$COLOR_STATE" 16b7fe7641d7b4c4e8cf129a35ee7625db4ec53b4dfb03ce0a0416183f7b3fab
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
"$PY" -m py_compile "$EVAL" "$EVAL_BASE"
mkdir -p "$ROOT"

mapfile -t candidates < <("$PY" - "$TRAIN_STATE" <<'PY'
import hashlib,json,sys
from pathlib import Path
state_path=Path(sys.argv[1])
if not state_path.is_file(): raise SystemExit('missing Stage153 training state')
state=json.loads(state_path.read_text(encoding='utf-8'))
if state.get('status')!='training_complete_pending_stage154_validation':
    raise SystemExit(f"Stage153 training is not validation-ready: {state.get('status')}")
for key in ('test_accessed','stage148_test_reused','frozen_video_used','production_model_modified','backend_gates_run','deployment_performed'):
    if state.get(key) is not False: raise SystemExit(f'Stage153 forbidden policy failed: {key}')
for filename in ('best.pt','last.pt','gate-best.pt'):
    meta=state.get('artifacts',{}).get(filename)
    if not meta: continue
    path=Path(meta['path'])
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=meta['sha256']: raise SystemExit(f'{filename} SHA mismatch')
    print(f"{filename[:-3]}\t{path}\t{actual}")
PY
)
[[ "${#candidates[@]}" -ge 2 ]] || { echo "fewer than two Stage153 checkpoints available" >&2; exit 67; }

for entry in "${candidates[@]}"; do
  IFS=$'\t' read -r variant checkpoint checkpoint_sha <<<"$entry"
  set +e
  "$PY" "$EVAL" \
    --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
    --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --body-checkpoint "$checkpoint" --expected-body-checkpoint-sha256 "$checkpoint_sha" \
    --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 0cd4a23d8388d160cc652120aa80f64df1f051a9efd99ae97ba1f9057d317dde \
    --body-specialist-subtype-threshold 0.50 \
    --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 fbbb9d78361e0b87800b1c4784a31df2c01fd107e0c4668d95317502e7671af4 \
    --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --datasets-safety-root "$BASE/datasets" --precision-target 0.93 \
    --device cuda --batch-size 64 --workers 8 --output "$ROOT/$variant/report.json"
  code=$?
  set -e
  [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
done

"$PY" - "$ROOT" "$BODY_STATE" "$TRAIN_STATE" "$SPECIALIST" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root,state,training_state,specialist=map(Path,sys.argv[1:])
training=json.loads(training_state.read_text(encoding='utf-8'))
checkpoint_by_sha={meta['sha256']:meta['path'] for name,meta in training['artifacts'].items() if name.endswith('.pt')}
variants=[]
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    path=directory/'report.json'; data=json.loads(path.read_text(encoding='utf-8'))
    assert data['policy']['split']=='validation'
    assert data['policy']['test_accessed'] is False and data['policy']['frozen_video_used'] is False
    body_sha=data['inputs']['body_checkpoint_sha256']
    variants.append({
      'variant':directory.name,'primary_checkpoint':checkpoint_by_sha[body_sha],
      'primary_checkpoint_sha256':body_sha,
      'specialist_checkpoint':str(specialist),
      'specialist_checkpoint_sha256':data['inputs']['body_specialist_checkpoint_sha256'],
      'qualified':bool(data['gates']['all_pass']),'thresholds':data['selection']['thresholds'],
      'static':data['static']['candidate'],'complex_coverage_gain':data['comparison']['body_complex_static_coverage_gain'],
      'track_final':data['track_fusion']['candidate']['track_final'],'stability':data['track_fusion']['candidate']['stability'],
      'per_class':data['per_class'],'stratified':data['stratified'],'gates':data['gates']['gates'],
      'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    })
qualified=[item for item in variants if item['qualified']]
selected=max(qualified,key=lambda item:(item['static']['coverage'],item['track_final']['coverage'],item['complex_coverage_gain'],item['static']['precision'])) if qualified else None
diagnostic=max(variants,key=lambda item:(sum(item['gates'].values()),item['static']['coverage'],item['track_final']['coverage'],item['static']['precision']))
out={
 'schema_version':'stage154-body-night-small-validation-state-v1','created_at':datetime.now(timezone.utc).isoformat(),
 'status':'complete_validation_only','decision':'body_component_qualified_pending_integrated_gate' if selected else 'body_component_rejected_fail_closed',
 'qualified_variants':[item['variant'] for item in qualified],'selected_variant':selected['variant'] if selected else None,
 'selected_primary_checkpoint':selected['primary_checkpoint'] if selected else None,
 'selected_primary_checkpoint_sha256':selected['primary_checkpoint_sha256'] if selected else None,
 'selected_diagnostic':diagnostic['variant'],'variants':variants,
 'training_state_sha256':hashlib.sha256(training_state.read_bytes()).hexdigest(),
 'test_accessed':False,'stage148_test_reused':False,'frozen_video_used':False,'production_model_modified':False,
 'backend_gates_run':False,'deployment_performed':False,
}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('decision','qualified_variants','selected_variant','selected_diagnostic')},ensure_ascii=False))
PY

"$PY" - "$COLOR_STATE" "$BODY_STATE" "$COLOR" "$INTEGRATED_STATE" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
color_path,body_path,color_checkpoint,output=map(Path,sys.argv[1:])
color=json.loads(color_path.read_text(encoding='utf-8')); body=json.loads(body_path.read_text(encoding='utf-8'))
for role,state in (('color',color),('body',body)):
    for key in ('test_accessed','frozen_video_used','production_model_modified'):
        if state.get(key) is not False: raise SystemExit(f'{role} isolation failed: {key}')
def color_gates(v):
    return [v['static']['precision']>=.93,v['static']['coverage']>=.25,v['unknown_reduction']>=.20,
            v['track_final']['precision']>=.93,v['track_final']['coverage']>=.25,v['stability']['transition_stability']>=.95]
def body_gates(v):
    return [v['static']['precision']>=.93,v['static']['coverage']>=.45,v['complex_coverage_gain']>=.15,
            v['track_final']['precision']>=.93,v['track_final']['coverage']>=.45,v['stability']['transition_stability']>=.95]
colors=[v for v in color.get('variants',[]) if all(color_gates(v))]
bodies=[v for v in body.get('variants',[]) if all(body_gates(v))]
selected_color=max(colors,key=lambda v:(v['unknown_reduction'],v['static']['coverage'],v['track_final']['coverage'],v['static']['precision'])) if colors else None
selected_body=max(bodies,key=lambda v:(v['static']['coverage'],v['track_final']['coverage'],v['complex_coverage_gain'],v['static']['precision'])) if bodies else None
authorized=selected_color is not None and selected_body is not None
candidate=None
if authorized:
    candidate={
      'body_variant':selected_body['variant'],'body_primary_checkpoint':selected_body['primary_checkpoint'],
      'body_primary_checkpoint_sha256':selected_body['primary_checkpoint_sha256'],
      'body_specialist_checkpoint':selected_body['specialist_checkpoint'],
      'body_specialist_checkpoint_sha256':selected_body['specialist_checkpoint_sha256'],
      'body_class_thresholds':selected_body['thresholds'],'body_report_sha256':selected_body['report_sha256'],
      'color_variant':selected_color['variant'],'color_checkpoint':str(color_checkpoint),
      'color_checkpoint_sha256':hashlib.sha256(color_checkpoint.read_bytes()).hexdigest(),
      'color_class_thresholds':selected_color['thresholds'],'color_report_sha256':selected_color['report_sha256'],
    }
out={
 'schema_version':'stage154-integrated-component-gate-state-v1','created_at':datetime.now(timezone.utc).isoformat(),
 'status':'pass_stage155_new_holdout_authorized' if authorized else 'component_repair_required_fail_closed',
 'stage155_new_holdout_authorized':authorized,'candidate':candidate,
 'inputs':{'color_state':str(color_path),'color_state_sha256':hashlib.sha256(color_path.read_bytes()).hexdigest(),
           'body_state':str(body_path),'body_state_sha256':hashlib.sha256(body_path.read_bytes()).hexdigest()},
 'test_accessed':False,'stage148_test_reused':False,'frozen_video_used':False,'production_model_modified':False,
 'backend_gates_run':False,'deployment_performed':False,
}
output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(output)+'.sha256').write_text(hashlib.sha256(output.read_bytes()).hexdigest()+'  '+output.name+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'candidate':candidate},ensure_ascii=False))
PY
