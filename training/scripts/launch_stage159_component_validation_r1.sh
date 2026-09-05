#!/usr/bin/env bash
set -euo pipefail

run_id="${STAGE159_RUN_ID:-ATTR-STAGE159-COMPONENT-VALIDATION-R2}"
base="/root/autodl-tmp/vcas"
py="/root/miniconda3/bin/python"
code_root="${STAGE159_CODE_ROOT:-${base}/code/stage159_component_validation_r2}"
eval_script="${code_root}/scripts/evaluate_stage159_component_validation.py"
labels="${base}/code/config/vehicle_labels.v2.json"
baseline="${base}/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
production_config="${code_root}/production_vehicle_analytics.yaml"
views="${base}/datasets/attribute-domain-v2/stage159-validation-views-r3"
body_manifest="${views}/body.validation-only.csv"
color_manifest="${views}/color.validation-only.csv"
body_state="${base}/runs/attributes/ATTR-STAGE156-FULLSCALE-BODY-TRAIN-R1.state.json"
color_state="${base}/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2.state.json"
output_root="${base}/runs/attributes/${run_id}"
outer_log="${base}/runs/attributes/${run_id}.log"
state="${base}/runs/attributes/${run_id}.state.json"

test ! -e "${output_root}"
test ! -e "${state}"
exec > >(tee -a "${outer_log}") 2>&1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "${path}"
  actual="$(sha256sum "${path}" | awk '{print tolower($1)}')"
  test "${actual}" = "${expected,,}"
}

assert_sha256 "${eval_script}" e8147f470e7a3d48f46d67c4b41f32bc7963c7248da1a0f640f069a9ed037e1d
assert_sha256 "${code_root}/scripts/test_evaluate_stage159_component_validation.py" a49f487298b8154922f64c6b184a4d6b581e32add3cf6fc737d6a1d437ca6ff3
assert_sha256 "${code_root}/scripts/evaluate_stage109_color_class_thresholds.py" 5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
assert_sha256 "${code_root}/scripts/evaluate_v2_decoupled_shared_validation.py" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
assert_sha256 "${labels}" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "${baseline}" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "${production_config}" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa
assert_sha256 "${body_manifest}" 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406
assert_sha256 "${color_manifest}" 34cb19d09eb31c92cd832a44a072a92d27746a8e8df723070d50488ca6e3b023
cd "${code_root}/scripts"
"${py}" -m unittest -v test_evaluate_stage159_component_validation.py

echo "waiting for Stage158 fresh color training to complete"
while tmux has-session -t VCAS-STAGE158-FRESH-COLOR-TRAIN-R2 2>/dev/null; do
  sleep 30
done
while pgrep -f '[t]rain_attribute.py .*ATTR-STAGE158' >/dev/null 2>&1; do
  sleep 15
done
mkdir -p "${output_root}"

mapfile -t body_candidates < <("${py}" - "${body_state}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
if d.get('status')!='training_complete_pending_stage157_validation': raise SystemExit(f"body state not ready: {d.get('status')}")
seen=set()
for variant,data in sorted(d['candidates'].items()):
 for artifact in ('best.pt','last.pt'):
  meta=data['artifacts'][artifact]; path=Path(meta['path']); actual=hashlib.sha256(path.read_bytes()).hexdigest()
  if actual!=meta['sha256']: raise SystemExit(f'body checkpoint SHA mismatch: {path}')
  if actual not in seen:
   seen.add(actual); print(f"{variant}__{artifact[:-3]}\t{path}\t{actual}")
PY
)
mapfile -t color_candidates < <("${py}" - "${color_state}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8'))
if d.get('status')!='training_complete_pending_stage159_validation': raise SystemExit(f"color state not ready: {d.get('status')}")
seen=set()
for variant,data in sorted(d['candidates'].items()):
 for artifact in ('best.pt','last.pt'):
  meta=data['artifacts'][artifact]; path=Path(meta['path']); actual=hashlib.sha256(path.read_bytes()).hexdigest()
  if actual!=meta['sha256']: raise SystemExit(f'color checkpoint SHA mismatch: {path}')
  if actual not in seen:
   seen.add(actual); print(f"{variant}__{artifact[:-3]}\t{path}\t{actual}")
PY
)
test "${#body_candidates[@]}" -ge 2
test "${#color_candidates[@]}" -ge 2

for entry in "${body_candidates[@]}"; do
  IFS=$'\t' read -r variant checkpoint checkpoint_sha <<<"${entry}"
  set +e
  "${py}" "${eval_script}" \
    --head body \
    --manifest "${body_manifest}" --expected-manifest-sha256 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406 \
    --labels "${labels}" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --candidate-checkpoint "${checkpoint}" --expected-candidate-sha256 "${checkpoint_sha}" \
    --baseline-checkpoint "${baseline}" --expected-baseline-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --production-config "${production_config}" --expected-production-config-sha256 fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa \
    --resolution-root "${base}" --allowed-image-root "${base}/datasets" --allowed-image-root "${base}/sources" \
    --precision-target 0.935 --device cuda --batch-size 64 --workers 8 \
    --output "${output_root}/body/${variant}/report.json"
  code=$?
  set -e
  test "${code}" = 0 -o "${code}" = 2
done

for entry in "${color_candidates[@]}"; do
  IFS=$'\t' read -r variant checkpoint checkpoint_sha <<<"${entry}"
  set +e
  "${py}" "${eval_script}" \
    --head color \
    --manifest "${color_manifest}" --expected-manifest-sha256 34cb19d09eb31c92cd832a44a072a92d27746a8e8df723070d50488ca6e3b023 \
    --labels "${labels}" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --candidate-checkpoint "${checkpoint}" --expected-candidate-sha256 "${checkpoint_sha}" \
    --baseline-checkpoint "${baseline}" --expected-baseline-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --production-config "${production_config}" --expected-production-config-sha256 fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa \
    --resolution-root "${base}" --allowed-image-root "${base}/datasets" --allowed-image-root "${base}/sources" \
    --precision-target 0.935 --device cuda --batch-size 64 --workers 8 \
    --output "${output_root}/color/${variant}/report.json"
  code=$?
  set -e
  test "${code}" = 0 -o "${code}" = 2
done

"${py}" - "${output_root}" "${state}" "${body_state}" "${color_state}" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root,state,body_state,color_state=map(Path,sys.argv[1:])
def collect(head):
 out=[]
 for path in sorted((root/head).glob('*/report.json')):
  d=json.loads(path.read_text(encoding='utf-8'))
  if d['policy']['split']!='validation' or d['policy']['test_accessed'] is not False or d['policy']['frozen_video_used'] is not False: raise SystemExit('validation isolation failed')
  out.append({'variant':path.parent.name,'qualified':bool(d['gates']['all_pass']),'checkpoint':d['inputs']['candidate_checkpoint'],'checkpoint_sha256':d['inputs']['candidate_checkpoint_sha256'],'thresholds':d['selection']['thresholds'],'overall':d['static']['candidate_overall'],'complex':d['static']['candidate_complex'],'comparison':d['comparison'],'track':d['track_fusion'],'per_class':d['per_class'],'stratified':d['stratified'],'gates':d['gates']['gates'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 return out
body=collect('body'); color=collect('color')
body_pass=[v for v in body if v['qualified']]; color_pass=[v for v in color if v['qualified']]
selected_body=max(body_pass,key=lambda v:(v['complex']['coverage'],v['overall']['coverage'],v['track']['track_final']['coverage'],v['overall']['precision'])) if body_pass else None
selected_color=max(color_pass,key=lambda v:(v['comparison']['color_complex_unknown_relative_reduction'],v['overall']['coverage'],v['track']['track_final']['coverage'],v['overall']['precision'])) if color_pass else None
authorized=selected_body is not None and selected_color is not None
out={'schema_version':'stage159-component-validation-state-v1','created_at':datetime.now(timezone.utc).isoformat(),'status':'pass_stage160_new_holdout_build_authorized' if authorized else 'component_repair_required_fail_closed','stage160_new_holdout_build_authorized':authorized,'selected_body':selected_body,'selected_color':selected_color,'body_variants':body,'color_variants':color,'inputs':{'body_training_state_sha256':hashlib.sha256(body_state.read_bytes()).hexdigest(),'color_training_state_sha256':hashlib.sha256(color_state.read_bytes()).hexdigest()},'test_accessed':False,'stage148_test_reused':False,'stage155_holdout_reused':False,'frozen_video_used':False,'production_model_modified':False,'backend_gates_run':False,'deployment_performed':False}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'body_qualified':[v['variant'] for v in body_pass],'color_qualified':[v['variant'] for v in color_pass]},ensure_ascii=False))
PY

find "${output_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${output_root}/SHA256SUMS"
