#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage102_partial_color_r1"
TRAIN="$CODE/scripts/train_attribute.py"
DATASET_CODE="$CODE/src/attribute_dataset.py"
HIERARCHY_CODE="$CODE/src/attribute_hierarchy.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST_DIR="$BASE/datasets/attribute-domain-v2/stage150-vehicle-rear-color-domain-r2"
MANIFEST="$MANIFEST_DIR/attribute_manifest.stage150-vehicle-rear-color-domain.csv"
MANIFEST_REPORT="$MANIFEST_DIR/stage150-vehicle-rear-color-domain-manifest-report.json"
MANIFEST_STATE="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-MANIFEST-R2.state.json"
INIT="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2/gate-best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-DOMAIN-R1"
RUN="$ROOT/ATTR-STAGE150-COLOR-CONVNEXT-256-VEHICLE-REAR-DOMAIN-R1"
TRAIN_LOG="$ROOT/train.log"
LAUNCH_LOG="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-DOMAIN-R1.launch.log"
STATE="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-DOMAIN-R1.state.json"
SESSION=VCAS-STAGE150-COLOR-DOMAIN-R1

TRAIN_SHA=65cf1c9bd3ef9c2c0648ff5f8b63bbb47959bb885322af728be9567a1eed6f16
DATASET_SHA=a7cebd29288f7beb627a6ec12777df88628935bd496e7611a732e6b1d953609b
HIERARCHY_SHA=1ffe558a91091ce54ccf885870a5aafb19bdbd912bec8a4935b7a49564f5428f
LABELS_SHA=22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
MANIFEST_SHA=56a421f68cb2455b13707be7d2787fc05d456b385f36ec4af561703db752e63f
MANIFEST_REPORT_SHA=f25b51d02d4ba2c4b29d7a03dc991fa6326a2c14f7c044734bd3a9685f6e97f8
MANIFEST_STATE_SHA=99abbe6d1ffa31bbd79e113b2b22f4beee12eeb4a71f1f5f22b7cac8909085b7
INIT_SHA=811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

write_failure_state() {
  local return_code="$1"
  "$PY" - "$STATE" "$return_code" "$MANIFEST" "$TRAIN_LOG" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import json,sys
state_path=Path(sys.argv[1])
payload={
  'schema_version':'stage150-color-domain-training-state-v1',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'training_failed_preserved',
  'return_code':int(sys.argv[2]),
  'manifest':sys.argv[3],
  'train_log':sys.argv[4],
  'policy':{
    'research_only':True,'deployable':False,'test_accessed':False,
    'stage148_test_reused':False,'frozen_video_used':False,
    'production_model_modified':False,'backend_gates_run':False,'deployment_performed':False,
  },
}
state_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
PY
}

if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash '$0' --worker >>'$LAUNCH_LOG' 2>&1"
  echo "started tmux:$SESSION"
  echo "run=$RUN"
  echo "log=$TRAIN_LOG"
  exit 0
fi

for required in "$PY" "$TRAIN" "$DATASET_CODE" "$HIERARCHY_CODE" "$LABELS" "$MANIFEST" "$MANIFEST_REPORT" "$MANIFEST_STATE" "$INIT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" "$TRAIN_SHA"
assert_sha256 "$DATASET_CODE" "$DATASET_SHA"
assert_sha256 "$HIERARCHY_CODE" "$HIERARCHY_SHA"
assert_sha256 "$LABELS" "$LABELS_SHA"
assert_sha256 "$MANIFEST" "$MANIFEST_SHA"
assert_sha256 "$MANIFEST_REPORT" "$MANIFEST_REPORT_SHA"
assert_sha256 "$MANIFEST_STATE" "$MANIFEST_STATE_SHA"
assert_sha256 "$INIT" "$INIT_SHA"

"$PY" - "$MANIFEST_REPORT" "$MANIFEST_STATE" <<'PY'
import json,sys
report=json.load(open(sys.argv[1],encoding='utf-8'))
state=json.load(open(sys.argv[2],encoding='utf-8'))
assert report['status']=='pass_research_only'
assert state['status']=='pass_research_only_pending_training'
assert report['output']['train_rows']>=118338
assert report['output']['validation_rows']>=500
assert report['output']['accepted_supplement_train']>=3500
assert report['integrity']['train_validation_group_overlap']==0
assert report['integrity']['train_validation_exact_sha_overlap']==0
assert report['integrity']['train_validation_dhash_distance_le_4_rows']==0
assert report['policy']['canonical_dhash_recomputed_from_saved_pixels'] is True
assert report['policy']['legacy_stage102_validation_imported'] is False
assert report['policy']['future_holdout_imported'] is False
assert report['policy']['stage148_test_reused'] is False
assert report['policy']['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN" "$DATASET_CODE" "$HIERARCHY_CODE"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite training root: $ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite training state: $STATE" >&2; exit 67; }
mkdir -p "$ROOT"

set +e
"$PY" "$TRAIN" \
  --manifest "$MANIFEST" \
  --labels "$LABELS" \
  --input-size 256 \
  --architecture convnext_tiny \
  --resize-mode stretch \
  --epochs 6 \
  --batch-size 40 \
  --workers 8 \
  --learning-rate 0.000002 \
  --weight-decay 0.0001 \
  --body-loss-weight 0.0 \
  --color-loss-weight 1.3 \
  --focal-gamma 0.0 \
  --color-focal-gamma 1.5 \
  --class-weighting inverse_sqrt \
  --label-smoothing 0.01 \
  --gradient-clip-norm 5.0 \
  --freeze-backbone-epochs 0 \
  --patience 3 \
  --type-threshold 0.99 \
  --color-threshold 0.70 \
  --gate-type-precision 0.93 \
  --gate-type-coverage 0.45 \
  --gate-color-precision 0.93 \
  --gate-color-coverage 0.25 \
  --seed 20260905 \
  --augmentation-profile color_scene \
  --selection-head color \
  --body-hierarchy none \
  --coarse-car-loss-weight 0.0 \
  --coarse-truck-loss-weight 0.0 \
  --coarse-color-loss-weight 0.5 \
  --night-sample-weight 1.5 \
  --occlusion-sample-weight 1.1 \
  --hard-sample-weight 0.1 \
  --small-sample-weight 0.5 \
  --color-sample-weight 0.25 \
  --pseudo-label-weight 0.5 \
  --init-checkpoint "$INIT" \
  --run-kind formal \
  --dataset-version attribute-domain-v2-stage150-vehicle-rear-color-domain-r2 \
  --code-revision stage150-color-domain-r1-canonical-pixel-dedup \
  --skip-test \
  --output-dir "$RUN" >"$TRAIN_LOG" 2>&1
train_rc=$?
set -e
if [[ "$train_rc" -ne 0 ]]; then
  write_failure_state "$train_rc"
  exit "$train_rc"
fi

for required in "$RUN/best.pt" "$RUN/last.pt" "$RUN/metrics.json" "$RUN/model_card.json"; do
  [[ -f "$required" ]] || { write_failure_state 68; echo "missing training artifact: $required" >&2; exit 68; }
done

"$PY" - "$STATE" "$RUN" "$MANIFEST" "$MANIFEST_REPORT" "$INIT" "$TRAIN_LOG" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
state_path,run_dir,manifest,report,init,train_log=map(Path,sys.argv[1:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
artifacts={}
for name in ('best.pt','last.pt','gate-best.pt','metrics.json','model_card.json','test_metrics.json'):
    path=run_dir/name
    if path.is_file():
        artifacts[name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
payload={
  'schema_version':'stage150-color-domain-training-state-v1',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'training_complete_pending_stage151_validation',
  'run_dir':str(run_dir),
  'manifest':str(manifest),'manifest_sha256':sha(manifest),
  'manifest_report':str(report),'manifest_report_sha256':sha(report),
  'initialization_checkpoint':str(init),'initialization_checkpoint_sha256':sha(init),
  'train_log':str(train_log),
  'artifacts':artifacts,
  'policy':{
    'research_only':True,'deployable':False,'test_accessed':False,
    'stage148_test_reused':False,'frozen_video_used':False,
    'production_model_modified':False,'backend_gates_run':False,'deployment_performed':False,
  },
}
state_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
state_path.with_suffix(state_path.suffix+'.sha256').write_text(
    f"{sha(state_path)}  {state_path.name}\n",encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False))
PY

echo "Stage150 color-domain training complete; Stage151 validation remains locked"
