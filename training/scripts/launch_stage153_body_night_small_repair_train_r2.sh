#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage141_letterbox_r1"
TRAIN="$CODE/scripts/train_attribute.py"
HIERARCHY="$CODE/src/attribute_hierarchy.py"
MODEL="$CODE/src/multitask_mobilenet_v3.py"
COMMON="$CODE/src/common.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
DATA="$BASE/datasets/attribute-domain-v2/stage153-lvad-body-repair-r3"
MANIFEST="$DATA/attribute_manifest.stage153-lvad-body-repair.csv"
REPORT="$DATA/stage153-lvad-body-repair-manifest-report.json"
MANIFEST_STATE="$BASE/runs/attributes/ATTR-STAGE153-LVAD-BODY-REPAIR-MANIFEST-R3.state.json"
INIT="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE153-BODY-NIGHT-SMALL-REPAIR-R2"
RUN="$ROOT/ATTR-STAGE153-BODY-CONVNEXT-256-LVAD-REPAIR-R2"
STATE="$ROOT.state.json"

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

[[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage153 R2 training" >&2; exit 66; }
assert_sha256 "$TRAIN" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
assert_sha256 "$HIERARCHY" 1ffe558a91091ce54ccf885870a5aafb19bdbd912bec8a4935b7a49564f5428f
assert_sha256 "$MODEL" a66e1fc94a538cd29cebcf565d0e8a0e8a694be47e62572e5f25c038e6319bb8
assert_sha256 "$COMMON" a7be6dad0629cd0e4cf50955731558339351b8653bf681c4b7874592e2713a1d
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" ae2adb257dee46eb0e456e3174051c48f540cc57f2d051799d4a1b4b32c6d579
assert_sha256 "$REPORT" 5c2c4249ccd10f94194828bddd50f4b18a88c0595b9e41fc026320561b7b4371
assert_sha256 "$MANIFEST_STATE" 9e787ae445da0de5911ae042af73fb930cda69d9887c5a294b0528e39b514918
assert_sha256 "$INIT" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec

"$PY" - "$REPORT" "$MANIFEST_STATE" <<'PY'
import csv,json,sys
report=json.load(open(sys.argv[1],encoding='utf-8'))
state=json.load(open(sys.argv[2],encoding='utf-8'))
assert report['status']=='pass_training_manifest_ready' and state['status']=='pass_training_manifest_ready'
assert report['output']['train_rows']==25157 and report['output']['real_night_coarse_rows']==7657
assert report['output']['rejected_ambiguous_truck_bus_rows']==730
assert report['output']['coarse_family_counts']=={'car':7657}
assert report['output']['quotas']['night_fraction']>=0.30
assert report['output']['quotas']['small_fraction']>=0.20
assert report['output']['quotas']['occluded_or_truncated_fraction']>=0.15
assert report['leakage']['train_validation_group_overlap']==0
assert report['leakage']['train_validation_exact_sha_overlap']==0
assert report['leakage']['train_rows_near_validation_dhash']==0
assert report['policy']['exact_replay_coarse_metadata_cleared'] is True
assert report['policy']['mixed_truck_bus_rows_excluded'] is True
assert report['policy']['fine_labels_fabricated'] is False
assert report['policy']['stage148_test_reused'] is False and report['policy']['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN" "$HIERARCHY" "$MODEL" "$COMMON"

mkdir -p "$ROOT"
set +e
"$PY" "$TRAIN" \
  --manifest "$MANIFEST" --labels "$LABELS" \
  --input-size 256 --architecture convnext_tiny --resize-mode stretch \
  --epochs 6 --batch-size 40 --workers 8 --learning-rate 0.000001 --weight-decay 0.0001 \
  --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 1.5 --color-focal-gamma 0.0 \
  --class-weighting none --label-smoothing 0.01 --gradient-clip-norm 5.0 \
  --freeze-backbone-epochs 1 --patience 4 \
  --type-threshold 0.80 --color-threshold 0.99 --gate-type-precision 0.93 --gate-type-coverage 0.45 \
  --seed 20260830 --augmentation-profile hard_scene --selection-head body \
  --body-hierarchy truck_family --truck-subtype-threshold 0.65 \
  --coarse-car-loss-weight 0.20 --coarse-truck-loss-weight 0.0 \
  --night-sample-weight 1.5 --occlusion-sample-weight 1.25 --hard-sample-weight 0.2 \
  --small-sample-weight 1.25 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
  --init-checkpoint "$INIT" --run-kind formal \
  --dataset-version attribute-domain-v2-stage153-lvad-body-night-small-repair-r3 \
  --code-revision stage153-body-night-small-repair-r2 --skip-test \
  --output-dir "$RUN" >"$ROOT/train.log" 2>&1
TRAIN_RC=$?
set -e

"$PY" - "$RUN" "$STATE" "$TRAIN_RC" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
run,state,return_code=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
artifacts={}
for name in ('best.pt','last.pt','gate-best.pt','metrics.json','model_card.json','test_metrics.json'):
    path=run/name
    if path.is_file():artifacts[name]={'path':str(path),'sha256':sha(path)}
test_status=None
if (run/'test_metrics.json').is_file():test_status=json.loads((run/'test_metrics.json').read_text(encoding='utf-8')).get('status')
complete=return_code==0 and all(name in artifacts for name in ('best.pt','last.pt','metrics.json','model_card.json','test_metrics.json')) and test_status=='not_run'
payload={
  'schema_version':'stage153-body-night-small-repair-training-state-v2',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'training_complete_pending_stage154_validation' if complete else 'training_failed_closed',
  'return_code':return_code,'run':str(run),'artifacts':artifacts,'test_metrics_status':test_status,
  'manifest_sha256':'ae2adb257dee46eb0e456e3174051c48f540cc57f2d051799d4a1b4b32c6d579',
  'initialization_checkpoint_sha256':'e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec',
  'research_only':True,'non_deployable':True,'test_accessed':False,'stage148_test_reused':False,
  'frozen_video_used':False,'production_model_modified':False,'backend_gates_run':False,'deployment_performed':False,
}
state.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False))
PY
exit "$TRAIN_RC"
