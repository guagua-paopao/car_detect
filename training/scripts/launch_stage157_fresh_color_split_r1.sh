#!/usr/bin/env bash
set -euo pipefail

run_id="ATTR-STAGE157-FRESH-COLOR-SPLIT-R1"
base="/root/autodl-tmp/vcas"
code_root="${base}/code/stage157_color_validation_r1/scripts"
dataset_root="${base}/datasets/attribute-domain-v2"
input_fine_root="${dataset_root}/stage157-kaggle-fine-color-r1"
output_root="${dataset_root}/stage157-fresh-color-split-r1"
run_root="${base}/runs/attributes/${run_id}"
log="${base}/runs/attributes/${run_id}.log"

test ! -e "${output_root}"
test ! -e "${run_root}"
mkdir -p "${output_root}" "${run_root}"
exec > >(tee -a "${log}") 2>&1

cd "${code_root}"
/root/miniconda3/bin/python -m unittest -v test_build_stage157_fresh_color_split.py
/root/miniconda3/bin/python build_stage157_fresh_color_split.py \
  --stage150-manifest "${dataset_root}/stage150-vehicle-rear-color-domain-r2/attribute_manifest.stage150-vehicle-rear-color-domain.csv" \
  --expected-stage150-sha256 56a421f68cb2455b13707be7d2787fc05d456b385f36ec4af561703db752e63f \
  --fine-cc0-manifest "${input_fine_root}/attribute_manifest.stage157-kaggle-fine-color.csv" \
  --fine-cc0-report "${input_fine_root}/stage157-kaggle-fine-color-recovery-report.json" \
  --vfg-validation-manifest "${dataset_root}/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv" \
  --dataset-root "${dataset_root}" \
  --output-training-manifest "${output_root}/attribute_manifest.stage157-fresh-color-train.csv" \
  --output-validation-manifest "${output_root}/attribute_manifest.stage157-fresh-color-validation.csv" \
  --output-report "${output_root}/stage157-fresh-color-split-report.json" \
  --dvm-target-per-color 250 \
  --cc0-target-per-color 8 \
  --minimum-training-rows 118338 \
  --minimum-validation-per-color 200 \
  --workers 12

sha256sum \
  "${output_root}/attribute_manifest.stage157-fresh-color-train.csv" \
  "${output_root}/attribute_manifest.stage157-fresh-color-validation.csv" \
  "${output_root}/stage157-fresh-color-split-report.json" \
  "${log}" > "${run_root}/SHA256SUMS"
/root/miniconda3/bin/python - <<'PY'
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

base = Path('/root/autodl-tmp/vcas')
run_id = 'ATTR-STAGE157-FRESH-COLOR-SPLIT-R1'
output_root = base / 'datasets/attribute-domain-v2/stage157-fresh-color-split-r1'
run_root = base / 'runs/attributes' / run_id
report_path = output_root / 'stage157-fresh-color-split-report.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
state = {
    'schema_version': 'stage157-fresh-color-split-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'run_id': run_id,
    'status': 'completed_research_only' if report.get('status') == 'pass_research_only' else 'failed_closed',
    'report': str(report_path),
    'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
    'training_manifest': report['output']['training_manifest'],
    'training_manifest_sha256': report['output']['training_manifest_sha256'],
    'validation_manifest': report['output']['validation_manifest'],
    'validation_manifest_sha256': report['output']['validation_manifest_sha256'],
    'required_initialization': report['policy']['required_initialization'],
    'production_modified': False,
    'frozen_video_used': False,
    'deployment_performed': False,
}
state_path = run_root / 'state.json'
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
(run_root / 'state.json.sha256').write_text(
    hashlib.sha256(state_path.read_bytes()).hexdigest() + '  state.json\n', encoding='utf-8'
)
if state['status'] != 'completed_research_only':
    raise SystemExit(2)
PY
