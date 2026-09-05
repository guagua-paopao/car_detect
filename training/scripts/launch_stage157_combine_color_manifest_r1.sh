#!/usr/bin/env bash
set -euo pipefail

run_id="ATTR-STAGE157-COMBINE-COLOR-MANIFEST-R1"
base="/root/autodl-tmp/vcas"
code_root="${base}/code/stage157_color_validation_r2/scripts"
split_root="${base}/datasets/attribute-domain-v2/stage157-fresh-color-split-r2"
output_root="${base}/datasets/attribute-domain-v2/stage157-fresh-color-combined-r1"
run_root="${base}/runs/attributes/${run_id}"
log="${base}/runs/attributes/${run_id}.log"

test ! -e "${output_root}"
test ! -e "${run_root}"
mkdir -p "${output_root}" "${run_root}"
exec > >(tee -a "${log}") 2>&1

/root/miniconda3/bin/python "${code_root}/combine_stage157_color_train_validation.py" \
  --split-report "${split_root}/stage157-fresh-color-split-report.json" \
  --train-manifest "${split_root}/attribute_manifest.stage157-fresh-color-train.csv" \
  --validation-manifest "${split_root}/attribute_manifest.stage157-fresh-color-validation.csv" \
  --output-manifest "${output_root}/attribute_manifest.stage157-fresh-color-combined.csv" \
  --output-report "${output_root}/stage157-combined-color-manifest-report.json"

sha256sum \
  "${output_root}/attribute_manifest.stage157-fresh-color-combined.csv" \
  "${output_root}/stage157-combined-color-manifest-report.json" \
  "${log}" > "${run_root}/SHA256SUMS"
/root/miniconda3/bin/python - <<'PY'
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

base = Path('/root/autodl-tmp/vcas')
run_id = 'ATTR-STAGE157-COMBINE-COLOR-MANIFEST-R1'
output_root = base / 'datasets/attribute-domain-v2/stage157-fresh-color-combined-r1'
run_root = base / 'runs/attributes' / run_id
report_path = output_root / 'stage157-combined-color-manifest-report.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
state = {
    'schema_version': 'stage157-combined-color-manifest-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'completed_research_only' if report.get('status') == 'pass_research_only' else 'failed_closed',
    'report': str(report_path),
    'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
    'manifest': report['output']['manifest'],
    'manifest_sha256': report['output']['manifest_sha256'],
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
