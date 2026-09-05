#!/usr/bin/env bash
set -euo pipefail

run_id="ATTR-STAGE159-VALIDATION-VIEWS-R3"
base="/root/autodl-tmp/vcas"
code_root="${base}/code/stage159_validation_r3/scripts"
dataset_root="${base}/datasets/attribute-domain-v2"
output_root="${dataset_root}/stage159-validation-views-r3"
run_root="${base}/runs/attributes/${run_id}"
log="${base}/runs/attributes/${run_id}.log"

test ! -e "${output_root}"
test ! -e "${run_root}"
mkdir -p "${output_root}" "${run_root}"
exec > >(tee -a "${log}") 2>&1

cd "${code_root}"
/root/miniconda3/bin/python -m unittest -v test_build_stage159_validation_views.py
/root/miniconda3/bin/python build_stage159_validation_views.py \
  --body-manifest "${dataset_root}/stage156-fullscale-body-repair-r1/attribute_manifest.stage156-fullscale-body-repair.csv" \
  --expected-body-manifest-sha256 d9f4e6f8718f389f1669b20933bb5089382bc84fba23dcd0f89a0bb5166194d5 \
  --color-manifest "${dataset_root}/stage157-fresh-color-combined-r2/attribute_manifest.stage157-fresh-color-combined.csv" \
  --expected-color-manifest-sha256 db92921c1633183b018be70091dfe9f26cc55fa942b844c4749b2add5e430f23 \
  --dataset-root "${dataset_root}" \
  --sources-root "${base}/sources" \
  --output-body "${output_root}/body.validation-only.csv" \
  --output-color "${output_root}/color.validation-only.csv" \
  --output-report "${output_root}/stage159-validation-views-report.json"

sha256sum "${output_root}"/* "${log}" > "${run_root}/SHA256SUMS"
/root/miniconda3/bin/python - <<'PY'
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

base = Path('/root/autodl-tmp/vcas')
run_id = 'ATTR-STAGE159-VALIDATION-VIEWS-R3'
output_root = base / 'datasets/attribute-domain-v2/stage159-validation-views-r3'
run_root = base / 'runs/attributes' / run_id
report_path = output_root / 'stage159-validation-views-report.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
state = {
    'schema_version': 'stage159-validation-views-state-v3',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'completed_validation_only' if report.get('status') == 'pass_validation_only' else 'failed_closed',
    'report': str(report_path),
    'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
    'body_manifest': report['output']['body_manifest'],
    'body_manifest_sha256': report['output']['body_manifest_sha256'],
    'color_manifest': report['output']['color_manifest'],
    'color_manifest_sha256': report['output']['color_manifest_sha256'],
    'test_accessed': False,
    'frozen_video_used': False,
    'production_modified': False,
}
state_path = run_root / 'state.json'
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
(run_root / 'state.json.sha256').write_text(
    hashlib.sha256(state_path.read_bytes()).hexdigest() + '  state.json\n', encoding='utf-8'
)
if state['status'] != 'completed_validation_only':
    raise SystemExit(2)
PY
