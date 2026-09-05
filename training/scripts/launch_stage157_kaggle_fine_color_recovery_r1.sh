#!/usr/bin/env bash
set -euo pipefail

run_id="ATTR-STAGE157-KAGGLE-FINE-COLOR-RECOVERY-R1"
base="/root/autodl-tmp/vcas"
code_root="${base}/code/stage157_color_validation_r1/scripts"
source_root="${base}/sources/kaggle-vehicle-color-cc0-v1"
input_root="${base}/datasets/attribute-domain-v2/kaggle-color-cc0-v1"
output_root="${base}/datasets/attribute-domain-v2/stage157-kaggle-fine-color-r1"
run_root="${base}/runs/attributes/${run_id}"
log="${base}/runs/attributes/${run_id}.log"

test ! -e "${output_root}"
test ! -e "${run_root}"
mkdir -p "${output_root}" "${run_root}"
exec > >(tee -a "${log}") 2>&1

cd "${code_root}"
/root/miniconda3/bin/python -m unittest -v test_recover_stage157_kaggle_cc0_fine_colors.py
/root/miniconda3/bin/python recover_stage157_kaggle_cc0_fine_colors.py \
  --archive "${source_root}/archive.zip" \
  --expected-archive-sha256 a9be2abeb7e497ce6dc7f11204f957b594b80b6efc761bc433dbfa0f673b50d6 \
  --input-manifest "${input_root}/attribute_supplement.csv" \
  --expected-input-sha256 fb08331aaa567025c4621ad0b5c036666998ae731b529c54eb8f916602662063 \
  --output-manifest "${output_root}/attribute_manifest.stage157-kaggle-fine-color.csv" \
  --output-report "${output_root}/stage157-kaggle-fine-color-recovery-report.json"

sha256sum \
  "${output_root}/attribute_manifest.stage157-kaggle-fine-color.csv" \
  "${output_root}/stage157-kaggle-fine-color-recovery-report.json" \
  "${log}" > "${run_root}/SHA256SUMS"
/root/miniconda3/bin/python - <<'PY'
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

base = Path('/root/autodl-tmp/vcas')
run_id = 'ATTR-STAGE157-KAGGLE-FINE-COLOR-RECOVERY-R1'
output_root = base / 'datasets/attribute-domain-v2/stage157-kaggle-fine-color-r1'
run_root = base / 'runs/attributes' / run_id
report_path = output_root / 'stage157-kaggle-fine-color-recovery-report.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
state = {
    'schema_version': 'stage157-kaggle-fine-color-recovery-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'run_id': run_id,
    'status': 'completed' if report.get('status') == 'pass' else 'failed_closed',
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
if state['status'] != 'completed':
    raise SystemExit(2)
PY
