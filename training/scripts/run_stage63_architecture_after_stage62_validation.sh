#!/usr/bin/env bash
set -euo pipefail

validation_session="vcas_stage62_domain_validation"
validation_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE62-DOMAIN-VALIDATION-V1.state.json"
project_root="/root/autodl-tmp/vcas/code"
python_bin="/root/miniconda3/bin/python"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE63-ARCHITECTURE-DOMAIN-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE63-ARCHITECTURE-DOMAIN-V1.state.json"

while tmux has-session -t "${validation_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${validation_state}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("fail_closed: Stage62 validation state is missing")
state = json.loads(path.read_text(encoding="utf-8"))
if state.get("status") != "complete":
    raise SystemExit(f"fail_closed: Stage62 validation did not complete: {state.get('status')}")
for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
    if state.get(key) is not False:
        raise SystemExit(f"fail_closed: Stage62 validation policy violation: {key}")
PY

test ! -e "${output_root}"
test ! -e "${state_path}"
cd "${project_root}"
exec "${python_bin}" training/scripts/run_stage62_domain_consistency_matrix.py \
  --matrix /root/autodl-tmp/vcas/runs/attributes/plans/stage63-architecture-domain-matrix.json \
  --supervised-manifest /root/autodl-tmp/vcas/datasets/attribute-domain-v2/attribute_manifest.stage59-preweather-hard-v2.csv \
  --supervised-report /root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage59-preweather-v2/manifest-report.json \
  --unlabeled-manifest /root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage61-combined-adverse-proposals-v1/attribute-proposals.csv \
  --unlabeled-report /root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage61-combined-adverse-proposals-v1/combine-report.json \
  --attribution /root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage61-combined-adverse-proposals-v1/openimages-attribution.csv \
  --attribution-report /root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage61-combined-adverse-proposals-v1/attribution-report.json \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --code-revision stage63-architecture-domain-v1 \
  --device cuda
