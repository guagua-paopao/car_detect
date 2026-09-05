#!/usr/bin/env bash
set -euo pipefail

validation_session="vcas_stage63_architecture_validation"
validation_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE63-ARCHITECTURE-VALIDATION-V1.state.json"
project_root="/root/autodl-tmp/vcas/code"
python_bin="/root/miniconda3/bin/python"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-TAXONOMY-V2-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-TAXONOMY-V2-V1.state.json"

while tmux has-session -t "${validation_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${validation_state}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("fail_closed: Stage63 validation state is missing")
state = json.loads(path.read_text(encoding="utf-8"))
if state.get("status") != "complete":
    raise SystemExit(f"fail_closed: Stage63 validation did not complete: {state.get('status')}")
for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
    if state.get(key) is not False:
        raise SystemExit(f"fail_closed: Stage63 validation policy violation: {key}")
PY

test ! -e "${output_root}"
test ! -e "${state_path}"
cd "${project_root}"
exec "${python_bin}" training/scripts/run_stage64_taxonomy_v2_matrix.py \
  --matrix /root/autodl-tmp/vcas/runs/attributes/plans/stage64-taxonomy-v2-matrix.json \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --code-revision stage64-taxonomy-v2-v1 \
  --device cuda
