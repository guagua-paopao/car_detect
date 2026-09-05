#!/usr/bin/env bash
set -euo pipefail

validation_session="vcas_stage64_taxonomy_v2_validation"
validation_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-TAXONOMY-V2-VALIDATION-V1.state.json"
training_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-TAXONOMY-V2-V1.state.json"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-ADVERSE-PARTIAL-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-ADVERSE-PARTIAL-V1.state.json"
leakage_session="vcas_stage65_leakage_audit"
leakage_report="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-PERCEPTUAL-LEAKAGE-AUDIT-V1.json"

while tmux has-session -t "${validation_session}" 2>/dev/null; do
  sleep 30
done

while tmux has-session -t "${leakage_session}" 2>/dev/null; do
  sleep 30
done

/root/miniconda3/bin/python - "${validation_state}" "${training_state}" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"fail_closed: missing Stage64 evidence: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = "complete" if "VALIDATION" in path.name else "complete_validation_only"
    if state.get("status") != expected:
        raise SystemExit(f"fail_closed: Stage64 state did not complete: {path}: {state.get('status')}")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if state.get(key) is not False:
            raise SystemExit(f"fail_closed: Stage64 policy violation: {key}")
PY

/root/miniconda3/bin/python - "${leakage_report}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("fail_closed: Stage65 perceptual leakage report is missing")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "pass":
    raise SystemExit(f"fail_closed: Stage65 perceptual leakage audit failed: {report.get('status')}")
if report.get("exact_sha_cross_split_overlap") != 0 or report.get("source_group_cross_split_overlap") != 0:
    raise SystemExit("fail_closed: Stage65 exact/group leakage found")
if report.get("dhash", {}).get("cross_split_near_pairs") != 0:
    raise SystemExit("fail_closed: Stage65 perceptual leakage found")
PY

test ! -e "${output_root}"
test ! -e "${state_path}"
cd /root/autodl-tmp/vcas/code
exec /root/miniconda3/bin/python training/scripts/run_stage65_adverse_partial_matrix.py \
  --matrix /root/autodl-tmp/vcas/runs/attributes/plans/stage65-adverse-partial-matrix.json \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --code-revision stage65-adverse-partial-v1 \
  --device cuda
