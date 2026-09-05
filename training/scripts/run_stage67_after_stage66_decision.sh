#!/usr/bin/env bash
set -euo pipefail

decision_session="vcas_stage67_followup_decision"
decision_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-FOLLOWUP-DECISION-V1.json"
stage66_report="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-VALIDATION-V1/validation-screen-report.json"
matrix_path="/root/autodl-tmp/vcas/runs/attributes/plans/stage67-uvh26-controlled-matrix.json"
runner="/root/autodl-tmp/vcas/code/training_stage66/scripts/run_stage67_uvh26_controlled_matrix.py"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-CONTROLLED-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-CONTROLLED-V1.state.json"
log_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-CONTROLLED-V1.log"
python_bin="/root/miniconda3/bin/python"

exec >>"${log_path}" 2>&1
while tmux has-session -t "${decision_session}" 2>/dev/null; do
  sleep 30
done

action=$("${python_bin}" - "${decision_path}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("fail_closed: Stage67 decision is missing")
d = json.loads(p.read_text())
if d.get("status") != "pass":
    raise SystemExit("fail_closed: Stage67 decision did not pass")
policy = d.get("policy", {})
for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
    if policy.get(key) is not False:
        raise SystemExit(f"fail_closed: Stage67 decision policy violation: {key}")
if policy.get("deployment_paused_by_user") is not True:
    raise SystemExit("fail_closed: Stage67 decision lost deployment pause")
action = d.get("action", "")
allowed = {
    "skip_stage67_stage66_full_candidate_available",
    "skip_stage67_type_gates_met_focus_next_round_on_color",
    "run_stage67_uvh26_controlled_type_followup",
}
if action not in allowed:
    raise SystemExit(f"fail_closed: unexpected Stage67 action: {action}")
print(action)
PY
)

if [[ "${action}" != "run_stage67_uvh26_controlled_type_followup" ]]; then
  echo "Stage67 training not required: ${action}"
  exit 0
fi

test -f "${stage66_report}"
test -f "${matrix_path}"
test -f "${runner}"
test ! -e "${output_root}"
test ! -e "${state_path}"

exec "${python_bin}" "${runner}" \
  --matrix "${matrix_path}" \
  --decision "${decision_path}" \
  --stage66-validation-report "${stage66_report}" \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --device cuda \
  --code-revision stage67-uvh26-controlled-v1
