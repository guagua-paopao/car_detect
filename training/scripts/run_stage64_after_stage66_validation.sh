#!/usr/bin/env bash
set -euo pipefail

upstream_session="vcas_stage66_licensed_adverse_validation"
upstream_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-VALIDATION-V1.state.json"
decision_session="vcas_stage67_followup_decision"
decision_output="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-FOLLOWUP-DECISION-V1.json"
stage67_validation_session="vcas_stage67_uvh26_validation"
stage67_validation_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-VALIDATION-V1.state.json"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-TAXONOMY-V2-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-TAXONOMY-V2-V1.state.json"

while tmux has-session -t "${upstream_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${decision_session}" 2>/dev/null; do
  sleep 30
done

stage67_action=$("${python_bin}" - "${upstream_state}" "${decision_output}" <<'PY'
import json, sys
from pathlib import Path
state_path, decision_path = map(Path, sys.argv[1:])
if not state_path.is_file():
    raise SystemExit("fail_closed: Stage66 validation state is missing")
state = json.loads(state_path.read_text())
if state.get("status") != "complete":
    raise SystemExit(f"fail_closed: Stage66 validation did not complete: {state.get('status')}")
for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
    if state.get(key) is not False:
        raise SystemExit(f"fail_closed: Stage66 validation policy violation: {key}")
if not decision_path.is_file():
    raise SystemExit("fail_closed: Stage67 follow-up decision is missing")
decision = json.loads(decision_path.read_text())
if decision.get("status") != "pass":
    raise SystemExit("fail_closed: Stage67 follow-up decision did not pass")
policy = decision.get("policy", {})
for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
    if policy.get(key) is not False:
        raise SystemExit(f"fail_closed: Stage67 decision policy violation: {key}")
if policy.get("deployment_paused_by_user") is not True:
    raise SystemExit("fail_closed: Stage67 decision lost deployment pause")
action = decision.get("action", "")
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

if [[ "${stage67_action}" == "run_stage67_uvh26_controlled_type_followup" ]]; then
  while true; do
    if [[ -f "${stage67_validation_state}" ]]; then
      stage67_status=$("${python_bin}" - "${stage67_validation_state}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
status = d.get("status")
if status not in {"waiting_for_training", "running", "complete"}:
    raise SystemExit(f"fail_closed: unexpected Stage67 validation status: {status}")
if status == "complete":
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
        if d.get(key) is not False:
            raise SystemExit(f"fail_closed: Stage67 validation policy violation: {key}")
print(status)
PY
)
      if [[ "${stage67_status}" == "complete" ]]; then
        break
      fi
    fi
    if ! tmux has-session -t "${stage67_validation_session}" 2>/dev/null; then
      echo "fail_closed: Stage67 validation session ended without a complete state" >&2
      exit 1
    fi
    sleep 60
  done
fi

test ! -e "${output_root}"
test ! -e "${state_path}"
cd "${project_root}"
exec "${python_bin}" training/scripts/run_stage64_taxonomy_v2_matrix.py \
  --matrix /root/autodl-tmp/vcas/runs/attributes/plans/stage64-taxonomy-v2-matrix.json \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --device cuda \
  --code-revision stage64-taxonomy-v2-v1
