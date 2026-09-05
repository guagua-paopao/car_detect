#!/usr/bin/env bash
set -euo pipefail

decision_session="vcas_stage67_followup_decision"
decision_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-FOLLOWUP-DECISION-V1.json"
project_root="/root/autodl-tmp/vcas/code"
python_bin="/root/miniconda3/bin/python"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-VALIDATION-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-VALIDATION-V1.state.json"
log_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-VALIDATION-V1.log"

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
  echo "Stage67 validation not required: ${action}"
  exit 0
fi

cd "${project_root}"
exec "${python_bin}" training/scripts/run_stage62_domain_validation_after_training.py \
  --training-state /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-CONTROLLED-V1.state.json \
  --training-matrix /root/autodl-tmp/vcas/runs/attributes/plans/stage67-uvh26-controlled-matrix.json \
  --expected-training-matrix-sha256 ccfae19e5c0be7420b8cd6ac6975e10eb702e2cba51e90e93be7b5e882a4db22 \
  --training-root /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE67-UVH26-CONTROLLED-V1 \
  --training-session vcas_stage67_uvh26 \
  --hard-manifest /root/autodl-tmp/vcas/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv \
  --expected-hard-manifest-sha256 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40 \
  --ua-manifest /root/autodl-tmp/vcas/datasets/attribute-domain-v2/ua-detrac-tracks-v1/attribute_manifest.weather-v2.csv \
  --expected-ua-manifest-sha256 6d8d0e61f64a2a3e837670667ef35e15e83590d1df52321258cf78996001a50f \
  --vfg-manifest /root/autodl-tmp/vcas/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv \
  --expected-vfg-manifest-sha256 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3 \
  --production-checkpoint /root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt \
  --expected-production-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
  --labels /root/autodl-tmp/vcas/code/config/vehicle_labels.v1.json \
  --expected-labels-sha256 c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f \
  --expected-static-evaluator-sha256 974a684d34dc45e4572d8efa5e703bee2de486f3b8ba44649a1909bef37ef2bb \
  --expected-track-sweeper-sha256 e3310eda88ae9e8afec5337408fd5fe9f977a0a176ac1b8f55375722c195e502 \
  --expected-vfg-comparison-sha256 5a924c27bc927a82c79c158341e4c3188333edd76d07f4b4cf84a050a19a6899 \
  --expected-track-evaluator-sha256 20881e01aec678e5f59f2a4eebb851cf9c015ccca5d4009780024a988459da90 \
  --scripts-root /root/autodl-tmp/vcas/code/training/scripts \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --python "${python_bin}" \
  --device cuda \
  --timeout-hours 120 \
  --resume-waiting
