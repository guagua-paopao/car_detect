#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp/vcas
py=/root/miniconda3/bin/python
code_root=${base}/code/stage159_component_validation_r6
eval_script=${code_root}/scripts/evaluate_stage159_component_validation.py
labels=${base}/code/config/vehicle_labels.v2.json
baseline=${base}/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
production_config=${code_root}/production_vehicle_analytics.yaml
views=${base}/datasets/attribute-domain-v2/stage159-validation-views-r3
body_manifest=${views}/body.validation-only.csv
stage161_root=${base}/runs/attributes/ATTR-STAGE161-BODY-FOCUS-DISTILL-R1
output_root=${base}/runs/attributes/ATTR-STAGE162-BODY-VALIDATION-R2
outer_log=${base}/runs/attributes/ATTR-STAGE162-BODY-VALIDATION-R2.log
state=${base}/runs/attributes/ATTR-STAGE162-BODY-VALIDATION-R2.state.json

test ! -e "${output_root}"; test ! -e "${state}"
exec > >(tee -a "${outer_log}") 2>&1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "${path}"
  actual="$(sha256sum "${path}" | awk '{print tolower($1)}')"
  test "${actual}" = "${expected,,}"
}

assert_sha256 "${eval_script}" e8147f470e7a3d48f46d67c4b41f32bc7963c7248da1a0f640f069a9ed037e1d
assert_sha256 "${code_root}/scripts/test_evaluate_stage159_component_validation.py" a49f487298b8154922f64c6b184a4d6b581e32add3cf6fc737d6a1d437ca6ff3
assert_sha256 "${code_root}/scripts/evaluate_stage109_color_class_thresholds.py" 5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
assert_sha256 "${code_root}/scripts/evaluate_v2_decoupled_shared_validation.py" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
assert_sha256 "${labels}" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "${baseline}" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "${production_config}" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa
assert_sha256 "${body_manifest}" 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406

cd "${code_root}/scripts"
"${py}" -m unittest -v test_evaluate_stage159_component_validation.py

while tmux has-session -t VCAS-STAGE161-BODY-FOCUS-R2 2>/dev/null; do sleep 30; done
while pgrep -f '[t]rain_attribute.py .*stage161-body-focus-r1' >/dev/null 2>&1; do sleep 15; done

for variant in ATTR-STAGE161-BODY-CONVNEXT-256-FOCUS-R1 ATTR-STAGE161-BODY-CONVNEXT-288-FOCUS-R1; do
  for artifact in best.pt last.pt; do
    test -s "${stage161_root}/${variant}/${artifact}"
    test -s "${stage161_root}/${variant}/metrics.json"
    test -s "${stage161_root}/${variant}/model_card.json"
  done
done

mkdir -p "${output_root}/body"
for variant in ATTR-STAGE161-BODY-CONVNEXT-256-FOCUS-R1 ATTR-STAGE161-BODY-CONVNEXT-288-FOCUS-R1; do
  for artifact in best last; do
    checkpoint=${stage161_root}/${variant}/${artifact}.pt
    checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print tolower($1)}')
    report_dir=${output_root}/body/${variant}__${artifact}
    set +e
    "${py}" "${eval_script}" --head body \
      --manifest "${body_manifest}" --expected-manifest-sha256 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406 \
      --labels "${labels}" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --candidate-checkpoint "${checkpoint}" --expected-candidate-sha256 "${checkpoint_sha}" \
      --baseline-checkpoint "${baseline}" --expected-baseline-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --production-config "${production_config}" --expected-production-config-sha256 fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa \
      --resolution-root "${base}" --allowed-image-root "${base}/datasets" --allowed-image-root "${base}/sources" \
      --precision-target 0.935 --device cuda --batch-size 64 --workers 8 --output "${report_dir}/report.json"
    code=$?
    set -e
    test "${code}" = 0 -o "${code}" = 2
  done
done

"${py}" - "${output_root}" "${state}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root, state = map(Path, sys.argv[1:])
reports = []
for path in sorted((root / "body").glob("*/report.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    policy = data.get("policy", {})
    if policy.get("split") != "validation" or policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
        raise SystemExit(f"validation isolation failed: {path}")
    reports.append({"variant": path.parent.name, "qualified": bool(data["gates"]["all_pass"]), "candidate_checkpoint": data["inputs"]["candidate_checkpoint"], "candidate_checkpoint_sha256": data["inputs"]["candidate_checkpoint_sha256"], "overall": data["static"]["candidate_overall"], "complex": data["static"]["candidate_complex"], "comparison": data["comparison"], "track": data["track_fusion"], "gates": data["gates"]["gates"], "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
if len(reports) != 4:
    raise SystemExit(f"expected 4 reports, got {len(reports)}")
qualified = [r for r in reports if r["qualified"]]
selected = max(qualified, key=lambda r: (r["complex"]["coverage"], r["overall"]["coverage"], r["track"]["track_final"]["coverage"], r["overall"]["precision"])) if qualified else None
result = {"schema_version": "stage162-body-validation-state-v1", "created_at": datetime.now(timezone.utc).isoformat(), "status": "body_candidate_pass" if selected else "body_candidates_rejected_fail_closed", "selected": selected, "candidates": reports, "selection_split": "validation", "precision_target": 0.935, "test_accessed": False, "frozen_video_used": False, "production_model_modified": False, "backend_gates_run": False, "deployment_performed": False}
state.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(state.with_suffix(state.suffix + ".sha256")).write_text(hashlib.sha256(state.read_bytes()).hexdigest() + "  " + state.name + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "selected": selected["variant"] if selected else None}, ensure_ascii=False))
PY

find "${output_root}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${output_root}/SHA256SUMS"
