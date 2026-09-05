#!/usr/bin/env bash
set -euo pipefail

upstream_session="vcas_stage63_architecture_validation"
upstream_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE63-ARCHITECTURE-VALIDATION-V1.state.json"
dedup_session="vcas_stage66_hash_dedup"
dedup_report="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-HASH-DEDUP-V2.report.json"
near_dedup_session="vcas_stage66_near_dedup"
near_dedup_report="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-NEAR-DEDUP-V3.report.json"
strata_session="vcas_stage66_strata_v3"
leakage_session="vcas_stage66_full_leakage_v3"
matrix_path="/root/autodl-tmp/vcas/runs/attributes/plans/stage66-licensed-adverse-architecture-matrix.json"
python_bin="/root/miniconda3/bin/python"
training_root="/root/autodl-tmp/vcas/code/training_stage66"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-V1.state.json"
log_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE66-LICENSED-ADVERSE-V1.log"

exec >>"${log_path}" 2>&1
while tmux has-session -t "${upstream_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${dedup_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${near_dedup_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${strata_session}" 2>/dev/null; do
  sleep 30
done
while tmux has-session -t "${leakage_session}" 2>/dev/null; do
  sleep 30
done

"${python_bin}" - "${upstream_state}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("fail_closed: Stage63 validation state is missing")
d = json.loads(p.read_text())
if d.get("status") != "complete":
    raise SystemExit(f"fail_closed: Stage63 validation did not complete: {d.get('status')}")
for key in ("test_accessed", "frozen_video_used", "production_model_modified", "deployment_performed"):
    if d.get(key) is not False:
        raise SystemExit(f"fail_closed: Stage63 validation policy violation: {key}")
if d.get("evaluator_policy_revision") != "attribute-truth-routed-v2":
    raise SystemExit("fail_closed: Stage63 validation did not use truth-routed v2")
PY

"${python_bin}" - "${dedup_report}" "${near_dedup_report}" "${matrix_path}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report_path, near_report_path, matrix_path = map(Path, sys.argv[1:])
if not report_path.is_file():
    raise SystemExit("fail_closed: Stage66 full-crop hash/dedup report is missing")
report = json.loads(report_path.read_text())
if report.get("status") != "pass":
    raise SystemExit(f"fail_closed: Stage66 full-crop hash/dedup failed: {report.get('status')}")
intermediate_manifest = Path(report["output_manifest"])
if not intermediate_manifest.is_file() or sha256(intermediate_manifest) != report.get("output_manifest_sha256"):
    raise SystemExit("fail_closed: Stage66 hashed manifest does not match its report")
if not near_report_path.is_file():
    raise SystemExit("fail_closed: Stage66 near-dedup report is missing")
near_report = json.loads(near_report_path.read_text())
if near_report.get("status") != "pass":
    raise SystemExit(f"fail_closed: Stage66 near-dedup failed: {near_report.get('status')}")
if near_report.get("input_manifest_sha256") != sha256(intermediate_manifest):
    raise SystemExit("fail_closed: Stage66 near-dedup input lineage mismatch")
manifest = Path(near_report["output_manifest"])
if not manifest.is_file() or sha256(manifest) != near_report.get("output_manifest_sha256"):
    raise SystemExit("fail_closed: Stage66 near-dedup manifest does not match its report")
if near_report.get("residual_cross_split_near_pairs") != 0:
    raise SystemExit("fail_closed: Stage66 near-dedup left cross-split near pairs")
matrix = json.loads(matrix_path.read_text())
if Path(matrix.get("manifest", "")) != manifest:
    raise SystemExit("fail_closed: Stage66 matrix has not been finalized to the hashed manifest")
if matrix.get("manifest_sha256", "").lower() != sha256(manifest):
    raise SystemExit("fail_closed: Stage66 matrix hashed-manifest SHA mismatch")
if Path(matrix.get("dedup_report", "")) != report_path:
    raise SystemExit("fail_closed: Stage66 matrix does not pin the dedup report")
if matrix.get("dedup_report_sha256", "").lower() != sha256(report_path):
    raise SystemExit("fail_closed: Stage66 matrix dedup-report SHA mismatch")
if Path(matrix.get("near_dedup_report", "")) != near_report_path:
    raise SystemExit("fail_closed: Stage66 matrix does not pin the near-dedup report")
if matrix.get("near_dedup_report_sha256", "").lower() != sha256(near_report_path):
    raise SystemExit("fail_closed: Stage66 matrix near-dedup-report SHA mismatch")
PY

test ! -e "${output_root}"
test ! -e "${state_path}"
cd "${training_root}"
exec "${python_bin}" scripts/run_stage66_licensed_adverse_matrix.py \
  --matrix "${matrix_path}" \
  --output-root "${output_root}" \
  --state "${state_path}" \
  --device cuda \
  --code-revision stage66-licensed-adverse-v1
