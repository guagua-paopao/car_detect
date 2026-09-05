#!/usr/bin/env bash
set -euo pipefail

pool_session="VCAS-BUILD-STAGE70-UA-V2"
base="/root/autodl-tmp/vcas"
pool="${base}/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2"
run="${base}/runs/attributes/ATTR-STAGE70-UA-COLOR-PSEUDO-V2"
auditor="${base}/code/training/scripts/build_stage70_uadetrac_color_pseudolabels.py"
labels="${base}/code/config/vehicle_labels.v1.json"
production="${base}/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
dvm_teacher="${base}/runs/attributes/ATTR-STAGE68-DVM-COLOR-PRETRAIN-V1/best.pt"

while tmux has-session -t "=${pool_session}" 2>/dev/null; do
  sleep 30
done

if [[ -e "${run}" ]]; then
  echo "fail_closed: refusing to overwrite ${run}" >&2
  exit 21
fi
mkdir -p "${run}"

/root/miniconda3/bin/python - "${pool}/dataset_report.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("fail_closed: Stage70 v2 pool report is missing")
x=json.loads(p.read_text(encoding="utf-8"))
if x.get("status") != "pass":
    raise SystemExit("fail_closed: Stage70 v2 pool did not pass")
if x.get("color_supervised_rows") != 0:
    raise SystemExit("fail_closed: source pool unexpectedly contains color truth")
if x.get("deduplication", {}).get("post_filter_cross_split_count") != 0:
    raise SystemExit("fail_closed: post-filter perceptual leakage remains")
if x.get("video_leaks") or x.get("track_leaks"):
    raise SystemExit("fail_closed: video or track leakage remains")
policy=x.get("policy", {})
if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
    raise SystemExit("fail_closed: isolation policy is not proven")
PY

[[ "$(sha256sum "${auditor}" | awk '{print $1}')" == "5ab978d5d9c9935a5b9aa2ce4943d9d701e01a542a1ec93aa14c6df41365127e" ]]
[[ "$(sha256sum "${labels}" | awk '{print $1}')" == "c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f" ]]
[[ "$(sha256sum "${production}" | awk '{print $1}')" == "6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383" ]]
[[ "$(sha256sum "${dvm_teacher}" | awk '{print $1}')" == "9a5a887c37827660ac5042c7da02d2215b13ef8fc0580fb9ada4f20cfa5160f6" ]]

exec /root/miniconda3/bin/python "${auditor}" \
  --manifest "${pool}/attribute_manifest.training-only.csv" \
  --dataset-root "${pool}" \
  --labels "${labels}" \
  --expected-labels-sha256 c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f \
  --checkpoint "production=${production}" \
  --checkpoint "dvm=${dvm_teacher}" \
  --output-manifest "${run}/attribute_manifest.color-pseudo.train-only.csv" \
  --output-report "${run}/color-pseudolabel-report.json" \
  --teacher-confidence 0.75 \
  --track-evidence-agreement 0.80 \
  --minimum-evidence-frames 3 \
  --audit-frames-per-track 5 \
  --output-frames-per-track 3 \
  --maximum-per-class 6000 \
  --minimum-total 500 \
  --minimum-classes 5 \
  --batch-size 128 \
  --workers 6 \
  --device cuda
