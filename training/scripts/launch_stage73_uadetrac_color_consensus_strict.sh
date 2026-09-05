#!/usr/bin/env bash
set -euo pipefail

# Re-run the unchanged Stage70 fail-closed pseudo-label contract with the
# independently trained Stage71 ConvNeXt color teacher. This is a train-only
# audit: a failing report is evidence, never permission to import labels.
session_name="VCAS-STAGE73-UA-COLOR-STRICT"
base="/root/autodl-tmp/vcas"
python_bin="/root/miniconda3/bin/python"
auditor="${base}/code/training/scripts/build_stage70_uadetrac_color_pseudolabels.py"
pool="${base}/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2"
manifest="${pool}/attribute_manifest.training-only.csv"
pool_report="${pool}/dataset_report.json"
labels="${base}/code/config/vehicle_labels.v1.json"
production="${base}/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
stage71_color="${base}/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/best.pt"
run="${base}/runs/attributes/ATTR-STAGE73-UA-COLOR-CONSENSUS-STRICT-V1"
output_manifest="${run}/attribute_manifest.color-pseudo.train-only.csv"
output_report="${run}/color-pseudolabel-report.json"
log_path="${run}/stage73-color-consensus.log"

declare -A expected=(
  ["${auditor}"]="5ab978d5d9c9935a5b9aa2ce4943d9d701e01a542a1ec93aa14c6df41365127e"
  ["${manifest}"]="ab1f20ab1ef2e9da0b26b48d49efd0d4b9d324a22ec58dcdb2e102b371ebc369"
  ["${pool_report}"]="c44e4c6f0acd35ada56abed744c375b32fa6b94417f4fe7b8d693d3ad2e659ca"
  ["${labels}"]="c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f"
  ["${production}"]="6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383"
  ["${stage71_color}"]="14bf423d277085c7decec236f1cc3931274ad778d0101e5d839be43648254121"
)
for path in "${!expected[@]}"; do
  [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 2; }
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected[${path}]}" ]] || {
    echo "SHA256 mismatch: ${path}" >&2
    exit 3
  }
done

"${python_bin}" -c '
import json,sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("status") == "pass"
assert payload.get("color_supervised_rows") == 0
assert payload.get("deduplication", {}).get("post_filter_cross_split_count") == 0
assert not payload.get("video_leaks") and not payload.get("track_leaks")
policy=payload.get("policy", {})
assert policy.get("test_accessed") is False and policy.get("frozen_video_used") is False
' "${pool_report}"

if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "session already exists: ${session_name}" >&2
  exit 4
fi
if [[ -e "${run}" ]]; then
  echo "refusing to overwrite: ${run}" >&2
  exit 5
fi
if pgrep -f '^/root/miniconda3/bin/python .*/train_attribute.py' >/dev/null; then
  echo "model training is active" >&2
  exit 6
fi
available_kib="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
[[ "${available_kib}" =~ ^[0-9]+$ ]] && (( available_kib >= 4194304 )) || {
  echo "less than 4 GiB free" >&2
  exit 7
}

tmux new-session -d -s "${session_name}" \
  "mkdir -p '${run}' && \
   '${python_bin}' '${auditor}' \
     --manifest '${manifest}' \
     --dataset-root '${pool}' \
     --labels '${labels}' \
     --expected-labels-sha256 '${expected[${labels}]}' \
     --checkpoint 'production=${production}' \
     --checkpoint 'stage71_color=${stage71_color}' \
     --output-manifest '${output_manifest}' \
     --output-report '${output_report}' \
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
     --device cuda \
     > '${log_path}' 2>&1 && \
   sha256sum '${output_manifest}' > '${output_manifest}.sha256' && \
   sha256sum '${output_report}' > '${output_report}.sha256'"

echo "started tmux:${session_name}"
echo "run=${run}"
