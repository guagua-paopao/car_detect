#!/usr/bin/env bash
set -euo pipefail

session_name="VCAS-STAGE72B-BODY-TAXONOMY-V2"
python_bin="/root/miniconda3/bin/python"
builder_script="/root/autodl-tmp/vcas/code/training/scripts/build_stage72_body_taxonomy_v2_manifest.py"
expected_builder_sha="7dc5379483578db504eea15c8d6920c350e9770aabaed3e025f0ee82a0e7b490"
input_manifest="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage71-teacher-manifests-v5/attribute_manifest.stage71-body-teacher.csv"
expected_input_sha="89c625bd44e28749453b0470488207234d10f6e52ffe343c1f4e70eb49b861bb"
labels="/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json"
expected_labels_sha="22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f"
datasets_root="/root/autodl-tmp/vcas/datasets"
output_root="$datasets_root/attribute-domain-v2/stage72-body-taxonomy-v2-v1"
output_manifest="$output_root/attribute_manifest.stage72-body-taxonomy-v2.csv"
output_report="$output_root/stage72-body-taxonomy-v2-report.json"
log_path="$output_root/stage72-body-taxonomy-v2-builder.log"

for required_file in "$python_bin" "$builder_script" "$input_manifest" "$labels"; do
  [[ -f "$required_file" ]] || { echo "missing required file: $required_file" >&2; exit 2; }
done
[[ "$(sha256sum "$builder_script" | awk '{print $1}')" == "$expected_builder_sha" ]] || {
  echo "Stage72B builder SHA256 changed" >&2; exit 3;
}
[[ "$(sha256sum "$input_manifest" | awk '{print $1}')" == "$expected_input_sha" ]] || {
  echo "Stage71 body manifest SHA256 changed" >&2; exit 4;
}
[[ "$(sha256sum "$labels" | awk '{print $1}')" == "$expected_labels_sha" ]] || {
  echo "taxonomy-v2 labels SHA256 changed" >&2; exit 5;
}
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "session already exists: $session_name" >&2; exit 6;
fi
if pgrep -f '^/root/miniconda3/bin/python .*/train_attribute.py' >/dev/null \
  || pgrep -f '^/root/miniconda3/bin/python .*/recover_stage71_' >/dev/null \
  || pgrep -f '^/root/miniconda3/bin/python .*/build_stage72_.*manifest.py' >/dev/null; then
  echo "training, Stage71 recovery, or another Stage72 builder is active" >&2; exit 7;
fi
if [[ -e "$output_manifest" || -e "$output_report" || -e "$log_path" ]]; then
  echo "refusing to overwrite existing Stage72B evidence" >&2; exit 8;
fi
available_kib="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || { echo "unable to determine free disk" >&2; exit 9; }
(( available_kib >= 4194304 )) || { echo "less than 4 GiB free; refusing Stage72B" >&2; exit 10; }

tmux new-session -d -s "$session_name" \
  "mkdir -p '$output_root' && \
   '$python_bin' '$builder_script' \
     --input-manifest '$input_manifest' \
     --expected-input-sha256 '$expected_input_sha' \
     --labels '$labels' \
     --expected-labels-sha256 '$expected_labels_sha' \
     --datasets-safety-root '$datasets_root' \
     --output-manifest '$output_manifest' \
     --output-report '$output_report' \
     --near-duplicate-hamming 1 \
     --minimum-train-rows 100000 \
     --minimum-validation-rows 10000 \
     > '$log_path' 2>&1 && \
   sha256sum '$output_manifest' > '$output_manifest.sha256' && \
   sha256sum '$output_report' > '$output_report.sha256'"

echo "started tmux:$session_name"
echo "log=$log_path"
