#!/usr/bin/env bash
set -euo pipefail

# Stage72A is deliberately data-only.  It must wait until Stage71 releases the
# remote training/I/O path, and it never reads test or frozen-video assets.
session_name="VCAS-STAGE72A-DVM-FINE-COLOR"
python_bin="/root/miniconda3/bin/python"
builder_script="/root/autodl-tmp/vcas/code/training/scripts/build_stage72_dvm_fine_color_manifest.py"
expected_builder_sha="870c8d517ef431cf8453a36067792692062f13fbc618208c271e3f040c58a938"
source_manifest="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage68-dvm-color-pool-v1/attribute_manifest.audit-clean-v2.csv"
expected_source_sha="9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b"
output_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage72-dvm-fine-color-v1"
output_manifest="$output_root/attribute_manifest.stage72-dvm-fine-color-train.csv"
output_report="$output_root/stage72-dvm-fine-color-report.json"
log_path="$output_root/stage72-dvm-fine-color-builder.log"

for required_file in "$python_bin" "$builder_script" "$source_manifest"; do
  [[ -f "$required_file" ]] || { echo "missing required file: $required_file" >&2; exit 2; }
done

[[ "$(sha256sum "$builder_script" | awk '{print $1}')" == "$expected_builder_sha" ]] || {
  echo "Stage72 builder SHA256 changed" >&2
  exit 3
}
[[ "$(sha256sum "$source_manifest" | awk '{print $1}')" == "$expected_source_sha" ]] || {
  echo "Stage68 clean DVM manifest SHA256 changed" >&2
  exit 4
}

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "session already exists: $session_name" >&2
  exit 5
fi
if pgrep -f '^/root/miniconda3/bin/python .*/train_attribute.py' >/dev/null \
  || pgrep -f '^/root/miniconda3/bin/python .*/recover_stage71_' >/dev/null; then
  echo "model training or Stage71 recovery is still active; Stage72A must wait" >&2
  exit 6
fi
if [[ -e "$output_manifest" || -e "$output_report" || -e "$log_path" ]]; then
  echo "refusing to overwrite existing Stage72A evidence" >&2
  exit 7
fi

available_kib="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || { echo "unable to determine free disk" >&2; exit 8; }
(( available_kib >= 4194304 )) || { echo "less than 4 GiB free; refusing Stage72A" >&2; exit 9; }

tmux new-session -d -s "$session_name" \
  "mkdir -p '$output_root' && \
   '$python_bin' '$builder_script' \
     --dvm-manifest '$source_manifest' \
     --expected-dvm-sha256 '$expected_source_sha' \
     --output-manifest '$output_manifest' \
     --output-report '$output_report' \
     --minimum-train-rows 118338 \
     --minimum-gray-rows 1000 \
     --minimum-silver-rows 1000 \
     --minimum-yellow-rows 500 \
     --minimum-brown-rows 500 \
     > '$log_path' 2>&1 && \
   sha256sum '$output_manifest' > '$output_manifest.sha256' && \
   sha256sum '$output_report' > '$output_report.sha256'"

echo "started tmux:$session_name"
echo "log=$log_path"
