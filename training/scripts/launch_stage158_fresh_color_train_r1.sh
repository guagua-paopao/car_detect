#!/usr/bin/env bash
set -euo pipefail

run_id="${STAGE158_RUN_ID:-ATTR-STAGE158-FRESH-COLOR-TRAIN-R2}"
base="/root/autodl-tmp/vcas"
labels="${base}/code/config/vehicle_labels.v2.json"
manifest_root="${base}/datasets/attribute-domain-v2/stage157-fresh-color-combined-r2"
manifest="${manifest_root}/attribute_manifest.stage157-fresh-color-combined.csv"
combined_report="${manifest_root}/stage157-combined-color-manifest-report.json"
code_root="${STAGE158_CODE_ROOT:-${base}/code/stage158_fresh_color_train_r2/scripts}"
trainer="${code_root}/train_attribute.py"
run_root="${base}/runs/attributes/${run_id}"
outer_log="${base}/runs/attributes/${run_id}.log"
state="${base}/runs/attributes/${run_id}.state.json"

test ! -e "${run_root}"
test ! -e "${state}"
mkdir -p "${run_root}"
exec > >(tee -a "${outer_log}") 2>&1

test "$(sha256sum "${manifest}" | awk '{print $1}')" = "db92921c1633183b018be70091dfe9f26cc55fa942b844c4749b2add5e430f23"
test "$(sha256sum "${combined_report}" | awk '{print $1}')" = "7827f5f996e6951603483399558b0db5085b45ede91d3aab9d4b1932924d872c"
test "$(sha256sum "${trainer}" | awk '{print $1}')" = "8f657b3a4dde5cee366fb0f3af159d93698daac09d4ae6103f68435c2d64bd0c"
test "$(sha256sum "${code_root}/test_train_attribute_stage158_sampling.py" | awk '{print $1}')" = "d982538e406c4c7dab5fa7646a819adc4b9a580721bd7943299cc9909b2db70f"
test "$(sha256sum "${code_root}/finalize_stage158_fresh_color_training.py" | awk '{print $1}')" = "2a11941a4672f13c9219dadc1efd6c9003b82553a0fdd994949f37662cc41638"
test "$(sha256sum "${code_root}/test_finalize_stage158_fresh_color_training.py" | awk '{print $1}')" = "bd5d1c449966c4e06de836daa496bce1c6666b577cc7cdb677005c02568fc5d0"
test "$(sha256sum "${labels}" | awk '{print $1}')" = "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f"
test -f /root/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth
cd "${code_root}"
/root/miniconda3/bin/python -m unittest -v test_finalize_stage158_fresh_color_training.py
/root/miniconda3/bin/python -m unittest -v test_train_attribute_stage158_sampling.py

echo "waiting for Stage156 body training to release the GPU"
while tmux has-session -t VCAS-STAGE156-FULLSCALE-BODY-TRAIN-R1 2>/dev/null; do
  sleep 30
done
while pgrep -f '/train_attribute.py .*ATTR-STAGE156' >/dev/null 2>&1; do
  sleep 15
done

common=(
  --manifest "${manifest}"
  --labels "${labels}"
  --architecture convnext_tiny
  --epochs 8
  --workers 8
  --learning-rate 0.00003
  --weight-decay 0.0001
  --body-loss-weight 0.0
  --color-loss-weight 1.3
  --focal-gamma 0.0
  --color-focal-gamma 1.5
  --class-weighting inverse_sqrt
  --sampler-color-class-weighting inverse_sqrt
  --sampler-color-class-weight-cap 4.0
  --label-smoothing 0.02
  --gradient-clip-norm 5.0
  --freeze-backbone-epochs 1
  --patience 3
  --type-threshold 0.99
  --color-threshold 0.70
  --gate-type-precision 0.93
  --gate-type-coverage 0.45
  --gate-color-precision 0.93
  --gate-color-coverage 0.25
  --seed 20260906
  --augmentation-profile color_scene
  --selection-head color
  --body-hierarchy none
  --coarse-car-loss-weight 0.0
  --coarse-truck-loss-weight 0.0
  --coarse-color-loss-weight 0.5
  --night-sample-weight 1.5
  --occlusion-sample-weight 1.1
  --hard-sample-weight 0.1
  --small-sample-weight 0.5
  --color-sample-weight 0.25
  --pseudo-label-weight 0.5
  --run-kind formal
  --skip-test
)

/root/miniconda3/bin/python "${trainer}" "${common[@]}" \
  --input-size 256 \
  --resize-mode stretch \
  --batch-size 40 \
  --dataset-version attribute-domain-v2-stage158-fresh-color-256-r1 \
  --code-revision stage158-fresh-imagenet-color-256-balanced-r2 \
  --output-dir "${run_root}/ATTR-STAGE158-COLOR-CONVNEXT-256-FRESH-R1" \
  2>&1 | tee "${run_root}/candidate-256.log"

/root/miniconda3/bin/python "${trainer}" "${common[@]}" \
  --input-size 288 \
  --resize-mode letterbox \
  --batch-size 32 \
  --dataset-version attribute-domain-v2-stage158-fresh-color-288-r1 \
  --code-revision stage158-fresh-imagenet-color-288-letterbox-balanced-r2 \
  --output-dir "${run_root}/ATTR-STAGE158-COLOR-CONVNEXT-288-LETTERBOX-FRESH-R1" \
  2>&1 | tee "${run_root}/candidate-288.log"

/root/miniconda3/bin/python "${code_root}/finalize_stage158_fresh_color_training.py" \
  --combined-report "${combined_report}" \
  --expected-combined-report-sha256 7827f5f996e6951603483399558b0db5085b45ede91d3aab9d4b1932924d872c \
  --trainer "${trainer}" \
  --expected-trainer-sha256 8f657b3a4dde5cee366fb0f3af159d93698daac09d4ae6103f68435c2d64bd0c \
  --labels "${labels}" \
  --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --candidate-256-dir "${run_root}/ATTR-STAGE158-COLOR-CONVNEXT-256-FRESH-R1" \
  --candidate-288-dir "${run_root}/ATTR-STAGE158-COLOR-CONVNEXT-288-LETTERBOX-FRESH-R1" \
  --output "${state}"

sha256sum "${outer_log}" "${run_root}/candidate-256.log" "${run_root}/candidate-288.log" > "${run_root}/LOG_SHA256SUMS"
