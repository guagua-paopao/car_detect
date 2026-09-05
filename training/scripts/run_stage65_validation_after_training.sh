#!/usr/bin/env bash
set -euo pipefail

exec /root/miniconda3/bin/python /root/autodl-tmp/vcas/code/training/scripts/run_stage64_taxonomy_v2_validation.py \
  --training-state /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-ADVERSE-PARTIAL-V1.state.json \
  --training-matrix /root/autodl-tmp/vcas/runs/attributes/plans/stage65-adverse-partial-matrix.json \
  --expected-training-matrix-sha256 03f163223bbae01666d256cc6e73a89e18b01cceb8f86eedbba367c4f484e45d \
  --training-root /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-ADVERSE-PARTIAL-V1 \
  --training-session vcas_stage65_adverse_partial \
  --taxonomy-manifest /root/autodl-tmp/vcas/datasets/attribute-domain-v2/attribute_manifest.stage65-adverse-supervised-v1.csv \
  --expected-taxonomy-manifest-sha256 0c19498e66ed00be80d7d398b903d0e8bbbe5153334f342bbc1f196d9ab611f8 \
  --views-report /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE64-VALIDATION-VIEWS-V1.report.json \
  --expected-views-report-sha256 3e1c0e0bf5ed969fd237f8a484da535a89ea0745a933d51a8ba281c5e9214f1f \
  --production-checkpoint /root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt \
  --expected-production-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
  --labels /root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json \
  --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --expected-static-evaluator-sha256 974a684d34dc45e4572d8efa5e703bee2de486f3b8ba44649a1909bef37ef2bb \
  --expected-track-evaluator-sha256 e3310eda88ae9e8afec5337408fd5fe9f977a0a176ac1b8f55375722c195e502 \
  --scripts-root /root/autodl-tmp/vcas/code/training/scripts \
  --output-root /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-ADVERSE-PARTIAL-VALIDATION-V1 \
  --state /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE65-ADVERSE-PARTIAL-VALIDATION-V1.state.json \
  --python /root/miniconda3/bin/python \
  --device cuda \
  --timeout-hours 168 \
  --resume-waiting
