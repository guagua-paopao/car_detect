#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/vcas/code/training_stage97_eval_r1
EVALUATOR="$ROOT/scripts/evaluate_v2_decoupled_shared_validation.py"
MANIFEST=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv
LABELS=/root/autodl-tmp/vcas/code/config/vehicle_labels.v2.json
BODY=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt
COLOR=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3/best.pt
BASELINE=/root/autodl-tmp/vcas/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
SPECIALIST_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE94-TRUCK-SUBTYPE-R1/ATTR-STAGE94-TRUCK-SUBTYPE-CONVNEXT-256-R1
OUTPUT_ROOT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE97-SPECIALIST-ROUTER-VALIDATION-R1

test ! -e "$OUTPUT_ROOT"
echo "447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38  $EVALUATOR" | sha256sum -c -
echo "70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6  $MANIFEST" | sha256sum -c -
echo "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f  $LABELS" | sha256sum -c -
echo "e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec  $BODY" | sha256sum -c -
echo "a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47  $COLOR" | sha256sum -c -
echo "6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383  $BASELINE" | sha256sum -c -
echo "0f716377586fe340876042f3e7213383dd6cc2a6ec087a5fde0f737186ddf232  $SPECIALIST_ROOT/best.pt" | sha256sum -c -
echo "ef379809f0046562633926ee56c5125a7b75c699a1c85126193af47dfd2e70c6  $SPECIALIST_ROOT/gate-best.pt" | sha256sum -c -

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"
for variant in best gate-best; do
  specialist="$SPECIALIST_ROOT/$variant.pt"
  if [[ "$variant" == best ]]; then
    specialist_sha=0f716377586fe340876042f3e7213383dd6cc2a6ec087a5fde0f737186ddf232
  else
    specialist_sha=ef379809f0046562633926ee56c5125a7b75c699a1c85126193af47dfd2e70c6
  fi
  "$PYTHON" "$EVALUATOR" \
    --manifest "$MANIFEST" \
    --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
    --labels "$LABELS" \
    --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --body-checkpoint "$BODY" \
    --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
    --body-specialist-checkpoint "$specialist" \
    --expected-body-specialist-checkpoint-sha256 "$specialist_sha" \
    --body-specialist-subtype-threshold 0.80 \
    --color-checkpoint "$COLOR" \
    --expected-color-checkpoint-sha256 a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47 \
    --baseline-checkpoint "$BASELINE" \
    --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --output "$OUTPUT_ROOT/$variant.json" \
    --datasets-safety-root /root/autodl-tmp/vcas/datasets \
    --device cuda --batch-size 64 --workers 8 --precision-gate 0.93
  sha256sum "$OUTPUT_ROOT/$variant.json" > "$OUTPUT_ROOT/$variant.json.sha256"
done

sha256sum "$OUTPUT_ROOT"/*.json > "$OUTPUT_ROOT/reports.sha256"
