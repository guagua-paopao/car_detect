#!/usr/bin/env bash
set -euo pipefail

export STAGE159_RUN_ID=ATTR-STAGE159-COMPONENT-VALIDATION-R4
export STAGE159_CODE_ROOT=/root/autodl-tmp/vcas/code/stage159_component_validation_r4
exec /root/autodl-tmp/vcas/code/stage159_component_validation_r4/scripts/launch_stage159_component_validation_r1.sh
