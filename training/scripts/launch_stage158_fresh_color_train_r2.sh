#!/usr/bin/env bash
set -euo pipefail

export STAGE158_RUN_ID=ATTR-STAGE158-FRESH-COLOR-TRAIN-R2
export STAGE158_CODE_ROOT=/root/autodl-tmp/vcas/code/stage158_fresh_color_train_r2/scripts
exec /root/autodl-tmp/vcas/code/stage158_fresh_color_train_r2/scripts/launch_stage158_fresh_color_train_r1.sh
