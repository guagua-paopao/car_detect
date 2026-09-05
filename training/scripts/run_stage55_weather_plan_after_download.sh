#!/usr/bin/env bash
set -euo pipefail

download_session="vcas_stage55_oi_labels_download"
python_bin="/root/miniconda3/bin/python"
project_root="/root/autodl-tmp/vcas/code"
source_root="/root/autodl-tmp/vcas/sources/openimages-v7"
cache_root="/root/autodl-tmp/vcas/datasets/dataset-large-v1-staging/_openimages_cache/open-images-v7/train"
stage_root="/root/autodl-tmp/vcas/datasets/attribute-domain-v2/openimages-stage55-weather-plan-v1"

while tmux has-session -t "=${download_session}" 2>/dev/null; do sleep 30; done

test ! -e "${stage_root}"
exec "${python_bin}" "${project_root}/training/scripts/build_openimages_weather_vehicle_plan.py" \
  --human-labels "${source_root}/oidv7-train-annotations-human-imagelabels.csv" \
  --detections "${cache_root}/labels/detections.csv" \
  --image-metadata "${cache_root}/metadata/image_ids.csv" \
  --exclude-card "/root/autodl-tmp/vcas/datasets/dataset-large-v1-staging/dataset-large-v1.json" \
  --exclude-card "/root/autodl-tmp/vcas/datasets/dataset-v1/dataset-pilot-det-v1.json" \
  --download-state "${source_root}/oidv7-train-annotations-human-imagelabels.download.json" \
  --output-plan "${stage_root}/weather-vehicle-plan.csv" \
  --output-report "${stage_root}/source-plan-report.json" \
  --small-area-ratio 0.01
