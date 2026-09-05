#!/usr/bin/env python3
import csv
import json
from pathlib import Path

path = Path('/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE177-FULL-SCENE-AUDIT-R2/attribute_manifest.stage177-scene-audited.csv')
fields = ['image_path', 'video_id', 'track_group', 'frame_number', 'source_frame_id', 'source_image_id', 'source_vehicle_type', 'body_type', 'weather', 'stage177_scene_label', 'stage177_effective_representative']
rows = []
with path.open('r', encoding='utf-8-sig', newline='') as stream:
    for row in csv.DictReader(stream):
        if row.get('source_dataset', '').strip().upper() == 'UA-DETRAC':
            rows.append({field: row.get(field, '') for field in fields})
            if len(rows) == 8:
                break
print(json.dumps(rows, indent=2))
