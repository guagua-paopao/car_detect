#!/usr/bin/env python3
import json
from pathlib import Path

path = Path('/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE209-UNUSED-SUPERVISED-INVENTORY-R1/stage209-unused-supervised-inventory.json')
data = json.loads(path.read_text(encoding='utf-8'))
rows = []
for manifest, stats in data.get('per_manifest', {}).items():
    novel = int(stats.get('exact_novel_rows', 0) or 0)
    eligible = int(stats.get('metadata_eligible', 0) or 0)
    if novel or eligible:
        rows.append((novel, eligible, manifest, stats.get('exact_novel_class_counts_before_global_dedup', {}), stats.get('metadata_eligible_source_counts', {})))
for novel, eligible, manifest, classes, sources in sorted(rows, reverse=True):
    print(json.dumps({
        'novel_before_global_dedup': novel,
        'eligible': eligible,
        'manifest': manifest,
        'classes': classes,
        'sources': sources,
    }, ensure_ascii=False))
