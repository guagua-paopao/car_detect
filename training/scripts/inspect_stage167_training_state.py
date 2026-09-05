#!/usr/bin/env python3
import json
from pathlib import Path

root = Path('/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1')
for name in ('model_card.json', 'metrics.json'):
    data = json.loads((root / name).read_text(encoding='utf-8'))
    print(f'=== {name} ===')
    def walk(value, prefix=''):
        if isinstance(value, dict):
            for key, child in value.items():
                path = f'{prefix}.{key}' if prefix else key
                lower = path.lower()
                if isinstance(child, (dict, list)):
                    walk(child, path)
                elif any(token in lower for token in ('unlabel', 'night', 'sample', 'weight', 'manifest', 'consistency', 'train_rows')):
                    print(path, json.dumps(child, ensure_ascii=False))
    walk(data)
