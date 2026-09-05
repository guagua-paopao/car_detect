#!/usr/bin/env python3
"""Import and instantiate the isolated attribute training bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    for required in (
        root / "scripts" / "train_attribute.py",
        root / "scripts" / "migrate_attribute_checkpoint_taxonomy_v2.py",
        root / "src" / "attribute_dataset.py",
        root / "src" / "attribute_hierarchy.py",
        root / "src" / "multitask_mobilenet_v3.py",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    sys.path.insert(0, str(root))
    from src.attribute_hierarchy import car_family_indices, coarse_truck_family_indices
    from src.multitask_mobilenet_v3 import MultiTaskVehicleAttributes

    body = ["sedan", "suv", "mpv", "van", "pickup", "bus", "truck", "light_truck", "heavy_truck", "other", "unknown"]
    car = car_family_indices(body)
    truck = coarse_truck_family_indices(body)
    model = MultiTaskVehicleAttributes(11, 11, architecture="convnext_tiny", pretrained=False)
    body_head = model.body_type_head
    color_head = model.color_head
    body_output = body_head.out_features if hasattr(body_head, "out_features") else body_head[-1].out_features
    color_output = color_head.out_features if hasattr(color_head, "out_features") else color_head[-1].out_features
    result = {
        "status": "pass",
        "architecture": model.architecture,
        "body_outputs": int(body_output),
        "color_outputs": int(color_output),
        "car_family_indices": list(car),
        "truck_family_indices": list(truck),
    }
    if result["architecture"] != "convnext_tiny" or result["body_outputs"] != 11 or result["color_outputs"] != 11:
        raise RuntimeError(f"isolated bundle smoke mismatch: {result}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
