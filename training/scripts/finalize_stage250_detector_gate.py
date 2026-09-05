#!/usr/bin/env python3
"""Apply conservative two-domain validation gates to a Stage250 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-base", "baseline-bmd", "candidate-base", "candidate-bmd"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    reports = {key: load(getattr(args, key)) for key in ("baseline_base", "baseline_bmd", "candidate_base", "candidate_bmd")}
    metric = lambda key, name: float(reports[key]["metrics"][name])
    gates = {
        "base_map50_95_no_regression": metric("candidate_base", "map50_95") >= metric("baseline_base", "map50_95") - 0.005,
        "base_recall_no_regression": metric("candidate_base", "recall_mean") >= metric("baseline_base", "recall_mean") - 0.005,
        "bmd_map50_95_no_regression": metric("candidate_bmd", "map50_95") >= metric("baseline_bmd", "map50_95") - 0.010,
        "bmd_recall_no_regression": metric("candidate_bmd", "recall_mean") >= metric("baseline_bmd", "recall_mean") - 0.010,
        "base_material_gain": (
            metric("candidate_base", "map50_95") >= metric("baseline_base", "map50_95") + 0.010
            or metric("candidate_base", "recall_mean") >= metric("baseline_base", "recall_mean") + 0.010
        ),
    }
    payload = {
        "schema_version": "1.0", "stage": "DET-STAGE250-HARD-REPLAY-R1",
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "validation_pass" if all(gates.values()) else "validation_rejected",
        "gates": gates,
        "metrics": {key: reports[key]["metrics"] for key in reports},
        "policy": {"research_only": True, "deployable": False, "test_accessed": False, "frozen_video_used": False},
    }
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="ascii", newline="\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
