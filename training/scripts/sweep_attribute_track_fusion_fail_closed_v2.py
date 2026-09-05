#!/usr/bin/env python3
"""Validation-only track sweep using fail-closed temporal fusion v2.

The existing v1 inference and metric implementation is reused without editing
its pinned source. Only the temporal fusion primitive is replaced. This keeps
the running Stage71 static-validation evidence reproducible while adding the
stricter conflict/quality policy as a separate downstream gate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent


def ensure_script_import_path() -> None:
    value = str(SCRIPT_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def fail_closed_fuse(
    values: list[dict],
    window: int,
    minimum_share: float,
    minimum_margin: float,
) -> list[str]:
    ensure_script_import_path()
    from track_fusion_fail_closed_v2 import fuse_sequence_fail_closed

    return fuse_sequence_fail_closed(
        values,
        window=window,
        minimum_share=minimum_share,
        minimum_margin=minimum_margin,
        switch_confirmations=3,
        conflict_unknown_after=2,
        override_ratio=1.15,
    )


def argument_value(arguments: list[str], name: str) -> str:
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"required argument missing: {name}") from error


def enforce_validation_only(arguments: list[str]) -> Path:
    if argument_value(arguments, "--split") != "validation":
        raise ValueError("fail-closed v2 sweep is validation-only")
    output = Path(argument_value(arguments, "--output"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation evidence: {output}")
    return output


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    output = enforce_validation_only(arguments)
    ensure_script_import_path()

    import evaluate_attribute_track_fusion as evaluator
    import sweep_attribute_track_fusion as sweep_v1
    from track_fusion_fail_closed_v2 import __file__ as fusion_source

    evaluator.fuse_sequence = fail_closed_fuse
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *arguments]
        result = sweep_v1.main()
    finally:
        sys.argv = original_argv
    if result != 0:
        return result

    report = json.loads(output.read_text(encoding="utf-8"))
    protocol = report.setdefault("protocol", {})
    if protocol.get("split") != "validation" or protocol.get("test_used") is not False:
        raise RuntimeError("upstream sweep violated validation-only policy")
    if protocol.get("frozen_video_used") is not False:
        raise RuntimeError("upstream sweep accessed frozen video")
    report["schema_version"] = "attribute-track-fusion-sweep-fail-closed-v2"
    protocol["fusion_policy"] = {
        "quality_weighted": True,
        "switch_confirmations": 3,
        "conflict_unknown_after": 2,
        "override_ratio": 1.15,
        "low_quality_frames_may_override": False,
        "alternating_conflicts_may_switch": False,
    }
    protocol["v1_sweeper"] = str(Path(sweep_v1.__file__).resolve())
    protocol["v1_evaluator"] = str(Path(evaluator.__file__).resolve())
    protocol["v2_fusion"] = str(Path(fusion_source).resolve())
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
