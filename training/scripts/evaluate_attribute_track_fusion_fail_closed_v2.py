#!/usr/bin/env python3
"""Evaluate fixed fail-closed v2 track fusion on validation or one-time test.

Unlike the validation sweeper, this evaluator never searches parameters. Test
runs must provide exactly one window plus explicit share, margin, and per-head
thresholds that were selected earlier on validation.
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


def argument_values(arguments: list[str], name: str) -> list[str]:
    try:
        start = arguments.index(name) + 1
    except ValueError as error:
        raise ValueError(f"required argument missing: {name}") from error
    values: list[str] = []
    for value in arguments[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    if not values:
        raise ValueError(f"required argument has no value: {name}")
    return values


def validate_fixed_protocol(arguments: list[str]) -> tuple[Path, str]:
    split = argument_value(arguments, "--split")
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    output = Path(argument_value(arguments, "--output"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite track evidence: {output}")
    windows = argument_values(arguments, "--fusion-windows")
    if len(windows) != 1:
        raise ValueError("fixed v2 evaluation requires exactly one fusion window")
    for name in (
        "--fusion-min-share",
        "--fusion-min-margin",
        "--type-threshold",
        "--color-threshold",
    ):
        argument_value(arguments, name)
    return output, split


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    output, split = validate_fixed_protocol(arguments)
    ensure_script_import_path()

    import evaluate_attribute_track_fusion as evaluator
    from track_fusion_fail_closed_v2 import __file__ as fusion_source

    evaluator.fuse_sequence = fail_closed_fuse
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *arguments]
        result = evaluator.main()
    finally:
        sys.argv = original_argv
    if result != 0:
        return result

    report = json.loads(output.read_text(encoding="utf-8"))
    protocol = report.setdefault("protocol", {})
    if protocol.get("split") != split or protocol.get("test_used_for_selection") is not False:
        raise RuntimeError("base evaluator violated the fixed evaluation protocol")
    if protocol.get("frozen_video_used") is not False:
        raise RuntimeError("base evaluator accessed frozen video")
    report["schema_version"] = "attribute-track-fusion-fixed-fail-closed-v2"
    protocol["parameter_search"] = False
    protocol["parameters_source"] = "independent validation"
    protocol["fusion_policy"] = {
        "quality_weighted": True,
        "switch_confirmations": 3,
        "conflict_unknown_after": 2,
        "override_ratio": 1.15,
        "low_quality_frames_may_override": False,
        "alternating_conflicts_may_switch": False,
    }
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
