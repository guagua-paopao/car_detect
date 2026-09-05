#!/usr/bin/env python3
"""Extract the independent body/color subgraphs from the deployed attribute ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    body = output_dir / "vehicle-attribute-body.onnx"
    color = output_dir / "vehicle-attribute-color.onnx"
    report_path = output_dir / "extraction-report.json"

    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    outputs = {item.name for item in model.graph.output}
    if outputs != {"body_type", "color"}:
        raise RuntimeError(f"unexpected deployed outputs: {sorted(outputs)}")

    onnx.utils.extract_model(
        str(source), str(body), ["images"], ["body_type"], check_model=True
    )
    onnx.utils.extract_model(
        str(source), str(color), ["images"], ["color"], check_model=True
    )

    branches = {}
    for name, path, expected_output in (
        ("body", body, "body_type"),
        ("color", color, "color"),
    ):
        branch = onnx.load(str(path))
        onnx.checker.check_model(branch)
        actual_outputs = [item.name for item in branch.graph.output]
        if actual_outputs != [expected_output]:
            raise RuntimeError(f"{name} branch output mismatch: {actual_outputs}")
        branches[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "nodes": len(branch.graph.node),
            "initializers": len(branch.graph.initializer),
            "output": expected_output,
        }

    report = {
        "schema_version": "1.0",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_nodes": len(model.graph.node),
        "method": "onnx.utils.extract_model dependency-preserving extraction",
        "input": "images",
        "branches": branches,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
