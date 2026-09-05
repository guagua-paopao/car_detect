#!/usr/bin/env python3
"""Verify an AutoDL host before a VCAS GPU training run and record evidence."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from src.common import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-vram-gb", type=float, default=20.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=80.0)
    return parser.parse_args()


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc)}
    return {
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    root = args.workspace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    free_disk_gb = disk.free / (1024**3)
    if free_disk_gb < args.min_free_disk_gb:
        errors.append(
            f"free disk {free_disk_gb:.1f} GiB is below {args.min_free_disk_gb:.1f} GiB"
        )

    torch_info: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "compiled_cuda": torch.version.cuda,
        }
        if cuda_available:
            properties = torch.cuda.get_device_properties(0)
            vram_gb = properties.total_memory / (1024**3)
            torch_info.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "vram_gb": round(vram_gb, 2),
                    "cudnn_version": torch.backends.cudnn.version(),
                }
            )
            if vram_gb < args.min_vram_gb:
                errors.append(
                    f"GPU VRAM {vram_gb:.1f} GiB is below {args.min_vram_gb:.1f} GiB"
                )
        else:
            errors.append("torch.cuda.is_available() is false")
    except ImportError as exc:
        torch_info = {"status": "unavailable", "error": str(exc)}
        errors.append("PyTorch is not installed")

    evidence = {
        "schema_version": "1.0",
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "fail" if errors else "pass",
        "workspace_root": str(root),
        "platform": platform.platform(),
        "python": sys.version,
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(free_disk_gb, 2),
        },
        "torch": torch_info,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,temperature.gpu",
                "--format=csv,noheader",
            ]
        ),
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
        "errors": errors,
    }
    write_json(args.output.resolve(), evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
