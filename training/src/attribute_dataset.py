from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


COARSE_COLOR_GROUP_CODES = {
    "silver_gray": 1,
    "yellow_orange": 2,
    "brown_beige": 3,
}


def head_is_supervised(row: dict[str, str], field: str) -> bool:
    column = f"{field}_supervised"
    if column not in row or not row[column].strip():
        return True
    value = row[column].strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid {column} value: {row[column]!r}")


class VehicleAttributeDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        body_types: list[str],
        colors: list[str],
        transform: Any,
        training: bool,
        return_pseudo: bool = False,
        return_body_family: bool = False,
        return_color_group: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.body_types = body_types
        self.colors = colors
        self.body_index = {label: index for index, label in enumerate(body_types)}
        self.color_index = {label: index for index, label in enumerate(colors)}
        self.transform = transform
        self.return_pseudo = return_pseudo
        self.return_body_family = return_body_family
        self.return_color_group = return_color_group

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = [row for row in rows if row.get("split") == split]
        if any("review_status" in row for row in selected):
            selected = [
                row for row in selected if row.get("review_status") == "approved"
            ]
        if training:
            selected = [
                row
                for row in selected
                if (
                    (
                        head_is_supervised(row, "body_type")
                        and row.get("body_type") in self.body_index
                    )
                    or (
                        head_is_supervised(row, "color")
                        and row.get("color") in self.color_index
                    )
                    or (
                        self.return_body_family
                        and row.get("coarse_body_family", "").strip().lower() in {"car", "truck"}
                    )
                    or (
                        self.return_color_group
                        and row.get("coarse_color_group", "").strip().lower()
                        in COARSE_COLOR_GROUP_CODES
                    )
                )
            ]
        if not selected:
            raise ValueError(
                f"{self.manifest_path} has no usable {split!r} samples; "
                "finish annotation and set review_status=approved"
            )
        self.rows = selected

    def __len__(self) -> int:
        return len(self.rows)

    def targets(self) -> tuple[list[int], list[int]]:
        body_targets = [
            (
                self.body_index.get(row.get("body_type", ""), -100)
                if head_is_supervised(row, "body_type")
                else -100
            )
            for row in self.rows
        ]
        color_targets = [
            (
                self.color_index.get(row.get("color", ""), -100)
                if head_is_supervised(row, "color")
                else -100
            )
            for row in self.rows
        ]
        return body_targets, color_targets

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / row["image_path"]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        tensor = self.transform(rgb)
        body_target = (
            self.body_index.get(row.get("body_type", ""), -100)
            if head_is_supervised(row, "body_type")
            else -100
        )
        color_target = (
            self.color_index.get(row.get("color", ""), -100)
            if head_is_supervised(row, "color")
            else -100
        )
        if self.return_pseudo and self.return_body_family:
            pseudo = row.get("pseudo_label", "false").strip().lower() in {"true", "1", "yes"}
            family_name = row.get("coarse_body_family", "").strip().lower()
            family = {"car": 1, "truck": 2}.get(family_name, 0)
            if self.return_color_group:
                color_group_name = row.get("coarse_color_group", "").strip().lower()
                color_group = COARSE_COLOR_GROUP_CODES.get(color_group_name, 0)
                return tensor, body_target, color_target, str(path), pseudo, family, color_group
            return tensor, body_target, color_target, str(path), pseudo, family
        if self.return_pseudo and self.return_color_group:
            pseudo = row.get("pseudo_label", "false").strip().lower() in {"true", "1", "yes"}
            color_group_name = row.get("coarse_color_group", "").strip().lower()
            color_group = COARSE_COLOR_GROUP_CODES.get(color_group_name, 0)
            return tensor, body_target, color_target, str(path), pseudo, 0, color_group
        if self.return_pseudo:
            pseudo = row.get("pseudo_label", "false").strip().lower() in {"true", "1", "yes"}
            return tensor, body_target, color_target, str(path), pseudo
        return tensor, body_target, color_target, str(path)
