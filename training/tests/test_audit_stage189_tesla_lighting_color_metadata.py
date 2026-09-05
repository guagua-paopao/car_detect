import csv
import hashlib
from pathlib import Path

from training.scripts import audit_stage189_tesla_lighting_color_metadata as module


def test_exact_color_mapping_does_not_force_ambiguous_achromatic_labels(tmp_path: Path) -> None:
    rows = []
    colors = list(module.EXACT_COLOR) + sorted(module.AMBIGUOUS_COLOR)
    models = list(module.EXACT_BODY) + ["Other car"]
    lighting = ["Light", "Medium", "Dark"]
    for index in range(3026):
        rows.append({
            "color": colors[index % len(colors)],
            "image": f"img_{index:04d}.jpg",
            "lighting": lighting[index % len(lighting)],
            "model": models[index % len(models)],
            "year_3": "", "year_s": "", "year_x": "", "year_y": "",
        })
    labels = tmp_path / "tesla_dataset_labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name in ("labels_description.txt", "label-studio-code.xml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    expected = {
        path.name: (path.stat().st_size, hashlib.md5(path.read_bytes()).hexdigest())
        for path in tmp_path.iterdir()
    }
    original = module.EXPECTED
    try:
        module.EXPECTED = expected
        report = module.run(tmp_path, tmp_path / "report.json")
    finally:
        module.EXPECTED = original
    assert report["status"] == "pass_metadata_only_image_archive_pending_no_training"
    assert report["counts"]["exact_color_rows"] < report["counts"]["rows"]
    assert report["counts"]["confirmed_natural_night_rows"] == 0
    assert report["gates"]["training_authorized"] is False
