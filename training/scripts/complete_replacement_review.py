#!/usr/bin/env python3
"""Complete exactly 379 visually accepted replacement review decisions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# These candidates were visually rejected for being synthetic, partial, heavily
# occluded, multi-vehicle, interior-only, or otherwise unsuitable.
REJECTED = {
    17, 19, 32, 44, 54, 59, 60, 62, 81, 86, 99, 102, 106, 108,
    120, 129, 130, 135, 139, 140, 141, 144, 146, 151, 152, 157,
    160, 164, 174, 179, 182, 190, 193, 198, 207, 212, 223, 231,
    232, 243, 247, 251, 254, 263, 265, 276, 281, 282, 286, 287,
    292, 295, 306, 310, 311, 312, 313, 314, 315, 323, 329, 332,
    340, 343, 348, 352, 359, 367, 370, 372, 385, 388, 389, 392,
    393, 396, 400, 411, 423, 426, 427, 430, 432, 441, 443, 445,
    447, 448, 450, 464, 471, 474,
}


# Visual corrections to the candidate teacher labels.
CORRECTIONS_TEXT = """
4,mpv,other
6,sedan,silver_gray
10,heavy_truck,silver_gray
11,pickup,brown_beige
13,suv,other
18,bus,silver_gray
25,suv,red
26,other,other
33,sedan,silver_gray
34,suv,other
36,bus,silver_gray
39,mpv,green
47,light_truck,brown_beige
49,mpv,white
53,bus,brown_beige
63,bus,red
67,sedan,yellow_orange
71,heavy_truck,red
76,sedan,blue
80,light_truck,silver_gray
89,bus,white
93,mpv,white
94,mpv,green
97,suv,silver_gray
100,heavy_truck,black
103,sedan,red
109,heavy_truck,white
110,other,green
116,light_truck,blue
119,pickup,red
121,sedan,white
126,bus,brown_beige
128,heavy_truck,silver_gray
134,heavy_truck,green
143,light_truck,red
148,mpv,brown_beige
150,sedan,green
155,heavy_truck,black
158,other,green
163,heavy_truck,white
166,mpv,green
170,light_truck,yellow_orange
172,heavy_truck,blue
173,heavy_truck,black
176,other,red
178,sedan,brown_beige
184,other,black
186,sedan,yellow_orange
189,bus,other
191,light_truck,red
197,pickup,brown_beige
202,sedan,yellow_orange
204,sedan,other
208,heavy_truck,other
211,sedan,yellow_orange
213,sedan,brown_beige
215,bus,other
218,light_truck,blue
222,sedan,green
224,light_truck,white
227,pickup,black
228,mpv,black
236,other,other
238,sedan,brown_beige
241,suv,black
242,bus,yellow_orange
244,heavy_truck,black
246,mpv,black
252,bus,green
255,mpv,silver_gray
256,other,black
259,suv,blue
261,bus,brown_beige
267,sedan,black
268,suv,black
269,light_truck,brown_beige
271,heavy_truck,red
272,heavy_truck,white
274,sedan,yellow_orange
275,pickup,black
277,suv,brown_beige
278,light_truck,green
283,mpv,yellow_orange
285,mpv,brown_beige
290,pickup,white
293,suv,green
296,light_truck,other
301,sedan,black
302,heavy_truck,black
305,pickup,white
308,heavy_truck,white
309,light_truck,black
318,mpv,black
319,mpv,green
320,light_truck,red
324,bus,red
327,sedan,green
330,suv,yellow_orange
334,suv,brown_beige
342,suv,white
347,light_truck,black
350,mpv,brown_beige
351,mpv,red
354,suv,yellow_orange
355,bus,black
361,suv,white
363,light_truck,blue
371,light_truck,white
375,mpv,brown_beige
383,sedan,brown_beige
387,heavy_truck,blue
391,mpv,brown_beige
397,heavy_truck,green
399,other,brown_beige
403,light_truck,blue
404,bus,red
405,heavy_truck,yellow_orange
407,sedan,brown_beige
415,sedan,brown_beige
419,suv,blue
429,heavy_truck,yellow_orange
431,sedan,green
435,van,brown_beige
439,sedan,green
440,pickup,white
444,bus,black
455,sedan,yellow_orange
459,light_truck,brown_beige
461,heavy_truck,green
463,sedan,yellow_orange
475,heavy_truck,brown_beige
479,sedan,yellow_orange
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-count", type=int, default=379)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corrections: dict[int, tuple[str, str]] = {}
    for line in CORRECTIONS_TEXT.splitlines():
        index_text, body, color = line.split(",")
        corrections[int(index_text)] = (body, color)

    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("review CSV has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if len(rows) != 600:
        raise ValueError(f"expected 600 rows, found {len(rows)}")

    accepted = 0
    accepted_indices: list[int] = []
    for index, row in enumerate(rows):
        for field in ("body_type", "color", "crop_quality", "viewpoint"):
            row[f"reviewed_{field}"] = ""
        if index >= 480 or index in REJECTED or accepted >= args.accepted_count:
            row["review_status"] = "rejected" if index in REJECTED else "pending"
            continue
        body = row["teacher_body_type"]
        color = row["teacher_color"]
        if index in corrections:
            body, color = corrections[index]
        viewpoint = row.get("teacher_viewpoint") or "unknown"
        row["reviewed_body_type"] = body
        row["reviewed_color"] = color
        row["reviewed_crop_quality"] = "good"
        row["reviewed_viewpoint"] = viewpoint
        row["review_status"] = "completed"
        accepted += 1
        accepted_indices.append(index)

    if accepted != args.accepted_count:
        raise RuntimeError(f"accepted {accepted}, expected {args.accepted_count}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS: accepted {accepted} replacements; "
        f"last accepted index={accepted_indices[-1]}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
