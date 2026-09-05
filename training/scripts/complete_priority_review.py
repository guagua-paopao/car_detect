#!/usr/bin/env python3
"""Write the completed Codex visual review into a 600-row review queue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Every row not listed here was visually rejected as a poor attribute crop.
# Columns: zero-based queue index, body_type, color, crop_quality, viewpoint.
GOOD_ROWS = """
1,suv,black,usable,rear
2,suv,black,good,side
6,van,white,usable,side
7,heavy_truck,black,usable,front
10,mpv,silver_gray,good,rear
12,heavy_truck,yellow_orange,usable,side
14,other,black,usable,front
17,heavy_truck,white,usable,side
18,suv,black,good,front
19,suv,blue,good,front_three_quarter
21,sedan,white,good,side
25,sedan,brown_beige,usable,front_three_quarter
28,other,black,usable,front
29,suv,yellow_orange,usable,rear_three_quarter
31,suv,black,usable,side
33,van,black,usable,rear
34,sedan,red,usable,side
35,sedan,white,usable,front
38,bus,white,usable,side
40,sedan,green,usable,rear
48,sedan,green,usable,side
50,sedan,red,usable,front_three_quarter
52,sedan,yellow_orange,usable,rear
54,sedan,red,usable,side
57,van,white,usable,rear_three_quarter
61,other,black,usable,side
63,sedan,red,good,side
64,sedan,silver_gray,usable,front
65,other,white,good,front
66,sedan,silver_gray,good,rear
69,suv,black,usable,front_three_quarter
73,sedan,black,usable,rear
74,sedan,green,usable,side
75,sedan,green,usable,side
76,bus,white,usable,side
83,suv,silver_gray,usable,side
84,sedan,black,usable,rear
85,sedan,black,usable,rear
86,van,green,usable,side
90,van,silver_gray,usable,front
92,mpv,black,usable,rear
94,other,black,usable,front
95,sedan,silver_gray,usable,front
96,bus,red,usable,rear
100,suv,white,usable,side
104,sedan,white,usable,side
106,other,black,usable,front_three_quarter
107,other,red,usable,side
108,pickup,blue,usable,side
109,sedan,silver_gray,usable,side
110,sedan,white,usable,front
112,sedan,silver_gray,usable,front
113,mpv,black,usable,side
115,sedan,black,usable,rear
116,sedan,silver_gray,usable,rear
117,sedan,blue,good,side
118,pickup,white,good,rear
120,mpv,silver_gray,usable,rear
124,pickup,white,good,rear_three_quarter
125,pickup,white,good,rear_three_quarter
126,van,white,usable,side
128,other,silver_gray,usable,side
129,pickup,red,usable,side
130,sedan,silver_gray,usable,rear
133,van,white,usable,front_three_quarter
139,mpv,white,usable,rear
141,suv,silver_gray,usable,rear_three_quarter
142,sedan,black,usable,rear
143,pickup,brown_beige,usable,side
144,van,red,usable,front
146,van,white,usable,rear
147,mpv,silver_gray,usable,rear
148,mpv,silver_gray,usable,rear
149,suv,black,usable,rear_three_quarter
155,other,brown_beige,usable,side
156,other,black,usable,side
157,sedan,silver_gray,usable,front
159,pickup,red,usable,side
161,suv,brown_beige,usable,side
162,other,blue,usable,front_three_quarter
166,sedan,white,usable,side
167,sedan,yellow_orange,usable,rear
169,sedan,red,good,rear
170,sedan,black,usable,side
171,sedan,blue,usable,side
173,sedan,brown_beige,usable,front
181,pickup,white,usable,rear
186,sedan,black,usable,side
188,sedan,black,usable,rear
190,sedan,yellow_orange,usable,side
191,mpv,silver_gray,usable,front
197,sedan,black,good,front
198,sedan,green,usable,side
199,sedan,black,usable,rear
200,pickup,black,usable,side
201,suv,silver_gray,usable,rear
202,mpv,red,usable,rear
203,mpv,red,usable,rear
206,mpv,black,good,side
210,other,black,good,front
212,suv,silver_gray,usable,rear
213,van,white,usable,front
214,other,brown_beige,usable,rear
215,mpv,black,good,side
220,mpv,black,usable,rear
221,van,black,usable,front
226,van,silver_gray,usable,rear
228,other,black,usable,front
229,suv,red,usable,side
230,sedan,silver_gray,usable,front
231,sedan,blue,usable,side
233,other,red,usable,side
234,mpv,blue,usable,rear
241,sedan,silver_gray,usable,rear
243,sedan,silver_gray,usable,side
246,suv,black,usable,side
248,mpv,silver_gray,usable,side
253,sedan,brown_beige,good,side
269,sedan,red,usable,side
272,suv,black,usable,front
278,van,black,usable,front
279,sedan,black,usable,side
291,sedan,white,usable,rear
294,pickup,black,usable,rear
298,other,silver_gray,usable,front
302,suv,black,usable,rear
304,suv,silver_gray,usable,rear
307,sedan,black,usable,side
314,sedan,red,good,side
316,mpv,black,usable,rear
318,suv,green,usable,side
327,suv,white,usable,side
330,sedan,black,usable,side
331,suv,silver_gray,usable,front
334,suv,black,usable,side
335,suv,black,usable,front
344,sedan,white,usable,front
346,sedan,white,usable,front_three_quarter
347,other,silver_gray,usable,side
348,suv,black,usable,front_three_quarter
350,van,silver_gray,usable,front
354,sedan,green,usable,rear
356,suv,brown_beige,usable,front
358,other,silver_gray,usable,rear_three_quarter
360,sedan,black,usable,rear
363,pickup,white,usable,rear
364,sedan,black,usable,rear
368,sedan,blue,usable,rear
369,mpv,red,usable,side
372,sedan,black,good,front
373,suv,black,usable,front
379,van,white,usable,rear
381,pickup,red,usable,side
385,suv,red,usable,rear
389,sedan,brown_beige,usable,side
390,sedan,yellow_orange,usable,side
394,sedan,white,usable,side
395,sedan,brown_beige,good,front
402,mpv,brown_beige,usable,rear
405,mpv,white,usable,rear
406,mpv,white,usable,side
410,sedan,black,usable,side
412,sedan,black,usable,rear
414,van,red,usable,front
415,sedan,silver_gray,good,rear
418,sedan,brown_beige,usable,side
421,suv,red,usable,rear
422,suv,black,usable,front
428,pickup,blue,usable,rear
435,sedan,red,usable,front
436,mpv,brown_beige,good,front_three_quarter
443,other,red,usable,rear
453,sedan,black,usable,front
455,suv,white,usable,side
459,sedan,silver_gray,usable,side
461,sedan,black,usable,side
462,sedan,yellow_orange,usable,side
463,sedan,brown_beige,usable,rear_three_quarter
468,light_truck,silver_gray,usable,side
469,mpv,white,usable,rear
473,van,brown_beige,usable,side
474,van,brown_beige,usable,side
478,sedan,yellow_orange,usable,side
480,mpv,silver_gray,usable,rear
482,sedan,silver_gray,usable,front
483,heavy_truck,black,good,rear
484,mpv,red,usable,side
488,sedan,black,usable,side
489,mpv,brown_beige,usable,front
490,mpv,brown_beige,usable,side
491,mpv,silver_gray,usable,rear
492,sedan,silver_gray,usable,side
495,other,green,usable,side
497,mpv,silver_gray,usable,rear
498,sedan,yellow_orange,usable,rear
500,suv,black,usable,rear
501,mpv,black,usable,side
502,mpv,white,usable,rear
506,van,white,usable,front
510,van,white,usable,front
512,suv,silver_gray,usable,rear_three_quarter
515,van,yellow_orange,usable,side
516,sedan,brown_beige,good,side
521,mpv,green,usable,front
526,pickup,red,good,side
533,sedan,red,usable,front
544,suv,white,usable,side
549,sedan,blue,usable,side
560,mpv,black,usable,rear_three_quarter
561,sedan,black,usable,rear_three_quarter
562,suv,black,usable,rear
563,mpv,red,usable,rear_three_quarter
564,sedan,silver_gray,usable,side
566,suv,black,good,rear
567,suv,silver_gray,usable,side
568,suv,silver_gray,usable,front_three_quarter
569,sedan,silver_gray,usable,side
570,other,black,usable,front
571,other,white,usable,front
574,suv,red,usable,side
575,mpv,white,usable,rear
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("review CSV has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if len(rows) != 600:
        raise ValueError(f"expected 600 rows, found {len(rows)}")

    accepted: dict[int, tuple[str, str, str, str]] = {}
    for line in GOOD_ROWS.splitlines():
        index_text, body, color, quality, viewpoint = line.split(",")
        index = int(index_text)
        if index in accepted:
            raise ValueError(f"duplicate reviewed index: {index}")
        accepted[index] = (body, color, quality, viewpoint)

    for index, row in enumerate(rows):
        body, color, quality, viewpoint = accepted.get(
            index,
            ("unknown", "unknown", "poor", "unknown"),
        )
        row["reviewed_body_type"] = body
        row["reviewed_color"] = color
        row["reviewed_crop_quality"] = quality
        row["reviewed_viewpoint"] = viewpoint
        row["review_status"] = "completed"

    output = args.output or args.review_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS: completed {len(rows)} rows "
        f"({len(accepted)} accepted, {len(rows) - len(accepted)} poor) -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
