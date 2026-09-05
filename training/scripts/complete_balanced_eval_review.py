#!/usr/bin/env python3
"""Write visually reviewed, high-quality validation/test attribute queues."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALIASES = {
    "silver": "silver_gray",
    "yellow": "yellow_orange",
    "brown": "brown_beige",
}

VALIDATION_REJECTED = {
    2, 3, 17, 19, 27, 66, 83, 86, 87, 89, 90, 96, 100, 107, 115,
    124, 125, 130, 143, 144, 146, 153, 156, 166, 167, 175, 178, 179,
}
VALIDATION_LABELS = """
0 light_truck other
1 heavy_truck red
4 pickup red
5 other brown
6 pickup black
7 bus red
8 heavy_truck yellow
9 pickup blue
10 other silver
11 sedan green
12 other silver
13 heavy_truck yellow
14 heavy_truck red
15 other blue
16 suv black
18 bus red
20 other yellow
21 suv blue
22 pickup black
23 bus red
24 mpv other
25 sedan white
26 sedan yellow
28 heavy_truck red
29 suv yellow
30 suv black
31 sedan white
32 pickup black
33 bus blue
34 pickup white
35 van white
36 sedan white
37 bus yellow
38 van other
39 sedan black
40 sedan red
41 van black
42 pickup red
43 sedan white
44 sedan yellow
45 bus white
46 other blue
47 sedan white
48 van white
49 sedan yellow
50 sedan black
51 bus other
52 sedan red
53 sedan white
54 bus blue
55 other brown
56 sedan white
57 suv black
58 pickup brown
59 sedan white
60 bus white
61 other yellow
62 sedan black
63 suv red
64 sedan yellow
65 sedan brown
67 other yellow
68 suv black
69 bus red
70 suv black
71 sedan white
72 bus silver
73 other yellow
74 sedan black
75 suv red
76 sedan white
77 sedan silver
78 van blue
79 sedan white
80 sedan black
81 bus brown
82 sedan brown
84 bus silver
85 sedan silver
88 sedan silver
91 sedan black
92 sedan white
93 bus yellow
94 sedan silver
95 sedan silver
97 sedan green
98 sedan red
99 heavy_truck red
101 sedan white
102 other black
103 sedan white
104 suv red
105 suv brown
106 other white
108 sedan blue
109 sedan red
110 suv white
111 other white
112 sedan blue
113 sedan silver
114 other red
116 pickup silver
117 other black
118 other brown
119 other red
120 sedan black
121 other white
122 mpv silver
123 sedan silver
126 sedan white
127 other yellow
128 sedan black
129 sedan blue
131 suv white
132 other silver
133 other yellow
134 sedan black
135 sedan yellow
136 other red
137 sedan silver
138 sedan white
139 other yellow
140 suv black
141 suv black
142 sedan red
145 sedan yellow
147 sedan white
148 other blue
149 pickup white
150 other white
151 other yellow
152 other black
154 pickup red
155 sedan white
157 other yellow
158 sedan black
159 other brown
160 other red
161 other silver
162 sedan white
163 sedan yellow
164 other silver
165 sedan black
168 sedan white
169 sedan white
170 other black
171 other black
172 other red
173 other silver
174 sedan white
176 other black
177 other silver
"""

TEST_REJECTED = {
    0, 4, 14, 16, 17, 20, 23, 26, 35, 38, 41, 47, 49, 50, 51, 68,
    72, 75, 78, 81, 84, 89, 90, 94, 99, 107, 117, 133, 154, 169, 170,
    178,
}
TEST_LABELS = """
1 suv yellow
2 other silver
3 other red
5 suv silver
6 bus white
7 heavy_truck yellow
8 other black
9 other yellow
10 other red
11 pickup silver
12 bus blue
13 heavy_truck green
15 sedan black
18 mpv brown
19 suv white
21 other black
22 other yellow
24 bus red
25 heavy_truck white
27 sedan green
28 sedan yellow
29 suv silver
30 bus green
31 pickup red
32 mpv silver
33 other black
34 other white
36 suv yellow
37 heavy_truck brown
39 other brown
40 other red
42 pickup red
43 other yellow
44 other white
45 bus blue
46 heavy_truck white
48 sedan black
52 suv blue
53 van yellow
54 heavy_truck yellow
55 other silver
56 other blue
57 bus other
58 suv red
59 other blue
60 other white
61 bus other
62 pickup yellow
63 other yellow
64 other silver
65 van blue
66 heavy_truck red
67 sedan black
69 van red
70 heavy_truck white
71 pickup green
73 van white
74 other yellow
76 heavy_truck black
77 suv black
79 bus white
80 other black
82 van white
83 other silver
85 bus red
86 other blue
87 other brown
88 other blue
91 other white
92 other yellow
93 sedan silver
95 other red
96 sedan silver
97 other white
98 other yellow
100 sedan black
101 sedan silver
102 pickup silver
103 sedan black
104 suv yellow
105 other silver
106 other other
108 other brown
109 other white
110 other yellow
111 sedan black
112 sedan black
113 other red
114 other green
115 other red
116 pickup yellow
118 other black
119 other red
120 sedan silver
121 other white
122 other silver
123 other black
124 pickup blue
125 other red
126 sedan green
127 other white
128 other silver
129 van yellow
130 other black
131 other red
132 sedan white
134 other silver
135 other yellow
136 other blue
137 other red
138 sedan silver
139 other white
140 other black
141 other yellow
142 other black
143 other yellow
144 sedan silver
145 other white
146 other brown
147 other yellow
148 other yellow
149 suv red
150 other black
151 other white
152 other white
153 sedan yellow
155 sedan blue
156 mpv silver
157 sedan white
158 sedan black
159 other yellow
160 other black
161 other red
162 suv silver
163 other white
164 other silver
165 other yellow
166 other white
167 other red
168 other silver
171 other yellow
172 other white
173 other black
174 suv silver
175 other white
176 sedan black
177 other yellow
179 other red
"""


def parse_labels(value: str) -> dict[int, tuple[str, str]]:
    result = {}
    for raw_line in value.strip().splitlines():
        index_text, body_type, color = raw_line.split()
        result[int(index_text)] = (body_type, ALIASES.get(color, color))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.split == "validation":
        rejected = VALIDATION_REJECTED
        labels = parse_labels(VALIDATION_LABELS)
    else:
        rejected = TEST_REJECTED
        labels = parse_labels(TEST_LABELS)
    expected = set(range(len(rows)))
    if rejected & set(labels):
        raise ValueError("an index cannot be both accepted and rejected")
    if rejected | set(labels) != expected:
        missing = sorted(expected - rejected - set(labels))
        extra = sorted((rejected | set(labels)) - expected)
        raise ValueError(f"review coverage mismatch: missing={missing}, extra={extra}")

    completed = []
    for index, row in enumerate(rows):
        if index in rejected:
            continue
        body_type, color = labels[index]
        item = dict(row)
        item["reviewed_body_type"] = body_type
        item["reviewed_color"] = color
        item["reviewed_crop_quality"] = "good"
        item["reviewed_viewpoint"] = row.get("teacher_viewpoint") or "unknown"
        completed.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(completed)
    print(
        f"PASS: {args.split} accepted={len(completed)} "
        f"rejected={len(rejected)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
