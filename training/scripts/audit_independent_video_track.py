"""Audit candidate/baseline attribute stability on an independent local video.

This is a behavior audit, not a labeled accuracy evaluation.  It deliberately
uses the unused source clip 04_city_motorcycle.mp4 (not a segment in the frozen
60s composite) and records that no color/type ground truth is available.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import onnxruntime as ort


BODY = ["sedan", "suv", "mpv", "van", "pickup", "bus", "light_truck", "heavy_truck", "other", "unknown"]
COLOR = ["black", "white", "silver_gray", "red", "blue", "green", "yellow_orange", "brown_beige", "other", "unknown"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def letterbox(img, size=960):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = r
    return out, scale, dx, dy


def nms_detections(session, frame, conf=0.35, iou=0.5):
    h, w = frame.shape[:2]
    lb, scale, dx, dy = letterbox(frame)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    raw = session.run(None, {session.get_inputs()[0].name: x})[0][0].T
    # YOLO detect export: cx,cy,w,h followed by class scores.
    boxes, scores, classes = [], [], []
    for row in raw:
        cls = int(np.argmax(row[4:]))
        score = float(row[4 + cls])
        if score < conf:
            continue
        cx, cy, bw, bh = map(float, row[:4])
        x1 = max(0, min(w - 1, (cx - bw / 2 - dx) / scale))
        y1 = max(0, min(h - 1, (cy - bh / 2 - dy) / scale))
        x2 = max(0, min(w - 1, (cx + bw / 2 - dx) / scale))
        y2 = max(0, min(h - 1, (cy + bh / 2 - dy) / scale))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(score)
        classes.append(cls)
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    return [(boxes[int(k)], scores[int(k)], classes[int(k)]) for k in np.asarray(keep).reshape(-1)] if len(keep) else []


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def classify(body_session, color_session, crops, body_temp=1.0, color_temp=1.0, color_baseline_session=None, color_baseline_temp=0.75, color_base_gate=0.8, color_candidate_gate=0.85, color_margin=-0.1):
    if not crops:
        return []
    arr = []
    for crop in crops:
        im = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        arr.append(np.transpose(im, (2, 0, 1)))
    inp = np.stack(arr)
    bout = body_session.run(None, {body_session.get_inputs()[0].name: inp})
    cout = color_session.run(None, {color_session.get_inputs()[0].name: inp})
    bp = softmax(bout[0] / body_temp)
    cp = softmax(cout[1] / color_temp)
    bp_color = None
    if color_baseline_session is not None:
        bout_color = color_baseline_session.run(None, {color_baseline_session.get_inputs()[0].name: inp})
        bp_color = softmax(bout_color[1] / color_baseline_temp)
    result = []
    for b, c in zip(bp, cp):
        bi = int(np.argmax(b))
        if bp_color is None:
            ci = int(np.argmax(c)); cconf = float(c[ci])
        else:
            base = bp_color[len(result)]
            non_base = base.copy(); non_base[9] = -1.0
            non_cand = c.copy(); non_cand[9] = -1.0
            bci = int(np.argmax(non_base)); bconf = float(non_base[bci])
            cci = int(np.argmax(non_cand)); cc = float(non_cand[cci])
            if bconf < color_base_gate and cc >= color_candidate_gate and cc >= bconf + color_margin:
                ci, cconf = cci, cc
            else:
                ci, cconf = bci, bconf
        result.append({"body": BODY[bi], "body_conf": float(b[bi]), "color": COLOR[ci], "color_conf": cconf})
    return result


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    den = aw * ah + bw * bh - inter
    return inter / den if den else 0.0


def audit(video: Path, det_path: Path, body_path: Path, color_path: Path, body_temp: float, color_temp: float, sample_every: int, fusion_min_share: float, fusion_min_margin: float, color_baseline_path: Path | None = None, color_baseline_temp: float = .75, color_base_gate: float = .8, color_candidate_gate: float = .85, color_margin: float = -.1):
    det = ort.InferenceSession(str(det_path), providers=["CPUExecutionProvider"])
    body_attr = ort.InferenceSession(str(body_path), providers=["CPUExecutionProvider"])
    color_attr = ort.InferenceSession(str(color_path), providers=["CPUExecutionProvider"])
    color_baseline_attr = ort.InferenceSession(str(color_baseline_path), providers=["CPUExecutionProvider"]) if color_baseline_path else None
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracks = {}
    next_id = 1
    frames = 0
    sampled = 0
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if idx % sample_every:
            continue
        sampled += 1
        ds = nms_detections(det, frame)
        crops = [frame[max(0,b[1]):min(frame.shape[0],b[1]+b[3]), max(0,b[0]):min(frame.shape[1],b[0]+b[2])] for b,_,_ in ds]
        attrs = classify(body_attr, color_attr, crops, body_temp, color_temp, color_baseline_attr, color_baseline_temp, color_base_gate, color_candidate_gate, color_margin)
        used = set()
        for (box, det_conf, _), a in zip(ds, attrs):
            best, best_i = 0.0, None
            for tid, t in tracks.items():
                if tid in used or idx - t["last"] > sample_every * 4:
                    continue
                v = iou(box, t["box"])
                if v > best:
                    best, best_i = v, tid
            if best_i is None or best < 0.15:
                best_i = next_id; next_id += 1
                tracks[best_i] = {"box": box, "last": idx, "body": [], "color": [], "conf": []}
            used.add(best_i)
            t = tracks[best_i]
            t.update(box=box, last=idx)
            q = max(0.01, float(det_conf) * math.sqrt(max(1, box[2] * box[3]) / (frame.shape[0] * frame.shape[1])))
            t["body"].append((a["body"], a["body_conf"], q))
            t["color"].append((a["color"], a["color_conf"], q))
            t["conf"].append((a["body_conf"], a["color_conf"]))
    cap.release()
    rows = []
    for tid, t in tracks.items():
        if len(t["body"]) < 3:
            continue
        def fuse(seq, window=5, min_share=0.60, min_margin=0.12):
            out = []
            for i in range(len(seq)):
                hist = seq[max(0, i - window + 1):i + 1]
                scores = defaultdict(float)
                for label, conf, quality in hist:
                    # Confidence/quality weighting; conflicting windows abstain.
                    scores[label] += max(0.01, float(conf)) * max(0.01, float(quality))
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                total = sum(scores.values()) or 1.0
                if not ranked or ranked[0][1] / total < min_share or (len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) / total < min_margin):
                    out.append(("unknown", 0.0))
                else:
                    out.append((ranked[0][0], ranked[0][1] / total))
            return out

        def stab(key, fused_key=None):
            vals = [x[0] for x in t[key] if x[1] >= 0.8]
            base = {"frames": len(vals), "stability": max(Counter(vals).values()) / len(vals) if vals else None, "switches": sum(a != b for a,b in zip(vals, vals[1:]))}
            if fused_key:
                fv = [x[0] for x in t[fused_key] if x[0] != "unknown"]
                base["fused_frames"] = len(fv)
                base["fused_stability"] = max(Counter(fv).values()) / len(fv) if fv else None
                base["fused_switches"] = sum(a != b for a,b in zip(fv, fv[1:]))
                base["fused_abstention_rate"] = 1.0 - len(fv) / len(t[fused_key]) if t[fused_key] else None
            return base
        t["body_fused"] = fuse(t["body"], min_share=fusion_min_share, min_margin=fusion_min_margin)
        t["color_fused"] = fuse(t["color"], min_share=fusion_min_share, min_margin=fusion_min_margin)
        rows.append({"track_id": tid, "observations": len(t["body"]), "body": stab("body", "body_fused"), "color": stab("color", "color_fused")})
    def summary(key):
        x = [r[key]["stability"] for r in rows if r[key]["stability"] is not None]
        fx = [r[key]["fused_stability"] for r in rows if r[key].get("fused_stability") is not None]
        sw = [r[key]["switches"] for r in rows if r[key]["stability"] is not None]
        fsw = [r[key]["fused_switches"] for r in rows if r[key].get("fused_stability") is not None]
        abst = [r[key]["fused_abstention_rate"] for r in rows if r[key].get("fused_abstention_rate") is not None]
        return {"tracks": len(x), "stability_rate_mean": float(np.mean(x)) if x else None, "stability_ge_095": float(np.mean(np.asarray(x) >= .95)) if x else None, "label_switches_total": int(sum(sw)), "fused": {"tracks": len(fx), "stability_rate_mean": float(np.mean(fx)) if fx else None, "stability_ge_095": float(np.mean(np.asarray(fx) >= .95)) if fx else None, "label_switches_total": int(sum(fsw)), "abstention_rate_mean": float(np.mean(abst)) if abst else None}, "unknown_rate_not_available": True}
    return {"schema_version":"independent-video-track-behavior-v1","video":str(video),"video_sha256":sha256(video),"fps":fps,"total_frames":total,"processed_frames":frames,"sample_every":sample_every,"sampled_frames":sampled,"tracks":len(rows),"body":summary("body"),"color":summary("color"),"per_track":rows,"ground_truth":"none; behavior proxy only; not a release accuracy or color-unknown gate","frozen_video_used":False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True); ap.add_argument("--detector", type=Path, required=True); ap.add_argument("--attribute", type=Path); ap.add_argument("--body-attribute", type=Path); ap.add_argument("--color-attribute", type=Path)
    ap.add_argument("--body-temperature", type=float, default=1.0); ap.add_argument("--color-temperature", type=float, default=1.0); ap.add_argument("--color-baseline-attribute", type=Path); ap.add_argument("--color-baseline-temperature", type=float, default=.75); ap.add_argument("--color-base-gate", type=float, default=.8); ap.add_argument("--color-candidate-gate", type=float, default=.85); ap.add_argument("--color-router-margin", type=float, default=-.1); ap.add_argument("--sample-every", type=int, default=10); ap.add_argument("--fusion-min-share", type=float, default=.60); ap.add_argument("--fusion-min-margin", type=float, default=.12); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); body_path=a.body_attribute or a.attribute; color_path=a.color_attribute or a.attribute;
    if not body_path or not color_path: ap.error("provide --attribute or both --body-attribute and --color-attribute")
    out = audit(a.video, a.detector, body_path, color_path, a.body_temperature, a.color_temperature, a.sample_every, a.fusion_min_share, a.fusion_min_margin, a.color_baseline_attribute, a.color_baseline_temperature, a.color_base_gate, a.color_candidate_gate, a.color_router_margin); out["fusion_policy"]={"window":5,"min_share":a.fusion_min_share,"min_margin":a.fusion_min_margin}; out["body_attribute"]=str(body_path); out["color_attribute"]=str(color_path); out["color_baseline_attribute"]=str(a.color_baseline_attribute) if a.color_baseline_attribute else None; out["color_router"]={"base_gate":a.color_base_gate,"candidate_gate":a.color_candidate_gate,"margin":a.color_router_margin} if a.color_baseline_attribute else None; a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"output":str(a.output),"tracks":out["tracks"],"body":out["body"],"color":out["color"]}, ensure_ascii=False))

if __name__ == "__main__": main()
