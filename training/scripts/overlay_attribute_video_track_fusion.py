#!/usr/bin/env python3
"""Render attribute predictions with short-term IoU track fusion.

This is an evaluation/replay tool.  It does not use the frozen video for
training or threshold selection; it reports frame and track stability only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=1, keepdims=True)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = max(0, ax2-ax1) * max(0, ay2-ay1); ab = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / max(1e-6, aa + ab - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True); ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True); ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output-video", type=Path, required=True); ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--type-threshold", type=float, default=.75); ap.add_argument("--color-threshold", type=float, default=.70)
    ap.add_argument("--iou-threshold", type=float, default=.30); ap.add_argument("--history", type=int, default=5)
    args = ap.parse_args()
    import onnxruntime as ort
    labels = json.loads(args.labels.read_text(encoding="utf-8")); bodies = labels["body_types"]; colors = labels["colors"]
    sess = ort.InferenceSession(str(args.model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    det_rows = [json.loads(x) for x in args.detections.read_text(encoding="utf-8").splitlines() if x.strip()]
    cap = cv2.VideoCapture(str(args.video));
    if not cap.isOpened(): raise RuntimeError(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.; width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(det_rows) != nframes: raise RuntimeError(f"detections {len(det_rows)} != frames {nframes}")
    args.output_video.parent.mkdir(parents=True, exist_ok=True); args.output_report.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracks = {}; next_id = 1; frame_index = 0; switches = 0; observations = 0; stable = 0; body_counts = Counter(); color_counts = Counter(); window_body = Counter(); window_color = Counter(); track_seen = Counter()
    unknown_body = unknown_color = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        detections = det_rows[frame_index]["detections"]; crops=[]; meta=[]; used=set()
        # Greedy IoU association; detections are already confidence ordered.
        assignments=[]
        for di,d in enumerate(detections):
            box=tuple(int(round(v)) for v in d["xyxy"]); best=None; best_iou=0.
            for tid,t in tracks.items():
                if tid in used or t["class_name"] != d.get("class_name"): continue
                score=iou(box,t["box"])
                if score>best_iou: best_iou=score; best=tid
            if best is None or best_iou < args.iou_threshold:
                best=next_id; next_id+=1; tracks[best]={"class_name":d.get("class_name"),"box":box,"body":deque(maxlen=args.history),"color":deque(maxlen=args.history),"last_body":"unknown","last_color":"unknown","body_pending":None,"color_pending":None,"body_streak":0,"color_streak":0,"miss":0}
            used.add(best); tracks[best]["box"]=box; tracks[best]["miss"]=0; assignments.append((di,best,box))
            x1,y1,x2,y2=box; x1=max(0,min(width-1,x1)); x2=max(0,min(width,x2)); y1=max(0,min(height-1,y1)); y2=max(0,min(height,y2))
            if x2>x1 and y2>y1:
                c=cv2.resize(frame[y1:y2,x1:x2],(args.input_size,args.input_size),interpolation=cv2.INTER_LINEAR); c=cv2.cvtColor(c,cv2.COLOR_BGR2RGB).astype(np.float32)/255.; crops.append(np.transpose(c,(2,0,1))); meta.append((di,best))
        probs=[]
        for s in range(0,len(crops),args.batch):
            b=np.asarray(crops[s:s+args.batch],dtype=np.float32);
            if len(b):
                bl,cl=sess.run(None,{input_name:b}); probs.extend(zip(softmax(np.asarray(bl)),softmax(np.asarray(cl))))
        pred_by_det={}
        for (di,tid), (bp,cp) in zip(meta,probs):
            t=tracks[tid]; t["body"].append(bp); t["color"].append(cp); body_avg=np.mean(np.asarray(t["body"]),axis=0); color_avg=np.mean(np.asarray(t["color"]),axis=0); bi=int(body_avg.argmax()); ci=int(color_avg.argmax()); bc=float(body_avg[bi]); cc=float(color_avg[ci]); bnew=bodies[bi] if bc>=args.type_threshold else "unknown"; cnew=colors[ci] if cc>=args.color_threshold else "unknown"; observations+=1
            for key,new,conf,last in (("body",bnew,bc,"last_body"),("color",cnew,cc,"last_color")):
                if new != t[last] and new != "unknown":
                    if t[key+"_pending"] == new: t[key+"_streak"] += 1
                    else: t[key+"_pending"]=new; t[key+"_streak"]=1
                    if t[key+"_streak"] >= 3: t[last]=new; t[key+"_pending"]=None; t[key+"_streak"]=0; switches+=1
                elif new == t[last]: t[key+"_pending"]=None; t[key+"_streak"]=0
            pred_by_det[di]=(t["last_body"],bc,t["last_color"],cc,tid)
            track_seen[tid]+=1
        for tid,t in tracks.items(): t["miss"] += 1
        tracks={tid:t for tid,t in tracks.items() if t["miss"]<=10}
        in_window=36<=frame_index/fps<48
        for di,d in enumerate(detections):
            x1,y1,x2,y2=[int(round(v)) for v in d["xyxy"]]; b,bc,c,cc,tid=pred_by_det.get(di,("unknown",0.,"unknown",0.,0)); body_counts[b]+=1; color_counts[c]+=1;
            if in_window: window_body[b]+=1; window_color[c]+=1
            if b=="unknown": unknown_body+=1
            if c=="unknown": unknown_color+=1
            text=f"{d.get('class_name','vehicle')} {d.get('confidence',0):.2f} | {b} {bc:.2f} | {c} {cc:.2f} | T{tid}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,215,255),2); ty=max(18,y1-6); cv2.putText(frame,text,(x1,ty),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,text,(x1,ty),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,215,255),1,cv2.LINE_AA)
        cv2.rectangle(frame,(0,0),(width,30),(0,0,0),-1); cv2.putText(frame,f"TRACK-FUSED ATTRIBUTES | {frame_index/fps:05.1f}s",(10,21),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA); writer.write(frame); frame_index+=1
    cap.release(); writer.release();
    report={"schema_version":"1.0","video":{"path":str(args.video),"fps":fps,"frames":frame_index,"width":width,"height":height},"model":str(args.model),"settings":{"input_size":args.input_size,"type_threshold":args.type_threshold,"color_threshold":args.color_threshold,"iou_threshold":args.iou_threshold,"history":args.history,"providers":sess.get_providers()},"tracks":{"created":next_id-1,"observations":observations,"label_switch_count":switches,"stability_rate":1.0-switches/max(1,observations),"min_observations":min(track_seen.values()) if track_seen else 0},"all_video":{"body_type":dict(body_counts),"color":dict(color_counts)},"window_36_48s":{"body_type":dict(window_body),"color":dict(window_color)},"unknown_rates":{"body_type":unknown_body/max(1,observations),"color":unknown_color/max(1,observations)},"output_video":str(args.output_video)}
    args.output_report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
