#!/usr/bin/env python3
"""Merge independently selected per-head temperatures for an isolated candidate."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--body',type=Path,required=True); ap.add_argument('--color',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    b=json.loads(args.body.read_text(encoding='utf-8')); c=json.loads(args.color.read_text(encoding='utf-8'))
    out={'schema_version':'merged-head-calibration-v1','method':'per_head_temperature_scaling','body_type':{'temperature':float(b['body_type']['temperature']),'source':str(args.body)},'color':{'temperature':float(c['color']['temperature']),'source':str(args.color)},'frozen_video_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
