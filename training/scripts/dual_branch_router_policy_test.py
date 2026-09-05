#!/usr/bin/env python3
"""Pure policy contract for v32; TensorRT execution remains a separate gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def route(base_label, base_conf, cand_label, cand_conf, base_gate=.8, cand_gate=.9, margin=.05):
    if base_conf < base_gate and cand_conf >= cand_gate and cand_conf >= base_conf + margin:
        return cand_label, cand_conf, "candidate"
    if base_conf < base_gate:
        return "unknown", 0.0, "unknown"
    return base_label, base_conf, "baseline"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    vectors=[
        ("white",.92,"red",.99,"white","baseline"),
        ("unknown",.61,"blue",.94,"blue","candidate"),
        ("unknown",.79,"blue",.83,"unknown","unknown"),
        ("white",.79,"blue",.84,"unknown","unknown"),
        ("black",.79,"black",.90,"black","candidate"),
    ]
    results=[]
    for b,bc,c,cc,expected,source in vectors:
        label,conf,selected=route(b,bc,c,cc)
        assert label==expected and selected==source, (b,bc,c,cc,label,selected)
        results.append({'baseline':{'label':b,'confidence':bc},'candidate':{'label':c,'confidence':cc},'selected':selected,'label':label,'confidence':conf})
    report={'schema_version':'v32-adaptive-router-policy-contract-v1','policy':{'base_gate':.8,'candidate_gate':.9,'margin':.05,'unknown_on_conflict':True},'vectors':results,'gate':True,'frozen_video_used':False,'deployment_performed':False,'tensorrt_execution':'pending_candidate_toolchain'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
