#!/usr/bin/env python3
"""Select per-predicted-class confidence thresholds on validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attribute_dataset import VehicleAttributeDataset
from src.common import load_json
from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint


def collect(args, split, model, labels, transform, temperature, device):
    import torch
    from torch.utils.data import DataLoader
    ds = VehicleAttributeDataset(args.manifest, split=split, body_types=labels, colors=["unknown"], transform=transform, training=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    preds=[]; confs=[]; targets=[]; rows=[]
    model.eval()
    with torch.no_grad():
        for images, target, _, paths in loader:
            logits,_=model(images.to(device)); prob=(logits/temperature).softmax(1); conf,pred=prob.max(1)
            preds.extend(pred.cpu().tolist()); confs.extend(conf.cpu().tolist()); targets.extend(target.tolist())
            rows.extend(paths)
    return ds, preds, confs, targets, rows


def apply(preds, confs, targets, labels, thresholds):
    unknown=labels.index("unknown"); valid=[i for i,t in enumerate(targets) if t!=-100]
    selected=[i for i in valid if preds[i]!=unknown and confs[i]>=thresholds.get(labels[preds[i]],1.0)]
    return {
      "evaluated":len(valid),
      "accuracy":sum(preds[i]==targets[i] for i in valid)/len(valid) if valid else 0.0,
      "high_confidence_precision":sum(preds[i]==targets[i] for i in selected)/len(selected) if selected else 0.0,
      "high_confidence_coverage":len(selected)/len(valid) if valid else 0.0,
      "high_confidence_selected":len(selected),
      "predicted_unknown_rate":sum(preds[i]==unknown for i in valid)/len(valid) if valid else 0.0,
      "true_unknown_rate":sum(targets[i]==unknown for i in valid)/len(valid) if valid else 0.0,
    }


def select_thresholds(preds, confs, targets, labels, target_precision):
    unknown=labels.index("unknown"); thresholds={}; table={}
    for class_id,label in enumerate(labels):
        if class_id==unknown: continue
        values=[(confs[i],preds[i]==targets[i]) for i,t in enumerate(targets) if t!=-100 and preds[i]==class_id]
        candidates=sorted({round(v,5) for v,_ in values} | {0.0,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95}, reverse=False)
        valid=[]
        for threshold in candidates:
            chosen=[ok for conf,ok in values if conf>=threshold]
            precision=sum(chosen)/len(chosen) if chosen else 0.0
            if chosen and precision>=target_precision: valid.append((len(chosen),precision,threshold))
        if valid:
            count,precision,threshold=max(valid,key=lambda x:(x[0],-x[2]))
            thresholds[label]=threshold; table[label]={"threshold":threshold,"selected":count,"precision":precision,"candidates":len(values)}
        else:
            thresholds[label]=1.0; table[label]={"threshold":1.0,"selected":0,"precision":0.0,"candidates":len(values)}
    return thresholds,table


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--labels",type=Path,required=True); ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--calibration",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--target-precision",type=float,default=.93); ap.add_argument("--device",default="cuda"); args=ap.parse_args()
    import torch
    from torchvision import transforms
    label_data=load_json(args.labels); labels=label_data["body_types"]; cal=json.loads(args.calibration.read_text()); temperature=float(cal["body_type"]["temperature"]); ck=torch.load(args.checkpoint,map_location="cpu"); model=model_from_checkpoint(ck,pretrained=False); model.load_state_dict(ck["model_state"]); device=torch.device(args.device); model.to(device); size=int(ck.get("input_size",256)); transform=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    val_ds,vp,vc,vt,_=collect(args,"validation",model,labels,transform,temperature,device); test_ds,tp,tc,tt,_=collect(args,"test",model,labels,transform,temperature,device)
    thresholds,table=select_thresholds(vp,vc,vt,labels,args.target_precision)
    report={"schema_version":"class-threshold-calibration-v1","checkpoint":str(args.checkpoint),"temperature":temperature,"target_precision":args.target_precision,"thresholds":thresholds,"validation_per_class":table,"validation":apply(vp,vc,vt,labels,thresholds),"test":apply(tp,tc,tt,labels,thresholds),"frozen_video_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
