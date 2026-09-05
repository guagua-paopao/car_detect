#!/usr/bin/env python3
"""Evaluate independent body/color checkpoints as a fail-closed ensemble."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_json
from src.attribute_dataset import VehicleAttributeDataset
from src.multitask_mobilenet_v3 import model_from_checkpoint, IMAGENET_MEAN, IMAGENET_STD

def head_metrics(pred, target, conf, labels, threshold):
    unknown = labels.index("unknown")
    selected = [i for i,(p,c,t) in enumerate(zip(pred,conf,target)) if t != -100 and p != unknown and c >= threshold]
    valid = [i for i,t in enumerate(target) if t != -100]
    return {
        "evaluated": len(valid),
        "accuracy": sum(pred[i] == target[i] for i in valid)/len(valid) if valid else 0.0,
        "high_confidence_precision": sum(pred[i] == target[i] for i in selected)/len(selected) if selected else 0.0,
        "high_confidence_coverage": len(selected)/len(valid) if valid else 0.0,
        "high_confidence_selected": len(selected),
        "predicted_unknown_rate": sum(pred[i] == unknown for i in valid)/len(valid) if valid else 0.0,
        "true_unknown_rate": sum(target[i] == unknown for i in valid)/len(valid) if valid else 0.0,
    }

def run_split(args, split, body_model, color_model, body_temp, color_temp, body_types, colors, transform, device, color_unknown_fallback_threshold=None):
    import torch
    from torch.utils.data import DataLoader
    ds = VehicleAttributeDataset(args.manifest, split=split, body_types=body_types, colors=colors, transform=transform, training=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    bp=[]; bt=[]; bc=[]; cp=[]; ct=[]; cc=[]
    body_model.eval(); color_model.eval()
    with torch.no_grad():
        for images, body_target, color_target, _ in loader:
            images = images.to(device)
            bl,_ = body_model(images); _,cl = color_model(images)
            bprob=(bl/body_temp).softmax(1); cprob=(cl/color_temp).softmax(1)
            bconf,bpred=bprob.max(1); cconf,cpred=cprob.max(1)
            if color_unknown_fallback_threshold is not None:
                unknown_idx = colors.index("unknown")
                nonunknown = cprob.clone()
                nonunknown[:, unknown_idx] = -1.0
                fallback_conf, fallback_pred = nonunknown.max(1)
                use_fallback = (cpred == unknown_idx) & (fallback_conf >= color_unknown_fallback_threshold)
                cpred = cpred.clone()
                cconf = cconf.clone()
                cpred[use_fallback] = fallback_pred[use_fallback]
                cconf[use_fallback] = fallback_conf[use_fallback]
            bp.extend(bpred.cpu().tolist()); bt.extend(body_target.tolist()); bc.extend(bconf.cpu().tolist())
            cp.extend(cpred.cpu().tolist()); ct.extend(color_target.tolist()); cc.extend(cconf.cpu().tolist())
    return {"split":split,"body_type":head_metrics(bp,bt,bc,body_types,args.body_threshold),"color":head_metrics(cp,ct,cc,colors,args.color_threshold),"rows":len(ds)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--labels",type=Path,required=True)
    ap.add_argument("--body-checkpoint",type=Path,required=True); ap.add_argument("--color-checkpoint",type=Path,required=True)
    ap.add_argument("--body-calibration",type=Path,required=True); ap.add_argument("--color-calibration",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--device",default="cuda"); ap.add_argument("--body-threshold",type=float,default=.75); ap.add_argument("--color-threshold",type=float,default=.70)
    ap.add_argument("--color-unknown-fallback",action="store_true",help="select a non-unknown fallback threshold on validation only")
    args=ap.parse_args()
    import torch
    from torchvision import transforms
    labels=load_json(args.labels); body_types=labels["body_types"]; colors=labels["colors"]
    bc=json.loads(args.body_calibration.read_text()); cc=json.loads(args.color_calibration.read_text())
    body_temp=float(bc["body_type"]["temperature"]); color_temp=float(cc["color"]["temperature"])
    body_ck=torch.load(args.body_checkpoint,map_location="cpu"); color_ck=torch.load(args.color_checkpoint,map_location="cpu")
    body_model=model_from_checkpoint(body_ck,pretrained=False); color_model=model_from_checkpoint(color_ck,pretrained=False)
    body_model.load_state_dict(body_ck["model_state"]); color_model.load_state_dict(color_ck["model_state"])
    device=torch.device(args.device); body_model.to(device); color_model.to(device)
    size=int(body_ck["input_size"])
    transform=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    report={"schema_version":"attribute-ensemble-v1","body_checkpoint":str(args.body_checkpoint),"color_checkpoint":str(args.color_checkpoint),"body_temperature":body_temp,"color_temperature":color_temp,"frozen_video_used":False}
    fallback_threshold = None
    if args.color_unknown_fallback:
        candidates = [round(x, 2) for x in [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]]
        trials = []
        for threshold in candidates:
            trial = run_split(args,"validation",body_model,color_model,body_temp,color_temp,body_types,colors,transform,device,threshold)
            cm = trial["color"]
            trials.append({"threshold":threshold,"precision":cm["high_confidence_precision"],"coverage":cm["high_confidence_coverage"],"unknown_rate":cm["predicted_unknown_rate"]})
        valid = [x for x in trials if x["precision"] >= .93 and x["coverage"] >= .25]
        if valid:
            valid.sort(key=lambda x:(x["unknown_rate"], -x["coverage"], -x["precision"]))
            fallback_threshold = valid[0]["threshold"]
        report["color_unknown_fallback_policy"]={"enabled":True,"selected_threshold":fallback_threshold,"validation_trials":trials}
    report["validation"]=run_split(args,"validation",body_model,color_model,body_temp,color_temp,body_types,colors,transform,device,fallback_threshold)
    report["test"]=run_split(args,"test",body_model,color_model,body_temp,color_temp,body_types,colors,transform,device,fallback_threshold)
    report["release_target_met"]=all([report["validation"]["body_type"]["high_confidence_precision"]>=.93,report["test"]["body_type"]["high_confidence_precision"]>=.93,report["validation"]["body_type"]["high_confidence_coverage"]>=.45,report["test"]["body_type"]["high_confidence_coverage"]>=.45,report["validation"]["color"]["high_confidence_precision"]>=.93,report["test"]["color"]["high_confidence_precision"]>=.93,report["validation"]["color"]["high_confidence_coverage"]>=.25,report["test"]["color"]["high_confidence_coverage"]>=.25])
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
