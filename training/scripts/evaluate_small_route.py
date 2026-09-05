#!/usr/bin/env python3
"""Validation-selected routing between a small-target candidate and baseline."""
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


def collect(args, split, models, temperatures, labels, transform, device):
    import torch
    from torch.utils.data import DataLoader
    ds = VehicleAttributeDataset(args.manifest, split=split, body_types=labels, colors=["unknown"], transform=transform, training=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    outputs = [[] for _ in models]; targets=[]
    for model in models: model.eval()
    with torch.no_grad():
        for images, target, _, _ in loader:
            images=images.to(device)
            for idx,model in enumerate(models):
                logits,_=model(images); prob=(logits/temperatures[idx]).softmax(1); conf,pred=prob.max(1); outputs[idx].extend(zip(pred.cpu().tolist(),conf.cpu().tolist()))
            targets.extend(target.tolist())
    sizes=[row.get("vehicle_size", "unknown") for row in ds.rows]
    return outputs,targets,sizes


def route(outputs, targets, sizes, labels, cand_threshold, base_threshold, margin):
    unknown=labels.index("unknown"); pred=[]; conf=[]
    for idx,target in enumerate(targets):
        cp,cc=outputs[0][idx]; bp,bc=outputs[1][idx]
        c_ok=cp!=unknown and cc>=cand_threshold; b_ok=bp!=unknown and bc>=base_threshold
        if sizes[idx]=="small" and c_ok and (not b_ok or cc>=bc+margin): p,c=cp,cc
        elif b_ok: p,c=bp,bc
        elif c_ok: p,c=cp,cc
        else: p,c=unknown,0.0
        pred.append(p); conf.append(c)
    valid=[i for i,t in enumerate(targets) if t!=-100]; selected=[i for i in valid if pred[i]!=unknown]
    small=[i for i in valid if sizes[i]=="small"]; small_selected=[i for i in small if pred[i]!=unknown]
    def summarize(indices,selected_indices):
        return {"evaluated":len(indices),"precision":sum(pred[i]==targets[i] for i in selected_indices)/len(selected_indices) if selected_indices else 0.0,"coverage":len(selected_indices)/len(indices) if indices else 0.0,"selected":len(selected_indices),"accuracy":sum(pred[i]==targets[i] for i in indices)/len(indices) if indices else 0.0}
    return {"overall":summarize(valid,selected),"small":summarize(small,small_selected),"predicted_unknown_rate":sum(pred[i]==unknown for i in valid)/len(valid) if valid else 0.0}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--labels",type=Path,required=True); ap.add_argument("--candidate-checkpoint",type=Path,required=True); ap.add_argument("--baseline-checkpoint",type=Path,required=True); ap.add_argument("--candidate-calibration",type=Path,required=True); ap.add_argument("--baseline-calibration",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--device",default="cuda"); args=ap.parse_args()
    import torch
    from torchvision import transforms
    data=load_json(args.labels); labels=data["body_types"]; ccal=json.loads(args.candidate_calibration.read_text()); bcal=json.loads(args.baseline_calibration.read_text()); ctemp=float(ccal["body_type"]["temperature"]); btemp=float(bcal["body_type"]["temperature"]); cck=torch.load(args.candidate_checkpoint,map_location="cpu"); bck=torch.load(args.baseline_checkpoint,map_location="cpu"); cm=model_from_checkpoint(cck,pretrained=False); bm=model_from_checkpoint(bck,pretrained=False); cm.load_state_dict(cck["model_state"]); bm.load_state_dict(bck["model_state"]); device=torch.device(args.device); cm.to(device); bm.to(device); size=int(cck.get("input_size",256)); transform=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    val_out,val_t,val_s=collect(args,"validation",[cm,bm],[ctemp,btemp],labels,transform,device); test_out,test_t,test_s=collect(args,"test",[cm,bm],[ctemp,btemp],labels,transform,device)
    trials=[]
    for ct in [round(x/100,2) for x in range(45,91,5)]:
      for bt in [round(x/100,2) for x in range(45,91,5)]:
       for margin in (0.0,0.03,0.06):
        m=route(val_out,val_t,val_s,labels,ct,bt,margin); trials.append({"candidate_threshold":ct,"baseline_threshold":bt,"margin":margin,**m})
    valid=[x for x in trials if x["overall"]["precision"]>=.93 and x["overall"]["coverage"]>=.45 and x["small"]["precision"]>=.93]
    selected=max(valid,key=lambda x:(x["small"]["coverage"],x["overall"]["coverage"],x["overall"]["precision"])) if valid else {"candidate_threshold":.75,"baseline_threshold":.75,"margin":0.0}
    ct=float(selected["candidate_threshold"]); bt=float(selected["baseline_threshold"]); margin=float(selected["margin"])
    report={"schema_version":"small-route-evaluation-v1","candidate_checkpoint":str(args.candidate_checkpoint),"baseline_checkpoint":str(args.baseline_checkpoint),"selection":{"candidate_threshold":ct,"baseline_threshold":bt,"margin":margin,"validation_trials":trials},"validation":route(val_out,val_t,val_s,labels,ct,bt,margin),"test":route(test_out,test_t,test_s,labels,ct,bt,margin),"frozen_video_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
