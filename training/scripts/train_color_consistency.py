#!/usr/bin/env python3
"""Train an isolated candidate with unlabeled color-consistency regularization.

The unlabeled stream is train-only BMD raw data. It never creates a color
class target: only the model's color distribution is required to remain stable
under photometric perturbations. Validation/test are read from the labeled
rows of the same manifest and are never used for optimization decisions beyond
ordinary validation model selection.
"""
from __future__ import annotations
import argparse, csv, json, random, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_json, sha256_file, trainable_attribute_labels, write_json
from src.attribute_dataset import VehicleAttributeDataset
from src.multitask_mobilenet_v3 import (
    IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint, architecture_from_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--consistency-weight", type=float, default=.08)
    p.add_argument("--consistency-temperature", type=float, default=.75)
    p.add_argument("--raw-body-weight", type=float, default=0.0)
    p.add_argument("--color-threshold", type=float, default=.70)
    p.add_argument("--type-threshold", type=float, default=.75)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--dataset-version", default="bmd-raw-v29-consistency")
    return p.parse_args()


def evaluate(model, loader, device, body_count, color_count, body_unknown, color_unknown, type_thr, color_thr):
    model.eval(); bp=[]; bt=[]; cp=[]; ct=[]; bconf=[]; cconf=[]
    with torch.no_grad():
        for images, body, color, _ in loader:
            body_logits, color_logits = model(images.to(device))
            bpred = body_logits.argmax(1).cpu(); cpred = color_logits.argmax(1).cpu()
            bc = body_logits.softmax(1).amax(1).cpu(); cc = color_logits.softmax(1).amax(1).cpu()
            for p,t,c in zip(bpred.tolist(),body.tolist(),bc.tolist()):
                if t != -100: bp.append(p); bt.append(t); bconf.append(c)
            for p,t,c in zip(cpred.tolist(),color.tolist(),cc.tolist()):
                if t != -100: cp.append(p); ct.append(t); cconf.append(c)
    def f1(pred,tgt,n):
        vals=[]
        for k in range(n):
            tp=sum(p==k and t==k for p,t in zip(pred,tgt)); fp=sum(p==k and t!=k for p,t in zip(pred,tgt)); fn=sum(p!=k and t==k for p,t in zip(pred,tgt)); d=2*tp+fp+fn
            if d: vals.append(2*tp/d)
        return sum(vals)/len(vals) if vals else 0.0
    bs=[i for i,(p,c) in enumerate(zip(bp,bconf)) if p!=body_unknown and c>=type_thr]
    cs=[i for i,(p,c) in enumerate(zip(cp,cconf)) if p!=color_unknown and c>=color_thr]
    return {
        "body_type_accuracy": sum(p==t for p,t in zip(bp,bt))/len(bt) if bt else 0.0,
        "body_type_macro_f1": f1(bp,bt,body_count),
        "color_accuracy": sum(p==t for p,t in zip(cp,ct))/len(ct) if ct else 0.0,
        "color_macro_f1": f1(cp,ct,color_count),
        "body_type_evaluated": len(bt), "color_evaluated": len(ct),
        "body_type_high_confidence_precision": sum(bp[i]==bt[i] for i in bs)/len(bs) if bs else 0.0,
        "body_type_high_confidence_coverage": len(bs)/len(bt) if bt else 0.0,
        "color_high_confidence_precision": sum(cp[i]==ct[i] for i in cs)/len(cs) if cs else 0.0,
        "color_high_confidence_coverage": len(cs)/len(ct) if ct else 0.0,
    }


def main():
    a=parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    dev=torch.device(a.device); labels=load_json(a.labels); body_types,colors=trainable_attribute_labels(labels)
    ckp=torch.load(a.init_checkpoint,map_location="cpu"); arch=architecture_from_checkpoint(ckp)
    model=model_from_checkpoint(ckp,pretrained=False).to(dev); model.load_state_dict(ckp["model_state"])
    size=a.input_size; norm=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
    train_transform=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.28,contrast=.28,saturation=.14,hue=.025),transforms.ToTensor(),norm])
    eval_transform=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),norm])
    labeled=VehicleAttributeDataset(a.manifest,split="train",body_types=body_types,colors=colors,transform=train_transform,training=True,return_pseudo=True)
    validation=VehicleAttributeDataset(a.manifest,split="validation",body_types=body_types,colors=colors,transform=eval_transform,training=False)
    test=VehicleAttributeDataset(a.manifest,split="test",body_types=body_types,colors=colors,transform=eval_transform,training=False)
    with a.manifest.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    raw=[r for r in rows if r.get("split")=="train" and r.get("source_dataset") in {"BMD-45-RAW","BMD-45-RAW-COCO"} and r.get("color_supervised","false").lower() in {"false","0","no"}]
    if not raw: raise RuntimeError("no train-only raw rows for consistency stream")
    class Raw(Dataset):
        def __len__(self): return len(raw)
        def __getitem__(self,i):
            with Image.open((a.manifest.parent/raw[i]["image_path"]).resolve()) as im: im=im.convert("RGB")
            v1=ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(.82)).enhance(1.10)
            v2=ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.18)).enhance(.92)
            def t(x): return norm(transforms.ToTensor()(transforms.Resize((size,size),antialias=True)(x)))
            return torch.stack([t(v1),t(v2)])
    l_loader=DataLoader(labeled,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=="cuda",drop_last=True)
    r_loader=DataLoader(Raw(),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=="cuda",drop_last=True)
    v_loader=DataLoader(validation,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=="cuda")
    t_loader=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=="cuda")
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,a.epochs)); amp=dev.type=="cuda"; scaler=torch.cuda.amp.GradScaler(enabled=amp)
    body_targets,color_targets=labeled.targets(); bw=torch.bincount(torch.tensor([x for x in body_targets if x>=0]),minlength=len(body_types)).float(); cw=torch.bincount(torch.tensor([x for x in color_targets if x>=0]),minlength=len(colors)).float(); bw=(bw.clamp_min(1).rsqrt()/bw.clamp_min(1).rsqrt().mean()).to(dev); cw=(cw.clamp_min(1).rsqrt()/cw.clamp_min(1).rsqrt().mean()).to(dev)
    body_loss=nn.CrossEntropyLoss(weight=bw,label_smoothing=.05); color_loss=nn.CrossEntropyLoss(weight=cw,label_smoothing=.05); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); best=-1.; logs=[]
    def repeat(loader):
        while True:
            for item in loader:
                yield item
    for epoch in range(a.epochs):
        model.train(); running=0.; n=0; raw_it=repeat(r_loader)
        for batch in l_loader:
            if len(batch)==5: images,bt,ct,_,pseudo=batch
            else: images,bt,ct,_=batch; pseudo=torch.zeros_like(bt,dtype=torch.bool)
            raw_pair=next(raw_it).to(dev); images=images.to(dev); bt=bt.to(dev); ct=ct.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                bl,cl=model(images); loss=torch.zeros((),device=dev)
                bm=bt!=-100; cmask=ct!=-100
                if bm.any(): loss=loss+body_loss(bl[bm],bt[bm])
                if cmask.any(): loss=loss+color_loss(cl[cmask],ct[cmask])
                # Raw stream is intentionally label-free for color.
                _,c1=model(raw_pair[:,0]); _,c2=model(raw_pair[:,1]); temp=max(.1,a.consistency_temperature)
                p1=torch.softmax(c1/temp,1); p2=torch.softmax(c2/temp,1)
                cons=torch.nn.functional.mse_loss(p1,p2)*(temp*temp)
                loss=loss+float(a.consistency_weight)*cons
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update(); running+=float(loss.detach().cpu()); n+=1
        sched.step(); metrics=evaluate(model,v_loader,dev,len(body_types),len(colors),body_types.index("unknown"),colors.index("unknown"),a.type_threshold,a.color_threshold); metrics.update(epoch=epoch,train_loss=running/max(1,n),raw_consistency_weight=a.consistency_weight,raw_rows=len(raw)); logs.append(metrics); score=(metrics["body_type_macro_f1"]+metrics["color_macro_f1"])/2
        torch.save({"schema_version":"1.0","epoch":epoch,"best_score":max(best,score),"model_state":model.state_dict(),"architecture":arch,"input_size":size,"resize_mode":"stretch","body_types":body_types,"colors":colors,"labels_version":labels["labels_version"]},out/"last.pt")
        if score>best: best=score; torch.save({"schema_version":"1.0","epoch":epoch,"best_score":best,"model_state":model.state_dict(),"architecture":arch,"input_size":size,"resize_mode":"stretch","body_types":body_types,"colors":colors,"labels_version":labels["labels_version"]},out/"best.pt")
        print(json.dumps(metrics,ensure_ascii=False))
    best_ck=torch.load(out/"best.pt",map_location="cpu"); model.load_state_dict(best_ck["model_state"]); model.to(dev); test_metrics=evaluate(model,t_loader,dev,len(body_types),len(colors),body_types.index("unknown"),colors.index("unknown"),a.type_threshold,a.color_threshold)
    card={"schema_version":"color-consistency-candidate-v1","run_id":out.name,"started_at":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"finished_at":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"manifest":str(a.manifest),"frozen_video_used":False,"train_only_raw_rows":len(raw),"consistency_weight":a.consistency_weight,"consistency_temperature":a.consistency_temperature,"init_checkpoint":str(a.init_checkpoint),"validation":logs[-1],"test":test_metrics,"artifacts":{"best_pt":str(out/"best.pt"),"best_pt_sha256":sha256_file(out/"best.pt"),"last_pt":str(out/"last.pt"),"last_pt_sha256":sha256_file(out/"last.pt")},"release_eligible":False,"deployment_performed":False,"deployment_paused_by_user":True,"note":"isolated research candidate; requires independent hard-case, ONNX/TensorRT, C++ and track gates"}
    write_json(out/"model_card.json",card); write_json(out/"training_log.json",logs); write_json(out/"test_metrics.json",test_metrics)
    print(json.dumps({"validation":logs[-1],"test":test_metrics,"output":str(out)},ensure_ascii=False))


if __name__=="__main__": main()
