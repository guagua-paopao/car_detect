#!/usr/bin/env python3
"""Evaluate body/color checkpoints with independent transforms and scene strata."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_json
from src.multitask_mobilenet_v3 import IMAGENET_MEAN, IMAGENET_STD, model_from_checkpoint

def parse_bool(value):
    return str(value or '').strip().lower() in {'1','true','yes','y','on'}

def metadata(row):
    try: area = float(row.get('width','')) * float(row.get('height',''))
    except (TypeError, ValueError): area = 0.0
    size = str(row.get('vehicle_size','')).strip().lower()
    if str(row.get('small_target','')).strip().lower() in {'1','true','yes'}: size='small'
    if size not in {'small','medium','large'}:
        size = 'unknown' if area <= 0 else 'small' if area < 128*96 else 'medium' if area < 256*192 else 'large'
    if parse_bool(row.get('night')): lighting='night'
    else:
        try: mean=float(row.get('gray_mean') or row.get('photometric_mean') or '')
        except (TypeError, ValueError): mean=math.nan
        lighting='unknown' if math.isnan(mean) else 'low_light' if mean < 64 else 'moderate_light' if mean < 128 else 'daylight'
    if parse_bool(row.get('truncated')): occ='truncated'
    elif parse_bool(row.get('occluded')): occ='occluded'
    elif row.get('truncated') or row.get('occluded'): occ='visible'
    else: occ='unknown'
    return {'vehicle_size':size,'lighting':lighting,'occlusion_level':occ,'source_dataset':row.get('source_dataset') or 'unknown'}

def metrics(pred, target, conf, labels, threshold):
    unk = labels.index('unknown') if 'unknown' in labels else -1
    valid=[i for i,t in enumerate(target) if t != -100]
    selected=[i for i in valid if pred[i] != unk and conf[i] >= threshold]
    return {'evaluated':len(valid),'high_confidence_precision':sum(pred[i]==target[i] for i in selected)/len(selected) if selected else 0.0,'high_confidence_coverage':len(selected)/len(valid) if valid else 0.0,'high_confidence_selected':len(selected),'predicted_unknown_rate':sum(pred[i]==unk for i in valid)/len(valid) if valid else 0.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--body-checkpoint',type=Path,required=True); ap.add_argument('--color-checkpoint',type=Path,required=True)
    ap.add_argument('--body-calibration',type=Path,required=True); ap.add_argument('--color-calibration',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--body-threshold',type=float,default=.8); ap.add_argument('--color-threshold',type=float,default=.7); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    labels=load_json(args.labels)
    bck=torch.load(args.body_checkpoint,map_location='cpu'); cck=torch.load(args.color_checkpoint,map_location='cpu')
    body_labels=[str(x) for x in bck.get('body_types',labels['body_types'])]; color_labels=[str(x) for x in cck.get('colors',labels['colors'])]
    body_temp=float(json.loads(args.body_calibration.read_text())['body_type']['temperature']); color_temp=float(json.loads(args.color_calibration.read_text())['color']['temperature'])
    with args.manifest.open('r',encoding='utf-8-sig',newline='') as h: all_rows=list(csv.DictReader(h))
    device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu')
    bm=model_from_checkpoint(bck,pretrained=False).to(device); cm=model_from_checkpoint(cck,pretrained=False).to(device)
    bm.load_state_dict(bck['model_state']); cm.load_state_dict(cck['model_state']); bm.eval(); cm.eval()
    bt=transforms.Compose([transforms.Resize((int(bck.get('input_size',224)),int(bck.get('input_size',224))),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    ct=transforms.Compose([transforms.Resize((int(cck.get('input_size',224)),int(cck.get('input_size',224))),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    split_reports={}
    for split in ('validation','test'):
        rows=[r for r in all_rows if r.get('split')==split and (r.get('review_status','approved') or 'approved')=='approved']
        class Rows(Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self,i):
                r=rows[i]; p=Path(r['image_path']); p=p if p.is_absolute() else (args.manifest.parent/p).resolve()
                with Image.open(p) as im: im=im.convert('RGB')
                bi=body_labels.index(r['body_type']) if r.get('body_type') in body_labels and str(r.get('body_type_supervised','true')).lower() not in {'false','0','no'} else -100
                ci=color_labels.index(r['color']) if r.get('color') in color_labels and str(r.get('color_supervised','true')).lower() not in {'false','0','no'} else -100
                return bt(im),ct(im),bi,ci,i
        loader=DataLoader(Rows(),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
        bp=[];btg=[];bc=[];cp=[];ctg=[];cc=[]
        with torch.no_grad():
            for bi,ci,btarget,ctarget,_ in loader:
                bl,_=bm(bi.to(device)); _,cl=cm(ci.to(device)); bprob=(bl/body_temp).softmax(1); cprob=(cl/color_temp).softmax(1)
                bconf,bpred=bprob.max(1); cconf,cpred=cprob.max(1)
                bp+=bpred.cpu().tolist(); btg+=btarget.tolist(); bc+=bconf.cpu().tolist(); cp+=cpred.cpu().tolist(); ctg+=ctarget.tolist(); cc+=cconf.cpu().tolist()
        def strat(pred,target,conf,labs,thr,kind):
            groups=defaultdict(lambda:[[],[],[]])
            for i,(p,t,c) in enumerate(zip(pred,target,conf)):
                if t==-100: continue
                for k,v in metadata(rows[i]).items(): groups[f'{k}={v}'][0].append(p); groups[f'{k}={v}'][1].append(t); groups[f'{k}={v}'][2].append(c)
            return {k:metrics(v[0],v[1],v[2],labs,thr) for k,v in sorted(groups.items())}
        split_reports[split]={'rows':len(rows),'body_type':metrics(bp,btg,bc,body_labels,args.body_threshold),'color':metrics(cp,ctg,cc,color_labels,args.color_threshold),'stratified':{'body_type':strat(bp,btg,bc,body_labels,args.body_threshold,'body'),'color':strat(cp,ctg,cc,color_labels,args.color_threshold,'color')}}
    report={'schema_version':'attribute-ensemble-stratified-v1','manifest':str(args.manifest),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'body_checkpoint':str(args.body_checkpoint),'color_checkpoint':str(args.color_checkpoint),'body_temperature':body_temp,'color_temperature':color_temp,'thresholds':{'body_type':args.body_threshold,'color':args.color_threshold},'frozen_video_used':False,'validation':split_reports['validation'],'test':split_reports['test']}
    report['release_target_met']=all(report[s][h]['high_confidence_precision']>=p and report[s][h]['high_confidence_coverage']>=c for s in ('validation','test') for h,p,c in (('body_type',.93,.45),('color',.93,.25)))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'status':'pass','release_target_met':report['release_target_met'],'validation':{k:split_reports['validation'][k] for k in ('body_type','color')},'test':{k:split_reports['test'][k] for k in ('body_type','color')}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
