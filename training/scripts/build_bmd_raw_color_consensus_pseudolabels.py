"""Create conservative train-only BMD color pseudo-labels.

Only unlabeled BMD-45 raw *training* crops are considered.  A label is
accepted when the production teacher and isolated candidate agree, both have
high calibrated confidence, and the candidate agrees across brightness/
contrast perturbations.  This is a training artifact, never validation/test
ground truth and never a substitute for a labeled release gate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
from collections import Counter

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.multitask_mobilenet_v3 import model_from_checkpoint, IMAGENET_MEAN, IMAGENET_STD


def sha(p:Path):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--candidate-temperature',type=float,default=.825); ap.add_argument('--baseline-temperature',type=float,default=.75); ap.add_argument('--confidence',type=float,default=.88); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
    with a.manifest.open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    targets=[r for r in rows if r.get('split')=='train' and r.get('source_dataset')=='BMD-45-RAW' and r.get('color_supervised','false').lower() in {'false','0','no'}]
    if not targets: raise RuntimeError('no unlabeled BMD-45-RAW training rows')
    cck=torch.load(a.candidate,map_location='cpu'); bck=torch.load(a.baseline,map_location='cpu');
    cm=model_from_checkpoint(cck,pretrained=False); bm=model_from_checkpoint(bck,pretrained=False); cm.load_state_dict(cck['model_state']); bm.load_state_dict(bck['model_state']); cm.eval(); bm.eval()
    dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'); cm.to(dev); bm.to(dev)
    size=int(cck.get('input_size',224)); norm=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD); to_tensor=transforms.ToTensor()
    class D(Dataset):
        def __len__(self): return len(targets)
        def __getitem__(self,i):
            r=targets[i]; p=a.manifest.parent/ r['image_path']
            with Image.open(p) as im: im=im.convert('RGB')
            variants=[]
            for bright,contrast in [(1.0,1.0),(.78,1.08),(1.22,.92)]:
                x=ImageEnhance.Brightness(im).enhance(bright); x=ImageEnhance.Contrast(x).enhance(contrast); x=transforms.Resize((size,size),antialias=True)(x); variants.append(norm(to_tensor(x)))
            return torch.stack(variants),i
    loader=DataLoader(D(),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=='cuda')
    accepted={}; stats=Counter(); color_counts=Counter()
    with torch.no_grad():
        for batch,indexes in loader:
            # [B,3,C,H,W] -> three augmentation passes.
            b,c=batch.shape[0],batch.shape[1]; x=batch.to(dev)
            cp=[]; bp=[]; cc=[]; bc=[]
            for j in range(c):
                bl,cl=bm(x[:,j]); ql,rl=cm(x[:,j]);
                bp.append((bl/a.baseline_temperature).softmax(1)); cp.append((ql/a.candidate_temperature).softmax(1));
            # teacher/candidate must agree on base and all perturbations.
            for k,idx in enumerate(indexes.tolist()):
                bpred=[int(z[k].argmax()) for z in bp]; cpred=[int(z[k].argmax()) for z in cp]
                bconf=[float(z[k].max()) for z in bp]; cconf=[float(z[k].max()) for z in cp]
                label=cpred[0]; agree=(label==bpred[0] and label!=9 and all(z==label for z in cpred) and all(z==label for z in bpred))
                strong=agree and min(bconf+cconf)>=a.confidence
                stats['seen']+=1
                if agree: stats['teacher_candidate_agree']+=1
                if strong:
                    accepted[idx]=label; color_counts[cck['colors'][label]]+=1; stats['accepted']+=1
                else: stats['rejected']+=1
    out_rows=[]
    for i,r in enumerate(rows):
        nr=dict(r)
        if i in accepted:
            nr['color']=cck['colors'][accepted[i]]; nr['color_supervised']='true'; nr['annotation_source']='auto_consensus_pseudo_v1'; nr['color_review_status']='auto_consensus'; nr['review_method']='production_candidate_augmented_consensus'; nr['pseudo_label']='true'; nr['pseudo_label_confidence']=str(a.confidence)
        out_rows.append(nr)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]);
    for extra in ('pseudo_label','pseudo_label_confidence'):
        if extra not in fields: fields.append(extra)
    with a.output.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(out_rows)
    report={'schema_version':'bmd-raw-color-consensus-pseudolabel-v1','input_manifest':str(a.manifest),'output_manifest':str(a.output),'input_sha256':sha(a.manifest),'output_sha256':sha(a.output),'policy':'train-only BMD-45-RAW; production teacher and candidate agreement plus 3 brightness/contrast variants; rejected conflicts remain unknown','frozen_video_used':False,'test_or_validation_used':False,'threshold':a.confidence,'rows':dict(stats),'accepted_color_counts':dict(sorted(color_counts.items())),'accepted_rate':stats['accepted']/max(1,stats['seen']),'candidate_checkpoint':str(a.candidate),'baseline_checkpoint':str(a.baseline),'license_inherited_from_source':True}
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
