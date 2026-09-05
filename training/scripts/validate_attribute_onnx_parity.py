#!/usr/bin/env python3
"""Compare PyTorch checkpoint and exported calibrated ONNX predictions."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--onnx',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--calibration',type=Path,required=True); ap.add_argument('--labels',type=Path,required=True); ap.add_argument('--split',default='validation'); ap.add_argument('--limit',type=int,default=512); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    import torch, onnxruntime as ort
    from torchvision import transforms
    import sys; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.multitask_mobilenet_v3 import model_from_checkpoint
    ck=torch.load(args.checkpoint,map_location='cpu'); cal=json.loads(args.calibration.read_text()); size=int(ck['input_size']); mean=ck['normalization']['mean']; std=ck['normalization']['std']; tf=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(mean,std)])
    with args.manifest.open('r',encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if r.get('split')==args.split and r.get('review_status','approved')=='approved'][:args.limit]
    device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu'); model=model_from_checkpoint(ck,pretrained=False).to(device); model.load_state_dict(ck['model_state']); model.eval(); ort_s=ort.InferenceSession(str(args.onnx),providers=['CUDAExecutionProvider','CPUExecutionProvider']); name=ort_s.get_inputs()[0].name
    ptb=[]; ptc=[]; orb=[]; orc=[]; max_abs=0.;
    with torch.no_grad():
        for start in range(0,len(rows),64):
            chunk=rows[start:start+64]; batch=[]; raw=[]
            for r in chunk:
                with Image.open(args.manifest.parent/r['image_path']) as im:
                    rgb=im.convert('RGB'); batch.append(tf(rgb)); arr=np.asarray(rgb.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.0; raw.append(np.transpose(arr,(2,0,1)))
            x=torch.stack(batch); a,b=model(x.to(device)); a=a.cpu().numpy()/float(cal['body_type']['temperature']); b=b.cpu().numpy()/float(cal['color']['temperature']); oa,ob=ort_s.run(None,{name:np.asarray(raw,dtype=np.float32)})
            max_abs=max(max_abs,float(np.max(np.abs(a-oa))),float(np.max(np.abs(b-ob)))); ptb.extend(a.argmax(1).tolist()); ptc.extend(b.argmax(1).tolist()); orb.extend(np.asarray(oa).argmax(1).tolist()); orc.extend(np.asarray(ob).argmax(1).tolist())
    body_match=sum(x==y for x,y in zip(ptb,orb)); color_match=sum(x==y for x,y in zip(ptc,orc)); report={'schema_version':'1.0','split':args.split,'samples':len(rows),'body_top1_match_rate':body_match/max(1,len(rows)),'color_top1_match_rate':color_match/max(1,len(rows)),'max_abs_logit_delta':max_abs,'gate':body_match/max(1,len(rows))>=.995 and color_match/max(1,len(rows))>=.995}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
