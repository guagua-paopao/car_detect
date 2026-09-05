#!/usr/bin/env python3
"""Check whether a manifest contains labeled multi-frame vehicle tracks."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--min-frames',type=int,default=3); a=ap.parse_args()
 with a.manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 eligible=[r for r in rows if r.get('color_supervised','true').lower() not in {'false','0','no'} and r.get('color','').strip() not in {'','unknown'} and r.get('review_status','approved')=='approved']
 groups=defaultdict(list)
 for r in eligible:
  key=(r.get('video_id',''),r.get('track_group',''))
  groups[key].append(r)
 counts=Counter(len(v) for v in groups.values()); valid={k:v for k,v in groups.items() if len(v)>=a.min_frames}
 out={'schema_version':'labeled-continuous-manifest-audit-v1','manifest':str(a.manifest),'rows':len(rows),'eligible_labeled_color_rows':len(eligible),'groups':len(groups),'group_size_histogram':{str(k):v for k,v in sorted(counts.items())},'min_frames':a.min_frames,'eligible_groups':len(valid),'eligible_rows':sum(len(v) for v in valid.values()),'by_split':Counter(r.get('split') for v in valid.values() for r in v),'by_color':Counter(r.get('color') for v in valid.values() for r in v),'status':'pass' if valid else 'missing_labeled_continuous_evidence','policy':'test rows are evidence only; no threshold selection; frozen video excluded','frozen_video_used':False}
 out['by_split']=dict(out['by_split']); out['by_color']=dict(out['by_color']); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:out[k] for k in ('manifest','rows','eligible_labeled_color_rows','groups','eligible_groups','eligible_rows','status')},ensure_ascii=False))
if __name__=='__main__': main()
