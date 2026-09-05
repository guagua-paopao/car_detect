#!/usr/bin/env python3
"""Append an independently split, licensed VCoR color set without leakage."""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from collections import Counter
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-manifest',type=Path,required=True); ap.add_argument('--vcor-manifest',type=Path,required=True); ap.add_argument('--output-manifest',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); args=ap.parse_args()
    with args.base_manifest.open('r',encoding='utf-8-sig',newline='') as f: base=list(csv.DictReader(f))
    with args.vcor_manifest.open('r',encoding='utf-8-sig',newline='') as f: src=list(csv.DictReader(f))
    fields=list(base[0]); seen_paths={r.get('image_path','') for r in base}; seen_sha={r.get('sha256','') for r in base if r.get('sha256')}
    added=[]; skipped=Counter()
    for r in src:
        source=(args.vcor_manifest.parent/r['image_path']).resolve()
        if not source.exists(): skipped['missing_image']+=1; continue
        digest=r.get('sha256','') or sha(source)
        rel=os.path.relpath(source,args.output_manifest.parent.resolve()).replace('\\','/')
        if rel in seen_paths or digest in seen_sha: skipped['duplicate']+=1; continue
        n={k:r.get(k,'') for k in fields}; n['image_path']=rel; n['source_dataset']='VCoR-LARGE'; n['source_group']='VCoR-LARGE'; n['source_license']=r.get('source_license') or 'research-and-education-only; original-authors'; n['annotation_source']='official_dataset'; n['body_type']='unknown'; n['body_type_supervised']='false'; n['color_supervised']='true'; n['review_status']='approved'; n['review_method']='official_vcor_color_manifest+sha_dedup'; n['formal_train_eligible']='true'; n['license_train_eligible']='true'; n['sha256']=digest; added.append(n); seen_paths.add(rel); seen_sha.add(digest)
    merged=base+added; args.output_manifest.parent.mkdir(parents=True,exist_ok=True)
    with args.output_manifest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(merged)
    report={'schema_version':'vcor-color-expansion-manifest-v27','base_rows':len(base),'source_rows':len(src),'added_rows':len(added),'rows_after':len(merged),'added_split_counts':dict(sorted(Counter(r['split'] for r in added).items())),'added_color_counts':dict(sorted(Counter(r['color'] for r in added).items())),'skipped':dict(sorted(skipped.items())),'source_license':'research-and-education-only; original-authors','train_policy':'only rows split=train are used for training; validation/test remain held out','frozen_video_used':False,'output_manifest':str(args.output_manifest),'output_manifest_sha256':sha(args.output_manifest)}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
