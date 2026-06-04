"""Decompose the eic_annotate FP calls: is the confident peak (a) at coords where a MAF
compound exists (substitution -- real present compound, wrong isomer ID) or (b) where NO
MAF compound is (novel -- a real peak Metabolon didn't claim -> MAF-incompleteness, not
clearly our error)? Bounds how understated the closed-world precision is."""
import csv
from collections import defaultdict
import numpy as np
norm=lambda s:"".join(c for c in (s or "").lower() if c.isalnum())
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
SLOPE={"lc/ms neg":18.0,"lc/ms pos early":17.0,"lc/ms pos late":27.0,"lc/ms polar":13.0}
maf=defaultdict(list); mafnames=defaultdict(set)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    p=(r.get("platform") or "").strip().lower()
    try: mz=float(r["mz"]); ri=float(r["rt"])
    except: continue
    nm=norm(r.get("name") or "")
    if nm: maf[p].append((mz,ri)); mafnames[p].add(nm)
called=[r for r in csv.DictReader(open("/tmp/eic_all.csv")) if float(r["qvalue"])<0.05 and int(r["present"])>=2 and int(r["primary"])]
tp=sub=nov=0
for r in called:
    p=r["platform"]; nm=norm(r["name"]); mz=float(r["mz"]); ri=float(r["ri"])
    if nm in mafnames.get(p,set()): tp+=1; continue
    # FP: is there a MAF compound at same m/z (7ppm) + RI (within 5s)?
    s=SLOPE.get(p,18.0); near=any(abs(m-mz)<=mz*7e-6 and abs(rr-ri)<=5*s for m,rr in maf.get(p,[]))
    if near: sub+=1
    else: nov+=1
c=len(called)
print(f"called (FDR<5%, primary): {c}")
print(f"  TP (in MAF)                         : {tp} ({tp/c*100:.0f}%)")
print(f"  FP-substitution (MAF cmpd at coords): {sub} ({sub/c*100:.0f}%)  <- real present peak, wrong isomer ID")
print(f"  FP-novel (NO MAF cmpd at coords)    : {nov} ({nov/c*100:.0f}%)  <- real peak, Metabolon claimed nothing")
print(f"\nclosed-world precision (MAF=truth)       : {tp/c:.3f}")
print(f"precision if novels aren't counted FP    : {tp/(tp+sub):.3f}   (treats MAF-incompleteness as not-our-error)")
