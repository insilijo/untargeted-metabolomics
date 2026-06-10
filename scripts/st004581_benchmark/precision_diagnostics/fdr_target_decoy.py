"""Target-decoy FDR for library matching. Decoy = DD with m/z shifted +0.5 Da (same RI/
density, no true matches) -> decoy hits estimate the coincidental-match (false) rate.
Per-compound best score -> q-value -> at FDR<5%, how many compounds called & precision."""
import csv, numpy as np
from collections import defaultdict
norm=lambda s:"".join(ch for ch in (s or "").lower() if ch.isalnum())
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
maf=defaultdict(set)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    p=(r.get("platform") or "").strip().lower(); nm=norm(r.get("name") or "")
    if nm: maf[p].add(nm)
total_maf=sum(len(v) for v in maf.values())
def best(path):
    d={}
    for r in csv.DictReader(open(path)):
        mid=(r.get("match_id") or "").strip()
        if not mid: continue
        try: s=float(r["score"])
        except: continue
        k=((r.get("platform") or "").strip().lower(),mid)
        if k not in d or s<d[k]: d[k]=s
    return d
tgt=best("/tmp/tgt.csv"); dec=best("/tmp/dec.csv")
items=sorted(tgt.items(), key=lambda x:x[1])      # best (low) score first
tscore=np.array([s for _,s in items]); dscore=np.sort(list(dec.values()))
istp=np.array([1 if k[1] in maf.get(k[0],set()) else 0 for k,_ in items])
ndec=np.searchsorted(dscore,tscore,side="right").astype(float)
ntgt=np.arange(1,len(items)+1).astype(float)
fdr=ndec/ntgt
q=np.minimum.accumulate(fdr[::-1])[::-1]
print(f"target compounds called {len(items)}  decoy hits {len(dscore)}  MAF {total_maf}\n")
print(f"{'FDR<':>6}{'called':>8}{'TP':>6}{'precision':>11}{'recall':>9}{'%ofDD(~3214)':>13}")
for lvl in [0.01,0.05,0.10,0.20,0.50,1.01]:
    keep=q<lvl; c=keep.sum(); tp=int(istp[keep].sum())
    P=tp/c if c else 0; R=tp/total_maf
    print(f"{lvl:>6.2f}{c:>8}{tp:>6}{P:>11.3f}{R:>9.3f}{c/3214*100:>12.0f}%")
print("\n(Metabolon calls ~26% of the DD present; precision there is the closed-world number)")
