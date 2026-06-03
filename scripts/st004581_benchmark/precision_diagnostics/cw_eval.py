import csv,sys
from collections import defaultdict
norm=lambda s:"".join(ch for ch in (s or "").lower() if ch.isalnum())
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
maf=defaultdict(set)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    p=(r.get("platform") or "").strip().lower(); nm=norm(r.get("name") or "")
    if nm: maf[p].add(nm)
ann=defaultdict(set)
for r in csv.DictReader(open(sys.argv[1])):
    mid=(r.get("match_id") or "").strip(); p=(r.get("platform") or "").strip().lower()
    if mid: ann[p].add(mid)
called=sum(len(v) for v in ann.values()); tp=sum(len(ann[p]&maf[p]) for p in ann)
tm=sum(len(v) for v in maf.values())
P=tp/called if called else 0; R=tp/tm; F1=2*P*R/(P+R) if P+R else 0
print(f"  closed-world (presence): called {called}  TP {tp}  precision {P:.3f}  recall {R:.3f}  F1 {F1:.3f}")
