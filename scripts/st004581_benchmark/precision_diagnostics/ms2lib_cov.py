import sys, csv, json
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; P="/mnt/volume-hel1-1/data/processed/"
ik14=lambda s:(s or "")[:14]
# load neg libraries, index by ik14
libs={}
for nm,fn in [("massbank","ms2_library_massbank_full_neg.json"),("mona","ms2_library_mona_neg.json"),("gnps","ms2_library_gnps_neg.json")]:
    d=json.load(open(P+fn)); libs[nm]={ik14(k):v for k,v in d.items()}
    print(f"{nm} neg: {len(d)} full-ik -> {len(libs[nm])} ik14")
allik14=set().union(*[set(l) for l in libs.values()])
print(f"combined neg ik14: {len(allik14)}")
# MAF compounds per platform
from collections import defaultdict
plat_iks=defaultdict(set)
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    p=(r.get("platform") or "").strip().lower(); ik=ik14(r.get("inchikey"))
    if len(ik)>=14: plat_iks[p].add(ik)
print("\nNEG-library coverage of MAF compounds (by ik14):")
for p,iks in sorted(plat_iks.items()):
    cov=len(iks&allik14)
    print(f"  {p:<16} {len(iks):>4} compounds -> {cov:>4} have a neg reference spectrum ({cov/len(iks):.2f})")
neg=plat_iks["lc/ms neg"]
for nm in libs: print(f"    neg MAF in {nm}: {len(neg&set(libs[nm]))}")
