"""What ARE the 84 no-SMILES MAF neg compounds? Categorize: have InChIKey (lookup-able) /
lipid shorthand (parse-able to structure) / unknown 'X-#####' (no structure) / other."""
import sys, csv, re
from pathlib import Path
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
PLAT="lc/ms neg"; ik14=lambda s:(s or "")[:14]
maf=set()
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT: maf.add(M._norm(r.get("name") or ""))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# raw DD rows to see inchikey presence
rows={}
for r in csv.DictReader(open(DD)):
    nm=M._norm(r.get("biochemical") or r.get("name") or r.get("BIOCHEMICAL") or "")
    if nm: rows[nm]=r
lib0=M.load_library(Path(DD))
cats=dict(have_ik=0,lipid=0,unknown_X=0,plain_named=0); ex={k:[] for k in cats}
lipid_re=re.compile(r'\b(PC|PE|PS|PI|PG|PA|LPC|LPE|TAG|DAG|MAG|CE|SM|Cer|FA|DG|TG|GPC|GPE|sphingo|ceramide|acyl|oate|enoate|dienoate)\b',re.I)
n=0
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if nm not in maf: continue
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    if d is not None: continue   # has smiles
    n+=1
    raw=rows.get(nm,{}); ik=(raw.get("inchikey") or raw.get("InChIKey") or raw.get("INCHIKEY") or "").strip()
    if ik and len(ik)>=14: cats["have_ik"]+=1; (ex["have_ik"].append(nm) if len(ex["have_ik"])<6 else None)
    elif re.match(r'^[xX]-\s*\d',nm) or nm.startswith("x-"): cats["unknown_X"]+=1; (ex["unknown_X"].append(nm) if len(ex["unknown_X"])<6 else None)
    elif lipid_re.search(nm): cats["lipid"]+=1; (ex["lipid"].append(nm) if len(ex["lipid"])<6 else None)
    else: cats["plain_named"]+=1; (ex["plain_named"].append(nm) if len(ex["plain_named"])<8 else None)
print(f"84-class no-SMILES MAF neg compounds: {n}")
for k in cats: print(f"  {k:<14}{cats[k]:>4}   e.g. {ex[k][:5]}")
