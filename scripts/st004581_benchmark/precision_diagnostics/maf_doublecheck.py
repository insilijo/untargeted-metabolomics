"""Double-check the MAF matching: are some 'FP' DD compounds actually in the MAF under a
different name? Compare name-keying vs InChIKey-keying. If ik-match > name-match, the
name-key is losing real TPs."""
import sys, csv
from pathlib import Path
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
PLAT="lc/ms neg"; ik14=lambda s:(s or "").strip()[:14]
# MAF neg entries: name, ik, mz
maf_name=set(); maf_ik=set(); maf_rows=[]
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()!=PLAT: continue
    nm=M._norm(r.get("name") or ""); ik=ik14(r.get("inchikey"))
    maf_name.add(nm)
    if ik and len(ik)>=14: maf_ik.add(ik)
    maf_rows.append((nm,ik,r.get("name")))
print(f"MAF neg entries: {len(maf_rows)}  (distinct norm-names {len(maf_name)}, with ik14 {len(maf_ik)})")
# DD neg compounds
lib0=M.load_library(Path(DD))
dd=[(M._norm(c['name']), (c.get('ik14') or '').strip(), c['name']) for c in lib0.get(PLAT,[]) if M._norm(c['name'])]
name_hit=sum(1 for nm,ik,raw in dd if nm in maf_name)
ik_hit=sum(1 for nm,ik,raw in dd if ik and len(ik)>=14 and ik in maf_ik)
# DD compounds that match MAF by ik but NOT by name = miskeyed TPs
miskey=[(nm,ik,raw) for nm,ik,raw in dd if ik and len(ik)>=14 and ik in maf_ik and nm not in maf_name]
print(f"\nDD neg compounds: {len(dd)}")
print(f"  match MAF by NORM-NAME : {name_hit}")
print(f"  match MAF by INCHIKEY  : {ik_hit}")
print(f"  match by ik but NOT by name (miskeyed!): {len(miskey)}")
for nm,ik,raw in miskey[:15]: print(f"      ik={ik}  ddname='{raw}'  (norm='{nm}')")
# reverse: MAF entries whose name isn't in DD name set (so a DD synonym would be FP+FN)
dd_name=set(nm for nm,_,_ in dd); dd_ik=set(ik for _,ik,_ in dd if ik and len(ik)>=14)
maf_noname=[(nm,ik,raw) for nm,ik,raw in maf_rows if nm not in dd_name]
maf_noname_butik=[x for x in maf_noname if x[1] and len(x[1])>=14 and x[1] in dd_ik]
print(f"\nMAF entries whose NORM-NAME is absent from DD: {len(maf_noname)}")
print(f"  ...but whose INCHIKEY *is* in DD (name-key would miss): {len(maf_noname_butik)}")
for nm,ik,raw in maf_noname_butik[:15]: print(f"      ik={ik}  mafname='{raw}'  (norm='{nm}')")
