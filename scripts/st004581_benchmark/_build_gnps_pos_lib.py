import json, csv
recs=json.load(open("data/HOC/GNPS-LIBRARY.json"))
ik14=lambda s:(s or "")[:14]
pos={}
for r in recs:
    if "pos" not in (r.get("Ion_Mode") or "").lower(): continue
    ik=r.get("InChIKey_smiles") or r.get("InChIKey_inchi") or ""
    if not ik or ik=="N/A": continue
    k=ik14(ik)
    try: pk=json.loads(r["peaks_json"])
    except: continue
    if len(pk)<3: continue
    if k not in pos or len(pk)>len(pos[k]): pos[k]=[[float(m),float(i)] for m,i in pk]
json.dump(pos, open("data/st004581_work/gnps_pos_ms2_ik14.json","w"))
print(f"{len(pos)} unique pos ik14 spectra")
