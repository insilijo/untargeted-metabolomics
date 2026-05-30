import json, time, csv
t=time.time()
print("loading GNPS-LIBRARY.json (530MB)...", flush=True)
recs=json.load(open("data/HOC/GNPS-LIBRARY.json"))
print(f"  {len(recs)} records in {time.time()-t:.1f}s", flush=True)
ik14=lambda s:(s or "")[:14]
neg={}; npos=nneg=0
for r in recs:
    mode=(r.get("Ion_Mode") or "").lower()
    if "neg" not in mode: 
        npos+=1; continue
    nneg+=1
    ik=r.get("InChIKey_smiles") or r.get("InChIKey_inchi") or ""
    if not ik or ik=="N/A": continue
    k=ik14(ik)
    try: peaks=json.loads(r["peaks_json"])
    except: continue
    if len(peaks)<3: continue
    # keep the spectrum with most peaks per ik14
    if k not in neg or len(peaks)>len(neg[k]):
        neg[k]=[[float(m),float(i)] for m,i in peaks]
json.dump(neg, open("data/st004581_work/gnps_neg_ms2_ik14.json","w"))
print(f"neg records: {nneg}  pos/other: {npos}  -> {len(neg)} unique neg ik14 spectra", flush=True)
# coverage of Metabolon neg
maf=[r for r in csv.DictReader(open("/home/jgardner/squid_results/st004581/annotations_repaired.csv")) if r['platform']=='lc/ms neg' and r.get('unannotatable','')!='true']
cov=sum(1 for g in maf if ik14(g['inchikey']) in neg)
print(f"Metabolon neg GT with GNPS-neg experimental spectrum: {cov}/{len(maf)}", flush=True)
