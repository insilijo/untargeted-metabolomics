"""Build bootstrappable MS2 reference libraries for the public Metabolon DD + HMDB -- FROM SCRATCH.

Assumes NOTHING is on the machine: downloads every public source, then attaches the spectra to the
public compound lists so the kit-free bootstrap can confirm a mass-unique MS1 peak by reference-MS2
match >= TAU and use its observed RT as a calibration anchor. Library-MATCHING with self-bootstrapping
calibration, NOT de novo.

  python build_references.py all          # download everything + build (one shot)
  python build_references.py download     # just fetch sources into ./downloads
  python build_references.py fast|mona|hmdb|dd|emit   # individual build stages

Sources (auto-downloaded into ./downloads unless already present):
  GNPS-LIBRARY.json            external.gnps2.org           experimental MS2   (CC0)
  HMDB.json (GNPS HMDB lib)    external.gnps2.org           experimental MS2   (CC0)
  MassBank_NISTformat.msp      github MassBank-data latest  experimental MS2   (CC BY)
  MoNA-export-LipidBlast.json  mona.fiehnlab.ucdavis.edu    in-silico lipid    (CC BY)
  PubChem PUG REST             pubchem.ncbi.nlm.nih.gov     DD SMILES backfill (public)
HMDB compound universe is derived from the GNPS HMDB library (HMDB.json above) -- NOT the licence-gated
hmdb.ca XML -- so there is no hands-on / academic-gated download step.
Bundled (no clean public URL): inputs/metabolon_data_dictionary_PMC_OA_subset.csv (PMC-OA subset).

LICENCE: keep PRIVATE / INTERNAL. The compound identities trace to HMDB (redistributed via GNPS),
so do not surface this bundle publicly or fold it into the public Sextant repo without checking.
"""
import sys, os, csv, json, re, time, subprocess, urllib.request, urllib.parse, zipfile
csv.field_size_limit(10**7)

BASE = os.path.dirname(os.path.abspath(__file__))
DL   = os.path.join(BASE, "downloads"); OUT = os.path.join(BASE, "out"); INP = os.path.join(BASE, "inputs")
DD_CSV = os.path.join(INP, "metabolon_data_dictionary_PMC_OA_subset.csv")
MAF    = "/home/jgardner/squid_results/st004581/annotations_repaired.csv"   # optional: coverage report only

GNPS_LIBRARY = "https://external.gnps2.org/gnpslibrary/GNPS-LIBRARY.json"
GNPS_HMDB    = "https://external.gnps2.org/gnpslibrary/HMDB.json"
MASSBANK_REL = "https://api.github.com/repos/MassBank/MassBank-data/releases/latest"
MONA_PREDEF  = "https://mona.fiehnlab.ucdavis.edu/rest/downloads/predefined"
MONA_RETRIEVE= "https://mona.fiehnlab.ucdavis.edu/rest/downloads/retrieve/"
UA           = "Mozilla/5.0 (build_references)"

ik14   = lambda s: (s or "")[:14]
_norm  = lambda s: "".join(ch for ch in (s or "").lower() if ch.isalnum())
isfull = lambda s: bool(s) and len(s) >= 25 and s.count("-") >= 2
def polarity(s):
    s=(s or "").lower()
    return "neg" if "neg" in s else ("pos" if "pos" in s else "")
def cap_peaks(pk, n=200):
    out=[]
    for pr in pk:
        try:
            m,i=float(pr[0]),float(pr[1])
            if i>0: out.append([round(m,5),round(i,2)])
        except Exception: continue
    out.sort(key=lambda x:-x[1]); return out[:n]
def _f(x):
    try: return round(float(x),5)
    except Exception: return None

# ----------------------------------------------------------------------------- download helpers
def curl(url, dest, extra=()):
    if os.path.exists(dest) and os.path.getsize(dest)>1024:
        print(f"  have {os.path.basename(dest)} ({os.path.getsize(dest)//1024//1024}MB) -- skip"); return True
    print(f"  downloading {url} -> {os.path.basename(dest)}",flush=True)
    r=subprocess.run(["curl","-sL","--fail","-A",UA,*extra,"-o",dest,url])
    ok=r.returncode==0 and os.path.exists(dest) and os.path.getsize(dest)>1024
    print(f"    {'ok' if ok else 'FAILED'} ({os.path.getsize(dest)//1024//1024 if os.path.exists(dest) else 0}MB)")
    return ok
def http_json(url):
    req=urllib.request.Request(url, headers={"User-Agent":UA})
    return json.load(urllib.request.urlopen(req, timeout=60))

def stage_download():
    os.makedirs(DL, exist_ok=True)
    curl(GNPS_LIBRARY, DL+"/GNPS-LIBRARY.json")
    curl(GNPS_HMDB,    DL+"/HMDB.json")
    # MassBank: latest release -> NIST-format MSP asset
    try:
        rel=http_json(MASSBANK_REL); msp=next(a["browser_download_url"] for a in rel["assets"]
            if a["name"]=="MassBank_NISTformat.msp")
        curl(msp, DL+"/MassBank_NIST.msp")
    except Exception as e: print("  MassBank lookup failed:",e)
    # MoNA LipidBlast: predefined -> id -> retrieve zip -> unzip
    try:
        if not os.path.exists(DL+"/MoNA-export-LipidBlast.json"):
            pre=http_json(MONA_PREDEF)
            ent=next(x for x in pre if x.get("label")=="Libraries - LipidBlast")
            mid=ent["jsonExport"]["id"]; z=DL+"/MoNA-LipidBlast.zip"
            curl(MONA_RETRIEVE+mid, z)
            with zipfile.ZipFile(z) as zf:
                nm=[n for n in zf.namelist() if n.endswith(".json")][0]
                zf.extract(nm, DL); os.replace(DL+"/"+nm, DL+"/MoNA-export-LipidBlast.json")
            print("    unzipped MoNA ->", os.path.getsize(DL+"/MoNA-export-LipidBlast.json")//1024//1024,"MB")
        else: print("  have MoNA-export-LipidBlast.json -- skip")
    except Exception as e: print("  MoNA download failed:",e)
    # HMDB compounds come from the GNPS HMDB library (HMDB.json, already fetched above) -- no
    # licence-gated hmdb.ca XML needed. See stage_hmdb.

# ----------------------------------------------------------------------------- spectral index
_rk=None
def ik_from(smiles, inchi):
    global _rk
    if _rk is None:
        try:
            from rdkit import Chem; from rdkit.Chem import inchi as _ic
            from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*"); _rk=(Chem,_ic)
        except Exception: _rk=False
    if not _rk: return ""
    Chem,_ic=_rk
    try:
        if inchi and inchi.startswith("InChI="): return _ic.InchiToInchiKey(inchi) or ""
        if smiles:
            m=Chem.MolFromSmiles(smiles)
            if m is not None: return _ic.InchiToInchiKey(_ic.MolToInchi(m)) or ""
    except Exception: pass
    return ""
def emit_row(fh, ik, src, kind, mode, adduct, prec, peaks, name="", smiles=""):
    if not isfull(ik) or not peaks or len(peaks)<3: return 0
    fh.write(json.dumps({"ik":ik,"src":src,"kind":kind,"mode":mode,"adduct":adduct or "",
        "prec":_f(prec),"name":name or "","smiles":smiles or "","peaks":cap_peaks(peaks)})+"\n"); return 1

def stage_fast():
    os.makedirs(OUT, exist_ok=True); out=open(OUT+"/idx_fast.jsonl","w"); n=0
    for fn,src in [("GNPS-LIBRARY.json","gnps"),("HMDB.json","hmdb_gnps")]:
        p=DL+"/"+fn
        if not os.path.exists(p): print("missing",fn,"-- run download"); continue
        t=time.time(); recs=json.load(open(p)); print(f"{fn}: {len(recs)} recs {time.time()-t:.0f}s",flush=True)
        for r in recs:
            ik=(r.get("InChIKey_smiles") or r.get("InChIKey_inchi") or r.get("INCHIKEY") or "").strip()
            sm=r.get("Smiles") or r.get("SMILES") or ""; inchi=r.get("INCHI") or ""
            if not isfull(ik): ik=ik_from(sm,inchi) or ik
            try: pk=json.loads(r["peaks_json"])
            except Exception: continue
            n+=emit_row(out,ik,src,"experimental",polarity(r.get("Ion_Mode") or r.get("Ion_mode")),
                        r.get("Adduct"),r.get("Precursor_MZ"),pk,r.get("Compound_Name"),sm)
    m=DL+"/MassBank_NIST.msp"
    if os.path.exists(m): n+=parse_msp(m,out)
    else: print("missing MassBank MSP -- run download")
    out.close(); print(f"fast index rows: {n}",flush=True)

def parse_msp(path,out):
    n=0; meta={}; peaks=[]; inpk=False
    def flush():
        nonlocal n
        ik=meta.get("InChIKey","")
        if not isfull(ik): ik=ik_from(meta.get("SMILES",""),meta.get("InChI","")) or ik
        n+=emit_row(out,ik,"massbank","experimental",polarity(meta.get("Ion_mode")),
                    meta.get("Precursor_type"),meta.get("PrecursorMZ"),peaks,meta.get("Name"),meta.get("SMILES"))
    for line in open(path,encoding="utf-8",errors="replace"):
        line=line.rstrip("\n")
        if not line:
            if meta or peaks: flush()
            meta={};peaks=[];inpk=False; continue
        if inpk:
            sp=line.split()
            if len(sp)>=2:
                try: peaks.append([float(sp[0]),float(sp[1])])
                except Exception: pass
        elif ":" in line:
            k,v=line.split(":",1); k=k.strip()
            if k=="Num Peaks": inpk=True
            else: meta[k]=v.strip()
    if meta or peaks: flush()
    print(f"MassBank MSP rows: {n}",flush=True); return n

def stream_array(path,bufsize=8<<20):
    dec=json.JSONDecoder()
    with open(path) as f:
        buf=f.read(bufsize); i=buf.find("["); buf=buf[i+1:]
        while True:
            buf=buf.lstrip().lstrip(",").lstrip()
            if buf[:1]=="]": return
            if not buf:
                ch=f.read(bufsize)
                if not ch: return
                buf+=ch; continue
            try:
                obj,end=dec.raw_decode(buf); yield obj; buf=buf[end:]
            except json.JSONDecodeError:
                ch=f.read(bufsize)
                if not ch: return
                buf+=ch

def stage_mona():
    p=DL+"/MoNA-export-LipidBlast.json"
    if not os.path.exists(p): print("missing MoNA json -- run download"); return
    os.makedirs(OUT, exist_ok=True); out=open(OUT+"/idx_mona.jsonl","w"); n=seen=0; t=time.time()
    for rec in stream_array(p):
        seen+=1
        if seen%200000==0: print(f"  MoNA {seen} recs, {n} kept, {time.time()-t:.0f}s",flush=True)
        comp=(rec.get("compound") or [{}])[0]
        cmd={m.get("name"):m.get("value") for m in comp.get("metaData",[])}
        ik=cmd.get("InChIKey") or ""
        if not isfull(ik): ik=ik_from(cmd.get("SMILES"),cmd.get("InChI")) or ik
        pk=[]
        for tok in (rec.get("spectrum") or "").split():
            if ":" in tok:
                a,b=tok.split(":",1)
                try: pk.append([float(a),float(b)])
                except Exception: pass
        smd={m.get("name"):m.get("value") for m in rec.get("metaData",[])}
        n+=emit_row(out,ik,"mona_lipidblast","in_silico",polarity(smd.get("ionization mode")),
                    smd.get("precursor type"),smd.get("precursor m/z"),pk,
                    (comp.get("names") or [{}])[0].get("name"),cmd.get("SMILES"))
    out.close(); print(f"MoNA: scanned {seen}, kept {n}",flush=True)

# ----------------------------------------------------------------------------- corrected DD (+PubChem SMILES)
def pubchem_smiles(cids):
    """Batch CID -> SMILES via PubChem PUG REST (downloads; no local structure file needed)."""
    out={}; cids=[c for c in cids if c]; B=180
    for i in range(0,len(cids),B):
        chunk=cids[i:i+B]
        data=urllib.parse.urlencode({"cid":",".join(chunk)}).encode()
        url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/ConnectivitySMILES/JSON"
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,data=data,headers={"User-Agent":UA})
                d=json.load(urllib.request.urlopen(req,timeout=60))
                for p in d["PropertyTable"]["Properties"]:
                    s=p.get("ConnectivitySMILES") or p.get("SMILES") or p.get("CanonicalSMILES")
                    if s: out[str(p["CID"])]=s
                break
            except Exception:
                time.sleep(1.0*(attempt+1))
        time.sleep(0.25)   # PubChem rate limit (<=5 req/s)
        if (i//B)%10==0: print(f"  PubChem {i+len(chunk)}/{len(cids)} CIDs, {len(out)} smiles",flush=True)
    return out

def stage_dd():
    os.makedirs(OUT, exist_ok=True)
    name2ik={}
    if os.path.exists(MAF):
        for r in csv.DictReader(open(MAF)):
            if r.get("unannotatable","")!="true" and (r.get("inchikey") or "").strip():
                name2ik.setdefault(_norm(r.get("name")),r["inchikey"].strip())
    rows=list(csv.DictReader(open(DD_CSV,encoding="utf-8-sig")))
    cids=sorted({(r.get("PUBCHEM") or "").split(";")[0].split(",")[0].strip() for r in rows} - {""})
    print(f"DD {len(rows)} rows; fetching SMILES for {len(cids)} PubChem CIDs...",flush=True)
    cid2smi=pubchem_smiles(cids)
    w=csv.writer(open(OUT+"/metabolon_dd_corrected.csv","w"))
    w.writerow(["platform","name","inchikey","smiles","mass","ri","pubchem","ik_source"]); nbf=nsm=0
    for r in rows:
        nm=r.get("BIOCHEMICAL") or ""; ik=(r.get("INCHIKEY") or "").strip(); src="dd"
        if not isfull(ik):
            bf=name2ik.get(_norm(nm),"")
            if isfull(bf): ik=bf; src="maf_backfill"; nbf+=1
        cid=(r.get("PUBCHEM") or "").split(";")[0].split(",")[0].strip(); sm=cid2smi.get(cid,"")
        if sm: nsm+=1
        w.writerow([r.get("PLATFORM",""),nm,ik,sm,r.get("MASS",""),r.get("RI",""),cid,src])
    print(f"corrected DD: {len(rows)} rows (ik backfilled {nbf}, smiles {nsm}/{len(rows)})",flush=True)

# ----------------------------------------------------------------------------- HMDB compounds
def stage_hmdb():
    """HMDB compound universe FROM the GNPS HMDB library (external.gnps2.org/gnpslibrary/HMDB.json),
    NOT the licence-gated hmdb.ca XML. This is the HMDB subset that actually has reference MS2 --
    the right scope for an MS2 library, and it removes the only hands-on / academic-gated step."""
    p=DL+"/HMDB.json"
    if not os.path.exists(p): print("missing GNPS HMDB.json -- run download; SKIP"); return
    os.makedirs(OUT, exist_ok=True)
    recs=json.load(open(p)); seen={}
    for r in recs:
        ik=(r.get("InChIKey_smiles") or r.get("InChIKey_inchi") or r.get("INCHIKEY") or "").strip()
        sm=r.get("Smiles") or r.get("SMILES") or ""; inchi=r.get("INCHI") or ""
        if not isfull(ik): ik=ik_from(sm,inchi) or ik
        if isfull(ik) and ik not in seen: seen[ik]=(r.get("Compound_Name") or "", sm)
    w=csv.writer(open(OUT+"/hmdb_compounds.csv","w"))
    w.writerow(["accession","name","inchikey","smiles","formula","mono_mass"])
    for ik,(nm,sm) in seen.items(): w.writerow(["",nm,ik,sm,"",""])
    print(f"HMDB compounds (from GNPS HMDB library): {len(seen)}",flush=True)

# ----------------------------------------------------------------------------- emit
def stage_emit():
    dd={}
    for r in csv.DictReader(open(OUT+"/metabolon_dd_corrected.csv")):
        if isfull(r["inchikey"]): dd[r["inchikey"]]=(r["name"],r["smiles"])
    hmdb={}; hp=OUT+"/hmdb_compounds.csv"
    if os.path.exists(hp):
        for r in csv.DictReader(open(hp)):
            if isfull(r["inchikey"]): hmdb[r["inchikey"]]=(r["name"],r["smiles"])
    print(f"universes: DD {len(dd)} ik, HMDB {len(hmdb)} ik",flush=True)
    ddk=set(dd); hmk=set(hmdb)
    byik={}; ref={"neg":{},"pos":{}}; counts={}
    for jf in ["idx_fast.jsonl","idx_mona.jsonl"]:
        p=OUT+"/"+jf
        if not os.path.exists(p): continue
        for line in open(p):
            s=json.loads(line); ik=s["ik"]; counts[s["src"]]=counts.get(s["src"],0)+1
            if ik and s["mode"] in ref:
                k=ik14(ik); rank=(0 if s["kind"]=="experimental" else -1,len(s["peaks"])); cur=ref[s["mode"]].get(k)
                if cur is None or rank>cur[0]: ref[s["mode"]][k]=(rank,s["peaks"])
            if ik in ddk or ik in hmk:
                d=byik.setdefault(ik,[])
                if sum(1 for x in d if x["src"]==s["src"] and x["mode"]==s["mode"])<3: d.append(s)
    print("index spectra by source:",counts,flush=True)
    def write_mgf(path,univ):
        nb=nc=0
        with open(path,"w") as f:
            for ik,(name,smiles) in univ.items():
                specs=byik.get(ik,[])
                if not specs: continue
                nc+=1
                for sp in specs:
                    f.write("BEGIN IONS\n"); f.write(f"TITLE={name} | {ik} | {sp['src']}:{sp['kind']}\n")
                    f.write(f"INCHIKEY={ik}\n")
                    if smiles: f.write(f"SMILES={smiles}\n")
                    if sp["prec"]: f.write(f"PEPMASS={sp['prec']}\n")
                    if sp["mode"]: f.write(f"IONMODE={'Positive' if sp['mode']=='pos' else 'Negative'}\n")
                    if sp["adduct"]: f.write(f"ADDUCT={sp['adduct']}\n")
                    f.write(f"LIBRARY={sp['src']}\nSPECTRUMKIND={sp['kind']}\n")
                    for m,i in sp["peaks"]: f.write(f"{m} {i}\n")
                    f.write("END IONS\n\n"); nb+=1
        print(f"  {os.path.basename(path)}: {nc} compounds, {nb} spectra",flush=True); return nc,nb
    md=write_mgf(OUT+"/metabolon_dd_ms2.mgf",dd); mh=write_mgf(OUT+"/hmdb_ms2.mgf",hmdb)
    for mode in ("neg","pos"):
        json.dump({k:v[1] for k,v in ref[mode].items()}, open(OUT+f"/ms2_ref_{mode}_ik14.json","w"))
        print(f"  ms2_ref_{mode}_ik14.json: {len(ref[mode])} ik14",flush=True)
    open(OUT+"/README.md","w").write(README.format(sources=json.dumps(counts,indent=2),
        dd_n=len(dd),hmdb_n=len(hmdb),dd_mgf=md[1],hmdb_mgf=mh[1],
        ref_neg=len(ref["neg"]),ref_pos=len(ref["pos"])))
    print("wrote README",flush=True)

README="""# MS2 reference libraries for public Metabolon DD + HMDB (PRIVATE / INTERNAL)

Built from scratch by build_references.py. **Do not redistribute** -- HMDB is academic-only and taints
this bundle. Purpose: make the kit-free bootstrap work (mass-unique MS1 + reference-MS2 >= TAU -> anchor).

## Spectra by source
{sources}

## Outputs
- metabolon_dd_ms2.mgf  ({dd_mgf} spectra over DD compounds; DD universe {dd_n} ik)
- hmdb_ms2.mgf          ({hmdb_mgf} spectra over HMDB compounds; HMDB universe {hmdb_n} ik)
- ms2_ref_neg_ik14.json / ms2_ref_pos_ik14.json  (bootstrap drop-in {{ik14: peaks}}; experimental
  preferred over in-silico; neg={ref_neg}, pos={ref_pos})
- metabolon_dd_corrected.csv  (InChIKey backfilled name->MAF; SMILES via PubChem PUG REST)
"""

STAGES={"download":stage_download,"fast":stage_fast,"mona":stage_mona,"hmdb":stage_hmdb,"dd":stage_dd,"emit":stage_emit}
def stage_all():
    stage_download(); stage_dd(); stage_fast(); stage_mona(); stage_hmdb(); stage_emit()
if __name__=="__main__":
    for d in (DL,OUT): os.makedirs(d,exist_ok=True)
    arg=sys.argv[1] if len(sys.argv)>1 else "all"
    (stage_all if arg=="all" else STAGES[arg])()
