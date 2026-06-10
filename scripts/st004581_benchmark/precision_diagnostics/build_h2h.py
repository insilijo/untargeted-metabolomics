"""Transform /tmp/forest_sweep.py -> /tmp/forest_sweep_h2h.py:
  - PLATFORM + LIBRARY env switches (4 platforms x {dd,uni})
  - memory-efficient STREAMING EIC + m/z dedup + on-disk cache (storage fix)
  - universe candidate branch (metabolon_mass m/z, kit-structure-predicted RT, m/z-present prune)
Reuses forest_sweep's validated propagation + cardinality + IDPREC(MAF-only precision) verbatim."""
s=open("/tmp/forest_sweep.py").read()

# ---- PATCH A: header (platform/library/kit/adduct/mzml) ----
A_old='''PLAT="lc/ms neg"; DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
KIT="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]'''
A_new='''import os as _osf
PLATFORM=_osf.environ.get("PLATFORM","lc/ms neg"); LIBRARY=_osf.environ.get("LIBRARY","dd")
_METH={"lc/ms neg":("Method3","anchors_lc_ms_neg.csv",-1.007276),
       "lc/ms pos early":("Method1","anchors_lc_ms_pos_early.csv",1.007276),
       "lc/ms pos late":("Method2","anchors_lc_ms_pos_late.csv",1.007276),
       "lc/ms polar":("Method4","anchors_lc_ms_polar.csv",1.007276)}
_meth,_kitf,ADD=_METH[PLATFORM]
PLAT=PLATFORM; DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
KIT="/root/untargeted-metabolomics/data/anchor_panels/"+_kitf; SMI="/tmp/dd_pubchem_smiles.csv"
UNI="/mnt/volume-hel1-1/data/processed/compound_universe_broad.csv"; CACHE="/mnt/volume-hel1-1/cache"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/%s_*COLU*.mzML"%_meth))[:8]'''
assert A_old in s; s=s.replace(A_old,A_new)

# ---- PATCH B: candidate construction (dd vs uni) ----
B_old='''comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if not nm: continue
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=(c.get("ik14") or "")[:14]))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6'''
B_new='''import pickle as _pk, csv as _csvu
_csvu.field_size_limit(10**7)
# kit-trained structure model for universe RT prediction (no RI prior for ZINC)
_kitmdl=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=4).fit(kitX,kitY)
def _present_files():
    cp=CACHE+"/presf_%s_%d.pkl"%(_meth,int(FLOOR))
    if _osf.path.exists(cp): return _pk.load(open(cp,"rb"))
    arrs=[]
    for mp in MZML:
        acc=[]
        for spec in pymzml.run.Reader(mp):
            if spec.ms_level!=1: continue
            sm=np.asarray(spec.mz); si=np.asarray(spec.i)
            if len(sm): acc.append(sm[si>FLOOR])
        arrs.append(np.sort(np.concatenate(acc)) if acc else np.array([0.0]))
    _osf.makedirs(CACHE,exist_ok=True); _pk.dump(arrs,open(cp,"wb")); return arrs
comp=[]
if LIBRARY=="dd":
    for c in lib0.get(PLAT,[]):
        nm=M._norm(c["name"])
        if not nm: continue
        s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
        comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=(c.get("ik14") or "")[:14]))
else:
    from rdkit import Chem as _Chem
    from rdkit.Chem import Descriptors as _Desc
    _nmf=CACHE+"/uni_neutral.pkl"
    if _osf.path.exists(_nmf): neut=_pk.load(open(_nmf,"rb"))
    else:
        neut={}
        for r in _csvu.DictReader(open(UNI)):
            ik=(r.get("inchikey") or "")[:14]; sm=r.get("smiles")
            if not sm or ik in neut: continue
            m=_Chem.MolFromSmiles(sm)
            if m is None: continue
            try: neut[ik]=float(_Desc.ExactMolWt(m))
            except Exception: pass
        _osf.makedirs(CACHE,exist_ok=True); _pk.dump(neut,open(_nmf,"wb"))
    print("universe neutral masses (RDKit):",len(neut),flush=True)
    presf=_present_files(); print("present MS1 per-file arrays:",[len(a) for a in presf],flush=True)
    def _isp(mz):  # reproducible: m/z above FLOOR in >=MINREP injections
        t=mz*MZ_PPM*1e-6; c=0
        for a in presf:
            i=np.searchsorted(a,mz-t)
            if i<len(a) and a[i]<=mz+t:
                c+=1
                if c>=MINREP: return True
        return False
    seen=set(); rows=[]
    for r in _csvu.DictReader(open(UNI)):
        ik=(r.get("inchikey") or "")[:14]; sm=r.get("smiles")
        if not sm or ik in seen or ik not in neut: continue
        mz=neut[ik]+ADD              # ADD = -proton (neg) / +proton (pos/polar)
        if not _isp(mz): continue
        seen.add(ik); rows.append((ik,sm,mz))
    print("universe m/z-present candidates:",len(rows),"(computing descriptors...)",flush=True)
    for ik,sm,mz in rows:
        d=_descriptors(sm)
        dd=np.array(d) if d is not None else None
        pr=float(_kitmdl.predict(dd[None,:])[0]) if dd is not None else float(np.median(kitY))
        comp.append(dict(name="",mz=mz,desc=dd,pred=pr,ik=ik))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6'''
assert B_old in s; s=s.replace(B_old,B_new)

# ---- PATCH C: EIC extraction -> streaming + dedup + cache ----
C_old='''peaks=[[] for _ in range(n)]; rtspans=[]
for mp in MZML:
    rts=[];rows=[]
    for spec in pymzml.run.Reader(mp):
        if spec.ms_level!=1: continue
        sm=np.asarray(spec.mz); si=np.asarray(spec.i); rts.append(spec.scan_time_in_minutes()*60)
        if not len(sm): rows.append(np.zeros(n)); continue
        lo=np.searchsorted(sm,MZc-tol); hi=np.searchsorted(sm,MZc+tol)
        rows.append(np.array([si[lo[k]:hi[k]].sum() if hi[k]>lo[k] else 0.0 for k in range(n)]))
    rt=np.array(rts); mat=np.array(rows); rtspans.append((rt.min(),rt.max()))
    for k in range(n):
        col=mat[:,k]; loc=(col>FLOOR)&(col>=np.roll(col,1))&(col>=np.roll(col,-1))
        peaks[k].append(np.array([(rt[j],col[j]) for j in np.where(loc)[0]]) if loc.any() else np.empty((0,2)))
import time as _t;_T0=_t.time();print("EIC done %.0fs"%(_t.time()-_T0),flush=True)'''
C_new='''import time as _t
def _extract_stream(files,mzc):
    nn=len(mzc); t=mzc*MZ_PPM*1e-6; lo_mz=mzc-t; hi_mz=mzc+t
    pks=[[] for _ in range(nn)]; spans=[]
    for mp in files:
        pk=[[] for _ in range(nn)]; prev2=None; prev1=None; rt1=0.0; rmin=1e18; rmax=-1e18
        for spec in pymzml.run.Reader(mp):
            if spec.ms_level!=1: continue
            sm=np.asarray(spec.mz); si=np.asarray(spec.i); rt=spec.scan_time_in_minutes()*60
            rmin=min(rmin,rt); rmax=max(rmax,rt)
            if len(sm):
                cs=np.concatenate(([0.0],np.cumsum(si.astype(np.float64))))
                lo=np.searchsorted(sm,lo_mz); hi=np.searchsorted(sm,hi_mz); cur=cs[hi]-cs[lo]
            else: cur=np.zeros(nn)
            if prev2 is not None:
                ismax=(prev1>FLOOR)&(prev1>=prev2)&(prev1>=cur)
                for k in np.nonzero(ismax)[0]: pk[int(k)].append((rt1,prev1[int(k)]))
            prev2=prev1; prev1=cur; rt1=rt
        CAP=40   # keep top-CAP peaks per m/z per file (memory bound; propagation only needs strong peaks)
        for k in range(nn):
            if not pk[k]: pks[k].append(np.empty((0,2))); continue
            a=np.array(pk[k])
            if len(a)>CAP: a=a[np.argsort(a[:,1])[-CAP:]]
            pks[k].append(a)
        spans.append((rmin,rmax))
    return pks,spans
# dedup candidate m/z into unique bins (within tol) -> extract once per bin
_o=np.argsort(MZc); umz=[]; grp=np.zeros(n,int)
for k in _o:
    k=int(k)
    if umz and abs(MZc[k]-umz[-1])<=MZc[k]*MZ_PPM*1e-6: grp[k]=len(umz)-1
    else: umz.append(MZc[k]); grp[k]=len(umz)-1
umz=np.array(umz); nU=len(umz)
_ck=CACHE+"/eic_%s_%s_n%d_f%d.pkl"%(_meth,LIBRARY,nU,int(FLOOR))
_T0=_t.time()
if _osf.path.exists(_ck):
    upeaks,rtspans=_pk.load(open(_ck,"rb")); print("EIC cache hit (%d uniq m/z) %.0fs"%(nU,_t.time()-_T0),flush=True)
else:
    upeaks,rtspans=_extract_stream(MZML,umz)
    _osf.makedirs(CACHE,exist_ok=True); _pk.dump((upeaks,rtspans),open(_ck,"wb"))
    print("EIC streamed+cached (%d uniq m/z) %.0fs"%(nU,_t.time()-_T0),flush=True)
peaks=[[upeaks[grp[k]][inj] for inj in range(len(MZML))] for k in range(n)]'''
assert C_old in s; s=s.replace(C_old,C_new)

# ---- PATCH D: always print the IDPREC (MAF-only) line + a clean H2H summary ----
# force-enable the 3-metric summary regardless of env
D_anchor='''import os
if os.environ.get(\'DUMP\'):'''
D_new='''import os
# H2H: always emit recall / closed-world prec / MAF-only prec at M=1
_cls1=cluster(votes>=1); _cov=set(mid[k] for cl in _cls1 for k in cl if inmaf[k]); _rec=len(card_recall_set(_cov))/nmaf
_negmaf=[k for k in range(n) if inmaf[k]]; _cor=0; _sub=0; _nov=0
for cl in _cls1:
    if any(inmaf[k] or comp[k]["ik"] in full_maf_ik for k in cl): _cor+=1; continue
    kk=max(cl,key=lambda k:strength[k]); ap=gapex[kk]; mz=MZc[kk]
    if any(abs(MZc[j]-mz)<=mz*ISOBAR_PPM*1e-6 and abs(comp[j]["pred"]-ap)<=TIGHT for j in _negmaf): _sub+=1
    else: _nov+=1
print("H2H %s | %s | n=%d nmaf=%d | recall(card)=%.3f closed-world-prec=%.3f MAF-only-prec=%.3f (correct=%d subst=%d novel=%d)"%(
    PLATFORM,LIBRARY,n,nmaf,_rec,_cor/max(len(_cls1),1),_cor/max(_cor+_sub,1),_cor,_sub,_nov),flush=True)
if os.environ.get(\'DUMP\'):'''
assert D_anchor in s; s=s.replace(D_anchor,D_new)

# ---- PATCH E: drop vestigial feature-table read (FEAT no longer in header) ----
E_old='''df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"])'''
assert E_old in s; s=s.replace(E_old,"# (feature-table read removed: EIC comes from raw mzML)")

open("/tmp/forest_sweep_h2h.py","w").write(s)
import ast; ast.parse(s); print("forest_sweep_h2h.py built OK, lines:",s.count(chr(10)))
