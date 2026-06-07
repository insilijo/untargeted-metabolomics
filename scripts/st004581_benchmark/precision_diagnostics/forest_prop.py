"""Forest of precision-first propagation chains (the user's xgboost-forest-in-tandem idea).
K chains, each = bootstrapped seed + random descriptor subset, run independently with
EARLY-CLIP (stop the chain the moment held-out RT-prediction error traverses upward = drift
onset). Aggregate by CONSENSUS: admit a compound only if >=M of K chains independently admit
it. Drift FPs are idiosyncratic to whichever bad anchor a chain let in -> they don't reach
consensus -> voted out. M is the precision-first knob (higher M = stricter)."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
ANC="/root/untargeted-metabolomics/data/anchor_panels/anchors_all_platforms.csv"
_PCFG={"neg":("lc/ms neg","anchors_lc_ms_neg.csv","Method3"),
       "pos_early":("lc/ms pos early","anchors_lc_ms_pos_early.csv","Method1"),
       "pos_late":("lc/ms pos late","anchors_lc_ms_pos_late.csv","Method2"),
       "polar":("lc/ms polar","anchors_lc_ms_polar.csv","Method4")}
_pk=sys.argv[1] if len(sys.argv)>1 else "neg"
PLAT,_kf,_meth=_PCFG[_pk]
KIT=f"/root/untargeted-metabolomics/data/anchor_panels/{_kf}"; SMI="/tmp/dd_pubchem_smiles.csv"
_NINJ=int(sys.argv[3]) if len(sys.argv)>3 else 8
MZML=sorted(glob.glob(f"/root/SQuID-INC/data/st004581/mzml/{_meth}_*COLU*.mzML"))[:_NINJ]
MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=max(2,round(0.30*len(MZML))); NRAND=60
FDR_ADMIT=float(sys.argv[2]) if len(sys.argv)>2 else 0.01
print(f"## SWEEP CONFIG: FDR_ADMIT={FDR_ADMIT}  n_inj={len(MZML)}  MINREP={max(2,round(0.30*len(MZML)))}",flush=True)
ISOBAR_PPM=10.0; K=int(sys.argv[4]) if len(sys.argv)>4 else 10; NFEAT=15; ik14=lambda s:(s or "")[:14]
maf=set(); maf_ik=set(); name2id={}; ik2id={}; _eid=0
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    if (r.get("platform") or "").strip().lower()==PLAT:
        nm=M._norm(r.get("name") or ""); _ik=(r.get("inchikey") or "").strip()[:14]
        maf.add(nm)
        # one entry-id per MAF compound, keyed primarily by ik14 (so synonyms collapse)
        if len(_ik)>=14 and _ik in ik2id: eid=ik2id[_ik]
        elif nm in name2id: eid=name2id[nm]
        else: eid=_eid; _eid+=1
        name2id.setdefault(nm,eid)
        if len(_ik)>=14: maf_ik.add(_ik); ik2id.setdefault(_ik,eid)
NMAF=_eid  # distinct MAF compounds (synonyms collapsed by ik)
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"]); df["batch"]=df.source_file
lib0=M.load_library(Path(DD)); anchors=M.load_anchor_points(Path(ANC))
smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# ladder (RI->sec) for an INDEPENDENT prediction used as the dedup tie-break key
_cal={p:list(v) for p,v in anchors.items()}
for _plat,_e in lib0.items():
    _mzs=np.array(sorted(_c["mz"] for _c in _e))
    for _c in _e:
        _t=_c["mz"]*MZ_PPM*1e-6
        if (np.searchsorted(_mzs,_c["mz"]+_t)-np.searchsorted(_mzs,_c["mz"]-_t))==1: _cal.setdefault(_plat,[]).append((_c["mz"],_c["ri"]))
_,_,_,_pp=M.build_batch_ladders(df,_cal,MZ_PPM,1,True)
_agg=defaultdict(list)
for _sec,_ri in _pp[PLAT]: _agg[round(_ri,1)].append(_sec)
_RI=np.array(sorted(_agg)); _SEC=np.array([np.median(_agg[r]) for r in _RI]); _u,_ui=np.unique(_RI,return_index=True)
inv=PchipInterpolator(_u,_SEC[_ui],extrapolate=True)
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if not nm: continue
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=(c.get("ik14") or "")[:14]))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
# MATCH BY INCHIKEY-OR-NAME (DD carries synonym variants the name-key splits; see maf_doublecheck)
# mid = the distinct MAF-compound id this DD compound maps to (ik first so synonyms collapse), else -1
mid=np.array([(ik2id.get(c["ik"]) if len(c["ik"])>=14 and c["ik"] in ik2id else name2id.get(c["name"],-1)) if (c["name"] in maf or (len(c["ik"])>=14 and c["ik"] in maf_ik)) else -1 for c in comp])
inmaf=mid>=0; nmaf=NMAF; N=len(MZML)
def _recall(selmask):  # distinct MAF compounds covered (NOT DD synonyms)
    return len(set(mid[selmask&inmaf]))/nmaf
DDIM=len(_descriptors("CCO"))
print(f"neg compounds {n} (smiles {sum(c['desc'] is not None for c in comp)}, MAF {inmaf.sum()})  descdim {DDIM}  K {K}",flush=True)
peaks=[[] for _ in range(n)]; rtspans=[]
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
print("EIC cached",flush=True)
grng=np.random.RandomState(0); RC=[grng.uniform(lo+TIGHT,hi-TIGHT,NRAND) for lo,hi in rtspans]
def rep_apex(k,pred,W):
    nrep=0; best=None; bi=0.0
    for inj in range(N):
        P=peaks[k][inj]
        if not len(P): continue
        m=np.abs(P[:,0]-pred)<=W
        if m.any():
            nrep+=1; j=np.argmax(P[m,1])
            if P[m][j,1]>bi: bi=P[m][j,1]; best=P[m][j,0]
    return nrep,best
nullc=np.array([sum((len(peaks[k][inj]) and (np.abs(peaks[k][inj][:,0]-c)<=TIGHT).any()) for inj in range(N) for c in RC[inj])/(N*NRAND) for k in range(n)])
withdesc=[k for k in range(n) if comp[k]["desc"] is not None]
DX=np.array([comp[k]["desc"] for k in withdesc])   # descriptor matrix for predictable compounds
# seed kit
kitX=[]; kitY=[]
for r in csv.DictReader(open(KIT)):
    s=r.get("smiles"); d=_descriptors(s) if s else None
    try: sec=float(r["observed_rt_sec"])
    except: sec=None
    if d is not None and sec: kitX.append(np.array(d)); kitY.append(sec)
kitX=np.array(kitX); kitY=np.array(kitY)
def run_chain(cs):
    rng=np.random.RandomState(cs)
    fsub=np.sort(rng.choice(DDIM,NFEAT,replace=False))
    bidx=rng.choice(len(kitX),len(kitX),replace=True)
    vmask=~np.isin(np.arange(len(kitX)),np.unique(bidx))   # OOB seed = validation for early-clip
    Xtr=list(kitX[bidx][:,fsub]); ytr=list(kitY[bidx])
    vX=kitX[vmask][:,fsub]; vY=kitY[vmask]
    admitted=np.zeros(n,bool); best_val=np.inf; bad=0
    for rd in range(12):
        mdl=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=5).fit(np.array(Xtr),np.array(ytr))
        if len(vY):
            ve=np.median(np.abs(mdl.predict(vX)-vY))
            if ve>best_val*1.05: bad+=1
            else: best_val=min(best_val,ve); bad=0
            if bad>=2: break          # EARLY-CLIP: held-out error traversing upward = drift
        pr=mdl.predict(DX[:,fsub]); cand=[]
        for ii,k in enumerate(withdesc):
            if admitted[k]: continue
            nrep,apex=rep_apex(k,pr[ii],TIGHT)
            if nrep<MINREP or apex is None: continue
            if binom.sf(nrep-1,N,np.clip(nullc[k],1e-6,0.999))>FDR_ADMIT: continue
            cand.append((k,apex,abs(apex-pr[ii])))
        if not cand: break
        cand.sort(key=lambda x:x[2]); taken=[]; new=[]
        for k,apex,dp in cand:
            if any(abs(MZc[k]-MZc[kk])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(apex-aa)<=TIGHT for kk,aa in taken): continue
            taken.append((k,apex)); new.append((k,apex))
        if not new: break
        for k,apex in new: admitted[k]=True; Xtr.append(comp[k]["desc"][fsub]); ytr.append(apex)
    return admitted
votes=np.zeros(n)
for c in range(K):
    votes+=run_chain(c); print(f"  chain {c} done (admitted {int(votes.sum() if c==0 else 0) or ''})",flush=True) if c==0 else None
# ---- LADDER-FALLBACK: no-SMILES compounds reached via RI->sec ladder (no structure model).
# Only MASS-UNIQUE no-SMILES compounds (a peak at their m/z is unambiguously them; RT just
# confirms) -> keeps no-SMILES recall while cutting coincidental-peak FPs. ----
_so=np.argsort(MZc); _ss=MZc[_so]
massuniq=np.ones(n,bool)
for k in range(n):
    t=MZc[k]*MZ_PPM*1e-6
    massuniq[k]=(np.searchsorted(_ss,MZc[k]+t)-np.searchsorted(_ss,MZc[k]-t))<=1
nolad=0; nolad_skip=0
for k in range(n):
    if comp[k]["desc"] is not None: continue
    if not massuniq[k]: nolad_skip+=1; continue          # isobaric no-SMILES -> ladder can't disambiguate
    nrep,apex=rep_apex(k,comp[k]["pred"],TIGHT)
    if nrep<MINREP or apex is None: continue
    if binom.sf(nrep-1,N,np.clip(nullc[k],1e-6,0.999))>FDR_ADMIT: continue
    votes[k]=K; nolad+=1
print(f"  ladder-fallback admitted {nolad} mass-unique no-SMILES compounds (skipped {nolad_skip} isobaric)")
print(f"\n{'consensus>=M':>12}{'admitted':>9}{'recall':>8}{'precision':>10}")
for Mv in range(1,K+1):
    sel=votes>=Mv; a=int(sel.sum())
    if not a: continue
    print(f"{Mv:>12}{a:>9}{_recall(sel):>8.3f}{inmaf[sel].sum()/a:>10.3f}")

# ---- decomposition of the M=1 admitted set (honest true-precision accounting) ----
sel=votes>=1
strength=np.zeros(n); reprod=np.zeros(n); gapex=np.full(n,np.nan)
for k in range(n):
    allp=[p for inj in range(N) for p in peaks[k][inj] if len(peaks[k][inj])]
    flat=[row for inj in range(N) for row in (peaks[k][inj] if len(peaks[k][inj]) else [])]
    if flat:
        A=np.array(flat); strength[k]=A[:,1].max(); gapex[k]=A[A[:,1].argmax(),0]
        reprod[k]=sum(bool(len(peaks[k][inj])) and bool((np.abs(peaks[k][inj][:,0]-gapex[k])<=TIGHT).any()) for inj in range(N))
maf_idx=[j for j in range(n) if inmaf[j]]; maf_mz=np.array([MZc[j] for j in maf_idx])
cats=defaultdict(int)
for k in np.where(sel)[0]:
    if inmaf[k]: cats['TP']+=1; continue
    iso=np.abs(maf_mz-MZc[k])<=MZc[k]*ISOBAR_PPM*1e-6
    if iso.any():
        isoj=[maf_idx[t] for t in np.where(iso)[0]]
        cats['isobar_dup' if any(sel[j] for j in isoj) else 'substitution']+=1
    else:
        cats['novel_real' if (reprod[k]>=MINREP and strength[k]>2*FLOOR) else 'weak_noise']+=1
tot=int(sel.sum()); tp=cats['TP']; nov=cats['novel_real']; sub=cats['substitution']; idup=cats['isobar_dup']
print("\n--- M=1 admitted-set decomposition (n=%d) ---"%tot)
for c,v in sorted(cats.items(),key=lambda x:-x[1]): print(f"  {c:<14}{v:>5}  {v/tot:.3f}")
print(f"\nclosed-world precision (MAF only)                 : {tp/tot:.3f}")
print(f"true precision (TP + novel_real = real compounds) : {(tp+nov)/tot:.3f}")
print(f"genuine-error floor (TP / TP+substitution+isobar) : {tp/max(tp+sub+idup,1):.3f}")

# ---- global co-elution dedup: one call per (isobaric m/z, co-eluting apex) ----
def dedup(selmask):
    idx=[k for k in np.where(selmask)[0] if not np.isnan(gapex[k])]
    keep=np.zeros(n,bool); used=[]
    for k in sorted(idx,key=lambda k:(abs(gapex[k]-comp[k]["pred"]),-votes[k])):
        if any(abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT for j in used): continue
        keep[k]=True; used.append(k)
    return keep
print("\n--- AFTER global co-elution dedup (one call per peak) ---")
print(f"{'consensus>=M':>12}{'admitted':>9}{'recall':>8}{'precision':>10}{'truePrec':>10}")
for Mv in [1,2,3,4,5]:
    d=dedup(votes>=Mv); a=int(d.sum())
    if not a: continue
    tp_=inmaf[d].sum()
    nov_=sum(1 for k in np.where(d)[0] if not inmaf[k] and not (np.abs(maf_mz-MZc[k])<=MZc[k]*ISOBAR_PPM*1e-6).any() and reprod[k]>=MINREP and strength[k]>2*FLOOR)
    print(f"{Mv:>12}{a:>9}{tp_/nmaf:>8.3f}{tp_/a:>10.3f}{(tp_+nov_)/a:>10.3f}")

# ---- COMPOUND-CLUSTER scoring (squid_inc design): co-eluting isobars = ONE annotation,
# TP if ANY member is MAF. Lossless: keeps MAF compound in its cluster (recall held) while
# collapsing isobaric peers (precision up). ----
def cluster_score(selmask):
    idx=[k for k in np.where(selmask)[0] if not np.isnan(gapex[k])]
    clusters=[]
    for k in sorted(idx,key=lambda k:-strength[k]):
        for cl in clusters:
            j=cl[0]
            if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT:
                cl.append(k); break
        else: clusters.append([k])
    ncl=len(clusters)
    tp_cl=sum(any(inmaf[k] for k in cl) for cl in clusters)
    maf_cov=len(set(mid[k] for cl in clusters for k in cl if inmaf[k]))
    return ncl,tp_cl,maf_cov
print("\n--- COMPOUND-CLUSTER scoring (one annotation per co-eluting isobar group) ---")
print(f"{'consensus>=M':>12}{'clusters':>9}{'recall':>8}{'precision':>11}")
for Mv in [1,2,3,4,5]:
    ncl,tp_cl,maf_cov=cluster_score(votes>=Mv)
    if not ncl: continue
    print(f"{Mv:>12}{ncl:>9}{maf_cov/nmaf:>8.3f}{tp_cl/ncl:>11.3f}")

# ---- MISLOCATION RESCUE PASS: after propagation, train final model on all admitted, then
# do ONE wider-window search for the unadmitted (mislocated FNs). One-to-one by prediction-
# closeness (answer-key-first); FDR vs the WIDER null; cluster-score the combined set. ----
WIDE=20.0
admset=[k for k in range(n) if votes[k]>=1 and comp[k]["desc"] is not None and not np.isnan(gapex[k])]
if len(admset)>20:
    fmdl=HistGradientBoostingRegressor(max_iter=400,max_depth=4,learning_rate=0.05,min_samples_leaf=5).fit(
        np.array([comp[k]["desc"] for k in admset]), np.array([gapex[k] for k in admset]))
    # wider null per compound
    nullW=np.array([sum((len(peaks[k][inj]) and (np.abs(peaks[k][inj][:,0]-c)<=WIDE).any()) for inj in range(N) for c in RC[inj])/(N*NRAND) for k in range(n)])
    rescued=np.zeros(n,bool); rap={}
    cand=[]
    for k in range(n):
        if votes[k]>=1 or comp[k]["desc"] is None: continue
        pr=float(fmdl.predict([comp[k]["desc"]])[0])
        nrep,apex=rep_apex(k,pr,WIDE)
        if nrep<MINREP or apex is None: continue
        if binom.sf(nrep-1,N,np.clip(nullW[k],1e-6,0.999))>FDR_ADMIT: continue
        cand.append((k,apex,abs(apex-pr)))
    cand.sort(key=lambda x:x[2])
    taken=[]
    for k,apex,dp in cand:
        if any(abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(apex-aa)<=TIGHT for j,aa in taken): continue
        taken.append((k,apex)); rescued[k]=True; gapex[k]=apex
    sel2=(votes>=1)|rescued
    # cluster-score combined
    idx=[k for k in np.where(sel2)[0] if not np.isnan(gapex[k])]
    clusters=[]
    for k in sorted(idx,key=lambda k:-strength[k] if strength[k]>0 else 0):
        for cl in clusters:
            j=cl[0]
            if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT: cl.append(k); break
        else: clusters.append([k])
    ncl=len(clusters); tp_cl=sum(any(inmaf[k] for k in cl) for cl in clusters)
    maf_cov=len(set(mid[k] for cl in clusters for k in cl if inmaf[k]))
    nres_maf=sum(inmaf[k] for k in np.where(rescued)[0])
    print(f"\n--- MISLOCATION RESCUE (wide={WIDE}s, final model on {len(admset)} admitted) ---")
    print(f"rescued compounds            : {int(rescued.sum())}  (of which MAF: {nres_maf})")
    print(f"cluster recall  : {maf_cov}/{nmaf} = {maf_cov/nmaf:.3f}   (was 0.532 pre-rescue)")
    print(f"cluster precision: {tp_cl}/{ncl} = {tp_cl/ncl:.3f}   (was 0.658 pre-rescue)")

# ---- WHERE precision is lost: FP-cluster decomposition + conflation test ----
def fp_decomp(selmask):
    idx=[k for k in np.where(selmask)[0] if not np.isnan(gapex[k])]
    clusters=[]
    for k in sorted(idx,key=lambda k:(-strength[k] if strength[k]>0 else 0)):
        for cl in clusters:
            j=cl[0]
            if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT: cl.append(k); break
        else: clusters.append([k])
    covered=set(k for cl in clusters for k in cl if inmaf[k])
    fn=[k for k in range(n) if inmaf[k] and k not in covered]   # MAF compounds we MISSED
    fn_mz=np.array([MZc[k] for k in fn]) if fn else np.array([])
    cats=defaultdict(int); tp_sizes=[]; fp_n=0
    for cl in clusters:
        if any(inmaf[k] for k in cl): tp_sizes.append(len(cl)); continue
        fp_n+=1; mz=MZc[cl[0]]
        iso_fn = len(fn_mz)>0 and bool((np.abs(fn_mz-mz)<=mz*ISOBAR_PPM*1e-6).any())
        from_ladder = all(comp[k]["desc"] is None for k in cl)
        if iso_fn: cats["conflation: isobaric MISSED-MAF exists (wrong compound called for an answer-key peak)"]+=1
        elif from_ladder: cats["ladder no-SMILES FP (imprecise RI->sec)"]+=1
        else: cats["genuine novel/noise (no MAF compound at this m/z)"]+=1
    return clusters,cats,tp_sizes,fp_n
print("\n=== PRECISION-LOSS DECOMPOSITION (M=1 FP clusters) ===")
cl1,cats,tps,fpn=fp_decomp(votes>=1)
print(f"total clusters {len(cl1)}  TP {len(tps)}  FP {fpn}")
for c,v in sorted(cats.items(),key=lambda x:-x[1]): print(f"  {v:>4}  ({v/max(fpn,1):.2f} of FPs)  {c}")
import numpy as _np
print(f"\nTP-cluster size (conflation within correct calls): median {int(_np.median(tps))}  mean {_np.mean(tps):.2f}  max {max(tps)}  (1=clean, >1=isobaric candidates listed)")
print(f"  TP clusters that are CLEAN (size 1): {sum(1 for s in tps if s==1)}/{len(tps)}")

# ---- ADDUCT ATTRIBUTION: are the FP peaks adducts/isotopes of co-eluting stronger compounds? ----
fpl=df[df.platform==PLAT]; fmz=fpl.mz.to_numpy(); frt=fpl.rt.to_numpy(); fint=fpl.intensity.to_numpy()
_o=np.argsort(fmz); fmz,frt,fint=fmz[_o],frt[_o],fint[_o]
# neg-mode: peak at m/z X is a non-principal ion if a co-eluting STRONGER feature sits at X-delta
# (i.e. X = that compound's [M-H] + delta). deltas from [M-H]:
ADD={"13C-isotope":1.00336,"[M+Cl]":35.97668,"[M+FA-H]":46.00548,"[M+Hac-H]":60.02113,
     "[M+Na-2H]":20.97417,"[M+K-2H]":36.94816,"in-source +H2O":18.01056}
def feat_int(mz,rt):
    t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz-t); hi=np.searchsorted(fmz,mz+t)
    best=0.0
    for j in range(lo,hi):
        if abs(frt[j]-rt)<=TIGHT: best=max(best,fint[j])
    return best
def adduct_of(mz,rt):
    my=feat_int(mz,rt)
    for name,d in ADD.items():
        t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz-d-t); hi=np.searchsorted(fmz,mz-d+t)
        for j in range(lo,hi):
            if abs(frt[j]-rt)<=TIGHT and fint[j]>max(my,1)*1.3: return name
    return None
cl1,_,_,_=fp_decomp(votes>=1)
add_fp=defaultdict(int); nonadd=0; fp_tot=0
for cl in cl1:
    if any(inmaf[k] for k in cl): continue
    fp_tot+=1; k=cl[0]
    if np.isnan(gapex[k]): nonadd+=1; continue
    a=adduct_of(MZc[k],gapex[k])
    if a: add_fp[a]+=1
    else: nonadd+=1
print(f"\n=== ADDUCT ATTRIBUTION of M=1 FP clusters (n={fp_tot}) ===")
tot_add=sum(add_fp.values())
print(f"explained as adduct/isotope of a co-eluting STRONGER compound: {tot_add}  ({tot_add/max(fp_tot,1):.2f})")
for a,v in sorted(add_fp.items(),key=lambda x:-x[1]): print(f"    {v:>3}  {a}")
print(f"NOT adduct-explained (genuine novel / noise): {nonadd}  ({nonadd/max(fp_tot,1):.2f})")

# ---- BIDIRECTIONAL adduct+fragment FILTER -> precision recovery ----
LOSS={"-H2O":18.01056,"-CO2":43.98983,"-CO":27.99491,"-NH3":17.02655,"-CH2O":30.01056,"-hexose":162.05282,"-SO3":79.95682,"-H2O-H2O":36.02112}
def explained(mz,rt):
    my=feat_int(mz,rt)
    for name,d in ADD.items():   # our peak is a heavier adduct; principal lighter at mz-d
        t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz-d-t); hi=np.searchsorted(fmz,mz-d+t)
        for j in range(lo,hi):
            if abs(frt[j]-rt)<=TIGHT and fint[j]>max(my,1)*1.3: return name
    for name,d in LOSS.items():   # our peak is an in-source fragment; principal heavier at mz+loss
        t=mz*MZ_PPM*1e-6; lo=np.searchsorted(fmz,mz+d-t); hi=np.searchsorted(fmz,mz+d+t)
        for j in range(lo,hi):
            if abs(frt[j]-rt)<=TIGHT and fint[j]>max(my,1)*1.3: return name
    return None
filt=np.zeros(n,bool)
for k in np.where(votes>=1)[0]:
    if not np.isnan(gapex[k]) and explained(MZc[k],gapex[k]): filt[k]=True
filt_tp=int((filt&inmaf).sum()); filt_fp=int((filt&~inmaf).sum())
def cluster_f(selmask):
    idx=[k for k in np.where(selmask&~filt)[0] if not np.isnan(gapex[k])]
    clusters=[]
    for k in sorted(idx,key=lambda k:(-strength[k] if strength[k]>0 else 0)):
        for cl in clusters:
            j=cl[0]
            if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT: cl.append(k); break
        else: clusters.append([k])
    ncl=len(clusters); tp=sum(any(inmaf[k] for k in cl) for cl in clusters)
    cov=len(set(mid[k] for cl in clusters for k in cl if inmaf[k]))
    return ncl,tp,cov
print(f"\n=== AFTER bidirectional adduct+fragment FILTER ===")
print(f"filtered {int(filt.sum())} admits ({filt_fp} FP, {filt_tp} TP)")
print(f"{'>=M':>4}{'clusters':>9}{'recall':>8}{'precision':>11}")
for Mv in [1,2,3]:
    ncl,tp,cov=cluster_f(votes>=Mv)
    if ncl: print(f"{Mv:>4}{ncl:>9}{cov/nmaf:>8.3f}{tp/ncl:>11.3f}")

# ===== CLASSIFY THE MAF MISSES (FNs) =====
# recover the M=1 cluster coverage
_cl,_,_=cluster_score(votes>=1) if False else (None,None,None)
# rebuild clusters for votes>=1
_idx=[k for k in np.where(votes>=1)[0] if not np.isnan(gapex[k])]
_clusters=[]
for k in sorted(_idx,key=lambda k:(-strength[k] if strength[k]>0 else 0)):
    for cl in _clusters:
        j=cl[0]
        if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT: cl.append(k); break
    else: _clusters.append([k])
covered_mids=set(mid[k] for cl in _clusters for k in cl if inmaf[k])
by_mid=defaultdict(list)
for k in range(n):
    if inmaf[k]: by_mid[mid[k]].append(k)
adm=set(np.where(votes>=1)[0])
def anywhere_strong(k):  # reproducible strong peak at ANY rt
    cnt=defaultdict(int); mx=0.0
    for inj in range(N):
        for rt_,it in peaks[k][inj]:
            if it>2*FLOOR: cnt[round(rt_/(2*TIGHT))]+=1; mx=max(mx,it)
    return any(v>=MINREP for v in cnt.values())
cats=defaultdict(int)
for m_id,ks in by_mid.items():
    if m_id in covered_mids: continue                      # recovered, not a miss
    has_smiles=any(comp[k]["desc"] is not None for k in ks)
    at_pred=False
    for k in ks:
        nr,ap=rep_apex(k,comp[k]["pred"],TIGHT)
        if nr>=MINREP and ap is not None: at_pred=True; break
    anyw=any(anywhere_strong(k) for k in ks)
    sm="" if has_smiles else " [no-SMILES]"
    if at_pred:
        # peak at predicted RT but not admitted -> isobar-collapsed or FDR/weak
        iso=False
        for k in ks:
            for a in adm:
                if a not in ks and abs(MZc[a]-MZc[k])<=MZc[k]*ISOBAR_PPM*1e-6 and not np.isnan(gapex[a]) and abs(gapex[a]-comp[k]["pred"])<=TIGHT: iso=True; break
            if iso: break
        cats[("isobar-collapsed (a co-eluting isobar took the peak)" if iso else "at predicted RT but FDR/weak-rejected")+sm]+=1
    elif anyw:
        cats["RT-MISLOCATED (strong peak elsewhere at m/z)"+sm]+=1
    elif not has_smiles:
        cats["no-SMILES, no clear peak (unreachable + faint)"]+=1
    else:
        cats["TRULY ABSENT / sub-floor (no strong peak at m/z)"]+=1
tot=sum(cats.values())
print(f"\n===== MAF MISS (FN) CLASSIFICATION =====  total misses: {tot}  (of {nmaf} MAF; recall {1-tot/nmaf:.3f})")
for c,v in sorted(cats.items(),key=lambda x:-x[1]): print(f"  {v:>4}  ({v/tot:.2f})  {c}")

# ===== PROBABILISTIC OUTPUT: forest-consensus presence probability + calibration =====
prob=votes/K   # fraction of forest chains admitting = presence probability
_sel=votes>=1
_pidx=[k for k in np.where(_sel)[0] if not np.isnan(gapex[k])]
_pcl=[]
for k in sorted(_pidx,key=lambda k:(-strength[k] if strength[k]>0 else 0)):
    for cl in _pcl:
        j=cl[0]
        if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT: cl.append(k); break
    else: _pcl.append([k])
clp=[(max(prob[k] for k in cl), any(inmaf[k] for k in cl)) for cl in _pcl]
print("\n=== PRESENCE-PROBABILITY CALIBRATION (cluster prob = max member votes/K) ===")
print(f"{'prob bin':>12}{'clusters':>10}{'obs precision':>15}  (calibrated => obs ~ bin midpoint)")
for lo,hi in [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.001)]:
    inb=[tp for p,tp in clp if lo<=p<hi]
    if inb: print(f"  [{lo:.1f},{hi:.1f}){'':>4}{len(inb):>10}{sum(inb)/len(inb):>15.3f}")
print(f"\n=== P/R as continuous PROBABILITY threshold ===\n{'prob>=':>8}{'clusters':>9}{'recall':>8}{'precision':>11}")
for tau in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    cls=[cl for cl in _pcl if max(prob[k] for k in cl)>=tau]
    if not cls: continue
    tp=sum(any(inmaf[k] for k in cl) for cl in cls); cov=len(set(mid[k] for cl in cls for k in cl if inmaf[k]))
    print(f"{tau:>8.1f}{len(cls):>9}{cov/nmaf:>8.3f}{tp/len(cls):>11.3f}")
