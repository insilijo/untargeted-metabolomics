"""Slim sweep: kit (subsampled) drives BOTH the RI->sec ladder AND the structure-RT seed.
args: KIT_SIZE(0=all) KIT_SEED K  -> outputs cluster P/R per consensus-M (threshold curve)
+ cardinality recall + cross-platform precision."""
import sys, csv, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pymzml
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,"/root/SQuID-INC"); sys.path.insert(0,"/root/untargeted-metabolomics/scripts")
import library_match_rtri as M
from squid_inc.features.rt_model import _descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
KIT_SIZE=int(sys.argv[1]) if len(sys.argv)>1 else 0
KIT_SEED=int(sys.argv[2]) if len(sys.argv)>2 else 0
K=int(sys.argv[3]) if len(sys.argv)>3 else 6
import os as _osf
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
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/%s_*COLU*.mzML"%_meth))[:8]
import os as _osf; MZ_PPM=7.0; FLOOR=float(_osf.environ.get('FLOOR','50000')); TIGHT=6.0; MINREP=2; NRAND=60; FDR_ADMIT=0.05; ISOBAR_PPM=10.0; NFEAT=15; ik14=lambda s:(s or "")[:14]
maf=set(); maf_ik=set(); name2id={}; ik2id={}; _eid=0; full_maf_ik=set(); maf_rt={}
for r in csv.DictReader(open(GT)):
    if r.get("unannotatable","")=="true": continue
    _ik=(r.get("inchikey") or "").strip()[:14]
    if len(_ik)>=14: full_maf_ik.add(_ik)
    if (r.get("platform") or "").strip().lower()==PLAT:
        nm=M._norm(r.get("name") or "")
        maf.add(nm)
        if len(_ik)>=14 and _ik in ik2id: eid=ik2id[_ik]
        elif nm in name2id: eid=name2id[nm]
        else: eid=_eid; _eid+=1
        name2id.setdefault(nm,eid)
        if len(_ik)>=14: maf_ik.add(_ik); ik2id.setdefault(_ik,eid)
        try: maf_rt[eid]=float(r.get('rt') or r.get('observed_rt') or 'nan')
        except: pass
NMAF=_eid
# (feature-table read removed: EIC comes from raw mzML)
lib0=M.load_library(Path(DD)); smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# ---- KIT (subsampled) drives ladder + seed ----
kit=[]
for r in csv.DictReader(open(KIT)):
    s=r.get("smiles"); d=_descriptors(s) if s else None
    try: sec=float(r["observed_rt_sec"]); ri=float(r["ri"])
    except: sec=ri=None
    if d is not None and sec and ri: kit.append((d,sec,ri,s))
rng=np.random.RandomState(KIT_SEED)
import os as _os
KIT_MODE=_os.environ.get("KIT_MODE","random")
if KIT_SIZE and KIT_SIZE<len(kit):
    if KIT_MODE=="spread":
        ks=sorted(range(len(kit)),key=lambda i:kit[i][1])
        idx=sorted(set(ks[int(round(j*(len(ks)-1)/(KIT_SIZE-1)))] for j in range(KIT_SIZE)))
    elif KIT_MODE=="gap":
        # greedy: add the anchor the CURRENT ladder predicts worst (fills high-curvature regions)
        ri=[kit[i][2] for i in range(len(kit))]; sec=[kit[i][1] for i in range(len(kit))]
        o=sorted(range(len(kit)),key=lambda i:ri[i]); chosen=[o[0],o[-1]]
        def _lad(ch):
            cr=sorted(set(ch),key=lambda i:ri[i]); xs=[]; ys=[]
            for i in cr:
                if not xs or ri[i]>xs[-1]+1e-6: xs.append(ri[i]); ys.append(sec[i])
            return (PchipInterpolator(np.array(xs),np.array(ys),extrapolate=True) if len(xs)>=2 else None)
        while len(set(chosen))<KIT_SIZE:
            L=_lad(chosen)
            if L is None: break
            cand=[(abs(float(L(ri[i]))-sec[i]),i) for i in range(len(kit)) if i not in chosen]
            if not cand: break
            chosen.append(max(cand)[1])
        idx=sorted(set(chosen))
    elif KIT_MODE=='chemfn':
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        def _fp(sm):
            m=Chem.MolFromSmiles(sm) if sm else None
            return AllChem.GetMorganFingerprintAsBitVect(m,2,2048) if m else None
        kfps=[_fp(k[3]) for k in kit]
        maffps=[]
        for _c in lib0.get(PLAT,[]):
            if M._norm(_c['name']) in maf:
                _f=_fp(smi.get(_c['ik14']))
                if _f is not None: maffps.append(_f)
        def _d(a,b): return 1.0-DataStructs.TanimotoSimilarity(a,b)
        E=min(KIT_SIZE,10); chosen=[0]
        while len(chosen)<E:
            best=-1.0; bi=None
            for i in range(len(kit)):
                if i in chosen or kfps[i] is None: continue
                md=min(_d(kfps[i],kfps[c]) for c in chosen)
                if md>best: best=md; bi=i
            chosen.append(bi)
        mind=[min(_d(mf,kfps[c]) for c in chosen) for mf in maffps]
        while len(chosen)<KIT_SIZE:
            best=-1.0; bi=None
            for i in range(len(kit)):
                if i in chosen or kfps[i] is None: continue
                gain=sum(max(0.0,mind[m]-_d(maffps[m],kfps[i])) for m in range(len(maffps)))
                if gain>best: best=gain; bi=i
            chosen.append(bi)
            mind=[min(mind[m],_d(maffps[m],kfps[bi])) for m in range(len(maffps))]
        idx=sorted(set(chosen))
    elif KIT_MODE=='chem':
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(k[3]),2,2048) for k in kit]
        chosen=[0]
        while len(chosen)<KIT_SIZE:
            best=-1.0; bi=None
            for i in range(len(kit)):
                if i in chosen: continue
                md=min(1.0-DataStructs.TanimotoSimilarity(fps[i],fps[c]) for c in chosen)
                if md>best: best=md; bi=i
            chosen.append(bi)
        idx=sorted(set(chosen))
    else:
        idx=rng.choice(len(kit),KIT_SIZE,replace=False)
    kit=[kit[i] for i in idx]
# ladder from kit (RI->sec)
agg=defaultdict(list)
for d,sec,ri,_sm in kit: agg[round(ri,1)].append(sec)
xs=np.array(sorted(agg)); ys=np.array([np.median(agg[x]) for x in xs])
inv=PchipInterpolator(xs,ys,extrapolate=True) if len(xs)>=3 else (lambda r: np.full_like(np.asarray(r,float),np.median(ys)))
kitX=np.array([d for d,_,_,_ in kit]); kitY=np.array([sec for _,sec,_,_ in kit])
import pickle as _pk, csv as _csvu
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
        comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ri=float(c["ri"]),ik=(c.get("ik14") or "")[:14]))
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
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
mid=np.array([(ik2id.get(c["ik"]) if len(c["ik"])>=14 and c["ik"] in ik2id else name2id.get(c["name"],-1)) if (c["name"] in maf or (len(c["ik"])>=14 and c["ik"] in maf_ik)) else -1 for c in comp])
inmaf=mid>=0; nmaf=NMAF; N=len(MZML); DDIM=len(_descriptors("CCO"))
import time as _t
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
        CAP=int(__import__("os").environ.get("PEAK_CAP","200"))   # keep top-CAP peaks per m/z per file (memory bound; propagation only needs strong peaks)
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
peaks=[[upeaks[grp[k]][inj] for inj in range(len(MZML))] for k in range(n)]
grng=np.random.RandomState(0); RC=[grng.uniform(lo+TIGHT,hi-TIGHT,NRAND) for lo,hi in rtspans]
_gstr=np.zeros(n); gapex=np.full(n,np.nan); strength=np.zeros(n)
for k in range(n):
    best=0; ba=np.nan
    for inj in range(N):
        P=peaks[k][inj]
        if len(P):
            j=P[:,1].argmax()
            if P[j,1]>best: best=P[j,1]; ba=P[j,0]
    _gstr[k]=best; strength[k]=best; gapex[k]=ba
nullc=np.zeros(n)
for k in range(n):
    thr=0.5*_gstr[k]; hit=0; tot=0
    for inj in range(N):
        P=peaks[k][inj]
        for c in RC[inj]:
            tot+=1
            if len(P):
                m=np.abs(P[:,0]-c)<=TIGHT
                if m.any() and P[m,1].max()>=thr: hit+=1
    nullc[k]=hit/max(tot,1)
print("nullc done",flush=True)
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
withdesc=[k for k in range(n) if comp[k]["desc"] is not None]; DX=np.array([comp[k]["desc"] for k in withdesc])

import os as _om2
MS2MODE=_om2.environ.get("MS2MODE","vanilla")  # vanilla|gate|tiebreak|both
TAU=float(_om2.environ.get("TAU","0.5")); DDA_RT_W=float(_om2.environ.get("DDA_RT_W","12"))
ms2ent=np.full(n,np.nan); ms2hits=[[] for _ in range(n)]
def ms2_at(k,rt):  # best MS2 similarity of candidate k's ref to an observed DDA spectrum near this peak RT
    best=0.0
    for drt,sim in ms2hits[k]:
        if abs(drt-rt)<=DDA_RT_W and sim>best: best=sim
    return best
_POS=["ms2_library_massbank_full_pos.json","ms2_library_mona_pos.json","ms2_library_gnps_pos.json"]
_MS2LIBS={"Method3":["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"],"Method1":_POS,"Method2":_POS,"Method4":_POS}
if MS2MODE!="vanilla":
    import json as _json
    from squid_inc.features.ms2_similarity import entropy_similarity as _ent
    _our=set(comp[i]["ik"] for i in range(n) if comp[i]["ik"])
    _ref={}
    for _fn in _MS2LIBS.get(_meth,_MS2LIBS["Method3"]):
        try: _d=_json.load(open("/mnt/volume-hel1-1/data/processed/"+_fn))
        except Exception: continue
        for _k,_v in _d.items():
            _ik=_k[:14]
            if _ik in _our: _ref.setdefault(_ik,[]).append([(float(a),float(b)) for a,b in _v])
    _ms2=[]
    for _mp in MZML:
        for _sp in pymzml.run.Reader(_mp):
            if _sp.ms_level!=2: continue
            try: _p=float(_sp.selected_precursors[0]["mz"])
            except Exception: continue
            _rt=_sp.scan_time_in_minutes()*60
            _mz=np.asarray(_sp.mz); _ii=np.asarray(_sp.i)
            if len(_mz): _ms2.append((_p,_rt,[(float(a),float(b)) for a,b in zip(_mz.tolist(),_ii.tolist())]))
    _pm=np.array([x[0] for x in _ms2]); _po=np.argsort(_pm); _pm=_pm[_po]; _ms2=[_ms2[i] for i in _po]
    for k in range(n):
        ik=comp[k]["ik"]
        if ik not in _ref: continue
        t=MZc[k]*15e-6; lo=np.searchsorted(_pm,MZc[k]-t); hi=np.searchsorted(_pm,MZc[k]+t)
        if hi<=lo: continue
        hits=[]
        for j in range(lo,hi):
            _,drt,pk=_ms2[j]
            sim=max((_ent(pk,r,mz_tol=0.02) for r in _ref[ik]),default=0.0)
            hits.append((drt,sim))
        ms2hits[k]=hits
        if hits: ms2ent[k]=max(s for _,s in hits)
    print("MS2MODE=%s tau=%.2f | candidates w/ ref MS2: %d | w/ ref+obs(ms2ent): %d"%(
        MS2MODE,TAU,sum(1 for k in range(n) if comp[k]["ik"] in _ref),int((~np.isnan(ms2ent)).sum())),flush=True)

import os as _ob
if _ob.environ.get("SEED")=="bootstrap":
    TAU_SEED=float(_ob.environ.get("TAU_SEED","0.6"))
    _mzs=np.sort(MZc)
    def _massuniq(k):
        t=MZc[k]*ISOBAR_PPM*1e-6; lo=np.searchsorted(_mzs,MZc[k]-t); hi=np.searchsorted(_mzs,MZc[k]+t); return (hi-lo)==1
    seed=[k for k in range(n) if comp[k]["desc"] is not None and (not np.isnan(ms2ent[k])) and ms2ent[k]>=TAU_SEED and ms2hits[k] and _massuniq(k)]
    seedrt={k:max(ms2hits[k],key=lambda x:x[1])[0] for k in seed}   # observed RT where the confident MS2 matched
    nmu=sum(1 for k in range(n) if comp[k]["desc"] is not None and _massuniq(k))
    nms2=int((~np.isnan(ms2ent)).sum())
    print("BOOTSTRAP discovery: mass-unique=%d, MS2-scored=%d, confident seeds (unique AND MS2>=%.2f)=%d"%(nmu,nms2,TAU_SEED,len(seed)),flush=True)
    if len(seed)>=3:
        _ag=defaultdict(list)
        for k in seed: _ag[round(comp[k]["ri"],1)].append(seedrt[k])
        _xs=np.array(sorted(_ag)); _ys=np.array([np.median(_ag[x]) for x in _xs])
        _u,_ui=np.unique(_xs,return_index=True)
        if len(_u)>=3:
            inv=PchipInterpolator(_u,_ys[_ui],extrapolate=True)   # override ladder
            for k in range(n): comp[k]["pred"]=float(inv(comp[k]["ri"]))
        kitX=np.array([comp[k]["desc"] for k in seed]); kitY=np.array([seedrt[k] for k in seed])   # override structure seed
        print("BOOTSTRAP: ladder from %d RI points, structure model from %d anchors (NO spike-in kit)"%(len(_u),len(seed)),flush=True)
        if _ob.environ.get("NORI"):
            _m=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=3).fit(kitX,kitY)
            for _k in range(n):
                if comp[_k]["desc"] is not None: comp[_k]["pred"]=float(_m.predict(np.asarray(comp[_k]["desc"])[None,:])[0])
            print("NORI: comp pred from structure model trained on discovered anchors (NO library RI)",flush=True)
    else:
        print("BOOTSTRAP: too few confident seeds (%d) -- falling back to kit"%len(seed),flush=True)

def run_chain(cs):
    r=np.random.RandomState(cs); fsub=np.sort(r.choice(DDIM,NFEAT,replace=False))
    bidx=r.choice(len(kitX),len(kitX),replace=True)
    Xtr=list(kitX[bidx][:,fsub]); ytr=list(kitY[bidx]); admitted=np.zeros(n,bool)
    import os as _o2
    for rd in range(int(_o2.environ.get("ROUNDS","10"))):
        mdl=HistGradientBoostingRegressor(max_iter=250,max_depth=4,learning_rate=0.07,min_samples_leaf=4).fit(np.array(Xtr),np.array(ytr))
        pr=mdl.predict(DX[:,fsub]); cand=[]
        for ii,k in enumerate(withdesc):
            if admitted[k]: continue
            nr,ap=rep_apex(k,pr[ii],TIGHT)
            if nr<MINREP or ap is None: continue
            if binom.sf(nr-1,N,np.clip(nullc[k],1e-6,0.999))>FDR_ADMIT: continue
            cand.append((k,ap,abs(ap-pr[ii])))
        if not cand: break
        cand.sort(key=lambda x:x[2]); new=[]; taken=[]
        for k,ap,dp in cand:
            if any(abs(MZc[k]-MZc[kk])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(ap-aa)<=TIGHT for kk,aa in taken): continue
            taken.append((k,ap)); new.append((k,ap))
        if not new: break
        for k,ap in new:
            admitted[k]=True
            if MS2MODE in ("gate","both") and not (ms2ent[k]>=TAU): continue  # confirm-required: train only on MS2-confirmed admits (no-MS2 ZINC blocked)
            Xtr.append(comp[k]["desc"][fsub]); ytr.append(ap)
    return admitted
votes=np.zeros(n)
for c in range(K):
    _c0=_t.time(); votes+=run_chain(c); print("chain %d done %.0fs admitted=%d"%(c,_t.time()-_c0,int((votes>0).sum())),flush=True)
def cluster(selmask):
    idx=[k for k in np.where(selmask)[0] if not np.isnan(gapex[k])]; cls=[]
    for k in sorted(idx,key=lambda k:-strength[k]):
        for cl in cls:
            j=cl[0]
            if abs(MZc[k]-MZc[j])<=MZc[k]*ISOBAR_PPM*1e-6 and abs(gapex[k]-gapex[j])<=TIGHT: cl.append(k); break
        else: cls.append([k])
    return cls
# cardinality recovery for recall
order=np.argsort(MZc); groups=[]; cur=[int(order[0])]
for k in order[1:]:
    k=int(k)
    if abs(MZc[k]-MZc[cur[-1]])<=MZc[k]*ISOBAR_PPM*1e-6: cur.append(k)
    else: groups.append(cur); cur=[k]
groups.append(cur)
def card_recall_set(base_recovered):
    cr=set(base_recovered)
    for g in groups:
        mm=[k for k in g if inmaf[k]]
        if not mm: continue
        allpk=[(rt_,it) for k in g for inj in range(N) for rt_,it in peaks[k][inj] if it>2*FLOOR]
        if not allpk: continue
        rtb=defaultdict(list)
        for rt_,it in allpk: rtb[int(rt_//(2*TIGHT))].append((rt_,it))
        dist=[(np.median([r for r,_ in v]),max(i for _,i in v)) for b,v in rtb.items() if len(v)>=MINREP]
        if not dist: continue
        gg=list(g)  # ALL DD compounds at this m/z (blind candidate set, not just MAF)
        prt=np.array([comp[k]["pred"] for k in gg]); pkrt=np.array([d[0] for d in dist])
        rt_cost=np.abs(prt[:,None]-pkrt[None,:]); cost=rt_cost.copy()
        if MS2MODE in ("tiebreak","both"):
            for ai in range(len(gg)):
                if not ms2hits[gg[ai]]: continue
                for bi in range(len(pkrt)):
                    sim=ms2_at(gg[ai],pkrt[bi])
                    if sim>0: cost[ai,bi]=rt_cost[ai,bi]-2*TIGHT*sim  # MS2 match discounts -> wins the slot
        ri,ci=linear_sum_assignment(cost)
        for a,b in zip(ri,ci):
            if rt_cost[a,b]<=2*TIGHT and inmaf[gg[a]]: cr.add(mid[gg[a]])  # RT still gates; MS2 only decides who gets the peak
    return cr
def card_recall(b): return len(card_recall_set(b))/nmaf
print(f"KIT_SIZE={KIT_SIZE if KIT_SIZE else len(kit)} KIT_SEED={KIT_SEED} K={K} (kit used {len(kit)})")
print(f"{'M':>3}{'clusters':>9}{'recall':>8}{'precision':>11}{'+card_recall':>13}")
for Mv in range(1,K+1):
    cls=cluster(votes>=Mv)
    if not cls: continue
    cov=set(mid[k] for cl in cls for k in cl if inmaf[k])
    tp=sum(any(inmaf[k] for k in cl) for cl in cls)
    corr=sum(any(inmaf[k] or comp[k]["ik"] in full_maf_ik for k in cl) for cl in cls)  # xplat
    cr=card_recall(cov)
    print(f"{Mv:>3}{len(cls):>9}{len(cov)/nmaf:>8.3f}{corr/len(cls):>11.3f}{cr:>13.3f}")

import os
# H2H: always emit recall / closed-world prec / MAF-only prec at M=1
_cls1=cluster(votes>=1); _cov=set(mid[k] for cl in _cls1 for k in cl if inmaf[k]); _rec=len(card_recall_set(_cov))/nmaf
_negmaf=[k for k in range(n) if inmaf[k]]; _cor=0; _sub=0; _nov=0
for cl in _cls1:
    if any(inmaf[k] or comp[k]["ik"] in full_maf_ik for k in cl): _cor+=1; continue
    kk=max(cl,key=lambda k:strength[k]); ap=gapex[kk]; mz=MZc[kk]
    if any(abs(MZc[j]-mz)<=mz*ISOBAR_PPM*1e-6 and abs(comp[j]["pred"]-ap)<=TIGHT for j in _negmaf): _sub+=1
    else: _nov+=1
print("H2H %s | %s | MS2=%s | n=%d nmaf=%d | recall(card)=%.3f closed-world-prec=%.3f MAF-only-prec=%.3f (correct=%d subst=%d novel=%d)"%(
    PLATFORM,LIBRARY,MS2MODE,n,nmaf,_rec,_cor/max(len(_cls1),1),_cor/max(_cor+_sub,1),_cor,_sub,_nov),flush=True)
if os.environ.get("REPORT"):
    _rp=os.environ["REPORT"]; _cls=cluster(votes>=1)
    _f=open(_rp,"w"); _f.write("category\trep_ik\tmz\tapex_sec\tlog10_int\tnrep\tn_in_cluster\tms2_sim\n")
    _par=_cov=_oos=0
    for cl in _cls:
        kk=max(cl,key=lambda k:strength[k]); ik=comp[kk]["ik"]
        inp=any(inmaf[k] for k in cl); xpl=any((comp[k]["ik"] in full_maf_ik) for k in cl)
        nr,_=rep_apex(kk,comp[kk]["pred"],TIGHT)
        ap=gapex[kk]; ms2s=ms2_at(kk,ap) if not np.isnan(ap) else 0.0
        if inp: cat="parity"; _par+=1
        elif xpl: cat="coverage_win_xplat"; _cov+=1
        else: cat="out_of_scope"; _oos+=1
        _f.write("%s\t%s\t%.4f\t%.1f\t%.2f\t%d\t%d\t%.3f\n"%(cat,ik,MZc[kk],ap if not np.isnan(ap) else -1,np.log10(strength[kk]+1),nr,len(cl),ms2s))
    _f.close()
    print("REPORT %s|%s: parity=%d coverage-win(xplat)=%d out-of-scope-evidenced=%d -> %s"%(PLATFORM,LIBRARY,_par,_cov,_oos,_rp),flush=True)
if os.environ.get('DUMP'):
    cls1=cluster(votes>=1); cov1=set(mid[k] for cl in cls1 for k in cl if inmaf[k])
    crset=card_recall_set(cov1); seen=set()
    f=open(os.environ['DUMP'],'w'); f.write('mid\tmz\tpred\tlogI\tat_pred\thas_peak\trecovered\n')
    for k in range(n):
        if inmaf[k] and mid[k] not in seen:
            seen.add(mid[k]); nr,_=rep_apex(k,comp[k]['pred'],TIGHT)
            f.write('%d\t%.4f\t%.0f\t%.1f\t%d\t%d\t%d\n'%(mid[k],MZc[k],comp[k]['pred'],np.log10(strength[k]+1),int(nr>=MINREP),int(strength[k]>2*FLOOR),int(mid[k] in crset)))
    f.close()

import os as _orc
if _orc.environ.get("ROOTCAUSE"):
    from collections import Counter as _C
    cls1=cluster(votes>=1); cov=set(mid[k] for cl in cls1 for k in cl if inmaf[k]); crset=card_recall_set(cov)
    reps={}
    for k in range(n):
        if inmaf[k] and (mid[k] not in reps or strength[k]>strength[reps[mid[k]]]): reps[mid[k]]=k
    fn=[m for m in reps if m not in crset]
    cats=_C(); ex=_C()
    for m in fn:
        k=reps[m]; sm=comp[k]["desc"] is not None
        nr,_=rep_apex(k,comp[k]["pred"],TIGHT); anyp=strength[k]>2*FLOOR
        iso=sum(1 for j in range(n) if inmaf[j] and abs(MZc[j]-MZc[k])<=MZc[k]*ISOBAR_PPM*1e-6)
        if not anyp:
            cats["NO PEAK at m/z (acquisition floor / absent)"+("" if sm else " [no-SMILES]")]+=1
        elif nr>=MINREP:
            cats["peak AT predicted-RT, not admitted (gate / cardinality miss)"+(" [iso%d]"%iso if iso>=2 else "")]+=1
        else:
            cats["RT-MISLOCATED (peak elsewhere, prediction wrong)"+("" if sm else " [no-SMILES]")+(" [iso%d]"%iso if iso>=2 else "")]+=1
    tot=sum(cats.values())
    print("\n===== DEFINITIVE FN ROOT-CAUSE (kit=%d, recall=%.3f, %d FNs) ====="%(len(kit),len(crset)/nmaf,tot))
    for c,v in cats.most_common(): print("  %3d (%.2f)  %s"%(v,v/tot,c))
    # rollups
    nopk=sum(v for c,v in cats.items() if "NO PEAK" in c)
    mis=sum(v for c,v in cats.items() if "MISLOCATED" in c)
    adm=sum(v for c,v in cats.items() if "AT predicted" in c)
    nosm=sum(v for c,v in cats.items() if "no-SMILES" in c)
    print("  --- rollup: no-peak %d (%.2f) | mislocated %d (%.2f) | admission-residual %d (%.2f) | (no-SMILES cross-cut %d)"%(nopk,nopk/tot,mis,mis/tot,adm,adm/tot,nosm))

if _osf.environ.get("IDPREC"):
    cls1=cluster(votes>=1); cov=set(mid[k] for cl in cls1 for k in cl if inmaf[k]); rec=len(card_recall_set(cov))/nmaf
    negmaf=[k for k in range(n) if inmaf[k]]
    correct=0; subst=0; novel=0
    for cl in cls1:
        if any(inmaf[k] or comp[k]["ik"] in full_maf_ik for k in cl): correct+=1; continue
        kk=max(cl,key=lambda k:strength[k]); ap=gapex[kk]; mz=MZc[kk]
        if any(abs(MZc[j]-mz)<=mz*ISOBAR_PPM*1e-6 and abs(comp[j]["pred"]-ap)<=TIGHT for j in negmaf): subst+=1
        else: novel+=1
    print("FLOOR=%d  recall(card)=%.3f  closed-world-prec=%.3f  IDENTITY-prec=%.3f  (correct=%d subst=%d novel-excluded=%d)"%(FLOOR,rec,correct/max(len(cls1),1),correct/max(correct+subst,1),correct,subst,novel))

if _osf.environ.get("SENSE"):
    from scipy.stats import spearmanr
    cls1=cluster(votes>=1); cov=set(mid[k] for cl in cls1 for k in cl if inmaf[k]); crset=card_recall_set(cov)
    reps={}
    for k in range(n):
        if inmaf[k] and (mid[k] not in reps or strength[k]>strength[reps[mid[k]]]): reps[mid[k]]=k
    ours=[]; mafr=[]; preds=[]
    for m in crset:
        if m in reps and m in maf_rt and not _math.isnan(maf_rt.get(m,float("nan"))) and not np.isnan(gapex[reps[m]]):
            ours.append(float(gapex[reps[m]])); mafr.append(maf_rt[m]); preds.append(float(comp[reps[m]]["pred"]))
    if len(ours)>5:
        rho=spearmanr(ours,mafr)[0]; rho_pred=spearmanr(preds,mafr)[0]
        print("SENSE-CHECK (recovered=%d):"%len(ours))
        print("  Spearman(OUR observed peak-sec, MAF reported rt) = %.3f   <- recoveries real if high"%rho)
        print("  Spearman(our PREDICTED sec, MAF reported rt)      = %.3f   (model vs ground-truth)"%rho_pred)
        # examples sorted by MAF rt
        idx=sorted(range(len(ours)),key=lambda i:mafr[i])
        print("  examples (MAF_rt, our_peak_sec, our_pred_sec):")
        for i in idx[::max(1,len(idx)//10)][:10]:
            print("    %.1f  %.0f  %.0f"%(mafr[i],ours[i],preds[i]))
        # FN side: do we miss across the whole rt range or specific regions?
        fnrt=[maf_rt[m] for m in reps if m not in crset and m in maf_rt and not _math.isnan(maf_rt.get(m,float("nan")))]
        import numpy as _np
        print("  recovered MAF_rt: median %.0f range [%.0f,%.0f] | missed MAF_rt: n=%d median %.0f"%(_np.median(mafr),min(mafr),max(mafr),len(fnrt),_np.median(fnrt) if fnrt else -1))
