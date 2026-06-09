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
PLAT="lc/ms neg"; DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"; FEAT="/tmp/feat_colu.parquet"
KIT="/root/untargeted-metabolomics/data/anchor_panels/anchors_lc_ms_neg.csv"; SMI="/tmp/dd_pubchem_smiles.csv"
MZML=sorted(glob.glob("/root/SQuID-INC/data/st004581/mzml/Method3_*COLU*.mzML"))[:8]
MZ_PPM=7.0; FLOOR=50000.0; TIGHT=6.0; MINREP=2; NRAND=60; FDR_ADMIT=0.05; ISOBAR_PPM=10.0; NFEAT=15; ik14=lambda s:(s or "")[:14]
maf=set(); maf_ik=set(); name2id={}; ik2id={}; _eid=0; full_maf_ik=set()
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
NMAF=_eid
df=pd.read_parquet(FEAT); df["platform"]=df.source_file.str.split("_").str[0].map(M.DEFAULT_PREFIX_MAP)
df=df.dropna(subset=["platform"])
lib0=M.load_library(Path(DD)); smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
# ---- KIT (subsampled) drives ladder + seed ----
kit=[]
for r in csv.DictReader(open(KIT)):
    s=r.get("smiles"); d=_descriptors(s) if s else None
    try: sec=float(r["observed_rt_sec"]); ri=float(r["ri"])
    except: sec=ri=None
    if d is not None and sec and ri: kit.append((d,sec,ri))
rng=np.random.RandomState(KIT_SEED)
import os as _os
KIT_MODE=_os.environ.get("KIT_MODE","random")
if KIT_SIZE and KIT_SIZE<len(kit):
    if KIT_MODE=="spread":
        ks=sorted(range(len(kit)),key=lambda i:kit[i][1])  # sort by observed RT (sec)
        idx=sorted(set(ks[int(round(j*(len(ks)-1)/(KIT_SIZE-1)))] for j in range(KIT_SIZE)))
    else:
        idx=rng.choice(len(kit),KIT_SIZE,replace=False)
    kit=[kit[i] for i in idx]
# ladder from kit (RI->sec)
agg=defaultdict(list)
for d,sec,ri in kit: agg[round(ri,1)].append(sec)
xs=np.array(sorted(agg)); ys=np.array([np.median(agg[x]) for x in xs])
inv=PchipInterpolator(xs,ys,extrapolate=True) if len(xs)>=3 else (lambda r: np.full_like(np.asarray(r,float),np.median(ys)))
kitX=np.array([d for d,_,_ in kit]); kitY=np.array([s for _,s,_ in kit])
comp=[]
for c in lib0.get(PLAT,[]):
    nm=M._norm(c["name"])
    if not nm: continue
    s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
    comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=(c.get("ik14") or "")[:14]))
n=len(comp); MZc=np.array([c["mz"] for c in comp]); tol=MZc*MZ_PPM*1e-6
mid=np.array([(ik2id.get(c["ik"]) if len(c["ik"])>=14 and c["ik"] in ik2id else name2id.get(c["name"],-1)) if (c["name"] in maf or (len(c["ik"])>=14 and c["ik"] in maf_ik)) else -1 for c in comp])
inmaf=mid>=0; nmaf=NMAF; N=len(MZML); DDIM=len(_descriptors("CCO"))
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
import time as _t;_T0=_t.time();print("EIC done %.0fs"%(_t.time()-_T0),flush=True)
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
def run_chain(cs):
    r=np.random.RandomState(cs); fsub=np.sort(r.choice(DDIM,NFEAT,replace=False))
    bidx=r.choice(len(kitX),len(kitX),replace=True)
    Xtr=list(kitX[bidx][:,fsub]); ytr=list(kitY[bidx]); admitted=np.zeros(n,bool)
    for rd in range(10):
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
        for k,ap in new: admitted[k]=True; Xtr.append(comp[k]["desc"][fsub]); ytr.append(ap)
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
        prt=np.array([comp[k]["pred"] for k in mm]); pkrt=np.array([d[0] for d in dist])
        cost=np.abs(prt[:,None]-pkrt[None,:]); ri,ci=linear_sum_assignment(cost)
        for a,b in zip(ri,ci):
            if cost[a,b]<=2*TIGHT: cr.add(mid[mm[a]])
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
if os.environ.get('DUMP'):
    cls1=cluster(votes>=1); cov1=set(mid[k] for cl in cls1 for k in cl if inmaf[k])
    crset=card_recall_set(cov1); seen=set()
    f=open(os.environ['DUMP'],'w'); f.write('mid\tmz\tpred\tlogI\tat_pred\thas_peak\trecovered\n')
    for k in range(n):
        if inmaf[k] and mid[k] not in seen:
            seen.add(mid[k]); nr,_=rep_apex(k,comp[k]['pred'],TIGHT)
            f.write('%d\t%.4f\t%.0f\t%.1f\t%d\t%d\t%d\n'%(mid[k],MZc[k],comp[k]['pred'],np.log10(strength[k]+1),int(nr>=MINREP),int(strength[k]>2*FLOOR),int(mid[k] in crset)))
    f.close()
