"""forest_sweep_ms2.py -> forest_sweep_bootstrap.py: SEED=bootstrap mode.
Instead of the spike-in kit, discover the seed from the data: candidates that are MASS-UNIQUE in the
library (no isobaric competitor) AND have a high-confidence reference-MS2 match. Their (RI, observed-RT)
build the ladder and (descriptors, observed-RT) seed the structure model; then propagate as usual."""
s=open("/tmp/forest_sweep_ms2.py").read()

# 1) carry library RI on each candidate (needed to rebuild the ladder from discovered seeds)
old='''        s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
        comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=(c.get("ik14") or "")[:14]))'''
new='''        s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
        comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ri=float(c["ri"]),ik=(c.get("ik14") or "")[:14]))'''
assert old in s; s=s.replace(old,new)

# 2) bootstrap-seed block: rebuild ladder + structure seed from discovered anchors (before run_chain)
anchor="def run_chain(cs):"
boot='''import os as _ob
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
    else:
        print("BOOTSTRAP: too few confident seeds (%d) -- falling back to kit"%len(seed),flush=True)

def run_chain(cs):'''
assert anchor in s; s=s.replace(anchor,boot,1)

# 3) tag the H2H line with the seed mode
s=s.replace('print("H2H %s | %s | MS2=%s |','print("H2H %s | %s | SEED="+_ob.environ.get("SEED","kit")+" | MS2=%s |' if False else 'print("H2H %s | %s | MS2=%s |')

open("/tmp/forest_sweep_bootstrap.py","w").write(s)
import ast; ast.parse(s); print("forest_sweep_bootstrap.py built OK, lines:",s.count(chr(10)))
