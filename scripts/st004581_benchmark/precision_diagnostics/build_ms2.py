"""forest_sweep_h2h.py -> forest_sweep_ms2.py: 4-arm MS2 ablation (MS2MODE=vanilla|gate|tiebreak|both).
 - gate: confirm-required training (only MS2-confirmed admits become RT anchors; blocks no-MS2 ZINC decoys)
 - tiebreak: MS2-aware cardinality cost (award contested peak to the MS2-matching candidate, not closest-RT)"""
s=open("/tmp/forest_sweep_h2h.py").read()

# ---- PATCH 1: replace MS2GATE block with MS2MODE block (observed MS2 w/ RT + ms2_at tiebreak fn) ----
P1_old='''import os as _om2
MS2GATE=_om2.environ.get("MS2GATE")  # tau threshold or None
ms2ent=np.full(n,np.nan)
if MS2GATE:
    import json as _json
    from squid_inc.features.ms2_similarity import entropy_similarity as _ent
    _refL={}
    for _fn in ["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"]:
        try:
            _d=_json.load(open("/mnt/volume-hel1-1/data/processed/"+_fn))
            for _k,_v in _d.items():
                _ik=_k[:14]
                if any(comp[i]["ik"]==_ik for i in range(0)): pass
        except Exception: pass
    _our=set(comp[i]["ik"] for i in range(n) if comp[i]["ik"])
    _ref={}
    for _fn in ["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"]:
        _d=_json.load(open("/mnt/volume-hel1-1/data/processed/"+_fn))
        for _k,_v in _d.items():
            _ik=_k[:14]
            if _ik in _our: _ref.setdefault(_ik,[]).append([(float(a),float(b)) for a,b in _v])
    _ms2=[]
    for _mp in MZML:
        for _sp in pymzml.run.Reader(_mp):
            if _sp.ms_level!=2: continue
            try: _p=_sp.selected_precursors[0]["mz"]
            except Exception: continue
            _mz=np.asarray(_sp.mz); _ii=np.asarray(_sp.i)
            if len(_mz): _ms2.append((float(_p),[(float(a),float(b)) for a,b in zip(_mz.tolist(),_ii.tolist())]))
    _pm=np.array([x[0] for x in _ms2])
    for k in range(n):
        ik=comp[k]["ik"]
        if ik not in _ref: continue
        t=MZc[k]*15e-6; cand=[x for x in _ms2 if abs(x[0]-MZc[k])<=t]
        if not cand: continue
        ms2ent[k]=max((_ent(pk,r,mz_tol=0.02) for _,pk in cand for r in _ref[ik]),default=np.nan)
    print("MS2GATE on: tau=%s, compounds with MS2 ref+obs: %d"%(MS2GATE,int((~np.isnan(ms2ent)).sum())),flush=True)'''
P1_new='''import os as _om2
MS2MODE=_om2.environ.get("MS2MODE","vanilla")  # vanilla|gate|tiebreak|both
TAU=float(_om2.environ.get("TAU","0.5")); DDA_RT_W=float(_om2.environ.get("DDA_RT_W","12"))
ms2ent=np.full(n,np.nan); ms2hits=[[] for _ in range(n)]
def ms2_at(k,rt):  # best MS2 similarity of candidate k's ref to an observed DDA spectrum near this peak RT
    best=0.0
    for drt,sim in ms2hits[k]:
        if abs(drt-rt)<=DDA_RT_W and sim>best: best=sim
    return best
_MS2LIBS={"Method3":["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"]}
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
        MS2MODE,TAU,sum(1 for k in range(n) if comp[k]["ik"] in _ref),int((~np.isnan(ms2ent)).sum())),flush=True)'''
assert P1_old in s; s=s.replace(P1_old,P1_new)

# ---- PATCH 2: run_chain training gate -> confirm-required ----
P2_old='''            if MS2GATE and (not np.isnan(ms2ent[k])) and ms2ent[k]<float(MS2GATE): continue  # MS2-refuted: admit but DON'T train on it'''
P2_new='''            if MS2MODE in ("gate","both") and not (ms2ent[k]>=TAU): continue  # confirm-required: train only on MS2-confirmed admits (no-MS2 ZINC blocked)'''
assert P2_old in s; s=s.replace(P2_old,P2_new)

# ---- PATCH 3: card_recall_set -> MS2-aware cost (assignment), RT still gates admission ----
P3_old='''        gg=list(g)  # ALL DD compounds at this m/z (blind candidate set, not just MAF)
        prt=np.array([comp[k]["pred"] for k in gg]); pkrt=np.array([d[0] for d in dist])
        cost=np.abs(prt[:,None]-pkrt[None,:]); ri,ci=linear_sum_assignment(cost)
        for a,b in zip(ri,ci):
            if cost[a,b]<=2*TIGHT and inmaf[gg[a]]: cr.add(mid[gg[a]])'''
P3_new='''        gg=list(g)  # ALL DD compounds at this m/z (blind candidate set, not just MAF)
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
            if rt_cost[a,b]<=2*TIGHT and inmaf[gg[a]]: cr.add(mid[gg[a]])  # RT still gates; MS2 only decides who gets the peak'''
assert P3_old in s; s=s.replace(P3_old,P3_new)

# ---- PATCH 4: H2H summary line includes MS2MODE ----
P4_old='''print("H2H %s | %s | n=%d nmaf=%d | recall(card)=%.3f closed-world-prec=%.3f MAF-only-prec=%.3f (correct=%d subst=%d novel=%d)"%(
    PLATFORM,LIBRARY,n,nmaf,_rec,_cor/max(len(_cls1),1),_cor/max(_cor+_sub,1),_cor,_sub,_nov),flush=True)'''
P4_new='''print("H2H %s | %s | MS2=%s | n=%d nmaf=%d | recall(card)=%.3f closed-world-prec=%.3f MAF-only-prec=%.3f (correct=%d subst=%d novel=%d)"%(
    PLATFORM,LIBRARY,MS2MODE,n,nmaf,_rec,_cor/max(len(_cls1),1),_cor/max(_cor+_sub,1),_cor,_sub,_nov),flush=True)'''
assert P4_old in s; s=s.replace(P4_old,P4_new)

# ---- PATCH 5: REPORT block (win-vs-Metabolon 4-part output incl. out-of-scope-with-evidence) ----
R_anchor="if os.environ.get('DUMP'):"
R_block='''if os.environ.get("REPORT"):
    _rp=os.environ["REPORT"]; _cls=cluster(votes>=1)
    _f=open(_rp,"w"); _f.write("category\\trep_ik\\tmz\\tapex_sec\\tlog10_int\\tnrep\\tn_in_cluster\\tms2_sim\\n")
    _par=_cov=_oos=0
    for cl in _cls:
        kk=max(cl,key=lambda k:strength[k]); ik=comp[kk]["ik"]
        inp=any(inmaf[k] for k in cl); xpl=any((comp[k]["ik"] in full_maf_ik) for k in cl)
        nr,_=rep_apex(kk,comp[kk]["pred"],TIGHT)
        ap=gapex[kk]; ms2s=ms2_at(kk,ap) if not np.isnan(ap) else 0.0
        if inp: cat="parity"; _par+=1
        elif xpl: cat="coverage_win_xplat"; _cov+=1
        else: cat="out_of_scope"; _oos+=1
        _f.write("%s\\t%s\\t%.4f\\t%.1f\\t%.2f\\t%d\\t%d\\t%.3f\\n"%(cat,ik,MZc[kk],ap if not np.isnan(ap) else -1,np.log10(strength[kk]+1),nr,len(cl),ms2s))
    _f.close()
    print("REPORT %s|%s: parity=%d coverage-win(xplat)=%d out-of-scope-evidenced=%d -> %s"%(PLATFORM,LIBRARY,_par,_cov,_oos,_rp),flush=True)
if os.environ.get('DUMP'):'''
assert R_anchor in s, "DUMP anchor for REPORT splice missing"; s=s.replace(R_anchor,R_block)

open("/tmp/forest_sweep_ms2.py","w").write(s)
import ast; ast.parse(s); print("forest_sweep_ms2.py built OK, lines:",s.count(chr(10)))
