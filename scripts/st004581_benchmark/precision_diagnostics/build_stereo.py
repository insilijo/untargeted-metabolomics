"""forest_sweep_h2h.py -> forest_sweep_stereo.py: stereo/geometric-isomer-aware scoring.
Entity key = FULL InChIKey (differentiates fumarate/maleate etc.; synonyms keep same full key).
DD candidates carry full InChIKey + their distinct RI -> cardinality assigns each isomer to its own
peak via the RI-based prediction. Adds an isomer-pair recovery report."""
s=open("/tmp/forest_sweep_h2h.py").read()

# ---- 1: MAF entity key ik14 -> full InChIKey ----
P1_old='''    _ik=(r.get("inchikey") or "").strip()[:14]
    if len(_ik)>=14: full_maf_ik.add(_ik)
    if (r.get("platform") or "").strip().lower()==PLAT:
        nm=M._norm(r.get("name") or "")
        maf.add(nm)
        if len(_ik)>=14 and _ik in ik2id: eid=ik2id[_ik]
        elif nm in name2id: eid=name2id[nm]
        else: eid=_eid; _eid+=1
        name2id.setdefault(nm,eid)
        if len(_ik)>=14: maf_ik.add(_ik); ik2id.setdefault(_ik,eid)'''
P1_new='''    _ik=(r.get("inchikey") or "").strip()          # FULL InChIKey (stereo-aware)
    if len(_ik)>=20: full_maf_ik.add(_ik)
    if (r.get("platform") or "").strip().lower()==PLAT:
        nm=M._norm(r.get("name") or "")
        maf.add(nm)
        if len(_ik)>=20 and _ik in ik2id: eid=ik2id[_ik]
        elif nm in name2id: eid=name2id[nm]
        else: eid=_eid; _eid+=1
        name2id.setdefault(nm,eid)
        if len(_ik)>=20: maf_ik.add(_ik); ik2id.setdefault(_ik,eid)'''
assert P1_old in s; s=s.replace(P1_old,P1_new)

# ---- 2: full-InChIKey DD loader (alongside lib0 which stays for chemfn kit mode) ----
P2_old='''lib0=M.load_library(Path(DD)); smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}'''
P2_new='''lib0=M.load_library(Path(DD)); smi={ik14(r["inchikey"]):r["smiles"] for r in csv.DictReader(open(SMI)) if r.get("smiles")}
from collections import defaultdict as _dd
DDFULL=_dd(list)  # platform -> [{name,mz,ri,ik(full)}]  (full InChIKey retained)
for _r in csv.DictReader(open(DD,encoding="utf-8-sig")):
    _P=(_r.get("PLATFORM") or "").strip().lower()
    try: _mz=float(_r.get("MASS")); _ri=float(_r.get("RI"))
    except (TypeError,ValueError): continue
    if _mz<=0 or _ri<=0: continue
    DDFULL[_P].append(dict(name=_r.get("BIOCHEMICAL") or "",mz=_mz,ri=_ri,ik=(_r.get("INCHIKEY") or "").strip()))'''
assert P2_old in s; s=s.replace(P2_old,P2_new)

# ---- 3: DD candidate branch uses full InChIKey ----
P3_old='''    for c in lib0.get(PLAT,[]):
        nm=M._norm(c["name"])
        if not nm: continue
        s=smi.get(c["ik14"]); d=_descriptors(s) if s else None
        comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=(c.get("ik14") or "")[:14]))'''
P3_new='''    for c in DDFULL.get(PLAT,[]):
        nm=M._norm(c["name"])
        if not nm: continue
        s=smi.get(ik14(c["ik"])); d=_descriptors(s) if s else None
        comp.append(dict(name=nm,mz=c["mz"],desc=(np.array(d) if d is not None else None),pred=float(inv(c["ri"])),ik=c["ik"]))'''
assert P3_old in s; s=s.replace(P3_old,P3_new)

# ---- 4: mid matches by full InChIKey ----
P4_old='''mid=np.array([(ik2id.get(c["ik"]) if len(c["ik"])>=14 and c["ik"] in ik2id else name2id.get(c["name"],-1)) if (c["name"] in maf or (len(c["ik"])>=14 and c["ik"] in maf_ik)) else -1 for c in comp])'''
P4_new='''mid=np.array([(ik2id.get(c["ik"]) if len(c["ik"])>=20 and c["ik"] in ik2id else name2id.get(c["name"],-1)) if (c["name"] in maf or (len(c["ik"])>=20 and c["ik"] in maf_ik)) else -1 for c in comp])'''
assert P4_old in s; s=s.replace(P4_old,P4_new)

# ---- 5: isomer-pair recovery report (appended before DUMP/REPORT blocks) ----
P5_anchor="if os.environ.get('DUMP'):"
P5_block='''if True:
    # isomer-pair recovery: MAF entities sharing an InChIKey14 skeleton (stereo/geometric isomers)
    from collections import defaultdict as _dd2
    id2ik={}; id2nm={}
    for _r in csv.DictReader(open(GT)):
        if _r.get("unannotatable")=="true" or (_r.get("platform") or "").strip().lower()!=PLAT: continue
        _fik=(_r.get("inchikey") or "").strip(); _nm=M._norm(_r.get("name") or "")
        _e=ik2id.get(_fik) if len(_fik)>=20 and _fik in ik2id else name2id.get(_nm)
        if _e is not None: id2ik[_e]=_fik; id2nm[_e]=_r.get("name")
    sk=_dd2(list)
    for _e,_fik in id2ik.items():
        if len(_fik)>=20: sk[_fik[:14]].append(_e)
    pairs={k:v for k,v in sk.items() if len(v)>=2}
    _cls1=cluster(votes>=1); _rec=card_recall_set(set(mid[k] for cl in _cls1 for k in cl if inmaf[k]))
    both=one=none=0
    for k,ents in pairs.items():
        got=sum(1 for e in ents if e in _rec)
        if got==len(ents): both+=1
        elif got>0: one+=1
        else: none+=1
    npair_ent=sum(len(v) for v in pairs.values())
    print("ISOMER-PAIRS %s: %d skeletons w/ >=2 isomers (%d distinct compounds) | recovered ALL=%d, SOME=%d, NONE=%d"%(
        PLATFORM,len(pairs),npair_ent,both,one,none),flush=True)
if os.environ.get('DUMP'):'''
assert P5_anchor in s; s=s.replace(P5_anchor,P5_block)

open("/tmp/forest_sweep_stereo.py","w").write(s)
import ast; ast.parse(s); print("forest_sweep_stereo.py built OK, lines:",s.count(chr(10)))
