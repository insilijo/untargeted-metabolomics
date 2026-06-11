s=open("/tmp/forest_sweep_bootstrap.py").read()
# (1) pos MS2 libs for Method1/2/4
old='_MS2LIBS={"Method3":["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"]}'
new=('_POS=["ms2_library_massbank_full_pos.json","ms2_library_mona_pos.json","ms2_library_gnps_pos.json"]\n'
     '_MS2LIBS={"Method3":["ms2_library_massbank_full_neg.json","ms2_library_mona_neg.json","ms2_library_gnps_neg.json"],'
     '"Method1":_POS,"Method2":_POS,"Method4":_POS}')
assert old in s, "ms2libs anchor missing"; s=s.replace(old,new)
# (2) NORI: structure-model pred (no RI ladder), appended in the bootstrap block
anc='        print("BOOTSTRAP: ladder from %d RI points, structure model from %d anchors (NO spike-in kit)"%(len(_u),len(seed)),flush=True)'
nori_lines=[
 anc,
 '        if _ob.environ.get("NORI"):',
 '            _m=HistGradientBoostingRegressor(max_iter=300,max_depth=4,learning_rate=0.06,min_samples_leaf=3).fit(kitX,kitY)',
 '            for _k in range(n):',
 '                if comp[_k]["desc"] is not None: comp[_k]["pred"]=float(_m.predict(np.asarray(comp[_k]["desc"])[None,:])[0])',
 '            print("NORI: comp pred from structure model trained on discovered anchors (NO library RI)",flush=True)',
]
assert anc in s, "nori anchor missing"; s=s.replace(anc,"\n".join(nori_lines))
open("/tmp/forest_sweep_bootstrap.py","w").write(s)
import ast; ast.parse(s); print("patched: pos MS2 libs + NORI mode, lines:",s.count(chr(10)))
