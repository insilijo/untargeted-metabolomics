"""Build a candidate library for sparse_anchor_annotate.py from a structure source -- vendor-free.

Takes a compound list (inchikey, name, optionally smiles) and emits a per-platform candidate library
with COMPUTED m/z over a comprehensive adduct set. m/z comes from:
  1. SMILES -> RDKit ExactMolWt   (preferred), else
  2. lipid-shorthand name parse -> molecular formula -> exact mass   (recovers structureless lipids)

This reproduces (and exceeds) a vendor's observed m/z without any vendor data: on ST004581 the
comprehensive adduct enumeration recovered the DD's observed-m/z recall, and the lipid parser recovers
~50-60% of the structureless lipids whose mass is encoded in the name.

USAGE:  python build_library.py SOURCE_CSV OUT_CSV
  SOURCE_CSV columns (case-insensitive): inchikey, name, optionally smiles
  OUT_CSV: platform,name,inchikey,smiles,mz,adduct,ri   (ri left blank -> structure-RT / quasi-library)

Adduct set (computed from neutral monoisotopic mass M), enumerated per platform:
  neg-relevant: [M-H] [M+Cl] [M+FA-H] [M-H2O-H] [M-2H]2- [M-H-CO2]
  pos-relevant: [M+H] [M+Na] [M+NH4] [M+K] [M+H-H2O] [M+H-NH3] [M+2H]2+ [2M+H]
  both:         [neutral]
(Cross-mode ions are included because real platforms aren't single-polarity -- ST004581's neg platform
 has [M+H]+ observations and the pos platforms have [M-H]-.)
"""
import sys, csv, re
csv.field_size_limit(10**7)
ik14 = lambda s: (s or "")[:14]
_norm = lambda s: "".join(ch for ch in (s or "").lower() if ch.isalnum())
P = 1.007276
EM = {"C":12.0,"H":1.0078250319,"O":15.9949146221,"N":14.0030740052,"P":30.97376151,"S":31.97207069}

# ---- comprehensive adduct set: (label, m/z = f(neutral mass M), platform-relevance) ----
_NEG = [("[M-H]",lambda M:M-P),("[M+Cl]",lambda M:M+34.969402),("[M+FA-H]",lambda M:M+44.998201),
        ("[M-H2O-H]",lambda M:M-19.017841),("[M-2H]2-",lambda M:(M-2*P)/2),("[M-H-CO2]",lambda M:M-44.997106)]
_POS = [("[M+H]",lambda M:M+P),("[M+Na]",lambda M:M+22.989218),("[M+NH4]",lambda M:M+18.033823),
        ("[M+K]",lambda M:M+38.963158),("[M+H-H2O]",lambda M:M-17.002740),("[M+H-NH3]",lambda M:M-16.019273),
        ("[M+2H]2+",lambda M:(M+2*P)/2),("[2M+H]",lambda M:2*M+P)]
_BOTH = [("[neutral]",lambda M:M)]
ADDUCTS = {  # platform -> primary + cross-mode + neutral (real platforms aren't single-polarity)
    "lc/ms neg":       _NEG + _POS + _BOTH,
    "lc/ms pos early": _POS + _NEG + _BOTH,
    "lc/ms pos late":  _POS + _NEG + _BOTH,
    "lc/ms polar":     _POS + _NEG + _BOTH,
}

# ---- lipid-shorthand -> formula -> neutral mass (for compounds with no SMILES) ----
_ACYL = {"acetyl":(2,0),"propionyl":(3,0),"butyryl":(4,0),"isobutyryl":(4,0),"valeryl":(5,0),"hexanoyl":(6,0),
"octanoyl":(8,0),"decanoyl":(10,0),"lauroyl":(12,0),"myristoyl":(14,0),"myristoleoyl":(14,1),"palmitoyl":(16,0),
"palmitoleoyl":(16,1),"margaroyl":(17,0),"stearoyl":(18,0),"oleoyl":(18,1),"linoleoyl":(18,2),"linolenoyl":(18,3),
"arachidoyl":(20,0),"arachidonoyl":(20,4),"eicosapentaenoyl":(20,5),"behenoyl":(22,0),"docosahexaenoyl":(22,6),
"lignoceroyl":(24,0),"nervonoyl":(24,1),"dodecenoyl":(12,1),"undecenoyl":(11,1),"decenoyl":(10,1)}
_CLS = {"tag":({"C":3,"H":8,"O":3},3),"triacylglycerol":({"C":3,"H":8,"O":3},3),
"dag":({"C":3,"H":8,"O":3},2),"diacylglycerol":({"C":3,"H":8,"O":3},2),
"mag":({"C":3,"H":8,"O":3},1),"monoacylglycerol":({"C":3,"H":8,"O":3},1),"glycerol":({"C":3,"H":8,"O":3},None),
"gpc":({"C":8,"H":20,"N":1,"O":6,"P":1},None),"gpe":({"C":5,"H":14,"N":1,"O":6,"P":1},None),
"gpi":({"C":9,"H":19,"O":11,"P":1},None),"gps":({"C":6,"H":14,"N":1,"O":8,"P":1},None),
"gpa":({"C":3,"H":9,"O":6,"P":1},None),"gpg":({"C":6,"H":15,"O":8,"P":1},None),
"carnitine":({"C":7,"H":15,"N":1,"O":3},1)}
def _lipid_neutral_mass(nm):
    l = (nm or "").lower()
    cls = next((k for k in _CLS if k in l), None)
    if cls is None: return None
    bb, _ = _CLS[cls]
    chains = [(int(a),int(b)) for a,b in re.findall(r"(\d{1,2}):(\d{1,2})", nm)]
    if not chains:
        chains = [_ACYL[k] for k in _ACYL if k in l]
    if not chains: return None
    f = dict(bb)
    for n,db in chains:
        f["C"]=f.get("C",0)+n; f["H"]=f.get("H",0)+(2*n-2*db)+(-2); f["O"]=f.get("O",0)+2-1  # acyl, esterify -H2O
    if "hydroxy" in l: f["O"]=f.get("O",0)+l.count("hydroxy")             # modifier: +O per hydroxy
    if "-dc" in l or "dicarboxyl" in l: f["O"]=f.get("O",0)+2             # dicarboxylic: +O2
    if "enyl" in l or "plasm" in l: f["O"]=f.get("O",0)-1                 # plasmalogen vinyl ether: -O
    return sum(EM[e]*k for e,k in f.items())

def _neutral_mass(smiles, name):
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            m = Chem.MolFromSmiles(smiles)
            if m is not None: return float(Descriptors.ExactMolWt(m))
        except Exception: pass
    return _lipid_neutral_mass(name)

def main(src, out):
    def col(r,*names):
        for n in names:
            if n in r and r[n] not in (None,""): return r[n]
        return ""
    w = csv.writer(open(out,"w")); w.writerow(["platform","name","inchikey","smiles","mz","adduct","ri"])
    n_out=n_smiles=n_lipid=n_none=0; seen=set()
    for r in csv.DictReader(open(src, encoding="utf-8-sig")):
        ik = col(r,"INCHIKEY","inchikey").strip(); nm = col(r,"BIOCHEMICAL","name","compound"); sm = col(r,"SMILES","smiles")
        if not ik or ik14(ik) in seen: continue
        seen.add(ik14(ik))
        M = _neutral_mass(sm, nm)
        if M is None: n_none+=1; continue
        if sm: n_smiles+=1
        else: n_lipid+=1
        if not (50 < M < 1500): continue
        for plat, adds in ADDUCTS.items():
            for lab, fn in adds:
                mz = fn(M)
                if mz > 30: w.writerow([plat, nm, ik, sm, "%.5f"%mz, lab, ""]); n_out+=1
    print("library rows: %d | compounds: %d (smiles %d, lipid-parse %d, unresolved %d)" % (
        n_out, n_smiles+n_lipid, n_smiles, n_lipid, n_none))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python build_library.py SOURCE_CSV OUT_CSV"); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
