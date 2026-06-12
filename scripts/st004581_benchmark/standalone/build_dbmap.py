"""Build a cross-database membership map (ik14 -> databases) for sparse_anchor_annotate --rich.

usage:  python build_dbmap.py OUT.csv  LABEL=SOURCE [LABEL=SOURCE ...]
  SOURCE: a .csv with an 'inchikey' column, or a .json keyed by inchikey.
  e.g.    python build_dbmap.py dbmap.csv HMDB=hmdb.csv GNPS=gnps_neg.json ZINC=zinc.csv
"""
import sys, csv, json
from collections import defaultdict
csv.field_size_limit(10**7)
ik14 = lambda s: (s or "")[:14]

def inchikeys(path):
    if path.endswith(".json"):
        for k in json.load(open(path)):
            yield k
    else:
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            yield r.get("inchikey") or r.get("INCHIKEY") or ""

def main(out, sources):
    db = defaultdict(set)
    for spec in sources:
        label, path = spec.split("=", 1)
        for ik in inchikeys(path):
            if ik14(ik): db[ik14(ik)].add(label)
    w = csv.writer(open(out, "w")); w.writerow(["ik14", "databases"])
    for ik, s in db.items(): w.writerow([ik, ";".join(sorted(s))])
    print("db map: %d compounds across %d sources" % (len(db), len(sources)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python build_dbmap.py OUT.csv LABEL=SOURCE ..."); sys.exit(1)
    main(sys.argv[1], sys.argv[2:])
