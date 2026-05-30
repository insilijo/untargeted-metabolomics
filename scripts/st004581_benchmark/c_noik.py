"""What ARE the no-InChIKey data-dictionary entries, and are they ever in the MAF?"""
import csv, re
from collections import Counter, defaultdict
DDCSV="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
MAFCSV="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
ik14=lambda s:(s or "")[:14]
norm=lambda s:re.sub(r"\s+"," ",(s or "").strip().lower())

DD=list(csv.DictReader(open(DDCSV, encoding="utf-8-sig")))
MAF=list(csv.DictReader(open(MAFCSV)))

# MAF: do any GT rows lack an inchikey?
maf_noik=sum(1 for r in MAF if not (r.get("inchikey") or "").strip())
maf_names=set(norm(r["name"]) for r in MAF)
maf_names_ik=set(norm(r["name"]) for r in MAF if (r.get("inchikey") or "").strip())
print(f"MAF rows: {len(MAF)};  MAF rows with NO inchikey: {maf_noik}")

# DD no-ik characterization
dd_total=len(DD); dd_noik=[r for r in DD if not (r.get("INCHIKEY") or "").strip()]
print(f"DD rows: {dd_total};  DD with NO inchikey: {len(dd_noik)} ({len(dd_noik)/dd_total:.0%})")
# is the name an unnamed Metabolon unknown? (e.g. 'X - 12345')
unk=lambda n: bool(re.match(r"^x\s*-\s*\d+", norm(n)))
n_unknown=sum(1 for r in dd_noik if unk(r["BIOCHEMICAL"]))
print(f"  of no-ik DD: unnamed 'X-#####' unknowns: {n_unknown} ({n_unknown/len(dd_noik):.0%}); named: {len(dd_noik)-n_unknown}")
# do no-ik DD names appear in the MAF (i.e., real compound, just missing structure in public DD)?
named_noik=[r for r in dd_noik if not unk(r["BIOCHEMICAL"])]
in_maf=sum(1 for r in named_noik if norm(r["BIOCHEMICAL"]) in maf_names)
in_maf_ik=sum(1 for r in named_noik if norm(r["BIOCHEMICAL"]) in maf_names_ik)
print(f"  named no-ik DD entries whose NAME is in the MAF at all: {in_maf}/{len(named_noik)}")
print(f"  ... and in the MAF WITH an inchikey (so structure exists, public DD just dropped it): {in_maf_ik}/{len(named_noik)}")
print("\n  examples of named no-ik DD entries:")
for r in named_noik[:12]:
    nm=norm(r["BIOCHEMICAL"]); tag="IN-MAF(ik)" if nm in maf_names_ik else ("IN-MAF" if nm in maf_names else "not-in-MAF")
    print(f"    {r['BIOCHEMICAL'][:46]:<46} {r['PLATFORM']:<16} {tag}")
print("\n  examples of unnamed unknowns:")
for r in [r for r in dd_noik if unk(r['BIOCHEMICAL'])][:5]:
    print(f"    {r['BIOCHEMICAL'][:46]:<46} {r['PLATFORM']}")
