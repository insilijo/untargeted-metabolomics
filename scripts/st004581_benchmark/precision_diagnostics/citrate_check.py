import csv
GT="/root/SQuID-INC/data/st004581/annotations_repaired.csv"
DD="/root/SQuID-INC/data/external/metabolon_data_dictionary_PMC_OA_subset_4.14.2024.csv"
CIT_IK14="KRKNYBCHXYNG"  # citric acid / citrate
print("=== citrate in the MAF (annotations_repaired) — all platforms ===")
for r in csv.DictReader(open(GT)):
    nm=(r.get("name") or ""); ik=(r.get("inchikey") or "")
    if "citr" in nm.lower() or ik[:12]==CIT_IK14:
        print(f"  platform={r.get('platform'):<16} name='{nm}'  ik={ik[:20]}  unannotatable={r.get('unannotatable')}")
print("\n=== isocitrate / aconitate (isobaric m/z 191.020 friends) in MAF ===")
for r in csv.DictReader(open(GT)):
    nm=(r.get("name") or "").lower()
    if any(x in nm for x in ["isocitr","aconit","citrate"]):
        print(f"  platform={r.get('platform'):<16} name='{r.get('name')}'  ik={(r.get('inchikey') or '')[:14]}")
print("\n=== citrate in the DD library (any platform) ===")
for r in csv.DictReader(open(DD)):
    nm=(r.get("biochemical") or r.get("name") or "")
    ik=(r.get("inchikey") or "")
    if "citrate" in nm.lower() and "iso" not in nm.lower():
        print(f"  DD: name='{nm}'  ik={ik[:20]}  platform/method={r.get('platform') or r.get('method') or '?'}")
