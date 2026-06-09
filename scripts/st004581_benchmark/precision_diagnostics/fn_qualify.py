import csv
def load(f):
    d={}
    for r in csv.DictReader(open(f),delimiter='\t'): d[r['mid']]=r
    return d
d6=load('/tmp/fnq_6.tsv'); d14=load('/tmp/fnq_14.tsv'); d30=load('/tmp/fnq_30.tsv')
mids=set(d30); N=len(mids)
always=[]; anch14=[]; anch30=[]; never=[]
for m in mids:
    r6=int(d6[m]['recovered']); r14=int(d14[m]['recovered']); r30=int(d30[m]['recovered'])
    if r6: always.append(m)
    elif r14: anch14.append(m)
    elif r30: anch30.append(m)
    else: never.append(m)
anchor=anch14+anch30
print(f"=== MAF compound recovery vs kit size (N={N}) ===")
print(f"  always recovered (kit=6 already)      : {len(always):>3} ({len(always)/N:.2f})")
print(f"  ANCHOR-recovered, needed kit 6->14    : {len(anch14):>3} ({len(anch14)/N:.2f})")
print(f"  ANCHOR-recovered, needed kit 14->30   : {len(anch30):>3} ({len(anch30)/N:.2f})")
print(f"  NEVER recovered (still FN at kit=30)  : {len(never):>3} ({len(never)/N:.2f})")
print(f"\n  => anchors recover {len(anchor)} failures ({len(anchor)/N:.2f} of MAF); that's the kit-size recall gain.")
imp=sum(1 for m in anchor if int(d6[m]['at_pred'])==0 and int(d30[m]['at_pred'])==1)
print(f"\n=== MECHANISM: of {len(anchor)} anchor-recovered, RT-prediction fixed (peak NOT at predicted-RT@kit6 -> IS @kit30): {imp} ({imp/max(len(anchor),1):.2f}) ===")
nopeak=sum(1 for m in never if int(d30[m]['has_peak'])==0); haspeak=len(never)-nopeak
print(f"\n=== NEVER-recovered ({len(never)}) split ===")
print(f"  no strong peak anywhere (ACQUISITION floor, NOT anchor): {nopeak} ({nopeak/max(len(never),1):.2f})")
print(f"  has a peak but unrecovered (isobar/mislocation residual): {haspeak} ({haspeak/max(len(never),1):.2f})")
print("\n=== predicted-RT region: anchor-recovered vs never (kit-design map) ===")
ar=[float(d30[m]['pred']) for m in anchor]; nv=[float(d30[m]['pred']) for m in never]
for lo,hi in [(0,80),(80,160),(160,260),(260,360),(360,600)]:
    print(f"  RT[{lo:>3},{hi:>3})s: anchor-recovered {sum(lo<=x<hi for x in ar):>2}   never {sum(lo<=x<hi for x in nv):>2}")
