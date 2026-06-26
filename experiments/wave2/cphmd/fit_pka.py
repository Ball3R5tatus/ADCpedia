#!/usr/bin/env python3
"""Fit per-residue pKa from a GB-CpHMD pH ladder (cphstats .dat per pH).
Reads pka_gpu_pH{3..7}.dat in outputs/wave2/cphmd/<CAND>/, fits a Hill curve
  frac_prot(pH) = 1 / (1 + 10**(n*(pH - pKa)))
to each residue's protonated fraction, and reports the titrating residues
(those whose fraction crosses 0.5 within the sampled window).
Usage: python fit_pka.py <CAND> [pH list e.g. 3,4,5,6,7]
"""
import sys, re, glob, os

cand = sys.argv[1] if len(sys.argv) > 1 else "cand_4"
phs = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["3","4","5","6","7"])]
D = f"/home/galeito/ADCpedia/outputs/wave2/cphmd/{cand}"

# residue -> {pH: frac_prot}
data = {}
line_re = re.compile(r"^(\S+\s+\d+)\s*:.*Frac Prot\s+([0-9.]+)")
for ph in phs:
    f = f"{D}/pka_gpu_pH{int(ph)}.dat"
    if not os.path.exists(f):
        print(f"!! missing {f}"); continue
    for ln in open(f):
        m = line_re.match(ln.strip())
        if m:
            res, frac = m.group(1).strip(), float(m.group(2))
            data.setdefault(res, {})[ph] = frac

# Hill fit by coarse->fine grid (no scipy dependency)
def hill(ph, pka, n):
    return 1.0 / (1.0 + 10.0 ** (n * (ph - pka)))

def fit(points):  # points = list of (pH, frac)
    xs = [p for p, _ in points]; ys = [y for _, y in points]
    best = None
    pka_lo, pka_hi = min(phs) - 1, max(phs) + 1
    for i in range(int((pka_hi - pka_lo) / 0.02) + 1):
        pka = pka_lo + i * 0.02
        for j in range(int((3.0 - 0.3) / 0.02) + 1):
            n = 0.3 + j * 0.02
            sse = sum((hill(x, pka, n) - y) ** 2 for x, y in zip(xs, ys))
            if best is None or sse < best[2]:
                best = (pka, n, sse)
    return best

proximal = {"AS4 122", "GL4 213", "GL4 219", "AS4 219", "GL4 218"}
rows = []
for res, d in data.items():
    pts = sorted(d.items())
    fr = [y for _, y in pts]
    if not fr: continue
    titrates = (max(fr) > 0.5 and min(fr) < 0.5)         # crosses midpoint in window
    if titrates and len(pts) >= 3:
        pka, n, sse = fit(pts)
        rmse = (sse / len(pts)) ** 0.5
        rows.append((res, pka, n, rmse, d))

rows.sort(key=lambda r: r[1])
print(f"\n=== {cand}: fitted pKa (titrating residues, pH window {phs}) ===")
print(f"{'residue':<10} {'pKa':>6} {'Hill_n':>7} {'RMSE':>6}   frac_prot per pH")
for res, pka, n, rmse, d in rows:
    fr = "  ".join(f"{int(p)}:{d.get(p,float('nan')):.2f}" for p in phs)
    tag = "  <-- PROXIMAL" if res in proximal else ""
    print(f"{res:<10} {pka:6.2f} {n:7.2f} {rmse:6.3f}   {fr}{tag}")

# save csv
with open(f"{D}/pka_summary.csv", "w") as o:
    o.write("residue,pKa,hill_n,rmse," + ",".join(f"frac_pH{int(p)}" for p in phs) + ",proximal\n")
    for res, pka, n, rmse, d in rows:
        o.write(f"{res.replace(' ','_')},{pka:.2f},{n:.2f},{rmse:.3f}," +
                ",".join(f"{d.get(p,'')}" for p in phs) + f",{res in proximal}\n")
print(f"\nsaved {D}/pka_summary.csv  ({len(rows)} titrating residues; "
      f"{len(data)-len(rows)} non-titrating/outside-window)")
