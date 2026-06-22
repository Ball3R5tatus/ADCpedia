#!/usr/bin/env python3
"""Aggregate MMAE-core self-RMSD across N=3 replicates per candidate, apply the
pre-registered decision criterion (separation iff mean+-SD ranges disjoint AND
Welch p<0.05), and plot. Self-RMSD averaged over the 2nd half (25-50 ns)."""
import os, math, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ANA = '/home/galeito/ADCpedia/md/analysis'
REPS = ['prod', 'prod_rep2', 'prod_rep3']
CANDS = {'cand_4': 'tab:blue', 'cand_5': 'tab:orange'}
HALF_NS = 25.0   # average over t >= 25 ns

def load_xvg(path):
    xs, ys = [], []
    for l in open(path):
        if l[0] in '#@': continue
        p = l.split()
        if len(p) >= 2: xs.append(float(p[0])); ys.append(float(p[1]) * 10.0)  # nm->A
    return xs, ys

def mean_second_half(path):
    xs, ys = load_xvg(path)
    h = [v for t, v in zip(xs, ys) if t >= HALF_NS]
    return st.mean(h), (xs, ys)

# --- collect per-replicate means ---
data = {}; series = {}
for cand in CANDS:
    vals = []
    for rep in REPS:
        f = f'{ANA}/{cand}/mmae_selfrmsd_{rep}.xvg'
        m, sxy = mean_second_half(f)
        vals.append(m); series[(cand, rep)] = sxy
    data[cand] = vals

print('=== MMAE-core self-RMSD, mean over 25-50 ns (A) ===')
for cand in CANDS:
    v = data[cand]
    print(f'  {cand}: rep1={v[0]:.2f}  rep2={v[1]:.2f}  rep3={v[2]:.2f}'
          f'  ->  mean={st.mean(v):.2f}  SD={st.pstdev(v):.2f}  sampleSD={st.stdev(v):.2f}')

a, b = data['cand_4'], data['cand_5']
ma, mb = st.mean(a), st.mean(b)
sa, sb = st.stdev(a), st.stdev(b)            # sample SD (n-1)
na = nb = 3
# Welch t-test (unequal variance)
se = math.sqrt(sa**2/na + sb**2/nb)
t = (ma - mb) / se if se > 0 else float('inf')
df = (sa**2/na + sb**2/nb)**2 / ((sa**2/na)**2/(na-1) + (sb**2/nb)**2/(nb-1))
try:
    from scipy import stats
    p = 2 * stats.t.sf(abs(t), df)
    pmethod = 'scipy'
except Exception:
    # Welch-Satterthwaite p via numerical incomplete beta (no scipy)
    def betacf(x, a, bb):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a+bb, a+1.0, a-1.0
        c = 1.0; d = 1.0 - qab*x/qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1.0/d; h = d
        for m in range(1, MAXIT+1):
            m2 = 2*m
            aa = m*(bb-m)*x/((qam+m2)*(a+m2))
            d = 1.0+aa*d; d = FPMIN if abs(d)<FPMIN else d
            c = 1.0+aa/c; c = FPMIN if abs(c)<FPMIN else c
            d = 1.0/d; h *= d*c
            aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d = 1.0+aa*d; d = FPMIN if abs(d)<FPMIN else d
            c = 1.0+aa/c; c = FPMIN if abs(c)<FPMIN else c
            d = 1.0/d; de = d*c; h *= de
            if abs(de-1.0) < EPS: break
        return h
    def betai(a, bb, x):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        lbeta = math.lgamma(a)+math.lgamma(bb)-math.lgamma(a+bb)
        front = math.exp(math.log(x)*a + math.log(1-x)*bb - lbeta)/a
        if x < (a+1)/(a+bb+2): return front*betacf(x, a, bb)
        else: return 1.0 - front*betacf(x, bb, a) * (a/bb)  # not used near here
    p = betai(df/2, 0.5, df/(df + t*t))
    pmethod = 'manual-incbeta'

print(f'\n=== Welch t-test (cand_4 vs cand_5) [{pmethod}] ===')
print(f'  cand_4 = {ma:.2f} +- {sa:.2f} A   (n=3)')
print(f'  cand_5 = {mb:.2f} +- {sb:.2f} A   (n=3)')
print(f'  gap |mean4-mean5| = {abs(ma-mb):.2f} A')
print(f'  t = {t:.3f}   df = {df:.2f}   p = {p:.4f}')

# pre-registered criterion: disjoint mean+-SD ranges AND p<0.05
lo4, hi4 = ma-sa, ma+sa
lo5, hi5 = mb-sb, mb+sb
disjoint = (hi4 < lo5) or (hi5 < lo4)
print(f'\n  cand_4 range [{lo4:.2f}, {hi4:.2f}]   cand_5 range [{lo5:.2f}, {hi5:.2f}]')
print(f'  ranges disjoint? {disjoint}    p<0.05? {p<0.05}')
verdict = 'SEPARATION' if (disjoint and p < 0.05) else 'NO SEPARATION'
print(f'\n  >>> PRE-REGISTERED VERDICT: {verdict} <<<')

# --- plot: time-series (left) + bar mean+-SD with points (right) ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
for cand, col in CANDS.items():
    for i, rep in enumerate(REPS):
        xs, ys = series[(cand, rep)]
        axL.plot(xs, ys, color=col, alpha=0.45, lw=0.8,
                 label=cand if i == 0 else None)
axL.axvline(HALF_NS, ls=':', c='k', lw=0.8)
axL.set_xlabel('time (ns)'); axL.set_ylabel('MMAE-core self-RMSD (A)')
axL.set_title('All 6 replicate trajectories'); axL.legend()

xpos = {'cand_4': 0, 'cand_5': 1}
for cand, col in CANDS.items():
    m = st.mean(data[cand]); sd = st.stdev(data[cand])
    axR.bar(xpos[cand], m, yerr=sd, capsize=8, color=col, alpha=0.6, width=0.5)
    for v in data[cand]:
        axR.plot(xpos[cand], v, 'o', color='k', ms=7, zorder=3)
axR.set_xticks([0, 1]); axR.set_xticklabels(['cand_4', 'cand_5'])
axR.set_ylabel('self-RMSD mean over 25-50 ns (A)')
axR.set_title(f'mean+-SD (n=3)   Welch p={p:.3f}\nVERDICT: {verdict}')
plt.tight_layout()
out = f'{ANA}/selfrmsd_n3_summary.png'
plt.savefig(out, dpi=130)
print(f'\nplot -> {out}')
