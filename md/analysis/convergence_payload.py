"""Payload (MMAE-core) stability + convergence diagnostics for cand_1 vs cand_4.
Outputs: payload_panel.png, convergence_panel.png, convergence_stats.csv."""
import csv, os, glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/galeito/ADCpedia/md/analysis'
EQ = 25.0  # production window start (ns) for equilibrated stats

# auto-discover candidates that have the full payload-diagnostic output
COLORMAP = {'cand_1': '#e07b39', 'cand_4': '#1f6f8b', 'cand_5': '#2ca02c', 'cand_8': '#9467bd'}
FALLBACK = ['#8c564b', '#d62728', '#17becf', '#bcbd22']
def discover():
    need = ['mmae_self_rmsd.xvg', 'mmae_rg.xvg', 'lig_prot_ncontacts.xvg', 'lig_rmsd.xvg']
    cands, fi = [], 0
    for d in sorted(glob.glob(f'{BASE}/cand_*')):
        c = os.path.basename(d)
        if all(os.path.exists(f'{d}/{n}') for n in need):
            col = COLORMAP.get(c)
            if col is None:
                col = FALLBACK[fi % len(FALLBACK)]; fi += 1
            cands.append((c, c.replace('_', ' '), col))
    return cands
CANDS = discover()
print('candidates:', [c[0] for c in CANDS])

def read_xvg(cand, name, tcol=0, ycol=1, yscale=1.0, tscale=1.0):
    xs, ys = [], []
    for line in open(f'{BASE}/{cand}/{name}'):
        line = line.strip()
        if not line or line[0] in '@#':
            continue
        p = line.split()
        xs.append(float(p[tcol]) * tscale); ys.append(float(p[ycol]) * yscale)
    return np.asarray(xs), np.asarray(ys)

# observable registry: key -> (file, ycol, yscale, tscale, label, unit)
OBS = {
    'mmae_rmsd':  ('mmae_self_rmsd.xvg',      1, 10.0, 1.0,    'MMAE self-RMSD',     'Å'),
    'mmae_rg':    ('mmae_rg.xvg',             1, 10.0, 1e-3,   'MMAE R$_g$',         'Å'),
    'contacts':   ('lig_prot_ncontacts.xvg',  1, 1.0,  1.0,    'Lig–prot contacts',  '#'),
    'lig_swing':  ('lig_rmsd.xvg',            1, 10.0, 1.0,    'Lig RMSD (prot-fit)','Å'),
}

def load(cand, key):
    f, yc, ys, ts, *_ = OBS[key]
    return read_xvg(cand, f, ycol=yc, yscale=ys, tscale=ts)

def block_sem(x):
    """Flyvbjerg–Petersen block averaging -> (plateau SEM, tau_int[frames], N_eff, stat.ineff s)."""
    x = np.asarray(x, float); N = len(x)
    naive = x.std(ddof=1) / np.sqrt(N)
    sizes, sems = [], []
    cur, b = x.copy(), 1
    while len(cur) >= 8:
        sems.append(cur.std(ddof=1) / np.sqrt(len(cur))); sizes.append(b)
        if len(cur) % 2:
            cur = cur[:-1]
        cur = 0.5 * (cur[0::2] + cur[1::2]); b *= 2
    sizes, sems = np.array(sizes), np.array(sems)
    nblocks = N / sizes
    plateau = sems[nblocks >= 10].max() if (nblocks >= 10).any() else sems.max()
    s = (plateau / naive) ** 2                    # statistical inefficiency
    tau = (s - 1.0) / 2.0                          # integrated autocorr time (frames)
    neff = N / s
    return plateau, tau, neff, s

def drift(t, y):
    """linear slope (units per ns) over the window."""
    a, b = np.polyfit(t, y, 1)                     # y = a*t + b
    return a

def sliding_mean(t, y, win_ns=5.0):
    """centered sliding-window average, edge-corrected (divide by actual count at boundaries)."""
    dt = np.median(np.diff(t)); w = max(1, int(round(win_ns / dt)))
    ker = np.ones(w)
    sums = np.convolve(y, ker, mode='same')
    cnts = np.convolve(np.ones_like(y), ker, mode='same')   # fewer points near edges
    return sums / cnts

def verdict(drift_tot, std):
    """convergence judged by systematic drift relative to thermal fluctuation."""
    r = abs(drift_tot) / std if std > 0 else np.inf
    return ('converged' if r < 0.5 else 'marginal' if r < 1.0 else 'DRIFTING'), r

# ---------- style ----------
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.titleweight': 'bold', 'axes.linewidth': 1.0,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9.5,
    'axes.edgecolor': '#333333', 'font.family': 'DejaVu Sans',
})
def clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, color='0.85', linewidth=0.6, alpha=0.6); ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=4, width=0.9, colors='#333333')

# ============== compute stats ==============
stats = {}
for cand, lab, col in CANDS:
    for key in OBS:
        t, y = load(cand, key)
        m = t >= EQ
        sem, tau_fr, neff, s = block_sem(y[m])
        dt = np.median(np.diff(t))
        sl = drift(t[m], y[m])
        win = t[m].max() - t[m].min()
        std = y[m].std(ddof=1)
        drift_tot = sl * win
        verd, ratio = verdict(drift_tot, std)
        stats[(cand, key)] = dict(mean=y[m].mean(), std=std, sem=sem, tau_ns=tau_fr * dt,
                                  neff=neff, slope=sl, drift_tot=drift_tot,
                                  ratio=ratio, conv=verd)

# ============== FIGURE 1: payload panel ==============
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
(axa, axb), (axc, axd) = ax
for cand, lab, col in CANDS:
    t, y = load(cand, 'mmae_rmsd'); st = stats[(cand, 'mmae_rmsd')]
    axa.plot(t, y, color=col, lw=0.9, alpha=0.85,
             label=f'{lab}  ({st["mean"]:.2f}±{st["sem"]:.2f} Å)')
    axa.axhline(st['mean'], ls='--', lw=1.2, color=col, alpha=0.9)
    tt, rg = load(cand, 'mmae_rg'); srg = stats[(cand, 'mmae_rg')]
    axb.plot(tt, rg, color=col, lw=0.9, alpha=0.85,
             label=f'{lab}  ({srg["mean"]:.2f}±{srg["sem"]:.2f} Å)')
    axb.axhline(srg['mean'], ls='--', lw=1.2, color=col, alpha=0.9)
    m = t >= EQ
    axc.hist(y[m], bins=40, color=col, alpha=0.55, density=True, label=lab)
    axd.hist(rg[tt >= EQ], bins=40, color=col, alpha=0.55, density=True, label=lab)
axa.set_title('Payload internal stability (MMAE self-fitted RMSD)')
axa.set_xlabel('Time (ns)'); axa.set_ylabel('MMAE RMSD (Å)'); axa.set_xlim(0, 50)
axa.legend(frameon=False, loc='upper left', title=f'mean±SEM ({EQ:.0f}–50 ns)')
axb.set_title('Payload compactness (MMAE radius of gyration)')
axb.set_xlabel('Time (ns)'); axb.set_ylabel('MMAE R$_g$ (Å)'); axb.set_xlim(0, 50)
axb.legend(frameon=False, loc='upper left', title=f'mean±SEM ({EQ:.0f}–50 ns)')
axc.set_title(f'MMAE RMSD distribution ({EQ:.0f}–50 ns)')
axc.set_xlabel('MMAE RMSD (Å)'); axc.set_ylabel('density'); axc.legend(frameon=False)
axd.set_title(f'MMAE R$_g$ distribution ({EQ:.0f}–50 ns)')
axd.set_xlabel('MMAE R$_g$ (Å)'); axd.set_ylabel('density'); axd.legend(frameon=False)
for a, lb in zip([axa, axb, axc, axd], 'abcd'):
    clean(a); a.text(-0.10, 1.04, f'({lb})', transform=a.transAxes,
                     fontsize=14, fontweight='bold', va='bottom', ha='left')
fig.suptitle('Payload (MMAE core) stability — ' + ', '.join(lab for _, lab, _ in CANDS),
             fontsize=15, fontweight='bold', y=1.00)
fig.tight_layout(pad=1.6, w_pad=2.5, h_pad=2.5)
fig.savefig(f'{BASE}/payload_panel.png', dpi=300, bbox_inches='tight')
print('WROTE payload_panel.png')

# ============== FIGURE 2: convergence panel ==============
fig2, ax2 = plt.subplots(2, 2, figsize=(12, 9))
order = ['mmae_rmsd', 'mmae_rg', 'contacts', 'lig_swing']
MARK = {'converged': '✓conv', 'marginal': '~marg', 'DRIFTING': '✗drift'}
def fmt_stat(s):
    """Show τ/N_eff/SEM only where the series is (near-)stationary; block-averaging
    error stats are invalid on a drifting series, so flag those as N_eff≈1."""
    if s['conv'] == 'DRIFTING':
        return f"drift/σ={s['ratio']:.2f} ✗drift  N_eff≈1 (non-stat.)"
    cav = '*' if s['conv'] == 'marginal' else ''
    return f"drift/σ={s['ratio']:.2f} {MARK[s['conv']]}  τ={s['tau_ns']:.1f}ns N_eff={s['neff']:.0f}{cav}"
for axx, key in zip(ax2.flat, order):
    _, _, _, _, label, unit = OBS[key]
    for cand, lab, col in CANDS:
        t, y = load(cand, key); st = stats[(cand, key)]
        sm = sliding_mean(t, y, 5.0)
        axx.plot(t, y, color=col, lw=0.4, alpha=0.16)
        axx.plot(t, sm, color=col, lw=1.8, label=lab)               # 5-ns sliding mean
        axx.axhline(st['mean'], ls='--', lw=1.0, color=col, alpha=0.7)
    axx.axvline(EQ, ls=':', lw=1.2, color='0.4')                    # production-window start
    txt = "\n".join(f"{lab}: {fmt_stat(stats[(cand, key)])}" for cand, lab, _ in CANDS)
    axx.text(0.97, 0.05, txt, transform=axx.transAxes, fontsize=8.0, va='bottom', ha='right',
             family='monospace', bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    axx.set_title(f'{label} — 5 ns sliding mean')
    axx.set_xlabel('Time (ns)'); axx.set_ylabel(f'{label} ({unit})'); axx.set_xlim(0, 50)
    axx.legend(frameon=False, loc='upper left')
    clean(axx)
fig2.suptitle('Convergence diagnostics — 5 ns sliding mean; verdict = |drift| vs fluctuation σ (production 25–50 ns)',
              fontsize=13, fontweight='bold', y=1.00)
fig2.tight_layout(pad=1.6, w_pad=2.5, h_pad=2.5)
fig2.savefig(f'{BASE}/convergence_panel.png', dpi=300, bbox_inches='tight')
print('WROTE convergence_panel.png')

# ============== CSV + console table ==============
with open(f'{BASE}/convergence_stats.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['candidate', 'observable', 'mean', 'std', 'SEM', 'tau_ns', 'N_eff',
                'slope_per_ns', 'drift_over_window', 'drift_over_std', 'verdict', 'error_stats_valid'])
    for cand, lab, _ in CANDS:
        for key in order:
            s = stats[(cand, key)]
            stat_ok = s['conv'] != 'DRIFTING'
            sem = f"{s['sem']:.3f}" if stat_ok else 'n/a(non-stat)'
            tau = f"{s['tau_ns']:.2f}" if stat_ok else 'n/a'
            neff = f"{s['neff']:.0f}" if stat_ok else '~1'
            valid = 'yes' if s['conv'] == 'converged' else 'caveat' if s['conv'] == 'marginal' else 'NO'
            w.writerow([cand, OBS[key][4], f"{s['mean']:.3f}", f"{s['std']:.3f}", sem,
                        tau, neff, f"{s['slope']:.4f}",
                        f"{s['drift_tot']:.3f}", f"{s['ratio']:.2f}", s['conv'], valid])
print('WROTE convergence_stats.csv\n')
print(f"{'cand':>7} {'observable':>20} {'mean':>8} {'std':>6} {'±SEM':>13} {'tau(ns)':>8} "
      f"{'N_eff':>6} {'drift/σ':>8}  verdict")
print('  (SEM/τ/N_eff suppressed where non-stationary — block-averaging assumes stationarity)')
for cand, lab, _ in CANDS:
    for key in order:
        s = stats[(cand, key)]
        stat_ok = s['conv'] != 'DRIFTING'
        sem = f"{s['sem']:>13.2f}" if stat_ok else f"{'n/a(non-stat)':>13}"
        tau = f"{s['tau_ns']:>8.1f}" if stat_ok else f"{'n/a':>8}"
        neff = f"{s['neff']:>6.0f}" if stat_ok else f"{'~1':>6}"
        print(f"{lab:>7} {OBS[key][4]:>20} {s['mean']:>8.2f} {s['std']:>6.2f} {sem} "
              f"{tau} {neff} {s['ratio']:>8.2f}  {s['conv']}")
