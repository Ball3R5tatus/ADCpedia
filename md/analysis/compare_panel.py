"""Publication-quality 2x2 comparison panel: cand_1 vs cand_4 MD stability.
Overlays the two candidates on the decision-relevant metrics.
Inputs: GROMACS .xvg (time in ns; distances/RMSD in nm)."""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/galeito/ADCpedia/md/analysis'
OUT = f'{BASE}/cand1_vs_cand4_panel.png'
NM2A = 10.0

CANDS = [
    ('cand_1', 'cand 1', '#e07b39'),
    ('cand_4', 'cand 4', '#1f6f8b'),
]

# per-candidate list of self-fitted domain RMSD files (3 protein chains + 4 HER2 domains)
DOMAIN_FILES = {
    'cand_1': ['dom_chA.xvg', 'dom_chB.xvg', 'dom_chC.xvg',
               'dom_hI.xvg', 'dom_hII.xvg', 'dom_hIII.xvg', 'dom_hIV.xvg'],
    'cand_4': ['domA_rmsd.xvg', 'domB_rmsd.xvg', 'domC_rmsd.xvg',
               'her2_I_rmsd.xvg', 'her2_II_rmsd.xvg', 'her2_III_rmsd.xvg', 'her2_IV_rmsd.xvg'],
}

def read_xvg(cand, name):
    xs, ys = [], []
    with open(f'{BASE}/{cand}/{name}') as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in '@#':
                continue
            p = line.split()
            xs.append(float(p[0])); ys.append(float(p[1]))
    return np.asarray(xs), np.asarray(ys)

def tail_stats(x, y, frm=25.0):
    """mean +/- std over the equilibrated window (t >= frm ns)."""
    m = x >= frm
    return y[m].mean(), y[m].std()

# ---------- style ----------
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.titleweight': 'bold', 'axes.linewidth': 1.0,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'axes.edgecolor': '#333333', 'font.family': 'DejaVu Sans',
})

def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='0.85', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=4, width=0.9, colors='#333333')

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(axa, axb), (axc, axd) = axes

# ---------- (a) covalent C-SG distance ----------
for cand, lab, col in CANDS:
    t, d = read_xvg(cand, 'c3sg_dist.xvg'); d *= NM2A
    m, s = tail_stats(t, d)
    axa.plot(t, d, color=col, lw=0.9, alpha=0.85, label=f'{lab}  ({m:.2f} Å)')
    axa.axhline(m, ls='--', lw=1.3, color=col, alpha=0.9)
axa.set_title('Covalent thioether bond')
axa.set_xlabel('Time (ns)'); axa.set_ylabel('C–SG distance (Å)')
axa.legend(frameon=False, loc='upper right', title='mean (25–50 ns)')

# ---------- (b) mean per-domain self-fitted CA RMSD (internal stability) ----------
for cand, lab, col in CANDS:
    stack = []
    for fn in DOMAIN_FILES[cand]:
        t, r = read_xvg(cand, fn)
        stack.append(r * NM2A)
    n = min(len(a) for a in stack)
    stack = np.vstack([a[:n] for a in stack]); t = t[:n]
    mean_dom = stack.mean(axis=0); spread = stack.std(axis=0)
    m, s = tail_stats(t, mean_dom)
    axb.fill_between(t, mean_dom - spread, mean_dom + spread, color=col, alpha=0.18, lw=0)
    axb.plot(t, mean_dom, color=col, lw=1.3, label=f'{lab}  ({m:.1f}±{s:.1f} Å)')
axb.set_title('Mean per-domain C$_\\alpha$ RMSD (self-fitted)')
axb.set_xlabel('Time (ns)'); axb.set_ylabel('C$_\\alpha$ RMSD (Å)')
axb.legend(frameon=False, loc='upper left', title='mean over 7 domains (25–50 ns)')

# ---------- (c) ligand RMSD ----------
for cand, lab, col in CANDS:
    t, r = read_xvg(cand, 'lig_rmsd.xvg'); r *= NM2A
    m, s = tail_stats(t, r)
    axc.plot(t, r, color=col, lw=1.1, label=f'{lab}  ({m:.1f}±{s:.1f} Å)')
axc.set_title('Payload (ligand) RMSD')
axc.set_xlabel('Time (ns)'); axc.set_ylabel('Ligand RMSD (Å)')
axc.legend(frameon=False, loc='upper left', title='mean (25–50 ns)')

# ---------- (d) ligand-protein contacts ----------
for cand, lab, col in CANDS:
    t, n = read_xvg(cand, 'lig_prot_ncontacts.xvg')
    m, s = tail_stats(t, n)
    axd.plot(t, n, color=col, lw=0.9, alpha=0.85, label=f'{lab}  ({m:.0f}±{s:.0f})')
axd.set_title('Ligand-protein contacts')
axd.set_xlabel('Time (ns)'); axd.set_ylabel('Contacts < 0.4 nm')
axd.legend(frameon=False, loc='upper left', title='mean (25–50 ns)')

for ax in (axa, axb, axc, axd):
    ax.set_xlim(0, 50)
    clean(ax)

for ax, lab in zip([axa, axb, axc, axd], 'abcd'):
    ax.text(-0.10, 1.04, f'({lab})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')

fig.suptitle('MD stability comparison — cand 1 vs cand 4 (50 ns each)',
             fontsize=15, fontweight='bold', y=1.00)
fig.tight_layout(pad=1.6, w_pad=2.5, h_pad=2.5)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'WROTE {OUT}')
for cand, lab, _ in CANDS:
    t, d = read_xvg(cand, 'c3sg_dist.xvg'); md, _ = tail_stats(t, d*NM2A)
    dom = np.vstack([read_xvg(cand, fn)[1]*NM2A for fn in DOMAIN_FILES[cand]]).mean(axis=0)
    mr, _ = tail_stats(t, dom)
    t, lr = read_xvg(cand, 'lig_rmsd.xvg'); ml, _ = tail_stats(t, lr*NM2A)
    t, nc = read_xvg(cand, 'lig_prot_ncontacts.xvg'); mn, _ = tail_stats(t, nc)
    print(f'{lab}: C-SG={md:.2f}A  domRMSD={mr:.1f}A  ligRMSD={ml:.1f}A  contacts={mn:.0f}')
