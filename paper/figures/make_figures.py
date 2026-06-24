#!/usr/bin/env python3
"""
Publication figures for the negative-result ADC-linker MD paper.
Run from repo root with system python3 (matplotlib 3.11, numpy):
    python3 paper/figures/make_figures.py
Reads real GROMACS .xvg / dssp.dat under md/analysis/; summary numbers (where
no per-frame file exists, e.g. buggy-vs-fixed) are taken from the master doc and
clearly tagged in the figure caption. Writes PNG (300 dpi) + PDF to paper/figures/.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats as ss

A = "md/analysis"
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.linewidth": 0.8, "savefig.dpi": 300,
    "figure.dpi": 120, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False,
})
# colourblind-friendly
C = {"cand_4": "#0072B2", "cand_5": "#D55E00", "cand_1": "#009E73",
     "cand_8": "#CC79A7", "vedotin": "#000000", "accent": "#E69F00"}

def load_xvg(f):
    t, v = [], []
    if not os.path.exists(f):
        return None, None
    for ln in open(f):
        if not ln or ln[0] in "#@":
            continue
        a = ln.split()
        if len(a) >= 2:
            try:
                t.append(float(a[0])); v.append(float(a[1]))
            except ValueError:
                pass
    return np.array(t), np.array(v)

def win_mean(f, lo=25, scale=10.0):
    t, v = load_xvg(f)
    if t is None:
        return np.nan
    return v[t >= lo].mean() * scale

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# ----------------------------------------------------------------------------- F1
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(5.2, 7.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")
    steps = [
        ("ADCpedia dataset\n73 ADC linker-payloads", "#D9E8F5", None),
        ("DiffLinker fine-tune\nVal-Cit / urea / GGFG emerge\n0 near-copies, Tanimoto 0.44-0.59", "#D9E8F5", "GENERATION"),
        ("Raw 3D output\nconnectivity / valence / stereo loss", "#FBE7D6", None),
        ("Chemical-validity gate\n(fail-closed: valence, connectivity, stereo)", "#FBE7D6", "VALIDITY"),
        ("~9-10 usable-3D / 100\n[Wilson 95% CI]", "#FBE7D6", None),
        ("Covalent complex build\ntrastuzumab-Fab/HER2 @ Cys214\ncanonical MMAE stereo, thiosuccinimide", "#E4DCEC", "STRUCTURE"),
        ("Covalent MD  (50 ns x N=3)\namber99sb-ildn / GAFF2-AM1BCC / TIP3P", "#E4DCEC", None),
        ("Evaluation\npayload self-RMSD / DSSP / per-domain RMSD", "#DCEDDC", "EVALUATION"),
        ("FDA positive control\nbrentuximab-vedotin linker, same scaffold", "#DCEDDC", None),
        ("VERDICT: metric is non-discriminative\nbottleneck = evaluation, not generation", "#F5D9D9", None),
    ]
    n = len(steps); h = 1.55; gap = (20 - n * h) / (n + 1)
    y = 20 - gap - h
    boxes = []
    for txt, col, tag in steps:
        box = FancyBboxPatch((1.4, y), 7.2, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.0, edgecolor="#444", facecolor=col)
        ax.add_patch(box)
        ax.text(5.0, y + h / 2, txt, ha="center", va="center", fontsize=8.0)
        if tag:
            ax.text(0.95, y + h / 2, tag, ha="right", va="center", fontsize=6.6,
                    color="#666", rotation=90, fontweight="bold")
        boxes.append(y)
        y -= h + gap
    for i in range(n - 1):
        y0 = boxes[i]; y1 = boxes[i + 1] + h
        ax.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1),
                     arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#555"))
    ax.set_title("Generate → gate → build → simulate → evaluate", fontsize=10, pad=4)
    save(fig, "fig1_pipeline")

# ----------------------------------------------------------------------------- F2
def fig2_yield_funnel():
    # numbers per master §19.17-§19.18 (H4 λ=0.01, n=500, Wilson CI)
    stages = ["Generated", "Connected\n(single frag)", "MMFF-valid\n(3D-relaxable)",
              "+ ADC motif\n(Val-Cit / urea)"]
    vals = [100, 33, 33, 10]          # per 100
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ymax = 100
    for i, (s, v) in enumerate(zip(stages, vals)):
        w = v / ymax * 0.8
        ax.add_patch(plt.Rectangle((0.5 - w / 2, i), w, 0.7,
                     facecolor=C["cand_4"], alpha=0.30 + 0.18 * i, edgecolor="#333", lw=0.8))
        ax.text(0.5, i + 0.35, f"{v} / 100", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(1.0, i + 0.35, s, ha="left", va="center", fontsize=8.2)
    ax.text(0.5, len(stages) - 0.05, "usable-3D = 10.2 / 100   [95% CI 7.8–13.2]",
            ha="center", va="bottom", fontsize=7.6, color="#444", style="italic")
    ax.set_xlim(-0.05, 1.9); ax.set_ylim(-0.2, len(stages) + 0.4); ax.axis("off")
    ax.set_title("Chemical validity — not generation — is the bottleneck", fontsize=10)
    save(fig, "fig2_yield_funnel")

# ----------------------------------------------------------------------------- F3
def dssp_structured(f):
    """% structured (H,G,I,E,B) per frame; '=' separates chains."""
    if not os.path.exists(f):
        return None
    struct = set("HGIEB")
    frac = []
    for ln in open(f):
        ln = ln.strip().replace("=", "")
        if not ln:
            continue
        tot = len(ln)
        s = sum(1 for c in ln if c in struct)
        frac.append(100.0 * s / tot)
    return np.array(frac)

def fig3_gross_stability():
    cand_dom = {
        "cand_4": dict(light="domA_rmsd", heavy="domB_rmsd",
                       I="her2_I_rmsd", II="her2_II_rmsd", III="her2_III_rmsd", IV="her2_IV_rmsd"),
        "cand_1": dict(light="dom_chA", heavy="dom_chB",
                       I="dom_hI", II="dom_hII", III="dom_hIII", IV="dom_hIV"),
        "cand_5": dict(light="dom_chA", heavy="dom_chB",
                       I="dom_hI", II="dom_hII", III="dom_hIII", IV="dom_hIV"),
        "cand_8": dict(light="dom_chA", heavy="dom_chB",
                       I="dom_hI", II="dom_hII", III="dom_hIII", IV="dom_hIV"),
    }
    labels = ["LC", "HC", "HER2-I", "HER2-II", "HER2-III", "HER2-IV"]
    keys = ["light", "heavy", "I", "II", "III", "IV"]
    cands = ["cand_4", "cand_1", "cand_5", "cand_8"]
    fig, (axb, axd) = plt.subplots(1, 2, figsize=(8.4, 3.5),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    # (a) per-domain RMSD bars
    x = np.arange(len(labels)); w = 0.2
    for j, c in enumerate(cands):
        vals = [win_mean(f"{A}/{c}/{cand_dom[c][k]}.xvg", scale=10.0) for k in keys]
        axb.bar(x + (j - 1.5) * w, vals, w, label=c.replace("cand_", "cand "),
                color=C[c], alpha=0.9)
    axb.axhline(2.0, ls="--", lw=0.8, color="#888")
    axb.text(5.4, 2.05, "2 Å", color="#888", fontsize=7, va="bottom", ha="right")
    axb.set_xticks(x); axb.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axb.set_ylabel("self-fitted Cα RMSD (Å), 25–50 ns")
    axb.set_title("(a) Per-domain RMSD — all folded", fontsize=9.5)
    axb.legend(fontsize=7.2, ncol=2, loc="upper left")
    # (b) DSSP timeseries
    for c in cands:
        fr = dssp_structured(f"{A}/{c}/dssp.dat")
        if fr is None:
            continue
        t = np.linspace(0, 50, len(fr))
        axd.plot(t, fr, color=C[c], lw=1.0, alpha=0.85, label=c.replace("cand_", "cand "))
    axd.set_ylim(30, 55)
    axd.set_xlabel("time (ns)"); axd.set_ylabel("% structured (H/G/I/E/B)")
    axd.set_title("(b) Secondary structure — flat (no unfolding)", fontsize=9.5)
    axd.legend(fontsize=7.2, ncol=2, loc="lower left")
    fig.tight_layout()
    save(fig, "fig3_gross_stability")

# ----------------------------------------------------------------------------- F4
def fig4_artifact():
    # master §20.5-§20.6 (2nd-half means)
    data = {  # metric: (cand4_buggy, cand4_fixed, cand5_buggy, cand5_fixed)
        "MMAE self-RMSD (Å)":  (1.21, 3.51, 4.26, 4.68),
        "MMAE R$_g$ (Å)":       (5.33, 5.65, 6.36, 4.77),
        "lig–prot contacts":    (181, 180, 504, 96),
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1))
    for ax, (metric, (c4b, c4f, c5b, c5f)) in zip(axes, data.items()):
        x = np.arange(2); w = 0.36
        ax.bar(x - w / 2, [c4b, c5b], w, color="#bbbbbb", edgecolor="#333", label="buggy (5-valent C)")
        ax.bar(x + w / 2, [c4f, c5f], w, color=[C["cand_4"], C["cand_5"]], edgecolor="#333", label="fixed (4-valent C)")
        ax.set_xticks(x); ax.set_xticklabels(["cand 4", "cand 5"])
        ax.set_title(metric, fontsize=9)
    axes[0].set_ylabel("2nd-half mean")
    axes[0].legend(fontsize=7, loc="upper left")
    # annotate the inversion
    axes[0].annotate("artifact:\nlooked rigid", xy=(0 + 0.18, 1.21), xytext=(0.1, 2.6),
                     fontsize=6.8, ha="center", color="#333",
                     arrowprops=dict(arrowstyle="->", lw=0.7, color="#333"))
    fig.suptitle("A topology bug created (and inverted) the apparent payload ranking", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_artifact")

# ----------------------------------------------------------------------------- F5 / F6 helpers
def rep_means(cand, lo=25):
    out = []
    for r in ["", "_rep2", "_rep3"]:
        m = win_mean(f"{A}/{cand}/mmae_selfrmsd_prod{r}.xvg", lo=lo, scale=10.0)
        if not np.isnan(m):
            out.append(m)
    return np.array(out)

def fig5_n3():
    c4, c5 = rep_means("cand_4"), rep_means("cand_5")
    t, p = ss.ttest_ind(c4, c5, equal_var=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for i, (c, name) in enumerate([(c4, "cand 4"), (c5, "cand 5")]):
        col = C["cand_4"] if i == 0 else C["cand_5"]
        ax.bar(i, c.mean(), 0.55, yerr=c.std(ddof=1), capsize=4,
               color=col, alpha=0.45, edgecolor="#333", lw=1.0)
        ax.scatter(np.full(len(c), i) + np.linspace(-0.12, 0.12, len(c)), c,
                   color=col, s=40, zorder=5, edgecolor="white", lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cand 4", "cand 5"])
    ax.set_ylabel("MMAE-core self-RMSD (Å), 25–50 ns")
    ax.set_ylim(0, 6)
    ax.text(0.5, 5.6, f"Welch p = {p:.3f}  (N=3)\nno separation",
            ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f3f3f3", ec="#999"))
    ax.set_title("Replicates: payload RMSD does not discriminate", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig5_n3_selfrmsd")
    return c4, c5

def fig6_three_systems(c4, c5):
    ved = rep_means("cand_vedotin")
    p4 = ss.ttest_ind(ved, c4, equal_var=False).pvalue
    p5 = ss.ttest_ind(ved, c5, equal_var=False).pvalue
    order = [("cand 4", c4, C["cand_4"]), ("vedotin\n(FDA)", ved, C["vedotin"]),
             ("cand 5", c5, C["cand_5"])]
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i, (name, d, col) in enumerate(order):
        ax.bar(i, d.mean(), 0.55, yerr=d.std(ddof=1), capsize=4,
               color=col, alpha=0.40, edgecolor="#333", lw=1.0)
        ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d,
                   color=col, s=42, zorder=5, edgecolor="white", lw=0.6)
    ax.axhspan(min(c4.min(), c5.min(), ved.min()), max(c4.max(), c5.max(), ved.max()),
               color="#999", alpha=0.06)
    ax.set_xticks(range(3)); ax.set_xticklabels([o[0] for o in order])
    ax.set_ylabel("MMAE-core self-RMSD (Å), 25–50 ns")
    ax.set_ylim(0, 6)
    ax.text(0.5, 5.7, f"vedotin vs cand 4: p = {p4:.2f}", ha="center", fontsize=8)
    ax.text(0.5, 5.25, f"vedotin vs cand 5: p = {p5:.2f}", ha="center", fontsize=8)
    ax.text(1.0, 0.35, "FDA linker lands INSIDE the candidate envelope\n→ metric non-discriminative",
            ha="center", fontsize=7.8, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff4e6", ec=C["accent"]))
    ax.set_title("Positive control: an approved ADC linker is indistinguishable", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig6_vedotin_control")

# ----------------------------------------------------------------------------- F7
def fig7_site():
    # master §20.13
    fig, (axs, axd) = plt.subplots(1, 2, figsize=(8.0, 3.2),
                                   gridspec_kw=dict(width_ratios=[1.1, 1.0]))
    # static: distance from Cys214 SG to nearest charged residues
    res = ["Glu213\n(−)", "Asp122\n(−)", "Arg211\n(+)", "Lys136\n(+)"]
    dist = [4.5, 8.0, 8.9, 10.3]
    cols = ["#D55E00", "#D55E00", "#0072B2", "#0072B2"]
    axs.barh(range(len(res)), dist, color=cols, alpha=0.85, edgecolor="#333")
    axs.axvline(6.0, ls="--", lw=0.9, color="#888")
    axs.text(6.1, -0.4, "catalytic\nrange ~5-7 Å", fontsize=6.8, color="#666", va="bottom")
    axs.set_yticks(range(len(res))); axs.set_yticklabels(res, fontsize=7.6)
    axs.set_xlabel("distance to Cys214 SG (Å)")
    axs.set_title("(a) Static site — acidic, no proximal base", fontsize=9)
    axs.invert_yaxis()
    axs.text(0.5, len(res) - 0.6, "pKa(SG) = 9.88", fontsize=7.6, color="#333")
    # dynamic: succinimide -- nearest basic-N distance distribution summary
    axd.axis("off")
    txt = ("(b) Dynamic (vedotin, 50 ns)\n\n"
           "succinimide ↔ nearest basic-N:\n"
           "   mean 5.5 Å,  min 2.6 Å\n"
           "   54 % of frames < 5 Å\n"
           "   8 % < 4 Å\n\n"
           "contacts: Fab Lys/Arg only\n(transient, flexible-linker driven)\n"
           "→ no PERSISTENT proximal base\n→ weak self-stabilisation predicted")
    axd.text(0.05, 0.95, txt, va="top", ha="left", fontsize=8.0,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f3f3f3", ec="#999"))
    fig.suptitle("Cys214 conjugation-site microenvironment (proxy site)", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "fig7_site_microenv")

if __name__ == "__main__":
    fig1_pipeline()
    fig2_yield_funnel()
    fig3_gross_stability()
    fig4_artifact()
    c4, c5 = fig5_n3()
    fig6_three_systems(c4, c5)
    fig7_site()
    print("\nAll figures in", OUT)
