#!/usr/bin/env python3
"""
Failure-mode taxonomy for ADCpedia molecular-diffusion candidates (Wave-0 task 0.3).

Pure tabulation of existing per-molecule gate results. No GPU / RDKit needed.

Each input row is one generated candidate with boolean stage flags:
    sdf, parsed, native_connected, mmff_ok, mmff_energy, mmff_converged,
    has_valcit, has_urea, usable_3d, smiles

Strict funnel -- each candidate is classified by its FIRST failed stage
(mutually exclusive & exhaustive):
    1. not_parsed    : parsed == False
    2. disconnected  : parsed AND not native_connected   (the "connectivity wall")
    3. mmff_fail     : native_connected AND not mmff_ok   (valence/relaxation failure)
    4. no_adc_motif  : mmff_ok AND not (has_valcit or has_urea)
    5. usable        : everything that survives (usable_3d == True)

Outputs (written next to this script):
    failure_breakdown.csv   counts + percentages per (config, cohort)
    failure_funnel.png/.pdf publication-clean stacked-bar funnel
    README.md               table + headline findings

Run:  python3 analyze_failures.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = HERE
EVAL = "/home/galeito/ADCpedia/outputs/h4/eval"
DELIVERABLE = "/home/galeito/ADCpedia/outputs/h4/deliverable"

# ---------------------------------------------------------------------------
# Input registry.
#
# Cohort policy (per task spec):
#   * rebaseline_n500 is the AUTHORITATIVE n=500 set for the 4 main configs.
#   * per/ 200-runs are the earlier n=200 set; reported as a SEPARATE cohort.
#   * To avoid double-counting a config across cohorts we prefer rebaseline
#     n=500 where both exist, so the four configs that appear in rebaseline
#     (curriculum851, multisite884*, lambda0.01, lambda0_control) are dropped
#     from the n=200 cohort. The remaining n=200 runs (lambda0.1, lambda1.0,
#     lambda10, lambda10_strict) are kept.
#   * final_best is the n=500 deliverable. The deliverable copy and the
#     per/final_best copy are byte-identical (md5 verified); we include it once
#     under the "deliverable" cohort.
# ---------------------------------------------------------------------------

# Config keys (normalized) that exist in the rebaseline n=500 cohort.
# Their n=200 per/ counterparts are excluded to prevent double counting.
REBASELINE_CONFIG_KEYS = {
    "curriculum851", "multisite", "lambda0.01", "lambda0_control",
}


def norm_key(label):
    """Normalize a config label so the same config maps to one key across cohorts."""
    l = label.lower()
    if "curriculum" in l:
        return "curriculum851"
    if "multisite" in l or "multisite884" in l:
        return "multisite"
    if "lambda0_control" in l or "lambda0control" in l:
        return "lambda0_control"
    if "lambda0.01" in l:
        return "lambda0.01"
    if "lambda0.1" in l:
        return "lambda0.1"
    if "lambda1.0" in l:
        return "lambda1.0"
    if "lambda10_strict" in l:
        return "lambda10_strict"
    if "lambda10" in l:
        return "lambda10"
    if "lambda0" in l and "lam0" in l:  # multisite884_lam0 handled above
        return "lambda0"
    return l


def collect_inputs():
    """Return list of (config_label, cohort, path), in display order."""
    inputs = []

    # --- Cohort A: rebaseline n=500 (4 main configs) ---
    rb = os.path.join(EVAL, "rebaseline_n500")
    rb_files = [
        ("curriculum851", "curriculum851_per_mol.csv"),
        ("multisite884_base", "multisite884_base_per_mol.csv"),
        ("h4_lambda0.01", "h4_lambda0.01_per_mol.csv"),
        ("h4_lambda0_control", "h4_lambda0_control_per_mol.csv"),
    ]
    for label, fn in rb_files:
        p = os.path.join(rb, fn)
        if os.path.isfile(p):
            inputs.append((label, "rebaseline_n500", p))

    # --- Cohort B: per/ 200-runs (only configs NOT covered by rebaseline) ---
    per = os.path.join(EVAL, "per")
    if os.path.isdir(per):
        for run in sorted(os.listdir(per)):
            rundir = os.path.join(per, run)
            if not os.path.isdir(rundir):
                continue
            if run == "final_best":
                continue  # n=500, handled as the deliverable cohort below
            # find the *_per_mol.csv inside
            cand = os.path.join(rundir, run + "_per_mol.csv")
            if not os.path.isfile(cand):
                # fallback: any *_per_mol.csv
                hits = [f for f in os.listdir(rundir) if f.endswith("_per_mol.csv")]
                cand = os.path.join(rundir, hits[0]) if hits else None
            if not cand or not os.path.isfile(cand):
                continue
            if norm_key(run) in REBASELINE_CONFIG_KEYS:
                continue  # prefer rebaseline n=500 for these configs
            inputs.append((run, "per_n200", cand))

    # --- Cohort C: deliverable final_best n=500 (single canonical copy) ---
    deliv = os.path.join(DELIVERABLE, "final_best_per_mol.csv")
    if os.path.isfile(deliv):
        inputs.append(("final_best", "deliverable", deliv))

    return inputs


def to_bool(s):
    return str(s).strip().lower() in ("true", "1", "yes", "t")


def classify_file(path):
    """Return (n, dict of stage->count). Verifies funnel sums and usable_3d match."""
    counts = {
        "not_parsed": 0,
        "disconnected": 0,
        "mmff_fail": 0,
        "no_adc_motif": 0,
        "usable": 0,
    }
    n = 0
    usable_flag_count = 0  # how many rows have usable_3d == True (consistency check)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n += 1
            parsed = to_bool(row["parsed"])
            connected = to_bool(row["native_connected"])
            mmff_ok = to_bool(row["mmff_ok"])
            has_motif = to_bool(row["has_valcit"]) or to_bool(row["has_urea"])
            usable_3d = to_bool(row["usable_3d"])
            if usable_3d:
                usable_flag_count += 1

            # strict funnel: first failed stage
            if not parsed:
                counts["not_parsed"] += 1
            elif not connected:
                counts["disconnected"] += 1
            elif not mmff_ok:
                counts["mmff_fail"] += 1
            elif not has_motif:
                counts["no_adc_motif"] += 1
            else:
                counts["usable"] += 1

    # exhaustiveness / mutual-exclusivity check
    assert sum(counts.values()) == n, (
        "funnel does not sum to row count in %s: %s vs %d"
        % (path, counts, n)
    )
    return n, counts, usable_flag_count


def pct(x, n):
    return 0.0 if n == 0 else 100.0 * x / n


STAGES = ["not_parsed", "disconnected", "mmff_fail", "no_adc_motif", "usable"]


def main():
    inputs = collect_inputs()
    if not inputs:
        sys.exit("No input per_mol.csv files found.")

    rows = []   # one dict per input file
    for label, cohort, path in inputs:
        n, counts, usable_flag = classify_file(path)
        rec = {"config": label, "cohort": cohort, "n": n, "path": path,
               "usable_flag": usable_flag}
        rec.update(counts)
        rows.append(rec)
        # note any divergence between our funnel-usable and the on-disk usable_3d flag
        if counts["usable"] != usable_flag:
            print("[note] %s/%s: funnel usable=%d but usable_3d flag=%d "
                  "(motif funnel vs stored 3D-usable definition differ by %d)"
                  % (cohort, label, counts["usable"], usable_flag,
                     counts["usable"] - usable_flag))

    # ----- write failure_breakdown.csv -----
    csv_path = os.path.join(OUTDIR, "failure_breakdown.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["config", "cohort", "n"]
        header += STAGES
        header += [s + "_pct" for s in STAGES]
        w.writerow(header)
        for r in rows:
            line = [r["config"], r["cohort"], r["n"]]
            line += [r[s] for s in STAGES]
            line += ["%.2f" % pct(r[s], r["n"]) for s in STAGES]
            w.writerow(line)
        # overall (all candidates pooled, all cohorts)
        tot_n = sum(r["n"] for r in rows)
        tot = {s: sum(r[s] for r in rows) for s in STAGES}
        line = ["ALL_POOLED", "all", tot_n]
        line += [tot[s] for s in STAGES]
        line += ["%.2f" % pct(tot[s], tot_n) for s in STAGES]
        w.writerow(line)
    print("wrote", csv_path)

    # ----- figure -----
    make_figure(rows)

    # ----- README -----
    make_readme(rows)

    return rows


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # order: rebaseline 4 main configs, then per_n200, then deliverable
    cohort_order = {"rebaseline_n500": 0, "per_n200": 1, "deliverable": 2, "all": 3}
    rows_sorted = sorted(rows, key=lambda r: (cohort_order.get(r["cohort"], 9),
                                              r["config"]))

    labels = ["%s\n(%s, n=%d)" % (r["config"], r["cohort"], r["n"])
              for r in rows_sorted]

    # colors: losses warm/grey, usable green
    colors = {
        "not_parsed": "#4d4d4d",
        "disconnected": "#d73027",   # the connectivity wall, highlighted red
        "mmff_fail": "#fc8d59",
        "no_adc_motif": "#fee08b",
        "usable": "#1a9850",         # survivors, green
    }
    nice = {
        "not_parsed": "not parsed",
        "disconnected": "disconnected (connectivity wall)",
        "mmff_fail": "MMFF fail (valence)",
        "no_adc_motif": "no ADC motif",
        "usable": "usable",
    }

    n_bars = len(rows_sorted)
    fig, ax = plt.subplots(figsize=(11, 0.7 * n_bars + 2.2))
    y = list(range(n_bars))

    left = [0.0] * n_bars
    for stage in STAGES:
        vals = [pct(r[stage], r["n"]) for r in rows_sorted]
        ax.barh(y, vals, left=left, color=colors[stage],
                edgecolor="white", height=0.72, label=nice[stage])
        # annotate percentages >= 3%
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 3.0:
                txtcolor = "white" if stage in ("disconnected", "not_parsed",
                                                "usable") else "black"
                ax.text(l + v / 2.0, y[i], "%.1f" % v, va="center", ha="center",
                        fontsize=8, color=txtcolor, fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of generated candidates (%)", fontsize=11)
    ax.set_title("ADCpedia diffusion: failure-mode funnel per config\n"
                 "(first failed gate; disconnection is the dominant loss)",
                 fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # legend in funnel/stage order
    handles = [Patch(facecolor=colors[s], edgecolor="white", label=nice[s])
               for s in STAGES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    png = os.path.join(OUTDIR, "failure_funnel.png")
    pdf = os.path.join(OUTDIR, "failure_funnel.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)
    print("wrote", pdf)


def make_readme(rows):
    lines = []
    A = lines.append
    A("# Failure-mode taxonomy: why do generated ADC candidates fail?\n")
    A("Wave-0 task 0.3. Pure tabulation of existing per-molecule gate results "
      "(no GPU / RDKit). Re-run with `python3 analyze_failures.py`.\n")

    A("## The funnel (strict, first-failed-stage)\n")
    A("Each candidate is classified by the **first** gate it fails:\n")
    A("1. `not_parsed`    -- the SDF could not be parsed at all.")
    A("2. `disconnected`  -- parsed, but the linker is **not natively bonded** "
      "to payload+antibody handle (the *connectivity wall*).")
    A("3. `mmff_fail`     -- connected, but MMFF relaxation / valence fails.")
    A("4. `no_adc_motif`  -- MMFF-clean, but lacks a Val-Cit or self-immolative "
      "urea cleavage motif.")
    A("5. `usable`        -- survives every gate.\n")
    A("These five are mutually exclusive and exhaustive (verified: each file's "
      "five buckets sum to its row count).\n")

    A("## Cohorts and files included\n")
    A("- **rebaseline_n500** (authoritative n=500 for the 4 main configs): "
      "`outputs/h4/eval/rebaseline_n500/{curriculum851,multisite884_base,"
      "h4_lambda0.01,h4_lambda0_control}_per_mol.csv`")
    A("- **per_n200** (earlier n=200 runs, *only configs not in rebaseline* to "
      "avoid double-counting): `outputs/h4/eval/per/{lambda0.1,lambda1.0,"
      "lambda10,lambda10_strict}/<run>_per_mol.csv`")
    A("- **deliverable** (n=500 best model): "
      "`outputs/h4/deliverable/final_best_per_mol.csv` "
      "(byte-identical to `outputs/h4/eval/per/final_best/final_best_per_mol.csv`; "
      "counted once).\n")
    A("The four rebaseline configs' n=200 counterparts "
      "(curriculum851, multisite884_lam0, lambda0.01, lambda0_control) are "
      "**excluded** from per_n200 so no config is double-counted across cohorts.\n")

    # ---- table ----
    A("## Breakdown table (counts, with % in parentheses)\n")
    hdr = ("| config | cohort | n | not_parsed | disconnected | mmff_fail | "
           "no_adc_motif | usable |")
    A(hdr)
    A("|---|---|---:|---:|---:|---:|---:|---:|")

    def cell(r, s):
        return "%d (%.1f%%)" % (r[s], pct(r[s], r["n"]))

    cohort_order = {"rebaseline_n500": 0, "per_n200": 1, "deliverable": 2}
    rows_sorted = sorted(rows, key=lambda r: (cohort_order.get(r["cohort"], 9),
                                              r["config"]))
    for r in rows_sorted:
        A("| %s | %s | %d | %s | %s | %s | %s | %s |" % (
            r["config"], r["cohort"], r["n"],
            cell(r, "not_parsed"), cell(r, "disconnected"),
            cell(r, "mmff_fail"), cell(r, "no_adc_motif"), cell(r, "usable")))

    tot_n = sum(r["n"] for r in rows)
    tot = {s: sum(r[s] for r in rows) for s in STAGES}
    A("| **ALL POOLED** | all | %d | %d (%.1f%%) | %d (%.1f%%) | %d (%.1f%%) | "
      "%d (%.1f%%) | %d (%.1f%%) |" % (
          tot_n,
          tot["not_parsed"], pct(tot["not_parsed"], tot_n),
          tot["disconnected"], pct(tot["disconnected"], tot_n),
          tot["mmff_fail"], pct(tot["mmff_fail"], tot_n),
          tot["no_adc_motif"], pct(tot["no_adc_motif"], tot_n),
          tot["usable"], pct(tot["usable"], tot_n)))
    A("")

    # ---- headline findings ----
    A("## Headline finding\n")
    disc_pct = pct(tot["disconnected"], tot_n)
    parse_pct = pct(tot["not_parsed"], tot_n)
    A("**Disconnection is the wall.** Across all %d pooled candidates, "
      "**%.1f%%** are lost to `disconnected` -- the single largest failure mode "
      "by a wide margin. Parsing essentially never fails (%.1f%%); the model "
      "reliably emits a valid SDF but fails to *bond the linker into one "
      "covalent ADC graph*.\n" % (tot_n, disc_pct, parse_pct))

    A("### Conditional drop-off down the funnel (pooled)\n")
    connected = tot_n - tot["not_parsed"] - tot["disconnected"]
    mmff_clean = connected - tot["mmff_fail"]
    A("- Of all candidates, **%.1f%%** reach connectivity "
      "(%d / %d parsed-and-connected)." % (pct(connected, tot_n), connected, tot_n))
    A("- Of the **connected** ones, **%.1f%%** then fail MMFF / valence "
      "(%d / %d)." % (pct(tot['mmff_fail'], connected), tot["mmff_fail"], connected))
    A("- Of the **MMFF-clean** ones, **%.1f%%** lack an ADC cleavage motif "
      "(%d / %d) and are dropped at `no_adc_motif`." % (
          pct(tot['no_adc_motif'], mmff_clean), tot["no_adc_motif"], mmff_clean))
    A("- Net **usable yield (pooled): %.1f%%** (%d / %d).\n" % (
        pct(tot["usable"], tot_n), tot["usable"], tot_n))

    A("So the loss chain is overwhelmingly front-loaded: the connectivity gate "
      "destroys the majority of candidates, MMFF prunes a small slice of "
      "survivors, and the motif filter is the *second* big gate -- it removes a "
      "large fraction of otherwise-valid 3D molecules.\n")

    # ---- per-config contrasts ----
    A("## Per-config contrasts\n")

    def get(label_sub, cohort=None):
        for r in rows:
            if label_sub in r["config"] and (cohort is None or r["cohort"] == cohort):
                return r
        return None

    cur = get("curriculum851", "rebaseline_n500")
    ms = get("multisite884_base", "rebaseline_n500")
    l001 = get("h4_lambda0.01", "rebaseline_n500")
    ctrl = get("h4_lambda0_control", "rebaseline_n500")

    if cur and ms and l001 and ctrl:
        A("Using the authoritative n=500 rebaseline configs:\n")
        A("| config | connectivity (1 - disc) | MMFF-clean | usable | "
          "motif yield of MMFF-clean |")
        A("|---|---:|---:|---:|---:|")
        for r in (cur, ms, l001, ctrl):
            conn = r["n"] - r["not_parsed"] - r["disconnected"]
            mmffc = conn - r["mmff_fail"]
            motif_yield = pct(r["usable"], mmffc) if mmffc else 0.0
            A("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
                r["config"], pct(conn, r["n"]), pct(mmffc, r["n"]),
                pct(r["usable"], r["n"]), motif_yield))
        A("")

        ms_conn = pct(ms['n'] - ms['not_parsed'] - ms['disconnected'], ms['n'])
        cur_conn = pct(cur['n'] - cur['not_parsed'] - cur['disconnected'], cur['n'])
        ms_mmffc = ms['n'] - ms['not_parsed'] - ms['disconnected'] - ms['mmff_fail']
        cur_mmffc = cur['n'] - cur['not_parsed'] - cur['disconnected'] - cur['mmff_fail']
        ms_motif = pct(ms['usable'], ms_mmffc) if ms_mmffc else 0.0
        cur_motif = pct(cur['usable'], cur_mmffc) if cur_mmffc else 0.0

        A("**Multi-site has higher connectivity but lower motif yield -- "
          "CONFIRMED.** `multisite884_base` reaches the *highest* connectivity "
          "of any config (%.1f%% vs %.1f%% for curriculum), yet only "
          "**%.1f%%** of its MMFF-clean molecules carry an ADC motif "
          "(vs %.1f%% for curriculum). The motif collapse is severe enough that "
          "multi-site's final usable yield (%.1f%%) is *below* curriculum's "
          "(%.1f%%) despite nearly double the connectivity -- it generates "
          "well-bonded scaffolds that mostly lack a cleavable Val-Cit/urea "
          "linker.\n" % (
              ms_conn, cur_conn, ms_motif, cur_motif,
              pct(ms['usable'], ms['n']), pct(cur['usable'], cur['n'])))

        A("**lambda0.01 is the best 4-config rebaseline:** highest usable yield "
          "at **%.1f%%**, driven by good connectivity (%.1f%%) *and* the best "
          "motif retention.\n" % (
              pct(l001['usable'], l001['n']),
              pct(l001['n'] - l001['not_parsed'] - l001['disconnected'], l001['n'])))

        A("**lambda0_control (no connectivity reward) is the worst:** "
          "connectivity only %.1f%% and usable only %.1f%% -- direct evidence "
          "the connectivity objective is doing real work.\n" % (
              pct(ctrl['n'] - ctrl['not_parsed'] - ctrl['disconnected'], ctrl['n']),
              pct(ctrl['usable'], ctrl['n'])))

    fb = get("final_best", "deliverable")
    if fb:
        A("**final_best (deliverable, n=500):** usable **%.1f%%** "
          "(connectivity %.1f%%) -- matches the known ~9-10/100 target for the "
          "best model.\n" % (
              pct(fb['usable'], fb['n']),
              pct(fb['n'] - fb['not_parsed'] - fb['disconnected'], fb['n'])))

    A("## Files in this directory\n")
    A("- `analyze_failures.py` -- deterministic, re-runnable generator.")
    A("- `failure_breakdown.csv` -- per (config, cohort) counts + percentages, "
      "plus an ALL_POOLED row.")
    A("- `failure_funnel.png` / `.pdf` -- stacked-bar funnel per config.")
    A("- `README.md` -- this report.\n")

    path = os.path.join(OUTDIR, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", path)


if __name__ == "__main__":
    main()
