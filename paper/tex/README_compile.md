# Compiling the manuscript

`manuscript.tex` is a **modern two-column coloured preprint template** (custom, `article`-based —
chemRxiv accepts any PDF). TeX Live is installed locally; it compiles to **`manuscript.pdf` (7 pages)**.

## Build
```bash
cd paper/tex
latexmk -pdf -bibtex manuscript.tex      # one command; runs bibtex + reruns automatically
# or:  pdflatex manuscript ; bibtex manuscript ; pdflatex manuscript ; pdflatex manuscript
```
Clean intermediates: `latexmk -C manuscript`.

## What it needs (all standard; pre-installed on Overleaf)
`lmodern, microtype, xcolor, colortbl, booktabs, array, enumitem, titlesec, caption, tcolorbox,
fancyhdr, natbib, hyperref, mhchem, graphicx`. Bibliography style: `unsrtnat` (numbered).

Locally these came from: `texlive-latex-base/-recommended/-extra`, `texlive-publishers`,
`texlive-science`, `texlive-bibtex-extra`, `lmodern`, `latexmk` (+ `poppler-utils` only for PDF preview).

## Files
- `manuscript.tex` — the paper (two-column, coloured palette navy/blue/orange).
- `references.bib` — 15 entries. **Verify DOIs/pages before submission** (see header comment;
  esp. `igashov2024`, `mslati2026`).
- Figures: vector PDFs in `../figures/` via `\graphicspath{{../figures/}}` (regenerate with
  `python3 ../figures/make_figures.py`).

## On Overleaf
Upload `manuscript.tex` + `references.bib`, and the `figures/*.pdf` into a `figures/` subfolder
(or change `\graphicspath` to `{{figures/}}`). Compiler: **pdfLaTeX**.

## Customising the look
- Palette: the `\definecolor` block (`ink`, `accent`, `accent2`, `tint`).
- Single-column instead: change `\documentclass[10pt,twocolumn]{article}` → `[11pt]{article}`,
  and `figure*`/`table*` → `figure`/`table`.
- The "Key result" callout: the `keybox` `tcolorbox` (after the positive-control result).
