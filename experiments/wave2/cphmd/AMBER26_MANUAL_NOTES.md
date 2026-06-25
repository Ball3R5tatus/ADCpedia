# Amber26 Reference Manual — notes relevant to our CpHMD work
(Manual at /home/galeito/Amber26/Amber26.pdf, 1112 pp; full text dump was /tmp/amber26.txt)

## Discrete CpHMD (Ch. 28, Mongan/Case MC method) — what our GB pilot uses
- Implicit solvent: `icnstph=1`, set pH via `solvph`, MC attempts every `ntcnstph` steps.
  Guidance: ~100 fs effective period per residue (ntcnstph=5, dt=0.002 ≈ 10 residues).
- REFERENCE ENERGIES (in the cpin) were derived with EXACTLY:
  `cut=30.0, igb=<cpinutil -igb>, saltcon=0.1, nrespa=1, temp0=300.0, ntc=2, ntf=2`.
  → our mdin matches these → results are valid. nrespa must NEVER change.
- ff99SB-derived; **ff14SB is charge-compatible** (manual says so) → valid.
- cpinutil.py: `-igb 2` default; restrict residues with `-resnames/-resnums`; for
  EXPLICIT solvent must use `-op` (special carboxylate radii) — NOT needed for GB.
- Flags: `-cpin`, `-cpout` (protonation history), `-cprestrt` (restart; use as cpin on restart).
- Analyze (28.4/28.7): cphstats reads cpin+cpout → fraction protonated → pKa.

## GPU constant-pH (production speed) — needs pmemd.cuda (licensed = pmemd26 free non-commercial)
- Continuous CpHMD (Ch. 30, λ-dynamics, Chen/Shen): "All-atom PME continuous constant pH MD
  supported on GPUs" (pmemd.cuda). Different method from Ch.28 discrete MC.
- Explicit-solvent discrete CpHMD: `icnstph=2, ntcnstph=100, ntrelax=...`; pH-REMD (Ch.27.4.3)
  across pH is the efficient titration (genremdinputs.py).
- pmemd.cuda can run constant pH (manual §28.3 example uses pmemd.cuda).
- GB on GPU is only for SMALL implicit systems. Our 280k explicit system → pmemd.cuda PME.

## pmemd26 build (the GPU path we lack)
- pmemd26.tar.bz2 is SELF-CONTAINED in Amber26 (no AmberTools source needed); free for
  non-commercial. Our Amber22.tar.bz2 = pmemd22 (NOT self-contained, wrong version for
  AmberTools26). To get GPU: download pmemd26 (free non-commercial form) → build with CUDA.

## Bottom line for the pilot
Our GB discrete CpHMD (sander CPU, icnstph=1, the params above) is correctly configured per
the manual and runnable now. GPU explicit-solvent / continuous CpHMD would need pmemd26.
