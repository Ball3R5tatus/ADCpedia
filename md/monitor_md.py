#!/usr/bin/env python3
"""Real-time GROMACS production monitor for the ADC pilot/batch runs.

Reads prod.log (step + simulated time), computes instantaneous + average
ns/day, progress %, and ETA. READ-ONLY — does not touch the running GPU job,
safe to run in a separate WSL terminal while mdrun is going.

Usage:
    python3 monitor_md.py [cid]      # default cid=4
    (Ctrl-C stops the MONITOR, not the simulation)

Optional: pass a different log name as 2nd arg if you used -deffnm something
    python3 monitor_md.py 4 prod
"""
import sys, os, re, time
from collections import deque

cid     = sys.argv[1] if len(sys.argv) > 1 else "4"
deffnm  = sys.argv[2] if len(sys.argv) > 2 else "prod"
RUN     = os.path.expanduser(f"~/ADCpedia/md/runs/cand_{cid}")
LOG     = f"{RUN}/{deffnm}.log"

DT_PS    = 0.002            # timestep (ps) — from the mdp
NSTEPS   = 25_000_000       # total steps (50 ns at 2 fs) — from prod.tpr
TOTAL_PS = NSTEPS * DT_PS   # 50000 ps
REFRESH  = 20               # seconds between screen updates


def latest_step_time():
    """Return (step, sim_time_ps) from the most recent 'Step Time' block."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 30000))      # tail ~30 KB is plenty
            tail = f.read().decode(errors="ignore")
    except FileNotFoundError:
        return None, None, False
    finished = "Finished mdrun" in tail
    # GROMACS writes:  "           Step           Time"
    #                  "        790000     1580.00000"
    pairs = re.findall(r"^\s*(\d+)\s+([\d.]+)\s*$", tail, re.M)
    best = None
    for s, t in pairs:
        s, t = int(s), float(t)
        if abs(t - s * DT_PS) < 1.0:          # sanity: t ≈ step*dt
            best = (s, t)
    if best:
        return best[0], best[1], finished
    return None, None, finished


def fmt_eta(hours):
    if hours == float("inf"):
        return "  --  "
    if hours < 1:
        return f"{hours*60:4.0f}min"
    return f"{hours:5.1f}h"


hist = deque(maxlen=12)        # (wall, step) for the moving-average rate
start_wall = time.time()
start_step = None

print(f"Monitoring cand_{cid} ({deffnm}.log) — refresh {REFRESH}s — "
      f"Ctrl-C stops the monitor (sim keeps running)\n")

try:
    while True:
        step, simt, finished = latest_step_time()
        now = time.time()

        if step is None:
            print("  waiting for prod.log step data ...", end="\r", flush=True)
            time.sleep(REFRESH)
            continue

        if start_step is None:
            start_step = step
        hist.append((now, step))

        # instantaneous ns/day over the moving window
        nsday_inst = 0.0
        if len(hist) >= 2:
            dw = hist[-1][0] - hist[0][0]
            ds = hist[-1][1] - hist[0][1]
            if dw > 0:
                nsday_inst = (ds * DT_PS / 1000) / (dw / 86400)

        # average ns/day since this monitor started
        nsday_avg = 0.0
        dw_tot = now - start_wall
        if dw_tot > 5:
            nsday_avg = ((step - start_step) * DT_PS / 1000) / (dw_tot / 86400)

        pct = 100 * simt / TOTAL_PS
        remaining_ns = (TOTAL_PS - simt) / 1000
        eta_h = remaining_ns / nsday_inst * 24 if nsday_inst > 0 else float("inf")

        bar_len = 28
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)

        line = (f"[{bar}] {pct:5.1f}%  "
                f"{simt/1000:5.2f}/{TOTAL_PS/1000:.0f} ns  "
                f"inst {nsday_inst:5.1f}  avg {nsday_avg:5.1f} ns/day  "
                f"ETA {fmt_eta(eta_h)}   ")
        print("\r" + line, end="", flush=True)

        if finished:
            print("\n\n*** Finished mdrun — production complete. ***")
            break

        time.sleep(REFRESH)

except KeyboardInterrupt:
    print("\n\nMonitor stopped. The simulation continues in the background.")
