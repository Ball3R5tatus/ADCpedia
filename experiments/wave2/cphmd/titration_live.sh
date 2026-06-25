#!/usr/bin/env bash
# Live auto-refreshing GB CpHMD titration dashboard (cand_4).
# Re-renders titration_dashboard.sh every N seconds in place (default 5s).
#   bash titration_live.sh        # refresh every 5 s
#   bash titration_live.sh 10     # refresh every 10 s
# Press Ctrl+C to stop.
INT=${1:-5}
DASH="$(cd "$(dirname "$0")" && pwd)/titration_dashboard.sh"
trap 'printf "\n  (live monitor stopped)\n"; exit 0' INT TERM
command -v tput >/dev/null && tput civis 2>/dev/null   # hide cursor
while true; do
  out="$(bash "$DASH" 2>/dev/null)"
  clear 2>/dev/null || printf '\033[H\033[2J'
  printf '%s\n' "$out"
  printf "\n  (LIVE — refresh every %ss — Ctrl+C to stop)\n" "$INT"
  sleep "$INT"
done
