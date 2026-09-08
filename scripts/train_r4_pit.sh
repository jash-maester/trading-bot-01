#!/usr/bin/env bash
# Phase 2: R4 signal on a point-in-time universe.
#
#   RUN_TAG=r4pit bash scripts/train_r4_pit.sh
#
# ~3 hours for 8 windows, matching r4_v2 (04:30 -> 07:17 on 2026-09-06). The
# ticker axis is the same 504 per window, so the per-step cost is unchanged;
# only WHICH names occupy it moves.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-r4pit}"
STATUS="logs/r4pit_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

say "R4 walk-forward on the point-in-time universe"
if uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
     > "logs/${TAG}_train.log" 2>&1; then
    stamp "train OK"
    grep -E "R4 GATE|horizon|universe:" "logs/${TAG}_train.log" | tail -20
else
    stamp "train FAIL"; tail -40 "logs/${TAG}_train.log"; exit 1
fi
stamp "chain DONE"
say "complete — status in $STATUS"
