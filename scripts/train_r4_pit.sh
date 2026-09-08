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
# walk.n_windows=12 because configs/walk/default.yaml sets 4, and r4_v2 ran 12
# (truncating to 8 usable). Four windows would give the gate's t-test df=3 and
# t_crit 3.182 instead of df=7 and 2.365, on a span ending 2020 instead of 2024
# -- a different and much harder test, reported against r4_v2's number as if it
# were the same one.
if uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
     walk.n_windows=12 \
     > "logs/${TAG}_train.log" 2>&1; then
    stamp "train OK"
    grep -E "R4 GATE|horizon|universe:" "logs/${TAG}_train.log" | tail -20
else
    stamp "train FAIL"; tail -40 "logs/${TAG}_train.log"; exit 1
fi
stamp "chain DONE"
say "complete — status in $STATUS"
