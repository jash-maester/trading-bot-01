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
# EXACT PARITY WITH r4_v2 on everything except the universe. Three settings had
# to be stated rather than inherited, and each would have broken the comparison
# in a different way:
#
#   n_windows=12   configs/walk/default.yaml says 4. Four windows gives the
#                  gate df=3 and t_crit 3.182 instead of df=7 and 2.365, over a
#                  span ending 2020 -- a harder test reported against r4_v2's
#                  number as if it were the same one.
#   data_end       unset, the bhav panel runs to 2026-09 and TEN windows fit,
#                  two of which test on 2024-07..2026-06. That is the 2025+
#                  holdout Phase 4 depends on, consumed as training evidence.
#                  train_signal.py documents this exact trap.
#   data_start     the panel opens 2010-01-04; pinning it keeps window
#                  boundaries identical to r4_v2's.
if uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
     walk.n_windows=12 walk.data_start=2010-01-01 walk.data_end=2024-12-31 \
     > "logs/${TAG}_train.log" 2>&1; then
    stamp "train OK"
    grep -E "R4 GATE|horizon|universe:" "logs/${TAG}_train.log" | tail -20
else
    stamp "train FAIL"; tail -40 "logs/${TAG}_train.log"; exit 1
fi
stamp "chain DONE"
say "complete — status in $STATUS"
