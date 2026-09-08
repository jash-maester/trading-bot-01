#!/usr/bin/env bash
# Can the allocator be made to beat equal-weight on a point-in-time universe?
#
#   KSET=20,30,50 RUN_TAG=swA bash scripts/pit_sweep.sh
#
# Phase 3's null control decomposed the failure exactly:
#
#   equal_weight   0.119
#   allocator      0.105   <- signal recovers +2.7pp
#   null_signal    0.078   <- concentration + turnover costs -4.1pp
#
# The signal is worth 2.7pp over random selection; concentrating into 20 of
# ~350 names and trading 3.45x turnover costs 4.1pp. Net -1.4pp. So the fix, if
# there is one, is to cut the cost of concentration rather than to find more
# signal -- and there are exactly two levers.
#
#   K      top-20 of ~350 is the extreme 6% tail, where prediction error
#          dominates and idiosyncratic variance is highest. An IC of +0.0225 is
#          better harvested broadly.
#   band   turnover is 3.45 against equal-weight's 1.10, and the demat bill is
#          Rs 113,409 against Rs 57,694 -- roughly 1.4%/yr of the gap. The
#          no-trade band exists, is set to 0.0, and has never been swept here.
#
# Split across two processes by K because the box has 20 cores and the
# allocator is CPU-bound.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-sw}"
KSET="${KSET:-20,30,50}"
SIGNAL="${SIGNAL_TAG:-r4_pit}"
STATUS="logs/sweep_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

say "sweep K=[${KSET}] x band=[0,0.005,0.01] on the point-in-time universe"
if uv run python scripts/run_allocator.py data=bhav_v1 \
      +split="oos_${SIGNAL}" +signal_tag="$SIGNAL" +require_gate_pass=true \
      +apply_tax=true \
      ++allocator.universe_from_panel=true \
      ++allocator.null_control=true \
      ++allocator.k_grid="[${KSET}]" \
      ++allocator.band_grid=[0.0,0.005,0.01] \
      ++allocator.freq_grid=[monthly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.risk_grid='[{name: none}, {name: volstop, stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 21}]' \
      ++allocator.nav_dir="audit/navs_${TAG}" \
      > "logs/${TAG}_sweep.log" 2>&1; then
    stamp "sweep OK"
    grep -E "^equal_weight|^null_signal|^allocator" "logs/${TAG}_sweep.log" | head -24
else
    stamp "sweep FAIL"; tail -40 "logs/${TAG}_sweep.log"; exit 1
fi
stamp "chain DONE"
