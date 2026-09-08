#!/usr/bin/env bash
# Task 2: does a quarterly hold beat a monthly one?
#
#   RUN_TAG=q bash scripts/pit_quarterly.sh
#
# Two effects at once, which is why it is worth measuring separately from the
# band. Quarterly cuts turnover a further ~3x, AND it pushes holdings toward
# the twelve-month STCG boundary where the rate drops from 20% + cess to 12.5%.
# The band already captured much of the first; the second is untouched.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TAG="${RUN_TAG:-q}"
SIGNAL="${SIGNAL_TAG:-r4_pit}"
STATUS="logs/quarterly_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

say "monthly vs quarterly, band swept, K=20/30"
if uv run python scripts/run_allocator.py data=bhav_v1 \
      +split="oos_${SIGNAL}" +signal_tag="$SIGNAL" +require_gate_pass=true \
      +apply_tax=true \
      ++allocator.universe_from_panel=true \
      ++allocator.null_control=true \
      ++allocator.k_grid=[20,30] \
      ++allocator.band_grid=[0.0,0.010] \
      ++allocator.freq_grid=[monthly,quarterly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.risk_grid='[{name: none}, {name: volstop, stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 21}]' \
      ++allocator.nav_dir=audit/navs_q \
      > "logs/${TAG}_alloc.log" 2>&1; then
    stamp "alloc OK"
    grep -E "^equal_weight|^null_signal|^allocator" "logs/${TAG}_alloc.log" | head -20
else
    stamp "alloc FAIL"; tail -40 "logs/${TAG}_alloc.log"; exit 1
fi

uv run python scripts/allocator_gate.py --nav-dir audit/navs_q \
    --baseline equal_weight_monthly --out audit/r5_gate_quarterly.json \
    > "logs/${TAG}_gate.log" 2>&1 && stamp "gate OK" || stamp "gate FAIL"
sed -n '/^arm  /,/^-\{60,\}$/p' "logs/${TAG}_gate.log" 2>/dev/null | head -18
grep -E "of .* arm" "logs/${TAG}_gate.log" 2>/dev/null | tail -1
stamp "chain DONE"
