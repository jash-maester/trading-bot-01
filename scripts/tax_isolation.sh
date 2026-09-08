#!/usr/bin/env bash
# What does capital-gains tax actually cost this strategy?
#
#   RUN_TAG=tax bash scripts/tax_isolation.sh
#
# apply_tax=true has been on for every number in Phases 1-5, and the tax term
# has never been isolated. It matters for what to do next: if the edge is 3%/yr
# NET of a 2%/yr drag, the fix is cadence; if the drag is 0.3%, the fix is the
# signal. Those are different projects.
#
# STCG is 20% + 4% cess under s111A and LTCG 12.5% under s112A, with the
# boundary at twelve months -- so a monthly-rebalanced book realises almost
# everything at the higher rate. Identical arms, identical seed, tax the only
# difference.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-tax}"
SIGNAL="${SIGNAL_TAG:-r4_pit}"
STATUS="logs/tax_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

for tax in true false; do
    say "apply_tax=${tax}"
    if uv run python scripts/run_allocator.py data=bhav_v1 \
          +split="oos_${SIGNAL}" +signal_tag="$SIGNAL" +require_gate_pass=true \
          +apply_tax="$tax" \
          ++allocator.universe_from_panel=true \
          ++allocator.null_control=false \
          ++allocator.k_grid=[20] \
          ++allocator.band_grid=[0.0,0.010] \
          ++allocator.freq_grid=[monthly] \
          ++allocator.horizon_grid=[20d] \
          ++allocator.risk_grid='[{name: none}, {name: volstop, stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 21}]' \
          > "logs/${TAG}_tax_${tax}.log" 2>&1; then
        stamp "tax=${tax} OK"
        grep -E "^equal_weight|^allocator" "logs/${TAG}_tax_${tax}.log" | head -6
    else
        stamp "tax=${tax} FAIL"; tail -30 "logs/${TAG}_tax_${tax}.log"; exit 1
    fi
done
stamp "chain DONE"
say "complete"
