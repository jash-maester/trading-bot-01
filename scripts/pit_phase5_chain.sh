#!/usr/bin/env bash
# Phase 5: does the backtest agree with the paper broker on the same weights?
#
#   RUN_TAG=p5 bash scripts/pit_phase5_chain.sh
#
# Every number in Phases 1-4 comes from the backtest env. If the broker would
# have done something different, those numbers describe a system nobody can
# run. The env and the broker have diverged twice before -- min_trade_value
# honoured by one and ignored by the other (11% NAV gap), and floor() deciding
# WHETHER a trade happened rather than its size -- and neither fix was ever
# checked end to end on this universe.
#
# An exact match is not the bar and not expected: the broker settles cash T+1,
# fills sells before buys, and clips rather than redistributes above the weight
# cap. What matters is that the gap stays small and does not TREND, because a
# widening gap is what both past divergences looked like.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-p5}"
SIGNAL="${SIGNAL_TAG:-r4_pit}"
HOLD="${HOLD_TAG:-${SIGNAL}_holdout}"
STATUS="logs/phase5_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

# In-sample span first, then the holdout: a rule applied on one side and not
# the other shows up on whichever span exercises it, and they are not the same
# trades.
run_parity() {   # run_parity <name> <split> <tag>
    local name="$1" split="$2" tag="$3"
    say "parity: ${name} (split=${split}, signal=${tag})"
    if uv run python scripts/broker_parity.py \
         --split "$split" --signal-tag "$tag" \
         --panels-root data/panels_bhav \
         > "logs/${TAG}_parity_${name}.log" 2>&1; then
        stamp "parity ${name} OK"
        grep -iE "gap|drift|trend|PASS|FAIL|nav" "logs/${TAG}_parity_${name}.log" | tail -12
    else
        stamp "parity ${name} FAIL"
        tail -30 "logs/${TAG}_parity_${name}.log"
        return 1
    fi
}

ok=0
if [ -f "data/panels_bhav/oos_${SIGNAL}.parquet" ]; then
    run_parity in_sample "oos_${SIGNAL}" "$SIGNAL" || ok=1
else
    say "skipped in-sample parity: oos_${SIGNAL}.parquet not built"
    stamp "parity in_sample SKIP"
fi

if [ -f "data/signal/${HOLD}/predictions.parquet" ]; then
    run_parity holdout holdout "$HOLD" || ok=1
else
    say "skipped holdout parity: ${HOLD} not built"
    stamp "parity holdout SKIP"
fi

stamp "chain DONE"
say "complete — status in $STATUS"
exit $ok
