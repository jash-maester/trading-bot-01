#!/usr/bin/env bash
# Record every stop event the standing risk arms fire, for the news-veto study.
#
# `13_fundamentals_and_news.md` §4 designs the news arm as a VETO ON THE STOP,
# not a ranking signal: a held name breaching its stop is sold only if the fall
# carries adverse news, and held if it looks like noise. That design has a
# precondition nobody has measured — a veto is only worth building if stopping
# is destroying value on a separable subset of events.
#
# This produces the evidence, and needs no news at all. It replays the arms that
# already ran (`audit/F3`) and logs, per stopped name, what it was bought at,
# what it was sold at, and which threshold fired.
#
#   RUN_TAG=stopev bash scripts/stop_events_run.sh
#
# Nothing here is new modelling: identical grid to the standing comparison, with
# +allocator.stop_events_dir switched on.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-stopev}"
SIGNAL="${SIGNAL_TAG:-r4_v2}"
SPLIT="${SPLIT:-oos_r4_v2}"
STATUS="logs/stopev_${TAG}.status"
mkdir -p logs audit/stop_events
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

say "allocator grid with stop-event recording (${SIGNAL} on ${SPLIT}, after tax)"
if uv run python scripts/run_allocator.py data=kite_v1 \
      +split="$SPLIT" +signal_tag="$SIGNAL" +require_gate_pass=true \
      +apply_tax=true \
      ++allocator.null_control=false \
      ++allocator.k_grid=[20] \
      ++allocator.freq_grid=[monthly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.stop_events_dir=audit/stop_events \
      > "logs/${TAG}_alloc.log" 2>&1; then
    stamp "grid OK"
    grep -E "stop event|^allocator|^equal_weight" "logs/${TAG}_alloc.log" | tail -20
else
    stamp "grid FAIL"; tail -40 "logs/${TAG}_alloc.log"; exit 1
fi

say "events written"
ls -la audit/stop_events/
stamp "chain DONE"
