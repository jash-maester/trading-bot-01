#!/usr/bin/env bash
# Does the fundamental blend earn money, not just IC?
#
# `audit/F2_FUNDAMENTAL_IC.md` shows a three-feature earnings-change score is
# near-independent of R4 and lifts pooled 20d IC from +0.0439 to +0.0517. IC is
# not money: it says nothing about whether the reordering survives a flat
# Rs 15.34 demat debit per scrip sold, 0.1% STT on both legs, and 20% STCG.
# This runs the same allocator grid over the blended signal and the plain one,
# so the two tables differ in exactly one thing.
#
#   RUN_TAG=fund bash scripts/fundamentals_chain.sh
#
# Stages skip when their artefact exists, so an interrupted run resumes. Each
# writes a line to logs/fundamentals_<tag>.status.
#
# WHY THERE IS NO HOLDOUT STAGE. NSE's corporates-financial-results endpoint
# returns nothing broadcast after roughly January 2025 — measured 2026-09-07
# against INFY, RELIANCE, TCS and BPCL, all of which return zero filings for
# 2025-02-01..2026-09-07. The blend can therefore be judged on the walk-forward
# windows and NOT on the 2025+ holdout. Adding a holdout stage here would
# silently score a signal whose fundamental component is entirely absent, which
# would read as "the blend adds nothing" when the truth is "the data stops".
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-fund}"
BASE_SIGNAL="${BASE_SIGNAL:-r4_v2}"
STATUS="logs/fundamentals_${TAG}.status"
mkdir -p logs
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

# The grid is cut to monthly / 20d / K in {20,30}: the cadence and horizon the
# standing comparison uses, so these rows sit directly beside the ones already
# in MLflow. Every risk arm in configs/allocator/default.yaml runs regardless.
GRID='+allocator.k_grid=[20,30] +allocator.freq_grid=[monthly] +allocator.horizon_grid=[20d]'

run_stage() {   # run_stage <name> <logfile> <command...>
    local name="$1" logf="$2"; shift 2
    say "$name"
    if "$@" > "$logf" 2>&1; then
        stamp "$name OK"; tail -25 "$logf"
    else
        stamp "$name FAIL"; say "$name FAILED — see $logf"; tail -40 "$logf"; return 1
    fi
}

# ── Stage 1: the blend, three selected features ──────────────────────────────
if [ -f "data/signal/${BASE_SIGNAL}_fund/gate.json" ]; then
    say "stage 1 skipped: data/signal/${BASE_SIGNAL}_fund/gate.json exists"
    stamp "stage1 SKIP"
else
    run_stage "stage1 blend(selected)" "logs/${TAG}_blend.log" \
        uv run python scripts/blend_signal.py \
            --signal-tag "$BASE_SIGNAL" --out-tag "${BASE_SIGNAL}_fund" || exit 1
fi

# ── Stage 2: the blend, ALL change features, no selection ────────────────────
# Stage 1's three features were chosen by looking at the same six windows the
# blend is then scored on. That is selection on the evaluation set, and it can
# manufacture a lift on its own. This arm uses every change feature the module
# builds, chosen before any IC was measured, so it cannot benefit from that
# choice. If the lift survives here it is not selection.
if [ -f "data/signal/${BASE_SIGNAL}_fundall/gate.json" ]; then
    say "stage 2 skipped: data/signal/${BASE_SIGNAL}_fundall/gate.json exists"
    stamp "stage2 SKIP"
else
    run_stage "stage2 blend(all-change)" "logs/${TAG}_blendall.log" \
        uv run python scripts/blend_signal.py \
            --signal-tag "$BASE_SIGNAL" --out-tag "${BASE_SIGNAL}_fundall" \
            --all-change-features || exit 1
fi

# ── Stages 3-5: allocator grid, one per signal, after tax ────────────────────
for sig in "$BASE_SIGNAL" "${BASE_SIGNAL}_fund" "${BASE_SIGNAL}_fundall"; do
    say "allocator grid on ${sig}"
    if RUN_TAG="${TAG}_${sig}" SIGNAL_TAG="$sig" SPLIT="oos_${BASE_SIGNAL}" \
       EXTRA="+apply_tax=true $GRID" bash scripts/orchestrate.sh \
         > "logs/${TAG}_alloc_${sig}.log" 2>&1; then
        stamp "alloc ${sig} OK"
    else
        stamp "alloc ${sig} FAIL"; say "allocator FAILED for ${sig}"
        tail -40 "logs/${TAG}_alloc_${sig}.log"; exit 1
    fi
    # The comparison is the whole point, so surface the table, not the tail.
    sed -n '/^strategy/,/^-\{60,\}$/p' "logs/${TAG}_${sig}_allocator.log" | tail -30
done

stamp "chain DONE"
say "chain complete — status in $STATUS"
echo
echo "Three tables above, identical in every way except the signal driving them:"
echo "  ${BASE_SIGNAL}          price only"
echo "  ${BASE_SIGNAL}_fund     + three selected earnings-change features"
echo "  ${BASE_SIGNAL}_fundall  + every change feature, no selection"
