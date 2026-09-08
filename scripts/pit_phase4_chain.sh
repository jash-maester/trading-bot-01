#!/usr/bin/env bash
# Phase 4: the 2025-26 holdout, on the point-in-time universe.
#
#   RUN_TAG=p4 bash scripts/pit_phase4_chain.sh
#
# The holdout is the only span no window of Phase 2 was scored on -- which is
# why walk.data_end was capped at 2024-12-31 and why a run that quietly widened
# to ten windows had to be stopped. Spending it is a one-way decision.
#
# THE MODEL IS STALE BY CONSTRUCTION. Phase 2's last window trained through
# ~2022, so predicting 2025-26 asks it to extrapolate. predict_signal.py writes
# `staleness_years` into index.json for exactly this reason. A deployment would
# refit; this measures what the frozen fit is worth, which is the conservative
# question.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-p4}"
SIGNAL="${SIGNAL_TAG:-r4_pit}"
HOLD="${HOLD_TAG:-${SIGNAL}_holdout}"
STATUS="logs/phase4_${TAG}.status"
NAVDIR="audit/navs_p4"
mkdir -p logs "$NAVDIR"
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

if [ ! -f "data/signal/${SIGNAL}/signal_model.pt" ]; then
    stamp "PRECONDITION FAIL: no trained model"
    say "data/signal/${SIGNAL}/signal_model.pt missing — Phase 2 produced no model"
    exit 1
fi

# ── Stage 1: a holdout slice of the point-in-time panel ─────────────────────
# 2025-04 onward, matching the span the fixed-universe holdout used, so the two
# are comparable. The PIT panel already carries per-date eligibility in
# is_tradeable, so no universe decision is made here.
if [ -f data/panels_bhav/holdout.parquet ]; then
    say "stage 1 skipped: holdout.parquet exists"; stamp "stage1 SKIP"
else
    say "stage 1: holdout slice 2025-04-01 .. panel end"
    if uv run python scripts/slice_panel.py \
         --in data/panels_bhav/full.parquet \
         --out data/panels_bhav/holdout.parquet \
         --from 2025-04-01 --to 2026-09-04 \
         > "logs/${TAG}_slice.log" 2>&1; then
        stamp "stage1 OK"; tail -4 "logs/${TAG}_slice.log"
    else
        stamp "stage1 FAIL"; tail -30 "logs/${TAG}_slice.log"; exit 1
    fi
fi

# ── Stage 2: run the frozen model forward ───────────────────────────────────
if [ -f "data/signal/${HOLD}/predictions.parquet" ]; then
    say "stage 2 skipped: ${HOLD} exists"; stamp "stage2 SKIP"
else
    say "stage 2: frozen ${SIGNAL} over the holdout"
    if uv run python scripts/predict_signal.py \
         --signal-dir "data/signal/${SIGNAL}" \
         --panel data/panels_bhav/holdout.parquet \
         --out-tag "$HOLD" \
         > "logs/${TAG}_predict.log" 2>&1; then
        stamp "stage2 OK"; grep -iE "staleness|rows|tickers" "logs/${TAG}_predict.log" | tail -5
    else
        stamp "stage2 FAIL"; tail -30 "logs/${TAG}_predict.log"; exit 1
    fi
fi

# ── Stage 3: allocator on the holdout ───────────────────────────────────────
# require_gate_pass=false and the runner stamps "SIGNAL GATE DID NOT PASS" on
# every table: a frozen-model holdout has no gate of its own by design, and the
# banner is what stops these numbers being quoted as validated.
say "stage 3: allocator on the holdout"
if uv run python scripts/run_allocator.py data=bhav_v1 \
      +split=holdout +signal_tag="$HOLD" +require_gate_pass=false \
      +apply_tax=true \
      ++allocator.universe_from_panel=true \
      ++allocator.null_control=true \
      ++allocator.k_grid=[20,30] \
      ++allocator.freq_grid=[monthly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.nav_dir="$NAVDIR" \
      > "logs/${TAG}_alloc.log" 2>&1; then
    stamp "stage3 OK"
    grep -E "universe from|SIGNAL GATE|^equal_weight|^null_signal|^allocator" \
        "logs/${TAG}_alloc.log" | head -14
else
    stamp "stage3 FAIL"; tail -40 "logs/${TAG}_alloc.log"; exit 1
fi

# ── Stage 4: baselines on the same holdout, for a like-for-like bar ─────────
say "stage 4: baselines on the holdout"
if uv run python scripts/run_baselines.py data=bhav_v1 \
      +split=holdout +apply_tax=true \
      '++baselines.freq_grid=[monthly]' \
      '++baselines.universe_from_panel=true' \
      '++baselines.nav_dir=audit/navs_p4' \
      > "logs/${TAG}_baselines.log" 2>&1; then
    stamp "stage4 OK"
    sed -n '/^baseline  /,$p' "logs/${TAG}_baselines.log" | head -12
else
    stamp "stage4 FAIL"; tail -30 "logs/${TAG}_baselines.log"; exit 1
fi

# ── Stage 5: the paired gate, against whichever bar is strongest ────────────
for base in equal_weight_frozen equal_weight; do
    say "holdout gate vs ${base}"
    uv run python scripts/allocator_gate.py --nav-dir "$NAVDIR" \
        --baseline "$base" --out "audit/r5_holdout_pit_vs_${base}.json" \
        > "logs/${TAG}_gate_${base}.log" 2>&1 \
        && { stamp "gate vs ${base} OK"; sed -n '/^arm  /,/^-\{60,\}$/p' "logs/${TAG}_gate_${base}.log" | head -16; } \
        || { stamp "gate vs ${base} FAIL"; tail -15 "logs/${TAG}_gate_${base}.log"; }
done

stamp "chain DONE"
say "complete — status in $STATUS"
