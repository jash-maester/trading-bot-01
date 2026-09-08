#!/usr/bin/env bash
# The cross-sectional-input experiment. RUN, AND THE HYPOTHESIS WAS REFUTED --
# audit/S3_CROSS_SECTIONAL_INPUTS.md. Kept so the run reproduces; do not read
# the reasoning below as a live proposal.
#
# Result: 20d gate IC fell from +0.0280 (t 4.19, PASS) to +0.0131 (t 1.52,
# FAIL); on the unseen holdout from +0.0459 to +0.0117, 3 of 15 months
# favouring it. The motivating gap was in-sample only and reverses out of it.
#
# The cross-sectional-input experiment.
#
#   RUN_TAG=xs bash scripts/pit_xs_chain.sh
#
# WHY. R4's target is a per-date cross-sectional z-score, but its inputs were
# standardised by ONE global (mean, std) per feature, frozen over the whole
# train split. The model was asked "where does this stock sit relative to its
# peers today?" and handed absolute numbers, which cannot answer it whenever
# the market-wide level moves -- and it moves enormously.
#
# Measured on r4_pit_long's OWN OOS rows, 20d, identical dates and names,
# blocked by year (scripts/signal_feature_diagnostic.py):
#
#   trained 15-feature model      +0.0275   11/14 years up
#   -realized_vol_60d, ranked     +0.0535   12/14   <- one feature, no fitting
#   -realized_vol_20d, ranked     +0.0462   12/14
#
# A cross-sectional rank of a single feature roughly doubles the model. The
# ranking operation is the missing piece, so this run gives the model the same
# operation on its inputs: van der Waerden scores within each date's tradeable
# cross-section (`supervised.cross_sectional_normalise`).
#
# WHY THIS AND NOT THE VOL FEATURE. Low-vol is regime-dependent and would have
# been a trap: on the 2025-26 holdout it scores +0.0312 at the monthly-block
# level with t 0.72, only 10/15 months up, and it turns sharply negative from
# 2026-03 (-0.3510, -0.1139, -0.1460). The model, by contrast, HOLDS on that
# same holdout (+0.0243 at 20d, t 4.48, 68.5% of dates positive). So the model
# generalises and the feature does not; the change worth making is structural,
# not a new factor.
#
# PARITY. Same config file, same panel, same windows, same seed as
# r4_pit_long. Exactly two overrides: the tag and xs_normalise. If anything
# else moves, the comparison answers nothing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-xs}"
SIGNAL="${SIGNAL_TAG:-r4_pit_xs}"
BASE="${BASE_TAG:-r4_pit_long}"
STATUS="logs/xs_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PANEL_START=$(uv run python -c "
import polars as pl
print(pl.read_parquet('data/panels_bhav/full.parquet', columns=['date'])['date'].min())
" 2>/dev/null | tail -1)
say "panel starts ${PANEL_START}; base=${BASE} signal=${SIGNAL}"
stamp "panel_start ${PANEL_START}"

# ── Stage 1: train ──────────────────────────────────────────────────────────
say "stage 1: train ${SIGNAL} (xs_normalise=rank)"
rm -rf "data/signal/${SIGNAL}"
if uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
     train.tag="${SIGNAL}" train.xs_normalise=rank \
     walk.n_windows=20 walk.data_start="${PANEL_START}" walk.data_end=2024-12-31 \
     > "logs/${TAG}_train.log" 2>&1; then
    stamp "stage1 OK"
    grep -E "R4 GATE|horizon +[0-9]+d:|identity feature stats" "logs/${TAG}_train.log" | tail -8
else
    stamp "stage1 FAIL"; tail -40 "logs/${TAG}_train.log"; exit 1
fi

# ── Stage 2: OOS split, so the allocator can read it later ──────────────────
say "stage 2: OOS split"
if uv run python scripts/make_oos_split.py --tag "${SIGNAL}" \
     --panels-root data/panels_bhav --out "oos_${SIGNAL}" \
     > "logs/${TAG}_split.log" 2>&1; then
    stamp "stage2 OK"
else
    stamp "stage2 FAIL"; tail -30 "logs/${TAG}_split.log"; exit 1
fi

# ── Stage 3: the frozen model over the unseen holdout ───────────────────────
say "stage 3: ${SIGNAL} over the 2025-26 holdout"
if uv run python scripts/predict_signal.py \
     --signal-dir "data/signal/${SIGNAL}" \
     --panel data/panels_bhav/holdout.parquet \
     --out-tag "${SIGNAL}_holdout" \
     > "logs/${TAG}_predict.log" 2>&1; then
    stamp "stage3 OK"
    grep -iE "input representation|rows|tickers" "logs/${TAG}_predict.log" | tail -4
else
    stamp "stage3 FAIL"; tail -30 "logs/${TAG}_predict.log"; exit 1
fi

# ── Stage 4: head to head, both spans, identical rows ───────────────────────
say "stage 4: head-to-head vs ${BASE}"
SIGNAL="$SIGNAL" BASE="$BASE" uv run python scripts/xs_head_to_head.py \
    2>&1 | tee "logs/${TAG}_compare.log"
stamp "stage4 OK"
stamp "chain DONE"
say "complete — status in $STATUS"
