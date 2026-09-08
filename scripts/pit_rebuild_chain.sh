#!/usr/bin/env bash
# Phase 1 of the point-in-time rebuild: how much of the measured edge is a
# universe chosen in 2026?
#
# Measured 2026-09-08: of the 605 names carrying >= Rs 5 crore of median daily
# turnover in 2021, the 504-name universe holds 292 -- 48.3% -- and 55 of those
# 605 had stopped trading by 2026. Every headline number in this repository
# sits on that list.
#
#   RUN_TAG=pit bash scripts/pit_rebuild_chain.sh
#
# Stages skip when their artefact exists, so an interrupted run resumes.
#
# WHAT PHASE 1 DOES AND DOES NOT ANSWER. It re-runs the R2 BASELINES on a
# point-in-time universe. Baselines do not train, so this needs no GPU and no
# retraining, and it answers the question that decides whether Phase 2 is worth
# it: does the bar R5 has to clear -- MomentumTopK, which beats equal-weight on
# CAGR at monthly cadence -- fall when the universe stops being survivorship
# selected?
#
# PREDICTION, recorded before the run: momentum-top-K falls MORE than
# equal-weight, because it buys whatever ran hardest and a list drawn in 2026
# guarantees those names survived to be drawn. Every absolute return should
# fall; the question is whether the GAP between them survives.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-pit}"
FROM="${OOS_FROM:-2016-09-28}"
TO="${OOS_TO:-2024-06-28}"
STATUS="logs/pit_${TAG}.status"
mkdir -p logs
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

run_stage() {   # run_stage <name> <logfile> <command...>
    local name="$1" logf="$2"; shift 2
    say "$name"
    if "$@" > "$logf" 2>&1; then
        stamp "$name OK"; tail -20 "$logf"
    else
        stamp "$name FAIL"; say "$name FAILED — see $logf"; tail -40 "$logf"; return 1
    fi
}

# ── Stage 1: bhavcopy -> OhlcvStore ──────────────────────────────────────────
if [ -d data/bhav_ohlcv ] && [ -n "$(ls -A data/bhav_ohlcv 2>/dev/null)" ]; then
    say "stage 1 skipped: data/bhav_ohlcv is populated"; stamp "stage1 SKIP"
else
    run_stage "stage1 to-store" "logs/${TAG}_store.log" \
        uv run python scripts/bhavcopy_to_store.py || exit 1
fi

# ── Stage 2: industries (one request, cached) ────────────────────────────────
if [ -f data/ext/industries.parquet ]; then
    say "stage 2 skipped: data/ext/industries.parquet exists"; stamp "stage2 SKIP"
else
    run_stage "stage2 industries" "logs/${TAG}_ind.log" \
        uv run python scripts/fetch_industries.py || exit 1
fi

# ── Stage 3: the coverage measurement, regenerated ───────────────────────────
run_stage "stage3 coverage" "logs/${TAG}_coverage.log" \
    uv run python scripts/pit_universe_report.py --max-names 504 || exit 1

# ── Stage 4: the point-in-time panel ─────────────────────────────────────────
if [ -f data/panels_bhav/full.parquet ]; then
    say "stage 4 skipped: data/panels_bhav/full.parquet exists"; stamp "stage4 SKIP"
else
    run_stage "stage4 panel" "logs/${TAG}_panel.log" \
        uv run python scripts/build_features.py data=bhav_v1 || exit 1
fi

# ── Stage 5: cut it to the same span the Kite comparison used ────────────────
if [ -f data/panels_bhav/oos_pit.parquet ]; then
    say "stage 5 skipped: oos_pit.parquet exists"; stamp "stage5 SKIP"
else
    run_stage "stage5 slice" "logs/${TAG}_slice.log" \
        uv run python scripts/slice_panel.py \
            --in data/panels_bhav/full.parquet \
            --out data/panels_bhav/oos_pit.parquet \
            --from "$FROM" --to "$TO" || exit 1
fi

# ── Stage 6: the baselines, on the point-in-time universe ────────────────────
say "stage 6: R2 baselines on the point-in-time universe"
if uv run python scripts/run_baselines.py data=bhav_v1 \
      +split=oos_pit +apply_tax=true \
      '++baselines.freq_grid=[monthly]' \
      '++baselines.nav_dir=audit/navs_pit' \
      > "logs/${TAG}_baselines.log" 2>&1; then
    stamp "stage6 OK"
    sed -n '/^baseline  /,$p' "logs/${TAG}_baselines.log" | head -30
else
    stamp "stage6 FAIL"; tail -40 "logs/${TAG}_baselines.log"; exit 1
fi

stamp "chain DONE"
say "complete — status in $STATUS"
echo
echo "Compare against the SAME baselines on the 2026-chosen universe:"
echo "  equal_weight   monthly  Sharpe 1.280  CAGR 0.257  MDD -0.522"
echo "  momentum_topk  monthly  Sharpe 0.896  CAGR 0.273  MDD -0.725"
echo "  sixty_forty    monthly  Sharpe 1.408  CAGR 0.155  MDD -0.325"
echo "  random         monthly  Sharpe 0.954  CAGR 0.188  MDD -0.576"
