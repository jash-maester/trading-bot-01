#!/usr/bin/env bash
# Phase 3: the R5 allocator on the point-in-time universe, and its gate.
#
#   RUN_TAG=p3 bash scripts/pit_phase3_chain.sh
#
# Everything R5 has been measured on so far used the 504 names a 2026
# instrument dump happened to contain. Phase 1 showed what that was worth:
# momentum-top-K went from CAGR +0.273 on that list to -0.002 on a
# point-in-time universe. This re-runs the allocator against a signal trained
# on the real thing.
#
# THE GATE IS RUN TWICE, AGAINST TWO BARS, ON PURPOSE. R5's criterion says
# "beats best R2 baseline", and on the point-in-time universe the strongest
# baseline is no longer momentum -- Phase 1 measured equal_weight_frozen at
# CAGR 0.137, equal_weight at 0.132, momentum at -0.002. Reporting only the
# equal-weight comparison is what flattered every arm the first time.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-p3}"
SIGNAL="${SIGNAL_TAG:-r4_pit}"
STATUS="logs/phase3_${TAG}.status"
NAVDIR="audit/navs_p3"
mkdir -p logs "$NAVDIR"
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

if [ ! -f "data/signal/${SIGNAL}/predictions.parquet" ]; then
    stamp "PRECONDITION FAIL: no signal"
    say "data/signal/${SIGNAL}/predictions.parquet is missing — Phase 2 has not produced a signal"
    exit 1
fi

# ── Stage 1: an OOS slice cut to the signal's own prediction span ────────────
if [ -f "data/panels_bhav/oos_${SIGNAL}.parquet" ]; then
    say "stage 1 skipped: oos_${SIGNAL}.parquet exists"; stamp "stage1 SKIP"
else
    say "stage 1: OOS split matched to ${SIGNAL}"
    if uv run python scripts/make_oos_split.py --tag "$SIGNAL" \
         --panels-root data/panels_bhav --out "oos_${SIGNAL}" \
         > "logs/${TAG}_split.log" 2>&1; then
        stamp "stage1 OK"; tail -6 "logs/${TAG}_split.log"
    else
        stamp "stage1 FAIL"; tail -30 "logs/${TAG}_split.log"; exit 1
    fi
fi

# ── Stage 2: the allocator grid, after tax, on the PIT universe ──────────────
# universe_from_panel is the load-bearing flag: without it the env is built
# over the fixed 504 and this whole phase measures the old universe on new bars.
say "stage 2: allocator grid on the point-in-time universe"
if uv run python scripts/run_allocator.py data=bhav_v1 \
      +split="oos_${SIGNAL}" +signal_tag="$SIGNAL" +require_gate_pass=true \
      +apply_tax=true \
      ++allocator.universe_from_panel=true \
      ++allocator.null_control=true \
      ++allocator.k_grid=[20,30] \
      ++allocator.freq_grid=[monthly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.nav_dir="$NAVDIR" \
      > "logs/${TAG}_alloc.log" 2>&1; then
    stamp "stage2 OK"
    grep -E "universe from|^equal_weight|^null_signal|^allocator" "logs/${TAG}_alloc.log" | head -14
else
    stamp "stage2 FAIL"; tail -40 "logs/${TAG}_alloc.log"; exit 1
fi

# ── Stage 3: bring the Phase 1 baselines into the same NAV directory ─────────
# The gate is paired, so both arms must be the same days on the same universe.
if [ -d audit/navs_pit ]; then
    cp -n audit/navs_pit/nav_*.parquet "$NAVDIR"/ 2>/dev/null || true
    say "baseline NAVs available: $(ls "$NAVDIR" | grep -c '^nav_')"
    stamp "stage3 OK"
else
    stamp "stage3 SKIP (no audit/navs_pit)"
fi

# ── Stage 4: the gate, against both bars ────────────────────────────────────
for base in equal_weight_frozen equal_weight; do
    say "gate vs ${base}"
    if uv run python scripts/allocator_gate.py --nav-dir "$NAVDIR" \
         --baseline "$base" --out "audit/r5_gate_pit_vs_${base}.json" \
         > "logs/${TAG}_gate_${base}.log" 2>&1; then
        stamp "gate vs ${base} OK"
        sed -n '/^arm  /,/^-\{60,\}$/p' "logs/${TAG}_gate_${base}.log" | head -18
        grep -E "of .* arm" "logs/${TAG}_gate_${base}.log" | tail -1
    else
        stamp "gate vs ${base} FAIL"; tail -20 "logs/${TAG}_gate_${base}.log"
    fi
done

stamp "chain DONE"
say "complete — status in $STATUS"
echo
echo "Phase 1 bars on the SAME point-in-time universe, for comparison:"
echo "  equal_weight_frozen  Sharpe 0.691  CAGR 0.137  MDD -0.567"
echo "  equal_weight         Sharpe 0.681  CAGR 0.132  MDD -0.593"
echo "  momentum_topk        Sharpe -0.008 CAGR -0.002 MDD -0.814"
echo "  random               Sharpe 0.329  CAGR 0.060  MDD -0.627"
