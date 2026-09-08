#!/usr/bin/env bash
# Task 3: more independent windows, which is what a window-level t-test needs.
#
#   RUN_TAG=ext bash scripts/pit_extend_span.sh
#
# The in-sample result sits at t 1.98 against a t_crit of 2.365 on 8 windows.
# More sweeping is how that becomes 2.1 by accident; more SPAN is how it
# becomes evidence. NSE's classic archive was probed on 2026-09-08 and serves
# 2005-01-03 cleanly (816 rows, 814 EQ/BE), so the walk-forward can start six
# years earlier.
#
#   8 windows   df=7   t_crit 2.365
#  ~13 windows  df=12  t_crit 2.179
#
# That is a lower bar AND more evidence, and it costs no new modelling -- the
# same signal, the same allocator, measured over more independent periods.
#
# ISIN IS ABSENT BEFORE ~2011, so rename resolution cannot reach the added
# years. It does not bite for the universe rule, which is turnover-based, but a
# caller doing identity work on 2005-2010 must know the column is empty.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-ext}"
FROM="${FROM:-2005-01-01}"
STATUS="logs/extend_${TAG}.status"
mkdir -p logs
: > "$STATUS"
say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }
set -a; [ -f .env ] && . ./.env; set +a

run_stage() {   # run_stage <name> <log> <cmd...>
    local name="$1" logf="$2"; shift 2
    say "$name"
    if "$@" > "$logf" 2>&1; then
        stamp "$name OK"; tail -14 "$logf"
    else
        stamp "$name FAIL"; say "$name FAILED"; tail -40 "$logf"; return 1
    fi
}

# ── 1: extend the archive. Cached days are not refetched. ───────────────────
run_stage "stage1 fetch" "logs/${TAG}_fetch.log" \
    uv run python scripts/fetch_bhavcopy.py --from "$FROM" || exit 1

# ── 2: rebuild the store from scratch over the wider span ───────────────────
# Wiped rather than appended: back_adjust runs over the whole frame, and a
# half-updated store is the kind of thing that looks fine and is not.
say "stage 2: rebuild the store"
rm -rf data/bhav_ohlcv
run_stage "stage2 store" "logs/${TAG}_store.log" \
    uv run python scripts/bhavcopy_to_store.py || exit 1

# ── 3: rebuild the panel ────────────────────────────────────────────────────
say "stage 3: rebuild the point-in-time panel"
rm -rf data/panels_bhav
run_stage "stage3 panel" "logs/${TAG}_panel.log" \
    uv run python scripts/build_features.py data=bhav_v1 \
        data.start_date="${FROM}" || exit 1

# ── 4: retrain over the longer span ─────────────────────────────────────────
# data_start is the panel's own first session; n_windows high enough that the
# span, not the count, is the binding constraint.
say "stage 4: retrain R4 over the longer span"
PANEL_START=$(uv run python -c "
import polars as pl
print(pl.read_parquet('data/panels_bhav/full.parquet', columns=['date'])['date'].min())
" 2>/dev/null | tail -1)
say "panel starts ${PANEL_START}"
rm -rf data/signal/r4_pit_long
if uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
     train.tag=r4_pit_long \
     walk.n_windows=20 walk.data_start="${PANEL_START}" walk.data_end=2024-12-31 \
     > "logs/${TAG}_train.log" 2>&1; then
    stamp "stage4 OK"
    grep -E "walk-forward window|R4 GATE|horizon +[0-9]+d:" "logs/${TAG}_train.log" | head -6
else
    stamp "stage4 FAIL"; tail -40 "logs/${TAG}_train.log"; exit 1
fi

# ── 5: allocator + gate on the longer span ──────────────────────────────────
run_stage "stage5 split" "logs/${TAG}_split.log" \
    uv run python scripts/make_oos_split.py --tag r4_pit_long \
        --panels-root data/panels_bhav --out oos_r4_pit_long || exit 1

say "stage 6: allocator, band swept"
if uv run python scripts/run_allocator.py data=bhav_v1 \
      +split=oos_r4_pit_long +signal_tag=r4_pit_long +require_gate_pass=true \
      +apply_tax=true \
      ++allocator.universe_from_panel=true \
      ++allocator.null_control=true \
      ++allocator.k_grid=[20,30] \
      ++allocator.band_grid=[0.0,0.005,0.010] \
      ++allocator.freq_grid=[monthly,quarterly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.nav_dir=audit/navs_long \
      > "logs/${TAG}_alloc.log" 2>&1; then
    stamp "stage6 OK"
    grep -E "^equal_weight|^null_signal|^allocator" "logs/${TAG}_alloc.log" | head -20
else
    stamp "stage6 FAIL"; tail -40 "logs/${TAG}_alloc.log"; exit 1
fi

run_stage "stage7 gate" "logs/${TAG}_gate.log" \
    uv run python scripts/allocator_gate.py --nav-dir audit/navs_long \
        --baseline equal_weight_monthly --out audit/r5_gate_long.json || true

stamp "chain DONE"
say "complete — status in $STATUS"
