#!/usr/bin/env bash
# P3 adaptive shadow book, nightly end-of-day replay (audit/P3_ADAPTIVE_SHADOW.md).
#
#   bash scripts/adaptive_daily.sh        # after scripts/paper_daily.sh has run
#   DRY=1 bash scripts/adaptive_daily.sh  # everything except the record append
#
# Separate from paper_daily.sh ON PURPOSE: P2's script is not edited, and a
# failure here can never stop or change the P2 record. It reuses the forward
# panel P2 built (data/panels_forward/full.parquet) and refuses to run if that
# panel is behind the bhavcopy (P2's run failed or has not run yet).
#
# Refits are NOT done here: they need the host GPU (scripts/refit_adaptive.py,
# ~18 min on MPS vs ~60 min at 7.4 GB in this VM). This only stitches the
# registered refits' predictions (cheap) and replays the books.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
set -a; [ -f .env ] && . ./.env; set +a
export TRADER_TCN_FULL=1

# ── FROZEN (audit/P3) ────────────────────────────────────────────────────────
ADAPT_FROM="2026-10-01"                 # first rebalance at this session's open
FREEZE_DATE="2026-09-30"                # scoring strictly after this session
CAPITAL=100000
K=30; BAND=0.010; FREQ=semimonthly; HOR=20d
NULL_SEEDS="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
# ─────────────────────────────────────────────────────────────────────────────
DIR=audit/paper/adaptive; LATEST=$DIR/latest; SNAPS=$DIR/snapshots; RECORD=$DIR/record.jsonl
LOGS=logs/paper; mkdir -p "$LOGS" "$SNAPS"
TODAY=$(date -u +%F); STATUS="$LOGS/adaptive_$TODAY.status"
echo "=== run $(date -u +%FT%TZ) ===" >> "$STATUS"
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
fail(){ stamp "FAIL $*"; exit 1; }
maxdate(){ uv run python -c "import polars as pl;print(pl.scan_parquet('$1').select(pl.col('date').max()).collect().item())"; }

BARS=$(maxdate data/ext/bhavcopy.parquet)
DATA_DATE=$(maxdate data/panels_forward/full.parquet)
[ "$BARS" = "$DATA_DATE" ] || { stamp "forward panel ends $DATA_DATE, bhavcopy $BARS: P2's run has not built today's panel -- waiting"; exit 0; }
if [[ "$DATA_DATE" < "$ADAPT_FROM" ]] && [ -z "${DRY:-}" ]; then
    stamp "data through $DATA_DATE is before ADAPT_FROM=$ADAPT_FROM -- nothing to replay yet"; exit 0
fi
if [ -d "$SNAPS/$DATA_DATE" ] && [ -z "${DRY:-}" ]; then
    stamp "no new session: $DATA_DATE already recorded"; exit 0
fi
stamp "start data=$DATA_DATE"

# 1 the stitched signal (newest refit per date, no lookahead)
uv run python scripts/adaptive_signal.py > "$LOGS/adaptive_signal_$TODAY.log" 2>&1 || fail "adaptive_signal"
stamp "signal OK ($(head -1 "$LOGS/adaptive_signal_$TODAY.log"))"

# 2 its own paper panel: first traded session = ADAPT_FROM
uv run python scripts/make_paper_panel.py --src data/panels_forward/full.parquet \
    --out data/panels_forward/paper_adaptive.parquet --trade-start "$ADAPT_FROM" --lookback 60 \
    > "$LOGS/adaptive_panel_$TODAY.log" 2>&1 || fail "make_paper_panel"
N_LIVE=$(uv run python -c "import polars as pl;print(pl.scan_parquet('data/panels_forward/paper_adaptive.parquet').filter(pl.col('date')>=pl.lit('$ADAPT_FROM').str.to_date()).select(pl.col('date').n_unique()).collect().item())")
if [ "$N_LIVE" -lt 2 ] && [ -z "${DRY:-}" ]; then
    stamp "first session $ADAPT_FROM is the latest; its close is marked when the next session publishes -- nothing to record yet"; exit 0
fi

# 3 replay: adaptive signal + 20 random books, semimonthly, same machinery as P2
rm -rf "$LATEST"
uv run python scripts/run_allocator.py data=bhav_v1 data.panels_root=data/panels_forward \
    +split=paper_adaptive +signal_tag=adaptive_signal +require_gate_pass=false +apply_tax=true \
    env.initial_cash="$CAPITAL" \
    "++allocator.initial_full_deploy=true" "++allocator.topup_cash=true" \
    "++allocator.universe_from_panel=true" "++allocator.null_control=true" \
    "++allocator.null_seeds=$NULL_SEEDS" \
    "++allocator.k_grid=[$K]" "++allocator.band_grid=[$BAND]" "++allocator.freq_grid=[$FREQ]" \
    "++allocator.horizon_grid=[$HOR]" "++allocator.nav_dir=$LATEST" \
    > "$LOGS/adaptive_alloc_$TODAY.log" 2>&1 || fail "run_allocator"
stamp "allocator OK ($(ls "$LATEST" | wc -l | tr -d ' ') books)"

# 4 record (append-only), determinism check, snapshot
if [ -n "${DRY:-}" ]; then stamp "DRY: record not appended"; exit 0; fi
uv run python scripts/paper_record.py --nav-dir "$LATEST" --record "$RECORD" --snapshots "$SNAPS" \
    --freeze-date "$FREEZE_DATE" --data-date "$DATA_DATE" --record-from "$ADAPT_FROM" --freq "$FREQ" \
    > "$LOGS/adaptive_record_$TODAY.log" 2>&1 || fail "paper_record"
rm -rf "$SNAPS/$DATA_DATE" && cp -r "$LATEST" "$SNAPS/$DATA_DATE"
stamp "DONE $(uv run python -c "import json;s=json.load(open('$LATEST/summary.json'));print(f\"data={s['data_date']} scored={s['scored_sessions']} rank={s['rank']}/{s['n_null']+1}\")")"
