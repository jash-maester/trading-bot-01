#!/usr/bin/env bash
# The forward paper loop. One run per trading day, after NSE publishes (~18:30 IST).
#
#   bash scripts/paper_daily.sh            # normal
#   DRY=1 bash scripts/paper_daily.sh      # everything except the record append
#
# Pre-registered in audit/P2_PAPER_PERMUTATION_TEST.md. Nothing in the FROZEN
# block may change without a new pre-registration and a reset clock.
#
# Kite-free by construction: NSE's daily bhavcopy and index archive, no token.
# Runs on the Mac -- the working copy -- because the training box reboots on
# Windows' schedule, and because the whole chain takes ~5 minutes here (the
# frozen model scored on CPU is 3.5 of them).
#
# STATE. None is persisted. Each run replays the forward span from scratch;
# determinism carries positions, the band and the stops across days. The
# record of what the book did each day is scripts/paper_record.py's job, and
# it is append-only.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"   # scheduler-independent: cron and launchd both start minimal
set -a; [ -f .env ] && . ./.env; set +a

# ── FROZEN (audit/P2) ────────────────────────────────────────────────────────
SIGNAL_DIR="${SIGNAL_DIR:-data/signal/r4_pit_long}"   # artefact in force; annual refit replaces it
PANEL_START="2024-06-01"                                # 365d eligibility + 60d features before the warm-up
FREEZE_DATE="2026-09-09"                                # last warm-up session; scoring starts strictly after
K=30; BAND=0.010; FREQ=monthly; HOR=20d
NULL_SEEDS="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
# ─────────────────────────────────────────────────────────────────────────────
PAPER=audit/paper; LATEST=$PAPER/latest; SNAPS=$PAPER/snapshots; RECORD=$PAPER/record.jsonl
LOGS=logs/paper; mkdir -p "$LOGS" "$SNAPS"
TODAY=$(date -u +%F); STATUS="$LOGS/$TODAY.status"
echo "=== run $(date -u +%FT%TZ) ===" >> "$STATUS"    # append: two attempts a day share the file
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }
fail(){ stamp "FAIL $*"; [ -n "${MW:-}" ] && kill "$MW" 2>/dev/null; exit 1; }
maxdate(){ uv run python -c "import polars as pl;print(pl.scan_parquet('$1').select(pl.col('date').max()).collect().item())"; }
nextday(){ uv run python -c "from datetime import date,timedelta;print(date.fromisoformat('$1')+timedelta(days=1))"; }

bash scripts/memwatch.sh "$LOGS/mem_$TODAY.csv" 15 & MW=$!
stamp "start  signal=$SIGNAL_DIR  freeze=$FREEZE_DATE"

# ── 1 new sessions: bhavcopy + index, weekday calendar, 404 = not yet published ──
LAST=$(maxdate data/ext/bhavcopy.parquet)
FROM=$(nextday "$LAST")
uv run python scripts/fetch_bhavcopy.py --calendar weekdays --from "$FROM" --to "$TODAY" --sleep 0.6 \
    > "$LOGS/fetch_$TODAY.log" 2>&1 || fail "fetch_bhavcopy"
# The index has its OWN last date. NSE publishes the bhavcopy and the index
# file at different times, so after a run that fetched one but not the other,
# deriving both ranges from the bhavcopy would ask for the index from a date
# past the one it is missing -- and the dead-beta gate below would then fail
# on every run forever. Seen 2026-09-10 17:31 IST: bhavcopy out, index not.
IDX_LAST=$(uv run python -c "from datetime import datetime;from trader.data.storage import OhlcvStore;print(str(OhlcvStore('data/kite_ohlcv').load(tickers=['^NSEI'],start=datetime(2026,1,1))['date'].max())[:10])")
IDX_FROM=$(nextday "$IDX_LAST")
uv run python scripts/fetch_nse_index.py --from "$IDX_FROM" --to "$TODAY" \
    > "$LOGS/index_$TODAY.log" 2>&1 || fail "fetch_nse_index"
DATA_DATE=$(maxdate data/ext/bhavcopy.parquet)
IDX_DATE=$(uv run python -c "from datetime import datetime;from trader.data.storage import OhlcvStore;print(str(OhlcvStore('data/kite_ohlcv').load(tickers=['^NSEI'],start=datetime(2026,1,1))['date'].max())[:10])")
[ "$DATA_DATE" = "$IDX_DATE" ] || fail "bhavcopy ends $DATA_DATE but ^NSEI ends $IDX_DATE -- refusing to build a panel with a dead beta"
if [ -d "$SNAPS/$DATA_DATE" ] && [ -z "${DRY:-}" ]; then
    stamp "no new session: $DATA_DATE already recorded"; kill "$MW"; exit 0
fi
stamp "data through $DATA_DATE (fetched from $FROM)"

# ── 1b corporate actions: refresh this year, keep every other year ──────────
# The fetcher never re-reads a cached year, and it writes exactly the years it
# was asked for: requesting 2026 alone would REPLACE the file with 2026 only
# and silently drop every historical split -- a fake -50% return at each one.
# So: delete this year's cache, request the full range (old years come from
# cache instantly), and refuse to continue if the result is implausibly small.
YEAR=$(date -u +%Y)
rm -f "data/raw/nse/corpactions/ca_${YEAR}.json"
uv run python scripts/fetch_corporate_actions.py --from 2010 --to "$YEAR" \
    > "$LOGS/ca_$TODAY.log" 2>&1 || stamp "WARN corporate-actions refresh failed; existing file stands"
uv run python -c "import polars as pl,sys;n=pl.read_parquet('data/ext/corporate_actions.parquet').height;print(f'corporate actions: {n} rows');sys.exit(0 if n>=1000 else 1)" \
    || fail "corporate_actions.parquet has fewer than 1000 rows -- refusing to back-adjust with a truncated file"

# ── 2 store: rebuild beside, verify, swap ────────────────────────────────────
rm -rf data/bhav_ohlcv_new
uv run python scripts/bhavcopy_to_store.py --root data/bhav_ohlcv_new > "$LOGS/store_$TODAY.log" 2>&1 || fail "store rebuild"
uv run python - "$DATA_DATE" <<'PY' || fail "store verify"
import sys; from datetime import datetime
from trader.data.storage import OhlcvStore
want = sys.argv[1]; old, new = OhlcvStore("data/bhav_ohlcv"), OhlcvStore("data/bhav_ohlcv_new")
no, nn = len(old.tickers()), len(new.tickers())
assert abs(nn - no) <= 10, f"ticker count {no} -> {nn}"
for t in ("^NSEI", next(t for t in new.tickers() if t.startswith("RELIANCE"))):
    e = str(new.load(tickers=[t], start=datetime(2026,1,1))["date"].max())
    assert e == want, f"{t} ends {e}, want {want}"
PY
rm -rf data/bhav_ohlcv_prev && mv data/bhav_ohlcv data/bhav_ohlcv_prev && mv data/bhav_ohlcv_new data/bhav_ohlcv
stamp "store OK"

# ── 3 forward panel ──────────────────────────────────────────────────────────
rm -rf data/panels_forward
uv run python scripts/build_features.py data=bhav_v1 data.panels_root=data/panels_forward \
    data.start_date="$PANEL_START" data.end_date="$DATA_DATE" \
    +data.train_end=2026-03-31 +data.val_start=2026-04-01 +data.val_end=2026-06-30 +data.test_start=2026-07-01 \
    > "$LOGS/panel_$TODAY.log" 2>&1 || fail "build_features"
uv run python - "$DATA_DATE" <<'PY' || fail "panel verify (dead feature or wrong end date)"
import sys, polars as pl
from trader.data.features import FEATURE_COLS
d = pl.read_parquet("data/panels_forward/full.parquet"); last = d["date"].max()
assert str(last) == sys.argv[1], f"panel ends {last}"
tr = d.filter(pl.col("is_tradeable") & (pl.col("date") == last))
dead = [c for c in FEATURE_COLS if not tr[c].std()]
assert not dead, f"DEAD FEATURES on {last}: {dead}"
print(f"panel ok: {tr.height} tradeable on {last}")
PY
stamp "panel OK"

# ── 4 score the frozen model ─────────────────────────────────────────────────
rm -rf data/signal/paper_signal
uv run python scripts/predict_signal.py --signal-dir "$SIGNAL_DIR" --panel data/panels_forward/full.parquet \
    --out-tag paper_signal --device cpu > "$LOGS/predict_$TODAY.log" 2>&1 || fail "predict_signal"
stamp "predict OK ($(grep -oE '[0-9,]+ rows' "$LOGS/predict_$TODAY.log" | tail -1))"

# ── 5 the 21 books (+3 overlays recorded for context) ────────────────────────
rm -rf "$LATEST"
uv run python scripts/run_allocator.py data=bhav_v1 data.panels_root=data/panels_forward \
    +split=full +signal_tag=paper_signal +require_gate_pass=false +apply_tax=true \
    "++allocator.universe_from_panel=true" "++allocator.null_control=true" \
    "++allocator.null_seeds=$NULL_SEEDS" \
    "++allocator.k_grid=[$K]" "++allocator.band_grid=[$BAND]" "++allocator.freq_grid=[$FREQ]" \
    "++allocator.horizon_grid=[$HOR]" "++allocator.nav_dir=$LATEST" \
    > "$LOGS/alloc_$TODAY.log" 2>&1 || fail "run_allocator"
stamp "allocator OK ($(ls "$LATEST" | wc -l | tr -d ' ') books)"

# ── 6 record (append-only), determinism check, snapshot ──────────────────────
if [ -n "${DRY:-}" ]; then
    stamp "DRY: record not appended"; kill "$MW"; exit 0
fi
uv run python scripts/paper_record.py --nav-dir "$LATEST" --record "$RECORD" --snapshots "$SNAPS" \
    --freeze-date "$FREEZE_DATE" --data-date "$DATA_DATE" > "$LOGS/record_$TODAY.log" 2>&1 || fail "paper_record"
rm -rf "$SNAPS/$DATA_DATE" && cp -r "$LATEST" "$SNAPS/$DATA_DATE"
kill "$MW" 2>/dev/null
stamp "DONE $(uv run python -c "import json;s=json.load(open('$LATEST/summary.json'));print(f\"data={s['data_date']} scored={s['scored_sessions']} rank={s['rank']}/{s['n_null']+1} det_diff={s['determinism']['max_abs_rel_diff']:.2e}\")")"
