#!/usr/bin/env bash
# audit/R_OVERNIGHT_2026-09-25.md E2: r4_pit_long with a ListNet loss, otherwise identical.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
set -a; [ -f .env ] && . ./.env; set +a
TAG=r4_pit_long_listnet; LOG=logs/retrain; mkdir -p "$LOG"
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG/E2.status"; }
# Wait for the GPU: E0 (retrain_parity.sh) must finish first.
while pgrep -f "scripts/retrain_parity.sh" >/dev/null; do sleep 30; done
st "E0 finished; starting E2"
PANEL_START=$(uv run python -c "import polars as pl;print(pl.read_parquet('data/panels_bhav/full.parquet',columns=['date'])['date'].min())")
bash scripts/memwatch.sh "$LOG/mem_E2.csv" 30 & MW=$!; trap 'kill $MW 2>/dev/null' EXIT
rm -rf "data/signal/$TAG"
uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
    train.tag="$TAG" +train.loss=listnet \
    walk.n_windows=20 walk.data_start="$PANEL_START" walk.data_end=2024-12-31 \
    > "$LOG/E2.train.log" 2>&1 || { st "E2 train FAIL"; tail -30 "$LOG/E2.train.log"; exit 1; }
st "E2 train OK"
grep -E "R4 GATE|horizon +[0-9]+d:" "$LOG/E2.train.log" | tail -3 | tee -a "$LOG/E2.status"
uv run python scripts/topk_gate.py --signal "$TAG" --panel data/panels_bhav/full.parquet \
    --k 30 --horizon 20 --n-null 20 --which top > "$LOG/E2.topk.log" 2>&1
grep -vE "INFO|^\s*$" "$LOG/E2.topk.log" | tail -6 | tee -a "$LOG/E2.status"
cp "data/signal/$TAG/topk_gate.json" "audit/topk_gate/${TAG}_top30.json" 2>/dev/null
st "E2 DONE"
