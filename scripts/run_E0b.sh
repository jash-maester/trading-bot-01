#!/usr/bin/env bash
# audit/R_OVERNIGHT_2026-09-25.md E0b: seed-to-seed floor on MPS. Waits for E2.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
set -a; [ -f .env ] && . ./.env; set +a
TAG=r4_pit_long_mps_s43; LOG=logs/retrain
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG/E0b.status"; }
while pgrep -f "scripts/run_E2.sh" >/dev/null; do sleep 30; done
st "E2 finished; starting E0b"
PANEL_START=$(uv run python -c "import polars as pl;print(pl.read_parquet('data/panels_bhav/full.parquet',columns=['date'])['date'].min())")
rm -rf "data/signal/$TAG"
uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal seed=43 \
    train.tag="$TAG" walk.n_windows=20 walk.data_start="$PANEL_START" walk.data_end=2024-12-31 \
    > "$LOG/E0b.train.log" 2>&1 || { st "E0b train FAIL"; tail -30 "$LOG/E0b.train.log"; exit 1; }
st "E0b train OK"
uv run python scripts/compare_signal_artefacts.py --a r4_pit_long_mps --b "$TAG" \
    --panel data/panels_bhav/oos_r4_pit_long.parquet > "$LOG/E0b.seed_floor.txt" 2>&1
tail -6 "$LOG/E0b.seed_floor.txt" | tee -a "$LOG/E0b.status"
st "E0b DONE"
