#!/usr/bin/env bash
# Mac (MPS) parity retrain of r4_pit_long -- proves the annual refit
# (audit/P2, first due 2027-04-01) can run on this machine now that the CUDA
# box is gone. Writes to a SEPARATE tag; the paper loop keeps using
# r4_pit_long, which P2 freezes until the refit.
#
#   bash scripts/retrain_parity.sh                                  # ~1.9 h
#   TRADER_MATMUL_PRECISION=medium bash scripts/retrain_parity.sh   # ~1.6 h
#
# ETA from probes on 2026-09-25 (60 steps + one validation pass, W1, MPS):
#   original encoder      83.2 s  -> ~4.3 h for the 123-epoch run
#   pruned TCN forward    36.4 s  -> ~1.9 h   (exact maths; the default)
#   pruned + "medium"     30.4 s  -> ~1.6 h   (TF32-like fp32 matmuls)
# The CUDA reference was 123 epochs / 12,022 steps.
#
# Parity is STATISTICAL, not bit-exact: MPS and CUDA float arithmetic differ.
# The comparison (scripts/compare_signal_artefacts.py) checks the gate verdict,
# per-window IC, the top-K gate, and the per-date rank correlation of the two
# models' predictions on identical rows.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
set -a; [ -f .env ] && . ./.env; set +a

TAG="${TAG:-r4_pit_long_mps}"
REF="${REF:-r4_pit_long}"
LOG=logs/retrain; mkdir -p "$LOG"; STAMP=$(date -u +%Y%m%dT%H%MZ)
STATUS="$LOG/${TAG}_${STAMP}.status"
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$STATUS"; }

# The panel must be the one the reference trained on, or parity is meaningless.
want=$(cat data/panels_bhav/full.sha256); got=$(shasum -a 256 data/panels_bhav/full.parquet | awk '{print $1}')
[ "$want" = "$got" ] || { stamp "FAIL panel sha256 $got != recorded $want"; exit 1; }
PANEL_START=$(uv run python -c "import polars as pl;print(pl.read_parquet('data/panels_bhav/full.parquet',columns=['date'])['date'].min())")

bash scripts/memwatch.sh "$LOG/mem_${TAG}_${STAMP}.csv" 30 & MW=$!
trap 'kill $MW 2>/dev/null' EXIT
stamp "start tag=$TAG ref=$REF panel_start=$PANEL_START (panel sha verified) tcn=${TRADER_TCN_FULL:+full}${TRADER_TCN_FULL:-pruned} matmul=${TRADER_MATMUL_PRECISION:-highest}"

rm -rf "data/signal/$TAG"
# Identical to scripts/pit_extend_span.sh stage 4, except the tag.
if uv run python scripts/train_signal.py data=bhav_v1 train=r4_pit model=signal \
     train.tag="$TAG" \
     walk.n_windows=20 walk.data_start="$PANEL_START" walk.data_end=2024-12-31 \
     > "$LOG/${TAG}_${STAMP}.train.log" 2>&1; then
    stamp "train OK"
else
    stamp "train FAIL"; tail -30 "$LOG/${TAG}_${STAMP}.train.log"; exit 1
fi

uv run python scripts/compare_signal_artefacts.py --a "$REF" --b "$TAG" \
    --panel data/panels_bhav/oos_r4_pit_long.parquet \
    | tee "$LOG/${TAG}_${STAMP}.parity.txt"
stamp "DONE -- parity report $LOG/${TAG}_${STAMP}.parity.txt"
