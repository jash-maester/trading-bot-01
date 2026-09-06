#!/usr/bin/env bash
# Run the R4 -> R5 chain to completion on the training box, one stage after the
# next, writing a per-stage log and a machine-readable status line.
#
# It runs ON THE BOX on purpose. The previous watcher lived on the Mac and died
# when that machine rebooted, taking the only progress signal with it. Nothing
# here depends on the controlling SSH session surviving.
#
#   RUN_TAG=r4_v1 bash scripts/orchestrate.sh
#
# Stages are idempotent and skipped when their artefact already exists, so a
# re-run after an interruption resumes rather than restarting.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Two independent names, and conflating them cost a run: SIGNAL_TAG selects the
# R4 artefacts under data/signal/, while RUN_TAG only labels this chain's logs.
# Re-running the grid against the SAME signal with a different code revision
# needs a new RUN_TAG and the OLD SIGNAL_TAG.
SIGNAL_TAG="${SIGNAL_TAG:-r4_v1}"
TAG="${RUN_TAG:-$SIGNAL_TAG}"
SPLIT="${SPLIT:-oos}"
STATUS="logs/orchestrate_${TAG}.status"
mkdir -p logs
: > "$STATUS"

say() { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

# ── Stage 1: OOS panel slice matched to the prediction span ──────────────────
if [ -f "data/panels_kite/${SPLIT}.parquet" ]; then
    say "stage 1 skipped: data/panels_kite/${SPLIT}.parquet exists"
    stamp "stage1 SKIP"
else
    say "stage 1: build ${SPLIT} split"
    if uv run python scripts/make_oos_split.py --tag "$SIGNAL_TAG" --out "$SPLIT" \
         > "logs/${TAG}_oos_split.log" 2>&1; then
        stamp "stage1 OK"
    else
        stamp "stage1 FAIL"; say "stage 1 FAILED — see logs/${TAG}_oos_split.log"; exit 1
    fi
fi

# ── Stage 2: R5 deterministic allocator grid ─────────────────────────────────
# require_gate_pass=false is deliberate and loud: R4's gate returned FAIL, and
# this stage exists to find out whether that verdict matters economically. Every
# MLflow run it writes carries signal_gate_verdict=FAIL as a param, so no result
# from tonight can later be mistaken for one that cleared the gate.
say "stage 2: allocator grid on ${SPLIT} (signal ${SIGNAL_TAG}, run ${TAG}, gate override ON)"
if uv run python scripts/run_allocator.py data=kite_v1 \
      +split="$SPLIT" +signal_tag="$SIGNAL_TAG" +require_gate_pass=false \
      +allocator.null_control=true \
      > "logs/${TAG}_allocator.log" 2>&1; then
    stamp "stage2 OK"
    say "stage 2 complete"
    tail -30 "logs/${TAG}_allocator.log"
else
    stamp "stage2 FAIL"; say "stage 2 FAILED — see logs/${TAG}_allocator.log"
    tail -30 "logs/${TAG}_allocator.log"; exit 1
fi

stamp "chain DONE"
say "chain complete — status in $STATUS"
