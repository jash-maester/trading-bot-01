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
# The OOS slice is cut to the signal's prediction span, so it is per-signal:
# r4_v2's warm-up prefix starts its predictions 59 trading days before r4_v1's,
# and a slice built for one silently truncates the other.
SPLIT="${SPLIT:-oos_${SIGNAL_TAG}}"
# Whether run_allocator must see a PASS in gate.json. Default on: the gate is
# now a window-level test (12_gate_decision.md) that r4_v1 and r4_v2 both
# clear. Set REQUIRE_GATE=false only to measure a failed signal deliberately;
# the verdict is stamped into every MLflow run either way.
REQUIRE_GATE="${REQUIRE_GATE:-true}"
# Extra Hydra overrides appended verbatim to the allocator stage. Tax is the
# reason this exists: `+apply_tax=true` was passed by hand on every run that
# quoted a post-tax number, so the chain could not reproduce them. Anything
# passed here is echoed into the log and the status file, so a table can be
# traced back to the flags that produced it.
EXTRA="${EXTRA:-}"
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
# `++`, not `+`, on every allocator key. `+` means "append a key that does not
# exist", and until 2026-09-07 none of these existed -- `cfg.allocator` was read
# by run_allocator.py and populated by nothing. configs/allocator/default.yaml
# now defines them, so `+` fails with "An item is already at
# 'allocator.null_control'". `++` appends OR overrides and is correct either
# way. `split`, `signal_tag` and `require_gate_pass` are still genuinely absent
# from the struct, so those keep a single `+`.
say "stage 2: allocator grid on ${SPLIT} (signal ${SIGNAL_TAG}, run ${TAG}, require_gate_pass=${REQUIRE_GATE}) ${EXTRA}"
stamp "stage2 EXTRA=${EXTRA:-<none>}"
# shellcheck disable=SC2086  # EXTRA is a deliberate word-split list of overrides
if uv run python scripts/run_allocator.py data=kite_v1 \
      +split="$SPLIT" +signal_tag="$SIGNAL_TAG" +require_gate_pass="$REQUIRE_GATE" \
      ++allocator.null_control=true $EXTRA \
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
