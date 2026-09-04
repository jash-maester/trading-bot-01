#!/usr/bin/env bash
# A2 — run the whole 4060 measurement suite unattended, in cost order.
#
# Each step writes its own JSON under outputs/a2/ via the script's `emit`,
# so a later step dying does not lose the earlier results.  Launch detached:
#
#   ssh jashm@192.168.1.7 'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/profiling/a2_all.sh'
#
# then poll:  a2_run.sh cat <name>.json
set -uo pipefail
cd /mnt/d/trading-bot-01
export PATH="$HOME/.local/bin:$PATH"
mkdir -p outputs/a2
P="uv run python scripts/profiling/a2_ppo_update.py"
O="--outdir outputs/a2"

run() { echo "### $* ###"; timeout "$1" bash -c "${*:2}"; echo "### rc=$? ###"; }

{
echo "=== a2_all start $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,power.draw --format=csv,noheader

# 1. closed-form FLOPs (instant)
run 300 "$P analytic --n-tickers 504 --lookback 60 $O"

# 2. encoder-invocation ratio (seconds, no model)
run 900 "$P dates --panel synthetic --n-tickers 504 --seeds 0,1,2 $O"
run 900 "$P dates --panel real --seeds 0,1,2 $O"

# 3. hardware probe (~2 min)
run 900 "uv run python scripts/profiling/a2_hw_probe.py | tee outputs/a2/hw_probe.json"

# 4. per-gradient-step scaling, uninstrumented.  Smallest first so that a
#    later OOM/kill still leaves the cheap cells on disk.
run 7200 "$P scale --tickers 163,300,504,645 --lookbacks 20,30,60 --iters 5 $O"

# 5. VRAM / host-RAM ceiling sweep
run 7200 "$P ceiling --tickers 163,300,504,645 --minibatches 32,64,128,256,512 --lookback 60 $O"

# 5b. eager vs torch.compile at the production shape (production sets
#     compile: true on mlp_regime, so this is the configuration that ships).
run 3600 "$P scale --tickers 504 --lookbacks 60 --iters 5 --compile --outdir outputs/a2/compiled"

# 6. torch.profiler over 6 gradient steps at production shape
run 3600 "$P profile --n-tickers 504 --lookback 60 --steps 6 $O"
run 3600 "$P profile --n-tickers 163 --lookback 60 --steps 6 $O"

# 7. instrumented byte accounting.  n_steps/n_minibatches reduced 8x so the
#    run is ~16 gradient steps; the per-transition byte formula it verifies
#    scales exactly to the production 256/32 shape.
run 7200 "$P update --n-tickers 504 --n-steps 32 --n-minibatches 4 --updates 1 $O"

# 8. clean end-to-end update wall clock, no instrumentation
run 10800 "$P update --n-tickers 504 --n-steps 32 --n-minibatches 4 --updates 1 --no-probe $O"
run 10800 "$P update --panel real --n-steps 32 --n-minibatches 4 --updates 1 --no-probe $O"

echo "=== a2_all done $(date -Is) ==="
} > outputs/a2/a2_all.log 2>&1
