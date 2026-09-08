#!/usr/bin/env bash
# Does the stop's 21-step quarantine bar a trade that was worth making?
#
# `audit/N1_STOP_VETO_HEADROOM.md` measured what happens after a stop fires, to
# decide whether the news veto in `13` §4 was worth building. It answered that
# question and raised a sharper one on the way.
#
# Over 60 trading days a stopped name beats the market by +4.36% (stop10 arm),
# and the DEPTH of the fall that triggered the stop predicts the size of the
# recovery -- Spearman -0.148, p 0.000, n 734. That is mean reversion, it is
# idiosyncratic rather than market, and it is large. Meanwhile the overlay bars
# re-buying a stopped name for `stop_cooldown_steps` = 21. If R4 would have
# picked the name back up inside that window, the quarantine is refusing a
# trade the data says was good.
#
# This sweeps the cooldown alone -- same signal, same K, same cadence, same
# stop thresholds -- so any difference is attributable to the quarantine.
#
#   RUN_TAG=cooldown bash scripts/cooldown_sweep.sh
#
# PREDICTION, recorded before the run: the effect is SMALL and possibly
# negative. The 60-day reversion is real, but R4 ranks on momentum-shaped
# features and a name that just fell 12% is unlikely to re-enter its top 20
# inside 21 days, so the quarantine may rarely bind at all. Where it does bind,
# re-buying and re-stopping the same name pays the flat Rs 15.34 demat debit
# twice. I expect |delta CAGR| < 1pp between cooldown 0 and 21.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-cooldown}"
SIGNAL="${SIGNAL_TAG:-r4_v2}"
SPLIT="${SPLIT:-oos_r4_v2}"
STATUS="logs/cooldown_${TAG}.status"
mkdir -p logs
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

# Both stop shapes, four cooldowns each, plus the no-stop reference.
GRID='[{name: none},
{name: v_cd0,  stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 0},
{name: v_cd5,  stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 5},
{name: v_cd21, stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 21},
{name: v_cd42, stop_vol_mult: 1.0, stop_vol_horizon_days: 20, stop_cooldown_steps: 42},
{name: f_cd0,  stop_loss: 0.10, stop_cooldown_steps: 0},
{name: f_cd5,  stop_loss: 0.10, stop_cooldown_steps: 5},
{name: f_cd21, stop_loss: 0.10, stop_cooldown_steps: 21},
{name: f_cd42, stop_loss: 0.10, stop_cooldown_steps: 42}]'
GRID=$(echo "$GRID" | tr -d '\n')

say "cooldown sweep (${SIGNAL} on ${SPLIT}, after tax, K=20 monthly 20d)"
if uv run python scripts/run_allocator.py data=kite_v1 \
      +split="$SPLIT" +signal_tag="$SIGNAL" +require_gate_pass=true \
      +apply_tax=true \
      ++allocator.null_control=false \
      ++allocator.k_grid=[20] \
      ++allocator.freq_grid=[monthly] \
      ++allocator.horizon_grid=[20d] \
      ++allocator.risk_grid="$GRID" \
      > "logs/${TAG}_alloc.log" 2>&1; then
    stamp "sweep OK"
    sed -n '/^strategy/,/^-\{60,\}$/p' "logs/${TAG}_alloc.log" | tail -20
else
    stamp "sweep FAIL"; tail -40 "logs/${TAG}_alloc.log"; exit 1
fi
stamp "chain DONE"
say "complete — status in $STATUS"
