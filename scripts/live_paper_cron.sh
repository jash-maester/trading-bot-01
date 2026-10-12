#!/usr/bin/env bash
# Cron entry point for scripts/live_paper.py inside the `paper` container.
#   live_paper_cron.sh deploy | mark
# Sources .env on every run so the daily Kite login takes effect without a
# container restart. Read-only Kite use only (profile, quote).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
mkdir -p logs/paper
# TRADER_LIVE_TAG separates the books' logs (unset = the frozen P2 book).
LOG="logs/paper/live${TRADER_LIVE_TAG:+_$TRADER_LIVE_TAG}_$(date +%F).log"
echo "=== $(date -Is) live_paper $* ===" >> "$LOG"
uv run python scripts/live_paper.py "$@" --prices live >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
