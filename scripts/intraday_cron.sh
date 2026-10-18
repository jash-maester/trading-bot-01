#!/usr/bin/env bash
# Cron entry point for the intraday Phase 0 collectors in the `paper` container
# (14_weekend_scoping_notes.md section A7). Data capture only; nothing trades.
#   intraday_cron.sh recorder | announcements
# Sources .env for KITE_API_KEY (the access token comes from the dashboard login).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
mkdir -p logs/intraday
case "${1:-}" in
  recorder)      LOG="logs/intraday/recorder_$(date +%F).log"; CMD=(scripts/tick_recorder.py) ;;
  announcements) LOG="logs/intraday/announcements_$(date +%F).log"; CMD=(scripts/nse_announcements.py) ;;
  *) echo "usage: $0 recorder|announcements" >&2; exit 2 ;;
esac
echo "=== $(date -Is) $1 ===" >> "$LOG"
uv run python "${CMD[@]}" >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
