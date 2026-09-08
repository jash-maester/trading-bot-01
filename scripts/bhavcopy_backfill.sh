#!/usr/bin/env bash
# Backfill NSE's full-market daily bars, 2010 to today.
#
# ~4,200 trading sessions at roughly 1.1s each: about 80 minutes on a cold
# cache, and near-instant on a warm one. Resumable -- every body is cached and
# a progress marker is flushed every 100 sessions -- so an interrupted run
# picks up where it stopped rather than starting over.
#
#   RUN_TAG=bhav bash scripts/bhavcopy_backfill.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-bhav}"
FROM="${FROM:-2010-01-01}"
STATUS="logs/bhavcopy_${TAG}.status"
mkdir -p logs
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

say "bhavcopy backfill from ${FROM}"
if uv run python scripts/fetch_bhavcopy.py --from "$FROM" \
     > "logs/${TAG}_fetch.log" 2>&1; then
    stamp "fetch OK"; tail -24 "logs/${TAG}_fetch.log"
else
    stamp "fetch FAIL"; tail -40 "logs/${TAG}_fetch.log"; exit 1
fi
stamp "chain DONE"
say "complete — status in $STATUS"
