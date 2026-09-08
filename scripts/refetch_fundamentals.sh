#!/usr/bin/env bash
# Re-run the fundamentals pipeline after the two coverage fixes.
#
# Neither fix needs a full re-download. The results index is cached per symbol
# and the XBRL bodies are cached per document, so this is mostly a re-PARSE:
#
#   * the six ampersand symbols (M&M, M&MFIN, J&KBANK, ARE&M, GVT&D, GMRP&UI)
#     were never successfully fetched, and the percent-encoded cache filename
#     differs from the old one, so they fetch fresh and nothing stale is hit;
#   * every other symbol reads from cache;
#   * fetch_xbrl_figures.py --restart re-parses all ~9,400 cached bodies with
#     the banking alias map, downloading only the ampersand tickers' documents.
#
# Expect a few minutes of network for the new symbols and a few more of pure
# parse, against ~3.5 hours for the original run.
#
#   RUN_TAG=refetch bash scripts/refetch_fundamentals.sh
#
# The old artefacts are kept beside the new ones as *.pre_banking.parquet so
# the before/after is checkable rather than asserted.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TAG="${RUN_TAG:-refetch}"
STATUS="logs/refetch_${TAG}.status"
mkdir -p logs
: > "$STATUS"

say()   { echo "=== $* ==="; }
stamp() { echo "$(date -Is) $*" >> "$STATUS"; }

set -a; [ -f .env ] && . ./.env; set +a

for f in data/ext/results.parquet data/ext/fundamentals.parquet; do
    [ -f "$f" ] && [ ! -f "${f%.parquet}.pre_banking.parquet" ] && \
        cp "$f" "${f%.parquet}.pre_banking.parquet" && say "kept ${f%.parquet}.pre_banking.parquet"
done

say "stage 1: results index (cached except the six ampersand symbols)"
if uv run python scripts/fetch_fundamentals.py --out data/ext/results.parquet \
     > "logs/${TAG}_index.log" 2>&1; then
    stamp "stage1 OK"; tail -22 "logs/${TAG}_index.log"
else
    stamp "stage1 FAIL"; tail -40 "logs/${TAG}_index.log"; exit 1
fi

say "stage 2: re-parse every XBRL body with the banking alias map"
if uv run python scripts/fetch_xbrl_figures.py --restart \
     > "logs/${TAG}_xbrl.log" 2>&1; then
    stamp "stage2 OK"; tail -12 "logs/${TAG}_xbrl.log"
else
    stamp "stage2 FAIL"; tail -40 "logs/${TAG}_xbrl.log"; exit 1
fi

say "stage 3: what changed"
if uv run python scripts/fundamentals_coverage.py > "logs/${TAG}_coverage.log" 2>&1; then
    stamp "stage3 OK"; cat "logs/${TAG}_coverage.log"
else
    stamp "stage3 FAIL"; tail -40 "logs/${TAG}_coverage.log"; exit 1
fi

stamp "chain DONE"
say "chain complete — status in $STATUS"
