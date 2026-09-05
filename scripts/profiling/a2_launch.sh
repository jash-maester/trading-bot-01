#!/usr/bin/env bash
# Detach a2_all.sh from the SSH connection.
#
# Exists because the remote SSH shell is PowerShell, which re-parses argv in
# transit: an inline `nohup ... & echo $!` loses its redirect and its ampersand,
# so the job starts with nowhere to write and dies unnoticed. Invoking a bare
# path is the only form that survives, so the nohup lives here instead.
set -uo pipefail
cd /mnt/d/trading-bot-01
mkdir -p outputs/a2
setsid nohup bash scripts/profiling/a2_all.sh > outputs/a2/launch.log 2>&1 < /dev/null &
sleep 2
pgrep -f a2_all.sh > outputs/a2/a2_all.pid 2>/dev/null || true
echo "launched; pid file:"
cat outputs/a2/a2_all.pid 2>/dev/null || echo "(none — check outputs/a2/launch.log)"
