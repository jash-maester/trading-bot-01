#!/usr/bin/env bash
# A2 — detachable launcher for the profiling runs on the Windows box.
#
# `win_bootstrap.sh run` execs the python process in the foreground, so its
# stdout dies with the ssh pipe.  A2's runs take 20-40 minutes, longer than the
# caller will hold the connection.  This wrapper nohups the job and writes both
# a log and (via the script's own `emit`) a JSON result under outputs/a2/.
#
#   start:  a2_run.sh start <tag> <subcommand and hydra-free args...>
#   poll:   a2_run.sh poll  <tag>
#   log:    a2_run.sh log   <tag>
#   ps:     a2_run.sh ps
set -uo pipefail
cd /mnt/d/trading-bot-01
export PATH="$HOME/.local/bin:$PATH"
mkdir -p outputs/a2

CMD="${1:-ps}"
case "$CMD" in
start)
    TAG="$2"; shift 2
    nohup uv run python scripts/profiling/a2_ppo_update.py "$@" \
        > "outputs/a2/${TAG}.log" 2>&1 &
    echo "started tag=$TAG pid=$! args=$*"
    ;;
poll)
    TAG="$2"
    if pgrep -f "a2_ppo_update.py" >/dev/null; then echo "STATUS: running";
    else echo "STATUS: idle"; fi
    echo "--- tail ${TAG}.log ---"
    tail -c 4000 "outputs/a2/${TAG}.log" 2>/dev/null || echo "(no log yet)"
    ;;
log)
    tail -c 200000 "outputs/a2/${2}.log"
    ;;
cat)
    cat "outputs/a2/${2}"
    ;;
ps)
    ps -eo pid,etime,rss,args | grep -E "a2_ppo_update" | grep -v grep
    free -g | head -2
    ;;
esac
