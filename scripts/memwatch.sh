#!/usr/bin/env bash
# Sample host/WSL memory and GPU while a long run executes.
#
#   bash scripts/memwatch.sh logs/mem_myrun.csv 15 &
#   MEMWATCH=$!
#   ... long run ...
#   kill $MEMWATCH
#
# WHY THIS EXISTS. On 2026-09-09 the training box rebooted uncleanly at
# ~10:55 IST, within a minute of a run finishing. Windows logged Event 41 with
# NO bugcheck, no crash dump and no WHEA error, and WSL logged no OOM kill --
# the signature of a power loss or a hard freeze, not a software crash. But the
# question "did our run exhaust memory and hang the host?" could not be
# answered, because nothing had recorded what the run actually used.
#
# It is a fair question there: .wslconfig sets memory=30G on a 31.6 GiB host,
# leaving Windows 1.6 GiB if WSL ever fills its cap. A WSL balloon starving
# Windows would produce exactly the evidence pattern observed. Our measured
# footprint was nowhere near that -- but "nowhere near" was inferred after the
# fact, not measured. This makes it measured.
set -uo pipefail
OUT="${1:-logs/memwatch.csv}"
INTERVAL="${2:-15}"
mkdir -p "$(dirname "$OUT")"
echo "ts,used_mb,avail_mb,cache_mb,swap_used_mb,gpu_used_mb,gpu_util" > "$OUT"

# The Mac is now a working copy of the box (both ran the same allocator sweep
# on 2026-09-09 and produced byte-identical numbers), so this has to sample on
# either. `free` and `date -Is` are GNU-only.
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

sample_linux() {
    read -r u a c <<<"$(free -m | awk '/^Mem:/{print $3, $7, $6}')"
    local s; s=$(free -m | awk '/^Swap:/{print $3}')
    echo "${u},${a},${c},${s}"
}

sample_darwin() {
    /usr/bin/vm_stat | /usr/bin/awk -v total="$(sysctl -n hw.memsize)" '
        /page size of/ { ps = $8 }
        /Pages free/            { free_p  = $3 }
        /Pages inactive/        { inact_p = $3 }
        /Pages speculative/     { spec_p  = $3 }
        /File-backed pages/     { file_p  = $3 }
        END {
            if (ps == 0) ps = 4096
            gsub(/\./, "", free_p); gsub(/\./, "", inact_p)
            gsub(/\./, "", spec_p); gsub(/\./, "", file_p)
            avail = (free_p + inact_p + spec_p) * ps / 1048576
            cache = file_p * ps / 1048576
            used  = total / 1048576 - avail
            printf "%d,%d,%d,%d", used, avail, cache, 0
        }'
}

case "$(uname -s)" in
    Darwin) SAMPLE=sample_darwin ;;
    *)      SAMPLE=sample_linux  ;;
esac

while true; do
    mem=$($SAMPLE)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)
    [ -z "$gpu" ] && gpu="0,0"
    echo "$(stamp),${mem},${gpu}" >> "$OUT"
    sleep "$INTERVAL"
done
