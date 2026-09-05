#!/usr/bin/env bash
# Bootstrap / verify the Windows training box, from inside WSL.
#
# This exists as a file rather than an inline `ssh ... "..."` command because the
# call chain is make -> sh -> ssh -> PowerShell -> wsl -> bash, and PowerShell
# re-parses argv on the way through. Anything with nested quotes, `&&`, or a
# Python one-liner gets mangled — the first version of this failed with a
# PowerShell ParserError on an embedded quote. Shipping a script and executing
# it by path means only the path has to survive the journey.
#
# Usage (from the Mac):  make win-setup   /   make win-test
set -euo pipefail

CMD="${1:-setup}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

say() { printf '\n=== %s ===\n' "$1"; }

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        say "installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv --version
}

export DOCKER_CONFIG="$(pwd)/.docker"

case "$CMD" in
services)
    # Anything with pipes, loops or $( ) belongs here rather than in an inline
    # ssh command — PowerShell re-parses argv in transit and mangles all three.
    mkdir -p .docker && echo '{}' > .docker/config.json
    DC="docker compose -f docker/docker-compose.yml"
    say "starting postgres + mlflow"
    $DC up -d postgres mlflow
    say "waiting for health"
    for _ in $(seq 1 45); do
        st=$($DC ps --format '{{.Health}}' | tr '\n' ' ')
        case "$st" in *starting*) sleep 4 ;; *) break ;; esac
    done
    $DC ps --format 'table {{.Service}}\t{{.Health}}\t{{.Ports}}'
    say "endpoints"
    pg_ok=$($DC exec -T postgres pg_isready -U trader -d trader 2>&1 || true)
    echo "  postgres: $pg_ok"
    echo "  mlflow:   $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:5555/health || echo unreachable)"
    say "schema"
    $DC exec -T postgres psql -U trader -d trader -tAc \
        "select table_schema||'.'||table_name from information_schema.tables
         where table_schema in ('market','ledger') order by 1" 2>&1 | sed 's/^/  /'
    $DC exec -T postgres psql -U trader -d trader -tAc \
        "select 'alembic: '||version_num from alembic_version" 2>&1 | sed 's/^/  /'
    ;;

runbg)
    # Background pass-through for long training runs. `run` execs in the
    # foreground, so the job dies when the SSH channel closes; this detaches it
    # with setsid+nohup, writes a timestamped log under logs/ and a pidfile, and
    # prints both so the caller can poll. Everything after the command name is
    # forwarded verbatim to `uv run python`.
    ensure_uv >/dev/null
    shift || true
    set -a; [ -f .env ] && . ./.env; set +a
    mkdir -p logs
    tag="${RUN_TAG:-run}"
    log="logs/${tag}.log"
    pidf="logs/${tag}.pid"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
        say "REFUSING: ${tag} already running as pid $(cat "$pidf")"
        exit 1
    fi
    say "launching (background): $*"
    setsid nohup uv run python "$@" > "$log" 2>&1 &
    echo $! > "$pidf"
    sleep 2
    say "pid $(cat "$pidf")  log $log"
    ;;

run)
    # Pass-through for Hydra apps: everything after the command name is forwarded
    # verbatim, e.g.  win_bootstrap.sh run scripts/train.py model=mlp_regime seed=1
    ensure_uv >/dev/null
    shift || true
    set -a; [ -f .env ] && . ./.env; set +a
    say "running: $*"
    exec uv run python "$@"
    ;;

setup)
    say "host"
    echo "pwd:    $(pwd)"
    echo "kernel: $(uname -sr)"
    echo "cores:  $(nproc)   ram: $(free -g | awk '/^Mem:/{print $2}') GB"

    ensure_uv

    say "uv sync"
    # No --extra zerodha: the training box never talks to Kite. Data arrives by
    # rsync, so kiteconnect and a live token are needless attack surface here.
    uv sync

    say "torch / CUDA"
    # Written to a temp file rather than passed with -c: a Python one-liner with
    # quotes is exactly what PowerShell mangles upstream.
    cat > /tmp/_probe.py <<'PY'
import torch
print("torch          ", torch.__version__)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print("vram           ", round(p.total_memory / 2**30, 2), "GiB")
    print("capability     ", f"{p.major}.{p.minor}")
    print("bf16 supported ", torch.cuda.is_bf16_supported())
else:
    print("!! CUDA NOT VISIBLE — training would fall back to CPU")
PY
    uv run python /tmp/_probe.py
    rm -f /tmp/_probe.py

    say "data present?"
    for d in data/panels data/panels_kite data/kite_ohlcv; do
        if [ -d "$d" ]; then
            printf '  %-22s %s\n' "$d" "$(du -sh "$d" | cut -f1)"
        else
            printf '  %-22s MISSING\n' "$d"
        fi
    done
    ;;

test)
    ensure_uv >/dev/null
    say "ruff";   uv run ruff check .
    say "mypy";   uv run mypy src
    say "pytest"; uv run pytest tests/unit -q
    ;;

*)
    echo "unknown command: $CMD (expected 'setup' or 'test')" >&2
    exit 2
    ;;
esac
