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

case "$CMD" in
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
