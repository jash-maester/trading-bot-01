"""Typer entry point for the ``trader`` console script.

``pyproject.toml`` has declared ``trader = "trader.cli:app"`` since M0, but this
module was never written — so the installed ``.venv/bin/trader`` shim existed
and crashed with ``ModuleNotFoundError`` on every invocation.

Scope is deliberately narrow. The pipeline entry points (``train``,
``walk_forward``, ``build_features``, ...) are Hydra applications: Hydra owns
``sys.argv``, resolves config groups from ``configs/``, and manages its own run
directory. Proxying that through Typer would either swallow Hydra's override
syntax or reimplement it badly, so those stay as ``scripts/*.py`` and this CLI
does not wrap them.

What it does instead is answer the questions that have no home today: is the
environment actually wired up, are the services reachable, did the data survive
the move. Those were all answered by ad-hoc shell during the Mac mini transfer;
this makes them a command.
"""
from __future__ import annotations

import os
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Operational helpers for the trading-bot project.",
)

# Repo root: this file lives at <root>/src/trader/cli.py
ROOT = Path(__file__).resolve().parents[2]

_OK = "ok"
_BAD = "FAIL"
_WARN = "warn"


def _status(label: str, ok: bool | None, detail: str = "") -> None:
    """Print one aligned status line. ``ok=None`` renders as a warning."""
    mark = _OK if ok else (_WARN if ok is None else _BAD)
    typer.echo(f"  [{mark:>4}] {label:<26} {detail}")


@app.command()
def version() -> None:
    """Show interpreter, key dependency versions, and the torch device."""
    import importlib.metadata as md
    import platform
    import sys

    typer.echo(f"{'python':<11}{sys.version.split()[0]}  ({platform.machine()})")
    for pkg in ("torch", "gymnasium", "polars", "pandas", "mlflow", "quantstats"):
        try:
            typer.echo(f"{pkg:<11}{md.version(pkg)}")
        except md.PackageNotFoundError:
            typer.echo(f"{pkg:<11}(not installed)")

    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device = "mps (Apple Metal)"
        else:
            device = "cpu"
        typer.echo(f"{'device':<11}{device}")
    except Exception as e:  # noqa: BLE001 — informational command, never fatal
        typer.echo(f"{'device':<11}unavailable ({e})")


@app.command()
def data() -> None:
    """Report panel presence and verify the tracked SHA256 provenance sidecars."""
    import hashlib

    panels = ROOT / "data" / "panels"
    if not panels.is_dir():
        typer.echo(f"no panel directory at {panels}")
        raise typer.Exit(code=1)

    failures = 0
    for split in ("train", "val", "test"):
        parquet = panels / f"{split}.parquet"
        sidecar = panels / f"{split}.sha256"
        if not parquet.exists():
            _status(f"{split}.parquet", False, "missing — run build_features.py")
            failures += 1
            continue

        size_mb = parquet.stat().st_size / 2**20
        if not sidecar.exists():
            _status(f"{split}.parquet", None, f"{size_mb:6.1f} MB, no .sha256 sidecar")
            continue

        # Streamed, not read whole: these files are tens of MB apiece.
        digest = hashlib.sha256()
        with parquet.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        recorded = sidecar.read_text().strip()
        match = digest.hexdigest() == recorded
        _status(
            f"{split}.parquet",
            match,
            f"{size_mb:6.1f} MB  sha256 {'matches' if match else 'MISMATCH'}",
        )
        failures += 0 if match else 1

    raise typer.Exit(code=1 if failures else 0)


@app.command()
def doctor() -> None:
    """Check that the environment and both services are actually usable."""
    import importlib.util

    typer.echo("environment")
    _status("repo root", (ROOT / "pyproject.toml").exists(), str(ROOT))
    _status(".env present", (ROOT / ".env").exists(), "not auto-loaded — source it yourself")

    try:
        import torch

        _status(
            "torch / device",
            True,
            f"{torch.__version__}, mps={torch.backends.mps.is_available()}",
        )
    except Exception as e:  # noqa: BLE001
        _status("torch / device", False, str(e))

    _status("quantstats", importlib.util.find_spec("quantstats") is not None)
    _status(
        "kiteconnect",
        None if importlib.util.find_spec("kiteconnect") is None else True,
        "optional — only needed for the live data feed",
    )

    typer.echo("\nservices")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    try:
        from sqlalchemy import text

        from trader.db.engine import get_engine

        with get_engine().connect() as conn:
            n = conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema IN ('market','ledger')"
                )
            ).scalar_one()
            rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        _status("postgres", True, f"{host}:{port}  {n} tables, alembic {rev}")
    except Exception as e:  # noqa: BLE001
        _status("postgres", False, f"{host}:{port}  {type(e).__name__}: {str(e)[:60]}")

    mlflow_port = os.getenv("MLFLOW_PORT", "5555")
    try:
        import urllib.request

        with urllib.request.urlopen(  # noqa: S310 — fixed localhost URL
            f"http://127.0.0.1:{mlflow_port}/health", timeout=3
        ) as resp:
            _status("mlflow", resp.status == 200, f"127.0.0.1:{mlflow_port}")
    except Exception as e:  # noqa: BLE001
        _status("mlflow", False, f"127.0.0.1:{mlflow_port}  {type(e).__name__}")

    typer.echo("\npipeline entry points are Hydra apps, not subcommands:")
    for script in ("build_universe", "fetch_data", "build_features", "train",
                   "walk_forward", "evaluate"):
        typer.echo(f"  uv run python scripts/{script}.py")


if __name__ == "__main__":
    app()
