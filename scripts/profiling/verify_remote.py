"""Confirm the panels and services on the training box before a run."""
import polars as pl, urllib.request
from trader.data.features import FEATURE_COLS
from trader.data.universe import active_tickers

for s in ("train", "val", "test"):
    d = pl.read_parquet(f"data/panels_kite/{s}.parquet", columns=["date", "ticker"])
    print(f"{s:6s} rows={d.height:>9,} tickers={d['ticker'].n_unique()} "
          f"{d['date'].min()}..{d['date'].max()}")

tr = pl.read_parquet("data/panels_kite/train.parquet").filter(pl.col("is_tradeable"))
dead = [c for c in FEATURE_COLS if (tr[c].std() or 0) <= 1e-12]
print("dead features:", dead or "none")
print("active universe:", len(active_tickers()))
try:
    urllib.request.urlopen("http://127.0.0.1:5555/health", timeout=5).read()
    print("mlflow: OK")
except Exception as e:
    print("mlflow: DOWN", e)
