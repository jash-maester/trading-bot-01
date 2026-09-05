"""Feature computation for the aligned OHLCV panel.

All features use strictly backward-looking windows (positive shifts only).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

if TYPE_CHECKING:  # omegaconf is only needed for the panel-path helper's type
    from omegaconf import DictConfig

_SQRT_252 = math.sqrt(252.0)
_RSI_ALPHA = 1.0 / 14.0
_ATR_ALPHA = 1.0 / 14.0

# Float feature columns that must be non-null (and non-NaN) on tradeable rows.
FEATURE_COLS: list[str] = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "realized_vol_20d",
    "realized_vol_60d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bbw_20",
    "z_close_20",
    "volume_z_20",
    "dollar_volume_20",
    "atr_14",
    "beta_nifty_60d",
]

# Longest backward-looking window each feature depends on, in TRADING days.
#
# This is the single source of truth for "how far back does a feature see", and
# the walk-forward purge guard (trader.training.walk_forward.compute_windows) is
# derived from it: a purge gap shorter than the longest window here means the
# train and val feature windows physically overlap, which is leakage.
#
# EWM features (rsi_14, atr_14, macd*) have no hard cut-off; the value recorded
# is their `min_samples` / effective span, which is what `compute_features`
# actually requires before emitting a non-null value.
#
# Keep in sync with the window literals below — `test_alignment_features.py`
# scans this module's source and fails if any `window_size=`/`span=`/`.shift(`
# literal exceeds MAX_FEATURE_LOOKBACK_DAYS, and if any FEATURE_COLS entry is
# missing from this table.
FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    "log_return_1d": 1,
    "log_return_5d": 5,
    "log_return_20d": 20,
    "realized_vol_20d": 20,
    "realized_vol_60d": 60,
    "rsi_14": 14,
    "macd": 26,          # slow EMA
    "macd_signal": 35,   # EMA9 of MACD(12, 26)
    "macd_hist": 35,
    "bbw_20": 20,
    "z_close_20": 20,
    "volume_z_20": 20,
    "dollar_volume_20": 20,
    "atr_14": 14,
    "beta_nifty_60d": 60,
}

MAX_FEATURE_LOOKBACK_DAYS: int = max(FEATURE_LOOKBACK_DAYS.values())

# Where panels live when a data config does not say otherwise.  Every entrypoint
# resolves through `resolve_panels_root` so that `data.panels_root` actually
# selects the dataset instead of being declared and ignored.
DEFAULT_PANELS_ROOT = "data/panels"

_OHLC = ["open", "high", "low", "close", "adj_close"]

_PANEL_SPLITS = ("train", "val", "test", "full")


class MissingBenchmarkError(ValueError):
    """Raised when beta is asked for without benchmark index returns.

    Its own type so a caller can tell "you forgot the index" apart from any
    other ValueError, and so the failure cannot be swallowed by a bare
    ``except Exception`` that was written for I/O problems.
    """


# ── Panel location ─────────────────────────────────────────────────────────────


def resolve_panels_root(cfg: DictConfig, project_root: Path) -> Path:
    """Resolve ``data.panels_root`` against `project_root` and log what it found.

    `data.panels_root` was declared in ``configs/data/kite_v1.yaml`` and read by
    exactly one entrypoint; every other script hardcoded ``data/panels``, so the
    Kite panel (live beta, 2005–2026, corporate-action masking) was written and
    then orphaned — no config override could select it, and a smoke run silently
    trained on the stale yfinance panel instead.

    The log line is not decoration.  The failure mode this whole helper exists to
    kill is *silence*: a run that does not say which panel it opened cannot be
    told apart from a run that opened the wrong one.  So the resolved path, the
    splits present, the ticker count and the date span always go to the log.

    Missing panels are not an error here — `build_features.py` calls this before
    the panels exist.  Callers that need a specific split check it themselves.
    """
    panels_root = project_root / str(cfg.data.get("panels_root", DEFAULT_PANELS_ROOT))
    logger.info(f"Panels root: {panels_root}  (data.panels_root)")

    present = [s for s in _PANEL_SPLITS if (panels_root / f"{s}.parquet").exists()]
    if not present:
        logger.warning(
            f"  no panel parquets under {panels_root} yet "
            "— run scripts/build_features.py"
        )
        return panels_root

    for split in present:
        path = panels_root / f"{split}.parquet"
        try:
            summary = (
                pl.scan_parquet(path)
                .select(
                    pl.col("ticker").n_unique().alias("n_tickers"),
                    pl.col("date").min().alias("d_min"),
                    pl.col("date").max().alias("d_max"),
                    pl.len().alias("n_rows"),
                )
                .collect()
            )
        except Exception as exc:  # noqa: BLE001 — a bad panel must name itself
            logger.warning(f"  {split}.parquet: unreadable ({type(exc).__name__}: {exc})")
            continue
        row = summary.row(0, named=True)
        logger.info(
            f"  {split}.parquet: {row['n_tickers']} tickers, "
            f"{row['n_rows']:,} rows, {row['d_min']}..{row['d_max']}"
        )
    return panels_root


def compute_features(
    df: pl.DataFrame,
    index_rets: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute all features for an aligned (date × ticker) panel.

    Parameters
    ----------
    df:
        Aligned panel from ``alignment.align_panel``.  Must have columns:
        date, ticker, open, high, low, close, adj_close, volume,
        sector_id, is_tradeable.
    index_rets:
        DataFrame with columns ``date`` and ``index_return`` (log return of the
        NIFTY 50 index), used for ``beta_nifty_60d``.  **Required.**  Passing
        ``None`` raises ``MissingBenchmarkError`` — see :func:`_add_beta` for
        why a missing benchmark must not be silently absorbed.

    Returns
    -------
    pl.DataFrame
        Input panel with all FEATURE_COLS added.  Non-tradeable rows (and
        warm-up rows without sufficient history) have is_tradeable forced to
        False; all feature nulls/NaNs are filled with 0.0 (sentinel).

    Raises
    ------
    MissingBenchmarkError
        If `index_rets` is None.
    """
    df = df.sort(["ticker", "date"])

    # Replace zero sentinels with null so they propagate correctly through all
    # feature arithmetic (zero adj_close → non-tradeable row from alignment).
    df = df.with_columns(
        [
            pl.when(pl.col(c) > 0.0)
            .then(pl.col(c))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(c)
            for c in _OHLC
        ]
    )

    # Volume: null for rows where adj_close is null (non-tradeable) so that
    # rolling volume stats are not polluted by sentinel rows.
    df = df.with_columns(
        pl.when(pl.col("adj_close").is_not_null())
        .then(pl.col("volume").cast(pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("_vol")
    )

    df = _add_returns(df)
    df = _add_volatility(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger(df)
    df = _add_volume_features(df)
    df = _add_atr(df)
    df = _add_beta(df, index_rets)

    df = df.drop("_vol")

    # ── enforce: is_tradeable=False where any feature is null or NaN ─────────
    bad = pl.lit(value=False)
    for col in FEATURE_COLS:
        if col in df.columns:
            bad = bad | pl.col(col).is_null() | pl.col(col).is_nan()

    df = df.with_columns(
        (pl.col("is_tradeable") & ~bad).alias("is_tradeable")
    )

    # ── sentinel fill: null/NaN → 0.0 on non-tradeable rows ──────────────────
    present = [c for c in FEATURE_COLS if c in df.columns]
    df = df.with_columns(
        [pl.col(c).fill_null(0.0).fill_nan(0.0) for c in present]
    )

    # Restore price sentinels to 0.0 (were set to null for safe arithmetic)
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in _OHLC])

    return df.sort(["date", "ticker"])


# ── private helpers ────────────────────────────────────────────────────────────


def _add_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("adj_close") / pl.col("adj_close").shift(1))
            .log()
            .over("ticker")
            .alias("log_return_1d"),
            (pl.col("adj_close") / pl.col("adj_close").shift(5))
            .log()
            .over("ticker")
            .alias("log_return_5d"),
            (pl.col("adj_close") / pl.col("adj_close").shift(20))
            .log()
            .over("ticker")
            .alias("log_return_20d"),
        ]
    )


def _add_volatility(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                pl.col("log_return_1d")
                .rolling_std(window_size=20, min_samples=20)
                .over("ticker")
                * _SQRT_252
            ).alias("realized_vol_20d"),
            (
                pl.col("log_return_1d")
                .rolling_std(window_size=60, min_samples=60)
                .over("ticker")
                * _SQRT_252
            ).alias("realized_vol_60d"),
        ]
    )


def _add_rsi(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("adj_close").diff(1).over("ticker").alias("_delta")
    )
    df = df.with_columns(
        [
            pl.col("_delta").clip(lower_bound=0.0).alias("_gain"),
            (-pl.col("_delta")).clip(lower_bound=0.0).alias("_loss"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("_gain")
            .ewm_mean(alpha=_RSI_ALPHA, min_samples=14)
            .over("ticker")
            .alias("_avg_gain"),
            pl.col("_loss")
            .ewm_mean(alpha=_RSI_ALPHA, min_samples=14)
            .over("ticker")
            .alias("_avg_loss"),
        ]
    )
    # Add small epsilon to avg_loss to avoid 0/0 NaN when both are zero.
    df = df.with_columns(
        (
            100.0
            - 100.0 / (1.0 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10))
        ).alias("rsi_14")
    )
    return df.drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def _add_macd(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("adj_close")
            .ewm_mean(span=12, min_samples=12)
            .over("ticker")
            .alias("_ema12"),
            pl.col("adj_close")
            .ewm_mean(span=26, min_samples=26)
            .over("ticker")
            .alias("_ema26"),
        ]
    )
    df = df.with_columns(
        (pl.col("_ema12") - pl.col("_ema26")).alias("macd")
    )
    df = df.with_columns(
        pl.col("macd")
        .ewm_mean(span=9, min_samples=9)
        .over("ticker")
        .alias("macd_signal")
    )
    df = df.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
    )
    return df.drop(["_ema12", "_ema26"])


def _add_bollinger(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("adj_close")
            .rolling_mean(window_size=20, min_samples=20)
            .over("ticker")
            .alias("_bb_mid"),
            pl.col("adj_close")
            .rolling_std(window_size=20, min_samples=20)
            .over("ticker")
            .alias("_bb_std"),
        ]
    )
    df = df.with_columns(
        [
            # bbw = (upper - lower) / middle = 4 * std / mid
            (4.0 * pl.col("_bb_std") / pl.col("_bb_mid")).alias("bbw_20"),
            (
                (pl.col("adj_close") - pl.col("_bb_mid"))
                / (pl.col("_bb_std") + 1e-10)
            ).alias("z_close_20"),
        ]
    )
    return df.drop(["_bb_mid", "_bb_std"])


def _add_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("_vol")
            .rolling_mean(window_size=20, min_samples=20)
            .over("ticker")
            .alias("_vol_mean_20"),
            pl.col("_vol")
            .rolling_std(window_size=20, min_samples=20)
            .over("ticker")
            .alias("_vol_std_20"),
            (pl.col("adj_close") * pl.col("_vol"))
            .rolling_mean(window_size=20, min_samples=20)
            .over("ticker")
            .alias("dollar_volume_20"),
        ]
    )
    df = df.with_columns(
        (
            (pl.col("_vol") - pl.col("_vol_mean_20"))
            / (pl.col("_vol_std_20") + 1e-10)
        ).alias("volume_z_20")
    )
    return df.drop(["_vol_mean_20", "_vol_std_20"])


def _add_atr(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("close").shift(1).over("ticker").alias("_prev_close")
    )
    df = df.with_columns(
        pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("_prev_close")).abs(),
                (pl.col("low") - pl.col("_prev_close")).abs(),
            ]
        ).alias("_tr")
    )
    df = df.with_columns(
        pl.col("_tr")
        .ewm_mean(alpha=_ATR_ALPHA, min_samples=14)
        .over("ticker")
        .alias("atr_14")
    )
    return df.drop(["_prev_close", "_tr"])


def _add_beta(df: pl.DataFrame, index_rets: pl.DataFrame | None) -> pl.DataFrame:
    """Rolling 60-day beta of each ticker against the benchmark index.

    Failure mode, chosen deliberately
    ---------------------------------
    This function used to return a constant ``1.0`` when no index was supplied.
    That is how ``beta_nifty_60d`` came to be constant 1.0 across all 276,005
    tradeable training rows: ``data/ohlcv`` has no ``^NSEI``, so `index_rets`
    was None, and nothing said so.  Downstream, `compute_feature_stats`
    substitutes ``(0.0, 1.0)`` for a zero-variance feature, which normalises the
    dead channel to a fixed **+1** — a constant bias into every convolution
    rather than the harmless zero everyone assumed.

    So: **raise**, rather than emit nulls.  Nulls were the other candidate and
    they are worse here, because of how they interact with the tradeability
    mask.  `compute_features` forces ``is_tradeable=False`` on any row with a
    null feature; a null beta is null on *every* row, so the panel would be
    written with zero tradeable rows and the real error would only surface much
    later, in the env, as an empty universe.  Raising fails at the point of the
    defect with the fix in the message, costs nothing, and cannot be misread.

    Nulls are still the right answer *per row* when the benchmark exists but
    does not cover a given date, or has degenerate variance over the window:
    that is a genuinely local gap, and `is_tradeable=False` on exactly those
    rows is the correct, non-destructive outcome.  Those rows previously also
    collapsed to beta = 1.0, contradicting the comment in
    ``scripts/build_features.py`` which claimed uncovered dates went null.
    """
    if index_rets is None:
        raise MissingBenchmarkError(
            "beta_nifty_60d needs benchmark index returns, but index_rets is None. "
            "A missing benchmark must never masquerade as beta = 1.0 — that is "
            "exactly the bug that left the channel dead across the whole training "
            "set. Fetch the index (configs/data/kite_v1.yaml: fetch_index: true, "
            "index_ticker: '^NSEI') and confirm it is in the OHLCV store; "
            "data/ohlcv (yfinance) does not contain it, data/kite_ohlcv does."
        )

    # Compute rolling index stats on the unique-date spine first, then join.
    # This avoids polluting per-ticker rolling windows with repeated index values.
    idx = (
        index_rets.sort("date")
        .with_columns(
            [
                pl.col("index_return")
                .rolling_mean(window_size=60, min_samples=30)
                .alias("_i_mean"),
                (pl.col("index_return").pow(2))
                .rolling_mean(window_size=60, min_samples=30)
                .alias("_i_e_r2"),
            ]
        )
        .select(["date", "index_return", "_i_mean", "_i_e_r2"])
    )
    df = df.join(idx, on="date", how="left")

    # Coverage check: a benchmark that is present but short is the same bug in a
    # quieter costume — it used to leave beta = 1.0 on every uncovered date.
    # Those rows now go null (→ is_tradeable=False), which is loud in the row
    # counts; say so here too, so it is loud in the log as well.
    panel_dates = df["date"].n_unique()
    covered = df.filter(pl.col("index_return").is_not_null())["date"].n_unique()
    if covered < panel_dates:
        logger.warning(
            f"Benchmark index covers {covered:,}/{panel_dates:,} panel dates. "
            f"beta_nifty_60d is null on the {panel_dates - covered:,} uncovered "
            "dates, so every row on those dates becomes non-tradeable. Widen the "
            "index fetch (data.start_date / data.end_date) if that is not intended."
        )

    df = df.with_columns(
        (pl.col("log_return_1d") * pl.col("index_return")).alias("_xy")
    )
    df = df.with_columns(
        [
            pl.col("_xy")
            .rolling_mean(window_size=60, min_samples=30)
            .over("ticker")
            .alias("_e_xy"),
            pl.col("log_return_1d")
            .rolling_mean(window_size=60, min_samples=30)
            .over("ticker")
            .alias("_e_x"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("_e_xy") - pl.col("_e_x") * pl.col("_i_mean")).alias("_cov"),
            (pl.col("_i_e_r2") - pl.col("_i_mean").pow(2)).alias("_var"),
        ]
    )
    # Null — NOT 1.0 — when the window is unusable (index not yet warmed up,
    # date uncovered by the benchmark, or degenerate index variance).  Null
    # propagates into is_tradeable=False for exactly those rows, which is the
    # honest outcome; 1.0 was a fabricated beta indistinguishable from a real one.
    df = df.with_columns(
        pl.when(pl.col("_var") > 1e-10)
        .then(pl.col("_cov") / pl.col("_var"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("beta_nifty_60d")
    )
    return df.drop(
        ["index_return", "_i_mean", "_i_e_r2", "_xy", "_e_xy", "_e_x", "_cov", "_var"]
    )
