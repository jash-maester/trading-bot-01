"""Feature computation for the aligned OHLCV panel.

All features use strictly backward-looking windows (positive shifts only).
"""
from __future__ import annotations

import math

import polars as pl

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

_OHLC = ["open", "high", "low", "close", "adj_close"]


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
        Optional DataFrame with columns ``date`` and ``index_return``
        (log return of the NIFTY 50 index).  When omitted, beta defaults to 1.

    Returns
    -------
    pl.DataFrame
        Input panel with all FEATURE_COLS added.  Non-tradeable rows (and
        warm-up rows without sufficient history) have is_tradeable forced to
        False; all feature nulls/NaNs are filled with 0.0 (sentinel).
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
    if index_rets is None:
        return df.with_columns(pl.lit(1.0, dtype=pl.Float64).alias("beta_nifty_60d"))

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
    df = df.with_columns(
        pl.when(pl.col("_var") > 1e-10)
        .then(pl.col("_cov") / pl.col("_var"))
        .otherwise(pl.lit(1.0, dtype=pl.Float64))
        .alias("beta_nifty_60d")
    )
    return df.drop(
        ["index_return", "_i_mean", "_i_e_r2", "_xy", "_e_xy", "_e_x", "_cov", "_var"]
    )
