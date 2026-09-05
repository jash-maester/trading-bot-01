"""Offline tests for the opt-in ext feature group.

The causality tests are the point of this file. For each ext feature we perturb
one input cell at ``t+1`` and assert row ``t`` is bit-identical, then perturb the
same cell at ``t-1`` and assert row ``t`` moves. A feature that fails the first
is leaking; one that fails the second is not reading its input at all — the
``beta_nifty_60d`` failure mode, where a dead channel sat unnoticed for months.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trader.data import features_ext
from trader.data.features import FEATURE_COLS, MAX_FEATURE_LOOKBACK_DAYS
from trader.data.features_ext import (
    EXT_FEATURE_COLS,
    EXT_FEATURE_LOOKBACK_DAYS,
    MAX_EXT_FEATURE_LOOKBACK_DAYS,
    combined_max_lookback_days,
    compute_ext_features,
    ext_feature_coverage,
    format_coverage_report,
    required_purge_months,
)

TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS"]
N_DAYS = 140
START = date(2024, 1, 1)
# Far enough in that every window (the longest is 61 sessions) is warm.
T_INDEX = 100


def _dates() -> list[date]:
    return [START + timedelta(days=i) for i in range(N_DAYS)]


def _panel() -> pl.DataFrame:
    dates = _dates()
    rows: list[dict[str, Any]] = []
    for di, day in enumerate(dates):
        for ti, ticker in enumerate(TICKERS):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "adj_close": 100.0 + ti * 10 + di * 0.1,
                    "volume": 1_000_000.0 + ti * 1000 + di,
                    "is_tradeable": True,
                    "log_return_1d": 0.001 * ((di + ti) % 7 - 3),
                }
            )
    return pl.DataFrame(rows)


def _delivery(dates: list[date] | None = None) -> pl.DataFrame:
    days = dates if dates is not None else _dates()
    rows: list[dict[str, Any]] = []
    for di, day in enumerate(days):
        for ti, ticker in enumerate(TICKERS):
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    # Deterministic, non-constant, and different per ticker.
                    "delivery_pct": 0.40 + 0.01 * ((di * 7 + ti * 3) % 21),
                }
            )
    return pl.DataFrame(rows)


def _flows() -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for di, day in enumerate(_dates()):
        rows.append(
            {
                "date": day,
                "fii_net_cr": 100.0 * ((di % 11) - 5),
                "dii_net_cr": -80.0 * ((di % 9) - 4),
            }
        )
    return pl.DataFrame(rows)


def _deals() -> pl.DataFrame:
    """A deal for AAA.NS every third session; nothing for the other tickers."""
    rows: list[dict[str, Any]] = []
    for di, day in enumerate(_dates()):
        if di % 3 == 0:
            rows.append(
                {
                    "date": day,
                    "ticker": "AAA.NS",
                    "net_value_inr": 5_000_000.0 * (1 if di % 2 == 0 else -1),
                }
            )
    return pl.DataFrame(rows)


def _row_at(frame: pl.DataFrame, index: int, ticker: str) -> dict[str, Any]:
    day = _dates()[index]
    return frame.filter((pl.col("date") == day) & (pl.col("ticker") == ticker)).row(
        0, named=True
    )


def _computed(**overrides: pl.DataFrame) -> pl.DataFrame:
    inputs: dict[str, pl.DataFrame] = {
        "delivery": _delivery(),
        "flows": _flows(),
        "deals": _deals(),
    }
    inputs.update(overrides)
    return compute_ext_features(
        _panel(),
        inputs["delivery"],
        inputs["flows"],
        inputs["deals"],
        observed_deal_dates=_dates(),
    )


# ── Contract: names, lookbacks, purge ────────────────────────────────────────


def test_ext_cols_are_disjoint_from_the_base_feature_set() -> None:
    assert set(EXT_FEATURE_COLS).isdisjoint(FEATURE_COLS)


def test_every_ext_feature_has_a_lookback_row() -> None:
    assert set(EXT_FEATURE_LOOKBACK_DAYS) == set(EXT_FEATURE_COLS), (
        "EXT_FEATURE_LOOKBACK_DAYS and EXT_FEATURE_COLS have diverged: "
        f"missing={set(EXT_FEATURE_COLS) - set(EXT_FEATURE_LOOKBACK_DAYS)}, "
        f"extra={set(EXT_FEATURE_LOOKBACK_DAYS) - set(EXT_FEATURE_COLS)}"
    )


def test_lookback_table_covers_every_window_literal_in_the_module() -> None:
    """Mirrors test_alignment_features' guard on features.py.

    A new 120-session window added without updating the table would silently
    invalidate the purge, so the source is scanned for window literals.
    """
    source = Path(features_ext.__file__).read_text()
    literals = [
        int(next(g for g in match if g))
        for match in re.findall(
            r"window_size=(\d+)|_(?:WIN|MIN): Final\[int\] = (\d+)", source
        )
    ]
    assert literals, "no window literals found — has the parsing convention changed?"
    # +1, not <=: every feature is shift(1) THEN a rolling window, so a window
    # of W reads W+1 rows back. The previous `max(literals) <=
    # MAX_EXT_FEATURE_LOOKBACK_DAYS` let a 61-day window pass against a 61-day
    # declared bound while the true lookback was 62 — one day of unpurged
    # overlap between train and val feature windows.
    assert max(literals) + 1 <= MAX_EXT_FEATURE_LOOKBACK_DAYS, (
        f"widest window {max(literals)} + 1 shift day exceeds the declared "
        f"MAX_EXT_FEATURE_LOOKBACK_DAYS={MAX_EXT_FEATURE_LOOKBACK_DAYS}"
    )

    # Every feature starts from a shift(1), so its lookback is window + 1.
    assert MAX_EXT_FEATURE_LOOKBACK_DAYS == max(EXT_FEATURE_LOOKBACK_DAYS.values())
    assert not re.search(r"\.shift\(\s*-", source), "a negative shift reads the future"


def test_ext_group_widens_the_purge_bound() -> None:
    # 61 > 60: enabling this group must widen the walk-forward purge guard.
    assert MAX_EXT_FEATURE_LOOKBACK_DAYS > MAX_FEATURE_LOOKBACK_DAYS
    assert combined_max_lookback_days(include_ext=False) == MAX_FEATURE_LOOKBACK_DAYS
    assert combined_max_lookback_days() == MAX_EXT_FEATURE_LOOKBACK_DAYS
    # 61 trading days / ~21 per month -> 3 months, the current default.
    assert required_purge_months() == 3
    assert required_purge_months(include_ext=False) == 3


# ── Contract: additive, non-destructive ──────────────────────────────────────


def test_existing_columns_are_untouched_and_row_order_preserved() -> None:
    panel = _panel()
    out = _computed()
    assert out.columns == [*panel.columns, *EXT_FEATURE_COLS]
    assert out.select(panel.columns).equals(panel)


def test_all_inputs_none_yields_all_null_columns() -> None:
    out = compute_ext_features(_panel())
    for name in EXT_FEATURE_COLS:
        assert out[name].null_count() == out.height, name
    coverage = ext_feature_coverage(out)
    assert coverage["coverage"].to_list() == [0.0] * len(EXT_FEATURE_COLS)


def test_duplicate_input_keys_are_refused() -> None:
    doubled = pl.concat([_delivery(), _delivery()])
    with pytest.raises(ValueError, match="duplicate"):
        compute_ext_features(_panel(), doubled)


def test_panel_carrying_an_ext_column_is_refused() -> None:
    panel = _panel().with_columns(pl.lit(0.0).alias("fii_net_5d"))
    with pytest.raises(ValueError, match="already carries"):
        compute_ext_features(panel, _delivery(), _flows(), _deals())


# ── Causality, per feature ───────────────────────────────────────────────────


def _perturb_delivery(index: int) -> pl.DataFrame:
    day = _dates()[index]
    return _delivery().with_columns(
        pl.when((pl.col("date") == day) & (pl.col("ticker") == "AAA.NS"))
        .then(pl.col("delivery_pct") * 0.5 + 0.05)
        .otherwise(pl.col("delivery_pct"))
        .alias("delivery_pct")
    )


def _perturb_flows(index: int) -> pl.DataFrame:
    day = _dates()[index]
    return _flows().with_columns(
        pl.when(pl.col("date") == day)
        .then(pl.col("fii_net_cr") + 9999.0)
        .otherwise(pl.col("fii_net_cr"))
        .alias("fii_net_cr"),
        pl.when(pl.col("date") == day)
        .then(pl.col("dii_net_cr") - 7777.0)
        .otherwise(pl.col("dii_net_cr"))
        .alias("dii_net_cr"),
    )


def _perturb_deals(index: int) -> pl.DataFrame:
    """Add a deal for AAA.NS on that session, or resize the one already there."""
    day = _dates()[index]
    base = _deals()
    if base.filter(pl.col("date") == day).height:
        return base.with_columns(
            pl.when(pl.col("date") == day)
            .then(pl.col("net_value_inr") * 3.0 + 1.0)
            .otherwise(pl.col("net_value_inr"))
            .alias("net_value_inr")
        )
    extra = pl.DataFrame(
        {"date": [day], "ticker": ["AAA.NS"], "net_value_inr": [12_345_678.0]}
    )
    return pl.concat([base, extra]).sort(["date", "ticker"])


# Index offsets chosen so the perturbed session has no baseline deal (i % 3 != 0)
# in both directions: T_INDEX = 100, so t-1 = 99 (99 % 3 == 0) is unusable for
# deals — use t-2 for the "past changes t" half of the deal case and t+1 for the
# future half.
_CASES = [
    ("delivery_pct_20d", _perturb_delivery, T_INDEX - 1),
    ("delivery_pct_z_60d", _perturb_delivery, T_INDEX - 1),
    ("fii_net_5d", _perturb_flows, T_INDEX - 1),
    ("dii_net_5d", _perturb_flows, T_INDEX - 1),
    ("bulk_deal_flag_5d", _perturb_deals, T_INDEX - 2),
    ("bulk_deal_net_20d", _perturb_deals, T_INDEX - 2),
]


def _recompute(perturb: Any, index: int) -> pl.DataFrame:
    if perturb is _perturb_delivery:
        return _computed(delivery=perturb(index))
    if perturb is _perturb_flows:
        return _computed(flows=perturb(index))
    return _computed(deals=perturb(index))


@pytest.mark.parametrize(("feature", "perturb", "past_index"), _CASES)
def test_future_perturbation_leaves_row_t_unchanged(
    feature: str, perturb: Any, past_index: int
) -> None:
    base = _row_at(_computed(), T_INDEX, "AAA.NS")[feature]
    assert base is not None, f"{feature} is null at t — the test proves nothing"
    for future in (T_INDEX, T_INDEX + 1, T_INDEX + 5):
        moved = _row_at(_recompute(perturb, future), T_INDEX, "AAA.NS")[feature]
        assert moved == pytest.approx(base), (
            f"{feature} at t changed when input at index {future} moved — "
            "the feature is reading day t or later"
        )


@pytest.mark.parametrize(("feature", "perturb", "past_index"), _CASES)
def test_past_perturbation_moves_row_t(feature: str, perturb: Any, past_index: int) -> None:
    base = _row_at(_computed(), T_INDEX, "AAA.NS")[feature]
    moved = _row_at(_recompute(perturb, past_index), T_INDEX, "AAA.NS")[feature]
    assert base is not None and moved is not None
    assert moved != pytest.approx(base), (
        f"{feature} at t did not move when its t-1 input changed — the channel is dead"
    )


def test_a_market_level_feature_is_broadcast_to_every_ticker() -> None:
    out = _computed()
    values = {t: _row_at(out, T_INDEX, t)["fii_net_5d"] for t in TICKERS}
    assert len({round(v, 9) for v in values.values()}) == 1


# ── Nulls, never zeros ───────────────────────────────────────────────────────


def test_missing_delivery_day_is_null_not_zero() -> None:
    """Drop a run of delivery days; the affected rows must be null, not 0."""
    days = _dates()
    dropped = set(days[T_INDEX - 8 : T_INDEX])
    partial = _delivery().filter(~pl.col("date").is_in(list(dropped)))
    out = _computed(delivery=partial)

    # At t the t-1 observation itself is gone, so both features are null.
    row = _row_at(out, T_INDEX, "AAA.NS")
    assert row["delivery_pct_20d"] is None
    assert row["delivery_pct_z_60d"] is None

    # One session later t-1 is real again: the 20-session mean still has only 12
    # of 20 real values (< 15) so it stays null, while the 60-session z-score has
    # 52 of 60 (>= 40) and is computed.
    later = _row_at(out, T_INDEX + 1, "AAA.NS")
    assert later["delivery_pct_20d"] is None
    assert later["delivery_pct_z_60d"] is not None

    # And the value must never have been fabricated as 0.0 anywhere.
    assert out.filter(pl.col("delivery_pct_20d") == 0.0).height == 0


def test_an_uncaptured_deal_date_is_null_while_a_captured_quiet_day_is_zero() -> None:
    panel = _panel()
    days = _dates()
    # Capture every date except the five sessions ending t-1.
    observed = [d for i, d in enumerate(days) if not (T_INDEX - 5 <= i <= T_INDEX - 1)]
    out_gap = compute_ext_features(
        panel, _delivery(), _flows(), _deals(), observed_deal_dates=observed
    )
    assert _row_at(out_gap, T_INDEX, "AAA.NS")["bulk_deal_flag_5d"] is None

    # BBB.NS never has a deal, but every date is captured: that is a real 0.0.
    out_full = _computed()
    assert _row_at(out_full, T_INDEX, "BBB.NS")["bulk_deal_flag_5d"] == pytest.approx(0.0)
    assert _row_at(out_full, T_INDEX, "BBB.NS")["bulk_deal_net_20d"] == pytest.approx(0.0)


def test_deal_flag_is_the_fraction_of_the_last_five_sessions_with_a_deal() -> None:
    out = _computed()
    flag = _row_at(out, T_INDEX, "AAA.NS")["bulk_deal_flag_5d"]
    # Deals land on every third session, so sessions 95..99 contain 96 and 99.
    assert flag == pytest.approx(2 / 5)


def test_degenerate_delivery_history_gives_a_null_z_score_not_zero() -> None:
    flat = _delivery().with_columns(pl.lit(0.5).alias("delivery_pct"))
    out = _computed(delivery=flat)
    row = _row_at(out, T_INDEX, "AAA.NS")
    assert row["delivery_pct_20d"] == pytest.approx(0.5)
    assert row["delivery_pct_z_60d"] is None


def test_bulk_deal_net_is_null_without_price_and_volume() -> None:
    panel = _panel().drop("volume")
    out = compute_ext_features(
        panel, _delivery(), _flows(), _deals(), observed_deal_dates=_dates()
    )
    assert out["bulk_deal_net_20d"].null_count() == out.height
    # The flag does not need price data and must still be populated.
    assert out["bulk_deal_flag_5d"].null_count() < out.height


# ── Coverage arithmetic ──────────────────────────────────────────────────────


def test_coverage_counts_are_exact() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 4,
            "ticker": TICKERS + ["DDD.NS"],
            "is_tradeable": [True, True, True, False],
            "fii_net_5d": [1.0, None, 3.0, 9.0],
            "dii_net_5d": [None, None, None, None],
        }
    )
    coverage = ext_feature_coverage(frame, features=["fii_net_5d", "dii_net_5d"])
    fii = coverage.filter(pl.col("feature") == "fii_net_5d").row(0, named=True)
    # is_tradeable filters the fourth row out of the denominator entirely.
    assert fii["n_cells"] == 3
    assert fii["n_present"] == 2
    assert fii["coverage"] == pytest.approx(2 / 3)
    assert fii["mean"] == pytest.approx(2.0)
    assert fii["dead"] is False

    dii = coverage.filter(pl.col("feature") == "dii_net_5d").row(0, named=True)
    assert dii["n_present"] == 0
    assert dii["coverage"] == pytest.approx(0.0)
    assert dii["mean"] is None
    # `dead` now means "unusable to train on", which an all-null channel is.
    # The distinction the previous assertion was protecting — empty is not the
    # same claim as constant — is carried by `status`, which is strictly more
    # precise than the boolean was.
    assert dii["dead"] is True
    assert dii["status"] == "EMPTY (all null/NaN)"
    assert fii["status"] == "ok"


def test_coverage_does_not_report_an_all_nan_channel_as_healthy() -> None:
    """The beta_nifty_60d regression.

    `is_not_null()` counts a NaN as present and the std of an all-NaN column is
    NaN — neither None nor below the floor — so a fully NaN-poisoned channel
    reported coverage 1.0000 and status "ok" from the very function that exists
    to catch a dead channel.
    """
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 3,
            "ticker": TICKERS,
            "is_tradeable": [True, True, True],
            "fii_net_5d": [float("nan")] * 3,
        }
    )
    row = ext_feature_coverage(frame, features=["fii_net_5d"]).row(0, named=True)
    assert row["n_present"] == 0, "NaN must not count as present"
    assert row["n_missing"] == 3
    assert row["coverage"] == pytest.approx(0.0), "an all-NaN channel is not 100% covered"
    assert row["dead"] is True
    assert row["status"] != "ok"


def test_coverage_flags_a_partially_nan_constant_channel_as_dead() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 3,
            "ticker": TICKERS,
            "is_tradeable": [True, True, True],
            "fii_net_5d": [1.0, float("nan"), 1.0],
        }
    )
    row = ext_feature_coverage(frame, features=["fii_net_5d"]).row(0, named=True)
    assert row["n_present"] == 2
    assert row["n_missing"] == 1
    assert row["dead"] is True, "constant across its finite values is dead"


def test_coverage_flags_a_constant_channel_as_dead() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 3,
            "ticker": TICKERS,
            "is_tradeable": [True] * 3,
            "fii_net_5d": [1.0, 1.0, 1.0],
        }
    )
    coverage = ext_feature_coverage(frame, features=["fii_net_5d"])
    assert coverage["dead"].to_list() == [True]
    assert coverage["coverage"].to_list() == [1.0]
    assert "DEAD" in format_coverage_report(coverage)


def test_coverage_without_the_tradeable_filter_uses_every_row() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 4,
            "ticker": TICKERS + ["DDD.NS"],
            "is_tradeable": [True, True, True, False],
            "fii_net_5d": [1.0, None, 3.0, 9.0],
        }
    )
    coverage = ext_feature_coverage(
        frame, features=["fii_net_5d"], tradeable_only=False
    )
    assert coverage["n_cells"].to_list() == [4]
    assert coverage["n_present"].to_list() == [3]


def test_coverage_on_a_real_computed_panel_is_high_but_not_fabricated() -> None:
    out = _computed()
    coverage = ext_feature_coverage(out)
    by_name = {r["feature"]: r for r in coverage.iter_rows(named=True)}
    assert set(by_name) == set(EXT_FEATURE_COLS)
    for name, row in by_name.items():
        # Warm-up rows are null, so nothing can be 100% covered on a 140-day panel.
        assert 0.0 < row["coverage"] < 1.0, name
        assert row["dead"] is False, name
    # The longest-lookback feature must be the least covered.
    assert (
        by_name["delivery_pct_z_60d"]["coverage"] < by_name["fii_net_5d"]["coverage"]
    )
    assert "coverage" in format_coverage_report(coverage)


def test_coverage_rejects_an_unknown_column() -> None:
    with pytest.raises(ValueError, match="not a column"):
        ext_feature_coverage(_computed(), features=["no_such_feature"])
