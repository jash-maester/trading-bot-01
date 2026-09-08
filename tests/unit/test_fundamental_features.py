"""Tests for cross-sectional fundamental features.

The bugs worth catching here are all silent ones: a ratio that fabricates a
number instead of admitting it is unknown, a year-on-year comparison that lines
up against the wrong quarter when a filing is missing, and a value that appears
on the panel a day before the market could have seen it.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from trader.data.fundamental_features import (
    CHANGE_COLS,
    LEVEL_COLS,
    as_of_panel,
    attach_earnings_yield,
    build_quarterly_features,
)


def _figures(rows: list[dict[str, object]]) -> pl.DataFrame:
    """A minimal figures frame with every column the builder reads."""
    base = {
        "ticker": "AAA.NS",
        "revenue": 100.0,
        "net_profit": 10.0,
        "pbt": 14.0,
        "tax_expense": 4.0,
        "employee_cost": 20.0,
        "finance_costs": 5.0,
        "eps_basic": 2.0,
        "paid_up_equity": 500.0,
        "face_value": 5.0,
    }
    return pl.DataFrame([{**base, **r} for r in rows])


def test_levels_are_ratios_of_revenue() -> None:
    f = build_quarterly_features(
        _figures([{"period_to": date(2023, 3, 31), "visible_from": date(2023, 5, 10)}])
    )
    row = f.row(0, named=True)
    assert row["f_net_margin"] == pytest.approx(0.10)
    assert row["f_pbt_margin"] == pytest.approx(0.14)
    assert row["f_employee_cost_ratio"] == pytest.approx(0.20)
    assert row["f_finance_cost_ratio"] == pytest.approx(0.05)
    assert row["f_tax_rate"] == pytest.approx(4.0 / 14.0)
    # paid_up_equity / face_value, the INFY-verified share count route.
    assert row["f_shares"] == pytest.approx(100.0)


def test_zero_denominator_is_null_not_zero() -> None:
    """A ratio over zero revenue is UNKNOWN.

    Filling it with 0.0 would place a fabricated value into a cross-sectional
    rank, which is the same class of error as a dead feature channel reading a
    constant — invisible, and wrong in a direction nobody chose.
    """
    f = build_quarterly_features(
        _figures(
            [
                {
                    "period_to": date(2023, 3, 31),
                    "visible_from": date(2023, 5, 10),
                    "revenue": 0.0,
                    "pbt": 0.0,
                }
            ]
        )
    )
    row = f.row(0, named=True)
    for col in ("f_net_margin", "f_pbt_margin", "f_tax_rate"):
        assert row[col] is None, f"{col} fabricated a value over a zero denominator"


def test_yoy_uses_the_filing_four_quarters_back() -> None:
    quarters = [
        (date(2022, 3, 31), date(2022, 5, 10), 100.0),
        (date(2022, 6, 30), date(2022, 8, 10), 110.0),
        (date(2022, 9, 30), date(2022, 11, 10), 120.0),
        (date(2022, 12, 31), date(2023, 2, 10), 130.0),
        (date(2023, 3, 31), date(2023, 5, 10), 150.0),
    ]
    f = build_quarterly_features(
        _figures(
            [
                {"period_to": p, "visible_from": v, "revenue": r}
                for p, v, r in quarters
            ]
        )
    ).sort("period_to")
    last = f.row(-1, named=True)
    # 150 against the SAME quarter a year earlier (100), not the previous one.
    assert last["f_revenue_growth_yoy"] == pytest.approx(0.50)
    # The sequential surprise is against 130.
    assert last["f_revenue_surprise"] == pytest.approx(150.0 / 130.0 - 1.0)
    # The first four quarters have no year-ago counterpart at all.
    assert f["f_revenue_growth_yoy"].head(4).null_count() == 4


def test_negative_base_growth_keeps_its_sign() -> None:
    """Profit going from -50 to +50 is an improvement, and must read positive.

    Dividing by a raw negative base flips the sign, so a company recovering from
    a loss would rank as the worst in the cross-section. The builder divides by
    ``abs(base)`` for profit and EPS for exactly this reason.
    """
    rows = [
        {"period_to": date(2022, 3, 31), "visible_from": date(2022, 5, 10), "net_profit": -50.0},
        {"period_to": date(2022, 6, 30), "visible_from": date(2022, 8, 10)},
        {"period_to": date(2022, 9, 30), "visible_from": date(2022, 11, 10)},
        {"period_to": date(2022, 12, 31), "visible_from": date(2023, 2, 10)},
        {"period_to": date(2023, 3, 31), "visible_from": date(2023, 5, 10), "net_profit": 50.0},
    ]
    f = build_quarterly_features(_figures(rows)).sort("period_to")
    # (50 - -50) / |-50| = +2.0: a swing of twice the size of the starting loss.
    assert f.row(-1, named=True)["f_profit_growth_yoy"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("base", "now", "expected"),
    [
        (100.0, 150.0, 0.50),   # ordinary growth, unchanged by the |base| form
        (100.0, 50.0, -0.50),   # ordinary decline
        (-50.0, -10.0, 0.80),   # a shrinking loss is an IMPROVEMENT
        (-50.0, -60.0, -0.20),  # a widening loss is a decline
        (-50.0, 50.0, 2.00),    # loss to profit
        (50.0, -50.0, -2.00),   # profit to loss
    ],
)
def test_growth_is_monotone_through_zero(base: float, now: float, expected: float) -> None:
    """Growth must order companies by direction of travel, whatever the sign.

    The naive ``now/base - 1`` ranks a company whose loss halved BELOW one whose
    loss doubled, because dividing by a negative base flips the comparison. Every
    loss-making company in the universe would have been mis-ranked.
    """
    quarters = [
        (date(2022, 3, 31), date(2022, 5, 10), base),
        (date(2022, 6, 30), date(2022, 8, 10), 1.0),
        (date(2022, 9, 30), date(2022, 11, 10), 1.0),
        (date(2022, 12, 31), date(2023, 2, 10), 1.0),
        (date(2023, 3, 31), date(2023, 5, 10), now),
    ]
    f = build_quarterly_features(
        _figures(
            [
                {"period_to": p, "visible_from": v, "net_profit": x}
                for p, v, x in quarters
            ]
        )
    ).sort("period_to")
    assert f.row(-1, named=True)["f_profit_growth_yoy"] == pytest.approx(expected)


def test_as_of_panel_never_reveals_a_figure_early() -> None:
    """The whole point of ``visible_from``.

    The filing covers the quarter ending 2023-03-31 but is only announced on
    2023-05-10. Every panel date before that must stay NaN, including dates
    inside and just after the reporting quarter.
    """
    f = build_quarterly_features(
        _figures([{"period_to": date(2023, 3, 31), "visible_from": date(2023, 5, 10)}])
    )
    dates = [date(2023, 3, 31), date(2023, 5, 9), date(2023, 5, 10), date(2023, 6, 1)]
    grid = as_of_panel(f, list(dates), ["AAA.NS"], ("f_net_margin",))["f_net_margin"]
    assert np.isnan(grid[0, 0]), "figure visible on its own period end — lookahead"
    assert np.isnan(grid[1, 0]), "figure visible the day before announcement"
    assert grid[2, 0] == pytest.approx(0.10)
    assert grid[3, 0] == pytest.approx(0.10), "figure did not carry forward"


def test_as_of_panel_carries_forward_until_superseded() -> None:
    f = build_quarterly_features(
        _figures(
            [
                {"period_to": date(2023, 3, 31), "visible_from": date(2023, 5, 10)},
                {
                    "period_to": date(2023, 6, 30),
                    "visible_from": date(2023, 8, 10),
                    "net_profit": 30.0,
                },
            ]
        )
    )
    dates = [date(2023, 5, 10), date(2023, 7, 1), date(2023, 8, 10), date(2023, 9, 1)]
    grid = as_of_panel(f, list(dates), ["AAA.NS"], ("f_net_margin",))["f_net_margin"]
    assert grid[0, 0] == pytest.approx(0.10)
    assert grid[1, 0] == pytest.approx(0.10), "stale between filings is correct"
    assert grid[2, 0] == pytest.approx(0.30), "new filing did not supersede the old"
    assert grid[3, 0] == pytest.approx(0.30)


def test_as_of_panel_isolates_tickers() -> None:
    """One ticker's filing must never leak into another's column."""
    f = build_quarterly_features(
        pl.concat(
            [
                _figures([{"period_to": date(2023, 3, 31), "visible_from": date(2023, 5, 10)}]),
                _figures(
                    [
                        {
                            "ticker": "BBB.NS",
                            "period_to": date(2023, 3, 31),
                            "visible_from": date(2023, 6, 15),
                            "net_profit": 40.0,
                        }
                    ]
                ),
            ]
        )
    )
    dates = [date(2023, 5, 10), date(2023, 6, 15)]
    grid = as_of_panel(f, list(dates), ["AAA.NS", "BBB.NS"], ("f_net_margin",))["f_net_margin"]
    assert grid[0, 0] == pytest.approx(0.10)
    assert np.isnan(grid[0, 1]), "BBB.NS had not filed yet on 2023-05-10"
    assert grid[1, 1] == pytest.approx(0.40)


def test_earnings_yield_prices_on_the_visibility_date() -> None:
    """Earnings and price must be available on the same morning.

    Four quarters of 10 profit is 40 TTM; 100 shares at ₹20 is a ₹2,000 market
    cap, so the yield is 2%. The price used is the one on ``visible_from``.
    """
    rows = [
        (date(2022, 3, 31), date(2022, 5, 10)),
        (date(2022, 6, 30), date(2022, 8, 10)),
        (date(2022, 9, 30), date(2022, 11, 10)),
        (date(2022, 12, 31), date(2023, 2, 10)),
    ]
    f = build_quarterly_features(
        _figures([{"period_to": p, "visible_from": v} for p, v in rows])
    )
    panel = pl.DataFrame(
        {
            "ticker": ["AAA.NS"] * 2,
            "date": [date(2023, 2, 10), date(2023, 2, 13)],
            "close": [20.0, 999.0],  # the second price must not be used
        }
    )
    out = attach_earnings_yield(f, panel).sort("period_to")
    assert out.row(-1, named=True)["f_ttm_profit"] == pytest.approx(40.0)
    assert out.row(-1, named=True)["f_earnings_yield"] == pytest.approx(40.0 / 2000.0)


def test_earnings_yield_is_null_without_a_price() -> None:
    f = build_quarterly_features(
        _figures([{"period_to": date(2023, 3, 31), "visible_from": date(2023, 5, 10)}])
    )
    empty = pl.DataFrame(
        schema={"ticker": pl.String, "date": pl.Date, "close": pl.Float64}
    )
    out = attach_earnings_yield(f, empty)
    assert out["f_earnings_yield"].null_count() == out.height


def test_every_declared_column_is_actually_produced() -> None:
    """LEVEL_COLS/CHANGE_COLS are consumed by name downstream.

    A renamed feature that silently stops being built would reach the model as
    an all-NaN channel, which is the `beta_nifty_60d` failure in CLAUDE.md.
    """
    rows = [
        (date(2022, 3, 31), date(2022, 5, 10)),
        (date(2022, 6, 30), date(2022, 8, 10)),
        (date(2022, 9, 30), date(2022, 11, 10)),
        (date(2022, 12, 31), date(2023, 2, 10)),
        (date(2023, 3, 31), date(2023, 5, 10)),
    ]
    f = build_quarterly_features(
        _figures([{"period_to": p, "visible_from": v} for p, v in rows])
    )
    f = attach_earnings_yield(
        f,
        pl.DataFrame(
            {"ticker": ["AAA.NS"], "date": [date(2023, 5, 10)], "close": [20.0]}
        ),
    )
    for col in (*LEVEL_COLS, *CHANGE_COLS):
        assert col in f.columns, f"{col} is declared but never built"
    last = f.sort("period_to").row(-1, named=True)
    assert all(last[c] is not None for c in LEVEL_COLS), "a level came out null on clean input"


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_quarterly_features(pl.DataFrame({"ticker": ["AAA.NS"]}))
