"""Unit tests for FIFO lot matching and Indian capital-gains tax.

The rates asserted are the post-Budget-2024 ones: 20% STCG (§111A), 12.5% LTCG
(§112A) with a ₹1,25,000 per-FY exemption, plus 4% health & education cess.
"""
from __future__ import annotations

from datetime import date

import pytest

from trader.env.tax import (
    CESS_RATE,
    LTCG_EXEMPTION_PER_FY,
    LTCG_RATE,
    STCG_RATE,
    FifoLotBook,
    Lot,
    TaxModel,
    add_months,
    financial_year,
    financial_year_label,
    is_long_term,
)

# ── Rate constants ────────────────────────────────────────────────────────────


def test_post_budget_2024_rates() -> None:
    assert STCG_RATE == 0.20
    assert LTCG_RATE == 0.125
    assert CESS_RATE == 0.04
    assert LTCG_EXEMPTION_PER_FY == 125_000.0


# ── Holding period ────────────────────────────────────────────────────────────


def test_exactly_twelve_months_is_still_short_term() -> None:
    """The statute says "more than twelve months", so the anniversary itself
    is short-term and the next day is long-term."""
    buy = date(2024, 3, 15)
    assert is_long_term(buy, date(2025, 3, 15)) is False
    assert is_long_term(buy, date(2025, 3, 16)) is True


def test_add_months_clamps_leap_day() -> None:
    assert add_months(date(2024, 2, 29), 12) == date(2025, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2024, 12, 31), 12) == date(2025, 12, 31)
    assert is_long_term(date(2024, 2, 29), date(2025, 2, 28)) is False
    assert is_long_term(date(2024, 2, 29), date(2025, 3, 1)) is True


# ── Financial year ────────────────────────────────────────────────────────────


def test_financial_year_runs_april_to_march() -> None:
    assert financial_year(date(2025, 4, 1)) == 2025
    assert financial_year(date(2026, 3, 31)) == 2025
    assert financial_year(date(2025, 3, 31)) == 2024
    assert financial_year(date(2025, 12, 31)) == 2025
    assert financial_year_label(2025) == "FY2025-26"
    assert financial_year_label(2099) == "FY2099-00"


# ── FIFO matching ─────────────────────────────────────────────────────────────


def test_eleven_month_hold_is_short_term_at_20_percent_plus_cess() -> None:
    book = FifoLotBook()
    book.buy("INFY", Lot(buy_date=date(2024, 5, 1), quantity=100, cost_basis_per_share=100.0))
    gain = book.sell("INFY", 100, date(2025, 4, 1), 150.0)

    assert gain.short_term == pytest.approx(5_000.0)
    assert gain.long_term == 0.0
    assert gain.proceeds == pytest.approx(15_000.0)
    assert gain.cost == pytest.approx(10_000.0)

    model = TaxModel()
    model.record(gain)
    liability = model.liability(financial_year(gain.sell_date))
    assert liability.stcg_tax == pytest.approx(1_000.0)  # 20% of 5,000
    assert liability.ltcg_tax == 0.0
    assert liability.cess == pytest.approx(40.0)  # 4% of 1,000
    assert liability.total == pytest.approx(1_040.0)


def test_thirteen_month_hold_is_long_term_with_exemption() -> None:
    book = FifoLotBook()
    book.buy("TCS", Lot(buy_date=date(2024, 1, 10), quantity=100, cost_basis_per_share=100.0))
    gain = book.sell("TCS", 100, date(2025, 2, 10), 2_100.0)

    assert gain.long_term == pytest.approx(200_000.0)
    assert gain.short_term == 0.0

    model = TaxModel()
    model.record(gain)
    liability = model.liability(2024)  # 10 Feb 2025 falls in FY2024-25
    assert liability.taxable_long_term == pytest.approx(75_000.0)  # 200k − 125k
    assert liability.ltcg_tax == pytest.approx(9_375.0)  # 12.5% of 75,000
    assert liability.cess == pytest.approx(375.0)
    assert liability.total == pytest.approx(9_750.0)


def test_sale_spanning_multiple_lots_splits_fifo() -> None:
    book = FifoLotBook()
    # Oldest lot is long-term by the sale date; the newer one is not.
    book.buy("HDFC", Lot(buy_date=date(2023, 1, 10), quantity=100, cost_basis_per_share=100.0))
    book.buy("HDFC", Lot(buy_date=date(2024, 12, 1), quantity=50, cost_basis_per_share=200.0))

    gain = book.sell("HDFC", 120, date(2025, 3, 1), 300.0)

    # 100 shares from the old lot: (300 − 100) × 100 = 20,000 long-term.
    assert gain.long_term == pytest.approx(20_000.0)
    # 20 shares from the new lot: (300 − 200) × 20 = 2,000 short-term.
    assert gain.short_term == pytest.approx(2_000.0)
    assert gain.total == pytest.approx(22_000.0)
    assert gain.cost == pytest.approx(100 * 100.0 + 20 * 200.0)

    # The partially consumed lot keeps its original buy date.
    remaining = book.open_lots("HDFC")
    assert len(remaining) == 1
    assert remaining[0].buy_date == date(2024, 12, 1)
    assert remaining[0].quantity == pytest.approx(30.0)
    assert book.open_quantity("HDFC") == pytest.approx(30.0)


def test_lots_consumed_oldest_first_even_at_worse_prices() -> None:
    book = FifoLotBook()
    book.buy("SBIN", Lot(buy_date=date(2024, 1, 1), quantity=10, cost_basis_per_share=500.0))
    book.buy("SBIN", Lot(buy_date=date(2024, 6, 1), quantity=10, cost_basis_per_share=100.0))
    # FIFO must use the ₹500 basis, not the cheaper (tax-favourable) one.
    gain = book.sell("SBIN", 10, date(2024, 9, 1), 400.0)
    assert gain.short_term == pytest.approx(-1_000.0)
    assert book.open_lots("SBIN")[0].cost_basis_per_share == pytest.approx(100.0)


def test_full_consumption_empties_the_book() -> None:
    book = FifoLotBook()
    book.buy("WIPRO", Lot(buy_date=date(2024, 1, 1), quantity=10, cost_basis_per_share=50.0))
    book.sell("WIPRO", 10, date(2024, 6, 1), 60.0)
    assert book.open_lots("WIPRO") == ()
    assert book.open_quantity("WIPRO") == 0.0


def test_selling_more_than_held_is_rejected() -> None:
    book = FifoLotBook()
    book.buy("ITC", Lot(buy_date=date(2024, 1, 1), quantity=10, cost_basis_per_share=50.0))
    with pytest.raises(ValueError, match="only"):
        book.sell("ITC", 11, date(2024, 6, 1), 60.0)
    with pytest.raises(ValueError, match="only"):
        book.sell("UNKNOWN", 1, date(2024, 6, 1), 60.0)


def test_non_positive_quantities_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        Lot(buy_date=date(2024, 1, 1), quantity=0, cost_basis_per_share=10.0)
    with pytest.raises(ValueError, match="non-negative"):
        Lot(buy_date=date(2024, 1, 1), quantity=1, cost_basis_per_share=-1.0)
    book = FifoLotBook()
    book.buy("ITC", Lot(buy_date=date(2024, 1, 1), quantity=10, cost_basis_per_share=50.0))
    with pytest.raises(ValueError, match="positive"):
        book.sell("ITC", 0, date(2024, 6, 1), 60.0)


def test_symbols_are_independent() -> None:
    book = FifoLotBook()
    book.buy("A", Lot(buy_date=date(2024, 1, 1), quantity=10, cost_basis_per_share=10.0))
    book.buy("B", Lot(buy_date=date(2020, 1, 1), quantity=10, cost_basis_per_share=10.0))
    gain = book.sell("A", 10, date(2024, 6, 1), 20.0)
    assert gain.short_term == pytest.approx(100.0)
    assert book.open_quantity("B") == 10.0


# ── Tax accumulation ──────────────────────────────────────────────────────────


def test_ltcg_below_exemption_is_untaxed() -> None:
    model = TaxModel()
    model.record_gains(date(2025, 6, 1), long_term=100_000.0)
    liability = model.liability(2025)
    assert liability.taxable_long_term == 0.0
    assert liability.ltcg_tax == 0.0
    assert liability.total == 0.0


def test_ltcg_exemption_applies_once_per_financial_year() -> None:
    """31 Mar and 1 Apr sit in different FYs, so each gets its own ₹1.25L."""
    same_year = TaxModel()
    same_year.record_gains(date(2025, 1, 15), long_term=100_000.0)
    same_year.record_gains(date(2025, 3, 31), long_term=100_000.0)
    # Both in FY2024-25: 200,000 − 125,000 = 75,000 taxable.
    assert same_year.financial_years == (2024,)
    assert same_year.liability(2024).taxable_long_term == pytest.approx(75_000.0)
    assert same_year.total_tax() == pytest.approx(9_375.0 * 1.04)

    split = TaxModel()
    split.record_gains(date(2025, 3, 31), long_term=100_000.0)
    split.record_gains(date(2025, 4, 1), long_term=100_000.0)
    # One day later crosses into FY2025-26, so both sit under the exemption.
    assert split.financial_years == (2024, 2025)
    assert split.total_tax() == 0.0


def test_short_term_and_long_term_taxed_separately_in_one_year() -> None:
    model = TaxModel()
    model.record_gains(date(2025, 6, 1), short_term=50_000.0, long_term=225_000.0)
    liability = model.liability(2025)
    assert liability.stcg_tax == pytest.approx(10_000.0)  # 20% of 50,000
    assert liability.ltcg_tax == pytest.approx(12_500.0)  # 12.5% of (225k − 125k)
    assert liability.cess == pytest.approx(0.04 * 22_500.0)
    assert liability.total == pytest.approx(22_500.0 * 1.04)


def test_losses_net_within_bucket_and_floor_at_zero_tax() -> None:
    model = TaxModel()
    model.record_gains(date(2025, 6, 1), short_term=100_000.0)
    model.record_gains(date(2025, 7, 1), short_term=-40_000.0)
    assert model.realised(2025) == (pytest.approx(60_000.0), 0.0)
    assert model.liability(2025).stcg_tax == pytest.approx(12_000.0)

    net_loss = TaxModel()
    net_loss.record_gains(date(2025, 6, 1), short_term=-10_000.0, long_term=-5_000.0)
    liability = net_loss.liability(2025)
    assert liability.short_term_gain == pytest.approx(-10_000.0)
    assert liability.total == 0.0


def test_long_term_loss_does_not_shelter_short_term_gain() -> None:
    """Documented simplification: no cross-head set-off is modelled."""
    model = TaxModel()
    model.record_gains(date(2025, 6, 1), short_term=100_000.0, long_term=-100_000.0)
    assert model.liability(2025).stcg_tax == pytest.approx(20_000.0)


def test_unseen_year_has_no_liability() -> None:
    model = TaxModel()
    assert model.realised(1999) == (0.0, 0.0)
    assert model.liability(1999).total == 0.0
    assert model.financial_years == ()
    assert model.total_tax() == 0.0


def test_total_tax_sums_across_years() -> None:
    model = TaxModel()
    model.record_gains(date(2024, 6, 1), short_term=10_000.0)
    model.record_gains(date(2025, 6, 1), short_term=20_000.0)
    assert model.financial_years == (2024, 2025)
    assert model.total_tax() == pytest.approx((2_000.0 + 4_000.0) * 1.04)
    assert [liability.financial_year for liability in model.liabilities()] == [2024, 2025]


def test_rates_are_overridable() -> None:
    """Rates change with every Budget; the model must not hard-code them."""
    model = TaxModel(stcg_rate=0.15, ltcg_rate=0.10, cess_rate=0.0, ltcg_exemption=0.0)
    model.record_gains(date(2025, 6, 1), short_term=100_000.0, long_term=100_000.0)
    assert model.liability(2025).total == pytest.approx(15_000.0 + 10_000.0)


def test_end_to_end_book_to_tax() -> None:
    book = FifoLotBook()
    model = TaxModel()
    book.buy("RELIANCE", Lot(date(2023, 4, 3), 200, 1_000.0))
    book.buy("RELIANCE", Lot(date(2025, 1, 6), 100, 1_200.0))

    model.record(book.sell("RELIANCE", 250, date(2025, 6, 2), 2_000.0))

    # 200 long-term @ +1,000 = 200,000; 50 short-term @ +800 = 40,000.
    short_term, long_term = model.realised(2025)
    assert long_term == pytest.approx(200_000.0)
    assert short_term == pytest.approx(40_000.0)
    liability = model.liability(2025)
    assert liability.total == pytest.approx((0.20 * 40_000.0 + 0.125 * 75_000.0) * 1.04)
    assert book.open_quantity("RELIANCE") == pytest.approx(50.0)
