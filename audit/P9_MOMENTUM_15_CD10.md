# P9 — Book 6: Book 3 with a 10-session stop re-entry ban

**Pre-registered 2026-10-19 (evening; the commit time is authoritative)**,
before the book's first trade. The owner said "Yes do it" after
`audit/S7_COOLDOWN_MOMENTUM.md`. That test found a 10-session ban beat the
standard 21-session ban on Book 3's signal on both 2011–24 and the 2025–26
holdout, with drawdown within the pre-set 2 pp limit. Nothing above
"Operations" is edited after the first fill. Books 1–5 are untouched and keep
21 sessions.

## What differs from Book 3 — exactly one thing

| | Book 3 (`audit/P6_MOMENTUM_15.md`) | **Book 6** |
|---|---|---|
| Re-entry ban after a stop-loss sale | 21 sessions | **10 sessions** |

Everything else is identical and shares Book 3's signal file
(`data/signal/momentum_12_1/predictions.parquet`, 12-1 momentum):
- K=15, monthly.
- Inverse-vol weights, 10% name cap, 25% sector cap.
- Band 0.010, turnover budget 0.30 (lifted at first funding).
- Whole shares plus top-up, ₹500 minimum trade.
- Volstop 1.0×daily vol×√20, clipped 5–30%.
- Zerodha delivery costs.

* Executor: `scripts/live_paper.py`, with the ban read from
  `TRADER_LIVE_COOLDOWN` (default 21, so Books 1–5 are unaffected).
* ₹1,00,000, `audit/paper/live_momentum10/`. Trade 09:21 / 09:25 / 09:35,
  marks every 15 min, close mark 15:40.
* **First funding Tue 2026-10-20**, signal dated 2026-10-19. First rebalance
  2026-11-02.

Book 6 starts one session after Book 3 (19 Oct), so the two books' paths
differ from day one by entry prices as well as by the ban. Their holdings
will be nearly identical until the first stop-loss sale.

## Reading

Descriptive, paired with Book 3. What it can show: whether the shorter ban
changes live returns, drawdown and turnover. The ban only matters after a
stop fires and the name is still in the top 15 within 10–21 sessions, which
S7 suggests happens a few times a year. **A few months of live data will
very likely show little or no difference, and that will not be evidence
either way.** Fixed read **2027-10-19**, alongside the other books.

Not permitted before the read:
- changing the ban, K or any shared rule;
- pausing or resetting after a loss;
- choosing between Books 3 and 6 on interim numbers.

## Operations

*(appended as it runs; not part of the pre-registration)*

## Results

*(live read at 2027-10-19)*
