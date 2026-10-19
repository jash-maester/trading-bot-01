# P8 — Book 5: rising delivery %, within sector, 15 names, monthly

**Pre-registered 2026-10-19 (~10:30 IST, see the commit time for the exact
moment), before the signal was computed on any period and before any
backtest or live number exists.** The owner asked on 19 Oct, after a friend
suggested that buy/sell volume carries predictive information: "maybe
instantiate a separate experiment with a different book ... test multiple
strategies on paper trail and let it run for say couple of months." Nothing
above "Operations" is edited after the first result. P2, P3, P6, P7 and their
books are not touched.

## Why this signal

NSE publishes, per stock per day, the share of traded quantity that was taken
into delivery (`DELIV_PER`), as opposed to bought and sold within the day.
The common Indian-market reading is that **rising** delivery is
accumulation: holders are buying to keep. The *level* differs structurally
across stocks (index heavyweights sit lower because of derivative-linked
churn; `src/trader/data/features_ext.py:41-47`), so the signal is the change
against the stock's own history, not the level. The repo has had this data
since 2020-07 (`data/ext/delivery.parquet`) and the features coded, but they
were never put into a model or tested (checked 19 Oct). Evidence quality for
delivery % as a return predictor: weak or UNVERIFIED (14_weekend_scoping_notes.md
§B2). This is a test, not a belief.

## Signal (fixed; `scripts/delivery_signal.py`)

For ticker i on date d, using only EQ-series rows (BE is trade-for-trade, so
its delivery is 100% by rule and NSE publishes it blank):

    rise = mean(delivery_pct over the 20 sessions ending d)
         − mean(delivery_pct over the 120 sessions ending d)

* Needs at least 16 non-null of the 20 and 100 non-null of the 120.
* Delivery for day d is published after d's close, the same as d's close
  price that the other books' signals use. The signal dated d trades at d+1,
  as every book does.
* z-scored across tradeable names on d **within sector** when the sector has
  ≥ 5 such names, otherwise across all names (same rule as P7).
* **Higher rise ranks higher.** Direction fixed now; not to be flipped after a
  result.

## Book rules (fixed)

Identical to Books 3 and 4 except the signal:
- K = 15, monthly (first session of the month).
- Inverse-vol weights, 10% name cap, 25% sector cap.
- Band 0.010, turnover budget 0.30 (lifted for the first funding).
- Whole shares plus top-up, ₹500 minimum trade.
- Volstop (1.0×daily vol×√20, clipped 5–30%), 21-session cooldown.
- Zerodha delivery costs.

Live: ₹1,00,000, `audit/paper/live_delivery/`. Trade 09:20 / 09:24 / 09:34,
marks every 15 min, close mark 15:39. The signal is rebuilt nightly after the
delivery data update. **First funding Tue 2026-10-20**, signal dated
2026-10-19; first rebalance 2026-11-02.

## Historical test (fixed; run once)

`scripts/run_allocator.py` with the matched null control (20 random books,
seeds 1–20, the same machinery), `data=bhav_v1`, K=[15], band 0.010, monthly,
horizon 20d, tax on, default risk arms. The verdict arm is volstop. Delivery
data starts 2020-07-01, so the periods are:
- `oos_deliv`: the `oos_r4_pit_long` panel restricted to dates ≥ 2020-10-15,
  so that NAVs start once the 120-session window is full (about 2021-01). It
  ends 2024-07-02 (about 3.5 years).
- `holdout`: as P7.

**Criterion, volstop arm, each period:** rank **1st of 21** against its own 20
random books **and** above equal-weight. Both periods are short, so a pass is
weak evidence for and a fail weak evidence against.

**Whatever the result, the book goes live on 20 Oct.** A failed historical
test is recorded plainly, and the live book then runs as a descriptive
comparison. Nothing is re-tuned.

Known limits:
- Delisting bias is unrecoverable (`audit/P4_survivorship.md:15`).
- The delivery fetcher overwrites its output for the range requested (found
  19 Oct). Nightly updates go through `scripts/delivery_update.py`, which
  merges.
- This is one signal, with no multiple-testing correction.

## Reading the live book

Descriptive, alongside Books 1–4. The owner's horizon is "a couple of
months"; the fixed read is **2027-10-19** (12 months), with an interim
descriptive look allowed at any time and no decisions taken on it.

Not permitted before the read:
- changing the windows, direction, K or any shared rule;
- pausing or resetting after a loss.

## Operations

*(appended as it runs; not part of the pre-registration)*

## Results

*(historical test appended when run; live read at 2027-10-19)*
