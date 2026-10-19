# S7 — Shorter stop re-entry ban (1–2 weeks) on the momentum book

**Pre-registered 2026-10-19 (evening; the commit time is authoritative)**,
before any run. The owner asked: "shorter ban -> maybe 1-2 weeks, do data
backed." No live book changes because of this document. If the criterion
passes, a NEW shadow book is proposed (Book 6). Books 1–5 and their rules are
untouched: the 21-session ban is part of every one of their
pre-registrations.

## Prior evidence

`audit/N1_STOP_VETO_HEADROOM.md:135-171`, on the r4 model's signal, K=20:
- In sample, a 5-step ban beat 21 by about 1.6 pp CAGR.
- On the 2025–26 holdout this reversed: 21 was best or tied, and the spread
  across 0–42 was about 0.9 pp.
- Separately, stopped names gained about +2% on average over the next 21
  sessions, and 54–55% recovered (`N1:191-195`). This is the case for letting
  them back in sooner.

That test was on a different signal (r4), which never beat random. This one is
on the momentum signal, the only one in the repo that ranked 1st of 21 on both
periods (`audit/P7_COMPOSITE_15.md`, info-only runs).

## Test (fixed; run once)

Signal `mom12_1_oos` / `mom12_1_holdout` (Book 3's 12-1 momentum), K=15,
monthly, band 0.010, horizon 20d, tax on, `data=bhav_v1`. Matched null
control with 20 random books (seeds 1–20) **per arm**. Three arms, identical
except the ban:

| arm | stop | `stop_cooldown_steps` |
|---|---|---|
| `vcd21` (control, = Books 1–5) | volstop 1.0×vol×√20 | 21 |
| `vcd10` (≈ 2 weeks) | same | 10 |
| `vcd5` (≈ 1 week) | same | 5 |

Periods: `oos_r4_pit_long` (2011-07 → 2024-07) and `holdout` (2025-04 →
2026-09).

## Criterion (fixed now)

A shorter ban **qualifies** only if, on **both** periods:
1. its cumulative log return is higher than `vcd21`'s, **and**
2. its max drawdown is no more than 2 percentage points worse than
   `vcd21`'s.

If both 5 and 10 qualify, the one with the higher 2011–24 cumulative log
return is chosen. This tie-break is fixed now; it is a selection between two,
and is recorded as such.

**If one qualifies:** propose Book 6, Book 3's rules with that ban. It is
pre-registered separately and goes live only on the owner's yes.
**If neither qualifies:** record it plainly; no Book 6; the 21-session ban
stays.

Ranks against each arm's own random books are reported for information only.

**Prediction, recorded before the run:** the differences will be small, under
1 pp CAGR, as in N1. The momentum signal is slow, so a stopped name rarely
re-enters the top 15 within 21 sessions, and the ban seldom binds. Expect
neither variant to win on both periods.

## Results

Run 2026-10-19 20:20–20:26 IST. Logs `logs/retrain/s7_{holdout,oos}_20261019.log`,
NAVs `audit/navs_s7/`. MLflow experiment `allocator`. Cumulative log return,
CAGR, MDD and rank are computed from the NAV parquets (scoring snippet in the
session; the numbers reproduce from the NAVs with the same formulas as
`scripts/p7_score.py`).

| period | arm | cum log | CAGR | MDD | rank / 21 | MLflow |
|---|---|---|---|---|---|---|
| 2011-09 → 2024-07 | **vcd21** | +2.1183 | +18.06% | −45.0% | 1 | `436e2f5a54124d03a960a7af74b3483b` |
| | vcd10 | +2.2404 | +19.20% | −46.6% | 1 | `f0ff41f44ec946dbb14767953ece602a` |
| | vcd5 | +2.2597 | +19.38% | −47.6% | 1 | `5190f77dc8554176b4ad9af5dc141148` |
| | equal-weight | +1.8375 | +15.49% | −51.2% | | `e5ebc6903825405384af666f744e8e7b` |
| 2025-06 → 2026-09 | **vcd21** | +0.1287 | +11.47% | −9.9% | 1 | `7faec08b2421450fa4ce9a6372319dbb` |
| | vcd10 | +0.1492 | +13.41% | −10.4% | 1 | `59ff5b0d9c4b4d7cb48e4fcdba83e57c` |
| | vcd5 | +0.1699 | +15.41% | −10.7% | 1 | `309084dacc1e4e5a8d537e373f582ffc` |
| | equal-weight | +0.0710 | +6.17% | −12.3% | | `74f8feef342c47c7b039b50810d1e52c` |

Reproducibility check: `vcd21` matches the P7 info-only momentum volstop run
exactly (+2.1183 and +0.1287), as it must.

**Criterion:**
- **vcd10 qualifies.** It is higher than vcd21 on both periods (+0.122 and
  +0.021 log). Its MDD is 1.6 pp and 0.5 pp worse, within the 2 pp limit.
- **vcd5 does not qualify.** It is higher on both periods, but its 2011–24
  MDD is 2.6 pp worse than vcd21's, over the 2 pp limit.
- **Chosen: 10 sessions.** The tie-break was not needed.

**Prediction scored: wrong.** The gain is +1.1 pp CAGR (2011–24) and
+1.9 pp (holdout), not under 1 pp. The ban binds more often on momentum than
expected.

Caveats:
- Three arms on one signal.
- The holdout is 1.2 years.
- Shorter bans buy return with somewhat deeper drawdowns. This is the
  trade-off the 2 pp limit was set to police.
- It says nothing about Books 1, 2, 4 or 5, which trade other signals and keep
  21 sessions.

**Next, per the criterion:** propose Book 6 (Book 3's rules with a 10-session
ban), pre-registered separately and live only on the owner's yes.
