# NSE fixtures

Small offline samples for `tests/unit/test_nse_flows.py`. Rows are real
records copied verbatim from files downloaded on 2026-09-05 (see the module
docstring of `trader.data.sources.nse_flows` for URL patterns and what was
verified), trimmed to a handful of symbols. Hand-constructed files are marked.

| File | Origin |
|---|---|
| `MTO_02012015.DAT` | real; `archives/equities/mto/MTO_02012015.DAT`, 8 of 1,536 records. Header line 3 carries settlement number/date (pre-2020s layout). |
| `MTO_04092026.DAT` | real; `archives/equities/mto/MTO_04092026.DAT`, 8 of 3,375 records. Header line 3 has no settlement fields (current layout). Includes post-rename symbols LTM, TMPV. |
| `MTO_unknown_layout.DAT` | **hand-constructed**: a 6-field record layout that has never been observed, to prove the parser fails loudly rather than guessing. |
| `sec_bhavdata_full_04092026.csv` | real; `products/content/sec_bhavdata_full_04092026.csv`, 8 of ~3,300 rows. STLTECH (BE series) carries `-` for DELIV_QTY/DELIV_PER exactly as NSE publishes it. |
| `bulk.csv` | real header + 3 real rows from `content/equities/bulk.csv`; the two LTM/GRAVITON rows are **hand-constructed** (a quoted client name with a comma, and a symbol that exercises the rename alias). |
| `block.csv` | real; `content/equities/block.csv`, complete (2 rows that day). |
| `fiidii_latest.json` | real; `api/fiidiiTradeReact` response, complete. |
| `flows_backfill.csv` | **hand-constructed** in the documented backfill schema (`FlowsSource.BACKFILL_COLUMNS`). |

## What the rename map is actually exercised on

`MTO_02012015.DAT` contains these symbols only: `20MICRONS`, `INFY`,
`RELIANCE`, `SBIN` (EQ/N1/N3), `TATAMOTORS`, `TCS`.

It contains **`TATAMOTORS` and no `LTIM`** — LTIMindtree did not exist in 2015,
so a 2015 file cannot carry that symbol. An earlier note claiming the fixture
spells both was wrong. Verify with:

```
grep -o "TATAMOTORS\|LTIM" tests/fixtures/nse/MTO_02012015.DAT | sort -u
```

`MTO_04092026.DAT` already carries the post-rename `LTM` and `TMPV`, so
`KITE_RENAMES` is exercised across this fixture pair on exactly **one** symbol:
`TATAMOTORS`. The conclusion the map is there for still holds — without it
every pre-rename year silently drops those names on a symbol-keyed join — but
the coverage here is one symbol, not two, and adding a pre-rename `LTIM` row to
a fixture would widen it.
