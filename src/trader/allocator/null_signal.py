"""A random signal that is the SAME random signal tomorrow.

The null control scores each (date, ticker) with seeded noise. A draw over a
``[T, N]`` grid is stable only while the grid is: the forward paper loop
appends one session a day, so the first scored run re-rolled every random
book's entire history -- 80 of 85 books diverged from the previous snapshot by
up to 12.3% while the signal books reproduced to 1e-11
(``audit/P2``, restart 1).

Here each cell is a function of ``(seed, ticker, calendar day)``: a per-ticker
generator seeded by a stable hash of the seed and the ticker name, indexed by
days since a fixed anchor. Appending a session, adding a ticker, or moving the
panel start leaves every other cell identical.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime

import numpy as np

ANCHOR = date(2020, 1, 1)
SPAN_DAYS = 6000          # ~16 years of calendar days from the anchor
SIGMA = 0.02              # matches the historical control's scale


def _ordinal(d: date | datetime) -> int:
    dd = d.date() if isinstance(d, datetime) else d
    k = (dd - ANCHOR).days
    if not 0 <= k < SPAN_DAYS:
        raise ValueError(f"{dd} is outside the null signal's anchor span")
    return k


def stable_null_signal(
    dates: list[date | datetime],
    universe: list[str],
    seed: int,
    support: np.ndarray | None = None,
) -> np.ndarray:
    """``[len(dates), len(universe)]`` noise, cell-stable under grid growth.

    ``support`` masks the noise to the finite cells of a real signal so the
    control faces the same candidate set as the arm it controls.
    """
    k = np.fromiter((_ordinal(d) for d in dates), dtype=np.int64, count=len(dates))
    out = np.empty((len(dates), len(universe)), dtype=np.float64)
    for j, t in enumerate(universe):
        h = int.from_bytes(hashlib.sha256(f"{seed}:{t}".encode()).digest()[:8], "little")
        out[:, j] = np.random.default_rng(h).normal(0.0, SIGMA, SPAN_DAYS)[k]
    if support is not None:
        if support.shape != out.shape:
            raise ValueError(f"support {support.shape} does not match {out.shape}")
        out = np.where(np.isfinite(support), out, np.nan)
    return out
