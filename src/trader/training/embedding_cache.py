"""Frozen-encoder embedding cache — the 47× from ``audit/A2_compute.md``.

Once the TCN encoder is frozen it is a pure function of the panel: the
embedding of stock *i* on date *t* depends only on the feature window ending at
*t-1* and on weights that never change.  A2 measured the encoder at 97.9% of
the arithmetic in a PPO update, so any RL loop that re-runs it is paying ~47×
for a lookup.  This module runs the encoder **once** over every date and stores
the result in the pinned signal-artefact layout::

    <out_dir>/embeddings.npy        float16 [T, N, D]   memory-mappable
    <out_dir>/index.json            {"dates": [ISO...], "tickers": [...],
                                     "embed_dim": D, "feature_cols": [...],
                                     "encoder_state_sha256": "...", ...}
    <out_dir>/cache_predictions.parquet
                                    date, ticker, r_hat_5d, r_hat_20d (f64),
                                    only when a ``predict_fn`` is supplied.
                                    DELIBERATELY not "predictions.parquet":
                                    that name is pinned to R4's OOS-only,
                                    tradeable-masked artefact, and this is the
                                    full date x ticker cross-product.

``index.json`` uses the pinned cross-stream field names ``feature_cols`` and
``encoder_state_sha256``.  The pre-rename spellings (``feature_columns``,
``encoder_sha256``) are written as aliases and accepted on read, so a cache
written by either side of the rename still opens.

Safety
------
The cache is keyed by a SHA-256 over the encoder's ``state_dict`` — every
tensor's name, shape and float32 bytes.  :class:`EmbeddingCache` refuses to open
a directory whose recorded hash differs from the encoder the caller says it is
using, because a stale cache is indistinguishable from a live one by shape
alone and would silently feed the wrong representation into every downstream
run.

Dates in the cache are those with a *full* lookback window — the first
``lookback`` dates of the panel are dropped, exactly as ``PanelTradingEnv``
drops them by starting at ``day_idx >= lookback``.  ``embeddings[i]`` is the
encoder applied to features ``[dates[i]-L, dates[i])``, i.e. the window ending
at the previous close, so it is what the env's ``obs["features"]`` would have
produced on ``dates[i]``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from trader.models.signal import state_dict_sha256

INDEX_FILE = "index.json"
EMBEDDINGS_FILE = "embeddings.npy"
# NOT "predictions.parquet".  That filename is PINNED to the R4 signal-artefact
# contract — "one row per (date, tradeable ticker); OOS predictions only", f64 —
# and R4's `supervised.write_artefacts` is its sole writer.  What this module can
# produce is a different thing: the full cached-date x ticker cross-product, with
# no tradeability mask and no OOS restriction, from whatever `predict_fn` it was
# handed.  Writing that to the pinned path made it indistinguishable from R4's
# artefact to any downstream reader (`scripts/run_allocator.py`,
# `scripts/train_allocator_rl.py`), which is how a diagnostic dump becomes a
# backtest input.  Different content, different filename.
PREDICTIONS_FILE = "cache_predictions.parquet"


class EncoderHashMismatch(RuntimeError):
    """The cache on disk was written by a different encoder."""


def encoder_state_sha256(module: nn.Module) -> str:
    """SHA-256 over every tensor in ``module.state_dict()`` (name, shape, float32 bytes).

    Delegates to :func:`trader.models.signal.state_dict_sha256` on purpose.
    There must be exactly **one** definition of "the encoder hash" in the repo:
    R4's signal job stamps ``index.json`` with
    :meth:`SignalModel.encoder_state_sha256`, and this module refuses to load a
    cache whose stamp disagrees.  Two hash functions that differ in, say,
    whether they mix in ``dtype`` would make every cross-stream cache look
    stale — a refusal indistinguishable from a genuinely wrong encoder.

    Order is by sorted key so the hash is independent of registration order.
    Buffers (e.g. ``FeatureNormalizer.mean/std``) are included — a change in the
    normalisation stats changes the embedding just as a weight change does.
    """
    return state_dict_sha256(module.state_dict())


def _to_date(d: Any) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def stack_panel_features(
    panel: pl.DataFrame,
    tickers: list[str],
    feature_columns: list[str],
) -> tuple[np.ndarray, list[date]]:
    """Pivot a long panel into a dense ``[T, N, F]`` float32 tensor plus its dates.

    Missing (date, ticker) rows are zero-filled, matching ``PanelTradingEnv``'s
    ``_build_ticker_arrays`` convention so the cached embedding equals what the
    env would have fed the encoder.
    """
    dates = sorted(_to_date(d) for d in panel["date"].unique().to_list())
    date_index = {d: i for i, d in enumerate(dates)}
    ticker_index = {t: i for i, t in enumerate(tickers)}
    T, N, F = len(dates), len(tickers), len(feature_columns)
    out = np.zeros((T, N, F), dtype=np.float32)

    sub = panel.filter(pl.col("ticker").is_in(tickers))
    row_t = np.array([date_index[_to_date(d)] for d in sub["date"].to_list()], dtype=np.int64)
    row_n = np.array([ticker_index[t] for t in sub["ticker"].to_list()], dtype=np.int64)
    for fi, col in enumerate(feature_columns):
        if col not in sub.columns:
            raise KeyError(f"feature column {col!r} not in panel")
        vals = sub[col].cast(pl.Float32).fill_null(0.0).to_numpy()
        out[row_t, row_n, fi] = vals
    return out, dates


@torch.no_grad()
def write_cache(
    encoder: nn.Module,
    panel: pl.DataFrame | Path,
    tickers: list[str],
    out_dir: Path,
    *,
    feature_columns: list[str],
    lookback: int,
    batch_days: int = 32,
    device: torch.device | None = None,
    predict_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Run a **frozen** encoder over every full-window date and write the cache.

    Parameters
    ----------
    encoder:
        Any module mapping ``[B, N, L, F] -> [B, N, D]`` (``TCNEncoder``).  It is
        put in ``eval()`` and its parameters are left untouched; the caller is
        responsible for it being the frozen, R4-trained encoder.
    panel:
        Long-format feature panel or a parquet path.
    tickers:
        Column order of the ``N`` axis.  Recorded in ``index.json``.
    feature_columns, lookback:
        Must match what the encoder was trained on — both are recorded.
    predict_fn:
        Optional ``[B, N, D] -> [B, N, 2]`` (5d, 20d standardised forward
        returns).  When given, ``cache_predictions.parquet`` is written too:
        the full cached-date x ticker cross-product with **no** tradeability
        mask and **no** OOS restriction.  It is a diagnostic dump, not the
        pinned R4 signal artefact, and the filename says so.
    """
    if isinstance(panel, Path):
        panel = pl.read_parquet(panel)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device or torch.device("cpu")

    was_training = encoder.training
    encoder.eval()
    encoder.to(dev)
    try:
        feats, dates = stack_panel_features(panel, tickers, feature_columns)
        T, N, _ = feats.shape
        if T <= lookback:
            raise ValueError(f"panel has {T} dates; need more than lookback={lookback}")
        valid_dates = dates[lookback:]
        T_out = len(valid_dates)

        embed_dim: int | None = None
        emb_out: np.ndarray | None = None
        preds: list[np.ndarray] = []

        for start in range(0, T_out, batch_days):
            idx = list(range(start, min(start + batch_days, T_out)))
            # window for cache row i ends at panel date lookback+i (exclusive)
            windows = np.stack(
                [feats[lookback + i - lookback : lookback + i] for i in idx], axis=0
            )                                                     # [B, L, N, F]
            x = torch.from_numpy(windows).permute(0, 2, 1, 3).contiguous().to(dev)  # [B, N, L, F]
            z = encoder(x)                                        # [B, N, D]
            if emb_out is None:
                embed_dim = int(z.shape[-1])
                emb_out = np.zeros((T_out, N, embed_dim), dtype=np.float16)
            emb_out[idx] = z.float().cpu().numpy().astype(np.float16)
            if predict_fn is not None:
                preds.append(predict_fn(z).float().cpu().numpy())    # [B, N, 2]
    finally:
        encoder.train(was_training)

    assert emb_out is not None and embed_dim is not None
    np.save(out_dir / EMBEDDINGS_FILE, emb_out)

    sha = encoder_state_sha256(encoder)
    meta: dict[str, Any] = {
        # Pinned cross-stream field names (R4 signal artefact contract).
        "dates": [d.isoformat() for d in valid_dates],
        "tickers": list(tickers),
        "embed_dim": embed_dim,
        "feature_cols": list(feature_columns),
        "encoder_state_sha256": sha,
        # Aliases for the pre-rename spellings, so a reader on either side of
        # the rename opens the same directory.  Kept in sync by construction.
        "feature_columns": list(feature_columns),
        "encoder_sha256": sha,
        # Provenance / shape, not part of the pinned contract.
        "lookback": int(lookback),
        "encoder_class": type(encoder).__name__,
        "dtype": "float16",
        "n_dates": T_out,
        "n_tickers": N,
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / INDEX_FILE).write_text(json.dumps(meta, indent=2))

    if predict_fn is not None and preds:
        p = np.concatenate(preds, axis=0)                          # [T_out, N, 2]
        frame = pl.DataFrame(
            {
                "date": [d for d in valid_dates for _ in tickers],
                "ticker": [t for _ in valid_dates for t in tickers],
                # f64: the pinned schema's dtype, and this frame is compared
                # against R4's in the seam tests.
                "r_hat_5d": p[:, :, 0].reshape(-1).astype(np.float64),
                "r_hat_20d": p[:, :, 1].reshape(-1).astype(np.float64),
            }
        )
        frame.write_parquet(out_dir / PREDICTIONS_FILE)
    return out_dir


class EmbeddingCache:
    """Memory-mapped reader for a directory written by :func:`write_cache`.

    Parameters
    ----------
    cache_dir:
        Directory holding ``embeddings.npy`` and ``index.json``.
    encoder / expected_sha256:
        The encoder this cache is expected to represent.  Exactly one may be
        given; if either is, the recorded hash must match or
        :class:`EncoderHashMismatch` is raised.  Passing neither opens the cache
        unchecked — only do that for inspection, never for training.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        encoder: nn.Module | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / INDEX_FILE).read_text())
        if encoder is not None and expected_sha256 is not None:
            raise ValueError("pass encoder or expected_sha256, not both")
        if encoder is not None:
            expected_sha256 = encoder_state_sha256(encoder)
        # Pinned name first, pre-rename alias second.  A cache with neither is
        # unverifiable, which is a refusal, not a default: an unchecked cache is
        # exactly the silent-wrong-representation failure this class exists for.
        recorded = meta.get("encoder_state_sha256") or meta.get("encoder_sha256")
        if recorded is None:
            raise EncoderHashMismatch(
                f"cache {self.dir}/{INDEX_FILE} records no encoder hash "
                "(expected 'encoder_state_sha256'); rebuild it with write_cache()"
            )
        if expected_sha256 is not None and recorded != expected_sha256:
            raise EncoderHashMismatch(
                f"cache {self.dir} was written by encoder "
                f"{recorded[:12]}…, requested "
                f"{expected_sha256[:12]}…; rebuild the cache with write_cache()"
            )
        self.meta = meta
        self.dates: list[date] = [date.fromisoformat(d) for d in meta["dates"]]
        self.tickers: list[str] = list(meta["tickers"])
        self.embed_dim: int = int(meta["embed_dim"])
        self.encoder_sha256: str = str(recorded)
        self.feature_cols: list[str] = list(
            meta.get("feature_cols") or meta.get("feature_columns") or []
        )
        self._date_index = {d: i for i, d in enumerate(self.dates)}
        # Proleptic-Gregorian ordinals, ascending because `dates` is.  Used for
        # the bisection in `window()`: `np.searchsorted` has no overload for a
        # `datetime.date` needle (it would land in the object-array path), so
        # the comparison is done on integers.
        self._date_ordinals = np.array([d.toordinal() for d in self.dates], dtype=np.int64)
        self._emb = np.load(self.dir / EMBEDDINGS_FILE, mmap_mode="r")
        if self._emb.shape != (len(self.dates), len(self.tickers), self.embed_dim):
            raise ValueError(
                f"embeddings shape {self._emb.shape} disagrees with index.json "
                f"({len(self.dates)}, {len(self.tickers)}, {self.embed_dim})"
            )

    def __len__(self) -> int:
        return len(self.dates)

    def index_of(self, d: date) -> int:
        try:
            return self._date_index[_to_date(d)]
        except KeyError:
            raise KeyError(f"{d} not in cache (range {self.dates[0]}..{self.dates[-1]})") from None

    def at(self, d: date) -> np.ndarray:
        """``[N, D]`` float16 embedding for one date (a view into the mmap)."""
        return np.asarray(self._emb[self.index_of(d)])

    def window(self, start: date, end: date) -> np.ndarray:
        """``[T', N, D]`` for all cached dates with ``start <= date <= end`` (inclusive)."""
        s, e = _to_date(start), _to_date(end)
        lo = int(np.searchsorted(self._date_ordinals, s.toordinal(), side="left"))
        hi = int(np.searchsorted(self._date_ordinals, e.toordinal(), side="right"))
        return np.asarray(self._emb[lo:hi])

    def as_array(self) -> np.ndarray:
        """The whole ``[T, N, D]`` mmap (no copy)."""
        return np.asarray(self._emb)
