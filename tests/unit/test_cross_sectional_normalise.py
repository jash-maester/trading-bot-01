"""Per-date cross-sectional normalisation of the model's input features.

The defect this fixes: the target is a per-date cross-sectional z-score, but
inputs were standardised by ONE global (mean, std) per feature frozen over the
whole train split.  The model was asked a relative question and handed absolute
numbers, so a market-wide shift in level looked like a cross-sectional signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from trader.training.supervised import cross_sectional_normalise


def _feats(vals: list[list[float]]) -> np.ndarray:
    """[T, N] -> [T, N, 1]."""
    return np.asarray(vals, dtype=np.float32)[:, :, None]


def test_market_wide_level_shift_produces_identical_output() -> None:
    """The property the global-stats version could not have.

    Day 2 is day 1 with every name's volatility tripled and lifted — the whole
    market moved, the cross-section did not.  The target (a per-date z-score)
    is identical on both days, so the input must be too.  Under a single frozen
    (mean, std) it would not have been: day 2 would read as uniformly "high".
    """
    day1 = [0.10, 0.20, 0.30, 0.40]
    day2 = [3 * v + 5.0 for v in day1]
    out = cross_sectional_normalise(
        _feats([day1, day2]), np.ones((2, 4), dtype=bool), min_cross_section=2
    )
    np.testing.assert_allclose(out[0, :, 0], out[1, :, 0], rtol=0, atol=1e-6)


def test_is_a_within_date_monotone_map() -> None:
    """Ranking within a date is preserved; that is the whole information content."""
    out = cross_sectional_normalise(
        _feats([[5.0, 1.0, 9.0, 3.0]]), np.ones((1, 4), dtype=bool), min_cross_section=2
    )
    assert list(np.argsort(out[0, :, 0])) == [1, 3, 0, 2]
    assert out[0, :, 0].mean() == pytest.approx(0.0, abs=1e-6)


def test_masked_names_are_zero_and_do_not_displace_ranks() -> None:
    """A non-tradeable name must not consume a rank slot.

    Two tradeable names among four: they must come out as the same two scores
    they would get on their own, not squeezed by two phantom neighbours.
    """
    mask = np.array([[True, False, True, False]])
    out = cross_sectional_normalise(
        _feats([[1.0, 999.0, 2.0, -999.0]]), mask, min_cross_section=2
    )[0, :, 0]
    assert out[1] == 0.0 and out[3] == 0.0
    alone = cross_sectional_normalise(
        _feats([[1.0, 2.0]]), np.ones((1, 2), dtype=bool), min_cross_section=2
    )[0, :, 0]
    np.testing.assert_allclose([out[0], out[2]], alone, atol=1e-6)


def test_no_lookahead_across_dates() -> None:
    """A date's scores depend on that date's cross-section only.

    Appending a later day — or changing one — must not move an earlier day's
    output by a float.  This is the check that the normaliser cannot leak.
    """
    a = _feats([[1.0, 2.0, 3.0]])
    b = _feats([[1.0, 2.0, 3.0], [100.0, -100.0, 0.0]])
    o_a = cross_sectional_normalise(a, np.ones((1, 3), dtype=bool), min_cross_section=2)
    o_b = cross_sectional_normalise(b, np.ones((2, 3), dtype=bool), min_cross_section=2)
    np.testing.assert_array_equal(o_a[0], o_b[0])


def test_thin_and_degenerate_dates_are_left_at_zero() -> None:
    """One tradeable name has no cross-section; a flat day has no spread."""
    thin = cross_sectional_normalise(
        _feats([[1.0, 2.0, 3.0]]), np.array([[True, False, False]]), min_cross_section=2
    )
    assert np.all(thin == 0.0)
    flat = cross_sectional_normalise(
        _feats([[7.0, 7.0, 7.0]]), np.ones((1, 3), dtype=bool), min_cross_section=2
    )
    np.testing.assert_allclose(flat[0, :, 0], np.zeros(3), atol=1e-6)


def test_heavy_tails_are_bounded() -> None:
    """dollar_volume_20 spans ~11 orders of magnitude; rank scores must not."""
    vals = [1.0, 10.0, 1e3, 1e6, 1e11]
    out = cross_sectional_normalise(
        _feats([vals]), np.ones((1, 5), dtype=bool), min_cross_section=2
    )
    assert np.abs(out).max() < 4.0


def test_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="xs_normalise must be one of"):
        cross_sectional_normalise(
            _feats([[1.0, 2.0]]), np.ones((1, 2), dtype=bool), mode="quantile"
        )


def test_inference_reads_xs_normalise_from_the_artefact() -> None:
    """The seam: `predict_signal.py` must not hand raw features to a rank model.

    A model fitted on per-date rank scores and scored on raw inputs still runs
    and still emits finite numbers — it is silently meaningless.  Pin that the
    inference path reads the mode from the checkpoint's own `train_cfg`, and
    that an artefact written before the key existed still gets None.
    """
    import re
    from pathlib import Path

    src = Path("scripts/predict_signal.py").read_text(encoding="utf-8")
    assert 'tcfg.get("xs_normalise")' in src, "inference must read the saved mode"
    call = re.search(r"build_panel_tensors\((.*?)\n    \)", src, re.S)
    assert call and "xs_normalise=xs_normalise" in call.group(1), (
        "build_panel_tensors must be passed the artefact's xs_normalise"
    )


def test_saved_train_cfg_carries_xs_normalise() -> None:
    """`asdict(train_cfg)` is what inference reads; the field must be in it."""
    from dataclasses import asdict

    from trader.training.supervised import SupervisedConfig

    assert asdict(SupervisedConfig(xs_normalise="rank"))["xs_normalise"] == "rank"
    assert asdict(SupervisedConfig())["xs_normalise"] is None
