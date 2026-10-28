"""The no-lookahead harness itself (F2.2): it must pass a causal stage and catch a leaky one.

Every later stage's tests call ``tests/integrity.py::assert_no_lookahead``; these tests pin
down what "catch" means so a green lookahead test elsewhere is worth something.
"""

import numpy as np
import pandas as pd
import pytest

from tests.integrity import assert_no_lookahead

D = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=40)), name="date")
T = D[24]  # the attack starts at D[25]; 15 rows after t


def random_walk(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, size=(len(D), 3))
    return pd.DataFrame(100.0 * np.exp(steps.cumsum(axis=0)), index=D, columns=["A", "B", "C"])


class TestAssertNoLookahead:
    def test_causal_rolling_mean_passes(self):
        assert_no_lookahead(lambda p: p.rolling(3).mean(), random_walk(), T)

    def test_series_output_is_supported(self):
        assert_no_lookahead(lambda p: p.mean(axis=1), random_walk(), T)

    def test_string_t_is_accepted(self):
        assert_no_lookahead(lambda p: p.shift(1), random_walk(), str(T.date()))

    def test_shift_from_the_future_is_caught(self):
        with pytest.raises(AssertionError, match="lookahead detected"):
            assert_no_lookahead(lambda p: p.shift(-1), random_walk(), T)

    def test_centred_window_is_caught(self):
        with pytest.raises(AssertionError, match="lookahead detected"):
            assert_no_lookahead(lambda p: p.rolling(3, center=True).mean(), random_walk(), T)

    def test_full_sample_normalisation_is_caught(self):
        with pytest.raises(AssertionError, match="lookahead detected"):
            assert_no_lookahead(lambda p: p / p.mean(), random_walk(), T)

    def test_backfill_across_t_is_caught(self):
        prices = random_walk()
        prices.loc[T, "B"] = np.nan  # bfill would take D[25]'s price, which is after t
        with pytest.raises(AssertionError, match="lookahead detected"):
            assert_no_lookahead(lambda p: p.bfill(), prices, T)

    def test_failure_names_the_first_differing_date(self):
        # shift(-1) puts D[25]'s (attacked) price on row t = D[24], and only there.
        with pytest.raises(AssertionError, match=r"first at 2020-02-04 \(1 row\(s\) differ\)"):
            assert_no_lookahead(lambda p: p.shift(-1), random_walk(), T)

    def test_input_is_not_modified(self):
        prices = random_walk()
        before = prices.copy()
        assert_no_lookahead(lambda p: p.rolling(3).mean(), prices, T)
        pd.testing.assert_frame_equal(prices, before)

    def test_t_must_be_in_the_index_with_a_row_after_it(self):
        with pytest.raises(ValueError, match="last row"):
            assert_no_lookahead(lambda p: p, random_walk(), D[-1])
        with pytest.raises(ValueError, match="not in the index"):
            assert_no_lookahead(lambda p: p, random_walk(), "2019-12-31")
