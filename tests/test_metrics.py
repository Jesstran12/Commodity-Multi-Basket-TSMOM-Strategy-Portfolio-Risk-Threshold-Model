"""Metrics: Phase 4's first slice (annualised return and vol, Sharpe, Sortino, maximum
drawdown, the ``performance`` bundle and the §8 metrics known-answer integrity test) and
Phase 5's rest of §5.1 (monthly returns, drawdown duration, Calmar, hit rates, skew, excess
kurtosis, best and worst month, and the ``summarise`` bundle).

Synthetic series use business days from 2020-01-01: d1..d12 =
01-01 01-02 01-03 01-06 01-07 01-08 01-09 01-10 01-13 01-14 01-15 01-16.
Every expected value is hand-computed; the arithmetic is in a comment next to it.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ANNUALISATION, load_config
from src.metrics import (
    METRIC_KEYS,
    PERFORMANCE_KEYS,
    annualised_return,
    annualised_vol,
    calmar,
    hit_rate,
    kurtosis,
    max_drawdown,
    max_drawdown_duration,
    monthly_returns,
    performance,
    sharpe,
    skew,
    sortino,
    summarise,
)

NA = np.nan
SQ = np.sqrt(ANNUALISATION)  # 15.8745...
D = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=12)), name="date")  # d1..d12


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=D[: len(values)], dtype=float, name="r")


# The §8 tiny series: r = [0.10, -0.05, 0.02, -0.10].
#   wealth: 1.10, 1.045, 1.0659, 0.95931 (= 1.10 * 0.95 * 1.02 * 0.90)
#   peaks (starting wealth 1 included): 1.10, 1.10, 1.10, 1.10
#   drawdowns: 0, -0.05, 1.0659/1.1 - 1 = -0.031, 0.95931/1.1 - 1 = -0.1279
#   mean = -0.03 / 4 = -0.0075; deviations 0.1075, -0.0425, 0.0275, -0.0925;
#   squares 0.01155625 + 0.00180625 + 0.00075625 + 0.00855625 = 0.022675; / 3 = 0.00755833;
#   std = 0.0869387; Sharpe = -0.0075 / 0.0869387 * sqrt(252) = -0.086268 * 15.8745 = -1.3695
#   downside: min(r, 0)^2 = 0, 0.0025, 0, 0.01 -> mean 0.003125 -> sqrt 0.0559017;
#   Sortino = -0.0075 / 0.0559017 * sqrt(252) = -0.134164 * 15.8745 = -2.1298
#   ann return = 0.95931^(252/4) - 1 = exp(63 * ln 0.95931) - 1 = exp(63 * -0.0415433) - 1
#              = exp(-2.61723) - 1 = -0.926984
TINY = series([0.10, -0.05, 0.02, -0.10])
TINY_WEALTH = 1.10 * 0.95 * 1.02 * 0.90
TINY_STD = np.sqrt(0.022675 / 3)
TINY_DOWNSIDE = np.sqrt(0.003125)


# ---------------------------------------------------------------- annualised_return


class TestAnnualisedReturn:
    def test_hand_computed(self):
        assert annualised_return(TINY) == pytest.approx(TINY_WEALTH ** (252 / 4) - 1)

    def test_constant_daily_return_compounds(self):
        # 252 days of 0.001: (1.001^252)^(252/252) - 1 = 1.001^252 - 1 = 0.28640...
        r = pd.Series(0.001, index=pd.bdate_range("2019-01-01", periods=252))
        assert annualised_return(r) == pytest.approx(1.001**252 - 1)

    def test_nan_rows_are_dropped_not_zero(self):
        # [0.10, NaN, -0.05]: two rows -> wealth 1.045 ^ (252/2) - 1.
        assert annualised_return(series([0.10, NA, -0.05])) == pytest.approx(1.045**126 - 1)

    def test_empty_is_nan_and_wipeout_is_minus_one(self):
        assert np.isnan(annualised_return(series([NA, NA])))
        assert annualised_return(series([0.5, -1.0, 0.2])) == -1.0

    def test_returns_a_float(self):
        assert type(annualised_return(TINY)) is float


# ------------------------------------------------------------------- annualised_vol


class TestAnnualisedVol:
    def test_hand_computed(self):
        assert annualised_vol(TINY) == pytest.approx(TINY_STD * SQ)

    def test_uses_sample_std_ddof_one(self):
        # [0.01, 0.03]: ddof 1 gives 0.02 / sqrt(2) = 0.014142; ddof 0 would give 0.01.
        assert annualised_vol(series([0.01, 0.03])) == pytest.approx(0.02 / np.sqrt(2) * SQ)

    def test_nan_rows_are_dropped(self):
        assert annualised_vol(series([0.01, NA, 0.03])) == pytest.approx(0.02 / np.sqrt(2) * SQ)

    def test_fewer_than_two_rows_is_nan(self):
        assert np.isnan(annualised_vol(series([0.01])))
        assert np.isnan(annualised_vol(series([NA])))

    def test_returns_a_float(self):
        assert type(annualised_vol(TINY)) is float


# --------------------------------------------------------------------------- sharpe


class TestSharpe:
    def test_hand_computed(self):
        assert sharpe(TINY) == pytest.approx(-0.0075 / TINY_STD * SQ)

    def test_risk_free_rate_is_an_annual_rate_spread_over_the_year(self):
        # rf 2.52% a year is exactly 0.0001 a day; every excess return drops by that, the
        # std does not change: Sharpe = (-0.0075 - 0.0001) / std * sqrt(252).
        assert sharpe(TINY, 0.0252) == pytest.approx(-0.0076 / TINY_STD * SQ)

    def test_default_risk_free_rate_is_zero_in_config_and_here(self):
        assert load_config().inference.risk_free_rate == 0.0
        assert sharpe(TINY) == sharpe(TINY, 0.0)

    def test_scale_invariant(self):
        assert sharpe(TINY * 3) == pytest.approx(sharpe(TINY))

    def test_zero_std_or_too_short_is_nan(self):
        assert np.isnan(sharpe(series([0.01, 0.01, 0.01])))
        assert np.isnan(sharpe(series([0.01])))
        assert np.isnan(sharpe(series([NA, NA])))

    def test_returns_a_float(self):
        assert type(sharpe(TINY)) is float


# -------------------------------------------------------------------------- sortino


class TestSortino:
    def test_hand_computed(self):
        assert sortino(TINY) == pytest.approx(-0.0075 / TINY_DOWNSIDE * SQ)

    def test_positive_days_stay_in_the_denominator_count(self):
        # [0.02, 0.02, -0.02, 0.02]: mean 0.04 / 4 = 0.01; downside sqrt((0 + 0 + 0.0004 + 0)
        # / 4) = sqrt(0.0001) = 0.01 (not sqrt(0.0004 / 1) = 0.02); Sortino 1.0 * sqrt(252).
        assert sortino(series([0.02, 0.02, -0.02, 0.02])) == pytest.approx(1.0 * SQ)

    def test_risk_free_rate_moves_the_target(self):
        # rf 0.0252 -> daily 0.0001: excess [0.0199, 0.0199, -0.0201, 0.0199], mean 0.0396 / 4
        # = 0.0099, downside sqrt(0.0201^2 / 4) = 0.01005; Sortino 0.0099 / 0.01005 * sqrt(252).
        out = sortino(series([0.02, 0.02, -0.02, 0.02]), 0.0252)
        assert out == pytest.approx(0.0099 / 0.01005 * SQ)

    def test_no_downside_is_nan(self):
        assert np.isnan(sortino(series([0.01, 0.02, 0.0])))

    def test_empty_is_nan_and_returns_a_float(self):
        assert np.isnan(sortino(series([NA])))
        assert type(sortino(TINY)) is float


# --------------------------------------------------------------------- max_drawdown


class TestMaxDrawdown:
    def test_hand_computed(self):
        assert max_drawdown(TINY) == pytest.approx(TINY_WEALTH / 1.10 - 1)

    def test_loss_on_the_first_day_counts_from_the_starting_wealth(self):
        # [-0.10, 0.05]: wealth 0.90, 0.945; peaks 1, 1 -> drawdowns -0.10, -0.055.
        assert max_drawdown(series([-0.10, 0.05])) == pytest.approx(-0.10)

    def test_drawdown_is_measured_from_the_running_peak(self):
        # [0.20, -0.10, -0.10, 0.50]: wealth 1.2, 1.08, 0.972, 1.458; peaks 1.2, 1.2, 1.2,
        # 1.458 -> drawdowns 0, -0.10, -0.19, 0 -> -0.19 (not 0.972 - 1 = -0.028).
        assert max_drawdown(series([0.20, -0.10, -0.10, 0.50])) == pytest.approx(-0.19)

    def test_never_below_a_peak_is_zero(self):
        assert max_drawdown(series([0.01, 0.02, 0.0, 0.03])) == 0.0

    def test_nan_rows_are_dropped(self):
        assert max_drawdown(series([-0.10, NA, 0.05])) == pytest.approx(-0.10)

    def test_empty_is_nan_and_returns_a_float(self):
        assert np.isnan(max_drawdown(series([NA, NA])))
        assert type(max_drawdown(TINY)) is float


# ---------------------------------------------------------------------- performance


class TestPerformance:
    def test_hand_computed_bundle(self):
        out = performance(series([NA, 0.10, -0.05, 0.02, -0.10, NA]))
        assert list(out) == list(PERFORMANCE_KEYS)
        assert out["n_days"] == 4
        assert out["first_date"] == D[1] and out["last_date"] == D[4]
        assert out["ann_return"] == pytest.approx(TINY_WEALTH ** (252 / 4) - 1)
        assert out["ann_vol"] == pytest.approx(TINY_STD * SQ)
        assert out["sharpe"] == pytest.approx(-0.0075 / TINY_STD * SQ)
        assert out["sortino"] == pytest.approx(-0.0075 / TINY_DOWNSIDE * SQ)
        assert out["max_drawdown"] == pytest.approx(TINY_WEALTH / 1.10 - 1)

    def test_types(self):
        out = performance(TINY)
        assert type(out["n_days"]) is int
        assert isinstance(out["first_date"], pd.Timestamp)
        for k in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"):
            assert type(out[k]) is float, k

    def test_risk_free_rate_is_passed_through(self):
        out = performance(TINY, 0.0252)
        assert out["sharpe"] == pytest.approx(sharpe(TINY, 0.0252))
        assert out["sortino"] == pytest.approx(sortino(TINY, 0.0252))

    def test_empty_series(self):
        out = performance(series([NA, NA]))
        assert out["n_days"] == 0 and out["first_date"] is None and out["last_date"] is None
        for k in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"):
            assert np.isnan(out[k]), k


# ----------------------------------------------------------------------- §8 integrity


class TestIntegrityMetricsKnownAnswer:
    """PROJECT_LOG.md §8 "metrics known-answer": max drawdown, Sharpe, Sortino on
    hand-computed tiny series (the arithmetic is at the top of this module)."""

    def test_max_drawdown_known_answer(self):
        # Trough 0.95931 against the peak 1.10 after the first day: -0.12790.
        assert max_drawdown(TINY) == pytest.approx(-0.127900, abs=1e-6)

    def test_sharpe_known_answer(self):
        assert sharpe(TINY) == pytest.approx(-1.3695, abs=1e-4)

    def test_sortino_known_answer(self):
        assert sortino(TINY) == pytest.approx(-2.1298, abs=1e-4)

    def test_annualised_return_and_vol_known_answer(self):
        assert annualised_return(TINY) == pytest.approx(-0.926984, abs=1e-6)
        assert annualised_vol(TINY) == pytest.approx(1.380109, abs=1e-6)  # 0.0869387 * 15.874508

    def test_a_steady_climb(self):
        # Five days of +1%: no drawdown; std 0 -> Sharpe NaN; no downside -> Sortino NaN;
        # ann return 1.01^(252/5 * 5) - 1 = 1.01^252 - 1.
        r = series([0.01] * 5)
        assert max_drawdown(r) == 0.0
        assert np.isnan(sharpe(r)) and np.isnan(sortino(r))
        assert annualised_return(r) == pytest.approx(1.01**252 - 1)


# =================================================================== Phase 5 (F5.1)

# A series across three calendar months, with a NaN row inside February:
#   2020-01-30 +0.10, 2020-01-31 -0.05 | 2020-02-03 +0.02, 2020-02-04 NaN | 2020-03-02 -0.10
#   January:  1.10 * 0.95 - 1 = +0.045, labelled 2020-01-31 (its last row)
#   February: +0.02, labelled 2020-02-03 (the NaN row is dropped first, so 02-03 is the last)
#   March:    -0.10, labelled 2020-03-02
M_DATES = pd.DatetimeIndex(
    ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04", "2020-03-02"], name="date"
)
MONTHS = pd.Series([0.10, -0.05, 0.02, NA, -0.10], index=M_DATES, name="r")


# ------------------------------------------------------------------ monthly_returns


class TestMonthlyReturns:
    def test_hand_computed(self):
        out = monthly_returns(MONTHS)
        expected = pd.Series(
            [1.10 * 0.95 - 1, 0.02, -0.10],
            index=pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-03-02"], name="date"),
            name="r",
        )
        pd.testing.assert_series_equal(out, expected, check_freq=False)

    def test_a_month_with_only_nan_rows_is_absent_not_zero(self):
        r = pd.Series(
            [0.01, NA, NA, 0.02],
            index=pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"]),
        )
        out = monthly_returns(r)
        assert list(out.index.month) == [1, 3]
        assert out.tolist() == pytest.approx([0.01, 0.02])

    def test_partial_months_at_both_ends_are_kept(self):
        # Twelve business days from 2020-01-01 sit in January only: one month, compounded.
        out = monthly_returns(TINY)
        assert len(out) == 1
        assert out.iloc[0] == pytest.approx(TINY_WEALTH - 1)
        assert out.index[0] == D[3]

    def test_years_do_not_merge_same_month_numbers(self):
        r = pd.Series([0.01, 0.02], index=pd.DatetimeIndex(["2020-01-15", "2021-01-15"]))
        assert len(monthly_returns(r)) == 2

    def test_empty(self):
        assert len(monthly_returns(series([NA, NA]))) == 0


# ------------------------------------------------------------- max_drawdown_duration


class TestMaxDrawdownDuration:
    def test_hand_computed(self):
        # TINY wealth 1.10, 1.045, 1.0659, 0.95931 against the peak 1.10: rows 2-4 are below
        # it and never recover -> a spell of 3 rows, counted to the end.
        assert max_drawdown_duration(TINY) == 3

    def test_longest_spell_wins(self):
        # [-0.10, 0.05, 0.10, -0.01]: wealth 0.90, 0.945, 1.0395, 1.029105; peaks 1, 1, 1.0395,
        # 1.0395 -> below on rows 1-2 (2 rows, recovered on row 3) and row 4 (1 row) -> 2.
        assert max_drawdown_duration(series([-0.10, 0.05, 0.10, -0.01])) == 2

    def test_back_exactly_at_the_peak_ends_the_spell(self):
        # [0.10, -0.50, 1.00, 0.01]: wealth 1.1, 0.55, 1.1 (exact in floating point), 1.111;
        # only row 2 is strictly below the peak -> 1.
        assert max_drawdown_duration(series([0.10, -0.50, 1.00, 0.01])) == 1

    def test_loss_on_the_first_day_starts_a_spell(self):
        # [-0.01, -0.01, 0.05]: wealth 0.99, 0.9801, 1.02905 against the starting 1 -> 2.
        assert max_drawdown_duration(series([-0.01, -0.01, 0.05])) == 2

    def test_nan_rows_are_dropped_before_counting(self):
        # The NaN row is not a day of the spell: [-0.01, NaN, -0.01, 0.05] -> 2, not 3.
        assert max_drawdown_duration(series([-0.01, NA, -0.01, 0.05])) == 2

    def test_never_below_and_empty_are_zero_and_an_int(self):
        assert max_drawdown_duration(series([0.01, 0.0, 0.02])) == 0
        assert max_drawdown_duration(series([NA])) == 0
        assert type(max_drawdown_duration(TINY)) is int


# --------------------------------------------------------------------------- calmar


class TestCalmar:
    def test_hand_computed(self):
        # ann return -0.926984 over |max drawdown| 0.127900 = -7.2477
        assert calmar(TINY) == pytest.approx(
            (TINY_WEALTH ** (252 / 4) - 1) / -(TINY_WEALTH / 1.10 - 1)
        )
        assert calmar(TINY) == pytest.approx(-7.2477, abs=1e-4)

    def test_positive_case(self):
        # [0.20, -0.10]: wealth 1.08 -> ann return 1.08^126 - 1; drawdown -0.10 -> / 0.10.
        assert calmar(series([0.20, -0.10])) == pytest.approx((1.08**126 - 1) / 0.10)

    def test_no_drawdown_or_empty_is_nan(self):
        assert np.isnan(calmar(series([0.01, 0.02])))
        assert np.isnan(calmar(series([NA])))
        assert type(calmar(TINY)) is float


# ------------------------------------------------------------------------- hit_rate


class TestHitRate:
    def test_hand_computed(self):
        # [0.01, 0.0, -0.01, NaN, 0.02]: four rows, two strictly positive -> 0.5. A zero
        # return is not a hit; the NaN row is not a row.
        assert hit_rate(series([0.01, 0.0, -0.01, NA, 0.02])) == 0.5

    def test_tiny(self):
        assert hit_rate(TINY) == 0.5  # 0.10 and 0.02 of four

    def test_monthly_hit_rate_is_hit_rate_of_monthly_returns(self):
        # Months +0.045, +0.02, -0.10 -> 2 / 3.
        assert hit_rate(monthly_returns(MONTHS)) == pytest.approx(2 / 3)

    def test_empty_is_nan_and_a_float(self):
        assert np.isnan(hit_rate(series([NA])))
        assert type(hit_rate(TINY)) is float


# --------------------------------------------------------------------- skew, kurtosis


class TestSkewAndKurtosis:
    # [0, 0, 0, 0.01], n = 4: mean 0.0025; deviations -0.0025 (x3), +0.0075.
    #   m2 = (3 * 6.25e-6 + 5.625e-5) / 4 = 1.875e-5
    #   m3 = (3 * -1.5625e-8 + 4.21875e-7) / 4 = 9.375e-8
    #   m4 = (3 * 3.90625e-11 + 3.1640625e-9) / 4 = 8.203125e-10
    #   g1 = m3 / m2^1.5 = 1.154701 (= 2 / sqrt 3); adjusted G1 = g1 * sqrt(n (n-1)) / (n-2)
    #      = 1.154701 * sqrt(12) / 2 = 2.0
    #   g2 = m4 / m2^2 - 3 = 2.333333 - 3 = -0.666667; adjusted excess
    #      G2 = ((n+1) g2 + 6) (n-1) / ((n-2)(n-3)) = (5 * -0.666667 + 6) * 3 / 2 = 4.0
    SPIKE = series([0.0, 0.0, 0.0, 0.01])

    def test_hand_computed_skew(self):
        assert skew(self.SPIKE) == pytest.approx(2.0)

    def test_hand_computed_excess_kurtosis(self):
        assert kurtosis(self.SPIKE) == pytest.approx(4.0)

    def test_mirror_image_flips_the_skew_and_keeps_the_kurtosis(self):
        assert skew(-self.SPIKE) == pytest.approx(-2.0)
        assert kurtosis(-self.SPIKE) == pytest.approx(4.0)

    def test_symmetric_series(self):
        # [0.03, -0.01] x 4: deviations +-0.02, so g1 = 0 and m4 / m2^2 = 1, g2 = -2;
        # G2 = ((9)(-2) + 6) * 7 / (6 * 5) = -2.8.
        r = series([0.03, -0.01] * 4)
        assert skew(r) == pytest.approx(0.0, abs=1e-12)
        assert kurtosis(r) == pytest.approx(-2.8)

    def test_nan_rows_dropped_and_scale_and_shift_invariant(self):
        r = series([0.0, NA, 0.0, 0.0, 0.01])
        assert skew(r) == pytest.approx(2.0)
        assert kurtosis(self.SPIKE * 7 + 0.3) == pytest.approx(4.0)

    def test_too_short_or_constant_is_nan(self):
        assert np.isnan(skew(series([0.01, 0.02])))  # skew needs 3 rows
        assert np.isnan(kurtosis(series([0.01, 0.02, 0.04])))  # kurtosis needs 4
        assert np.isnan(skew(series([0.25] * 5)))  # no variance, no shape (0.25 is exact)
        assert np.isnan(kurtosis(series([0.25] * 5)))

    def test_floats(self):
        assert type(skew(TINY)) is float and type(kurtosis(TINY)) is float


# ------------------------------------------------------------------------ summarise


class TestSummarise:
    def test_keys_extend_performance_keys_in_order(self):
        assert METRIC_KEYS[: len(PERFORMANCE_KEYS)] == PERFORMANCE_KEYS
        assert METRIC_KEYS[len(PERFORMANCE_KEYS) :] == (
            "max_drawdown_duration",
            "calmar",
            "hit_rate_daily",
            "hit_rate_monthly",
            "skew",
            "kurtosis",
            "best_month",
            "worst_month",
        )
        assert list(summarise(MONTHS)) == list(METRIC_KEYS)

    def test_hand_computed_bundle(self):
        out = summarise(MONTHS)
        # Rows: 0.10, -0.05, 0.02, -0.10 (the NaN dropped) -> the TINY statistics.
        assert out["n_days"] == 4
        assert out["first_date"] == M_DATES[0] and out["last_date"] == M_DATES[-1]
        assert out["sharpe"] == pytest.approx(sharpe(TINY))
        assert out["max_drawdown"] == pytest.approx(max_drawdown(TINY))
        assert out["max_drawdown_duration"] == 3
        assert out["calmar"] == pytest.approx(calmar(TINY))
        assert out["hit_rate_daily"] == 0.5
        assert out["hit_rate_monthly"] == pytest.approx(2 / 3)
        assert out["skew"] == pytest.approx(skew(TINY))
        assert out["kurtosis"] == pytest.approx(kurtosis(TINY))
        assert out["best_month"] == pytest.approx(0.045)
        assert out["worst_month"] == pytest.approx(-0.10)

    def test_matches_performance_on_the_shared_keys(self):
        out, perf = summarise(MONTHS, 0.0252), performance(MONTHS, 0.0252)
        for k in PERFORMANCE_KEYS:
            assert out[k] == perf[k] or (np.isnan(out[k]) and np.isnan(perf[k])), k

    def test_types(self):
        out = summarise(MONTHS)
        assert type(out["n_days"]) is int and type(out["max_drawdown_duration"]) is int
        for k in METRIC_KEYS:
            if k not in ("n_days", "max_drawdown_duration", "first_date", "last_date"):
                assert type(out[k]) is float, k

    def test_empty_series(self):
        out = summarise(series([NA, NA]))
        assert out["n_days"] == 0 and out["max_drawdown_duration"] == 0
        assert out["first_date"] is None
        for k in ("calmar", "hit_rate_daily", "hit_rate_monthly", "skew", "kurtosis"):
            assert np.isnan(out[k]), k
        assert np.isnan(out["best_month"]) and np.isnan(out["worst_month"])
