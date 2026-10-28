"""Inference (Phase 5, F5.2): the Sharpe t-statistic, the stationary block bootstrap and its
confidence interval, the Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio
(PROJECT_LOG.md §5.2).

Conventions the tests pin (the module docstring of ``src/inference.py`` states them):

* Sharpe ratios given to or returned by a function are **annualised**; the PSR formula works
  on the per-row Sharpe ``SR / sqrt(252)`` internally.
* Skew and kurtosis are ``src.metrics.skew`` and ``src.metrics.kurtosis`` (bias-adjusted,
  excess kurtosis), so the PSR's ``(gamma_4 - 1) / 4`` term is ``(kurtosis + 2) / 4``.
* The bootstrap's random draws follow a documented order, so a seed fixes every index.

Every expected value is hand-computed or rebuilt from the documented recipe in plain loops.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.config import ANNUALISATION, load_config
from src.inference import (
    EULER_GAMMA,
    INFERENCE_KEYS,
    bootstrap_sharpe_ci,
    deflated_sharpe,
    expected_max_sharpe,
    inference_summary,
    probabilistic_sharpe,
    sharpe_t_stat,
    stationary_bootstrap_indices,
)
from src.metrics import sharpe

NA = np.nan
SQ = np.sqrt(ANNUALISATION)
PHI = NormalDist()


def series(values, start: str = "2020-01-01") -> pd.Series:
    values = list(values)
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def noisy(n: int, mean: float = 0.0005, sd: float = 0.01, seed: int = 7) -> pd.Series:
    """A seeded i.i.d. normal daily series (test data only)."""
    return series(np.random.default_rng(seed).normal(mean, sd, n))


TINY = series([0.10, -0.05, 0.02, -0.10])

# [0.03, -0.01] x 4 (n = 8): mean 0.01, deviations +-0.02 -> skew 0, excess kurtosis -2.8
# (tests/test_metrics.py::TestSkewAndKurtosis). std (ddof 1) = sqrt(8 * 0.0004 / 7)
# = 0.0213809; per-row Sharpe = 0.01 / 0.0213809 = 0.467707 (= sqrt(7 / 32)).
ALT8 = series([0.03, -0.01] * 4)
ALT8_SR = np.sqrt(7 / 32)


# -------------------------------------------------------------------- sharpe_t_stat


class TestSharpeTStat:
    def test_hand_computed(self):
        # SR * sqrt(T / 252) with T = 4: -1.3695 * sqrt(4 / 252) = -1.3695 * 0.125988 = -0.17254
        assert sharpe_t_stat(TINY) == pytest.approx(sharpe(TINY) * np.sqrt(4 / 252))
        assert sharpe_t_stat(TINY) == pytest.approx(-0.17254, abs=1e-5)

    def test_equals_per_row_sharpe_times_root_t(self):
        r = noisy(500)
        per_row = r.mean() / r.std()
        assert sharpe_t_stat(r) == pytest.approx(per_row * np.sqrt(500))

    def test_counts_non_nan_rows_and_passes_the_risk_free_rate(self):
        r = series([0.10, NA, -0.05, 0.02, NA, -0.10])
        assert sharpe_t_stat(r) == pytest.approx(sharpe_t_stat(TINY))
        assert sharpe_t_stat(TINY, 0.0252) == pytest.approx(sharpe(TINY, 0.0252) * np.sqrt(4 / 252))

    def test_undefined_is_nan_and_a_float(self):
        assert np.isnan(sharpe_t_stat(series([0.01, 0.01])))
        assert type(sharpe_t_stat(TINY)) is float


# ------------------------------------------------------- stationary_bootstrap_indices


def loop_indices(n: int, block_length: int, n_resamples: int, seed: int) -> np.ndarray:
    """The documented recipe, written as plain loops (an independent re-statement).

    One generator ``np.random.default_rng(seed)``. For each resample in turn: draw
    ``starts = rng.integers(0, n, size=n)``, then ``u = rng.random(n)``; row 0 starts a
    block at ``starts[0]``; for t >= 1 a new block starts at ``starts[t]`` when
    ``u[t] < 1 / block_length``, otherwise the index is the previous one plus one, wrapping
    from ``n - 1`` to ``0``.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, n), dtype=np.int64)
    for k in range(n_resamples):
        starts = rng.integers(0, n, size=n)
        u = rng.random(n)
        out[k, 0] = starts[0]
        for t in range(1, n):
            out[k, t] = starts[t] if u[t] < 1 / block_length else (out[k, t - 1] + 1) % n
    return out


class TestStationaryBootstrapIndices:
    def test_matches_the_documented_recipe(self):
        for n, b, k, seed in [(7, 3, 5, 0), (20, 4, 3, 11), (1, 1, 2, 3), (50, 20, 4, 0)]:
            got = stationary_bootstrap_indices(n, b, k, seed)
            np.testing.assert_array_equal(got, loop_indices(n, b, k, seed))

    def test_shape_dtype_range(self):
        idx = stationary_bootstrap_indices(30, 5, 8, 0)
        assert idx.shape == (8, 30)
        assert np.issubdtype(idx.dtype, np.integer)
        assert idx.min() >= 0 and idx.max() < 30

    def test_block_length_one_is_the_iid_bootstrap(self):
        # Every u < 1, so every row is a fresh start: row k is the (2k)-th draw.
        rng = np.random.default_rng(5)
        first = rng.integers(0, 9, size=9)
        idx = stationary_bootstrap_indices(9, 1, 2, 5)
        np.testing.assert_array_equal(idx[0], first)

    def test_one_huge_block_wraps_around(self):
        # block_length larger than any draw can beat: one block per resample, a rotation
        # of 0..n-1 starting at the first start draw, wrapping from n-1 to 0.
        idx = stationary_bootstrap_indices(6, 10**12, 3, 2)
        for row in idx:
            np.testing.assert_array_equal(row, (row[0] + np.arange(6)) % 6)

    def test_seeded_and_deterministic(self):
        a = stationary_bootstrap_indices(40, 5, 10, 0)
        np.testing.assert_array_equal(a, stationary_bootstrap_indices(40, 5, 10, 0))
        assert not np.array_equal(a, stationary_bootstrap_indices(40, 5, 10, 1))

    def test_mean_block_length_is_about_block_length(self):
        # A block ends where the next index is not the previous one plus one (mod n); with
        # n = 5000 and p = 1/20 the mean run is ~20 (a fresh start equal to prev + 1 is rare).
        idx = stationary_bootstrap_indices(5000, 20, 4, 0)
        breaks = (idx[:, 1:] != (idx[:, :-1] + 1) % 5000).sum()
        mean_block = idx.size / (breaks + idx.shape[0])
        assert 18 < mean_block < 22

    def test_bad_arguments_raise(self):
        for args in [(0, 5, 10, 0), (10, 0, 10, 0), (10, 5, 0, 0)]:
            with pytest.raises(ValueError):
                stationary_bootstrap_indices(*args)


# --------------------------------------------------------------- bootstrap_sharpe_ci


class TestBootstrapSharpeCi:
    def test_matches_percentiles_of_resampled_sharpes(self):
        r = noisy(300)
        lo, hi = bootstrap_sharpe_ci(r, 10, 200, 0, 0.90)
        idx = stationary_bootstrap_indices(300, 10, 200, 0)
        x = r.to_numpy()[idx]
        sr = x.mean(axis=1) / x.std(axis=1, ddof=1) * SQ
        q_lo, q_hi = np.quantile(sr, [(1 - 0.90) / 2, (1 + 0.90) / 2])
        assert lo == pytest.approx(q_lo, rel=1e-12)
        assert hi == pytest.approx(q_hi, rel=1e-12)

    def test_contains_the_point_estimate(self):
        # F5.2 acceptance: the 95% interval contains the sample Sharpe.
        for seed in (1, 2, 3):
            r = noisy(2000, seed=seed)
            lo, hi = bootstrap_sharpe_ci(r, 20, 500, 0, 0.95)
            assert lo < sharpe(r) < hi

    def test_width_is_about_the_textbook_standard_error(self):
        # i.i.d. normal, T = 2520 (10 years): se(SR) ~ sqrt((1 + SR^2/2) / years) ~ 0.32,
        # so a 95% band is ~ +-0.62 wide around the estimate, ~1.24 in all.
        r = noisy(2520, mean=0.0003, seed=4)
        lo, hi = bootstrap_sharpe_ci(r, 20, 1000, 0, 0.95)
        assert 1.0 < hi - lo < 1.5

    def test_a_higher_level_is_wider(self):
        r = noisy(500)
        lo90, hi90 = bootstrap_sharpe_ci(r, 20, 400, 0, 0.90)
        lo99, hi99 = bootstrap_sharpe_ci(r, 20, 400, 0, 0.99)
        assert lo99 < lo90 < hi90 < hi99

    def test_nan_rows_dropped_and_risk_free_rate_shifts(self):
        r = noisy(300)
        with_gaps = r.copy()
        with_gaps.iloc[[5, 50]] = NA
        assert bootstrap_sharpe_ci(with_gaps, 10, 100, 0, 0.95) == bootstrap_sharpe_ci(
            with_gaps.dropna(), 10, 100, 0, 0.95
        )
        assert (
            bootstrap_sharpe_ci(r, 10, 100, 0, 0.95, 0.05)[0]
            < bootstrap_sharpe_ci(r, 10, 100, 0, 0.95)[0]
        )

    def test_seeded(self):
        r = noisy(300)
        assert bootstrap_sharpe_ci(r, 10, 100, 0, 0.95) == bootstrap_sharpe_ci(r, 10, 100, 0, 0.95)
        assert bootstrap_sharpe_ci(r, 10, 100, 0, 0.95) != bootstrap_sharpe_ci(r, 10, 100, 1, 0.95)

    def test_too_short_is_nan_and_floats(self):
        lo, hi = bootstrap_sharpe_ci(series([0.01]), 10, 50, 0, 0.95)
        assert np.isnan(lo) and np.isnan(hi)
        lo, hi = bootstrap_sharpe_ci(noisy(50), 5, 50, 0, 0.95)
        assert type(lo) is float and type(hi) is float


# -------------------------------------------------------------- probabilistic_sharpe


class TestProbabilisticSharpe:
    def test_zero_mean_series_is_one_half(self):
        # F5.2 acceptance: SR = 0 -> z = 0 -> Phi(0) = 0.5, whatever the moments.
        r = series([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
        assert probabilistic_sharpe(r) == pytest.approx(0.5, abs=1e-12)

    def test_hand_computed(self):
        # ALT8: per-row SR = 0.467707, skew 0, excess kurtosis -2.8, T = 8.
        #   variance term 1 - 0 * SR + (-2.8 + 2) / 4 * SR^2 = 1 - 0.2 * 7/32 = 0.95625
        #   z = SR * sqrt(T - 1) / sqrt(0.95625) = 0.467707 * 2.645751 / 0.977880 = 1.265428
        #   PSR = Phi(1.265428) = 0.897141
        z = ALT8_SR * np.sqrt(7) / np.sqrt(0.95625)
        assert probabilistic_sharpe(ALT8) == pytest.approx(PHI.cdf(z))
        assert probabilistic_sharpe(ALT8) == pytest.approx(0.897141, abs=1e-6)

    def test_benchmark_is_an_annualised_sharpe(self):
        # Against its own (annualised) Sharpe the probability is exactly one half.
        r = noisy(400)
        assert probabilistic_sharpe(r, benchmark_sharpe=sharpe(r)) == pytest.approx(0.5)

    def test_hand_computed_against_a_benchmark(self):
        # ALT8 against an annualised benchmark of 0.25 * sqrt(252) (per row 0.25):
        #   z = (0.467707 - 0.25) * 2.645751 / 0.977880 = 0.589028 -> Phi = 0.722079
        z = (ALT8_SR - 0.25) * np.sqrt(7) / np.sqrt(0.95625)
        assert probabilistic_sharpe(ALT8, 0.25 * SQ) == pytest.approx(PHI.cdf(z))

    def test_negative_skew_and_fat_tails_lower_a_positive_psr(self):
        # Same mean and std, worse shape -> lower PSR (the point of the higher moments).
        base = noisy(1000, mean=0.001, seed=9)
        z = (base - base.mean()) / base.std()
        crash = z.copy()
        crash.iloc[::50] = -4.0  # occasional large losses
        crash = (crash - crash.mean()) / crash.std()
        good = z * 0.01 + 0.001
        bad = crash * 0.01 + 0.001
        assert sharpe(good) == pytest.approx(sharpe(bad))
        assert probabilistic_sharpe(bad) < probabilistic_sharpe(good)

    def test_more_data_more_confidence(self):
        r = noisy(250, mean=0.001, seed=2)
        longer = series(list(r) * 4)
        assert probabilistic_sharpe(longer) > probabilistic_sharpe(r) > 0.5

    def test_risk_free_rate_and_nan_rows(self):
        r = series([0.03, NA, -0.01, 0.03, -0.01, 0.03, -0.01, 0.03, -0.01])
        assert probabilistic_sharpe(r) == pytest.approx(probabilistic_sharpe(ALT8))
        assert probabilistic_sharpe(ALT8, 0.0, 0.0252) < probabilistic_sharpe(ALT8)

    def test_undefined_is_nan(self):
        assert np.isnan(probabilistic_sharpe(series([0.01, 0.02, 0.03])))  # T < 4
        assert np.isnan(probabilistic_sharpe(series([0.25] * 10)))  # no variance (0.25 is exact)

    def test_a_float_in_the_unit_interval(self):
        p = probabilistic_sharpe(noisy(300))
        assert type(p) is float and 0.0 < p < 1.0


# --------------------------------------------------------------- expected_max_sharpe


class TestExpectedMaxSharpe:
    # E[max SR] = std * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e))),
    # gamma = 0.5772157 (Euler-Mascheroni).
    #   N = 2:  (1 - g) * 0 + g * Phi^-1(1 - 1/(2e) = 0.816060) = 0.577216 * 0.900453 = 0.519755
    #   N = 10: 0.422784 * 1.281552 + 0.577216 * Phi^-1(0.963212) = 0.541820 + 0.577216 *
    #           1.789242 = 1.574598
    def test_euler_gamma(self):
        assert EULER_GAMMA == pytest.approx(0.5772156649015329, abs=1e-15)

    def test_hand_computed(self):
        assert expected_max_sharpe(2, 1.0) == pytest.approx(0.519755, abs=1e-6)
        assert expected_max_sharpe(10, 1.0) == pytest.approx(1.574598, abs=1e-6)
        g = EULER_GAMMA
        exact = (1 - g) * PHI.inv_cdf(1 - 1 / 100) + g * PHI.inv_cdf(1 - 1 / (100 * np.e))
        assert expected_max_sharpe(100, 0.3) == pytest.approx(0.3 * exact)

    def test_one_trial_has_no_hurdle(self):
        assert expected_max_sharpe(1, 0.5) == 0.0

    def test_increasing_in_trials_and_linear_in_std(self):
        vals = [expected_max_sharpe(n, 0.4) for n in (1, 2, 10, 100, 1000)]
        assert all(a < b for a, b in zip(vals, vals[1:], strict=False))
        assert expected_max_sharpe(50, 0.6) == pytest.approx(2 * expected_max_sharpe(50, 0.3))
        assert expected_max_sharpe(50, 0.0) == 0.0

    def test_bad_arguments_raise(self):
        with pytest.raises(ValueError):
            expected_max_sharpe(0, 0.5)
        with pytest.raises(ValueError):
            expected_max_sharpe(10, -0.1)

    def test_a_float(self):
        assert type(expected_max_sharpe(10, 0.5)) is float


# ------------------------------------------------------------------- deflated_sharpe


class TestDeflatedSharpe:
    def test_is_psr_at_the_expected_max_sharpe(self):
        r = noisy(1500, mean=0.0006)
        hurdle = expected_max_sharpe(20, 0.5)
        assert deflated_sharpe(r, 20, 0.5) == pytest.approx(probabilistic_sharpe(r, hurdle))

    def test_one_trial_is_the_psr(self):
        r = noisy(1500, mean=0.0006)
        assert deflated_sharpe(r, 1, 0.5) == pytest.approx(probabilistic_sharpe(r))

    def test_decreases_with_trials(self):
        # F5.2 acceptance: the more configurations tried, the lower the deflated Sharpe.
        r = noisy(2500, mean=0.0006)
        vals = [deflated_sharpe(r, n, 0.3) for n in (1, 2, 5, 20, 192, 1000)]
        assert all(a > b for a, b in zip(vals, vals[1:], strict=False))

    def test_hand_computed(self):
        # ALT8 with N = 2 trials and an annualised trial-Sharpe std of sqrt(252) * 0.1
        # (per row 0.1): hurdle per row 0.1 * 0.519755 = 0.0519755;
        #   z = (0.467707 - 0.0519755) * 2.645751 / 0.977880 = 1.124803 -> Phi = 0.869664
        z = (ALT8_SR - 0.1 * expected_max_sharpe(2, 1.0)) * np.sqrt(7) / np.sqrt(0.95625)
        assert deflated_sharpe(ALT8, 2, 0.1 * SQ) == pytest.approx(PHI.cdf(z))


# ----------------------------------------------------------------- inference_summary


class TestInferenceSummary:
    def test_keys(self):
        assert INFERENCE_KEYS == (
            "sharpe_t_stat",
            "sharpe_ci_low",
            "sharpe_ci_high",
            "psr",
            "n_trials",
            "sharpe_hurdle",
            "dsr",
        )

    def test_bundles_the_functions_with_the_config(self):
        cfg = load_config(
            overrides={"inference.n_resamples": 300, "inference.block_length": 10}
        ).inference
        r = noisy(800, mean=0.0006)
        out = inference_summary(r, cfg, n_trials=20, trials_sharpe_std=0.4)
        assert list(out) == list(INFERENCE_KEYS)
        lo, hi = bootstrap_sharpe_ci(r, 10, 300, cfg.seed, cfg.ci_level, cfg.risk_free_rate)
        assert out["sharpe_t_stat"] == pytest.approx(sharpe_t_stat(r))
        assert (out["sharpe_ci_low"], out["sharpe_ci_high"]) == (lo, hi)
        assert out["psr"] == pytest.approx(probabilistic_sharpe(r))
        assert out["n_trials"] == 20 and type(out["n_trials"]) is int
        assert out["sharpe_hurdle"] == pytest.approx(expected_max_sharpe(20, 0.4))
        assert out["dsr"] == pytest.approx(deflated_sharpe(r, 20, 0.4))

    def test_defaults_are_one_trial(self):
        cfg = load_config(overrides={"inference.n_resamples": 50}).inference
        r = noisy(300)
        out = inference_summary(r, cfg)
        assert out["n_trials"] == 1 and out["sharpe_hurdle"] == 0.0
        assert out["dsr"] == pytest.approx(out["psr"])

    def test_risk_free_rate_comes_from_the_config(self):
        cfg = load_config(
            overrides={"inference.n_resamples": 50, "inference.risk_free_rate": 0.0252}
        ).inference
        out = inference_summary(TINY, cfg)
        assert out["sharpe_t_stat"] == pytest.approx(sharpe_t_stat(TINY, 0.0252))

    def test_config_defaults(self):
        i = load_config().inference
        assert (i.block_length, i.n_resamples, i.seed, i.ci_level) == (20, 2000, 0, 0.95)
