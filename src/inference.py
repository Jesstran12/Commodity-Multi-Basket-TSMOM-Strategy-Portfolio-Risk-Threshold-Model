"""Statistical inference on a strategy's Sharpe ratio (PROJECT_LOG.md §5.2).

A Sharpe ratio estimated from 25 years of daily data is still an estimate. This module puts
error bars and probabilities on it:

* :func:`sharpe_t_stat`: ``SR * sqrt(T / 252)``, the Sharpe ratio's t-statistic.
* :func:`stationary_bootstrap_indices` and :func:`bootstrap_sharpe_ci`: a confidence interval
  from the stationary block bootstrap of Politis and Romano (1994), which resamples blocks of
  random (geometric) length so that volatility clustering survives the resampling.
* :func:`probabilistic_sharpe`: the Probabilistic Sharpe Ratio of Bailey and López de Prado
  (2012), the probability that the true Sharpe exceeds a benchmark, given the sample
  length, skew and kurtosis.
* :func:`expected_max_sharpe` and :func:`deflated_sharpe`: the Deflated Sharpe Ratio of
  Bailey and López de Prado (2014), the PSR measured against the Sharpe ratio the best of
  ``N`` unskilled trials would reach by luck.

Conventions, shared by every function here:

* The input is a daily return series; **NaN rows are dropped first**, as in
  ``src/metrics.py``, and ``T`` is the number of rows left.
* A Sharpe ratio passed in or returned is **annualised** (``ANNUALISATION`` = 252, the
  risk-free rate an annual rate spread evenly over the year). The PSR formula works on the
  per-row Sharpe, ``SR / sqrt(252)``; the conversion happens inside.
* Skew and kurtosis are :func:`src.metrics.skew` and :func:`src.metrics.kurtosis`
  (bias-adjusted, the kurtosis in excess of the normal's 3).
* The bootstrap is seeded: the same seed gives the same draws, so every number here is
  reproducible to the last digit.
* Where a statistic is undefined the result is ``float("nan")``, never an exception; bad
  arguments (a non-positive count) raise ``ValueError``.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.config import ANNUALISATION, InferenceConfig
from src.metrics import _clean, kurtosis, sharpe, skew

EULER_GAMMA = float(np.euler_gamma)
"""The Euler-Mascheroni constant, 0.5772..., in the expected maximum of ``N`` Sharpe ratios."""

INFERENCE_KEYS = (
    "sharpe_t_stat",
    "sharpe_ci_low",
    "sharpe_ci_high",
    "psr",
    "n_trials",
    "sharpe_hurdle",
    "dsr",
)
"""Keys of :func:`inference_summary`, in order."""


# --------------------------------------------------------------------------- t-stat


def sharpe_t_stat(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """The Sharpe ratio's t-statistic: ``sharpe(returns, rf) * sqrt(T / 252)``.

    ``T`` is the number of non-NaN rows; the result equals the per-row Sharpe times
    ``sqrt(T)``. NaN where the Sharpe ratio is.
    """
    # TODO 49: implement sharpe_t_stat.
    #   ``n = len(_clean(returns))``; ``float(sharpe(returns, risk_free_rate) * np.sqrt(n /
    #   ANNUALISATION))`` (a NaN Sharpe stays NaN through the product).
    #   The stub imports only what its signatures need. Imports to add as you go:
    #   ``ANNUALISATION`` (``src.config``); ``_clean``, ``sharpe``, ``skew``, ``kurtosis``
    #   (``src.metrics``; ``_clean`` is the shared NaN-dropping helper there); ``NormalDist``
    #   (the standard library's ``statistics``) for TODOs 52-53. ``make format`` sorts them.
    #   Verified by tests/test_inference.py::TestSharpeTStat.
    n = len(_clean(returns))
    return float(sharpe(returns, risk_free_rate) * np.sqrt(n / ANNUALISATION))


# ------------------------------------------------------------------------ bootstrap


def stationary_bootstrap_indices(
    n: int, block_length: int, n_resamples: int, seed: int
) -> np.ndarray:
    """Row indices of ``n_resamples`` stationary-bootstrap resamples of a length-``n`` series.

    Returns an integer array of shape ``(n_resamples, n)`` with values in ``[0, n)``. Each
    resample is a chain of blocks: a block starts at a uniformly random row and runs forward
    one row at a time, wrapping from ``n - 1`` back to ``0``; at every step after the first a
    new block starts with probability ``p = 1 / block_length``, so block lengths are
    geometric with mean ``block_length``.

    **The draw order is part of the contract** (it fixes every index for a given seed): one
    generator ``rng = np.random.default_rng(seed)``; then, for each resample in turn,
    ``starts = rng.integers(0, n, size=n)`` followed by ``u = rng.random(n)``. Row ``t = 0``
    takes ``starts[0]``; row ``t >= 1`` takes ``starts[t]`` if ``u[t] < p``, otherwise the
    previous row's index plus one, modulo ``n``.

    Raises ``ValueError`` unless ``n``, ``block_length`` and ``n_resamples`` are all ``>= 1``.
    """
    # TODO 50: implement stationary_bootstrap_indices.
    #   Validate the three counts. ``rng = np.random.default_rng(seed)``; ``p = 1.0 /
    #   block_length``; ``pos = np.arange(n)``; ``out = np.empty((n_resamples, n),
    #   dtype=np.int64)``. For each resample (a Python loop over ``n_resamples`` is fine;
    #   keep the two draws in the documented order): ``starts = rng.integers(0, n, size=n)``;
    #   ``new = rng.random(n) < p``; ``new[0] = True``. The row where each row's block began:
    #   ``begin = np.maximum.accumulate(np.where(new, pos, 0))``; then ``out[k] =
    #   (starts[begin] + pos - begin) % n``. (A loop over rows works too, but is ~1000x
    #   slower on the snapshot's 6,410 rows x 2,000 resamples.)
    #   Verified by tests/test_inference.py::TestStationaryBootstrapIndices.
    if n < 1 or block_length < 1 or n_resamples < 1:
        raise ValueError("n, block_length and n_resamples must all be >= 1")
    rng = np.random.default_rng(seed)
    p = 1.0 / block_length
    pos = np.arange(n)
    out = np.empty((n_resamples, n), dtype=np.int64)
    for k in range(n_resamples):
        starts = rng.integers(0, n, size=n)
        new = rng.random(n) < p
        new[0] = True
        begin = np.maximum.accumulate(np.where(new, pos, 0))
        out[k] = (starts[begin] + pos - begin) % n
    return out


def bootstrap_sharpe_ci(
    returns: pd.Series,
    block_length: int,
    n_resamples: int,
    seed: int,
    ci_level: float,
    risk_free_rate: float = 0.0,
) -> tuple[float, float]:
    """Percentile confidence interval for the annualised Sharpe ratio, stationary bootstrap.

    The excess returns (non-NaN rows, minus ``risk_free_rate / 252``) are resampled with
    :func:`stationary_bootstrap_indices` (``n`` = their count, and the other three
    arguments as given); each resample's Sharpe ratio is ``mean / std(ddof=1) * sqrt(252)``
    over its row; resamples whose std is zero are dropped. The interval is
    ``np.quantile(sharpes, [(1 - ci_level) / 2, (1 + ci_level) / 2])`` (numpy's default
    linear interpolation), returned as ``(low, high)`` floats.

    ``(nan, nan)`` with fewer than two rows or no resample with a finite Sharpe.
    """
    # TODO 51: implement bootstrap_sharpe_ci.
    #   ``x = _clean(returns).to_numpy() - risk_free_rate / ANNUALISATION``; ``(nan, nan)`` if
    #   ``len(x) < 2``; ``sample = x[stationary_bootstrap_indices(len(x), block_length,
    #   n_resamples, seed)]`` (fancy indexing gives the ``(n_resamples, n)`` matrix);
    #   ``mean = sample.mean(axis=1)``, ``sd = sample.std(axis=1, ddof=1)``; keep ``sd > 0``;
    #   ``sr = mean / sd * np.sqrt(ANNUALISATION)``; ``lo, hi = np.quantile(sr, [...])``;
    #   ``(float(lo), float(hi))``.
    #   Verified by tests/test_inference.py::TestBootstrapSharpeCi.
    x = _clean(returns).to_numpy() - risk_free_rate / ANNUALISATION
    if len(x) < 2:
        return float("nan"), float("nan")
    sample = x[stationary_bootstrap_indices(len(x), block_length, n_resamples, seed)]
    mean = sample.mean(axis=1)
    sd = sample.std(axis=1, ddof=1)
    keep = sd > 0
    sr = mean[keep] / sd[keep] * np.sqrt(ANNUALISATION)
    if len(sr) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(sr, [(1 - ci_level) / 2, (1 + ci_level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------- PSR and DSR


def probabilistic_sharpe(
    returns: pd.Series, benchmark_sharpe: float = 0.0, risk_free_rate: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio: the probability that the true Sharpe ratio exceeds
    ``benchmark_sharpe`` (annualised), given the sample.

    With ``sr = sharpe(returns, rf) / sqrt(252)`` and ``sr_star = benchmark_sharpe /
    sqrt(252)`` (per row), ``T`` rows, ``g3 = skew(returns)`` and ``g4 = kurtosis(returns)``
    (excess):

        PSR = Phi( (sr - sr_star) * sqrt(T - 1) / sqrt(1 - g3 * sr + (g4 + 2) / 4 * sr**2) )

    with ``Phi`` the standard normal CDF (``statistics.NormalDist().cdf``). This is Bailey
    and López de Prado's formula, whose ``(gamma_4 - 1) / 4`` term uses the plain kurtosis
    ``gamma_4 = g4 + 3``. NaN with fewer than four rows, an undefined Sharpe, skew or
    kurtosis (zero variance), or a variance term that is not positive.
    """
    # TODO 52: implement probabilistic_sharpe.
    #   ``n = len(_clean(returns))``; NaN if ``n < 4``; ``sr = sharpe(returns,
    #   risk_free_rate) / np.sqrt(ANNUALISATION)``; ``g3, g4 = skew(returns),
    #   kurtosis(returns)`` (the skew of the excess returns is the skew of the returns: a
    #   constant shift changes neither moment); NaN if any of the three is NaN;
    #   ``var = 1 - g3 * sr + (g4 + 2) / 4 * sr**2``; NaN unless ``var > 0``;
    #   ``z = (sr - benchmark_sharpe / np.sqrt(ANNUALISATION)) * np.sqrt(n - 1) / np.sqrt(var)``;
    #   ``float(NormalDist().cdf(z))`` (``NormalDist`` from ``statistics``; there is no scipy
    #   in this project and none is needed).
    #   Verified by tests/test_inference.py::TestProbabilisticSharpe.
    n = len(_clean(returns))
    if n < 4:
        return float("nan")
    sr = sharpe(returns, risk_free_rate) / np.sqrt(ANNUALISATION)
    g3, g4 = skew(returns), kurtosis(returns)
    if np.isnan(sr) or np.isnan(g3) or np.isnan(g4):
        return float("nan")
    var = 1 - g3 * sr + (g4 + 2) / 4 * sr**2
    if not var > 0:
        return float("nan")
    z = (sr - benchmark_sharpe / np.sqrt(ANNUALISATION)) * np.sqrt(n - 1) / np.sqrt(var)
    return float(NormalDist().cdf(z))


def expected_max_sharpe(n_trials: int, trials_sharpe_std: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` independent unskilled trials.

    ``E[max SR] = trials_sharpe_std * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))``
    with ``g = EULER_GAMMA``, ``e = exp(1)`` and ``Phi^-1`` the standard normal inverse CDF
    (``statistics.NormalDist().inv_cdf``). ``trials_sharpe_std`` is the standard deviation
    of the trials' Sharpe ratios; the result is in the same units (annualised in, annualised
    out). ``0.0`` for a single trial (one trial has nothing to be lucky against; the formula
    itself is undefined at ``N = 1``).

    Raises ``ValueError`` if ``n_trials < 1`` or ``trials_sharpe_std < 0``.
    """
    # TODO 53: implement expected_max_sharpe.
    #   Validate; ``0.0`` if ``n_trials == 1``; ``inv = NormalDist().inv_cdf``; ``float(
    #   trials_sharpe_std * ((1 - EULER_GAMMA) * inv(1 - 1 / n_trials) + EULER_GAMMA *
    #   inv(1 - 1 / (n_trials * np.e))))``.
    #   Verified by tests/test_inference.py::TestExpectedMaxSharpe.
    if n_trials < 1 or trials_sharpe_std < 0:
        raise ValueError("n_trials must be >= 1 and trials_sharpe_std must be >= 0")
    if n_trials == 1:
        return 0.0
    inv = NormalDist().inv_cdf
    return float(
        trials_sharpe_std
        * ((1 - EULER_GAMMA) * inv(1 - 1 / n_trials) + EULER_GAMMA * inv(1 - 1 / (n_trials * np.e)))
    )


def deflated_sharpe(
    returns: pd.Series, n_trials: int, trials_sharpe_std: float, risk_free_rate: float = 0.0
) -> float:
    """Deflated Sharpe Ratio: :func:`probabilistic_sharpe` against the Sharpe ratio the best
    of ``n_trials`` unskilled trials is expected to reach, :func:`expected_max_sharpe`.

    ``trials_sharpe_std`` is the standard deviation of the **annualised** Sharpe ratios of
    the trials (the robustness grid's, in Phase 6). With one trial the hurdle is zero and the
    DSR equals the PSR against zero. The more trials, the higher the hurdle and the lower
    the DSR.
    """
    # TODO 54: implement deflated_sharpe.
    #   ``hurdle = expected_max_sharpe(n_trials, trials_sharpe_std)``; return
    #   ``probabilistic_sharpe(returns, hurdle, risk_free_rate)``.
    #   Verified by tests/test_inference.py::TestDeflatedSharpe.
    hurdle = expected_max_sharpe(n_trials, trials_sharpe_std)
    return probabilistic_sharpe(returns, hurdle, risk_free_rate)


def inference_summary(
    returns: pd.Series,
    inference: InferenceConfig,
    n_trials: int = 1,
    trials_sharpe_std: float = 0.0,
) -> dict[str, object]:
    """The §5.2 statistics of one series, as a dict with keys exactly ``INFERENCE_KEYS``.

    ``sharpe_t_stat`` (:func:`sharpe_t_stat`); ``sharpe_ci_low`` and ``sharpe_ci_high``
    (:func:`bootstrap_sharpe_ci` with ``inference.block_length``, ``n_resamples``, ``seed``
    and ``ci_level``); ``psr`` (:func:`probabilistic_sharpe` against zero); ``n_trials``
    (int, as given); ``sharpe_hurdle`` (:func:`expected_max_sharpe`, annualised); ``dsr``
    (:func:`deflated_sharpe`). Every statistic uses ``inference.risk_free_rate``.

    The defaults (one trial, zero spread) are the honest values before the robustness grid
    exists: the hurdle is zero and the DSR equals the PSR. Phase 6 passes the grid's trial
    count and the standard deviation of its Sharpe ratios.
    """
    # TODO 55: implement inference_summary.
    #   ``rf = inference.risk_free_rate``; ``lo, hi = bootstrap_sharpe_ci(returns,
    #   inference.block_length, inference.n_resamples, inference.seed, inference.ci_level,
    #   rf)``; build the dict in ``INFERENCE_KEYS`` order with ``int(n_trials)``.
    #   Verified by tests/test_inference.py::TestInferenceSummary.
    rf = inference.risk_free_rate
    lo, hi = bootstrap_sharpe_ci(
        returns,
        inference.block_length,
        inference.n_resamples,
        inference.seed,
        inference.ci_level,
        rf,
    )
    return {
        "sharpe_t_stat": sharpe_t_stat(returns, rf),
        "sharpe_ci_low": lo,
        "sharpe_ci_high": hi,
        "psr": probabilistic_sharpe(returns, 0.0, rf),
        "n_trials": int(n_trials),
        "sharpe_hurdle": expected_max_sharpe(n_trials, trials_sharpe_std),
        "dsr": deflated_sharpe(returns, n_trials, trials_sharpe_std, rf),
    }


__all__ = [
    "EULER_GAMMA",
    "INFERENCE_KEYS",
    "bootstrap_sharpe_ci",
    "deflated_sharpe",
    "expected_max_sharpe",
    "inference_summary",
    "probabilistic_sharpe",
    "sharpe_t_stat",
    "stationary_bootstrap_indices",
]
