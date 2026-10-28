"""Performance metrics on a daily return series (PROJECT_LOG.md §5.1).

Phase 4 delivered the five statistics the backtest report and the §8 known-answer test need:
annualised return, annualised volatility, Sharpe ratio, Sortino ratio and maximum drawdown,
plus :func:`performance`, which bundles them for one series. Phase 5 adds the rest of the
return-series statistics of §5.1 (monthly returns, maximum drawdown duration, Calmar, daily
and monthly hit rates, skew, excess kurtosis, best and worst month) and :func:`summarise`,
which bundles all of them. The book statistics of §5.1 (turnover, leverage, cap binding,
``N_live``) describe positions, not a return series; they come from the backtest and the
sizing results and are assembled in the metrics stage (``src/cli.py``). The inference
statistics of §5.2 live in ``src/inference.py``.

Conventions, shared by every function here:

* The input is a daily return series (a ``pd.Series`` indexed by date). **NaN rows are
  dropped first**: a strategy's return is NaN before its first position, and a benchmark's
  on days it has no print, and neither is a zero return.
* ``ANNUALISATION`` (252) is the only annualisation factor; the risk-free rate is an annual
  rate and is spread evenly over the year (``risk_free_rate / ANNUALISATION`` per day).
* Where a statistic is undefined (an empty series, a zero standard deviation, no downside
  day) the result is ``float("nan")``, never an exception, so a table of many series can
  always be built.
* Every result is a plain ``float``, except counts (``int``) and the monthly series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANNUALISATION

PERFORMANCE_KEYS = (
    "n_days",
    "first_date",
    "last_date",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
)
"""Keys of :func:`performance`, in order."""

METRIC_KEYS = PERFORMANCE_KEYS + (
    "max_drawdown_duration",
    "calmar",
    "hit_rate_daily",
    "hit_rate_monthly",
    "skew",
    "kurtosis",
    "best_month",
    "worst_month",
)
"""Keys of :func:`summarise`, in order: ``PERFORMANCE_KEYS`` then the Phase 5 statistics."""


def _clean(returns: pd.Series) -> pd.Series:
    """The non-NaN rows as floats (shared by every metric)."""
    return returns.dropna().astype(float)


# --------------------------------------------------------------------- F5.1 (slice)


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return: ``prod(1 + r) ** (ANNUALISATION / n) - 1``.

    ``n`` is the number of non-NaN rows, so the number is the constant annual rate that
    compounds to the same final wealth over the sample. NaN for an empty series; ``-1.0``
    if the wealth is wiped out (``prod(1 + r) <= 0``).
    """
    # TODO 31: implement annualised_return.
    #   ``r = _clean(returns)``; ``n = len(r)``; NaN if ``n == 0``;
    #   ``wealth = float((1 + r).prod())``; ``-1.0`` if ``wealth <= 0``; else
    #   ``wealth ** (ANNUALISATION / n) - 1``. Wrap the result in ``float(...)``.
    #   Verified by tests/test_metrics.py::TestAnnualisedReturn and
    #   ::TestIntegrityMetricsKnownAnswer.
    r = _clean(returns)
    n = len(r)
    if n == 0:
        return float("nan")
    wealth = float((1 + r).prod())
    if wealth <= 0:
        return -1.0
    return float(wealth ** (ANNUALISATION / n) - 1)


def annualised_vol(returns: pd.Series) -> float:
    """Annualised volatility: sample standard deviation (``ddof = 1``) times ``sqrt(252)``.

    NaN with fewer than two non-NaN rows.
    """
    # TODO 32: implement annualised_vol.
    #   ``r = _clean(returns)``; NaN if ``len(r) < 2``; ``float(r.std() * np.sqrt(ANNUALISATION))``
    #   (pandas' default ``ddof=1`` is the contract). The stub imports only what its
    #   signatures need: add ``import numpy as np`` here and keep ``make lint`` clean.
    #   Verified by tests/test_metrics.py::TestAnnualisedVol and
    #   ::TestIntegrityMetricsKnownAnswer.
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(ANNUALISATION))


def sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio: ``mean(excess) / std(excess, ddof=1) * sqrt(252)``.

    ``excess = r - risk_free_rate / ANNUALISATION`` row by row (an annual rate spread evenly
    over the year; the default ``0.0`` is the project's stated convention, §5.1). NaN with
    fewer than two rows or a zero standard deviation.
    """
    # TODO 33: implement sharpe.
    #   ``excess = _clean(returns) - risk_free_rate / ANNUALISATION``; NaN if ``len(excess) < 2``;
    #   ``sd = excess.std()``; NaN if ``sd == 0`` (or not finite); else
    #   ``float(excess.mean() / sd * np.sqrt(ANNUALISATION))``.
    #   Verified by tests/test_metrics.py::TestSharpe and ::TestIntegrityMetricsKnownAnswer.
    excess = _clean(returns) - risk_free_rate / ANNUALISATION
    if len(excess) < 2:
        return float("nan")
    sd = excess.std()
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ANNUALISATION))


def sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio: ``mean(excess) / downside * sqrt(252)``.

    ``excess`` as in :func:`sharpe`; ``downside`` is the target downside deviation
    ``sqrt(mean(min(excess, 0) ** 2))`` over **all** rows (the positive days count as zero
    deviations and stay in the denominator's count). NaN for an empty series or when no
    row is below the target (no downside, no ratio).
    """
    # TODO 34: implement sortino.
    #   ``excess = _clean(returns) - risk_free_rate / ANNUALISATION``; NaN if empty;
    #   ``downside = float(np.sqrt((excess.clip(upper=0.0) ** 2).mean()))``; NaN if
    #   ``downside == 0``; else ``float(excess.mean() / downside * np.sqrt(ANNUALISATION))``.
    #   Verified by tests/test_metrics.py::TestSortino and ::TestIntegrityMetricsKnownAnswer.
    excess = _clean(returns) - risk_free_rate / ANNUALISATION
    if len(excess) == 0:
        return float("nan")
    downside = float(np.sqrt((excess.clip(upper=0.0) ** 2).mean()))
    if downside == 0:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(ANNUALISATION))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the compounded wealth, as a non-positive fraction.

    ``wealth = cumprod(1 + r)`` starting from 1; ``peak`` is the running maximum of the
    wealth **with the starting wealth of 1 counted as a peak** (a loss on the first day is
    a drawdown); ``drawdown = wealth / peak - 1``; the result is its minimum, ``0.0`` for a
    series that never falls below a previous peak, NaN for an empty series. A drawdown of
    25% is reported as ``-0.25``.
    """
    # TODO 35: implement max_drawdown.
    #   ``r = _clean(returns)``; NaN if empty; ``wealth = (1 + r).cumprod()``;
    #   ``peak = wealth.cummax().clip(lower=1.0)`` (the clip is the starting wealth);
    #   ``float((wealth / peak - 1).min())``.
    #   Verified by tests/test_metrics.py::TestMaxDrawdown and ::TestIntegrityMetricsKnownAnswer.
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    wealth = (1 + r).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return float((wealth / peak - 1).min())


def performance(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, object]:
    """The five statistics of one return series, plus its span, as a dict.

    Keys exactly ``PERFORMANCE_KEYS`` in order: ``n_days`` (number of non-NaN rows, int),
    ``first_date`` and ``last_date`` (first and last non-NaN row, ``Timestamp``; ``None``
    when there is none), ``ann_return``, ``ann_vol``, ``sharpe``, ``sortino`` and
    ``max_drawdown`` (the functions above, floats; NaN where undefined).
    """
    # TODO 36: implement performance.
    #   ``r = _clean(returns)``; the two dates are ``r.index[0]`` / ``r.index[-1]`` when
    #   ``len(r)`` else ``None``; the five statistics are the functions above on ``returns``
    #   (they drop NaN themselves). Build the dict in ``PERFORMANCE_KEYS`` order with
    #   ``int(len(r))`` for the count.
    #   Verified by tests/test_metrics.py::TestPerformance.
    r = _clean(returns)
    first_date = r.index[0] if len(r) else None
    last_date = r.index[-1] if len(r) else None
    return {
        "n_days": int(len(r)),
        "first_date": first_date,
        "last_date": last_date,
        "ann_return": annualised_return(returns),
        "ann_vol": annualised_vol(returns),
        "sharpe": sharpe(returns, risk_free_rate),
        "sortino": sortino(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
    }


# ------------------------------------------------------------------------ F5.1 (rest)


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Calendar-month returns: the daily returns compounded within each calendar month.

    NaN rows are dropped first; each month that has at least one row gives
    ``prod(1 + r) - 1`` over its rows, labelled with the date of its **last** row (a
    ``DatetimeIndex``, so a partial first or last month of the sample is kept and labelled
    where it ends). A month with no row is absent, never a zero. The result keeps the
    input's name and index name; empty in, empty out.
    """
    # TODO 42: implement monthly_returns.
    #   ``r = _clean(returns)``; ``month = r.index.to_period("M")`` (one label per calendar
    #   month, years kept apart); ``out = (1 + r).groupby(month).prod() - 1``; relabel with
    #   each month's last date: ``last = r.index.to_series().groupby(month).max()`` and
    #   ``pd.Series(out.to_numpy(), index=pd.DatetimeIndex(last.to_numpy(),
    #   name=r.index.name), name=returns.name)``. Do not use ``resample``: it creates the
    #   months that have no rows, and their empty product is a fake 0% month.
    #   Verified by tests/test_metrics.py::TestMonthlyReturns.
    r = _clean(returns)
    month = r.index.to_period("M")
    out = (1 + r).groupby(month).prod() - 1
    last = r.index.to_series().groupby(month).max()
    return pd.Series(
        out.to_numpy(),
        index=pd.DatetimeIndex(last.to_numpy(), name=r.index.name),
        name=returns.name,
    )


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest drawdown spell, in rows (trading days): the longest run of consecutive
    non-NaN rows on which the wealth is **strictly below** its running peak.

    The wealth and the peak are those of :func:`max_drawdown` (the starting wealth of 1
    counts as a peak). A row back at or above the peak ends the spell; a spell still open
    on the last row is counted up to it. ``0`` when the wealth never falls below a peak,
    an empty series included.
    """
    # TODO 43: implement max_drawdown_duration.
    #   ``r = _clean(returns)``; ``0`` if empty; ``wealth = (1 + r).cumprod()``; ``peak =
    #   wealth.cummax().clip(lower=1.0)``; ``under = wealth < peak`` (a bool series).
    #   Runs of True: ``spell = (~under).cumsum()`` is constant along each run, so
    #   ``under.groupby(spell).sum().max()`` is the longest one. Return ``int(...)``.
    #   Verified by tests/test_metrics.py::TestMaxDrawdownDuration.
    r = _clean(returns)
    if r.empty:
        return 0
    wealth = (1 + r).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    under = wealth < peak
    spell = (~under).cumsum()
    return int(under.groupby(spell).sum().max())


def calmar(returns: pd.Series) -> float:
    """Calmar ratio: :func:`annualised_return` over the magnitude of :func:`max_drawdown`.

    ``ann_return / -max_drawdown``. NaN when the maximum drawdown is ``0`` (nothing to
    divide by) or NaN (empty series).
    """
    # TODO 44: implement calmar.
    #   ``dd = max_drawdown(returns)``; NaN unless ``dd < 0`` (a NaN fails the comparison
    #   too); else ``float(annualised_return(returns) / -dd)``.
    #   Verified by tests/test_metrics.py::TestCalmar.
    dd = max_drawdown(returns)
    if not dd < 0:
        return float("nan")
    return float(annualised_return(returns) / -dd)


def hit_rate(returns: pd.Series) -> float:
    """Share of non-NaN rows with a **strictly positive** return (a zero is not a hit).

    Applied to daily returns it is the daily hit rate; applied to :func:`monthly_returns`
    it is the monthly one. NaN for an empty series.
    """
    # TODO 45: implement hit_rate.
    #   ``r = _clean(returns)``; NaN if empty; ``float((r > 0).mean())``.
    #   Verified by tests/test_metrics.py::TestHitRate.
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def skew(returns: pd.Series) -> float:
    """Sample skewness, bias-adjusted (pandas ``Series.skew``: the adjusted Fisher-Pearson
    coefficient ``G1``).

    NaN with fewer than three non-NaN rows or a zero variance (no spread, no shape).
    """
    # TODO 46: implement skew.
    #   ``r = _clean(returns)``; NaN if ``len(r) < 3`` or ``r.std() == 0``; else
    #   ``float(r.skew())``. pandas returns 0.0, not NaN, for a constant series, so the
    #   variance check is yours to make.
    #   Verified by tests/test_metrics.py::TestSkewAndKurtosis.
    r = _clean(returns)
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    return float(r.skew())


def kurtosis(returns: pd.Series) -> float:
    """Sample **excess** kurtosis, bias-adjusted (pandas ``Series.kurt``, ``G2``; a normal
    distribution scores 0).

    NaN with fewer than four non-NaN rows or a zero variance.
    """
    # TODO 47: implement kurtosis.
    #   As TODO 46 with ``len(r) < 4`` and ``r.kurt()``.
    #   Verified by tests/test_metrics.py::TestSkewAndKurtosis.
    r = _clean(returns)
    if len(r) < 4 or r.std() == 0:
        return float("nan")
    return float(r.kurt())


def summarise(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, object]:
    """Every return-series statistic of §5.1 for one series, as a dict.

    Keys exactly ``METRIC_KEYS`` in order: the :func:`performance` keys and values, then
    ``max_drawdown_duration`` (int, rows), ``calmar``, ``hit_rate_daily`` (of the daily
    rows), ``hit_rate_monthly`` (of :func:`monthly_returns`), ``skew``, ``kurtosis``
    (excess), ``best_month`` and ``worst_month`` (the largest and smallest monthly return;
    NaN when there is no month). Floats are plain ``float``; NaN where undefined.
    """
    # TODO 48: implement summarise.
    #   ``out = performance(returns, risk_free_rate)`` (a dict already in order); ``months =
    #   monthly_returns(returns)``; add the eight keys in ``METRIC_KEYS`` order with the
    #   functions above; ``best_month = float(months.max()) if len(months) else
    #   float("nan")``, likewise ``worst_month`` with ``min``.
    #   Verified by tests/test_metrics.py::TestSummarise.
    out = performance(returns, risk_free_rate)
    months = monthly_returns(returns)
    out["max_drawdown_duration"] = max_drawdown_duration(returns)
    out["calmar"] = calmar(returns)
    out["hit_rate_daily"] = hit_rate(returns)
    out["hit_rate_monthly"] = hit_rate(months)
    out["skew"] = skew(returns)
    out["kurtosis"] = kurtosis(returns)
    out["best_month"] = float(months.max()) if len(months) else float("nan")
    out["worst_month"] = float(months.min()) if len(months) else float("nan")
    return out


__all__ = [
    "METRIC_KEYS",
    "PERFORMANCE_KEYS",
    "annualised_return",
    "annualised_vol",
    "calmar",
    "hit_rate",
    "kurtosis",
    "max_drawdown",
    "max_drawdown_duration",
    "monthly_returns",
    "performance",
    "sharpe",
    "skew",
    "sortino",
    "summarise",
]
