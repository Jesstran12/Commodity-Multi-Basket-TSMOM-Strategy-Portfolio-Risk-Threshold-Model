"""Benchmarks (PROJECT_LOG.md §4): equal-weight long-only, equal-weight vol-targeted, and
the external index, each as a return series the strategy's can be put next to.

Pure functions over wide frames (index = tz-naive daily ``DatetimeIndex``, columns =
tickers), no I/O. The two equal-weight benchmarks are **weights** frames on the panel
calendar; they run through the same engine as the strategy (``backtest.run``, same lag,
same cost model) so the comparison is like for like. The external index is a return series
on the panel calendar built from its own prints.

Conventions:

* An instrument is **eligible** for the equal-weight basket on a day when it has a return
  that day (a price today and yesterday): no lookback, no signal, so the basket starts on
  the panel's first day and grows as instruments enter. Weights are ``1 / N_eligible`` on
  the eligible names and NaN elsewhere.
* The vol-targeted basket applies §2.2 **step 2 and the cap only** (the portfolio scaling
  the strategy uses, through ``portfolio.portfolio_vol`` and ``portfolio.leverage_cap``),
  not the per-asset scaling of step 1: it is the equal-weight basket run at the strategy's
  risk, so the comparison is not just about leverage (§4.2).
* Both baskets are decided on the rebalance schedule and held (``portfolio.hold``), like
  the strategy.
* The comparison window is the strategy's life; aligning the three series to it is the
  stage's job (``cli.run_benchmarks``), not these functions'.
"""

from __future__ import annotations

import pandas as pd

from src.config import SizingConfig
from src.data import clean_returns
from src.portfolio import hold, leverage_cap, portfolio_vol, rebalance_dates

# ------------------------------------------------------------------- F4.3 benchmarks


def ew_weights(returns: pd.DataFrame, rebalance: str = "daily") -> pd.DataFrame:
    """§4.1: equal-weight long-only weights, ``1 / N_eligible(t)`` on the eligible names.

    An instrument is eligible on a row when its return there is not NaN; its weight is
    ``1 / N_eligible`` (the row's count of eligible names) and NaN where not eligible, so
    every row with at least one eligible name has a gross of exactly 1. With
    ``rebalance = "daily"`` that daily frame is returned as is; with ``"monthly"`` it is
    sampled on ``portfolio.rebalance_dates`` and held (``portfolio.hold``), NaN before the
    first rebalance date. Same index and columns as ``returns``, float dtype; the input is
    not modified. ``rebalance`` is validated by ``rebalance_dates``.
    """
    # TODO 37: implement ew_weights.
    #   ``eligible = returns.notna()``; ``n = eligible.sum(axis=1)``;
    #   ``daily = eligible.astype(float).div(n, axis=0).where(eligible)`` (``1 / n`` on the
    #   eligible cells, NaN elsewhere; a row with ``n == 0`` is all NaN already);
    #   ``hold(daily, rebalance_dates(returns.index, rebalance))`` (``hold`` on every row is
    #   the identity, so one expression covers both schedules). ``hold`` and
    #   ``rebalance_dates`` come from ``src.portfolio`` (as do ``portfolio_vol`` and
    #   ``leverage_cap`` for TODO 38); ``clean_returns`` for TODO 39 from ``src.data``.
    #   Verified by tests/test_benchmark.py::TestEwWeights and ::TestEwLongOnlyToyCase.
    eligible = returns.notna()
    n = eligible.sum(axis=1)
    daily = eligible.astype(float).div(n, axis=0).where(eligible)
    return hold(daily, rebalance_dates(returns.index, rebalance))


def ew_vol_targeted_weights(
    returns: pd.DataFrame, sizing: SizingConfig, rebalance: str
) -> tuple[pd.DataFrame, pd.Series]:
    """§4.2: the equal-weight basket scaled to the strategy's portfolio vol target.

    From the daily :func:`ew_weights` (``"daily"``) as raw weights: the pre-scaled basket's
    realised vol through ``portfolio.portfolio_vol`` (same window and estimator as the
    strategy, ``sizing.vol_window`` / ``sizing.vol_estimator``), every weight scaled by
    ``sizing.sigma_target_portfolio / portfolio_vol`` row by row (NaN where that vol is NaN
    or not ``> 0``), the gross clipped to ``sizing.leverage_cap`` through
    ``portfolio.leverage_cap``, then both the capped weights and the cap flag sampled on
    the ``rebalance`` schedule and held. The per-asset scaling of §2.2 step 1 is **not**
    applied: the basket stays equal-weight within the row.

    Returns ``(weights, cap_binds)``: ``weights`` with the index and columns of ``returns``
    (NaN before the first rebalance date on which the vol exists), ``cap_binds`` a bool
    series named ``"cap_binds"`` on the same index (False where there is no weight).
    Inputs are not modified.
    """
    # TODO 38: implement ew_vol_targeted_weights.
    #   ``raw = ew_weights(returns)``; ``pvol = portfolio_vol(raw, returns, sizing.vol_window,
    #   sizing.vol_estimator)``; ``scale = (sizing.sigma_target_portfolio / pvol).where(pvol
    #   > 0)``; ``capped, binds = leverage_cap(raw.mul(scale, axis=0), sizing.leverage_cap)``;
    #   ``dates = rebalance_dates(returns.index, rebalance)``; ``weights = hold(capped,
    #   dates)``; ``cap_binds = hold(binds.astype(float), dates).eq(1.0).rename("cap_binds")``
    #   (the float round trip from Phase 3's TODO 23).
    #   Verified by tests/test_benchmark.py::TestEwVolTargetedWeights.
    raw = ew_weights(returns)
    pvol = portfolio_vol(raw, returns, sizing.vol_window, sizing.vol_estimator)
    scale = (sizing.sigma_target_portfolio / pvol).where(pvol > 0)
    capped, binds = leverage_cap(raw.mul(scale, axis=0), sizing.leverage_cap)
    dates = rebalance_dates(returns.index, rebalance)
    weights = hold(capped, dates)
    cap_binds = hold(binds.astype(float), dates).eq(1.0).rename("cap_binds")
    return weights, cap_binds


def external_returns(prices: pd.Series, calendar: pd.DatetimeIndex, clip: float) -> pd.Series:
    """§4.3: the external index's daily returns on the panel calendar.

    The index trades on its own calendar (``data/benchmark.csv``, loaded by
    ``data.load_benchmark``). Its prints on days **outside** ``calendar`` are dropped first,
    so their move is absorbed into the next print's return; returns are then computed
    between consecutive remaining prints with ``data.clean_returns`` (the §3.3 definition
    and clip, so a bad print is tamed the same way as a commodity's); finally the series is
    placed on ``calendar``: NaN on a calendar day without a print (the next print's return
    then spans it). Nothing is lost either way: every move between two kept prints is in
    exactly one row. NaN before the index's first print and after its last.

    Returns a float series on ``calendar`` named after ``prices`` (its ticker). The input is
    not modified; NaN prices are ignored.
    """
    # TODO 39: implement external_returns.
    #   ``p = prices.dropna()``; ``p = p[p.index.isin(calendar)]``;
    #   ``r, _ = clean_returns(p.to_frame(), clip)`` (a one-column frame; ``clean_returns``
    #   wants a frame); ``r.iloc[:, 0].reindex(calendar).rename(prices.name)``. Do not
    #   ``reindex`` the **prices** onto the calendar before differencing: a missing print
    #   would then cost two returns (the day and the day after) instead of none.
    #   Verified by tests/test_benchmark.py::TestExternalReturns and ::TestSnapshot.
    p = prices.dropna()
    p = p[p.index.isin(calendar)]
    r, _ = clean_returns(p.to_frame(), clip)
    return r.iloc[:, 0].reindex(calendar).rename(prices.name)


__all__ = ["ew_vol_targeted_weights", "ew_weights", "external_returns"]
