"""Position sizing (PROJECT_LOG.md §2.2–§2.3): vol estimators, per-asset scaling, rebalance
schedule, portfolio vol scaling and the leverage cap.

Pure functions over wide frames (index = tz-naive daily ``DatetimeIndex``, columns =
tickers), no I/O. Inputs are the signal from ``signals.tsmom`` and the cleaned daily returns
from ``data.clean_returns``, on the same calendar and columns. Everything here is computed
from the close of the row it sits on; the execution lag (``positions = weights.shift(lag)``)
is Phase 4's job and is not applied here.

The pipeline, in :func:`size_positions`:

1. ``vol = realised_vol(returns)``: annualised realised vol per instrument.
2. ``raw, n_live = asset_weights(signal, vol)``: each live instrument scaled to
   ``sigma_target_asset`` and divided by the number of live instruments (§2.2 step 1).
3. ``portfolio_vol(raw, returns)``: realised vol of the pre-scaled book, and
   ``raw * sigma_target_portfolio / portfolio_vol`` (§2.2 step 2).
4. ``leverage_cap``: gross exposure clipped to ``leverage_cap`` (§2.2 step 3).
5. ``hold`` on the ``rebalance_dates``: weights decided on rebalance days only and held
   constant, in weight terms, until the next one (§2.3).

Row conventions: one row is one trading day on the panel calendar, so "``window`` days" is
"``window`` rows". A weight is NaN wherever no position can be sized (no signal, no vol
estimate, or no portfolio vol estimate yet); it is ``0`` where the signal is exactly flat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ANNUALISATION, REBALANCES, VOL_ESTIMATORS, SizingConfig

SUMMARY_KEYS = (
    "n_rebalance_dates",
    "first_weight_date",
    "n_days_with_position",
    "mean_n_live",
    "mean_gross_leverage",
    "max_gross_leverage",
    "n_cap_bind_days",
    "cap_bind_fraction",
    "n_unsizable",
)
"""Keys of the sizing summary, in order."""


@dataclass(frozen=True)
class SizingResult:
    """Everything :func:`size_positions` computes, daily, on the input calendar.

    * ``vol``: annualised realised vol per instrument (frame, NaN warm-up).
    * ``raw_weights``: §2.2 step-1 weights per instrument (frame; NaN where not live).
    * ``n_live``: number of live instruments per day (int series named ``"n_live"``).
    * ``portfolio_vol``: annualised realised vol of the pre-scaled book (float series named
      ``"portfolio_vol"``; NaN warm-up).
    * ``daily_weights``: the fully scaled and capped weights as they would be decided on
      every day (frame), before the rebalance schedule is applied.
    * ``weights``: ``daily_weights`` sampled on ``rebalance_dates`` and held (frame). This
      is the frame Phase 4 turns into positions.
    * ``cap_binds``: whether the leverage cap bound on the held weights (bool series named
      ``"cap_binds"``; False on days without a position).
    * ``rebalance_dates``: the schedule used (``DatetimeIndex``, a subset of the calendar).
    """

    vol: pd.DataFrame
    raw_weights: pd.DataFrame
    n_live: pd.Series
    portfolio_vol: pd.Series
    daily_weights: pd.DataFrame
    weights: pd.DataFrame
    cap_binds: pd.Series
    rebalance_dates: pd.DatetimeIndex


# ------------------------------------------------------------------- F3.1 vol estimators


def realised_vol(returns: pd.DataFrame, window: int, estimator: str = "rolling") -> pd.DataFrame:
    """Annualised realised volatility of daily returns, per column.

    ``estimator = "rolling"``: sample standard deviation (``ddof = 1``) of the trailing
    ``window`` rows ending on and including the current row, times ``sqrt(ANNUALISATION)``.
    The **full window must be valid**: NaN on the first ``window - 1`` rows of the frame and
    wherever any of the trailing ``window`` returns is NaN, so a gap of ``g`` NaN return rows
    costs ``g + window - 1`` vol rows (the gap and the ``window - 1`` rows after it).

    ``estimator = "ewma"``: exponentially weighted standard deviation with ``span = window``
    (pandas ``ewm(span=window, adjust=True, ignore_na=False).std(bias=False)``), times
    ``sqrt(ANNUALISATION)``. NaN until ``window`` non-NaN returns have been seen in the
    column, and NaN on every row whose own return is NaN (there is no price to size on that
    day); a gap otherwise only decays the older observations' weights.

    Same index and columns as ``returns``, float dtype; ``returns`` is not modified. Raises
    ``ValueError`` if ``window < 2`` or ``estimator`` is not in ``VOL_ESTIMATORS``.
    """
    # TODO 17: implement realised_vol.
    #   1. Validate ``window >= 2`` and ``estimator in VOL_ESTIMATORS`` (``ValueError``).
    #   2. rolling: ``returns.rolling(window, min_periods=window).std()`` (pandas' default
    #      ``ddof=1`` is the contract; ``min_periods=window`` gives the full-window rule).
    #      ewma: ``returns.ewm(span=window, min_periods=window).std()`` (the other keyword
    #      defaults are the contract), then ``.where(returns.notna())`` so a row with no
    #      return has no vol.
    #   3. Multiply by ``np.sqrt(ANNUALISATION)``. The stub imports only what its signatures
    #      need: add ``import numpy as np`` and ``ANNUALISATION``, ``VOL_ESTIMATORS`` (and
    #      ``REBALANCES`` for TODO 19) to the ``from src.config import ...`` line.
    #   Verified by tests/test_portfolio.py::TestRealisedVol and ::TestIntegrityNoLookahead.
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if estimator not in VOL_ESTIMATORS:
        raise ValueError(f"estimator must be one of {VOL_ESTIMATORS}, got {estimator!r}")

    if estimator == "rolling":
        daily_vol = returns.rolling(window, min_periods=window).std()
    else:
        daily_vol = returns.ewm(span=window, min_periods=window).std()
        daily_vol = daily_vol.where(returns.notna())

    return daily_vol * np.sqrt(ANNUALISATION)


# ---------------------------------------------------------------- F3.2 per-asset scaling


def asset_weights(
    signal: pd.DataFrame, vol: pd.DataFrame, sigma_target: float
) -> tuple[pd.DataFrame, pd.Series]:
    """§2.2 step 1: ``w_raw_i(t) = s_i(t) * (sigma_target / vol_i(t)) / N_live(t)``.

    An instrument is **live** on a day when its signal is not NaN and its vol is a finite
    number greater than zero (a signal of exactly ``0`` is live, with weight ``0``; a signal
    without a usable vol is not, and cannot be sized). ``N_live(t)`` is the number of live
    instruments on the row.

    Returns ``(raw_weights, n_live)``: ``raw_weights`` has the index and columns of
    ``signal``, float dtype, NaN wherever the instrument is not live; ``n_live`` is an
    integer series on the same index named ``"n_live"`` (``0`` on a row with no live
    instrument, where every weight is NaN). Inputs are not modified. Raises ``ValueError``
    if ``sigma_target <= 0`` or if ``vol`` does not have exactly the index and columns of
    ``signal``.
    """
    # TODO 18: implement asset_weights.
    #   1. Validate ``sigma_target > 0`` and ``vol.index.equals(signal.index)`` /
    #      ``vol.columns.equals(signal.columns)`` (``ValueError``).
    #   2. ``live = signal.notna() & (vol > 0) & np.isfinite(vol)`` (a NaN vol compares
    #      False in both, so ``vol > 0`` alone already excludes it; the ``isfinite`` guards
    #      an infinite vol); ``n_live = live.sum(axis=1)``.
    #   3. ``raw = (signal * sigma_target / vol).where(live)``, then divide each row by its
    #      count: ``raw.div(n_live, axis=0)`` (a row with ``n_live == 0`` is all NaN already,
    #      so the zero never divides a number).
    #   4. ``n_live.astype(int).rename("n_live")``.
    #   Verified by tests/test_portfolio.py::TestAssetWeights, ::TestIntegrityVolTargeting
    #   and ::TestIntegrityNoLookahead.
    if sigma_target <= 0:
        raise ValueError(f"sigma_target must be > 0, got {sigma_target}")
    if not (vol.index.equals(signal.index) and vol.columns.equals(signal.columns)):
        raise ValueError("vol must have exactly the index and columns of signal")

    live = signal.notna() & (vol > 0) & np.isfinite(vol)
    n_live = live.sum(axis=1)

    raw = (signal * sigma_target / vol).where(live)
    raw = raw.div(n_live, axis=0)

    return raw, n_live.astype(int).rename("n_live")


# --------------------------------------------------------------- F3.3 rebalance schedule


def rebalance_dates(index: pd.DatetimeIndex, rebalance: str = "monthly") -> pd.DatetimeIndex:
    """The days on which weights are decided (§2.3).

    ``"monthly"``: the last row of every calendar month present in ``index`` (the last
    trading day of the month on the panel calendar; a partial final month contributes its
    last row, which is harmless because a weight decided on the sample's last day never
    becomes a position). ``"daily"``: every row. The result is a sorted ``DatetimeIndex``,
    a subset of ``index``, with the same name; ``index`` must be sorted and unique. Raises
    ``ValueError`` for any other ``rebalance`` (see ``REBALANCES``) or an unsorted or
    non-unique ``index``.
    """
    # TODO 19: implement rebalance_dates.
    #   1. Validate ``rebalance in REBALANCES`` and ``index.is_monotonic_increasing and
    #      index.is_unique`` (``ValueError``).
    #   2. daily: return ``index`` as is. monthly: a row is a month-end when the next row is
    #      in another month, and the last row always is: ``month = index.to_period("M")``,
    #      ``last = np.append(month[1:] != month[:-1], True)``, ``index[last]``.
    #   Verified by tests/test_portfolio.py::TestRebalanceDates and
    #   ::TestIntegrityRebalanceSchedule.
    if rebalance not in REBALANCES:
        raise ValueError(f"rebalance must be one of {REBALANCES}, got {rebalance!r}")
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("index must be sorted and unique")

    if rebalance == "daily":
        return index

    month = index.to_period("M")
    last = np.append(month[1:] != month[:-1], True)
    return index[last]


def hold(values: pd.DataFrame | pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame | pd.Series:
    """Sample ``values`` on ``dates`` and hold each sample until the next date (§2.3).

    Every row of the output is a copy of the input's row on the most recent date in
    ``dates`` on or before it, NaNs included (a NaN decided on a rebalance day stays NaN
    until the next one; it is never filled from an earlier rebalance). Rows before the
    first date are NaN. Same index, columns and name as ``values``; a bool or integer
    series comes back as float or object once NaN is introduced, so callers cast. ``dates``
    must be a subset of ``values.index``; raises ``ValueError`` otherwise.
    """
    # TODO 20: implement hold.
    #   ``dates.isin(values.index).all()`` else ``ValueError``; then one expression:
    #   ``values.loc[dates].reindex(values.index, method="ffill")``. ``reindex`` with
    #   ``method="ffill"`` copies whole rows by label, which is exactly "hold the last
    #   decision, NaNs included"; ``DataFrame.ffill`` would be wrong (it fills a NaN cell
    #   from an earlier row).
    #   Verified by tests/test_portfolio.py::TestHold and ::TestIntegrityRebalanceSchedule.
    if not dates.isin(values.index).all():
        raise ValueError("every date in dates must be in values.index")

    return values.loc[dates].reindex(values.index, method="ffill")


# ------------------------------------------- F3.4 portfolio vol scaling and leverage cap


def portfolio_vol(
    raw_weights: pd.DataFrame, returns: pd.DataFrame, window: int, estimator: str = "rolling"
) -> pd.Series:
    """§2.2 step 2's denominator: annualised realised vol of the pre-scaled book.

    The pre-scaled book's daily return is ``sum_i w_raw_i(t-1) * r_i(t)``: the raw weights
    of the previous row applied to today's returns (positions formed at the previous close
    earn today's return; a weight paired with the return of the day it was formed on would
    be a portfolio nobody can hold). The sum skips NaN terms and is NaN on a row where no
    term is defined (no instrument was live the row before), then :func:`realised_vol` is
    applied with the same ``window`` and ``estimator`` as the per-asset stage.

    Returns a float series on ``returns.index`` named ``"portfolio_vol"``; NaN until the
    pre-scaled book has ``window`` valid daily returns. Inputs are not modified. Raises
    ``ValueError`` if ``raw_weights`` does not have exactly the index and columns of
    ``returns`` (``window`` and ``estimator`` are validated by :func:`realised_vol`).
    """
    # TODO 21: implement portfolio_vol.
    #   1. Validate the shape (``ValueError``).
    #   2. ``book = (raw_weights.shift(1) * returns).sum(axis=1, min_count=1)`` (``min_count``
    #      keeps a row with no defined term NaN instead of 0).
    #   3. ``realised_vol(book.to_frame("portfolio_vol"), window, estimator)["portfolio_vol"]``.
    #   Verified by tests/test_portfolio.py::TestPortfolioVol and ::TestIntegrityNoLookahead.
    same_index = raw_weights.index.equals(returns.index)
    same_columns = raw_weights.columns.equals(returns.columns)
    if not (same_index and same_columns):
        raise ValueError("raw_weights must have exactly the index and columns of returns")

    book = (raw_weights.shift(1) * returns).sum(axis=1, min_count=1)

    return realised_vol(book.to_frame("portfolio_vol"), window, estimator)["portfolio_vol"]


def leverage_cap(weights: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, pd.Series]:
    """§2.2 step 3: scale each row down so that ``sum_i |w_i| <= cap``.

    On a row whose gross exposure ``sum_i |w_i|`` (NaN terms skipped) exceeds ``cap``, every
    weight is multiplied by ``cap / gross``, so the row's gross becomes exactly ``cap`` and
    the relative sizes and signs are unchanged; any other row is returned as is. Returns
    ``(capped, binds)``: ``capped`` with the index, columns and NaNs of ``weights``, and
    ``binds`` a bool series on the same index named ``"cap_binds"``, True exactly on the
    rows that were scaled (False on a row with no weight at all). Input not modified.
    Raises ``ValueError`` if ``cap <= 0``.
    """
    # TODO 22: implement leverage_cap.
    #   ``gross = weights.abs().sum(axis=1, min_count=1)``; ``binds = gross > cap`` (NaN
    #   compares False); ``factor = (cap / gross).where(binds, 1.0)``;
    #   ``weights.mul(factor, axis=0)``; ``binds.rename("cap_binds")``.
    #   Verified by tests/test_portfolio.py::TestLeverageCap.
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")

    gross = weights.abs().sum(axis=1, min_count=1)
    binds = gross > cap
    factor = (cap / gross).where(binds, 1.0)

    return weights.mul(factor, axis=0), binds.rename("cap_binds")


def size_positions(
    signal: pd.DataFrame, returns: pd.DataFrame, sizing: SizingConfig, rebalance: str
) -> SizingResult:
    """The whole of §2.2–§2.3: from a signal and daily returns to held weights.

    ``sizing`` is the ``sizing`` section of the config (``vol_window``, ``vol_estimator``,
    ``sigma_target_asset``, ``sigma_target_portfolio``, ``leverage_cap``); ``rebalance`` is
    ``execution.rebalance``. The steps, each a function of this module, in this order:
    :func:`realised_vol` → :func:`asset_weights` → :func:`portfolio_vol` → scale the raw
    weights by ``sigma_target_portfolio / portfolio_vol`` row by row → :func:`leverage_cap`
    → :func:`rebalance_dates` and :func:`hold` on the capped weights and on the cap flag.
    Every quantity on a row uses that row's close and earlier only. A portfolio vol that
    is NaN or not greater than zero (a perfectly hedged synthetic book) leaves the row's
    weights NaN: no finite scale exists.

    Returns a :class:`SizingResult` (see its docstring for every field). ``weights`` is NaN
    before the first rebalance date on which a weight could be sized (portfolio vol is
    itself NaN for ``vol_window`` rows after the first live instrument, so the first
    weights land after that second warm-up), and on rows before that instrument's own first
    live rebalance day for each instrument. ``cap_binds`` is False wherever ``weights`` has
    no position. Raises ``ValueError`` if ``returns`` does not have exactly the index and
    columns of ``signal`` (the per-stage validations cover the parameters).
    """
    # TODO 23: implement size_positions.
    #   1. Validate the shape (``ValueError``).
    #   2. ``vol = realised_vol(returns, sizing.vol_window, sizing.vol_estimator)``;
    #      ``raw, n_live = asset_weights(signal, vol, sizing.sigma_target_asset)``;
    #      ``pvol = portfolio_vol(raw, returns, sizing.vol_window, sizing.vol_estimator)``;
    #      ``scale = (sizing.sigma_target_portfolio / pvol).where(pvol > 0)`` (NaN where
    #      the vol is missing or zero; NaN compares False); ``scaled = raw.mul(scale, axis=0)``;
    #      ``daily, binds = leverage_cap(scaled, sizing.leverage_cap)``;
    #      ``dates = rebalance_dates(signal.index, rebalance)``;
    #      ``weights = hold(daily, dates)``;
    #      ``cap_binds = hold(binds.astype(float), dates).eq(1.0).rename("cap_binds")``
    #      (the float round trip keeps the series bool with False before the first date).
    #   3. Return ``SizingResult(vol, raw, n_live, pvol, daily, weights, cap_binds, dates)``.
    #   Verified by tests/test_portfolio.py::TestSizePositions, ::TestIntegrityVolTargeting,
    #   ::TestIntegrityRebalanceSchedule, ::TestIntegrityNoLookahead and ::TestSnapshot.
    same_index = returns.index.equals(signal.index)
    same_columns = returns.columns.equals(signal.columns)
    if not (same_index and same_columns):
        raise ValueError("returns must have exactly the index and columns of signal")

    vol = realised_vol(returns, sizing.vol_window, sizing.vol_estimator)
    raw, n_live = asset_weights(signal, vol, sizing.sigma_target_asset)
    pvol = portfolio_vol(raw, returns, sizing.vol_window, sizing.vol_estimator)

    scale = (sizing.sigma_target_portfolio / pvol).where(pvol > 0)
    scaled = raw.mul(scale, axis=0)
    daily, binds = leverage_cap(scaled, sizing.leverage_cap)

    dates = rebalance_dates(signal.index, rebalance)
    weights = hold(daily, dates)
    cap_binds = hold(binds.astype(float), dates).eq(1.0).rename("cap_binds")

    return SizingResult(vol, raw, n_live, pvol, daily, weights, cap_binds, dates)


# ------------------------------------------------------------------------- summary


def sizing_summary(result: SizingResult) -> dict[str, float | pd.Timestamp | None]:
    """Headline numbers of a :class:`SizingResult`, for the phase's report.

    A day **has a position** when at least one held weight is not NaN. Keys, exactly
    ``SUMMARY_KEYS`` in order:

    * ``n_rebalance_dates``: ``len(result.rebalance_dates)`` (int);
    * ``first_weight_date``: first day with a position (``Timestamp``; ``None`` if none);
    * ``n_days_with_position``: number of such days (int);
    * ``mean_n_live``: mean of ``n_live`` over the days with a position (float; NaN if none);
    * ``mean_gross_leverage``, ``max_gross_leverage``: mean and max over the days with a
      position of ``sum_i |w_i|`` of the held weights (float; NaN if none);
    * ``n_cap_bind_days``: number of days on which ``cap_binds`` is True (int);
    * ``cap_bind_fraction``: that count over the days with a position (float; NaN if none);
    * ``n_unsizable``: number of (day, instrument) cells inside an instrument's traded
      span (from its first to its last non-NaN raw weight, inclusive) on which its raw
      weight is NaN: days on which an instrument already in the book could not be sized,
      for want of a signal (an unfilled gap or its echo) or of a vol estimate (the
      ``window - 1`` rows after a gap under the rolling estimator) (int). This is what
      binds in the per-asset stage.
    """
    # TODO 24: implement sizing_summary.
    #   ``held = result.weights``; ``has_pos = held.notna().any(axis=1)``;
    #   ``gross = held.abs().sum(axis=1)[has_pos]``; ``n_live = result.n_live[has_pos]``;
    #   ``first = held.index[has_pos][0] if has_pos.any() else None``.
    #   Traded span per column: ``defined = result.raw_weights.notna()``;
    #   ``span = defined.cummax() & defined[::-1].cummax()[::-1]`` (True from the first to
    #   the last defined row); ``n_unsizable = int((span & ~defined).sum().sum())``.
    #   Build the dict in ``SUMMARY_KEYS`` order; cast counts with ``int(...)``, floats with
    #   ``float(...)`` (``float("nan")`` when there is no day with a position).
    #   Verified by tests/test_portfolio.py::TestSizingSummary and ::TestSnapshot.
    held = result.weights
    has_pos = held.notna().any(axis=1)
    gross = held.abs().sum(axis=1)[has_pos]
    n_live = result.n_live[has_pos]
    first = held.index[has_pos][0] if has_pos.any() else None

    defined = result.raw_weights.notna()
    span = defined.cummax() & defined[::-1].cummax()[::-1]
    n_unsizable = int((span & ~defined).sum().sum())

    n_days = int(has_pos.sum())
    n_cap_bind_days = int(result.cap_binds.sum())
    if n_days > 0:
        mean_n_live = float(n_live.mean())
        mean_gross = float(gross.mean())
        max_gross = float(gross.max())
        cap_bind_fraction = float(n_cap_bind_days / n_days)
    else:
        mean_n_live = mean_gross = max_gross = cap_bind_fraction = float("nan")

    return {
        "n_rebalance_dates": int(len(result.rebalance_dates)),
        "first_weight_date": first,
        "n_days_with_position": n_days,
        "mean_n_live": mean_n_live,
        "mean_gross_leverage": mean_gross,
        "max_gross_leverage": max_gross,
        "n_cap_bind_days": n_cap_bind_days,
        "cap_bind_fraction": cap_bind_fraction,
        "n_unsizable": n_unsizable,
    }


__all__ = [
    "SUMMARY_KEYS",
    "SizingResult",
    "asset_weights",
    "hold",
    "leverage_cap",
    "portfolio_vol",
    "realised_vol",
    "rebalance_dates",
    "size_positions",
    "sizing_summary",
]
