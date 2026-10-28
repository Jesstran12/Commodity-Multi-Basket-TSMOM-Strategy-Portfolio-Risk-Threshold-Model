"""Sub-period analysis (PROJECT_LOG.md §5.3, F6.3): how the result is spread over time.

One Sharpe ratio over 25 years can hide a strategy that earned everything in one decade.
This module cuts the daily return series four ways:

* :func:`block_stats`: fixed calendar blocks, chosen a priori in the config
  (``robustness.subperiod_blocks``: 2000-2007, 2008-2013, 2014-2019, 2020-present);
* :func:`calendar_year_returns`: one compounded return per calendar year;
* :func:`rolling_sharpe` and :func:`rolling_summary`: the trailing three-year Sharpe
  ratio and where it was lowest and highest;
* :func:`regime_labels` and :func:`regime_stats`: a "calm" versus "stress" view, stress
  being the days on which the equal-weight basket's trailing volatility sat in the top
  quarter of its **full-sample** distribution. That threshold is only known at the end of
  the sample: **the regime view uses hindsight** and is reported with that caveat, never as
  something a trader could have acted on. Every other function here uses fixed calendar
  cuts or trailing windows only.

Conventions, shared by every function here:

* ``returns`` is a wide frame of daily return series indexed by date, one column per series
  (strategy gross, strategy net, the benchmarks); each column is measured over its own
  non-NaN rows (NaN rows dropped first, as in ``src/metrics.py``).
* Sharpe ratios and volatilities are annualised with ``ANNUALISATION``; ``rf`` is the
  annual risk-free rate (``inference.risk_free_rate``). The statistics are those of
  ``src.metrics`` (``annualised_return``, ``annualised_vol``, ``sharpe``), so a block or a
  regime is measured exactly as the whole sample is.
* Pure functions: inputs never modified; NaN where a statistic is undefined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANNUALISATION
from src.metrics import annualised_return, annualised_vol, sharpe
from src.portfolio import realised_vol

BLOCK_COLUMNS = (
    "block",
    "series",
    "first_date",
    "last_date",
    "n_days",
    "ann_return",
    "ann_vol",
    "sharpe",
)
"""Columns of :func:`block_stats`, in order."""

ROLLING_SUMMARY_COLUMNS = (
    "n_windows",
    "share_positive",
    "min",
    "min_date",
    "max",
    "max_date",
    "last",
    "last_date",
)
"""Columns of :func:`rolling_summary` (indexed by series), in order."""

REGIMES = ("calm", "stress")
"""The two regime labels, in report order."""

REGIME_COLUMNS = ("regime", "series", "n_days", "share_of_days", "ann_return", "ann_vol", "sharpe")
"""Columns of :func:`regime_stats`, in order."""


def block_label(start: str | None, end: str | None) -> str:
    """A calendar block in words: the years of its two ends joined by a hyphen, ``"start"``
    or ``"present"`` for an open end (``("2000-01-01", "2007-12-31")`` -> ``"2000-2007"``,
    ``("2020-01-01", None)`` -> ``"2020-present"``). Provided complete (not a TODO)."""
    return f"{start[:4] if start else 'start'}-{end[:4] if end else 'present'}"


def block_stats(
    returns: pd.DataFrame,
    blocks: tuple[tuple[str | None, str | None], ...],
    rf: float = 0.0,
) -> pd.DataFrame:
    """Each series' statistics inside each calendar block.

    ``blocks`` holds ``(start, end)`` date strings, both ends inclusive, ``None`` for an
    open end. One row per (block, series), blocks in the given order and within a block the
    columns of ``returns`` in order; columns exactly ``BLOCK_COLUMNS``; a ``RangeIndex``:

    * ``block``: :func:`block_label`; ``series``: the column name;
    * over the column's non-NaN rows with a date in the block: ``first_date`` and
      ``last_date`` (``NaT`` when there is none), ``n_days`` (``int``), and the
      ``annualised_return``, ``annualised_vol`` and ``sharpe`` (with ``rf``) of
      ``src.metrics``.

    A block in which a series has no row gives ``n_days = 0`` and NaN statistics, not an
    error.
    """
    # TODO 69: implement block_stats.
    #   For each ``(start, end)`` and each column: ``r = returns[col].dropna()``; ``inside =
    #   r.loc[start:end]`` (label slicing includes both ends, and ``None`` leaves that end
    #   open); one dict ``{"block": block_label(start, end), "series": col, "first_date":
    #   inside.index[0] if len(inside) else pd.NaT, "last_date": ..., "n_days":
    #   len(inside), "ann_return": annualised_return(inside), "ann_vol":
    #   annualised_vol(inside), "sharpe": sharpe(inside, rf)}``.
    #   ``pd.DataFrame(rows, columns=list(BLOCK_COLUMNS))``; make ``n_days`` ``int`` and the
    #   two date columns ``datetime64[ns]`` (``astype``) so an all-empty column keeps its type.
    #   Imports to add: ``annualised_return``, ``annualised_vol``, ``sharpe`` (``src.metrics``).
    #   Verified by tests/test_subperiods.py::TestBlockStats.
    rows = []
    for start, end in blocks:
        for col in returns.columns:
            r = returns[col].dropna()
            inside = r.loc[start:end]
            rows.append(
                {
                    "block": block_label(start, end),
                    "series": col,
                    "first_date": inside.index[0] if len(inside) else pd.NaT,
                    "last_date": inside.index[-1] if len(inside) else pd.NaT,
                    "n_days": len(inside),
                    "ann_return": annualised_return(inside),
                    "ann_vol": annualised_vol(inside),
                    "sharpe": sharpe(inside, rf),
                }
            )

    out = pd.DataFrame(rows, columns=list(BLOCK_COLUMNS))
    return out.astype(
        {"n_days": "int64", "first_date": "datetime64[ns]", "last_date": "datetime64[ns]"}
    )


def calendar_year_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """The compounded return of each series in each calendar year.

    Indexed by the year (``int``, named ``"year"``, ascending: every year in which at least
    one series has a non-NaN row), the columns of ``returns`` in order. Each value compounds
    the column's non-NaN rows of that year, ``prod(1 + r) - 1``; partial first and last
    years are kept, and a year in which a column has no row is NaN for it, not ``0``.
    """
    # TODO 70: implement calendar_year_returns.
    #   Per column: ``r = returns[col].dropna()``; ``(1 + r).groupby(r.index.year).prod() -
    #   1`` (a year with no row is simply absent). ``pd.concat(per_column, axis=1)`` aligns
    #   the years and leaves NaN where a column has none; then ``sort_index()``, the column
    #   names, and ``rename_axis("year")``. Watch the index dtype: it must be ``int``.
    #   Verified by tests/test_subperiods.py::TestCalendarYearReturns.
    per_column = []
    for col in returns.columns:
        r = returns[col].dropna()
        yearly = (1 + r).groupby(r.index.year).prod() - 1
        per_column.append(yearly.rename(col))

    out = pd.concat(per_column, axis=1).sort_index()
    out.index = out.index.astype("int64")
    out.columns = list(returns.columns)
    return out.rename_axis("year")


def rolling_sharpe(returns: pd.DataFrame, window: int, rf: float = 0.0) -> pd.DataFrame:
    """The trailing Sharpe ratio of each series over its last ``window`` observations.

    For each column on its own non-NaN rows: the mean of the excess return ``r -
    rf / ANNUALISATION`` over the trailing ``window`` rows, divided by their standard
    deviation (``ddof = 1``), times ``sqrt(ANNUALISATION)``; NaN until ``window``
    observations exist and where that standard deviation is not ``> 0``. The result is then
    put back on the full index (same index and columns as ``returns``), NaN on rows where
    the column itself is NaN. A value on row ``t`` uses the rows up to ``t`` only. The
    stage passes ``window = robustness.rolling_sharpe_years * ANNUALISATION``.

    Raises ``ValueError`` if ``window < 2``.
    """
    # TODO 71: implement rolling_sharpe.
    #   ``ValueError`` if ``window < 2``. Per column: ``x = returns[col].dropna() - rf /
    #   ANNUALISATION``; ``roll = x.rolling(window, min_periods=window)``; ``mean, std =
    #   roll.mean(), roll.std(ddof=1)``; ``(mean / std.where(std > 0)) *
    #   np.sqrt(ANNUALISATION)``; ``.reindex(returns.index)`` puts it back on the full index
    #   (NaN where the column is NaN). Assemble with ``pd.DataFrame(per_column,
    #   index=returns.index, columns=returns.columns)``.
    #   Imports to add: ``numpy as np``, ``ANNUALISATION`` (``src.config``).
    #   Verified by tests/test_subperiods.py::TestRollingSharpe (the lookahead test included).
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    per_column = {}
    for col in returns.columns:
        x = returns[col].dropna() - rf / ANNUALISATION
        roll = x.rolling(window, min_periods=window)
        mean, std = roll.mean(), roll.std(ddof=1)
        ratio = (mean / std.where(std > 0)) * np.sqrt(ANNUALISATION)
        per_column[col] = ratio.reindex(returns.index)

    return pd.DataFrame(per_column, index=returns.index, columns=returns.columns)


def rolling_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    """Where each rolling series was lowest, highest and last.

    ``rolling`` is :func:`rolling_sharpe`'s output. Indexed by series (the columns of
    ``rolling``, in order), columns exactly ``ROLLING_SUMMARY_COLUMNS``, over each column's
    non-NaN values: ``n_windows`` (``int``), ``share_positive`` (share ``> 0``), ``min`` and
    ``min_date`` (the row on which the lowest window ends; the first such row on a tie),
    ``max`` and ``max_date`` (likewise), ``last`` and ``last_date`` (the last non-NaN
    value). A column without a value has ``n_windows = 0``, NaN values and ``NaT`` dates.
    """
    # TODO 72: implement rolling_summary.
    #   Per column: ``x = rolling[col].dropna()``; if empty, ``n_windows = 0`` with NaN and
    #   ``pd.NaT``; else ``len(x)``, ``float((x > 0).mean())``, ``x.min()`` with
    #   ``x.idxmin()`` (the first minimum), ``x.max()`` with ``x.idxmax()``, ``x.iloc[-1]``
    #   with ``x.index[-1]``. ``pd.DataFrame(rows, index=rolling.columns,
    #   columns=list(ROLLING_SUMMARY_COLUMNS))``; ``n_windows`` as ``int``, the three date
    #   columns as ``datetime64[ns]``.
    #   Verified by tests/test_subperiods.py::TestRollingSummary.
    rows = []
    for col in rolling.columns:
        x = rolling[col].dropna()
        if len(x) == 0:
            rows.append(
                {
                    "n_windows": 0,
                    "share_positive": np.nan,
                    "min": np.nan,
                    "min_date": pd.NaT,
                    "max": np.nan,
                    "max_date": pd.NaT,
                    "last": np.nan,
                    "last_date": pd.NaT,
                }
            )
            continue
        rows.append(
            {
                "n_windows": len(x),
                "share_positive": float((x > 0).mean()),
                "min": float(x.min()),
                "min_date": x.idxmin(),
                "max": float(x.max()),
                "max_date": x.idxmax(),
                "last": float(x.iloc[-1]),
                "last_date": x.index[-1],
            }
        )

    out = pd.DataFrame(rows, index=rolling.columns, columns=list(ROLLING_SUMMARY_COLUMNS))
    return out.astype(
        {
            "n_windows": "int64",
            "min_date": "datetime64[ns]",
            "max_date": "datetime64[ns]",
            "last_date": "datetime64[ns]",
        }
    )


def regime_labels(returns: pd.Series, window: int, quantile: float) -> pd.Series:
    """Label each day ``"stress"`` or ``"calm"`` by a basket's trailing volatility.
    **Uses hindsight.**

    ``vol`` is the trailing ``window``-row volatility of ``returns`` exactly as
    ``portfolio.realised_vol(..., window, "rolling")`` computes it (sample std over a full
    window, annualised); the threshold is ``vol.quantile(quantile)`` over **the whole
    series' non-NaN values**, a number nobody knew until the sample ended. A row is
    ``"stress"`` where ``vol`` is strictly above the threshold, ``"calm"`` where it is at or
    below, and NaN where ``vol`` is NaN. Same index as ``returns``, named ``"regime"``.

    Raises ``ValueError`` unless ``0 < quantile < 1``.
    """
    # TODO 73: implement regime_labels.
    #   ``ValueError`` unless ``0 < quantile < 1``. ``vol = realised_vol(returns.to_frame(),
    #   window, "rolling").iloc[:, 0]``; ``threshold = vol.quantile(quantile)`` (NaN skipped);
    #   ``labels = pd.Series(np.where(vol > threshold, "stress", "calm"),
    #   index=returns.index).where(vol.notna())`` (NaN where the vol is), named ``"regime"``.
    #   Imports to add: ``realised_vol`` (``src.portfolio``).
    #   Verified by tests/test_subperiods.py::TestRegimeLabels.
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be strictly between 0 and 1, got {quantile}")

    vol = realised_vol(returns.to_frame(), window, "rolling").iloc[:, 0]
    threshold = vol.quantile(quantile)
    labels = pd.Series(np.where(vol > threshold, "stress", "calm"), index=returns.index)
    return labels.where(vol.notna()).rename("regime")


def regime_stats(returns: pd.DataFrame, labels: pd.Series, rf: float = 0.0) -> pd.DataFrame:
    """Each series' statistics on the calm days and on the stress days.

    ``labels`` is :func:`regime_labels`' output, aligned to ``returns.index`` first
    (``reindex``; a row without a label belongs to neither regime). One row per (regime,
    series), regimes in ``REGIMES`` order, series in column order; columns exactly
    ``REGIME_COLUMNS``; a ``RangeIndex``. Over the column's non-NaN rows with that label:
    ``n_days`` (``int``); ``share_of_days``, ``n_days`` over the column's non-NaN rows with
    either label; and the ``annualised_return``, ``annualised_vol`` and ``sharpe`` of
    ``src.metrics`` (the days of a regime are not consecutive; the statistics treat them as
    one series, the usual reading of a regime split).
    """
    # TODO 74: implement regime_stats.
    #   ``lab = labels.reindex(returns.index)``; for each regime in ``REGIMES`` and each
    #   column: ``r = returns[col]``; ``labelled = r[r.notna() & lab.notna()]``; ``days =
    #   labelled[lab[labelled.index] == regime]``; one dict with ``n_days = len(days)``,
    #   ``share_of_days = len(days) / len(labelled)`` (NaN if ``labelled`` is empty) and the
    #   three metrics of ``days``. ``pd.DataFrame(rows, columns=list(REGIME_COLUMNS))``.
    #   Verified by tests/test_subperiods.py::TestRegimeStats.
    lab = labels.reindex(returns.index)

    rows = []
    for regime in REGIMES:
        for col in returns.columns:
            r = returns[col]
            labelled = r[r.notna() & lab.notna()]
            days = labelled[lab[labelled.index] == regime]
            rows.append(
                {
                    "regime": regime,
                    "series": col,
                    "n_days": len(days),
                    "share_of_days": len(days) / len(labelled) if len(labelled) else np.nan,
                    "ann_return": annualised_return(days),
                    "ann_vol": annualised_vol(days),
                    "sharpe": sharpe(days, rf),
                }
            )

    return pd.DataFrame(rows, columns=list(REGIME_COLUMNS))


__all__ = [
    "BLOCK_COLUMNS",
    "REGIMES",
    "REGIME_COLUMNS",
    "ROLLING_SUMMARY_COLUMNS",
    "block_label",
    "block_stats",
    "calendar_year_returns",
    "regime_labels",
    "regime_stats",
    "rolling_summary",
    "rolling_sharpe",
]
