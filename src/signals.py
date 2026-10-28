"""Time-series momentum signal (PROJECT_LOG.md §2.1) and its per-instrument summary.

Pure functions over wide frames (index = tz-naive daily ``DatetimeIndex``, columns =
tickers), no I/O. The signal is computed on the **cleaned price index**
(``data.clean_price_index`` of ``data.clean_returns``), so the momentum measured is that of
the very series the P&L is later computed on, and a negative print (WTI, April 2020) can
never reach the ratio below.

Row conventions: one row is one trading day on the panel calendar (the union of the
universe's trading days, PROJECT_LOG.md §3.2), so "``lookback`` trading days ago" is
"``lookback`` rows up".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_VALUES = (-1.0, 0.0, 1.0)
"""The three values a non-NaN signal can take: short, flat, long."""

SUMMARY_COLUMNS = (
    "first_signal",
    "n_signal",
    "n_missing",
    "share_long",
    "share_short",
    "share_zero",
    "last_signal",
)
"""Columns of the signal summary table, in order."""


# ------------------------------------------------------------------ F2.1 TSMOM signal


def trailing_return(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Trailing return over ``lookback`` rows, ending ``skip`` rows ago.

    ``r(t) = P(t - skip) / P(t - skip - lookback) - 1`` per column, row-positional on the
    frame's calendar. With ``skip = 0`` this is PROJECT_LOG.md §2.1's ``P(t) / P(t-L) - 1``;
    ``skip = 21`` is the skip-month variant (the most recent 21 rows are left out of the
    window, which then ends 21 rows ago).

    NaN wherever **either endpoint** is NaN: the first ``lookback + skip`` rows of the
    frame, every row on which the instrument has no price, and every row whose base price
    ``P(t - skip - lookback)`` falls in an unfilled gap (so a gap of ``g`` price-index rows
    costs ``2g`` signal rows: the gap itself and its echo ``lookback + skip`` rows later).
    A NaN strictly inside the window does **not** make the return NaN: the clean price index
    carries its level across a gap (``data.clean_price_index``), so the return across the
    gap is well defined. Same index and columns as ``prices``, float dtype; ``prices`` is
    not modified.

    ``prices`` must be a positive series (the clean price index is; raw settlement prices
    with a negative print are not a valid input). Raises ``ValueError`` if ``lookback < 1``
    or ``skip < 0``.
    """
    # TODO 14: implement trailing_return.
    #   1. Validate: ``lookback < 1`` or ``skip < 0`` -> ``raise ValueError(...)``.
    #   2. Both endpoints are "the price k rows up": ``prices.shift(k)`` is positional (it
    #      moves values down k rows and leaves NaN at the top), which is exactly the "either
    #      endpoint NaN -> NaN" rule for free. ``end = prices.shift(skip)``,
    #      ``start = prices.shift(skip + lookback)``, return ``end / start - 1``.
    #   Do not use ``rolling``/``apply`` (slow, and a NaN inside the window must not matter),
    #   and do not ``ffill`` anything: a missing endpoint stays missing.
    #   Verified by tests/test_signals.py::TestTrailingReturn and ::TestIntegrityNoLookahead.
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")

    # 2. Both endpoints are "the price k rows up", placed on today's row so the two frames
    #    divide cell by cell. shift is positional and leaves NaN wherever there is no row to
    #    point at, which is the warm-up, the gap days and the gap echoes in one stroke.
    end = prices.shift(skip)
    start = prices.shift(skip + lookback)
    return end / start - 1


def tsmom(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """The TSMOM signal: the sign of :func:`trailing_return`.

    ``+1`` where the trailing return is positive, ``-1`` where negative, ``0`` where it is
    exactly zero (a flat price, or a price back at the level it had ``lookback`` rows ago),
    NaN wherever the trailing return is NaN (warm-up, missing prices, gap echoes). Same index
    and columns as ``prices``, float dtype, every non-NaN value in ``SIGNAL_VALUES``.
    ``prices`` is not modified. Same ``ValueError`` conditions as :func:`trailing_return`.
    """
    # TODO 15: implement tsmom. One expression: ``np.sign`` applied to the trailing return
    #   gives +1 / -1 / 0 / NaN with exactly the semantics above (``import numpy as np``).
    #   Verified by tests/test_signals.py::TestTsmom, ::TestIntegritySyntheticTrend,
    #   ::TestIntegrityNoLookahead and ::TestSnapshot.
    return np.sign(trailing_return(prices, lookback, skip))


# ------------------------------------------------------------------ signal summary


def signal_summary(signal: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument description of a signal frame, for the phase's numbers and reports.

    Parameters
    ----------
    signal:
        A frame as returned by :func:`tsmom`: every non-NaN value in ``SIGNAL_VALUES``;
        any other value raises ``ValueError``.

    Returns
    -------
    One row per column of ``signal``, index = ``signal.columns`` named ``"ticker"``, columns
    exactly ``SUMMARY_COLUMNS``, in order:

    * ``first_signal``: date of the first non-NaN signal (``NaT`` if there is none);
    * ``n_signal``: number of non-NaN signal rows (int);
    * ``n_missing``: number of NaN rows **after** ``first_signal`` (int; 0 if no signal at
      all). This is what binds in this phase: days on which an instrument had no signal
      after it had started, from unfilled gaps and their echoes;
    * ``share_long``, ``share_short``, ``share_zero``: fraction of the ``n_signal`` rows at
      ``+1``, ``-1`` and ``0``; they sum to 1; all NaN when ``n_signal`` is 0;
    * ``last_signal``: the value on the frame's last row (NaN if it is NaN there).
    """
    # TODO 16: implement signal_summary.
    #   1. Validate: the non-NaN values (``values = signal.to_numpy(dtype=float)``,
    #      ``values[~np.isnan(values)]``) must all be in ``SIGNAL_VALUES``
    #      (``np.isin(...).all()``), else ``raise ValueError``.
    #   2. One dict per column: ``s.first_valid_index()`` (None when the column is all NaN
    #      -> ``pd.NaT``), ``s.notna().sum()``, ``s.loc[first:].isna().sum()`` for the
    #      missing count, ``(s == 1).sum() / n`` etc. for the shares (NaN when ``n == 0``),
    #      ``s.iloc[-1]`` for the last value.
    #   3. ``pd.DataFrame(rows, index=pd.Index(signal.columns, name="ticker"),
    #      columns=list(SUMMARY_COLUMNS))``; make ``first_signal`` a datetime column
    #      (``pd.to_datetime``) so it stays typed when a NaT is in it. Cast the two counts
    #      with ``int(...)`` so the columns are integer, not object or float.
    #   Verified by tests/test_signals.py::TestSignalSummary and ::TestSnapshot.
    values = signal.to_numpy(dtype=float)
    present = values[~np.isnan(values)]
    if not np.isin(present, SIGNAL_VALUES).all():
        bad = sorted(set(present[~np.isin(present, SIGNAL_VALUES)]))
        raise ValueError(f"signal values outside {SIGNAL_VALUES}: {bad}")

    rows = []
    for ticker in signal.columns:
        s = signal[ticker]
        first = s.first_valid_index()
        n = int(s.notna().sum())
        if first is None:
            rows.append(
                {
                    "first_signal": pd.NaT,
                    "n_signal": 0,
                    "n_missing": 0,
                    "share_long": np.nan,
                    "share_short": np.nan,
                    "share_zero": np.nan,
                    "last_signal": s.iloc[-1],
                }
            )
            continue
        rows.append(
            {
                "first_signal": first,
                "n_signal": n,
                "n_missing": int(s.loc[first:].isna().sum()),
                "share_long": float((s == 1).sum() / n),
                "share_short": float((s == -1).sum() / n),
                "share_zero": float((s == 0).sum() / n),
                "last_signal": s.iloc[-1],
            }
        )

    out = pd.DataFrame(
        rows, index=pd.Index(signal.columns, name="ticker"), columns=list(SUMMARY_COLUMNS)
    )
    out["first_signal"] = pd.to_datetime(out["first_signal"])
    out["n_signal"] = out["n_signal"].astype(int)
    out["n_missing"] = out["n_missing"].astype(int)
    return out


__all__ = ["SIGNAL_VALUES", "SUMMARY_COLUMNS", "signal_summary", "trailing_return", "tsmom"]
