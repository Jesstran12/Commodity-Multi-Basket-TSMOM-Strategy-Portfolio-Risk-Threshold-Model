"""Backtest engine and costs (PROJECT_LOG.md §2.3–§2.4): from held weights to gross and net
daily returns.

Pure functions over wide frames (index = tz-naive daily ``DatetimeIndex``, columns =
tickers), no I/O. Inputs are the held weights from ``portfolio.size_positions`` (or a
benchmark's weights from ``benchmark``) and the cleaned daily returns from
``data.clean_returns``, on the same calendar and columns.

The pipeline, in :func:`run`:

1. ``positions = weights.shift(lag)``: the weights decided at the close of ``t - lag`` are
   the positions that earn the return from close ``t - 1`` to close ``t`` (§2.3; ``lag = 1``
   by default, ``2`` is the conservative variant).
2. ``turnover(t) = sum_i |pos_i(t) - pos_i(t-1)|`` with a NaN position counted as zero
   exposure, so entering and exiting are traded like any other change (§2.4).
3. ``cost(t) = turnover(t) * cost_bps * BASIS_POINT`` (one-way, per unit of notional).
4. ``gross(t) = sum_i pos_i(t) * r_i(t)`` and ``net(t) = gross(t) - cost(t)``.

Row conventions: one row is one trading day on the panel calendar, so "``lag`` days" is
"``lag`` rows". A position is NaN where there is none (the weight was NaN); the strategy's
return series is NaN **before its first position** and defined (zero when flat) from then
on, so that the metrics in ``metrics`` measure the strategy only over its own life.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ANNUALISATION, BASIS_POINT
from src.metrics import performance

SUMMARY_KEYS = (
    "first_date",
    "last_date",
    "n_days",
    "ann_return_gross",
    "ann_vol_gross",
    "sharpe_gross",
    "sortino_gross",
    "max_drawdown_gross",
    "ann_return_net",
    "ann_vol_net",
    "sharpe_net",
    "sortino_net",
    "max_drawdown_net",
    "total_cost",
    "ann_turnover",
    "mean_gross_leverage",
    "mean_n_live",
    "n_unpriced",
    "lag",
    "cost_bps",
)
"""Keys of the backtest summary, in order."""


@dataclass(frozen=True)
class BacktestResult:
    """Everything :func:`run` computes, daily, on the input calendar.

    * ``positions``: ``weights.shift(lag)`` (frame; NaN where there is no position).
    * ``gross_returns``: the book's daily return before costs (float series named
      ``"gross"``; NaN before the first position, then the sum of ``pos * r`` over the
      defined terms, ``0.0`` when nothing is held or priced).
    * ``costs``: the daily cost in return units (float series named ``"cost"``; ``0.0``
      wherever nothing is traded, the first row included).
    * ``net_returns``: ``gross_returns - costs`` (float series named ``"net"``; NaN exactly
      where ``gross_returns`` is).
    * ``turnover``: ``sum_i |pos_i(t) - pos_i(t-1)|`` (float series named ``"turnover"``;
      ``0.0`` on the first row and wherever nothing changes).
    * ``gross_leverage``: ``sum_i |pos_i(t)|`` (float series named ``"gross_leverage"``;
      NaN before the first position, ``0.0`` when flat after it).
    * ``n_live``: number of instruments with a position, a flat zero included (int series
      named ``"n_live"``; ``0`` where there is none).
    * ``unpriced``: per instrument, the number of days on which it had a position but no
      return (an unfilled gap and the day after it; those terms earn nothing) (int series
      indexed by ticker, named ``"n_unpriced"``). This is what binds in the engine.
    * ``lag``, ``cost_bps``: the parameters used.
    """

    positions: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    gross_leverage: pd.Series
    n_live: pd.Series
    unpriced: pd.Series
    lag: int
    cost_bps: float


# ----------------------------------------------------------------------- F4.1 engine


def positions_from_weights(weights: pd.DataFrame, lag: int) -> pd.DataFrame:
    """§2.3: ``positions = weights.shift(lag)``, row-positional on the frame's calendar.

    The weights decided at the close of row ``t - lag`` are the positions held over row
    ``t`` (and earn that row's return). The first ``lag`` rows are NaN (nothing was decided
    yet); a NaN weight is a NaN position. Same index, columns and dtype as ``weights``;
    ``weights`` is not modified. Raises ``ValueError`` if ``lag < 1`` (a lag of zero would
    pair a weight with the return of the day it was computed from, the lookahead the §8
    lag test exists to catch).
    """
    # TODO 25: implement positions_from_weights.
    #   Validate ``lag >= 1`` (``ValueError``); return ``weights.shift(lag)``. ``shift`` is
    #   positional (rows, not calendar days) and fills the top with NaN, which is exactly
    #   the contract; do not ``ffill`` anything.
    #   Verified by tests/test_backtest.py::TestPositionsFromWeights, ::TestIntegrityLag and
    #   ::TestIntegrityNoLookahead.
    if lag < 1:
        raise ValueError(f"lag must be >= 1 (got {lag}); lag 0 is same-day lookahead")
    return weights.shift(lag)


def turnover(positions: pd.DataFrame) -> pd.Series:
    """§2.4: ``TO(t) = sum_i |pos_i(t) - pos_i(t-1)|``, with NaN as zero exposure.

    A NaN position is no position, so entering from NaN into ``w`` trades ``|w|``, exiting
    to NaN trades ``|w|`` again, and a flip from ``w`` to ``-w`` trades ``2|w|``. The first
    row has no previous row and trades nothing (``0.0``), as does every row on which no
    position changes. Returns a float series on ``positions.index`` named ``"turnover"``;
    the input is not modified.
    """
    # TODO 26: implement turnover.
    #   ``exposure = positions.fillna(0.0)``; ``exposure.diff().abs().sum(axis=1)`` (the
    #   first row's ``diff`` is NaN in every column and ``sum`` skips NaN, giving 0.0 there);
    #   ``.rename("turnover")``.
    #   Verified by tests/test_backtest.py::TestTurnover and ::TestIntegrityZeroTurnover.
    exposure = positions.fillna(0.0)
    traded = exposure.diff().abs()
    return traded.sum(axis=1).rename("turnover")


def transaction_costs(turnover: pd.Series, cost_bps: float) -> pd.Series:
    """§2.4: ``cost(t) = TO(t) * cost_bps / 1e4`` in return units.

    ``cost_bps`` is the one-way cost per unit of notional traded in basis points
    (``costs.cost_bps``, 5 by default); ``BASIS_POINT`` from ``src.config`` does the unit
    conversion. Returns a float series on ``turnover.index`` named ``"cost"``; raises
    ``ValueError`` if ``cost_bps < 0``.
    """
    # TODO 27: implement transaction_costs.
    #   Validate ``cost_bps >= 0`` (``ValueError``); ``(turnover * cost_bps * BASIS_POINT)
    #   .rename("cost")``. Never write ``1e4`` or ``1e-4`` here: the constant is in
    #   ``src.config``.
    #   Verified by tests/test_backtest.py::TestTransactionCosts and
    #   ::TestIntegrityCostAccounting.
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0 (got {cost_bps})")
    return (turnover * cost_bps * BASIS_POINT).rename("cost")


def gross_returns(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """The book's daily return before costs: ``sum_i pos_i(t) * r_i(t)``.

    NaN terms (no position, or a position without a return that day) are skipped. The
    series is NaN on every row **before the first row with a position** (the strategy does
    not exist yet) and defined from then on: the sum of the defined terms, ``0.0`` when no
    term is defined (flat, or every held instrument unpriced). Returns a float series on
    ``returns.index`` named ``"gross"``. Raises ``ValueError`` if ``positions`` does not
    have exactly the index and columns of ``returns``; inputs are not modified.
    """
    # TODO 28: implement gross_returns.
    #   1. Validate the shape (``ValueError``).
    #   2. ``pnl = (positions * returns).sum(axis=1)`` (NaN skipped; 0.0 when nothing);
    #      ``started = positions.notna().any(axis=1).cummax()`` (False until the first row
    #      with a position, True from then on); ``pnl.where(started).rename("gross")``.
    #   Verified by tests/test_backtest.py::TestGrossReturns and ::TestIntegrityNoLookahead.
    if not (positions.index.equals(returns.index) and positions.columns.equals(returns.columns)):
        raise ValueError("positions must have exactly the index and columns of returns")
    pnl = (positions * returns).sum(axis=1)
    started = positions.notna().any(axis=1).cummax()
    return pnl.where(started).rename("gross")


# ------------------------------------------------------------------------ F4.2 costs


def run(weights: pd.DataFrame, returns: pd.DataFrame, lag: int, cost_bps: float) -> BacktestResult:
    """The whole engine: held weights and daily returns in, gross and net returns out.

    The steps, each a function of this module, in this order: :func:`positions_from_weights`
    → :func:`turnover` → :func:`transaction_costs` → :func:`gross_returns` → ``net = gross -
    cost``; then the bookkeeping series (``gross_leverage``, ``n_live``, ``unpriced``) from
    the positions. Returns a :class:`BacktestResult` (see its docstring for every field).
    ``weights`` and ``returns`` must have the same index and columns (``ValueError``
    otherwise); the per-stage validations cover ``lag`` and ``cost_bps``. Inputs are not
    modified.
    """
    # TODO 29: implement run.
    #   1. Validate the shape (``ValueError``).
    #   2. ``pos = positions_from_weights(weights, lag)``; ``to = turnover(pos)``;
    #      ``cost = transaction_costs(to, cost_bps)``; ``gross = gross_returns(pos, returns)``;
    #      ``net = (gross - cost).rename("net")`` (NaN where gross is NaN).
    #   3. ``lev = pos.abs().sum(axis=1).where(gross.notna()).rename("gross_leverage")``;
    #      ``n_live = pos.notna().sum(axis=1).astype(int).rename("n_live")``;
    #      ``unpriced = (pos.notna() & returns.isna()).sum().astype(int).rename("n_unpriced")``.
    #   4. Return ``BacktestResult(pos, gross, cost, net, to, lev, n_live, unpriced, lag,
    #      cost_bps)``.
    #   Verified by tests/test_backtest.py::TestRun, every ``TestIntegrity*`` class there and
    #   ::TestSnapshot.
    if not (weights.index.equals(returns.index) and weights.columns.equals(returns.columns)):
        raise ValueError("weights must have exactly the index and columns of returns")
    pos = positions_from_weights(weights, lag)
    to = turnover(pos)
    cost = transaction_costs(to, cost_bps)
    gross = gross_returns(pos, returns)
    net = (gross - cost).rename("net")
    lev = pos.abs().sum(axis=1).where(gross.notna()).rename("gross_leverage")
    n_live = pos.notna().sum(axis=1).astype(int).rename("n_live")
    unpriced = (pos.notna() & returns.isna()).sum().astype(int).rename("n_unpriced")
    return BacktestResult(pos, gross, cost, net, to, lev, n_live, unpriced, lag, cost_bps)


# ------------------------------------------------------------------------- summary


def backtest_summary(result: BacktestResult, risk_free_rate: float = 0.0) -> dict[str, object]:
    """Headline numbers of a :class:`BacktestResult`, gross **and** net, for the report.

    Keys exactly ``SUMMARY_KEYS`` in order. ``first_date``, ``last_date`` and ``n_days``
    describe the strategy's life (the non-NaN rows of ``gross_returns``; ``None`` / ``0``
    when there is none). The five ``*_gross`` and five ``*_net`` statistics are
    :func:`metrics.performance` of ``gross_returns`` and ``net_returns`` with
    ``risk_free_rate``. ``total_cost`` is the sum of ``costs``; ``ann_turnover`` is the mean
    daily turnover over the life times ``ANNUALISATION`` (the fraction of the book traded
    per year); ``mean_gross_leverage`` and ``mean_n_live`` are means over the life;
    ``n_unpriced`` is the sum of ``unpriced`` over instruments; ``lag`` and ``cost_bps`` are
    copied. Counts are ``int``, the rest ``float`` (NaN when there is no life).
    """
    # TODO 30: implement backtest_summary.
    #   ``life = result.gross_returns.notna()``; ``g = performance(result.gross_returns, rf)``,
    #   ``n = performance(result.net_returns, rf)``; the ten statistics are ``g[k]`` / ``n[k]``
    #   for ``k`` in ``("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown")`` under
    #   the ``_gross`` / ``_net`` suffix. ``ann_turnover = result.turnover[life].mean() *
    #   ANNUALISATION`` (NaN when ``life`` is all False, which ``.mean()`` of an empty series
    #   gives for free); likewise the two other means. Cast with ``int(...)`` / ``float(...)``
    #   and build the dict in ``SUMMARY_KEYS`` order. Imports to add as you go:
    #   ``from src.metrics import performance`` and ``ANNUALISATION`` from ``src.config``
    #   (``BASIS_POINT`` for TODO 27 likewise).
    #   Verified by tests/test_backtest.py::TestBacktestSummary and ::TestSnapshot.
    life = result.gross_returns.notna()
    g = performance(result.gross_returns, risk_free_rate)
    n = performance(result.net_returns, risk_free_rate)
    stats = ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown")
    out: dict[str, object] = {
        "first_date": g["first_date"],
        "last_date": g["last_date"],
        "n_days": int(g["n_days"]),
    }
    for k in stats:
        out[f"{k}_gross"] = float(g[k])
    for k in stats:
        out[f"{k}_net"] = float(n[k])
    out["total_cost"] = float(result.costs.sum())
    out["ann_turnover"] = float(result.turnover[life].mean() * ANNUALISATION)
    out["mean_gross_leverage"] = float(result.gross_leverage[life].mean())
    out["mean_n_live"] = float(result.n_live[life].mean())
    out["n_unpriced"] = int(result.unpriced.sum())
    out["lag"] = int(result.lag)
    out["cost_bps"] = float(result.cost_bps)
    return out


__all__ = [
    "SUMMARY_KEYS",
    "BacktestResult",
    "backtest_summary",
    "gross_returns",
    "positions_from_weights",
    "run",
    "transaction_costs",
    "turnover",
]
