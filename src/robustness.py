"""Robustness (PROJECT_LOG.md §5.3): the parameter grid, the in-sample / out-of-sample
selection exercise, and the cost and lag sweeps.

The headline result is the a-priori configuration of ``config/default.yaml``. This module
asks how much that result depends on the choices behind it:

* :func:`grid_specs` lists every configuration of the §5.3 grid (192 by design), and
  :func:`trial_config` turns one of them into a full :class:`~src.config.Config`;
* :func:`evaluate_trial` runs one configuration through the whole pipeline (signal, sizing,
  engine) at every cost of ``robustness.cost_bps_sweep`` and measures it;
  :func:`trial_fingerprint` identifies the inputs a cached result depends on, so the grid
  stage in ``src/cli.py`` can resume;
* :func:`trial_stats` gives the Deflated Sharpe Ratio its two inputs, the number of trials
  and the spread of their Sharpe ratios;
* :func:`is_oos` selects the lookback on the in-sample years only and reports what the
  selection does out of sample next to the a-priori default;
* :func:`cost_sweep`, :func:`lag_sweep` and :func:`grid_distribution` summarise the grid.

Conventions, shared by every function here:

* A **trial** is one configuration of the six grid parameters, ``GRID_KEYS``. Costs are not
  trials: a cost is an accounting assumption applied to the same positions, so each trial is
  measured at every cost of the sweep, and the **cost-0 rows are the trial's gross
  results**. ``N_trials`` is the number of configurations.
* The **grid frame** has one row per (trial, cost) with columns exactly ``GRID_COLUMNS``
  (``TRIAL_COLUMNS`` plus the fingerprint), rows in trial order (:func:`grid_specs`'s) and
  within a trial in sweep order, a ``RangeIndex`` and ``GRID_DTYPES``.
* Each trial is measured over **its own life**, from its first position to the end of the
  panel, so the default trial's net Sharpe is exactly the headline one. The in-sample rows
  are the life's rows on or before ``robustness.is_end``, the out-of-sample rows those on
  or after ``robustness.oos_start``.
* Pure functions: frames in, frames or dicts out, inputs never modified. The cache file
  itself is written by the stage in ``src/cli.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.backtest import run
from src.config import ANNUALISATION, Config, RobustnessConfig
from src.data import clean_price_index
from src.metrics import annualised_return, annualised_vol, max_drawdown, sharpe
from src.portfolio import size_positions
from src.signals import tsmom

GRID_KEYS = ("lookback", "vol_window", "estimator", "rebalance", "skip", "lag")
"""The six grid parameters of a trial, in the order they vary in :func:`grid_specs` (the
first slowest). Each maps to one config field: ``signal.lookback``, ``sizing.vol_window``,
``sizing.vol_estimator``, ``execution.rebalance``, ``signal.skip``, ``execution.lag``."""

TRIAL_COLUMNS = (
    *GRID_KEYS,
    "cost_bps",
    "first_date",
    "n_days",
    "n_days_is",
    "n_days_oos",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sharpe_is",
    "sharpe_oos",
    "max_drawdown",
    "ann_turnover",
    "mean_gross_leverage",
    "cap_bind_fraction",
)
"""Columns of :func:`evaluate_trial`'s frame, in order."""

GRID_COLUMNS = (*TRIAL_COLUMNS, "fingerprint")
"""Columns of the grid frame and of ``results/grid.parquet``, in order."""

GRID_DTYPES: dict[str, str] = {
    "lookback": "int64",
    "vol_window": "int64",
    "estimator": "str",
    "rebalance": "str",
    "skip": "int64",
    "lag": "int64",
    "cost_bps": "float64",
    "first_date": "datetime64[ns]",
    "n_days": "int64",
    "n_days_is": "int64",
    "n_days_oos": "int64",
    "ann_return": "float64",
    "ann_vol": "float64",
    "sharpe": "float64",
    "sharpe_is": "float64",
    "sharpe_oos": "float64",
    "max_drawdown": "float64",
    "ann_turnover": "float64",
    "mean_gross_leverage": "float64",
    "cap_bind_fraction": "float64",
    "fingerprint": "str",
}
"""The dtype of every grid column (apply with ``frame.astype(...)``), so that a frame built
in memory and one read back from Parquet compare equal."""

IS_OOS_COLUMNS = (
    "sharpe_is",
    "sharpe_oos",
    "degradation",
    "n_days_is",
    "n_days_oos",
    "selected",
    "default",
)
"""Columns of :func:`is_oos`'s table (indexed by ``lookback``), in order."""

IS_OOS_KEYS = (
    "selected_lookback",
    "selected_is",
    "selected_oos",
    "selected_degradation",
    "default_lookback",
    "default_is",
    "default_oos",
    "default_degradation",
    "oos_gap",
)
"""Keys of :func:`is_oos`'s summary, in order."""

COST_SWEEP_COLUMNS = (
    "ann_return",
    "sharpe",
    "max_drawdown",
    "grid_median_sharpe",
    "grid_share_positive",
)
"""Columns of :func:`cost_sweep` (indexed by ``cost_bps``), in order."""

LAG_SWEEP_COLUMNS = ("sharpe_gross", "sharpe_net", "ann_return_net", "grid_median_net")
"""Columns of :func:`lag_sweep` (indexed by ``lag``), in order."""

DISTRIBUTION_KEYS = (
    "n_trials",
    "sharpe_std",
    "sharpe_min",
    "sharpe_q25",
    "sharpe_median",
    "sharpe_q75",
    "sharpe_max",
    "share_positive",
    "default_sharpe",
    "default_rank",
    "best_trial",
    "worst_trial",
)
"""Keys of :func:`grid_distribution`, in order."""


# ------------------------------------------------------------------ provided helpers


def default_spec(cfg: Config) -> dict[str, object]:
    """The a-priori configuration as a trial: ``{key: value}`` for each of ``GRID_KEYS``,
    read from ``cfg``. Provided complete (not a TODO)."""
    return {
        "lookback": cfg.signal.lookback,
        "vol_window": cfg.sizing.vol_window,
        "estimator": cfg.sizing.vol_estimator,
        "rebalance": cfg.execution.rebalance,
        "skip": cfg.signal.skip,
        "lag": cfg.execution.lag,
    }


def matches(grid: pd.DataFrame, spec: Mapping[str, object]) -> pd.Series:
    """Boolean mask of the rows of ``grid`` whose columns equal every ``{key: value}`` of
    ``spec`` (a subset of ``GRID_KEYS`` and ``cost_bps``; an empty ``spec`` matches every
    row). Provided complete (not a TODO)."""
    mask = pd.Series(True, index=grid.index)
    for key, value in spec.items():
        mask &= grid[key] == value
    return mask


def trial_label(spec: Mapping[str, object]) -> str:
    """A trial in words, for the reports: ``"L 252 / vol 60 rolling / monthly / skip 0 / lag
    1"``. Provided complete (not a TODO)."""
    return (
        f"L {int(spec['lookback'])} / vol {int(spec['vol_window'])} {spec['estimator']} / "
        f"{spec['rebalance']} / skip {int(spec['skip'])} / lag {int(spec['lag'])}"
    )


# ------------------------------------------------------------------- F6.1 the grid


def grid_specs(robustness: RobustnessConfig) -> pd.DataFrame:
    """Every trial of the §5.3 grid: one row per configuration, columns exactly ``GRID_KEYS``.

    The values come from ``robustness.grid_lookbacks``, ``grid_vol_windows``,
    ``grid_estimators``, ``grid_rebalances``, ``grid_skips`` and ``grid_lags``, combined as
    ``itertools.product`` does in that order (the lookback varies slowest, the lag fastest),
    each list in its config order. The index is a ``RangeIndex`` named ``"trial"`` (the
    trial's number, from 0) and the dtypes are those of ``GRID_DTYPES``. The default config
    gives 4 x 3 x 2 x 2 x 2 x 2 = 192 rows.

    Raises ``ValueError`` if a list is empty or holds a value twice (a duplicate would count
    one configuration as two trials), or if a combination has ``skip >= lookback`` (no such
    signal exists).
    """
    # TODO 60: implement grid_specs.
    #   ``lists = {"lookback": robustness.grid_lookbacks, "vol_window": ...grid_vol_windows,
    #   "estimator": ...grid_estimators, "rebalance": ...grid_rebalances, "skip":
    #   ...grid_skips, "lag": ...grid_lags}`` (in ``GRID_KEYS`` order); ``ValueError`` if a
    #   list is empty or ``len(set(values)) != len(values)``.
    #   ``rows = list(itertools.product(*lists.values()))``; ``ValueError`` if any row has
    #   ``skip >= lookback``. ``pd.DataFrame(rows, columns=list(GRID_KEYS))``, then
    #   ``.astype({k: GRID_DTYPES[k] for k in GRID_KEYS})`` and ``.rename_axis("trial")``.
    #   Imports to add: ``itertools``.
    #   Verified by tests/test_robustness.py::TestGridSpecs.
    lists = {
        "lookback": robustness.grid_lookbacks,
        "vol_window": robustness.grid_vol_windows,
        "estimator": robustness.grid_estimators,
        "rebalance": robustness.grid_rebalances,
        "skip": robustness.grid_skips,
        "lag": robustness.grid_lags,
    }
    for key, values in lists.items():
        if len(values) == 0:
            raise ValueError(f"grid list {key!r} is empty")
        if len(set(values)) != len(values):
            raise ValueError(f"grid list {key!r} holds a duplicate: {list(values)}")

    rows = list(itertools.product(*lists.values()))

    for lookback, _vol_window, _estimator, _rebalance, skip, _lag in rows:
        if skip >= lookback:
            raise ValueError(f"skip {skip} >= lookback {lookback}: no such signal exists")

    frame = pd.DataFrame(rows, columns=list(GRID_KEYS))
    frame = frame.astype({key: GRID_DTYPES[key] for key in GRID_KEYS})
    return frame.rename_axis("trial")


def trial_config(cfg: Config, spec: Mapping[str, object]) -> Config:
    """``cfg`` with one trial's six parameters in place of the defaults.

    ``spec`` maps each of ``GRID_KEYS`` to a value (a row of :func:`grid_specs`, as a dict or
    a Series, or :func:`default_spec`). The result equals ``cfg`` except ``signal.lookback``,
    ``signal.skip``, ``sizing.vol_window``, ``sizing.vol_estimator``,
    ``execution.rebalance`` and ``execution.lag``, which take the trial's values as plain
    Python ``int`` and ``str``. ``cfg`` is not modified (the config is frozen; build new
    sections). Raises ``KeyError`` if ``spec`` lacks one of ``GRID_KEYS``.
    """
    # TODO 61: implement trial_config.
    #   ``dataclasses.replace`` builds a copy of a frozen dataclass with some fields changed:
    #   ``signal = dataclasses.replace(cfg.signal, lookback=int(spec["lookback"]),
    #   skip=int(spec["skip"]))``, the same for ``cfg.sizing`` (``vol_window``,
    #   ``vol_estimator=str(spec["estimator"])``) and ``cfg.execution`` (``rebalance``,
    #   ``lag``); then ``dataclasses.replace(cfg, signal=..., sizing=..., execution=...)``.
    #   ``spec["..."]`` raises the ``KeyError`` by itself.
    #   Imports to add: ``dataclasses``.
    #   Verified by tests/test_robustness.py::TestTrialConfig.
    signal = dataclasses.replace(
        cfg.signal,
        lookback=int(spec["lookback"]),
        skip=int(spec["skip"]),
    )
    sizing = dataclasses.replace(
        cfg.sizing,
        vol_window=int(spec["vol_window"]),
        vol_estimator=str(spec["estimator"]),
    )
    execution = dataclasses.replace(
        cfg.execution,
        rebalance=str(spec["rebalance"]),
        lag=int(spec["lag"]),
    )
    return dataclasses.replace(cfg, signal=signal, sizing=sizing, execution=execution)


def evaluate_trial(returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run one configuration through the pipeline and measure it at every cost of the sweep.

    ``returns`` is the cleaned daily return panel (``data.clean_returns``; wide, indexed by
    date, columns = tickers); ``cfg`` carries the trial (:func:`trial_config`). The pipeline
    is the backtest stage's: ``tsmom`` on ``clean_price_index(returns)`` with
    ``cfg.signal``; ``size_positions`` with ``cfg.sizing`` and ``cfg.execution.rebalance``;
    then, for each ``cost`` of ``cfg.robustness.cost_bps_sweep`` in order,
    ``backtest.run(weights, returns, cfg.execution.lag, cost)``.

    One row per cost, columns exactly ``TRIAL_COLUMNS``, ``RangeIndex``, ``GRID_DTYPES``:

    * the six ``GRID_KEYS`` read from ``cfg`` (:func:`default_spec`) and ``cost_bps``;
    * over the trial's **life** (the rows where the engine's gross return is not NaN):
      ``first_date`` (its first row; ``NaT`` if the trial never holds a position),
      ``n_days``, and of the net returns at that cost ``ann_return``, ``ann_vol``, ``sharpe``
      (with ``cfg.inference.risk_free_rate``) and ``max_drawdown`` (``src.metrics``);
    * ``sharpe_is`` and ``n_days_is`` over the life's rows on or before
      ``cfg.robustness.is_end``; ``sharpe_oos`` and ``n_days_oos`` over those on or after
      ``cfg.robustness.oos_start``;
    * the book over the life, equal at every cost: ``ann_turnover`` (mean daily turnover
      times ``ANNUALISATION``), ``mean_gross_leverage`` and ``cap_bind_fraction`` (mean of
      the sizing's ``cap_binds``).

    A statistic with no rows to measure is NaN (the metric functions already say so). The
    in-sample numbers depend on no return after ``is_end``: positions there are formed from
    earlier data only, and the in-sample rows stop at ``is_end``.
    """
    # TODO 62: implement evaluate_trial.
    #   ``signal = tsmom(clean_price_index(returns), cfg.signal.lookback, cfg.signal.skip)``;
    #   ``sized = size_positions(signal, returns, cfg.sizing, cfg.execution.rebalance)``;
    #   ``rf = cfg.inference.risk_free_rate``; ``is_end = pd.Timestamp(cfg.robustness.is_end)``,
    #   ``oos_start`` likewise.
    #   For each ``cost`` in ``cfg.robustness.cost_bps_sweep``: ``res = run(sized.weights,
    #   returns, cfg.execution.lag, cost)``; ``life = res.gross_returns.notna()``; ``net =
    #   res.net_returns[life]``; ``ins, oos = net[net.index <= is_end], net[net.index >=
    #   oos_start]``; a dict ``{**default_spec(cfg), "cost_bps": float(cost), "first_date":
    #   net.index[0] if len(net) else pd.NaT, "n_days": len(net), "n_days_is": len(ins),
    #   "n_days_oos": len(oos), "ann_return": annualised_return(net), ...,
    #   "sharpe_is": sharpe(ins, rf), "sharpe_oos": sharpe(oos, rf), ...,
    #   "ann_turnover": float(res.turnover[life].mean() * ANNUALISATION),
    #   "mean_gross_leverage": float(res.gross_leverage[life].mean()),
    #   "cap_bind_fraction": float(sized.cap_binds[life].mean())}``.
    #   ``pd.DataFrame(rows, columns=list(TRIAL_COLUMNS)).astype({c: GRID_DTYPES[c] for c in
    #   TRIAL_COLUMNS})``. The life is the same at every cost (costs never create a NaN), so
    #   the book statistics come out equal; computing them inside the loop is fine.
    #   Imports to add: ``ANNUALISATION`` (``src.config``), ``run`` (``src.backtest``),
    #   ``clean_price_index`` (``src.data``), ``annualised_return``, ``annualised_vol``,
    #   ``max_drawdown``, ``sharpe`` (``src.metrics``), ``size_positions``
    #   (``src.portfolio``), ``tsmom`` (``src.signals``).
    #   Verified by tests/test_robustness.py::TestEvaluateTrial and
    #   ::TestIntegrityIsOosSelection.
    signal = tsmom(clean_price_index(returns), cfg.signal.lookback, cfg.signal.skip)
    sized = size_positions(signal, returns, cfg.sizing, cfg.execution.rebalance)
    rf = cfg.inference.risk_free_rate
    is_end = pd.Timestamp(cfg.robustness.is_end)
    oos_start = pd.Timestamp(cfg.robustness.oos_start)

    rows = []
    for cost in cfg.robustness.cost_bps_sweep:
        res = run(sized.weights, returns, cfg.execution.lag, cost)
        life = res.gross_returns.notna()
        net = res.net_returns[life]
        ins = net[net.index <= is_end]
        oos = net[net.index >= oos_start]
        rows.append(
            {
                **default_spec(cfg),
                "cost_bps": float(cost),
                "first_date": net.index[0] if len(net) else pd.NaT,
                "n_days": len(net),
                "n_days_is": len(ins),
                "n_days_oos": len(oos),
                "ann_return": annualised_return(net),
                "ann_vol": annualised_vol(net),
                "sharpe": sharpe(net, rf),
                "max_drawdown": max_drawdown(net),
                "sharpe_is": sharpe(ins, rf),
                "sharpe_oos": sharpe(oos, rf),
                "ann_turnover": float(res.turnover[life].mean() * ANNUALISATION),
                "mean_gross_leverage": float(res.gross_leverage[life].mean()),
                "cap_bind_fraction": float(sized.cap_binds[life].mean()),
            }
        )

    frame = pd.DataFrame(rows, columns=list(TRIAL_COLUMNS))
    return frame.astype({column: GRID_DTYPES[column] for column in TRIAL_COLUMNS})


def trial_fingerprint(cfg: Config, panel_sha256: str) -> str:
    """The identity of every input a cached grid row depends on, as a SHA-256 hex digest.

    The payload is exactly this dict, serialised with ``json.dumps(payload,
    sort_keys=True)`` and hashed with ``hashlib.sha256(...encode()).hexdigest()``::

        {"panel_sha256": panel_sha256, "clip": cfg.data.clip,
         "sigma_target_asset": ..., "sigma_target_portfolio": ..., "leverage_cap": ...,
         "cost_bps_sweep": list(cfg.robustness.cost_bps_sweep),
         "is_end": ..., "oos_start": ..., "risk_free_rate": cfg.inference.risk_free_rate}

    (the sizing values from ``cfg.sizing``, the dates from ``cfg.robustness``). The grid
    lists, the default trial, the default cost and the paths are left out on purpose: a
    trial's row does not depend on them, so extending the grid or changing the default
    keeps the cache. A change to the code is not detected; delete ``results/grid.parquet``
    after changing the pipeline.
    """
    # TODO 63: implement trial_fingerprint.
    #   Build the dict exactly as above (``list(...)`` for the sweep, so it serialises as a
    #   JSON list); ``hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    #   .hexdigest()``. The test re-hashes the documented payload, so every key and value
    #   must match.
    #   Imports to add: ``hashlib``, ``json``.
    #   Verified by tests/test_robustness.py::TestTrialFingerprint.
    payload = {
        "panel_sha256": panel_sha256,
        "clip": cfg.data.clip,
        "sigma_target_asset": cfg.sizing.sigma_target_asset,
        "sigma_target_portfolio": cfg.sizing.sigma_target_portfolio,
        "leverage_cap": cfg.sizing.leverage_cap,
        "cost_bps_sweep": list(cfg.robustness.cost_bps_sweep),
        "is_end": cfg.robustness.is_end,
        "oos_start": cfg.robustness.oos_start,
        "risk_free_rate": cfg.inference.risk_free_rate,
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def trial_stats(grid: pd.DataFrame, cost_bps: float) -> tuple[int, float]:
    """The Deflated Sharpe Ratio's two inputs from the grid, at one cost.

    Returns ``(n_trials, sharpe_std)``: the number of grid rows at ``cost_bps`` (one per
    trial, so the number of configurations evaluated) and the sample standard deviation
    (``ddof = 1``) of their annualised ``sharpe`` over the finite values, ``0.0`` when fewer
    than two are finite. Raises ``ValueError`` if no row has that cost.
    """
    # TODO 64: implement trial_stats.
    #   ``at = grid.loc[grid["cost_bps"] == cost_bps, "sharpe"]``; ``ValueError`` if empty;
    #   ``finite = at[np.isfinite(at)]``; ``std = float(finite.std(ddof=1)) if len(finite) >=
    #   2 else 0.0``; return ``(int(len(at)), std)``.
    #   Imports to add: ``numpy as np``.
    #   Verified by tests/test_robustness.py::TestTrialStats.
    at = grid.loc[grid["cost_bps"] == cost_bps, "sharpe"]
    if len(at) == 0:
        raise ValueError(f"no grid row has cost_bps {cost_bps}")

    finite = at[np.isfinite(at)]
    std = float(finite.std(ddof=1)) if len(finite) >= 2 else 0.0
    return int(len(at)), std


# --------------------------------------------------------------- F6.2 IS/OOS


def is_oos(grid: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the lookback on the in-sample years, and see what it does out of sample.

    The candidates are the grid rows at ``cfg.costs.cost_bps`` whose five other grid
    parameters (``vol_window``, ``estimator``, ``rebalance``, ``skip``, ``lag``) equal the
    a-priori default's (:func:`default_spec`): only ``L`` is selected (§5.3). Returns:

    * a table indexed by ``lookback`` (ascending), columns exactly ``IS_OOS_COLUMNS``:
      ``sharpe_is``, ``sharpe_oos``, ``degradation = sharpe_is - sharpe_oos``, the two day
      counts (``int``), ``selected`` (``True`` on the one selected row) and ``default``
      (``True`` on ``cfg.signal.lookback``'s row);
    * a summary dict with keys exactly ``IS_OOS_KEYS``: the selected lookback (``int``)
      and its ``sharpe_is``, ``sharpe_oos`` and degradation; the same four for the default
      lookback; and ``oos_gap = selected_oos - default_oos`` (what selection gained, if
      positive, or cost, if negative, out of sample).

    The selected lookback has the highest finite ``sharpe_is``; a tie goes to the shorter
    lookback. The selection reads nothing but ``sharpe_is``: the out-of-sample columns cannot
    influence it. Raises ``ValueError`` if there is no candidate, the default lookback is
    not among them, or no ``sharpe_is`` is finite.
    """
    # TODO 65: implement is_oos.
    #   ``spec = default_spec(cfg)``; ``others = {k: v for k, v in spec.items() if k !=
    #   "lookback"}``; ``rows = grid[matches(grid, {**others, "cost_bps": cfg.costs.cost_bps})]
    #   .sort_values("lookback")`` (``ValueError`` if empty or without ``spec["lookback"]``).
    #   ``table = rows.set_index("lookback")[["sharpe_is", "sharpe_oos", "n_days_is",
    #   "n_days_oos"]]``; add ``degradation``; ``ValueError`` if ``table["sharpe_is"]`` has
    #   no finite value; ``best = table["sharpe_is"].idxmax()`` (``idxmax`` skips NaN and
    #   returns the first maximum, which on an ascending index is the shorter lookback);
    #   ``selected = table.index == best``; ``default = table.index == spec["lookback"]``;
    #   reorder to ``list(IS_OOS_COLUMNS)``. Then the summary in ``IS_OOS_KEYS`` order with
    #   ``int(...)`` lookbacks and ``float(...)`` Sharpe ratios.
    #   Verified by tests/test_robustness.py::TestIsOos and ::TestIntegrityIsOosSelection.
    spec = default_spec(cfg)
    others = {k: v for k, v in spec.items() if k != "lookback"}
    rows = grid[matches(grid, {**others, "cost_bps": cfg.costs.cost_bps})]
    rows = rows.sort_values("lookback")
    if len(rows) == 0:
        raise ValueError("no IS/OOS candidate: no grid row matches the default's other parameters")
    if spec["lookback"] not in rows["lookback"].tolist():
        raise ValueError(f"the default lookback {spec['lookback']} is not among the candidates")

    columns = ["sharpe_is", "sharpe_oos", "n_days_is", "n_days_oos"]
    table = rows.set_index("lookback")[columns].copy()
    table["degradation"] = table["sharpe_is"] - table["sharpe_oos"]
    if not np.isfinite(table["sharpe_is"]).any():
        raise ValueError("no candidate has a finite in-sample Sharpe ratio")

    best = table["sharpe_is"].idxmax()
    table["selected"] = table.index == best
    table["default"] = table.index == spec["lookback"]
    table = table[list(IS_OOS_COLUMNS)]

    sel = table.loc[best]
    dft = table.loc[spec["lookback"]]
    summary = {
        "selected_lookback": int(best),
        "selected_is": float(sel["sharpe_is"]),
        "selected_oos": float(sel["sharpe_oos"]),
        "selected_degradation": float(sel["degradation"]),
        "default_lookback": int(spec["lookback"]),
        "default_is": float(dft["sharpe_is"]),
        "default_oos": float(dft["sharpe_oos"]),
        "default_degradation": float(dft["degradation"]),
        "oos_gap": float(sel["sharpe_oos"] - dft["sharpe_oos"]),
    }
    return table, summary


# --------------------------------------------------------------- F6.4 sweeps


def cost_sweep(grid: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Net results against the cost assumption.

    Indexed by ``cost_bps`` (each cost of the grid, ascending), columns exactly
    ``COST_SWEEP_COLUMNS``: the default trial's (:func:`default_spec`) ``ann_return``,
    ``sharpe`` and ``max_drawdown`` at that cost; ``grid_median_sharpe``, the median
    ``sharpe`` of every trial at that cost; ``grid_share_positive``, the share of those
    trials with ``sharpe > 0``. The cost-0 row is the gross result. Raises ``ValueError``
    if the default trial is not in the grid.
    """
    # TODO 66: implement cost_sweep.
    #   ``mine = grid[matches(grid, default_spec(cfg))].set_index("cost_bps").sort_index()``
    #   (``ValueError`` if empty); ``by_cost = grid.groupby("cost_bps")["sharpe"]``;
    #   ``out = mine[["ann_return", "sharpe", "max_drawdown"]].assign(grid_median_sharpe=
    #   by_cost.median(), grid_share_positive=by_cost.apply(lambda x: float((x > 0).mean())))``
    #   (``assign`` aligns the two groupby results on the ``cost_bps`` index).
    #   Verified by tests/test_robustness.py::TestCostSweep.
    mine = grid[matches(grid, default_spec(cfg))].set_index("cost_bps").sort_index()
    if len(mine) == 0:
        raise ValueError("the default trial is not in the grid")

    by_cost = grid.groupby("cost_bps")["sharpe"]
    out = mine[["ann_return", "sharpe", "max_drawdown"]].assign(
        grid_median_sharpe=by_cost.median(),
        grid_share_positive=by_cost.apply(lambda x: float((x > 0).mean())),
    )
    return out[list(COST_SWEEP_COLUMNS)]


def lag_sweep(grid: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Results against the execution lag.

    Indexed by ``lag`` (each lag of the grid, ascending), columns exactly
    ``LAG_SWEEP_COLUMNS``: for the default trial with that lag (its other five grid
    parameters the defaults), ``sharpe_gross`` (the cost-0 row; NaN if the sweep has no
    zero cost), ``sharpe_net`` and ``ann_return_net`` (at ``cfg.costs.cost_bps``); and
    ``grid_median_net``, the median ``sharpe`` at ``cfg.costs.cost_bps`` of every trial
    with that lag. Raises ``ValueError`` if a lag has no default-trial row at
    ``cfg.costs.cost_bps``.
    """
    # TODO 67: implement lag_sweep.
    #   ``others = {k: v for k, v in default_spec(cfg).items() if k != "lag"}``; for each
    #   ``lag`` in ``sorted(grid["lag"].unique())``: ``net = grid[matches(grid, {**others,
    #   "lag": lag, "cost_bps": cfg.costs.cost_bps})]`` (``ValueError`` if empty), ``gross``
    #   the same at ``cost_bps = 0.0`` (NaN when empty), and the median ``sharpe`` of
    #   ``grid[matches(grid, {"lag": lag, "cost_bps": cfg.costs.cost_bps})]``. One dict per
    #   lag; ``pd.DataFrame(rows, index=pd.Index(lags, name="lag"),
    #   columns=list(LAG_SWEEP_COLUMNS))``, lags as ``int``.
    #   Verified by tests/test_robustness.py::TestLagSweep.
    others = {k: v for k, v in default_spec(cfg).items() if k != "lag"}
    cost = cfg.costs.cost_bps
    lags = [int(lag) for lag in sorted(grid["lag"].unique())]

    rows = []
    for lag in lags:
        net = grid[matches(grid, {**others, "lag": lag, "cost_bps": cost})]
        if len(net) == 0:
            raise ValueError(f"no default-trial row at lag {lag} and cost {cost}")
        gross = grid[matches(grid, {**others, "lag": lag, "cost_bps": 0.0})]
        same_lag = grid[matches(grid, {"lag": lag, "cost_bps": cost})]
        rows.append(
            {
                "sharpe_gross": float(gross["sharpe"].iloc[0]) if len(gross) else float("nan"),
                "sharpe_net": float(net["sharpe"].iloc[0]),
                "ann_return_net": float(net["ann_return"].iloc[0]),
                "grid_median_net": float(same_lag["sharpe"].median()),
            }
        )

    return pd.DataFrame(rows, index=pd.Index(lags, name="lag"), columns=list(LAG_SWEEP_COLUMNS))


def grid_distribution(grid: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Where the default sits among all the trials, at ``cfg.costs.cost_bps``.

    A dict with keys exactly ``DISTRIBUTION_KEYS``, over the grid rows at that cost:
    ``n_trials`` and ``sharpe_std`` (:func:`trial_stats`); the ``sharpe`` minimum, quartiles
    (``Series.quantile`` at 0.25, 0.5, 0.75, linear) and maximum; ``share_positive`` (share
    with ``sharpe > 0``); ``default_sharpe``; ``default_rank`` (``int``, 1 = best: one plus
    the number of trials with a strictly higher ``sharpe``); and ``best_trial`` and
    ``worst_trial``, the :func:`trial_label` of the trials with the highest and lowest
    ``sharpe`` (the first such row in grid order on a tie). Statistics over the finite
    Sharpe ratios. Raises ``ValueError`` if the default trial is not in the grid.
    """
    # TODO 68: implement grid_distribution.
    #   ``at = grid[grid["cost_bps"] == cfg.costs.cost_bps]``; ``mine = at[matches(at,
    #   default_spec(cfg))]`` (``ValueError`` if empty); ``n, std = trial_stats(grid,
    #   cfg.costs.cost_bps)``; ``x = at["sharpe"]``; ``finite = x[np.isfinite(x)]``; the
    #   quantiles with ``finite.quantile(q)``; ``default = float(mine["sharpe"].iloc[0])``;
    #   ``rank = 1 + int((finite > default).sum())``; the best and worst rows are
    #   ``at.loc[finite.idxmax()]`` and ``at.loc[finite.idxmin()]`` (first on a tie), each
    #   through ``trial_label``. Floats as ``float(...)``.
    #   Verified by tests/test_robustness.py::TestGridDistribution.
    at = grid[grid["cost_bps"] == cfg.costs.cost_bps]
    mine = at[matches(at, default_spec(cfg))]
    if len(mine) == 0:
        raise ValueError("the default trial is not in the grid")

    n, std = trial_stats(grid, cfg.costs.cost_bps)
    x = at["sharpe"]
    finite = x[np.isfinite(x)]
    default = float(mine["sharpe"].iloc[0])
    rank = 1 + int((finite > default).sum())

    return {
        "n_trials": n,
        "sharpe_std": std,
        "sharpe_min": float(finite.min()),
        "sharpe_q25": float(finite.quantile(0.25)),
        "sharpe_median": float(finite.quantile(0.5)),
        "sharpe_q75": float(finite.quantile(0.75)),
        "sharpe_max": float(finite.max()),
        "share_positive": float((finite > 0).mean()),
        "default_sharpe": default,
        "default_rank": rank,
        "best_trial": trial_label(at.loc[finite.idxmax()]),
        "worst_trial": trial_label(at.loc[finite.idxmin()]),
    }


__all__ = [
    "COST_SWEEP_COLUMNS",
    "DISTRIBUTION_KEYS",
    "GRID_COLUMNS",
    "GRID_DTYPES",
    "GRID_KEYS",
    "IS_OOS_COLUMNS",
    "IS_OOS_KEYS",
    "LAG_SWEEP_COLUMNS",
    "TRIAL_COLUMNS",
    "cost_sweep",
    "default_spec",
    "evaluate_trial",
    "grid_distribution",
    "grid_specs",
    "is_oos",
    "lag_sweep",
    "matches",
    "trial_config",
    "trial_fingerprint",
    "trial_label",
    "trial_stats",
]
