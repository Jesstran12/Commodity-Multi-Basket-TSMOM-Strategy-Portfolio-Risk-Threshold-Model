"""Command-line entry point: ``python -m src.cli <stage>``, and the stage functions it runs.

Each Makefile target maps to one stage. Stages are registered here as they land; an
unimplemented stage exits with a clear message rather than silently doing nothing. The
stage functions that read the snapshot and write ``results/`` (``run_backtest``,
``run_benchmarks``, ``run_metrics``, ``load_or_run_grid``, ``run_robustness``) live here
because this is one of the four modules allowed to do I/O (PROJECT_LOG.md §13); the engine,
the benchmarks, the metrics, the inference statistics, the robustness and sub-period
analyses and the report formatters stay pure.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtest import BacktestResult, backtest_summary, run
from src.benchmark import ew_vol_targeted_weights, ew_weights, external_returns
from src.config import ANNUALISATION, Config, dump_resolved, load_config
from src.data import clean_price_index, clean_returns, load_benchmark, sha256_of
from src.inference import (
    deflated_sharpe,
    expected_max_sharpe,
    inference_summary,
    probabilistic_sharpe,
)
from src.metrics import performance, summarise
from src.portfolio import SizingResult, size_positions, sizing_summary
from src.report import format_metrics_md, is_oos_md, metrics_json, subperiods_md, sweeps_md
from src.robustness import (
    GRID_COLUMNS,
    GRID_DTYPES,
    GRID_KEYS,
    cost_sweep,
    evaluate_trial,
    grid_distribution,
    grid_specs,
    is_oos,
    lag_sweep,
    trial_config,
    trial_fingerprint,
    trial_stats,
)
from src.signals import tsmom
from src.subperiods import (
    block_stats,
    calendar_year_returns,
    regime_labels,
    regime_stats,
    rolling_sharpe,
    rolling_summary,
)

ROOT = Path(__file__).resolve().parents[1]

RETURNS_FILE = "returns_daily.csv"
POSITIONS_FILE = "positions.csv"
BACKTEST_REPORT = "backtest.md"
BENCHMARKS_FILE = "benchmarks_daily.csv"
BENCHMARKS_REPORT = "benchmarks.md"
RESOLVED_CONFIG = "resolved_config.yaml"
METRICS_FILE = "metrics.json"
METRICS_TABLE = "metrics_table.md"
GRID_FILE = "grid.parquet"
IS_OOS_REPORT = "is_oos.md"
SUBPERIODS_REPORT = "subperiods.md"
SWEEPS_REPORT = "sweeps.md"
"""File names under ``paths.results_dir``. I/O settings, not strategy parameters."""

BENCHMARK_COLUMNS = (
    "ew_long_only_gross",
    "ew_long_only_net",
    "ew_vol_targeted_gross",
    "ew_vol_targeted_net",
    "external",
)
"""Columns of ``benchmarks_daily.csv``, in order."""

METRICS_SERIES = (
    "Strategy (gross)",
    "Strategy (net)",
    "EW long-only (net)",
    "EW vol-targeted (net)",
)
"""The first four series of the metrics table, in order; the external index's ticker is the
fifth."""

BOOK_KEYS = ("ann_turnover", "mean_gross_leverage", "mean_n_live", "cap_bind_fraction")
"""The §5.1 statistics of a book rather than of a return series, in the order they follow
``METRIC_KEYS`` in each metrics entry."""

TRIALS_KEYS = ("n_trials", "sharpe_std", "sharpe_hurdle", "psr", "dsr")
"""Keys of :class:`RobustnessRun`'s ``trials`` dict, in order: the grid's trial count and
Sharpe spread, and the default's net hurdle, PSR and DSR."""


@dataclass(frozen=True)
class BacktestRun:
    """What ``run_backtest`` computes: the inputs, the sizing and the engine's result."""

    returns: pd.DataFrame
    signal: pd.DataFrame
    sizing: SizingResult
    result: BacktestResult
    summary: dict[str, object]


@dataclass(frozen=True)
class BenchmarkRun:
    """What ``run_benchmarks`` computes: the three benchmarks on the strategy's life."""

    start: pd.Timestamp
    ew_long_only: BacktestResult
    ew_vol_targeted: BacktestResult
    ew_cap_binds: pd.Series
    external_ticker: str
    external: pd.Series
    table: pd.DataFrame
    summary: dict[str, dict[str, object]]


@dataclass(frozen=True)
class MetricsRun:
    """What ``run_metrics`` computes: the two earlier stages' runs, the five daily return
    series it measured (columns = series names, rows = the strategy's life) and the metrics
    entry of each (``{series name: {key: value}}``, in column order)."""

    backtest: BacktestRun
    benchmarks: BenchmarkRun
    returns: pd.DataFrame
    metrics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class GridRun:
    """What ``load_or_run_grid`` returns: the grid frame (``GRID_COLUMNS``, one row per trial
    and cost), the number of trials, how many were computed and how many read from the
    cache in this call, and the fingerprint the rows carry."""

    grid: pd.DataFrame
    n_trials: int
    n_computed: int
    n_cached: int
    fingerprint: str


@dataclass(frozen=True)
class RobustnessRun:
    """What ``run_robustness`` computes: the grid and every table of the three reports.

    ``trials`` has keys ``TRIALS_KEYS``; ``is_oos_table`` and ``is_oos_summary`` are
    ``robustness.is_oos``'s; ``cost_sweep``, ``lag_sweep`` and ``distribution`` the sweeps';
    ``returns`` the five daily series (as ``MetricsRun.returns``); ``blocks``, ``years``,
    ``rolling``, ``rolling_summary``, ``labels`` and ``regimes`` the sub-period tables."""

    grid: GridRun
    trials: dict[str, object]
    is_oos_table: pd.DataFrame
    is_oos_summary: dict[str, object]
    cost_sweep: pd.DataFrame
    lag_sweep: pd.DataFrame
    distribution: dict[str, object]
    returns: pd.DataFrame
    blocks: pd.DataFrame
    years: pd.DataFrame
    rolling: pd.DataFrame
    rolling_summary: pd.DataFrame
    labels: pd.Series
    regimes: pd.DataFrame


# ---------------------------------------------------------------------- stage helpers


def format_summary_md(title: str, sections: dict[str, dict[str, object]]) -> str:
    """Render ``{section: {key: value}}`` as Markdown: a heading, then one table per
    section with ``Quantity`` / ``Value`` columns. Floats print with 6 significant digits,
    dates as ``YYYY-MM-DD``, everything else with ``str``. Provided complete (not a TODO)."""
    lines = [f"# {title}", ""]
    for section, values in sections.items():
        lines += [f"## {section}", "", "| Quantity | Value |", "|---|---|"]
        for key, value in values.items():
            if isinstance(value, pd.Timestamp):
                text = f"{value:%Y-%m-%d}"
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            lines.append(f"| {key} | {text} |")
        lines.append("")
    return "\n".join(lines)


def _read_panel(cfg: Config, root: Path) -> pd.DataFrame:
    """The committed price panel as a wide frame indexed by date."""
    return pd.read_csv(root / cfg.data.panel_path, index_col="date", parse_dates=True)


# --------------------------------------------------------------- F4.4 backtest stage


def run_backtest(cfg: Config, root: str | Path) -> BacktestRun:
    """``make backtest``: the default strategy from the snapshot to ``results/``. No network.

    Steps, all relative to ``root``:

    1. read ``cfg.data.panel_path``; ``clean_returns`` with ``cfg.data.clip``; ``tsmom`` on
       the ``clean_price_index`` with ``cfg.signal``; ``size_positions`` with ``cfg.sizing``
       and ``cfg.execution.rebalance``; ``backtest.run`` on the held weights with
       ``cfg.execution.lag`` and ``cfg.costs.cost_bps``;
    2. ``backtest_summary`` with ``cfg.inference.risk_free_rate``;
    3. write, under ``root / cfg.paths.results_dir`` (created if needed):
       ``RETURNS_FILE``, the strategy's daily series over its life (from the first row with a
       position to the end) as a wide CSV indexed by ``date`` with columns exactly
       ``gross, net, cost, turnover, gross_leverage, n_live``; ``POSITIONS_FILE``, the
       positions over the same rows; ``BACKTEST_REPORT`` via :func:`format_summary_md` with
       a ``"Strategy"`` section (the summary) and a ``"Sizing"`` section
       (``sizing_summary`` of the sizing result); and ``RESOLVED_CONFIG`` via
       ``dump_resolved``.

    Returns a :class:`BacktestRun`. Deterministic: two runs on the same snapshot and config
    write byte-identical CSVs (the §8 determinism test).
    """
    # TODO 40: implement run_backtest.
    #   ``root = Path(root)``; ``panel = _read_panel(cfg, root)``; ``returns, _ =
    #   clean_returns(panel, cfg.data.clip)``; ``signal = tsmom(clean_price_index(returns),
    #   cfg.signal.lookback, cfg.signal.skip)``; ``sized = size_positions(signal, returns,
    #   cfg.sizing, cfg.execution.rebalance)``; ``result = run(sized.weights, returns,
    #   cfg.execution.lag, cfg.costs.cost_bps)``; ``summary = backtest_summary(result,
    #   cfg.inference.risk_free_rate)``.
    #   Files: ``life = result.gross_returns.notna()``; ``daily = pd.concat([result.gross_returns,
    #   result.net_returns, result.costs, result.turnover, result.gross_leverage,
    #   result.n_live], axis=1)[life]`` (the series names give the column names);
    #   ``daily.to_csv(results_dir / RETURNS_FILE)``; ``result.positions[life].to_csv(...)``;
    #   ``(results_dir / BACKTEST_REPORT).write_text(format_summary_md("Backtest", {"Strategy":
    #   summary, "Sizing": sizing_summary(sized)}))``; ``dump_resolved(cfg, results_dir /
    #   RESOLVED_CONFIG)``. ``results_dir.mkdir(parents=True, exist_ok=True)`` first.
    #   Imports to add: ``run``, ``backtest_summary`` (``src.backtest``), ``clean_returns``,
    #   ``clean_price_index`` (``src.data``), ``size_positions``, ``sizing_summary``
    #   (``src.portfolio``), ``tsmom`` (``src.signals``); for TODO 41 also ``ew_weights``,
    #   ``ew_vol_targeted_weights``, ``external_returns`` (``src.benchmark``),
    #   ``load_benchmark`` (``src.data``), ``performance`` (``src.metrics``) and
    #   ``ANNUALISATION`` (``src.config``).
    #   Verified by tests/test_cli.py::TestRunBacktest and ::TestIntegrityDeterminism.
    root = Path(root)
    panel = _read_panel(cfg, root)
    returns, _ = clean_returns(panel, cfg.data.clip)
    signal = tsmom(clean_price_index(returns), cfg.signal.lookback, cfg.signal.skip)
    sized = size_positions(signal, returns, cfg.sizing, cfg.execution.rebalance)
    result = run(sized.weights, returns, cfg.execution.lag, cfg.costs.cost_bps)
    summary = backtest_summary(result, cfg.inference.risk_free_rate)

    results_dir = root / cfg.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    life = result.gross_returns.notna()
    daily = pd.concat(
        [
            result.gross_returns,
            result.net_returns,
            result.costs,
            result.turnover,
            result.gross_leverage,
            result.n_live,
        ],
        axis=1,
    )[life]
    daily.to_csv(results_dir / RETURNS_FILE)
    result.positions[life].to_csv(results_dir / POSITIONS_FILE)
    report = format_summary_md("Backtest", {"Strategy": summary, "Sizing": sizing_summary(sized)})
    (results_dir / BACKTEST_REPORT).write_text(report)
    dump_resolved(cfg, results_dir / RESOLVED_CONFIG)
    return BacktestRun(returns, signal, sized, result, summary)


def run_benchmarks(cfg: Config, root: str | Path) -> BenchmarkRun:
    """``make benchmarks``: the three §4 benchmarks on the strategy's life, to ``results/``.

    Steps, all relative to ``root``:

    1. :func:`run_backtest`'s pipeline up to the engine (call it; it is cheap and keeps the
       two stages consistent), to get the cleaned returns and the strategy's first day with
       a position, ``start`` (first non-NaN gross return);
    2. the equal-weight long-only basket: ``backtest.run`` on
       ``ew_weights(returns, cfg.execution.rebalance)`` with the strategy's lag and costs;
       the vol-targeted basket: ``backtest.run`` on the weights of
       ``ew_vol_targeted_weights(returns, cfg.sizing, cfg.execution.rebalance)``, keeping
       its cap flag; the external index: ``load_benchmark(root / cfg.data.benchmark_path)``
       and :func:`external_returns` on the panel calendar with ``cfg.data.clip``;
    3. ``table``: a frame on the rows from ``start`` to the end of the calendar with
       columns exactly ``BENCHMARK_COLUMNS`` (gross and net of each basket, the external
       series), written as ``BENCHMARKS_FILE``;
    4. ``summary``: ``{"EW long-only (gross)": performance(...), "EW long-only (net)": ...,
       "EW vol-targeted (gross)": ..., "EW vol-targeted (net)": ..., "<ticker>": ...}`` of the
       five table columns with ``cfg.inference.risk_free_rate``, each extended with the
       basket's ``ann_turnover`` (mean daily turnover over the rows times 252) and, for the
       vol-targeted one, ``n_cap_bind_days`` over the rows; the external entry carries
       ``n_missing`` (rows in the table without a return, inside the index's own span) and
       ``n_after_last_print`` (rows after its last print). Written as ``BENCHMARKS_REPORT``
       via :func:`format_summary_md`.

    Returns a :class:`BenchmarkRun`.
    """
    # TODO 41: implement run_benchmarks.
    #   ``bt = run_backtest(cfg, root)``; ``returns = bt.returns``; ``start =
    #   bt.result.gross_returns.first_valid_index()``; ``lag, bps = cfg.execution.lag,
    #   cfg.costs.cost_bps``; ``lo = run(ew_weights(returns, cfg.execution.rebalance), returns,
    #   lag, bps)``; ``vt_w, vt_binds = ew_vol_targeted_weights(returns, cfg.sizing,
    #   cfg.execution.rebalance)``; ``vt = run(vt_w, returns, lag, bps)``; ``ticker, prices =
    #   load_benchmark(root / cfg.data.benchmark_path)``; ``ext = external_returns(prices,
    #   returns.index, cfg.data.clip)``.
    #   ``table = pd.DataFrame({...}, columns=list(BENCHMARK_COLUMNS)).loc[start:]`` from the
    #   five series; summary entries from ``performance(table[c], rf)`` plus the extras
    #   (``float(lo.turnover.loc[start:].mean() * ANNUALISATION)`` and so on; the external
    #   extras: ``span = ext.loc[start:].dropna()``; ``inside = table.index[(table.index >=
    #   span.index[0]) & (table.index <= span.index[-1])]``; ``n_missing = int(ext.loc[inside]
    #   .isna().sum())``; ``n_after_last_print = int((table.index > span.index[-1]).sum())``).
    #   Write the two files; return the ``BenchmarkRun``.
    #   Verified by tests/test_cli.py::TestRunBenchmarks.
    root = Path(root)
    bt = run_backtest(cfg, root)
    returns = bt.returns
    start = bt.result.gross_returns.first_valid_index()
    lag, bps = cfg.execution.lag, cfg.costs.cost_bps
    rf = cfg.inference.risk_free_rate

    lo = run(ew_weights(returns, cfg.execution.rebalance), returns, lag, bps)
    vt_w, vt_binds = ew_vol_targeted_weights(returns, cfg.sizing, cfg.execution.rebalance)
    vt = run(vt_w, returns, lag, bps)
    ticker, prices = load_benchmark(root / cfg.data.benchmark_path)
    ext = external_returns(prices, returns.index, cfg.data.clip)

    table = pd.DataFrame(
        {
            "ew_long_only_gross": lo.gross_returns,
            "ew_long_only_net": lo.net_returns,
            "ew_vol_targeted_gross": vt.gross_returns,
            "ew_vol_targeted_net": vt.net_returns,
            "external": ext,
        },
        columns=list(BENCHMARK_COLUMNS),
    ).loc[start:]

    lo_turnover = float(lo.turnover.loc[start:].mean() * ANNUALISATION)
    vt_turnover = float(vt.turnover.loc[start:].mean() * ANNUALISATION)
    n_cap_bind_days = int(vt_binds.loc[start:].sum())
    span = ext.loc[start:].dropna()
    inside = table.index[(table.index >= span.index[0]) & (table.index <= span.index[-1])]
    n_missing = int(ext.loc[inside].isna().sum())
    n_after_last_print = int((table.index > span.index[-1]).sum())

    summary: dict[str, dict[str, object]] = {
        "EW long-only (gross)": {
            **performance(table["ew_long_only_gross"], rf),
            "ann_turnover": lo_turnover,
        },
        "EW long-only (net)": {
            **performance(table["ew_long_only_net"], rf),
            "ann_turnover": lo_turnover,
        },
        "EW vol-targeted (gross)": {
            **performance(table["ew_vol_targeted_gross"], rf),
            "ann_turnover": vt_turnover,
            "n_cap_bind_days": n_cap_bind_days,
        },
        "EW vol-targeted (net)": {
            **performance(table["ew_vol_targeted_net"], rf),
            "ann_turnover": vt_turnover,
            "n_cap_bind_days": n_cap_bind_days,
        },
        ticker: {
            **performance(table["external"], rf),
            "n_missing": n_missing,
            "n_after_last_print": n_after_last_print,
        },
    }

    results_dir = root / cfg.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / BENCHMARKS_FILE)
    (results_dir / BENCHMARKS_REPORT).write_text(format_summary_md("Benchmarks", summary))
    return BenchmarkRun(start, lo, vt, vt_binds, ticker, ext, table, summary)


# ---------------------------------------------------------------- F5.3 metrics stage


def run_metrics(
    cfg: Config, root: str | Path, n_trials: int = 1, trials_sharpe_std: float = 0.0
) -> MetricsRun:
    """``make metrics``: every §5.1 and §5.2 statistic of the strategy (gross and net) and
    the three benchmarks, to ``results/metrics.json`` and ``results/metrics_table.md``.

    Steps, all relative to ``root``:

    1. ``bm = run_benchmarks(cfg, root)`` and ``bt = run_backtest(cfg, root)`` (both write
       their stage's files again, identically; each takes about a second and keeps every
       stage on the same inputs); ``life`` = the strategy's life (its non-NaN gross return
       rows);
    2. ``returns``: a frame on the life's rows with one column per series, named
       ``METRICS_SERIES`` then the external ticker: the strategy's gross and net returns,
       the ``ew_long_only_net``, ``ew_vol_targeted_net`` and ``external`` columns of the
       benchmark table;
    3. for each column, the entry ``{**summarise(col, rf), **book, **inference_summary(col,
       cfg.inference, n, std)}`` with ``rf = cfg.inference.risk_free_rate``, ``(n, std) =
       (n_trials, trials_sharpe_std)`` for the two strategy columns and ``(1, 0.0)`` for the
       three benchmarks, and ``book`` the ``BOOK_KEYS`` statistics over the life rows: for
       the strategy (both columns) ``ann_turnover``, ``mean_gross_leverage`` and
       ``mean_n_live`` from the backtest summary and ``cap_bind_fraction`` the mean of the
       sizing's ``cap_binds`` over the life; for each basket the same four from its
       own ``BacktestResult`` (mean daily turnover times 252, mean gross leverage, mean
       ``n_live``) and its cap flag (``NaN`` for the long-only basket, which has no cap);
       all four ``NaN`` for the external index;
    4. write ``METRICS_FILE`` (``metrics_json``) and ``METRICS_TABLE``
       (``format_metrics_md``) under ``root / cfg.paths.results_dir``.

    ``n_trials`` and ``trials_sharpe_std`` feed the Deflated Sharpe Ratio of the strategy:
    the ``metrics`` stage passes the robustness grid's (``trial_stats`` of
    :func:`load_or_run_grid`'s grid), and the defaults (one trial) are the honest values
    without a grid. The benchmarks were not selected from a search, so they are always
    measured at one trial. Returns a :class:`MetricsRun`. Deterministic: the bootstrap is
    seeded from ``cfg.inference``.
    """
    # TODO 59: implement run_metrics.
    #   ``root = Path(root)``; ``bm = run_benchmarks(cfg, root)``; ``bt = run_backtest(cfg,
    #   root)``; ``rf = cfg.inference.risk_free_rate``; ``life =
    #   bt.result.gross_returns.notna()``; ``rows = bt.result.gross_returns.index[life]``.
    #   ``names = [*METRICS_SERIES, bm.external_ticker]``; ``returns = pd.DataFrame({names[0]:
    #   bt.result.gross_returns[life], names[1]: bt.result.net_returns[life], names[2]:
    #   bm.table["ew_long_only_net"], names[3]: bm.table["ew_vol_targeted_net"], names[4]:
    #   bm.table["external"]}, index=rows, columns=names)``.
    #   Book statistics: a small helper ``book(result, cap_binds)`` returning
    #   ``{"ann_turnover": float(result.turnover[rows].mean() * ANNUALISATION),
    #   "mean_gross_leverage": float(result.gross_leverage[rows].mean()), "mean_n_live":
    #   float(result.n_live[rows].mean()), "cap_bind_fraction": float(cap_binds[rows].mean())
    #   if cap_binds is not None else float("nan")}`` works for all three books: the strategy
    #   with ``bt.sizing.cap_binds`` (its turnover, leverage and ``n_live`` means equal the
    #   backtest summary's by construction), the long-only basket with ``None``, the
    #   vol-targeted one with ``bm.ew_cap_binds``; the external index gets
    #   ``dict.fromkeys(BOOK_KEYS, float("nan"))``.
    #   ``metrics = {name: {**summarise(returns[name], rf), **book_of[name],
    #   **inference_summary(returns[name], cfg.inference, *trials_of[name])} for name in
    #   names}`` with ``trials_of`` giving ``(n_trials, trials_sharpe_std)`` for the two
    #   strategy names and ``(1, 0.0)`` for the rest; write the two files (``mkdir`` first);
    #   return the ``MetricsRun``.
    #   Imports to add: ``summarise`` (``src.metrics``), ``inference_summary``
    #   (``src.inference``), ``format_metrics_md`` and ``metrics_json`` (``src.report``).
    #   Verified by tests/test_cli.py::TestRunMetrics, ::TestMetricsSnapshot and
    #   ::TestIntegrityDeterminism.
    root = Path(root)
    bm = run_benchmarks(cfg, root)
    bt = run_backtest(cfg, root)
    rf = cfg.inference.risk_free_rate
    life = bt.result.gross_returns.notna()
    rows = bt.result.gross_returns.index[life]
    names = [*METRICS_SERIES, bm.external_ticker]
    returns = pd.DataFrame(
        {
            names[0]: bt.result.gross_returns[life],
            names[1]: bt.result.net_returns[life],
            names[2]: bm.table["ew_long_only_net"],
            names[3]: bm.table["ew_vol_targeted_net"],
            names[4]: bm.table["external"],
        },
        index=rows,
        columns=names,
    )

    def book(result: BacktestResult, cap_binds: pd.Series | None) -> dict[str, object]:
        return {
            "ann_turnover": float(result.turnover[rows].mean() * ANNUALISATION),
            "mean_gross_leverage": float(result.gross_leverage[rows].mean()),
            "mean_n_live": float(result.n_live[rows].mean()),
            "cap_bind_fraction": (
                float(cap_binds[rows].mean()) if cap_binds is not None else float("nan")
            ),
        }

    book_of = {
        names[0]: book(bt.result, bt.sizing.cap_binds),
        names[1]: book(bt.result, bt.sizing.cap_binds),
        names[2]: book(bm.ew_long_only, None),
        names[3]: book(bm.ew_vol_targeted, bm.ew_cap_binds),
        names[4]: dict.fromkeys(BOOK_KEYS, float("nan")),
    }
    trials_of = {name: (1, 0.0) for name in names}
    trials_of[names[0]] = trials_of[names[1]] = (n_trials, trials_sharpe_std)
    metrics = {
        name: {
            **summarise(returns[name], rf),
            **book_of[name],
            **inference_summary(returns[name], cfg.inference, *trials_of[name]),
        }
        for name in names
    }
    out = root / cfg.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / METRICS_FILE).write_text(metrics_json(metrics))
    (out / METRICS_TABLE).write_text(format_metrics_md(metrics))
    return MetricsRun(backtest=bt, benchmarks=bm, returns=returns, metrics=metrics)


# ------------------------------------------------------------ F6.1 grid stage


def load_or_run_grid(cfg: Config, root: str | Path, resume: bool = True) -> GridRun:
    """The robustness grid (F6.1), resumable through ``results/grid.parquet``. No network.

    Steps, all relative to ``root``:

    1. read ``cfg.data.panel_path`` and ``clean_returns`` it with ``cfg.data.clip``;
       ``fingerprint = trial_fingerprint(cfg, sha256_of(<the panel file>))``; ``specs =
       grid_specs(cfg.robustness)``;
    2. the cache: if ``resume`` and ``GRID_FILE`` exists under ``root /
       cfg.paths.results_dir``, read it (``pd.read_parquet``) and keep only the rows whose
       ``fingerprint`` equals this one; otherwise start with no cached rows;
    3. for each trial of ``specs``, in order: if the cache has rows with its ``GRID_KEYS``
       values, reuse them; otherwise compute ``evaluate_trial(returns, trial_config(cfg,
       spec))`` (the ``evaluate_trial`` imported into this module), add the ``fingerprint``
       column, and **write the file straight away** with every row known so far (the
       cached rows kept plus every trial computed in this call, in any order), so that an
       interrupted run loses at most the trial in progress and the next call resumes;
    4. the grid: the rows of every trial of ``specs``, in trial order and within a trial in
       ``cost_bps_sweep`` order, columns exactly ``GRID_COLUMNS`` with ``GRID_DTYPES`` and a
       ``RangeIndex``, written to ``GRID_FILE`` with ``to_parquet(index=False)`` (results
       directory created if needed). The file then holds exactly this grid: rows of a trial
       no longer in the grid, or of another fingerprint, are gone.

    ``resume=False`` ignores any existing file and recomputes every trial. Returns a
    :class:`GridRun` with ``n_trials = len(specs)`` and the counts of trials computed and
    reused in this call (they add up to ``n_trials``). A trial's rows are equal whether
    computed or read back (the dtypes make them so).
    """
    # TODO 76: implement load_or_run_grid.
    #   ``root = Path(root)``; ``panel_path = root / cfg.data.panel_path``; ``returns, _ =
    #   clean_returns(_read_panel(cfg, root), cfg.data.clip)``; ``fp =
    #   trial_fingerprint(cfg, sha256_of(panel_path))``; ``specs = grid_specs(cfg.robustness)``;
    #   ``out = root / cfg.paths.results_dir``; ``out.mkdir(parents=True, exist_ok=True)``;
    #   ``path = out / GRID_FILE``.
    #   Cache: ``cached = pd.read_parquet(path)`` when ``resume and path.exists()``, keeping
    #   ``cached[cached["fingerprint"] == fp]``; a dict ``{key tuple: rows}`` from
    #   ``cached.groupby(list(GRID_KEYS), sort=False)`` makes the lookup easy (the key tuple
    #   is ``tuple(spec[k] for k in GRID_KEYS)``; compare plain Python values, e.g. via
    #   ``spec.tolist()`` on a row from ``specs.iterrows()``).
    #   Loop over ``specs.iterrows()``: reuse, or ``rows = evaluate_trial(returns,
    #   trial_config(cfg, spec)).assign(fingerprint=fp)``, append to a list of new frames and
    #   ``pd.concat([cached, *new]).to_parquet(path, index=False)`` straight away.
    #   Final: ``grid = pd.concat(frames_in_trial_order, ignore_index=True)[list(GRID_COLUMNS)]
    #   .astype(GRID_DTYPES)``; within a trial the cached rows keep the sweep order they were
    #   written in (the fingerprint covers the sweep); write it and return the ``GridRun``.
    #   Imports to add: ``sha256_of`` (``src.data``) and, from ``src.robustness``,
    #   ``GRID_COLUMNS``, ``GRID_DTYPES``, ``GRID_KEYS``, ``evaluate_trial``, ``grid_specs``,
    #   ``trial_config``, ``trial_fingerprint``; ``trial_stats`` too (the ``metrics`` stage
    #   wrapper below already calls it).
    #   Verified by tests/test_cli.py::TestLoadOrRunGrid and ::TestStageWiring.
    root = Path(root)
    panel_path = root / cfg.data.panel_path
    returns, _ = clean_returns(_read_panel(cfg, root), cfg.data.clip)
    fp = trial_fingerprint(cfg, sha256_of(panel_path))
    specs = grid_specs(cfg.robustness)
    out = root / cfg.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / GRID_FILE

    if resume and path.exists():
        cached = pd.read_parquet(path)
        cached = cached[cached["fingerprint"] == fp]
    else:
        cached = pd.DataFrame(columns=list(GRID_COLUMNS))

    lookup = {}
    if len(cached):
        for key, rows in cached.groupby(list(GRID_KEYS), sort=False):
            lookup[tuple(k.item() if hasattr(k, "item") else k for k in key)] = rows

    frames = []
    new = []
    n_computed = 0
    n_cached = 0
    for _, spec in specs.iterrows():
        key = tuple(spec[list(GRID_KEYS)].tolist())
        if key in lookup:
            frames.append(lookup[key])
            n_cached += 1
            continue
        rows = evaluate_trial(returns, trial_config(cfg, spec)).assign(fingerprint=fp)
        new.append(rows)
        frames.append(rows)
        n_computed += 1
        known = [cached, *new] if len(cached) else new
        pd.concat(known, ignore_index=True).to_parquet(path, index=False)

    grid = pd.concat(frames, ignore_index=True)[list(GRID_COLUMNS)].astype(GRID_DTYPES)
    grid.to_parquet(path, index=False)
    return GridRun(
        grid=grid,
        n_trials=len(specs),
        n_computed=n_computed,
        n_cached=n_cached,
        fingerprint=fp,
    )


# ------------------------------------------------------ F6.2-F6.4 robustness stage


def run_robustness(cfg: Config, root: str | Path) -> RobustnessRun:
    """``make robustness``: the grid, the IS/OOS exercise, the sub-periods and the sweeps, to
    ``results/``. No network.

    Steps, all relative to ``root``, with ``rf = cfg.inference.risk_free_rate``:

    1. ``grid = load_or_run_grid(cfg, root)``; ``n, std = trial_stats(grid.grid,
       cfg.costs.cost_bps)``;
    2. ``is_oos``, ``cost_sweep``, ``lag_sweep`` and ``grid_distribution`` of ``grid.grid``
       with ``cfg``;
    3. ``bm = run_benchmarks(cfg, root)`` and ``bt = run_backtest(cfg, root)``; ``returns``,
       the five daily series over the strategy's life exactly as :func:`run_metrics` builds
       them (columns ``METRICS_SERIES`` then the external ticker);
    4. ``trials``, a dict with keys exactly ``TRIALS_KEYS``: ``n`` and ``std``, then for the
       net strategy column ``sharpe_hurdle = expected_max_sharpe(n, std)``, ``psr =
       probabilistic_sharpe(net, 0.0, rf)`` and ``dsr = deflated_sharpe(net, n, std, rf)``
       (the values ``metrics.json`` carries for ``Strategy (net)``);
    5. the sub-periods: ``block_stats(returns, cfg.robustness.subperiod_blocks, rf)``,
       ``calendar_year_returns(returns)``, ``rolling_sharpe(returns,
       cfg.robustness.rolling_sharpe_years * ANNUALISATION, rf)`` and its
       ``rolling_summary``; ``labels = regime_labels(bm.ew_long_only.net_returns,
       cfg.sizing.vol_window, cfg.robustness.regime_vol_quantile)`` (the basket over the
       whole panel, so its trailing vol exists from the strategy's first day), reindexed to
       ``returns.index``; ``regime_stats(returns, labels, rf)``;
    6. write, under ``root / cfg.paths.results_dir``: ``IS_OOS_REPORT`` (``is_oos_md`` with
       ``cfg.robustness.is_end``, ``oos_start`` and ``cfg.costs.cost_bps``),
       ``SUBPERIODS_REPORT`` (``subperiods_md`` with ``rolling_sharpe_years``,
       ``regime_vol_quantile`` and ``cfg.sizing.vol_window``) and ``SWEEPS_REPORT``
       (``sweeps_md`` with ``trials``, the distribution, the two sweeps and
       ``cfg.costs.cost_bps``).

    Returns a :class:`RobustnessRun`. Deterministic: two runs on the same snapshot and config
    write byte-identical reports (the §8 determinism test).
    """
    # TODO 77: implement run_robustness.
    #   ``root = Path(root)``; ``rf = cfg.inference.risk_free_rate``; ``rb =
    #   cfg.robustness``; ``grid = load_or_run_grid(cfg, root)``; ``g = grid.grid``; ``n, std
    #   = trial_stats(g, cfg.costs.cost_bps)``; ``table, summary = is_oos(g, cfg)``;
    #   ``costs, lags, dist = cost_sweep(g, cfg), lag_sweep(g, cfg), grid_distribution(g,
    #   cfg)``.
    #   ``bm = run_benchmarks(cfg, root)``; ``bt = run_backtest(cfg, root)``; the ``returns``
    #   frame as in TODO 59 (``life``, ``rows``, ``names``, the five columns).
    #   ``net = returns[METRICS_SERIES[1]]``; ``trials = {"n_trials": n, "sharpe_std": std,
    #   "sharpe_hurdle": expected_max_sharpe(n, std), "psr": probabilistic_sharpe(net, 0.0,
    #   rf), "dsr": deflated_sharpe(net, n, std, rf)}``.
    #   The sub-period tables as listed in step 5; ``rolling_summary`` of the rolling frame.
    #   ``out = root / cfg.paths.results_dir``; ``(out / IS_OOS_REPORT).write_text(is_oos_md(
    #   ...))`` and the other two; return the ``RobustnessRun`` (fields in its order).
    #   Imports to add: ``expected_max_sharpe``, ``probabilistic_sharpe``,
    #   ``deflated_sharpe`` (``src.inference``); ``is_oos``, ``cost_sweep``, ``lag_sweep``,
    #   ``grid_distribution`` (``src.robustness``); the six functions of ``src.subperiods``;
    #   ``is_oos_md``, ``subperiods_md``, ``sweeps_md`` (``src.report``).
    #   Verified by tests/test_cli.py::TestRunRobustness, ::TestRobustnessSnapshot,
    #   ::TestIntegrityDeterminism and ::TestStageWiring.
    root = Path(root)
    rf = cfg.inference.risk_free_rate
    rb = cfg.robustness
    grid = load_or_run_grid(cfg, root)
    g = grid.grid
    n, std = trial_stats(g, cfg.costs.cost_bps)
    table, summary = is_oos(g, cfg)
    costs, lags, dist = cost_sweep(g, cfg), lag_sweep(g, cfg), grid_distribution(g, cfg)

    bm = run_benchmarks(cfg, root)
    bt = run_backtest(cfg, root)
    life = bt.result.gross_returns.notna()
    rows = bt.result.gross_returns.index[life]
    names = [*METRICS_SERIES, bm.external_ticker]
    returns = pd.DataFrame(
        {
            names[0]: bt.result.gross_returns[life],
            names[1]: bt.result.net_returns[life],
            names[2]: bm.table["ew_long_only_net"],
            names[3]: bm.table["ew_vol_targeted_net"],
            names[4]: bm.table["external"],
        },
        index=rows,
        columns=names,
    )

    net = returns[METRICS_SERIES[1]]
    trials = {
        "n_trials": n,
        "sharpe_std": std,
        "sharpe_hurdle": expected_max_sharpe(n, std),
        "psr": probabilistic_sharpe(net, 0.0, rf),
        "dsr": deflated_sharpe(net, n, std, rf),
    }

    blocks = block_stats(returns, rb.subperiod_blocks, rf)
    years = calendar_year_returns(returns)
    rolling = rolling_sharpe(returns, rb.rolling_sharpe_years * ANNUALISATION, rf)
    roll_summary = rolling_summary(rolling)
    labels = regime_labels(
        bm.ew_long_only.net_returns, cfg.sizing.vol_window, rb.regime_vol_quantile
    ).reindex(returns.index)
    regimes = regime_stats(returns, labels, rf)

    out = root / cfg.paths.results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / IS_OOS_REPORT).write_text(
        is_oos_md(table, summary, rb.is_end, rb.oos_start, cfg.costs.cost_bps)
    )
    (out / SUBPERIODS_REPORT).write_text(
        subperiods_md(
            blocks,
            years,
            roll_summary,
            regimes,
            rb.rolling_sharpe_years,
            rb.regime_vol_quantile,
            cfg.sizing.vol_window,
        )
    )
    (out / SWEEPS_REPORT).write_text(sweeps_md(trials, dist, costs, lags, cfg.costs.cost_bps))

    return RobustnessRun(
        grid=grid,
        trials=trials,
        is_oos_table=table,
        is_oos_summary=summary,
        cost_sweep=costs,
        lag_sweep=lags,
        distribution=dist,
        returns=returns,
        blocks=blocks,
        years=years,
        rolling=rolling,
        rolling_summary=roll_summary,
        labels=labels,
        regimes=regimes,
    )


# ------------------------------------------------------------------------- stages


def _stage_data() -> None:
    from src.data import run_data

    cfg = load_config()
    dump_resolved(cfg, ROOT / cfg.paths.results_dir / RESOLVED_CONFIG)
    run_data(cfg, ROOT)


def _stage_panel() -> None:
    from src.data import run_panel

    cfg = load_config()
    dump_resolved(cfg, ROOT / cfg.paths.results_dir / RESOLVED_CONFIG)
    build = run_panel(cfg, ROOT)
    panel = build.prices[build.universe].dropna(how="all")
    print(f"panel: {len(panel)} rows, universe = {', '.join(build.universe)}")


def _stage_backtest() -> None:
    bt = run_backtest(load_config(), ROOT)
    s = bt.summary
    print(
        f"backtest: {s['n_days']} days from {s['first_date']:%Y-%m-%d}; "
        f"Sharpe gross {s['sharpe_gross']:.2f} / net {s['sharpe_net']:.2f}; "
        f"max drawdown net {s['max_drawdown_net']:.1%}; turnover {s['ann_turnover']:.2f}x/yr"
    )


def _stage_benchmarks() -> None:
    bm = run_benchmarks(load_config(), ROOT)
    for name, s in bm.summary.items():
        print(f"{name}: Sharpe {s['sharpe']:.2f}, max drawdown {s['max_drawdown']:.1%}")


def _stage_metrics() -> None:
    cfg = load_config()
    grid = load_or_run_grid(cfg, ROOT)
    n_trials, sharpe_std = trial_stats(grid.grid, cfg.costs.cost_bps)
    print(
        f"grid: {n_trials} trials ({grid.n_cached} cached, {grid.n_computed} computed), "
        f"net Sharpe std {sharpe_std:.3f}"
    )
    m = run_metrics(cfg, ROOT, n_trials, sharpe_std)
    for name, e in m.metrics.items():
        print(
            f"{name}: Sharpe {e['sharpe']:.2f} [{e['sharpe_ci_low']:.2f}, "
            f"{e['sharpe_ci_high']:.2f}], t {e['sharpe_t_stat']:.2f}, PSR {e['psr']:.3f}, "
            f"DSR {e['dsr']:.3f}"
        )


def _stage_robustness() -> None:
    r = run_robustness(load_config(), ROOT)
    g, t, s = r.grid, r.trials, r.is_oos_summary
    print(
        f"grid: {g.n_trials} trials ({g.n_cached} cached, {g.n_computed} computed); "
        f"net Sharpe std {t['sharpe_std']:.3f}, hurdle {t['sharpe_hurdle']:.3f}, "
        f"PSR {t['psr']:.3f}, DSR {t['dsr']:.3f}"
    )
    print(
        f"IS/OOS: selected L = {s['selected_lookback']} (IS {s['selected_is']:.2f}, "
        f"OOS {s['selected_oos']:.2f}); a priori L = {s['default_lookback']} "
        f"(IS {s['default_is']:.2f}, OOS {s['default_oos']:.2f})"
    )
    net = r.blocks[r.blocks["series"] == METRICS_SERIES[1]]
    print(
        "blocks (net Sharpe): "
        + ", ".join(f"{b} {x:.2f}" for b, x in zip(net["block"], net["sharpe"], strict=True))
    )
    print(
        "costs (net Sharpe): "
        + ", ".join(f"{c:g} bps {x:.2f}" for c, x in r.cost_sweep["sharpe"].items())
    )


STAGES: dict[str, Callable[[], None]] = {
    "data": _stage_data,
    "panel": _stage_panel,
    "backtest": _stage_backtest,
    "benchmarks": _stage_benchmarks,
    "metrics": _stage_metrics,
    "robustness": _stage_robustness,
}

KNOWN_STAGES = (
    "data",
    "panel",
    "backtest",
    "benchmarks",
    "metrics",
    "robustness",
    "figures",
    "report",
)


def main(argv: list[str] | None = None) -> int:
    """Run one pipeline stage. Returns a process exit code."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in KNOWN_STAGES:
        print(f"usage: python -m src.cli <{'|'.join(KNOWN_STAGES)}>", file=sys.stderr)
        return 2
    stage = args[0]
    fn = STAGES.get(stage)
    if fn is None:
        msg = f"stage '{stage}' is not implemented yet (see PROJECT_LOG.md Feature Log)"
        print(msg, file=sys.stderr)
        return 1
    fn()
    return 0


__all__ = [
    "BACKTEST_REPORT",
    "BENCHMARKS_FILE",
    "BENCHMARKS_REPORT",
    "BENCHMARK_COLUMNS",
    "BOOK_KEYS",
    "GRID_FILE",
    "IS_OOS_REPORT",
    "METRICS_FILE",
    "METRICS_SERIES",
    "METRICS_TABLE",
    "POSITIONS_FILE",
    "RESOLVED_CONFIG",
    "RETURNS_FILE",
    "SUBPERIODS_REPORT",
    "SWEEPS_REPORT",
    "TRIALS_KEYS",
    "BacktestRun",
    "BenchmarkRun",
    "GridRun",
    "MetricsRun",
    "RobustnessRun",
    "format_summary_md",
    "load_or_run_grid",
    "main",
    "run_backtest",
    "run_benchmarks",
    "run_metrics",
    "run_robustness",
]


if __name__ == "__main__":
    raise SystemExit(main())
