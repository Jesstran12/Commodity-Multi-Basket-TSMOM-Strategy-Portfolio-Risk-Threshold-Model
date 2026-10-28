"""Stages: ``run_backtest`` and ``run_benchmarks`` (Phase 4), ``run_metrics`` (Phase 5), and
``load_or_run_grid`` and ``run_robustness`` (Phase 6) from the committed snapshot into a
results directory, the files they write, the §8 determinism integrity test (extended to
``metrics.json`` in Phase 5 and to the robustness reports in Phase 6), and the CLI registry.
Everything here reads ``data/`` only; nothing touches the network or the repository's own
``results/``.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import src.cli as cli
from src.backtest import SUMMARY_KEYS, BacktestResult
from src.cli import (
    BACKTEST_REPORT,
    BENCHMARK_COLUMNS,
    BENCHMARKS_FILE,
    BENCHMARKS_REPORT,
    BOOK_KEYS,
    GRID_FILE,
    IS_OOS_REPORT,
    KNOWN_STAGES,
    METRICS_FILE,
    METRICS_SERIES,
    METRICS_TABLE,
    POSITIONS_FILE,
    RESOLVED_CONFIG,
    RETURNS_FILE,
    STAGES,
    SUBPERIODS_REPORT,
    SWEEPS_REPORT,
    TRIALS_KEYS,
    BacktestRun,
    BenchmarkRun,
    GridRun,
    MetricsRun,
    RobustnessRun,
    format_summary_md,
    load_or_run_grid,
    main,
    run_backtest,
    run_benchmarks,
    run_metrics,
    run_robustness,
)
from src.config import ANNUALISATION, Config, load_config
from src.inference import (
    INFERENCE_KEYS,
    deflated_sharpe,
    expected_max_sharpe,
    inference_summary,
    probabilistic_sharpe,
)
from src.metrics import METRIC_KEYS, PERFORMANCE_KEYS, monthly_returns, summarise
from src.portfolio import SUMMARY_KEYS as SIZING_KEYS
from src.report import (
    METRIC_ROWS,
    format_metrics_md,
    is_oos_md,
    metrics_json,
    subperiods_md,
    sweeps_md,
)
from src.robustness import (
    GRID_COLUMNS,
    GRID_DTYPES,
    cost_sweep,
    default_spec,
    grid_distribution,
    is_oos,
    lag_sweep,
    matches,
    trial_stats,
)
from src.robustness import evaluate_trial as robustness_evaluate_trial
from src.subperiods import (
    block_stats,
    calendar_year_returns,
    regime_labels,
    regime_stats,
    rolling_sharpe,
    rolling_summary,
)

ROOT = Path(__file__).resolve().parents[1]
RETURN_COLUMNS = ["gross", "net", "cost", "turnover", "gross_leverage", "n_live"]


def config_into(results_dir: Path) -> Config:
    """The default config with ``paths.results_dir`` pointed at a scratch directory."""
    return load_config(overrides={"paths.results_dir": str(results_dir)})


_CACHE: dict[str, object] = {}


def cached_backtest() -> tuple[BacktestRun, Path]:
    """One ``run_backtest`` into a scratch directory, shared by the read-only tests below.

    A plain helper, not a fixture, so that a stub raising ``NotImplementedError`` shows up
    as a failed test rather than an errored fixture.
    """
    if "backtest" not in _CACHE:
        out = Path(tempfile.mkdtemp(prefix="p4-backtest-"))
        _CACHE["backtest"] = (run_backtest(config_into(out), ROOT), out)
    return _CACHE["backtest"]


def cached_benchmarks() -> tuple[BenchmarkRun, Path]:
    if "benchmarks" not in _CACHE:
        out = Path(tempfile.mkdtemp(prefix="p4-benchmarks-"))
        _CACHE["benchmarks"] = (run_benchmarks(config_into(out), ROOT), out)
    return _CACHE["benchmarks"]


def cached_metrics() -> tuple[MetricsRun, Path]:
    """One default ``run_metrics`` (2,000 resamples per series) into a scratch directory."""
    if "metrics" not in _CACHE:
        out = Path(tempfile.mkdtemp(prefix="p5-metrics-"))
        _CACHE["metrics"] = (run_metrics(config_into(out), ROOT), out)
    return _CACHE["metrics"]


def quick_config(results_dir: Path) -> Config:
    """``config_into`` with 200 resamples: the same code path, a tenth of the bootstrap."""
    return load_config(
        overrides={"paths.results_dir": str(results_dir), "inference.n_resamples": 200}
    )


def small_grid_config(results_dir: Path, **overrides: object) -> Config:
    """``config_into`` with a two-trial grid (L 126 and 252, every other grid parameter at
    its a-priori value, four costs each) and 200 resamples. Keyword overrides use ``__`` for
    the dot: ``robustness__grid_lookbacks=[252]``."""
    base: dict[str, object] = {
        "paths.results_dir": str(results_dir),
        "inference.n_resamples": 200,
        "robustness.grid_lookbacks": [126, 252],
        "robustness.grid_vol_windows": [60],
        "robustness.grid_estimators": ["rolling"],
        "robustness.grid_rebalances": ["monthly"],
        "robustness.grid_skips": [0],
        "robustness.grid_lags": [1],
    }
    base.update({k.replace("__", "."): v for k, v in overrides.items()})
    return load_config(overrides=base)


def cached_robustness() -> tuple[RobustnessRun, Path, float]:
    """One default ``run_robustness`` (the full 192-trial grid, from an empty results
    directory) into a scratch directory, with its wall time in seconds."""
    if "robustness" not in _CACHE:
        out = Path(tempfile.mkdtemp(prefix="p6-robustness-"))
        started = time.perf_counter()
        r = run_robustness(config_into(out), ROOT)
        _CACHE["robustness"] = (r, out, time.perf_counter() - started)
    return _CACHE["robustness"]


# ------------------------------------------------------------------ format_summary_md


class TestFormatSummaryMd:
    def test_renders_sections_as_tables(self):
        md = format_summary_md(
            "Title",
            {
                "One": {"a": 1, "b": 0.123456789, "c": pd.Timestamp("2020-01-31")},
                "Two": {"x": None},
            },
        )
        lines = md.splitlines()
        assert lines[0] == "# Title"
        assert "## One" in lines and "## Two" in lines
        assert "| a | 1 |" in lines
        assert "| b | 0.123457 |" in lines
        assert "| c | 2020-01-31 |" in lines
        assert "| x | None |" in lines
        assert lines[lines.index("## One") + 2 : lines.index("## One") + 4] == [
            "| Quantity | Value |",
            "|---|---|",
        ]


# -------------------------------------------------------------------- run_backtest


class TestRunBacktest:
    def test_writes_the_four_files(self):
        _, out = cached_backtest()
        for name in (RETURNS_FILE, POSITIONS_FILE, BACKTEST_REPORT, RESOLVED_CONFIG):
            assert (out / name).exists(), name

    def test_returns_daily_csv_has_the_life_and_the_six_columns(self):
        bt, out = cached_backtest()
        daily = pd.read_csv(out / RETURNS_FILE, index_col="date", parse_dates=True)
        assert list(daily.columns) == RETURN_COLUMNS
        life = bt.result.gross_returns.notna()
        assert daily.index.equals(bt.result.gross_returns.index[life])
        assert daily.index[0] == pd.Timestamp("2001-04-02")
        assert daily.index[-1] == pd.Timestamp("2026-09-07")
        assert len(daily) == 6410
        assert daily["gross"].notna().all() and daily["net"].notna().all()
        assert_series_equal(daily["gross"], bt.result.gross_returns[life], check_names=False)
        assert_series_equal(daily["net"], bt.result.net_returns[life], check_names=False)

    def test_positions_csv_is_the_lagged_weights_over_the_life(self):
        bt, out = cached_backtest()
        pos = pd.read_csv(out / POSITIONS_FILE, index_col="date", parse_dates=True)
        life = bt.result.gross_returns.notna()
        assert list(pos.columns) == list(bt.returns.columns)
        assert_frame_equal(pos, bt.result.positions[life], check_names=False)
        assert_frame_equal(bt.result.positions, bt.sizing.weights.shift(1), check_names=False)

    def test_report_has_a_strategy_and_a_sizing_section(self):
        bt, out = cached_backtest()
        md = (out / BACKTEST_REPORT).read_text()
        assert md.startswith("# Backtest")
        assert "## Strategy" in md and "## Sizing" in md
        for key in SUMMARY_KEYS:
            assert f"| {key} |" in md, key
        for key in SIZING_KEYS:
            assert f"| {key} |" in md, key
        assert "| n_unpriced | 20 |" in md
        assert "| n_cap_bind_days | 0 |" in md

    def test_resolved_config_round_trips(self):
        bt, out = cached_backtest()
        import yaml

        resolved = yaml.safe_load((out / RESOLVED_CONFIG).read_text())
        assert resolved["sizing"]["vol_window"] == 60
        assert resolved["execution"]["lag"] == 1 and resolved["costs"]["cost_bps"] == 5
        assert resolved["paths"]["results_dir"] == str(out)

    def test_result_fields_and_summary(self):
        bt, _ = cached_backtest()
        assert isinstance(bt, BacktestRun) and isinstance(bt.result, BacktestResult)
        assert list(bt.summary) == list(SUMMARY_KEYS)
        assert bt.summary["lag"] == 1 and bt.summary["cost_bps"] == 5.0
        assert bt.summary["first_date"] == pd.Timestamp("2001-04-02")
        assert bt.summary["n_days"] == 6410
        assert 0.05 <= bt.summary["ann_vol_net"] <= 0.15

    def test_runs_well_under_a_minute(self, tmp_path):
        started = time.perf_counter()
        run_backtest(config_into(tmp_path), ROOT)
        assert time.perf_counter() - started < 60.0


# ----------------------------------------------------------------------- §8 integrity


class TestIntegrityDeterminism:
    """PROJECT_LOG.md §8 "determinism": two full pipeline runs on the snapshot produce
    identical results files. Phase 5 extends the check to ``metrics.json``."""

    def test_two_runs_write_byte_identical_results(self, tmp_path):
        a = run_backtest(config_into(tmp_path / "a"), ROOT)
        b = run_backtest(config_into(tmp_path / "b"), ROOT)
        for name in (RETURNS_FILE, POSITIONS_FILE, BACKTEST_REPORT):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), (
                name
            )
        assert_frame_equal(a.result.positions, b.result.positions)
        assert_series_equal(a.result.net_returns, b.result.net_returns)
        assert a.summary == b.summary

    def test_a_cached_run_matches_a_fresh_one(self, tmp_path):
        cached, out = cached_backtest()
        fresh = run_backtest(config_into(tmp_path), ROOT)
        assert (out / RETURNS_FILE).read_bytes() == (tmp_path / RETURNS_FILE).read_bytes()
        assert cached.summary == fresh.summary

    def test_two_runs_write_byte_identical_robustness_reports(self, tmp_path):
        # Phase 6: the grid and the three robustness reports, two runs from empty results
        # directories (so neither reads the other's cache).
        a = run_robustness(small_grid_config(tmp_path / "a"), ROOT)
        b = run_robustness(small_grid_config(tmp_path / "b"), ROOT)
        for name in (IS_OOS_REPORT, SUBPERIODS_REPORT, SWEEPS_REPORT):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), (
                name
            )
        assert_frame_equal(a.grid.grid, b.grid.grid)
        assert_frame_equal(
            pd.read_parquet(tmp_path / "a" / GRID_FILE), pd.read_parquet(tmp_path / "b" / GRID_FILE)
        )

    def test_two_runs_write_byte_identical_metrics_json(self, tmp_path):
        # The §8 wording: "two full pipeline runs on the snapshot produce identical
        # metrics.json". The bootstrap is seeded from the config, so this holds exactly.
        run_metrics(quick_config(tmp_path / "a"), ROOT)
        run_metrics(quick_config(tmp_path / "b"), ROOT)
        for name in (METRICS_FILE, METRICS_TABLE, RETURNS_FILE, BENCHMARKS_FILE):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), (
                name
            )


# ------------------------------------------------------------------ run_benchmarks


class TestRunBenchmarks:
    def test_writes_the_two_files_with_the_five_columns(self):
        bm, out = cached_benchmarks()
        table = pd.read_csv(out / BENCHMARKS_FILE, index_col="date", parse_dates=True)
        assert list(table.columns) == list(BENCHMARK_COLUMNS)
        assert_frame_equal(table, bm.table, check_names=False)
        assert (out / BENCHMARKS_REPORT).read_text().startswith("# Benchmarks")

    def test_aligned_to_the_strategys_life(self):
        bm, _ = cached_benchmarks()
        bt, _ = cached_backtest()
        assert bm.start == pd.Timestamp("2001-04-02") == bt.summary["first_date"]
        assert bm.table.index[0] == bm.start
        assert bm.table.index[-1] == pd.Timestamp("2026-09-07")
        assert len(bm.table) == 6410
        # The baskets exist on every row of the life; the external index stops with its
        # last print (2026-07-17, 36 rows before the panel's end) and, inside the life, has
        # 18 panel days without a print (the September 2001 closure, US holidays on which a
        # commodity market was open).
        assert bm.table[list(BENCHMARK_COLUMNS[:4])].notna().all().all()
        assert bm.table["external"].isna().sum() == 36 + 18

    def test_equal_weight_basket_is_long_only_gross_one(self):
        bm, _ = cached_benchmarks()
        pos = bm.ew_long_only.positions.loc[bm.start :]
        assert (pos.fillna(0) >= 0).all().all()
        assert pos.abs().sum(axis=1).to_numpy() == pytest.approx(1.0)
        # Monthly: the basket trades on the row after a month-end only.
        traded = bm.ew_long_only.turnover.loc[bm.start :]
        assert (traded > 0).sum() < len(traded) / 15

    def test_vol_targeted_basket_runs_near_the_target(self):
        bm, _ = cached_benchmarks()
        s = bm.summary["EW vol-targeted (net)"]
        assert 0.07 <= s["ann_vol"] <= 0.14
        assert s["n_cap_bind_days"] == 0
        assert not bm.ew_cap_binds.any()

    def test_summary_sections_and_keys(self):
        bm, _ = cached_benchmarks()
        assert list(bm.summary) == [
            "EW long-only (gross)",
            "EW long-only (net)",
            "EW vol-targeted (gross)",
            "EW vol-targeted (net)",
            "^BCOM",
        ]
        for name, s in bm.summary.items():
            assert list(s)[: len(PERFORMANCE_KEYS)] == list(PERFORMANCE_KEYS), name
        assert bm.summary["^BCOM"]["n_missing"] == 18
        assert bm.summary["^BCOM"]["n_after_last_print"] == 36
        assert bm.external_ticker == "^BCOM"
        lo_g, lo_n = bm.summary["EW long-only (gross)"], bm.summary["EW long-only (net)"]
        assert lo_n["sharpe"] < lo_g["sharpe"]
        assert lo_g["ann_turnover"] == lo_n["ann_turnover"] > 0

    def test_benchmarks_are_deterministic(self, tmp_path):
        bm, out = cached_benchmarks()
        again = run_benchmarks(config_into(tmp_path), ROOT)
        assert (out / BENCHMARKS_FILE).read_bytes() == (tmp_path / BENCHMARKS_FILE).read_bytes()
        assert again.summary == bm.summary


# --------------------------------------------------------------------- run_metrics


class TestRunMetrics:
    def test_series_names_and_constants(self):
        assert METRICS_SERIES == (
            "Strategy (gross)",
            "Strategy (net)",
            "EW long-only (net)",
            "EW vol-targeted (net)",
        )
        assert BOOK_KEYS == (
            "ann_turnover",
            "mean_gross_leverage",
            "mean_n_live",
            "cap_bind_fraction",
        )
        assert (METRICS_FILE, METRICS_TABLE) == ("metrics.json", "metrics_table.md")

    def test_writes_json_and_table_beside_the_earlier_stages_files(self):
        _, out = cached_metrics()
        for name in (METRICS_FILE, METRICS_TABLE, RETURNS_FILE, BENCHMARKS_FILE, RESOLVED_CONFIG):
            assert (out / name).exists(), name

    def test_five_columns_strategy_gross_and_net_and_three_benchmarks(self):
        m, _ = cached_metrics()
        assert isinstance(m, MetricsRun)
        assert list(m.metrics) == [*METRICS_SERIES, "^BCOM"]
        assert list(m.returns.columns) == list(m.metrics)
        for name, entry in m.metrics.items():
            assert list(entry) == [*METRIC_KEYS, *BOOK_KEYS, *INFERENCE_KEYS], name

    def test_daily_returns_are_the_earlier_stages_series_over_the_life(self):
        m, _ = cached_metrics()
        bt, bm = m.backtest, m.benchmarks
        life = bt.result.gross_returns.notna()
        assert m.returns.index.equals(bt.result.gross_returns.index[life])
        cols = m.returns.columns
        assert_series_equal(m.returns[cols[0]], bt.result.gross_returns[life], check_names=False)
        assert_series_equal(m.returns[cols[1]], bt.result.net_returns[life], check_names=False)
        assert_series_equal(m.returns[cols[2]], bm.table["ew_long_only_net"], check_names=False)
        assert_series_equal(m.returns[cols[3]], bm.table["ew_vol_targeted_net"], check_names=False)
        assert_series_equal(m.returns[cols[4]], bm.table["external"], check_names=False)

    def test_entries_are_summarise_and_inference_of_each_column(self):
        m, _ = cached_metrics()
        cfg = load_config()
        for name in ("Strategy (net)", "^BCOM"):
            entry = m.metrics[name]
            expected = summarise(m.returns[name], cfg.inference.risk_free_rate)
            for k in METRIC_KEYS:
                assert entry[k] == expected[k] or (np.isnan(entry[k]) and np.isnan(expected[k]))
        quick = load_config(overrides={"inference.n_resamples": 200}).inference
        a = inference_summary(m.returns["Strategy (net)"], quick)
        assert m.metrics["Strategy (net)"]["psr"] == pytest.approx(a["psr"])
        assert m.metrics["Strategy (net)"]["sharpe_t_stat"] == pytest.approx(a["sharpe_t_stat"])

    def test_headline_numbers_agree_with_the_earlier_stages(self):
        m, _ = cached_metrics()
        s = m.backtest.summary
        net, gross = m.metrics["Strategy (net)"], m.metrics["Strategy (gross)"]
        assert net["sharpe"] == s["sharpe_net"] and gross["sharpe"] == s["sharpe_gross"]
        assert net["max_drawdown"] == s["max_drawdown_net"]
        assert net["ann_return"] == s["ann_return_net"]
        lo = m.benchmarks.summary["EW long-only (net)"]
        assert m.metrics["EW long-only (net)"]["sharpe"] == lo["sharpe"]
        assert m.metrics["^BCOM"]["sharpe"] == m.benchmarks.summary["^BCOM"]["sharpe"]

    def test_book_statistics(self):
        m, _ = cached_metrics()
        s = m.backtest.summary
        for name in ("Strategy (gross)", "Strategy (net)"):
            e = m.metrics[name]
            assert e["ann_turnover"] == s["ann_turnover"]
            assert e["mean_gross_leverage"] == s["mean_gross_leverage"]
            assert e["mean_n_live"] == s["mean_n_live"]
            assert e["cap_bind_fraction"] == 0.0
        lo, vt = m.metrics["EW long-only (net)"], m.metrics["EW vol-targeted (net)"]
        assert lo["ann_turnover"] == m.benchmarks.summary["EW long-only (net)"]["ann_turnover"]
        assert vt["ann_turnover"] == m.benchmarks.summary["EW vol-targeted (net)"]["ann_turnover"]
        assert lo["mean_gross_leverage"] == pytest.approx(1.0)
        assert np.isnan(lo["cap_bind_fraction"])  # no cap on the plain basket
        assert vt["cap_bind_fraction"] == 0.0
        assert 12 < lo["mean_n_live"] <= 13 and lo["mean_n_live"] == vt["mean_n_live"]
        for k in BOOK_KEYS:
            assert np.isnan(m.metrics["^BCOM"][k]), k

    def test_one_trial_by_default_so_dsr_is_psr(self):
        m, _ = cached_metrics()
        for name, e in m.metrics.items():
            assert e["n_trials"] == 1 and e["sharpe_hurdle"] == 0.0, name
            assert e["dsr"] == e["psr"], name

    def test_trial_count_and_trial_std_are_passed_through(self, tmp_path):
        m = run_metrics(quick_config(tmp_path), ROOT, n_trials=192, trials_sharpe_std=0.2)
        for name in ("Strategy (gross)", "Strategy (net)"):
            e = m.metrics[name]
            assert e["n_trials"] == 192, name
            assert e["sharpe_hurdle"] == pytest.approx(expected_max_sharpe(192, 0.2)), name
            assert e["dsr"] < e["psr"], name

    def test_benchmarks_are_always_measured_at_one_trial(self, tmp_path):
        # Phase 6: nobody searched over the benchmarks, so the grid's trial count deflates
        # the strategy columns only.
        m = run_metrics(quick_config(tmp_path), ROOT, n_trials=192, trials_sharpe_std=0.2)
        for name in ("EW long-only (net)", "EW vol-targeted (net)", "^BCOM"):
            e = m.metrics[name]
            assert e["n_trials"] == 1 and e["sharpe_hurdle"] == 0.0, name
            assert e["dsr"] == e["psr"], name

    def test_files_are_the_report_helpers_output(self):
        m, out = cached_metrics()
        assert (out / METRICS_FILE).read_text() == metrics_json(m.metrics)
        assert (out / METRICS_TABLE).read_text() == format_metrics_md(m.metrics)
        loaded = json.loads((out / METRICS_FILE).read_text())
        assert list(loaded) == list(m.metrics)
        assert loaded["Strategy (net)"]["first_date"] == "2001-04-02"

    def test_table_renders_five_columns_and_every_row(self):
        _, out = cached_metrics()
        lines = (out / METRICS_TABLE).read_text().splitlines()
        assert lines[0] == (
            "| Metric | Strategy (gross) | Strategy (net) | EW long-only (net) "
            "| EW vol-targeted (net) | ^BCOM |"
        )
        assert len(lines) == 2 + len(METRIC_ROWS)
        assert {line.count("|") for line in lines} == {7}

    def test_ci_contains_the_point_estimate_for_every_series(self):
        # F5.2 acceptance on real data.
        m, _ = cached_metrics()
        for name, e in m.metrics.items():
            assert e["sharpe_ci_low"] < e["sharpe"] < e["sharpe_ci_high"], name
            assert 0.0 < e["psr"] < 1.0, name

    def test_runs_in_reasonable_time(self, tmp_path):
        started = time.perf_counter()
        run_metrics(config_into(tmp_path), ROOT)
        assert time.perf_counter() - started < 60.0


class TestMetricsSnapshot:
    """Real-data numbers of ``make metrics`` on the committed snapshot (pinned from the
    reference run; the structural counts exactly, the statistics to their reported digits)."""

    def test_months_and_days(self):
        m, _ = cached_metrics()
        # April 2001 to September 2026 is 306 calendar months (both partial ones kept);
        # ^BCOM stops in July 2026, 304 months and 6,356 days.
        for name in METRICS_SERIES:
            assert len(monthly_returns(m.returns[name])) == 306, name
            assert m.metrics[name]["n_days"] == 6410, name
        assert len(monthly_returns(m.returns["^BCOM"])) == 304
        assert m.metrics["^BCOM"]["n_days"] == 6356

    def test_net_drawdown_spell_is_the_phase_4_drawdown(self):
        # Phase 4: net peak 2015-12-17, recovered 2025-09-05. The spell is the rows strictly
        # in between, 2015-12-18 to 2025-09-04: 2,442 trading days, the longest of the life.
        m, _ = cached_metrics()
        net = m.returns["Strategy (net)"]
        assert m.metrics["Strategy (net)"]["max_drawdown_duration"] == 2442
        wealth = (1 + net).cumprod()
        under = wealth < wealth.cummax().clip(lower=1.0)
        spell = under.loc["2015-12-18":"2025-09-04"]
        assert spell.all() and len(spell) == 2442
        assert not under.loc["2015-12-17"] and not under.loc["2025-09-05"]

    def test_hit_rates_are_whole_counts(self):
        m, _ = cached_metrics()
        e = m.metrics["Strategy (net)"]
        assert e["hit_rate_daily"] == pytest.approx(3299 / 6410)  # 51.47% of days up
        assert e["hit_rate_monthly"] == pytest.approx(160 / 306)  # 52.29% of months up

    def test_strategy_net_inference(self):
        m, _ = cached_metrics()
        e = m.metrics["Strategy (net)"]
        # t = 0.342160 * sqrt(6410 / 252) = 0.342160 * 5.043462 = 1.72567
        assert e["sharpe_t_stat"] == pytest.approx(0.342160 * np.sqrt(6410 / 252), abs=1e-5)
        assert e["sharpe_ci_low"] == pytest.approx(-0.0341, abs=1e-4)
        assert e["sharpe_ci_high"] == pytest.approx(0.7189, abs=1e-4)
        assert e["psr"] == pytest.approx(0.9576, abs=1e-4)
        assert e["skew"] == pytest.approx(-0.1145, abs=1e-4)
        assert e["kurtosis"] == pytest.approx(2.2609, abs=1e-4)
        assert e["best_month"] == pytest.approx(0.1138, abs=1e-4)  # March 2020
        assert e["worst_month"] == pytest.approx(-0.1177, abs=1e-4)  # September 2011

    def test_the_strategy_interval_straddles_zero_and_the_basket_does_not(self):
        # The finding the table carries: 25 years of a 0.34 Sharpe is not enough to rule
        # out zero at 95% (PSR 0.958 < 0.975); the long-only basket's 0.71 is.
        m, _ = cached_metrics()
        net, lo = m.metrics["Strategy (net)"], m.metrics["EW long-only (net)"]
        assert net["sharpe_ci_low"] < 0 < net["sharpe_ci_high"]
        assert 0.95 < net["psr"] < 0.975
        assert lo["sharpe_ci_low"] > 0 and lo["psr"] > 0.999


# ------------------------------------------------------------------ load_or_run_grid


def grid_file(cfg: Config) -> Path:
    return Path(cfg.paths.results_dir) / GRID_FILE


class TestLoadOrRunGrid:
    def test_first_run_computes_every_trial_and_writes_the_cache(self, tmp_path):
        cfg = small_grid_config(tmp_path)
        g = load_or_run_grid(cfg, ROOT)
        assert isinstance(g, GridRun)
        assert (g.n_trials, g.n_computed, g.n_cached) == (2, 2, 0)
        assert grid_file(cfg).exists()
        assert len(g.fingerprint) == 64

    def test_columns_dtypes_and_order(self, tmp_path):
        g = load_or_run_grid(small_grid_config(tmp_path), ROOT).grid
        assert list(g.columns) == list(GRID_COLUMNS)
        assert isinstance(g.index, pd.RangeIndex) and len(g) == 8
        for col in GRID_COLUMNS:
            assert str(g[col].dtype) == str(pd.Series([], dtype=GRID_DTYPES[col]).dtype), col
        assert g["lookback"].tolist() == [126] * 4 + [252] * 4
        assert g["cost_bps"].tolist() == [0.0, 5.0, 10.0, 20.0] * 2
        assert g["fingerprint"].nunique() == 1

    def test_the_file_holds_the_grid(self, tmp_path):
        cfg = small_grid_config(tmp_path)
        g = load_or_run_grid(cfg, ROOT).grid
        assert_frame_equal(pd.read_parquet(grid_file(cfg)).astype(GRID_DTYPES), g)

    def test_a_second_run_reads_the_cache(self, tmp_path):
        cfg = small_grid_config(tmp_path)
        first = load_or_run_grid(cfg, ROOT)
        second = load_or_run_grid(cfg, ROOT)
        assert (second.n_computed, second.n_cached) == (0, 2)
        assert_frame_equal(first.grid, second.grid)

    def test_resumes_a_partial_cache(self, tmp_path):
        cfg = small_grid_config(tmp_path)
        full = load_or_run_grid(cfg, ROOT).grid
        partial = pd.read_parquet(grid_file(cfg))
        partial[partial["lookback"] == 252].to_parquet(grid_file(cfg), index=False)
        resumed = load_or_run_grid(cfg, ROOT)
        assert (resumed.n_computed, resumed.n_cached) == (1, 1)
        assert_frame_equal(resumed.grid, full)

    def test_an_interrupted_run_keeps_the_trials_it_finished(self, tmp_path, monkeypatch):
        # The stage calls the evaluate_trial imported into src.cli (its docstring says so), so
        # replacing that name simulates a crash in the middle of the grid.
        cfg = small_grid_config(tmp_path)
        real = robustness_evaluate_trial
        calls = []

        def flaky(returns, trial_cfg):
            calls.append(trial_cfg.signal.lookback)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(returns, trial_cfg)

        monkeypatch.setattr(cli, "evaluate_trial", flaky, raising=False)
        with pytest.raises(KeyboardInterrupt):
            load_or_run_grid(cfg, ROOT)
        saved = pd.read_parquet(grid_file(cfg))
        assert saved["lookback"].tolist() == [126] * 4
        monkeypatch.setattr(cli, "evaluate_trial", real)
        resumed = load_or_run_grid(cfg, ROOT)
        assert (resumed.n_computed, resumed.n_cached) == (1, 1)

    def test_another_fingerprint_is_recomputed(self, tmp_path):
        first = load_or_run_grid(small_grid_config(tmp_path), ROOT)
        changed = small_grid_config(tmp_path, sizing__leverage_cap=2.0)
        second = load_or_run_grid(changed, ROOT)
        assert (second.n_computed, second.n_cached) == (2, 0)
        assert second.fingerprint != first.fingerprint
        assert set(pd.read_parquet(grid_file(changed))["fingerprint"]) == {second.fingerprint}

    def test_resume_false_recomputes(self, tmp_path):
        cfg = small_grid_config(tmp_path)
        load_or_run_grid(cfg, ROOT)
        again = load_or_run_grid(cfg, ROOT, resume=False)
        assert (again.n_computed, again.n_cached) == (2, 0)

    def test_extending_the_grid_keeps_the_cache_and_the_order(self, tmp_path):
        load_or_run_grid(small_grid_config(tmp_path), ROOT)
        wider = small_grid_config(tmp_path, robustness__grid_lookbacks=[126, 189, 252])
        g = load_or_run_grid(wider, ROOT)
        assert (g.n_trials, g.n_computed, g.n_cached) == (3, 1, 2)
        assert g.grid["lookback"].tolist() == [126] * 4 + [189] * 4 + [252] * 4

    def test_shrinking_the_grid_shrinks_the_file(self, tmp_path):
        load_or_run_grid(small_grid_config(tmp_path), ROOT)
        narrow = small_grid_config(tmp_path, robustness__grid_lookbacks=[252])
        g = load_or_run_grid(narrow, ROOT)
        assert (g.n_computed, g.n_cached) == (0, 1)
        assert pd.read_parquet(grid_file(narrow))["lookback"].tolist() == [252] * 4

    def test_the_default_rows_are_the_headline_backtest(self, tmp_path):
        g = load_or_run_grid(small_grid_config(tmp_path), ROOT).grid
        bt, _ = cached_backtest()
        s = bt.summary
        default = g[matches(g, default_spec(load_config()))].set_index("cost_bps")
        # Each trial over its own life: the default's rows reproduce Phase 4's numbers.
        assert default.loc[5.0, "sharpe"] == s["sharpe_net"]
        assert default.loc[0.0, "sharpe"] == s["sharpe_gross"]
        assert default.loc[5.0, "max_drawdown"] == s["max_drawdown_net"]
        assert default.loc[5.0, "ann_turnover"] == pytest.approx(s["ann_turnover"])
        assert default.loc[5.0, "n_days"] == 6410
        assert default.loc[5.0, "first_date"] == pd.Timestamp("2001-04-02")


# ------------------------------------------------------------------- run_robustness


class TestRunRobustness:
    def test_writes_the_grid_and_the_three_reports(self):
        r, out, _ = cached_robustness()
        assert isinstance(r, RobustnessRun)
        for name in (GRID_FILE, IS_OOS_REPORT, SUBPERIODS_REPORT, SWEEPS_REPORT):
            assert (out / name).exists(), name

    def test_the_full_grid_is_192_trials_at_four_costs(self):
        r, _, _ = cached_robustness()
        assert (r.grid.n_trials, r.grid.n_computed, r.grid.n_cached) == (192, 192, 0)
        assert len(r.grid.grid) == 192 * 4
        assert not r.grid.grid.duplicated(
            subset=["lookback", "vol_window", "estimator", "rebalance", "skip", "lag", "cost_bps"]
        ).any()

    def test_runs_in_well_under_five_minutes(self):
        # F6.1 acceptance: the whole grid from an empty cache in under about five minutes.
        _, _, seconds = cached_robustness()
        assert seconds < 300.0

    def test_trials(self):
        r, _, _ = cached_robustness()
        t = r.trials
        assert list(t) == list(TRIALS_KEYS)
        n, std = trial_stats(r.grid.grid, 5.0)
        assert t["n_trials"] == n == 192 and t["sharpe_std"] == std
        net = r.returns["Strategy (net)"]
        assert t["sharpe_hurdle"] == expected_max_sharpe(n, std)
        assert t["psr"] == probabilistic_sharpe(net, 0.0, 0.0)
        assert t["dsr"] == deflated_sharpe(net, n, std, 0.0)
        assert t["dsr"] < t["psr"]  # deflated means lower

    def test_returns_are_the_metrics_stage_series(self):
        r, _, _ = cached_robustness()
        m, _ = cached_metrics()
        assert_frame_equal(r.returns, m.returns)

    def test_tables_are_the_pure_functions_output(self):
        r, _, _ = cached_robustness()
        cfg = load_config()
        g = r.grid.grid
        table, summary = is_oos(g, cfg)
        assert_frame_equal(r.is_oos_table, table)
        assert r.is_oos_summary == summary
        assert_frame_equal(r.cost_sweep, cost_sweep(g, cfg))
        assert_frame_equal(r.lag_sweep, lag_sweep(g, cfg))
        assert r.distribution == grid_distribution(g, cfg)
        rb = cfg.robustness
        assert_frame_equal(r.blocks, block_stats(r.returns, rb.subperiod_blocks, 0.0))
        assert_frame_equal(r.years, calendar_year_returns(r.returns))
        rolling = rolling_sharpe(r.returns, rb.rolling_sharpe_years * ANNUALISATION, 0.0)
        assert_frame_equal(r.rolling, rolling)
        assert_frame_equal(r.rolling_summary, rolling_summary(rolling))
        assert_frame_equal(r.regimes, regime_stats(r.returns, r.labels, 0.0))

    def test_regime_labels_are_the_baskets_over_the_whole_panel(self):
        r, _, _ = cached_robustness()
        bm, _ = cached_benchmarks()
        cfg = load_config()
        expected = regime_labels(
            bm.ew_long_only.net_returns, cfg.sizing.vol_window, cfg.robustness.regime_vol_quantile
        ).reindex(r.returns.index)
        assert_series_equal(r.labels, expected)
        assert r.labels.notna().all()  # the basket's vol exists before the strategy starts

    def test_reports_are_the_composers_output(self):
        r, out, _ = cached_robustness()
        cfg = load_config()
        rb = cfg.robustness
        assert (out / IS_OOS_REPORT).read_text() == is_oos_md(
            r.is_oos_table, r.is_oos_summary, rb.is_end, rb.oos_start, cfg.costs.cost_bps
        )
        assert (out / SUBPERIODS_REPORT).read_text() == subperiods_md(
            r.blocks,
            r.years,
            r.rolling_summary,
            r.regimes,
            rb.rolling_sharpe_years,
            rb.regime_vol_quantile,
            cfg.sizing.vol_window,
        )
        assert (out / SWEEPS_REPORT).read_text() == sweeps_md(
            r.trials, r.distribution, r.cost_sweep, r.lag_sweep, cfg.costs.cost_bps
        )

    def test_a_second_run_reuses_the_whole_grid(self):
        _, out, _ = cached_robustness()
        again = load_or_run_grid(config_into(out), ROOT)
        assert (again.n_computed, again.n_cached) == (0, 192)


class TestRobustnessSnapshot:
    """Real-data numbers of ``make robustness`` on the committed snapshot (pinned from the
    reference run; structural counts exactly, statistics to their reported digits)."""

    def test_trials_and_the_deflated_sharpe(self):
        r, _, _ = cached_robustness()
        t = r.trials
        assert t["n_trials"] == 192
        assert t["sharpe_std"] == pytest.approx(0.1574, abs=1e-4)
        # hurdle = 0.157403 * 2.752 (the N = 192 multiplier of Phase 5) = 0.4332
        assert t["sharpe_hurdle"] == pytest.approx(0.4332, abs=1e-4)
        assert t["psr"] == pytest.approx(0.9576, abs=1e-4)  # Phase 5's, unchanged
        assert t["dsr"] == pytest.approx(0.3233, abs=1e-4)
        # The default's net Sharpe (0.342) is below the hurdle: the DSR is under one half.
        assert t["dsr"] < 0.5

    def test_every_trial_has_a_sharpe_and_a_life_of_25_years(self):
        r, _, _ = cached_robustness()
        g = r.grid.grid
        assert g["sharpe"].notna().all() and g["sharpe_is"].notna().all()
        assert g["first_date"].min() == pd.Timestamp("2000-05-04")  # L 63, vol 20
        assert g["first_date"].max() == pd.Timestamp("2001-08-02")  # L 252, vol 120, lag 2
        assert (g["n_days_is"] + g["n_days_oos"] == g["n_days"]).all()
        assert (g["n_days_oos"] == 3442).all()  # 2013-01-02 to 2026-09-07 for every trial
        assert g["cap_bind_fraction"].max() < 0.002

    def test_the_selection_picks_the_a_priori_lookback(self):
        r, _, _ = cached_robustness()
        t, s = r.is_oos_table, r.is_oos_summary
        assert t["sharpe_is"].tolist() == pytest.approx([0.1877, 0.2639, -0.0148, 0.2936], abs=1e-4)
        assert t["sharpe_oos"].tolist() == pytest.approx(
            [-0.2238, 0.0729, 0.1865, 0.3827], abs=1e-4
        )
        assert t["n_days_is"].tolist() == [3156, 3093, 3030, 2968]
        assert s["selected_lookback"] == s["default_lookback"] == 252
        assert s["oos_gap"] == 0.0
        assert s["selected_degradation"] == pytest.approx(-0.0891, abs=1e-4)

    def test_cost_sweep_falls_gently(self):
        r, _, _ = cached_robustness()
        c = r.cost_sweep
        assert c["sharpe"].tolist() == pytest.approx([0.3680, 0.3422, 0.3163, 0.2646], abs=1e-4)
        assert c["sharpe"].is_monotonic_decreasing
        bt, _ = cached_backtest()
        assert c.loc[5.0, "sharpe"] == bt.summary["sharpe_net"]
        assert c.loc[0.0, "sharpe"] == bt.summary["sharpe_gross"]
        assert c["grid_share_positive"].tolist() == pytest.approx(
            [185 / 192, 143 / 192, 104 / 192, 70 / 192]
        )

    def test_lag_two_is_below_lag_one(self):
        r, _, _ = cached_robustness()
        lag = r.lag_sweep
        assert lag["sharpe_net"].tolist() == pytest.approx([0.3422, 0.2867], abs=1e-4)
        assert lag["sharpe_gross"].tolist() == pytest.approx([0.3680, 0.3125], abs=1e-4)

    def test_where_the_default_sits(self):
        r, _, _ = cached_robustness()
        d = r.distribution
        assert d["default_rank"] == 12
        assert d["sharpe_median"] == pytest.approx(0.1156, abs=1e-4)
        assert d["sharpe_min"] == pytest.approx(-0.2870, abs=1e-4)
        assert d["sharpe_max"] == pytest.approx(0.4752, abs=1e-4)
        assert d["best_trial"] == "L 252 / vol 120 rolling / daily / skip 0 / lag 1"
        assert d["worst_trial"] == "L 63 / vol 20 rolling / daily / skip 0 / lag 2"

    def test_calendar_blocks(self):
        r, _, _ = cached_robustness()
        net = r.blocks[r.blocks["series"] == "Strategy (net)"].set_index("block")
        assert net["n_days"].tolist() == [1709, 1511, 1508, 1682]
        assert net["n_days"].sum() == 6410  # the blocks tile the life
        assert net["sharpe"].tolist() == pytest.approx([0.5170, 0.2115, -0.2188, 0.7963], abs=1e-4)
        # The expected pattern: strongest 2000-2007 and 2020 on, weakest 2014-2019.
        assert net["sharpe"].idxmin() == "2014-2019" and net["sharpe"].idxmax() == "2020-present"

    def test_calendar_years_and_rolling_windows(self):
        r, _, _ = cached_robustness()
        assert r.years.index.tolist() == list(range(2001, 2027))
        net = r.years["Strategy (net)"]
        assert net.loc[2016] == pytest.approx(-0.2162, abs=1e-4)  # worst year
        assert net.loc[2025] == pytest.approx(0.2460, abs=1e-4)  # best year
        assert (net > 0).sum() == 16 and (net < 0).sum() == 10  # 2008 is -0.06%
        rs = r.rolling_summary
        # 6,410 days - 756 + 1 = 5,655 windows; ^BCOM 6,356 - 756 + 1 = 5,601.
        assert rs.loc["Strategy (net)", "n_windows"] == 5655
        assert rs.loc["^BCOM", "n_windows"] == 5601
        assert rs.loc["Strategy (net)", "min_date"] == pd.Timestamp("2019-01-24")
        assert rs.loc["Strategy (net)", "min"] == pytest.approx(-1.1037, abs=1e-4)
        assert rs.loc["Strategy (net)", "max"] == pytest.approx(1.2686, abs=1e-4)

    def test_regimes(self):
        r, _, _ = cached_robustness()
        g = r.regimes.set_index(["regime", "series"])
        assert g.loc[("stress", "Strategy (net)"), "n_days"] == 1544
        assert g.loc[("calm", "Strategy (net)"), "n_days"] == 4866
        assert g.loc[("stress", "Strategy (net)"), "sharpe"] == pytest.approx(-0.4763, abs=1e-4)
        assert g.loc[("calm", "Strategy (net)"), "sharpe"] == pytest.approx(0.6001, abs=1e-4)

    def test_metrics_with_the_grid_deflate_the_strategy(self, tmp_path):
        r, _, _ = cached_robustness()
        t = r.trials
        m = run_metrics(quick_config(tmp_path), ROOT, t["n_trials"], t["sharpe_std"])
        net, gross = m.metrics["Strategy (net)"], m.metrics["Strategy (gross)"]
        assert net["dsr"] == t["dsr"] and net["sharpe_hurdle"] == t["sharpe_hurdle"]
        assert gross["dsr"] == pytest.approx(0.3713, abs=1e-4)
        assert m.metrics["EW long-only (net)"]["dsr"] == m.metrics["EW long-only (net)"]["psr"]


class TestStageWiring:
    def test_metrics_stage_deflates_the_strategy_by_the_grid(self, tmp_path, monkeypatch, capsys):
        cfg = small_grid_config(tmp_path)
        monkeypatch.setattr(cli, "load_config", lambda: cfg)
        cli._stage_metrics()
        g = pd.read_parquet(tmp_path / GRID_FILE)
        n, std = trial_stats(g, cfg.costs.cost_bps)
        metrics = json.loads((tmp_path / METRICS_FILE).read_text())
        net = metrics["Strategy (net)"]
        assert net["n_trials"] == n == 2
        assert net["sharpe_hurdle"] == pytest.approx(expected_max_sharpe(2, std))
        assert metrics["^BCOM"]["n_trials"] == 1
        assert "grid: 2 trials" in capsys.readouterr().out

    def test_robustness_stage_prints_its_headline(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_config", lambda: small_grid_config(tmp_path))
        cli._stage_robustness()
        printed = capsys.readouterr().out
        assert "IS/OOS: selected L = " in printed and "DSR" in printed
        assert (tmp_path / SWEEPS_REPORT).exists()


# --------------------------------------------------------------------------- main


class TestMain:
    def test_backtest_and_benchmarks_are_registered(self):
        assert "backtest" in STAGES and "benchmarks" in STAGES
        assert set(STAGES) <= set(KNOWN_STAGES)

    def test_metrics_stage_is_registered_after_benchmarks(self):
        assert "metrics" in STAGES
        assert KNOWN_STAGES.index("metrics") == KNOWN_STAGES.index("benchmarks") + 1

    def test_makefile_runs_metrics_in_all(self):
        makefile = (ROOT / "Makefile").read_text()
        assert "\nmetrics:\n\t$(CLI) metrics\n" in makefile
        all_line = next(line for line in makefile.splitlines() if line.startswith("all:"))
        stages = all_line.split(":", 1)[1].split()
        assert stages.index("metrics") == stages.index("benchmarks") + 1

    def test_robustness_stage_is_registered_and_runs_after_metrics_in_all(self):
        assert "robustness" in STAGES
        makefile = (ROOT / "Makefile").read_text()
        assert "\nrobustness:\n\t$(CLI) robustness\n" in makefile
        all_line = next(line for line in makefile.splitlines() if line.startswith("all:"))
        stages = all_line.split(":", 1)[1].split()
        assert stages.index("robustness") == stages.index("metrics") + 1

    def test_usage_and_unimplemented_stage_exit_codes(self, capsys):
        assert main([]) == 2
        assert main(["nonsense"]) == 2
        assert main(["figures"]) == 1
        assert "not implemented" in capsys.readouterr().err

    def test_nan_free_results_for_the_strategy(self):
        bt, _ = cached_backtest()
        life = bt.result.gross_returns.notna()
        assert not np.isnan(bt.result.net_returns[life]).any()
