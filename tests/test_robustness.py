"""Robustness (Phase 6, F6.1, F6.2, F6.4): the grid's trials, one trial's evaluation, the
cache fingerprint, the Deflated Sharpe Ratio's inputs, the in-sample / out-of-sample
selection of the lookback, and the cost and lag sweeps.

The grid functions are tested on small hand-built grid frames (``grid_frame`` below) whose
expected values are read off by eye; ``evaluate_trial`` is tested on a seeded synthetic
return panel against the pipeline composed by hand from the Phase 2-5 functions it chains.
The full 192-trial grid on the snapshot is exercised in ``tests/test_cli.py``.
"""

import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.backtest import run
from src.config import ANNUALISATION, load_config
from src.data import clean_price_index
from src.metrics import annualised_return, annualised_vol, max_drawdown, sharpe
from src.portfolio import size_positions
from src.robustness import (
    COST_SWEEP_COLUMNS,
    DISTRIBUTION_KEYS,
    GRID_COLUMNS,
    GRID_DTYPES,
    GRID_KEYS,
    IS_OOS_COLUMNS,
    IS_OOS_KEYS,
    LAG_SWEEP_COLUMNS,
    TRIAL_COLUMNS,
    cost_sweep,
    default_spec,
    evaluate_trial,
    grid_distribution,
    grid_specs,
    is_oos,
    lag_sweep,
    matches,
    trial_config,
    trial_fingerprint,
    trial_label,
    trial_stats,
)
from src.signals import tsmom

NA = np.nan
CFG = load_config()
DEFAULT = {
    "lookback": 252,
    "vol_window": 60,
    "estimator": "rolling",
    "rebalance": "monthly",
    "skip": 0,
    "lag": 1,
}


def grid_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """A grid frame from partial rows: every missing grid key takes the default trial's
    value, ``cost_bps`` defaults to 5, every other column to NaN (dates NaT, counts 0)."""
    full = []
    for row in rows:
        r: dict[str, object] = {**DEFAULT, "cost_bps": 5.0}
        for col in GRID_COLUMNS:
            if col not in r:
                if col == "first_date":
                    r[col] = pd.NaT
                elif col.startswith("n_days"):
                    r[col] = 0
                elif col == "fingerprint":
                    r[col] = "f"
                else:
                    r[col] = NA
        r.update(row)
        full.append(r)
    return pd.DataFrame(full, columns=list(GRID_COLUMNS)).astype(GRID_DTYPES)


# ------------------------------------------------------------------------ constants


class TestConstants:
    def test_grid_keys_and_columns(self):
        assert GRID_KEYS == ("lookback", "vol_window", "estimator", "rebalance", "skip", "lag")
        assert TRIAL_COLUMNS[:7] == (*GRID_KEYS, "cost_bps")
        assert GRID_COLUMNS == (*TRIAL_COLUMNS, "fingerprint")
        assert set(GRID_DTYPES) == set(GRID_COLUMNS)
        assert len(set(GRID_COLUMNS)) == len(GRID_COLUMNS)

    def test_default_grid_lists_are_the_spec_grid(self):
        # PROJECT_LOG.md §5.3: 4 x 3 x 2 x 2 x 2 x 2 = 192 configurations, 4 costs each.
        r = CFG.robustness
        assert r.grid_lookbacks == (63, 126, 189, 252)
        assert r.grid_vol_windows == (20, 60, 120)
        assert r.grid_estimators == ("rolling", "ewma")
        assert r.grid_rebalances == ("monthly", "daily")
        assert r.grid_skips == (0, 21)
        assert r.grid_lags == (1, 2)
        assert r.cost_bps_sweep == (0.0, 5.0, 10.0, 20.0)
        assert r.regime_vol_quantile == 0.75

    def test_report_keys(self):
        assert IS_OOS_KEYS[0] == "selected_lookback" and IS_OOS_KEYS[-1] == "oos_gap"
        assert IS_OOS_COLUMNS[-2:] == ("selected", "default")
        assert COST_SWEEP_COLUMNS[:2] == ("ann_return", "sharpe")
        assert LAG_SWEEP_COLUMNS == (
            "sharpe_gross",
            "sharpe_net",
            "ann_return_net",
            "grid_median_net",
        )
        assert DISTRIBUTION_KEYS[:2] == ("n_trials", "sharpe_std")


class TestProvidedHelpers:
    def test_default_spec_reads_the_config(self):
        assert default_spec(CFG) == DEFAULT

    def test_matches(self):
        g = grid_frame([{"lookback": 63}, {"lookback": 252}, {"lookback": 252, "cost_bps": 0.0}])
        assert matches(g, {"lookback": 252}).tolist() == [False, True, True]
        assert matches(g, {"lookback": 252, "cost_bps": 5.0}).tolist() == [False, True, False]
        assert matches(g, {}).all()

    def test_trial_label(self):
        assert trial_label(DEFAULT) == "L 252 / vol 60 rolling / monthly / skip 0 / lag 1"


# ----------------------------------------------------------------------- grid_specs


def small_robustness(**changes):
    base = {
        "grid_lookbacks": (20, 40),
        "grid_vol_windows": (10,),
        "grid_estimators": ("rolling", "ewma"),
        "grid_rebalances": ("monthly",),
        "grid_skips": (0, 5),
        "grid_lags": (1,),
    }
    base.update(changes)
    return dataclasses.replace(CFG.robustness, **base)


class TestGridSpecs:
    def test_default_grid_has_192_trials(self):
        specs = grid_specs(CFG.robustness)
        assert len(specs) == 192
        assert list(specs.columns) == list(GRID_KEYS)
        assert not specs.duplicated().any()

    def test_index_and_dtypes(self):
        specs = grid_specs(CFG.robustness)
        assert isinstance(specs.index, pd.RangeIndex) and specs.index.name == "trial"
        assert specs.index[0] == 0 and specs.index[-1] == 191
        for key in GRID_KEYS:
            assert str(specs[key].dtype) == str(pd.Series([], dtype=GRID_DTYPES[key]).dtype), key

    def test_product_order_first_key_slowest(self):
        specs = grid_specs(CFG.robustness)
        # Trial 0: every list's first value; trial 1: the lag (fastest) moves to 2; trial 2:
        # the skip moves to 21; 192 / 4 = 48 trials per lookback.
        assert specs.iloc[0].tolist() == [63, 20, "rolling", "monthly", 0, 1]
        assert specs.iloc[1].tolist() == [63, 20, "rolling", "monthly", 0, 2]
        assert specs.iloc[2].tolist() == [63, 20, "rolling", "monthly", 21, 1]
        assert specs.iloc[-1].tolist() == [252, 120, "ewma", "daily", 21, 2]
        assert (specs["lookback"].iloc[:48] == 63).all() and specs["lookback"].iloc[48] == 126

    def test_default_trial_is_in_the_grid_once(self):
        specs = grid_specs(CFG.robustness)
        assert matches(specs, DEFAULT).sum() == 1

    def test_small_grid_follows_the_config_lists(self):
        specs = grid_specs(small_robustness())
        assert len(specs) == 2 * 1 * 2 * 1 * 2 * 1
        assert specs["lookback"].tolist() == [20, 20, 20, 20, 40, 40, 40, 40]
        assert specs["estimator"].tolist() == ["rolling", "rolling", "ewma", "ewma"] * 2
        assert specs["skip"].tolist() == [0, 5] * 4

    @pytest.mark.parametrize(
        "changes",
        [
            {"grid_lookbacks": ()},
            {"grid_lags": ()},
            {"grid_lookbacks": (20, 20)},
            {"grid_estimators": ("ewma", "ewma")},
            {"grid_skips": (0, 20)},  # skip 20 >= lookback 20
        ],
    )
    def test_invalid_grids_raise(self, changes):
        with pytest.raises(ValueError):
            grid_specs(small_robustness(**changes))


# ---------------------------------------------------------------------- trial_config


class TestTrialConfig:
    SPEC = {
        "lookback": 63,
        "vol_window": 20,
        "estimator": "ewma",
        "rebalance": "daily",
        "skip": 21,
        "lag": 2,
    }

    def test_replaces_the_six_parameters(self):
        c = trial_config(CFG, self.SPEC)
        assert (c.signal.lookback, c.signal.skip) == (63, 21)
        assert (c.sizing.vol_window, c.sizing.vol_estimator) == (20, "ewma")
        assert (c.execution.rebalance, c.execution.lag) == ("daily", 2)

    def test_everything_else_unchanged_and_input_untouched(self):
        c = trial_config(CFG, self.SPEC)
        assert c.sizing.leverage_cap == CFG.sizing.leverage_cap
        assert c.sizing.sigma_target_asset == CFG.sizing.sigma_target_asset
        assert c.costs == CFG.costs and c.data == CFG.data and c.robustness == CFG.robustness
        assert c.inference == CFG.inference and c.paths == CFG.paths
        assert CFG.signal.lookback == 252 and CFG.execution.lag == 1

    def test_accepts_a_grid_row_and_returns_python_types(self):
        row = grid_specs(CFG.robustness).iloc[1]  # numpy scalars in a Series
        c = trial_config(CFG, row)
        assert type(c.signal.lookback) is int and type(c.execution.lag) is int
        assert type(c.sizing.vol_estimator) is str
        assert (c.signal.lookback, c.execution.lag) == (63, 2)

    def test_default_spec_round_trips(self):
        assert trial_config(CFG, default_spec(CFG)) == CFG

    def test_missing_key_raises(self):
        spec = dict(self.SPEC)
        del spec["lag"]
        with pytest.raises(KeyError):
            trial_config(CFG, spec)


# --------------------------------------------------------------------- evaluate_trial

DAYS = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=520)), name="date")


def synthetic_returns(seed: int = 7) -> pd.DataFrame:
    """Three instruments, 520 business days from 2020-01-01 (to 2021-12-28), i.i.d. normal
    returns with drifts of opposite sign so the signal has both longs and shorts."""
    rng = np.random.default_rng(seed)
    drift = np.array([0.0008, -0.0006, 0.0002])
    values = drift + rng.normal(0.0, 0.012, size=(len(DAYS), 3))
    return pd.DataFrame(values, index=DAYS, columns=["AA", "BB", "CC"])


SYN_CFG = load_config(
    overrides={
        "signal.lookback": 40,
        "sizing.vol_window": 20,
        "robustness.cost_bps_sweep": [0, 5, 20],
        "robustness.is_end": "2020-12-31",
        "robustness.oos_start": "2021-01-01",
    }
)


def composed(returns: pd.DataFrame, cfg, cost: float):
    """The pipeline evaluate_trial must reproduce, composed by hand."""
    signal = tsmom(clean_price_index(returns), cfg.signal.lookback, cfg.signal.skip)
    sized = size_positions(signal, returns, cfg.sizing, cfg.execution.rebalance)
    result = run(sized.weights, returns, cfg.execution.lag, cost)
    return sized, result


class TestEvaluateTrial:
    def test_one_row_per_cost_with_the_trial_columns(self):
        out = evaluate_trial(synthetic_returns(), SYN_CFG)
        assert list(out.columns) == list(TRIAL_COLUMNS)
        assert isinstance(out.index, pd.RangeIndex) and len(out) == 3
        assert out["cost_bps"].tolist() == [0.0, 5.0, 20.0]
        for key, value in default_spec(SYN_CFG).items():
            assert (out[key] == value).all(), key

    def test_dtypes(self):
        out = evaluate_trial(synthetic_returns(), SYN_CFG)
        for col in TRIAL_COLUMNS:
            expected = pd.Series([], dtype=GRID_DTYPES[col]).dtype
            assert str(out[col].dtype) == str(expected), col

    def test_numbers_are_the_composed_pipeline(self):
        returns = synthetic_returns()
        out = evaluate_trial(returns, SYN_CFG)
        rf = SYN_CFG.inference.risk_free_rate
        for i, cost in enumerate((0.0, 5.0, 20.0)):
            sized, result = composed(returns, SYN_CFG, cost)
            life = result.gross_returns.notna()
            net = result.net_returns[life]
            row = out.iloc[i]
            assert row["first_date"] == net.index[0]
            assert row["n_days"] == len(net)
            assert row["ann_return"] == pytest.approx(annualised_return(net), rel=1e-12)
            assert row["ann_vol"] == pytest.approx(annualised_vol(net), rel=1e-12)
            assert row["sharpe"] == pytest.approx(sharpe(net, rf), rel=1e-12)
            assert row["max_drawdown"] == pytest.approx(max_drawdown(net), rel=1e-12)
            assert row["sharpe_is"] == pytest.approx(sharpe(net.loc[:"2020-12-31"], rf), rel=1e-12)
            assert row["sharpe_oos"] == pytest.approx(sharpe(net.loc["2021-01-01":], rf), rel=1e-12)
            turnover = result.turnover[life].mean() * ANNUALISATION
            assert row["ann_turnover"] == pytest.approx(turnover, rel=1e-12)
            assert row["mean_gross_leverage"] == pytest.approx(result.gross_leverage[life].mean())
            assert row["cap_bind_fraction"] == pytest.approx(sized.cap_binds[life].mean())

    def test_cost_zero_row_is_gross(self):
        returns = synthetic_returns()
        out = evaluate_trial(returns, SYN_CFG)
        _, result = composed(returns, SYN_CFG, 5.0)
        gross = result.gross_returns.dropna()
        assert out.iloc[0]["sharpe"] == pytest.approx(sharpe(gross), rel=1e-12)

    def test_sharpe_falls_as_costs_rise_and_the_book_does_not_change(self):
        out = evaluate_trial(synthetic_returns(), SYN_CFG)
        assert (
            out["sharpe"].is_monotonic_decreasing and out["sharpe"].iloc[0] > out["sharpe"].iloc[-1]
        )
        for col in ("ann_turnover", "mean_gross_leverage", "cap_bind_fraction", "n_days"):
            assert out[col].nunique() == 1, col

    def test_in_and_out_of_sample_days_split_the_life(self):
        out = evaluate_trial(synthetic_returns(), SYN_CFG)
        row = out.iloc[1]
        # is_end 2020-12-31 and oos_start 2021-01-01 are adjacent: every life row is in one.
        assert row["n_days_is"] + row["n_days_oos"] == row["n_days"]
        # The synthetic calendar has 262 business days in 2020 and 258 in 2021 (to
        # 2021-12-28); every 2021 row is in the life, which starts in 2020.
        assert row["n_days_oos"] == 258
        assert row["n_days_is"] > 0

    def test_the_trial_carries_its_own_parameters(self):
        returns = synthetic_returns()
        spec = {
            "lookback": 20,
            "vol_window": 10,
            "estimator": "ewma",
            "rebalance": "daily",
            "skip": 5,
            "lag": 2,
        }
        cfg = trial_config(SYN_CFG, spec)
        out = evaluate_trial(returns, cfg)
        for key, value in spec.items():
            assert (out[key] == value).all(), key
        _, result = composed(returns, cfg, 5.0)
        net = result.net_returns.dropna()
        assert out.iloc[1]["sharpe"] == pytest.approx(sharpe(net), rel=1e-12)
        assert out.iloc[1]["first_date"] == net.index[0]

    def test_a_trial_that_never_holds_a_position(self):
        returns = synthetic_returns().iloc[:60]  # too short for a 40-day signal and its vols
        out = evaluate_trial(returns, SYN_CFG)
        assert len(out) == 3
        assert out["first_date"].isna().all()
        assert (out["n_days"] == 0).all() and (out["n_days_is"] == 0).all()
        assert out["sharpe"].isna().all() and out["sharpe_oos"].isna().all()

    def test_inputs_not_modified(self):
        returns = synthetic_returns()
        before = returns.copy()
        evaluate_trial(returns, SYN_CFG)
        pd.testing.assert_frame_equal(returns, before)


class TestIntegrityIsOosSelection:
    """§5.3: the lookback is selected on in-sample data only. The in-sample statistics of a
    trial cannot depend on any return after ``is_end``, and the selection reads nothing but
    them."""

    def test_in_sample_statistics_are_blind_to_out_of_sample_returns(self):
        returns = synthetic_returns()
        attacked = returns.copy()
        rng = np.random.default_rng(1)
        future = attacked.index > pd.Timestamp("2020-12-31")
        attacked.loc[future] = rng.normal(0.0, 0.05, size=(int(future.sum()), 3))
        a = evaluate_trial(returns, SYN_CFG)
        b = evaluate_trial(attacked, SYN_CFG)
        for col in ("sharpe_is", "n_days_is", "first_date"):
            pd.testing.assert_series_equal(a[col], b[col], check_exact=True)
        assert not np.allclose(a["sharpe_oos"], b["sharpe_oos"])

    def test_selection_ignores_the_out_of_sample_columns(self):
        g = grid_frame(
            [
                {"lookback": 63, "sharpe_is": 0.2, "sharpe_oos": 0.9},
                {"lookback": 126, "sharpe_is": 0.5, "sharpe_oos": 0.1},
                {"lookback": 252, "sharpe_is": 0.3, "sharpe_oos": 0.4},
            ]
        )
        scrambled = g.assign(sharpe_oos=[-5.0, 7.0, NA], n_days_oos=[1, 2, 3])
        assert is_oos(g, CFG)[1]["selected_lookback"] == 126
        assert is_oos(scrambled, CFG)[1]["selected_lookback"] == 126


# ------------------------------------------------------------------ trial_fingerprint

SHA = "ab" * 32


class TestTrialFingerprint:
    def test_is_the_documented_payload_hashed(self):
        payload = {
            "panel_sha256": SHA,
            "clip": 0.5,
            "sigma_target_asset": 0.10,
            "sigma_target_portfolio": 0.10,
            "leverage_cap": 3.0,
            "cost_bps_sweep": [0.0, 5.0, 10.0, 20.0],
            "is_end": "2012-12-31",
            "oos_start": "2013-01-01",
            "risk_free_rate": 0.0,
        }
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        assert trial_fingerprint(CFG, SHA) == expected
        assert len(expected) == 64

    @pytest.mark.parametrize(
        "key,value",
        [
            ("data.clip", 0.4),
            ("sizing.sigma_target_asset", 0.2),
            ("sizing.sigma_target_portfolio", 0.2),
            ("sizing.leverage_cap", 2.0),
            ("robustness.cost_bps_sweep", [0, 5]),
            ("robustness.is_end", "2011-12-31"),
            ("robustness.oos_start", "2014-01-01"),
            ("inference.risk_free_rate", 0.02),
        ],
    )
    def test_changes_with_every_input_of_a_trial(self, key, value):
        changed = load_config(overrides={key: value})
        assert trial_fingerprint(changed, SHA) != trial_fingerprint(CFG, SHA)

    def test_changes_with_the_panel(self):
        assert trial_fingerprint(CFG, "cd" * 32) != trial_fingerprint(CFG, SHA)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("robustness.grid_lookbacks", [126, 252]),
            ("signal.lookback", 126),
            ("execution.lag", 2),
            ("costs.cost_bps", 10),
            ("paths.results_dir", "elsewhere"),
            ("inference.n_resamples", 10),
            ("robustness.regime_vol_quantile", 0.9),
        ],
    )
    def test_ignores_what_a_trial_row_does_not_depend_on(self, key, value):
        changed = load_config(overrides={key: value})
        assert trial_fingerprint(changed, SHA) == trial_fingerprint(CFG, SHA)


# ------------------------------------------------------------------------ trial_stats


class TestTrialStats:
    def test_count_and_sample_std_at_one_cost(self):
        # Sharpes 0.1, 0.3, 0.5 at 5 bps: mean 0.3, squared deviations 0.04 + 0 + 0.04 = 0.08,
        # / (3 - 1) = 0.04, std 0.2. The 0-bps rows are another cost and do not count.
        g = grid_frame(
            [
                {"lookback": 63, "sharpe": 0.1},
                {"lookback": 126, "sharpe": 0.3},
                {"lookback": 252, "sharpe": 0.5},
                {"lookback": 63, "cost_bps": 0.0, "sharpe": 9.0},
            ]
        )
        n, std = trial_stats(g, 5.0)
        assert n == 3 and isinstance(n, int)
        assert std == pytest.approx(0.2)

    def test_a_nan_sharpe_is_counted_as_a_trial_but_not_in_the_spread(self):
        g = grid_frame(
            [{"lookback": 63, "sharpe": 0.1}, {"lookback": 126, "sharpe": 0.5}, {"lookback": 252}]
        )
        n, std = trial_stats(g, 5.0)
        assert n == 3
        assert std == pytest.approx(np.std([0.1, 0.5], ddof=1))

    def test_one_trial_has_no_spread(self):
        assert trial_stats(grid_frame([{"sharpe": 0.4}]), 5.0) == (1, 0.0)

    def test_missing_cost_raises(self):
        with pytest.raises(ValueError):
            trial_stats(grid_frame([{"sharpe": 0.4}]), 10.0)


# ---------------------------------------------------------------------------- is_oos

IS_OOS_GRID = [
    {"lookback": 63, "sharpe_is": 0.20, "sharpe_oos": 0.30, "n_days_is": 2900, "n_days_oos": 3450},
    {"lookback": 126, "sharpe_is": 0.60, "sharpe_oos": 0.10, "n_days_is": 2850, "n_days_oos": 3450},
    {"lookback": 189, "sharpe_is": 0.60, "sharpe_oos": 0.50, "n_days_is": 2800, "n_days_oos": 3450},
    {"lookback": 252, "sharpe_is": 0.40, "sharpe_oos": 0.25, "n_days_is": 2750, "n_days_oos": 3450},
    # Not candidates: another vol window (higher IS Sharpe), another cost.
    {"lookback": 126, "vol_window": 20, "sharpe_is": 0.99, "sharpe_oos": 0.0},
    {"lookback": 252, "cost_bps": 0.0, "sharpe_is": 0.99, "sharpe_oos": 0.0},
]


class TestIsOos:
    def test_table(self):
        table, _ = is_oos(grid_frame(IS_OOS_GRID), CFG)
        assert list(table.columns) == list(IS_OOS_COLUMNS)
        assert table.index.name == "lookback" and table.index.tolist() == [63, 126, 189, 252]
        assert table["sharpe_is"].tolist() == [0.20, 0.60, 0.60, 0.40]
        # degradation = IS - OOS: -0.10, 0.50, 0.10, 0.15
        assert table["degradation"].tolist() == pytest.approx([-0.10, 0.50, 0.10, 0.15])
        assert table["n_days_is"].tolist() == [2900, 2850, 2800, 2750]
        assert table["selected"].tolist() == [False, True, False, False]
        assert table["default"].tolist() == [False, False, False, True]

    def test_summary_tie_goes_to_the_shorter_lookback(self):
        # 126 and 189 tie at 0.60 in sample: 126 is selected (its OOS 0.10 is worse luck).
        _, s = is_oos(grid_frame(IS_OOS_GRID), CFG)
        assert list(s) == list(IS_OOS_KEYS)
        assert s["selected_lookback"] == 126 and isinstance(s["selected_lookback"], int)
        assert s["selected_is"] == pytest.approx(0.60) and s["selected_oos"] == pytest.approx(0.10)
        assert s["selected_degradation"] == pytest.approx(0.50)
        assert s["default_lookback"] == 252
        assert s["default_is"] == pytest.approx(0.40) and s["default_oos"] == pytest.approx(0.25)
        assert s["default_degradation"] == pytest.approx(0.15)
        assert s["oos_gap"] == pytest.approx(0.10 - 0.25)

    def test_nan_in_sample_sharpe_is_never_selected(self):
        rows = [dict(r) for r in IS_OOS_GRID[:4]]
        rows[0]["sharpe_is"] = NA
        rows[1]["sharpe_is"] = NA
        rows[2]["sharpe_is"] = NA
        _, s = is_oos(grid_frame(rows), CFG)
        assert s["selected_lookback"] == 252

    def test_the_default_can_be_selected(self):
        rows = [dict(r) for r in IS_OOS_GRID[:4]]
        rows[3]["sharpe_is"] = 0.70
        table, s = is_oos(grid_frame(rows), CFG)
        assert s["selected_lookback"] == 252 and s["oos_gap"] == 0.0
        assert table.loc[252, "selected"] and table.loc[252, "default"]

    def test_errors(self):
        with pytest.raises(ValueError):
            is_oos(grid_frame([{"lookback": 63, "sharpe_is": 0.2}]), CFG)  # no default L
        with pytest.raises(ValueError):
            is_oos(grid_frame([{"lookback": 252, "cost_bps": 0.0, "sharpe_is": 0.2}]), CFG)
        with pytest.raises(ValueError):
            is_oos(grid_frame([{"lookback": 126}, {"lookback": 252}]), CFG)  # all NaN


# ------------------------------------------------------------------------ cost_sweep

SWEEP_GRID = [
    # The default trial at four costs, then one other trial at the same costs and one at
    # lag 2 (the lag sweep's second row).
    {"cost_bps": 0.0, "sharpe": 0.40, "ann_return": 0.040, "max_drawdown": -0.30},
    {"cost_bps": 5.0, "sharpe": 0.36, "ann_return": 0.036, "max_drawdown": -0.31},
    {"cost_bps": 10.0, "sharpe": 0.32, "ann_return": 0.032, "max_drawdown": -0.32},
    {"cost_bps": 20.0, "sharpe": 0.24, "ann_return": 0.024, "max_drawdown": -0.34},
    {"lookback": 63, "cost_bps": 0.0, "sharpe": 0.10},
    {"lookback": 63, "cost_bps": 5.0, "sharpe": -0.02},
    {"lookback": 63, "cost_bps": 10.0, "sharpe": -0.14},
    {"lookback": 63, "cost_bps": 20.0, "sharpe": -0.38},
    {"lag": 2, "cost_bps": 0.0, "sharpe": 0.38},
    {"lag": 2, "cost_bps": 5.0, "sharpe": 0.34, "ann_return": 0.034},
    {"lag": 2, "cost_bps": 10.0, "sharpe": 0.30},
    {"lag": 2, "cost_bps": 20.0, "sharpe": 0.22},
]


class TestCostSweep:
    def test_default_trial_at_each_cost_and_the_grid_median(self):
        out = cost_sweep(grid_frame(SWEEP_GRID), CFG)
        assert list(out.columns) == list(COST_SWEEP_COLUMNS)
        assert out.index.name == "cost_bps" and out.index.tolist() == [0.0, 5.0, 10.0, 20.0]
        assert out["sharpe"].tolist() == [0.40, 0.36, 0.32, 0.24]
        assert out["ann_return"].tolist() == [0.040, 0.036, 0.032, 0.024]
        assert out["max_drawdown"].tolist() == [-0.30, -0.31, -0.32, -0.34]
        # Medians of three trials: (0.40, 0.10, 0.38) -> 0.38; (0.36, -0.02, 0.34) -> 0.34;
        # (0.32, -0.14, 0.30) -> 0.30; (0.24, -0.38, 0.22) -> 0.22.
        assert out["grid_median_sharpe"].tolist() == pytest.approx([0.38, 0.34, 0.30, 0.22])
        # Share > 0: 3/3, 2/3, 2/3, 2/3.
        assert out["grid_share_positive"].tolist() == pytest.approx([1.0, 2 / 3, 2 / 3, 2 / 3])

    def test_costs_sorted_ascending(self):
        out = cost_sweep(grid_frame(list(reversed(SWEEP_GRID))), CFG)
        assert out.index.tolist() == [0.0, 5.0, 10.0, 20.0]
        assert out["sharpe"].tolist() == [0.40, 0.36, 0.32, 0.24]

    def test_missing_default_raises(self):
        with pytest.raises(ValueError):
            cost_sweep(grid_frame(SWEEP_GRID[4:8]), CFG)


class TestLagSweep:
    def test_default_trial_at_each_lag(self):
        out = lag_sweep(grid_frame(SWEEP_GRID), CFG)
        assert list(out.columns) == list(LAG_SWEEP_COLUMNS)
        assert out.index.name == "lag" and out.index.tolist() == [1, 2]
        assert out["sharpe_gross"].tolist() == [0.40, 0.38]
        assert out["sharpe_net"].tolist() == [0.36, 0.34]
        assert out["ann_return_net"].tolist() == [0.036, 0.034]
        # Lag 1 at 5 bps: the default (0.36) and L 63 (-0.02), median 0.17; lag 2: 0.34 alone.
        assert out["grid_median_net"].tolist() == pytest.approx([0.17, 0.34])

    def test_no_zero_cost_gives_nan_gross(self):
        rows = [r for r in SWEEP_GRID if r["cost_bps"] != 0.0]
        out = lag_sweep(grid_frame(rows), CFG)
        assert out["sharpe_gross"].isna().all()
        assert out["sharpe_net"].tolist() == [0.36, 0.34]

    def test_missing_default_at_a_lag_raises(self):
        rows = [r for r in SWEEP_GRID if not (r.get("lag") == 2 and r["cost_bps"] == 5.0)]
        with pytest.raises(ValueError):
            lag_sweep(grid_frame(rows), CFG)


# ------------------------------------------------------------------ grid_distribution


class TestGridDistribution:
    ROWS = [
        {"lookback": 63, "sharpe": 0.10},
        {"lookback": 126, "sharpe": 0.50},
        {"lookback": 189, "sharpe": -0.20},
        {"lookback": 252, "sharpe": 0.30},
        {"lookback": 252, "vol_window": 20, "sharpe": 0.50},
        {"lookback": 252, "cost_bps": 0.0, "sharpe": 9.0},  # another cost: ignored
    ]

    def test_values(self):
        d = grid_distribution(grid_frame(self.ROWS), CFG)
        assert list(d) == list(DISTRIBUTION_KEYS)
        # Sharpes at 5 bps: -0.20, 0.10, 0.30, 0.50, 0.50 (sorted).
        assert d["n_trials"] == 5
        assert d["sharpe_std"] == pytest.approx(np.std([0.1, 0.5, -0.2, 0.3, 0.5], ddof=1))
        assert d["sharpe_min"] == pytest.approx(-0.20)
        assert d["sharpe_q25"] == pytest.approx(0.10)  # linear: position 1 of 0..4
        assert d["sharpe_median"] == pytest.approx(0.30)
        assert d["sharpe_q75"] == pytest.approx(0.50)
        assert d["sharpe_max"] == pytest.approx(0.50)
        assert d["share_positive"] == pytest.approx(4 / 5)
        assert d["default_sharpe"] == pytest.approx(0.30)
        # Two trials (0.50, 0.50) are strictly better: rank 3.
        assert d["default_rank"] == 3 and isinstance(d["default_rank"], int)
        # The first 0.50 in grid order is L 126; the lowest is L 189.
        assert d["best_trial"] == "L 126 / vol 60 rolling / monthly / skip 0 / lag 1"
        assert d["worst_trial"] == "L 189 / vol 60 rolling / monthly / skip 0 / lag 1"

    def test_default_best_has_rank_one(self):
        rows = [{"lookback": 63, "sharpe": 0.1}, {"lookback": 252, "sharpe": 0.3}]
        assert grid_distribution(grid_frame(rows), CFG)["default_rank"] == 1

    def test_missing_default_raises(self):
        with pytest.raises(ValueError):
            grid_distribution(grid_frame(self.ROWS[:3]), CFG)
