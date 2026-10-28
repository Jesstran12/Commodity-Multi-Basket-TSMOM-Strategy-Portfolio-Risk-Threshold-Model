"""Backtest engine and costs (Phase 4): positions, turnover, costs, gross and net returns,
the backtest summary, and the §8 lag, zero-turnover, cost-accounting and no-lookahead
integrity tests.

Synthetic frames use business days from 2020-01-01: d1..d12 =
01-01 01-02 01-03 01-06 01-07 01-08 01-09 01-10 01-13 01-14 01-15 01-16.
Every expected value is hand-computed; the arithmetic is in a comment next to it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.backtest import (
    SUMMARY_KEYS,
    BacktestResult,
    backtest_summary,
    gross_returns,
    positions_from_weights,
    run,
    transaction_costs,
    turnover,
)
from src.config import ANNUALISATION, BASIS_POINT, SizingConfig, load_config
from src.data import clean_price_index, clean_returns
from src.portfolio import rebalance_dates, size_positions
from src.signals import tsmom
from tests.integrity import assert_no_lookahead

ROOT = Path(__file__).resolve().parents[1]
NA = np.nan
SQ = np.sqrt(ANNUALISATION)  # 15.8745...
D = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=12)), name="date")  # d1..d12


def frame(**columns: list[float]) -> pd.DataFrame:
    """Wide frame on d1..dn (n = the common length of the lists); NaN where a value is NA."""
    n = len(next(iter(columns.values())))
    return pd.DataFrame({k: v for k, v in columns.items()}, index=D[:n], dtype=float)


def series(values: list[float], name: str) -> pd.Series:
    """Float series on d1..dn."""
    return pd.Series(values, index=D[: len(values)], dtype=float, name=name)


# The worked example used by several classes below (lag 1, 10 bps):
#   weights  A: [NA, 0.5, 0.5, -0.5, -0.5]   B: [NA, NA, 0.5, 0.5, 0.5]
#   returns  A: [0.01, 0.02, -0.01, 0.03, 0.01]   B: [0.02, 0.01, 0.02, -0.02, 0.04]
#   positions (shift 1)   d1 [NA, NA]  d2 [NA, NA]  d3 [0.5, NA]  d4 [0.5, 0.5]  d5 [-0.5, 0.5]
#   gross: d1, d2 NaN (no position yet); d3 0.5 * -0.01 = -0.005;
#          d4 0.5 * 0.03 + 0.5 * -0.02 = 0.005; d5 -0.5 * 0.01 + 0.5 * 0.04 = 0.015
#   exposure (NaN -> 0): d1 [0, 0] d2 [0, 0] d3 [0.5, 0] d4 [0.5, 0.5] d5 [-0.5, 0.5]
#   turnover: d1 0 (no previous row); d2 0; d3 0.5; d4 0.5; d5 |-0.5 - 0.5| = 1.0
#   cost at 10 bps = turnover * 0.001: 0, 0, 0.0005, 0.0005, 0.001
#   net = gross - cost: NaN, NaN, -0.0055, 0.0045, 0.014
#   gross leverage: NaN, NaN, 0.5, 1.0, 1.0;  n_live: 0, 0, 1, 2, 2
W = frame(A=[NA, 0.5, 0.5, -0.5, -0.5], B=[NA, NA, 0.5, 0.5, 0.5])
R = frame(A=[0.01, 0.02, -0.01, 0.03, 0.01], B=[0.02, 0.01, 0.02, -0.02, 0.04])
POS = frame(A=[NA, NA, 0.5, 0.5, -0.5], B=[NA, NA, NA, 0.5, 0.5])


# ------------------------------------------------------------ positions_from_weights


class TestPositionsFromWeights:
    def test_lag_one_is_the_previous_row(self):
        assert_frame_equal(positions_from_weights(W, 1), POS)

    def test_lag_two_is_two_rows_up(self):
        out = positions_from_weights(W, 2)
        assert_frame_equal(out, frame(A=[NA, NA, NA, 0.5, 0.5], B=[NA, NA, NA, NA, 0.5]))

    def test_shift_is_positional_not_calendar(self):
        # d3 (Jan 3, Friday) -> d4 (Jan 6, Monday): one row, three calendar days.
        w = frame(A=[1.0, 2.0, 3.0, 4.0])
        assert positions_from_weights(w, 1)["A"].loc[D[3]] == 3.0

    def test_nan_weight_is_nan_position(self):
        out = positions_from_weights(frame(A=[1.0, NA, 1.0]), 1)
        assert out["A"].isna().tolist() == [True, False, True]

    def test_shape_dtype_and_input_untouched(self):
        w = frame(B=[1.0, 2.0, 3.0], A=[4.0, 5.0, 6.0])
        before = w.copy()
        out = positions_from_weights(w, 1)
        assert out.index.equals(w.index) and list(out.columns) == ["B", "A"]
        assert all(pd.api.types.is_float_dtype(d) for d in out.dtypes)
        assert_frame_equal(w, before)

    @pytest.mark.parametrize("lag", [0, -1])
    def test_rejects_lag_below_one(self, lag):
        with pytest.raises(ValueError):
            positions_from_weights(W, lag)

    def test_default_config_lag_is_one(self):
        assert load_config().execution.lag == 1


# ------------------------------------------------------------------------ turnover


class TestTurnover:
    def test_hand_computed(self):
        assert_series_equal(turnover(POS), series([0, 0, 0.5, 0.5, 1.0], "turnover"))

    def test_entry_and_exit_are_traded(self):
        # NaN -> 0.4 trades 0.4; 0.4 -> NaN trades 0.4 again.
        out = turnover(frame(A=[NA, 0.4, 0.4, NA]))
        assert out.tolist() == [0.0, 0.4, 0.0, 0.4]

    def test_flip_trades_twice_the_weight(self):
        out = turnover(frame(A=[0.3, -0.3, 0.3]))
        assert out.tolist() == pytest.approx([0.0, 0.6, 0.6])

    def test_first_row_trades_nothing_even_with_a_position(self):
        # A position on the first row has no previous row to be compared with.
        assert turnover(frame(A=[1.0, 1.0])).iloc[0] == 0.0

    def test_sums_across_instruments(self):
        out = turnover(frame(A=[0.0, 0.2], B=[0.0, -0.3], C=[NA, 0.1]))
        assert out.iloc[1] == pytest.approx(0.6)

    def test_name_dtype_index_and_input_untouched(self):
        p = frame(A=[NA, 0.5, 0.5])
        before = p.copy()
        out = turnover(p)
        assert out.name == "turnover" and out.index.equals(p.index)
        assert pd.api.types.is_float_dtype(out)
        assert_frame_equal(p, before)


# --------------------------------------------------------------- transaction_costs


class TestTransactionCosts:
    def test_hand_computed_ten_bps(self):
        to = series([0, 0, 0.5, 0.5, 1.0], "turnover")
        # 10 bps = 0.001 per unit traded.
        assert_series_equal(
            transaction_costs(to, 10.0), series([0, 0, 0.0005, 0.0005, 0.001], "cost")
        )

    def test_default_five_bps_on_a_unit_trade(self):
        bps = load_config().costs.cost_bps
        assert bps == 5.0
        out = transaction_costs(series([1.0], "turnover"), bps)
        assert out.iloc[0] == pytest.approx(0.0005)

    def test_zero_bps_costs_nothing(self):
        assert (transaction_costs(series([1.0, 2.0, 3.0], "turnover"), 0.0) == 0).all()

    def test_basis_point_constant(self):
        assert BASIS_POINT == 1e-4

    def test_name_and_bad_bps_rejected(self):
        out = transaction_costs(series([0.2], "turnover"), 5.0)
        assert out.name == "cost"
        with pytest.raises(ValueError):
            transaction_costs(series([0.2], "turnover"), -1.0)


# ------------------------------------------------------------------- gross_returns


class TestGrossReturns:
    def test_hand_computed(self):
        assert_series_equal(gross_returns(POS, R), series([NA, NA, -0.005, 0.005, 0.015], "gross"))

    def test_nan_before_the_first_position_and_zero_when_flat_after_it(self):
        pos = frame(A=[NA, NA, 0.5, NA, 0.5])
        r = frame(A=[0.01, 0.01, 0.02, 0.03, 0.04])
        # d1, d2: no position yet -> NaN. d3: 0.5 * 0.02 = 0.01. d4: flat -> 0.0 (not NaN:
        # the strategy exists and holds nothing). d5: 0.5 * 0.04 = 0.02.
        assert_series_equal(gross_returns(pos, r), series([NA, NA, 0.01, 0.0, 0.02], "gross"))

    def test_unpriced_position_earns_nothing(self):
        pos = frame(A=[NA, 0.5, 0.5], B=[NA, 0.5, 0.5])
        r = frame(A=[0.01, NA, 0.02], B=[0.01, 0.04, NA])
        # d2: A has no return -> only B counts: 0.5 * 0.04 = 0.02. d3: only A: 0.01.
        assert gross_returns(pos, r).tolist()[1:] == pytest.approx([0.02, 0.01])

    def test_all_unpriced_day_is_zero_not_nan(self):
        pos = frame(A=[NA, 0.5, 0.5])
        r = frame(A=[0.01, NA, 0.02])
        assert gross_returns(pos, r).tolist()[1] == 0.0

    def test_zero_weight_contributes_zero(self):
        pos = frame(A=[NA, 0.0])
        r = frame(A=[0.01, 0.05])
        assert gross_returns(pos, r).tolist() == pytest.approx([NA, 0.0], nan_ok=True)

    def test_name_index_and_inputs_untouched(self):
        p0, r0 = POS.copy(), R.copy()
        out = gross_returns(POS, R)
        assert out.name == "gross" and out.index.equals(R.index)
        assert_frame_equal(POS, p0)
        assert_frame_equal(R, r0)

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError):
            gross_returns(POS.iloc[:4], R)
        with pytest.raises(ValueError):
            gross_returns(POS.rename(columns={"B": "C"}), R)


# ----------------------------------------------------------------------------- run


class TestRun:
    def test_hand_computed_end_to_end(self):
        res = run(W, R, lag=1, cost_bps=10.0)
        assert_frame_equal(res.positions, POS)
        assert_series_equal(res.gross_returns, series([NA, NA, -0.005, 0.005, 0.015], "gross"))
        assert_series_equal(res.costs, series([0, 0, 0.0005, 0.0005, 0.001], "cost"))
        assert_series_equal(res.net_returns, series([NA, NA, -0.0055, 0.0045, 0.014], "net"))
        assert_series_equal(res.turnover, series([0, 0, 0.5, 0.5, 1.0], "turnover"))
        assert_series_equal(res.gross_leverage, series([NA, NA, 0.5, 1.0, 1.0], "gross_leverage"))
        assert res.n_live.tolist() == [0, 0, 1, 2, 2]
        assert res.lag == 1 and res.cost_bps == 10.0

    def test_lag_two_moves_everything_one_row_later(self):
        res = run(W, R, lag=2, cost_bps=10.0)
        # positions d4 [0.5, NA], d5 [0.5, 0.5]: gross d4 0.5 * 0.03 = 0.015,
        # d5 0.5 * 0.01 + 0.5 * 0.04 = 0.025; turnover d4 0.5, d5 0.5.
        assert res.gross_returns.tolist()[:3] == pytest.approx([NA, NA, NA], nan_ok=True)
        assert res.gross_returns.tolist()[3:] == pytest.approx([0.015, 0.025])
        assert res.turnover.tolist() == pytest.approx([0, 0, 0, 0.5, 0.5])

    def test_net_is_gross_minus_cost_wherever_gross_exists(self):
        res = run(W, R, lag=1, cost_bps=25.0)
        life = res.gross_returns.notna()
        diff = res.gross_returns[life] - res.net_returns[life]
        assert_series_equal(diff, res.costs[life], check_names=False)
        assert res.net_returns[~life].isna().all()

    def test_unpriced_counts_position_days_without_a_return(self):
        w = frame(A=[0.5, 0.5, 0.5, 0.5], B=[0.5, 0.5, 0.5, 0.5])
        r = frame(A=[0.01, NA, NA, 0.01], B=[0.01, 0.01, 0.01, NA])
        res = run(w, r, lag=1, cost_bps=5.0)
        # Positions from d2. A unpriced on d2, d3 (2); B on d4 (1).
        assert res.unpriced.tolist() == [2, 1]
        assert res.unpriced.name == "n_unpriced" and list(res.unpriced.index) == ["A", "B"]
        assert pd.api.types.is_integer_dtype(res.unpriced)

    def test_result_fields_names_and_types(self):
        res = run(W, R, lag=1, cost_bps=5.0)
        assert isinstance(res, BacktestResult)
        for s, name in (
            (res.gross_returns, "gross"),
            (res.costs, "cost"),
            (res.net_returns, "net"),
            (res.turnover, "turnover"),
            (res.gross_leverage, "gross_leverage"),
        ):
            assert s.name == name and s.index.equals(R.index)
            assert pd.api.types.is_float_dtype(s)
        assert pd.api.types.is_integer_dtype(res.n_live) and res.n_live.name == "n_live"
        assert list(res.positions.columns) == ["A", "B"]

    def test_inputs_untouched_and_misaligned_rejected(self):
        w0, r0 = W.copy(), R.copy()
        run(W, R, lag=1, cost_bps=5.0)
        assert_frame_equal(W, w0)
        assert_frame_equal(R, r0)
        with pytest.raises(ValueError):
            run(W, R.iloc[:4], lag=1, cost_bps=5.0)
        with pytest.raises(ValueError):
            run(W, R.rename(columns={"A": "C"}), lag=1, cost_bps=5.0)
        with pytest.raises(ValueError):
            run(W, R, lag=0, cost_bps=5.0)


# ----------------------------------------------------------------------- §8 integrity


class TestIntegrityLag:
    """PROJECT_LOG.md §8 "lag": positions at ``t`` equal weights at ``t - lag``."""

    IDX = pd.DatetimeIndex(list(pd.bdate_range("2019-01-01", periods=120)), name="date")

    def weights(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        w = pd.DataFrame(rng.normal(size=(120, 4)), index=self.IDX, columns=list("ABCD"))
        w.iloc[:7, 2] = np.nan  # a late starter
        w.iloc[50:53, 0] = np.nan  # a hole
        return w

    @pytest.mark.parametrize("lag", [1, 2, 5])
    def test_positions_at_t_equal_weights_at_t_minus_lag(self, lag):
        w = self.weights()
        pos = positions_from_weights(w, lag)
        for k in range(lag, len(w)):
            a, b = pos.iloc[k].to_numpy(), w.iloc[k - lag].to_numpy()
            assert np.array_equal(a, b, equal_nan=True), k
        assert pos.iloc[:lag].isna().all().all()

    @pytest.mark.parametrize("lag", [1, 2])
    def test_run_uses_the_configured_lag(self, lag):
        w = self.weights()
        r = pd.DataFrame(0.01, index=self.IDX, columns=w.columns)
        res = run(w, r, lag=lag, cost_bps=0.0)
        assert_frame_equal(res.positions, w.shift(lag))
        # With every return 0.01 the gross return is 0.01 * sum of the lagged weights.
        expected = (w.shift(lag).sum(axis=1) * 0.01).where(
            w.shift(lag).notna().any(axis=1).cummax()
        )
        assert_series_equal(res.gross_returns, expected.rename("gross"))

    def test_lag_zero_is_refused(self):
        with pytest.raises(ValueError):
            run(self.weights(), self.weights() * 0 + 0.01, lag=0, cost_bps=0.0)


class TestIntegrityZeroTurnover:
    """PROJECT_LOG.md §8 "zero-turnover, zero-cost": constant weights produce cost identically
    zero. The one trade a constant book makes is its entry (NaN -> w on the first position
    day), which is charged like any other change; after it, turnover and cost are exactly
    zero on every row, whatever the returns do."""

    IDX = pd.DatetimeIndex(list(pd.bdate_range("2019-01-01", periods=200)), name="date")

    def test_constant_weights_cost_nothing_after_the_entry(self):
        w = pd.DataFrame({"A": 0.4, "B": -0.3, "C": 0.2}, index=self.IDX)
        r = pd.DataFrame(
            np.random.default_rng(2).normal(0, 0.01, (200, 3)), index=self.IDX, columns=w.columns
        )
        res = run(w, r, lag=1, cost_bps=5.0)
        # Entry on row 2: |0.4| + |-0.3| + |0.2| = 0.9 traded, 0.9 * 5e-4 = 0.00045.
        assert res.turnover.iloc[1] == pytest.approx(0.9)
        assert res.costs.iloc[1] == pytest.approx(0.00045)
        assert (res.turnover.iloc[2:] == 0).all()
        assert (res.costs.iloc[2:] == 0).all()
        assert (res.costs.iloc[0] == 0) and (res.turnover.iloc[0] == 0)

    def test_unchanged_positions_trade_nothing_whatever_the_returns(self):
        pos = pd.DataFrame({"A": 1.0, "B": -2.0}, index=self.IDX)
        assert (turnover(pos) == 0).all()

    def test_net_equals_gross_after_the_entry(self):
        w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=self.IDX)
        r = pd.DataFrame(
            np.random.default_rng(3).normal(0, 0.01, (200, 2)), index=self.IDX, columns=w.columns
        )
        res = run(w, r, lag=1, cost_bps=20.0)
        assert_series_equal(res.net_returns.iloc[2:], res.gross_returns.iloc[2:], check_names=False)


class TestIntegrityCostAccounting:
    """PROJECT_LOG.md §8 "cost accounting": a single full flip of one unit weight costs exactly
    ``2 * bps / 1e4``."""

    @pytest.mark.parametrize("bps", [5.0, 10.0, 20.0])
    def test_unit_flip_costs_two_bps_over_ten_thousand(self, bps):
        w = frame(A=[1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        r = frame(A=[0.0] * 6)
        res = run(w, r, lag=1, cost_bps=bps)
        # Positions: d2-d4 +1, d5-d6 -1. The flip is on d5: |-1 - 1| = 2 traded.
        assert res.turnover.loc[D[4]] == pytest.approx(2.0)
        assert res.costs.loc[D[4]] == pytest.approx(2 * bps / 1e4)
        # With zero returns the net return on the flip day is minus the cost.
        assert res.net_returns.loc[D[4]] == pytest.approx(-2 * bps / 1e4)

    def test_total_cost_is_total_turnover_times_the_rate(self):
        res = run(W, R, lag=1, cost_bps=7.0)
        # Turnover 0.5 + 0.5 + 1.0 = 2.0 traded in all; 2.0 * 7e-4 = 0.0014.
        assert res.costs.sum() == pytest.approx(0.0014)
        assert res.costs.sum() == pytest.approx(res.turnover.sum() * 7.0 * BASIS_POINT)

    def test_entry_of_a_unit_weight_costs_one_bps_over_ten_thousand(self):
        res = run(frame(A=[1.0, 1.0, 1.0]), frame(A=[0.0, 0.0, 0.0]), lag=1, cost_bps=5.0)
        assert res.costs.tolist() == pytest.approx([0.0, 5e-4, 0.0])


class TestIntegrityNoLookahead:
    """PROJECT_LOG.md §8 "no lookahead", applied to the positions and the gross and net
    returns of the full pipeline (returns -> signal -> sizing -> engine) through the shared
    harness."""

    IDX = pd.DatetimeIndex(list(pd.bdate_range("2018-01-01", periods=80)), name="date")

    def prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        steps = rng.normal(0.0, 0.02, size=(len(self.IDX), 6))
        p = pd.DataFrame(100 * np.exp(steps.cumsum(axis=0)), index=self.IDX, columns=list("ABCDEF"))
        p.iloc[:15, 1] = np.nan  # a late starter
        p.iloc[30:34, 2] = np.nan  # an unfilled gap
        return p

    @staticmethod
    def pipeline(p: pd.DataFrame, rebalance: str, lag: int) -> BacktestResult:
        r = clean_returns(p, 0.5)[0]
        s = tsmom(clean_price_index(r), lookback=10)
        cfg = SizingConfig(
            sigma_target_asset=0.10,
            sigma_target_portfolio=0.10,
            leverage_cap=3.0,
            vol_window=5,
            vol_estimator="rolling",
        )
        return run(size_positions(s, r, cfg, rebalance).weights, r, lag, 5.0)

    @pytest.mark.parametrize("lag", [1, 2])
    @pytest.mark.parametrize("row", [40, 60])
    def test_positions_have_no_lookahead(self, lag, row):
        assert_no_lookahead(
            lambda p: self.pipeline(p, "daily", lag).positions, self.prices(), self.IDX[row]
        )

    @pytest.mark.parametrize("row", [40, 60])
    def test_gross_and_net_returns_have_no_lookahead(self, row):
        assert_no_lookahead(
            lambda p: self.pipeline(p, "daily", 1).gross_returns, self.prices(), self.IDX[row]
        )
        assert_no_lookahead(
            lambda p: self.pipeline(p, "daily", 1).net_returns, self.prices(), self.IDX[row]
        )

    @pytest.mark.parametrize("row", [40, 60])
    def test_turnover_and_leverage_have_no_lookahead(self, row):
        assert_no_lookahead(
            lambda p: self.pipeline(p, "daily", 1).turnover, self.prices(), self.IDX[row]
        )
        assert_no_lookahead(
            lambda p: self.pipeline(p, "daily", 1).gross_leverage, self.prices(), self.IDX[row]
        )

    # Monthly cuts sit on month-ends (Feb 28 = row 42, Mar 30 = row 64), as in Phase 3: the
    # schedule is calendar knowledge, and a mid-month truncation would create a rebalance
    # day the full sample does not have.
    @pytest.mark.parametrize("row", [42, 64])
    def test_monthly_positions_and_net_returns_have_no_lookahead(self, row):
        assert_no_lookahead(
            lambda p: self.pipeline(p, "monthly", 1).positions, self.prices(), self.IDX[row]
        )
        assert_no_lookahead(
            lambda p: self.pipeline(p, "monthly", 1).net_returns, self.prices(), self.IDX[row]
        )


# -------------------------------------------------------------------- backtest_summary


class TestBacktestSummary:
    def test_hand_computed(self):
        out = backtest_summary(run(W, R, lag=1, cost_bps=10.0))
        # Life: d3-d5 (3 days). Gross [-0.005, 0.005, 0.015]: wealth 0.995 * 1.005 * 1.015
        # = 1.014950... ; ann return = wealth^(252/3) - 1. Mean 0.005, deviations -0.01, 0,
        # 0.01 -> var 0.0002/2 = 1e-4, std 0.01 -> ann vol 0.01 * sqrt(252); Sharpe
        # 0.005 / 0.01 * sqrt(252) = 0.5 * 15.8745 = 7.937. Drawdown: wealth 0.995 < peak 1
        # -> -0.005 is the max drawdown. Net [-0.0055, 0.0045, 0.014].
        # Turnover over the life 0.5, 0.5, 1.0 -> mean 2/3 -> 168 per year. Leverage mean
        # (0.5 + 1 + 1) / 3 = 0.8333; n_live mean (1 + 2 + 2) / 3 = 1.6667. Total cost 0.002.
        wealth = 0.995 * 1.005 * 1.015
        assert out["first_date"] == D[2] and out["last_date"] == D[4] and out["n_days"] == 3
        assert out["ann_return_gross"] == pytest.approx(wealth ** (252 / 3) - 1)
        assert out["ann_vol_gross"] == pytest.approx(0.01 * SQ)
        assert out["sharpe_gross"] == pytest.approx(0.5 * SQ)
        assert out["max_drawdown_gross"] == pytest.approx(-0.005)
        assert out["max_drawdown_net"] == pytest.approx(-0.0055)
        assert out["total_cost"] == pytest.approx(0.002)
        assert out["ann_turnover"] == pytest.approx(2 / 3 * 252)
        assert out["mean_gross_leverage"] == pytest.approx(0.5 + 1 / 3)
        assert out["mean_n_live"] == pytest.approx(5 / 3)
        assert out["n_unpriced"] == 0
        assert out["lag"] == 1 and out["cost_bps"] == 10.0

    def test_keys_and_types(self):
        out = backtest_summary(run(W, R, lag=1, cost_bps=10.0))
        assert list(out) == list(SUMMARY_KEYS)
        for k in ("n_days", "n_unpriced", "lag"):
            assert type(out[k]) is int, k
        for k in SUMMARY_KEYS:
            if k not in ("first_date", "last_date", "n_days", "n_unpriced", "lag"):
                assert type(out[k]) is float, k
        assert isinstance(out["first_date"], pd.Timestamp)

    def test_net_is_below_gross_when_costs_are_paid(self):
        out = backtest_summary(run(W, R, lag=1, cost_bps=10.0))
        assert out["ann_return_net"] < out["ann_return_gross"]
        assert out["sharpe_net"] < out["sharpe_gross"]

    def test_no_position_at_all(self):
        out = backtest_summary(run(W * NA, R, lag=1, cost_bps=10.0))
        assert out["first_date"] is None and out["last_date"] is None and out["n_days"] == 0
        assert np.isnan(out["sharpe_net"]) and np.isnan(out["ann_turnover"])
        assert out["total_cost"] == 0.0 and out["n_unpriced"] == 0

    def test_risk_free_rate_lowers_the_sharpe(self):
        res = run(W, R, lag=1, cost_bps=10.0)
        assert backtest_summary(res, 0.05)["sharpe_gross"] < backtest_summary(res)["sharpe_gross"]


# -------------------------------------------------------------------------- snapshot


def snapshot_backtest() -> tuple[BacktestResult, pd.DataFrame]:
    """(engine result, cleaned returns) on the committed panel, with the default config."""
    cfg = load_config()
    panel = pd.read_csv(ROOT / cfg.data.panel_path, index_col="date", parse_dates=True)
    returns, _ = clean_returns(panel, cfg.data.clip)
    signal = tsmom(clean_price_index(returns), cfg.signal.lookback, cfg.signal.skip)
    sized = size_positions(signal, returns, cfg.sizing, cfg.execution.rebalance)
    return run(sized.weights, returns, cfg.execution.lag, cfg.costs.cost_bps), returns


class TestSnapshot:
    """Real-data checks on the committed snapshot in ``data/``. No network."""

    def test_life_starts_the_row_after_the_first_weights(self):
        res, returns = snapshot_backtest()
        # First weights 2001-03-30 (Phase 3); with lag 1 the first position, and the first
        # gross return, is on the next row, 2001-04-02; 6,410 rows to 2026-09-07.
        first = res.gross_returns.first_valid_index()
        assert first == pd.Timestamp("2001-04-02")
        assert first == returns.index[returns.index.get_loc(pd.Timestamp("2001-03-30")) + 1]
        assert res.gross_returns.notna().sum() == 6410
        assert res.gross_returns.loc[first:].notna().all()

    def test_costs_are_paid_only_on_the_row_after_a_month_end(self):
        res, _ = snapshot_backtest()
        cfg = load_config()
        idx = res.turnover.index
        traded = idx[res.turnover > 0]
        # Monthly held weights change on month-ends only (Phase 3); with lag 1 the positions
        # change, and the cost is paid, exactly one row later. The last month-end is the
        # panel's last row and has no row after it.
        month_ends = rebalance_dates(idx, "monthly")
        after = idx[idx.get_indexer(month_ends)[:-1] + 1]
        assert traded.isin(after).all()
        assert len(traded) > 250
        assert res.costs.sum() == pytest.approx(
            res.turnover.sum() * cfg.costs.cost_bps * BASIS_POINT
        )

    def test_unpriced_days_are_the_live_cattle_gap(self):
        res, _ = snapshot_backtest()
        # LE=F's 20-day gap in January 2006 (2006-01-03 to 2006-01-31) is 20 NaN return
        # rows with a position: the weight decided at the December month-end is held through
        # the gap and earns nothing. The 21st NaN return (2006-02-01, the first day after
        # the gap) is not an unpriced position: the January month-end decided NaN for LE=F
        # (no signal, no vol), so there is no position on it. Every other name is priced on
        # every day it is held.
        assert res.unpriced["LE=F"] == 20
        assert (res.unpriced.drop("LE=F") == 0).all()

    def test_headline_numbers_are_plausible(self):
        res, _ = snapshot_backtest()
        out = backtest_summary(res)
        assert 0.05 <= out["ann_vol_net"] <= 0.15
        assert -0.5 <= out["sharpe_net"] <= 1.5
        assert -0.6 <= out["max_drawdown_net"] <= -0.05
        assert 0 < out["sharpe_gross"] - out["sharpe_net"] < 0.1
        assert 1.0 <= out["ann_turnover"] <= 10.0

    def test_net_returns_have_no_lookahead_on_the_snapshot(self):
        cfg = load_config()
        panel = pd.read_csv(ROOT / cfg.data.panel_path, index_col="date", parse_dates=True)

        def net(p: pd.DataFrame) -> pd.Series:
            r, _ = clean_returns(p, cfg.data.clip)
            s = tsmom(clean_price_index(r), cfg.signal.lookback, cfg.signal.skip)
            w = size_positions(s, r, cfg.sizing, cfg.execution.rebalance).weights
            return run(w, r, cfg.execution.lag, cfg.costs.cost_bps).net_returns

        # A month-end cut (see TestIntegrityNoLookahead on the monthly schedule).
        assert_no_lookahead(net, panel, "2015-12-31", n_trials=1)
