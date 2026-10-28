"""Benchmarks (Phase 4): equal-weight long-only weights, the vol-targeted variant, the
external index on the panel calendar, and the §4.1 two-asset toy case through the engine.

Synthetic frames use business days from 2020-01-01: d1..d12 =
01-01 01-02 01-03 01-06 01-07 01-08 01-09 01-10 01-13 01-14 01-15 01-16.
Every expected value is hand-computed; the arithmetic is in a comment next to it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from src.backtest import run
from src.benchmark import ew_vol_targeted_weights, ew_weights, external_returns
from src.config import ANNUALISATION, SizingConfig, load_config
from src.data import clean_returns, load_benchmark
from src.portfolio import hold, rebalance_dates
from tests.integrity import assert_no_lookahead

ROOT = Path(__file__).resolve().parents[1]
NA = np.nan
SQ = np.sqrt(ANNUALISATION)  # 15.8745...
D = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=12)), name="date")  # d1..d12


def frame(**columns: list[float]) -> pd.DataFrame:
    """Wide frame on d1..dn (n = the common length of the lists); NaN where a value is NA."""
    n = len(next(iter(columns.values())))
    return pd.DataFrame({k: v for k, v in columns.items()}, index=D[:n], dtype=float)


def sizing(window: int = 2, estimator: str = "rolling", cap: float = 3.0) -> SizingConfig:
    return SizingConfig(
        sigma_target_asset=0.10,
        sigma_target_portfolio=0.10,
        leverage_cap=cap,
        vol_window=window,
        vol_estimator=estimator,
    )


def alternating(n: int, step: float) -> list[float]:
    """+step, -step, +step, ... : every 2-row sample std is step * sqrt(2)."""
    return [step if k % 2 == 0 else -step for k in range(n)]


# ---------------------------------------------------------------------- ew_weights


class TestEwWeights:
    def test_daily_is_one_over_the_eligible_count(self):
        r = frame(A=[0.01, 0.02, NA, 0.01], B=[NA, 0.01, 0.01, 0.01], C=[NA, NA, 0.02, 0.03])
        out = ew_weights(r)
        # d1: A alone -> 1. d2: A, B -> 0.5 each. d3: B, C -> 0.5 each, A NaN. d4: all -> 1/3.
        expected = frame(A=[1.0, 0.5, NA, 1 / 3], B=[NA, 0.5, 0.5, 1 / 3], C=[NA, NA, 0.5, 1 / 3])
        assert_frame_equal(out, expected)

    def test_gross_is_exactly_one_on_every_row_with_an_eligible_name(self):
        rng = np.random.default_rng(0)
        r = pd.DataFrame(rng.normal(0, 0.01, (12, 5)), index=D, columns=list("ABCDE"))
        r.iloc[:4, 0] = np.nan
        r.iloc[6, 2] = np.nan
        gross = ew_weights(r).sum(axis=1, min_count=1)
        assert gross.to_numpy() == pytest.approx(1.0)

    def test_row_with_no_eligible_name_is_all_nan(self):
        out = ew_weights(frame(A=[NA, 0.01], B=[NA, 0.02]))
        assert out.iloc[0].isna().all()
        assert out.iloc[1].tolist() == [0.5, 0.5]

    def test_monthly_holds_the_month_end_weights(self):
        # Business days Jan 28 - Feb 6 2020: Jan 28, 29, 30, 31 | Feb 3, 4, 5, 6.
        # Month-ends on this calendar: Jan 31 (row 3) and Feb 6 (row 7, the last row).
        idx = pd.DatetimeIndex(list(pd.bdate_range("2020-01-28", "2020-02-06")), name="date")
        r = pd.DataFrame(
            {"A": [0.01] * 8, "B": [NA, NA, 0.02, 0.02, NA, 0.02, 0.02, 0.02]}, index=idx
        )
        out = ew_weights(r, "monthly")
        # NaN before Jan 31; on Jan 31 both eligible -> 0.5 each, held through Feb 5 even
        # though B has no return on Feb 3; Feb 6 re-decided: both -> 0.5 again.
        assert out.iloc[:3].isna().all().all()
        assert_frame_equal(
            out.iloc[3:], pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=idx[3:])
        )

    def test_monthly_equals_hold_of_daily(self):
        rng = np.random.default_rng(1)
        idx = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=60)), name="date")
        r = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=idx, columns=list("ABC"))
        r.iloc[:10, 1] = np.nan
        expected = hold(ew_weights(r), rebalance_dates(idx, "monthly"))
        assert_frame_equal(ew_weights(r, "monthly"), expected)

    def test_shape_dtype_and_input_untouched(self):
        r = frame(B=[0.01, NA], A=[0.02, 0.03])
        before = r.copy()
        out = ew_weights(r)
        assert out.index.equals(r.index) and list(out.columns) == ["B", "A"]
        assert all(pd.api.types.is_float_dtype(d) for d in out.dtypes)
        assert_frame_equal(r, before)

    def test_rejects_unknown_schedule(self):
        with pytest.raises(ValueError):
            ew_weights(frame(A=[0.01, 0.02]), "weekly")


class TestEwLongOnlyToyCase:
    """PROJECT_LOG.md §9 F4.3 acceptance: the equal-weight long-only benchmark reproduces a
    hand-computed two-asset case through the engine (monthly, lag 1, 5 bps)."""

    # Business days Jan 28 - Feb 6 2020 (8 rows); month-ends Jan 31 (row 3), Feb 6 (row 7).
    IDX = pd.DatetimeIndex(list(pd.bdate_range("2020-01-28", "2020-02-06")), name="date")
    R = pd.DataFrame(
        {"A": [0.01] * 8, "B": [NA, NA, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]},
        index=IDX,
    )

    def test_monthly_two_asset_case(self):
        res = run(ew_weights(self.R, "monthly"), self.R, lag=1, cost_bps=5.0)
        # Weights 0.5 / 0.5 from Jan 31 (row 3); positions from Feb 3 (row 4).
        # Gross on Feb 3-6: 0.5 * 0.01 + 0.5 * 0.02 = 0.015 each day.
        # Turnover Feb 3: 0.5 + 0.5 = 1.0 -> cost 1.0 * 5e-4 = 0.0005; nothing after.
        # Net: Feb 3 0.0145, then 0.015.
        assert res.gross_returns.iloc[:4].isna().all()
        assert res.gross_returns.iloc[4:].to_numpy() == pytest.approx(0.015)
        assert res.turnover.tolist() == pytest.approx([0, 0, 0, 0, 1.0, 0, 0, 0])
        assert res.costs.tolist() == pytest.approx([0, 0, 0, 0, 0.0005, 0, 0, 0])
        assert res.net_returns.iloc[4:].to_numpy() == pytest.approx([0.0145, 0.015, 0.015, 0.015])
        assert res.gross_leverage.iloc[4:].to_numpy() == pytest.approx(1.0)
        assert res.n_live.tolist() == [0, 0, 0, 0, 2, 2, 2, 2]

    def test_daily_two_asset_case(self):
        res = run(ew_weights(self.R, "daily"), self.R, lag=1, cost_bps=5.0)
        # Daily weights: Jan 28-29 A 1.0 (B not eligible); Jan 30 on 0.5 / 0.5. Positions
        # one row later: Jan 29 A 1.0 -> gross 0.01; Jan 30 A 1.0 -> 0.01; Jan 31 0.5 / 0.5
        # -> 0.015 and turnover |0.5 - 1| + |0.5 - 0| = 1.0; Feb on 0.015, no turnover.
        # Entry Jan 29: turnover 1.0.
        assert res.gross_returns.tolist()[1:] == pytest.approx([0.01, 0.01] + [0.015] * 5)
        assert res.turnover.tolist() == pytest.approx([0, 1.0, 0, 1.0, 0, 0, 0, 0])
        assert res.costs.sum() == pytest.approx(2 * 5e-4)

    def test_no_cost_means_net_equals_gross(self):
        res = run(ew_weights(self.R, "monthly"), self.R, lag=1, cost_bps=0.0)
        assert_series_equal(res.net_returns, res.gross_returns, check_names=False)


# ------------------------------------------------------------ ew_vol_targeted_weights


class TestEwVolTargetedWeights:
    def test_single_instrument_hand_computed(self):
        # One eligible name: raw weight 1.0 every row. Book return b(t) = raw(t-1) * r(t) =
        # +-step from d2; rolling-2 std = step * sqrt(2), annualised step * sqrt(2) * sqrt(252),
        # first on d3. Scale 0.1 / that; weight = scale (gross < 3 with step 0.01).
        r = frame(A=alternating(8, 0.01))
        w, binds = ew_vol_targeted_weights(r, sizing(window=2), "daily")
        scale = 0.1 / (0.01 * np.sqrt(2) * SQ)
        assert w["A"].isna().tolist() == [True, True] + [False] * 6
        assert w["A"].iloc[2:].to_numpy() == pytest.approx(scale)
        assert not binds.any() and binds.name == "cap_binds" and binds.dtype == bool

    def test_two_identical_instruments_stay_equal_weight(self):
        # Raw 0.5 / 0.5; book return +-step again (0.5 * step + 0.5 * step), so the same
        # scale applies and each weight is scale / 2: equal-weight within the row, as §4.2
        # asks (no per-asset vol scaling).
        r = frame(A=alternating(8, 0.01), B=alternating(8, 0.01))
        w, _ = ew_vol_targeted_weights(r, sizing(window=2), "daily")
        scale = 0.1 / (0.01 * np.sqrt(2) * SQ)
        assert w.iloc[2:].to_numpy() == pytest.approx(scale / 2)

    def test_unequal_vols_do_not_change_the_split(self):
        # A twice as volatile as B: still 1/2 each (the strategy would give B twice A).
        r = frame(A=alternating(8, 0.02), B=alternating(8, 0.01))
        w, _ = ew_vol_targeted_weights(r, sizing(window=2), "daily")
        ratio = (w["A"] / w["B"]).dropna()
        assert ratio.to_numpy() == pytest.approx(1.0)

    def test_cap_binds_on_a_quiet_book(self):
        # step 0.001: scale 0.1 / (0.001 * sqrt(2) * sqrt(252)) = 4.454 > 3 -> capped to 3.
        r = frame(A=alternating(8, 0.001))
        w, binds = ew_vol_targeted_weights(r, sizing(window=2), "daily")
        assert w["A"].iloc[2:].to_numpy() == pytest.approx(3.0)
        assert binds.tolist() == [False, False] + [True] * 6

    def test_monthly_is_held_from_month_ends(self):
        idx = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=60)), name="date")
        r = pd.DataFrame({"A": alternating(60, 0.01), "B": alternating(60, 0.01)}, index=idx)
        w, binds = ew_vol_targeted_weights(r, sizing(window=2), "monthly")
        daily, _ = ew_vol_targeted_weights(r, sizing(window=2), "daily")
        dates = rebalance_dates(idx, "monthly")
        assert_index_equal(dates, idx[[22, 42, 59]])
        assert w.iloc[:22].isna().all().all()
        assert_frame_equal(w, hold(daily, dates))
        assert binds.loc[idx[:22]].tolist() == [False] * 22

    def test_vol_target_is_hit_on_iid_returns(self):
        rng = np.random.default_rng(11)
        idx = pd.DatetimeIndex(list(pd.bdate_range("2005-01-03", periods=3000)), name="date")
        r = pd.DataFrame(rng.normal(0, 0.20 / SQ, (3000, 6)), index=idx, columns=list("ABCDEF"))
        w, binds = ew_vol_targeted_weights(r, sizing(window=60), "monthly")
        book = (w.shift(1) * r).sum(axis=1, min_count=1).dropna()
        assert float(book.std() * SQ) == pytest.approx(0.10, rel=0.10)
        assert not binds.any()

    def test_shapes_and_inputs_untouched(self):
        r = frame(B=alternating(6, 0.01), A=alternating(6, 0.02))
        before = r.copy()
        w, binds = ew_vol_targeted_weights(r, sizing(window=2), "daily")
        assert w.index.equals(r.index) and list(w.columns) == ["B", "A"]
        assert binds.index.equals(r.index)
        assert_frame_equal(r, before)


# ------------------------------------------------------------------- external_returns


class TestExternalReturns:
    def test_returns_between_kept_prints_on_the_calendar(self):
        # Calendar d1..d6. Prints on d1, d2, d3, d5, d6 (d4 is a holiday for the index) and
        # one extra print on a Saturday (2020-01-04) that is not on the calendar.
        prices = pd.Series(
            [100.0, 102.0, 101.0, 999.0, 103.02, 100.0],
            index=pd.DatetimeIndex([D[0], D[1], D[2], "2020-01-04", D[4], D[5]]),
            name="IDX",
        )
        out = external_returns(prices, D[:6], clip=0.5)
        # d1 NaN (first print); d2 0.02; d3 101/102 - 1 = -0.0098039; d4 NaN (no print);
        # d5 103.02 / 101 - 1 = 0.02 (spans the holiday; the Saturday print is dropped and
        # never seen); d6 100 / 103.02 - 1 = -0.029315.
        expected = pd.Series(
            [NA, 0.02, 101 / 102 - 1, NA, 103.02 / 101 - 1, 100 / 103.02 - 1],
            index=D[:6],
            name="IDX",
        )
        assert_series_equal(out, expected)

    def test_a_missing_print_costs_one_row_not_two(self):
        prices = pd.Series([100.0, 110.0, 121.0], index=D[[0, 1, 3]], name="X")
        out = external_returns(prices, D[:4], clip=0.5)
        assert out.isna().tolist() == [True, False, True, False]
        assert out.iloc[3] == pytest.approx(0.10)

    def test_clipped_like_a_commodity(self):
        prices = pd.Series([100.0, 10.0, 100.0], index=D[:3], name="X")
        out = external_returns(prices, D[:3], clip=0.5)
        # -0.9 clips to -0.5; +9.0 clips to +0.5.
        assert out.tolist()[1:] == [-0.5, 0.5]

    def test_nan_before_first_and_after_last_print(self):
        prices = pd.Series([100.0, 101.0], index=D[[2, 3]], name="X")
        out = external_returns(prices, D[:6], clip=0.5)
        assert out.isna().tolist() == [True, True, True, False, True, True]

    def test_nan_prices_are_ignored(self):
        prices = pd.Series([100.0, NA, 110.0], index=D[:3], name="X")
        out = external_returns(prices, D[:3], clip=0.5)
        assert out.isna().tolist() == [True, True, False]
        assert out.iloc[2] == pytest.approx(0.10)

    def test_name_index_and_input_untouched(self):
        prices = pd.Series([100.0, 101.0, 102.0], index=D[:3], name="^BCOM")
        before = prices.copy()
        out = external_returns(prices, D[:5], clip=0.5)
        assert out.name == "^BCOM" and out.index.equals(D[:5])
        assert_series_equal(prices, before)

    def test_no_lookahead(self):
        rng = np.random.default_rng(3)
        idx = pd.DatetimeIndex(list(pd.bdate_range("2018-01-01", periods=80)), name="date")
        p = pd.DataFrame({"X": 100 * np.exp(rng.normal(0, 0.01, 80).cumsum())}, index=idx)
        assert_no_lookahead(lambda f: external_returns(f["X"], idx, 0.5), p, idx[40])


# -------------------------------------------------------------------------- snapshot


class TestSnapshot:
    """Real-data checks on the committed snapshot in ``data/``. No network."""

    def inputs(self) -> tuple[pd.DataFrame, str, pd.Series]:
        cfg = load_config()
        panel = pd.read_csv(ROOT / cfg.data.panel_path, index_col="date", parse_dates=True)
        returns, _ = clean_returns(panel, cfg.data.clip)
        ticker, prices = load_benchmark(ROOT / cfg.data.benchmark_path)
        return returns, ticker, prices

    def test_ew_basket_starts_on_the_panels_second_day_and_fills_up(self):
        returns, _, _ = self.inputs()
        w = ew_weights(returns)
        # Coffee and cotton have a return from the second row; all 13 names are eligible
        # together from live cattle's first return onward (the basket needs no lookback).
        assert w.notna().any(axis=1).idxmax() == returns.index[1]
        assert w.notna().sum(axis=1).max() == 13
        gross = w.sum(axis=1, min_count=1).dropna()
        assert gross.to_numpy() == pytest.approx(1.0)

    def test_external_index_calendar_counts(self):
        returns, ticker, prices = self.inputs()
        ext = external_returns(prices, returns.index, 0.5)
        assert ticker == "^BCOM" and ext.name == "^BCOM"
        # The index runs 1991-01-02 to 2026-07-17; the panel to 2026-09-07. Inside the
        # overlap, 20 panel days have no BCOM print (NaN here) and 2 BCOM prints fall on
        # days the panel does not have (2016-10-10 and 2016-11-11, dropped and absorbed).
        last = prices.index[-1]
        assert last == pd.Timestamp("2026-07-17")
        inside = returns.index[returns.index <= last]
        assert ext.loc[inside].isna().sum() == 20 + 1  # + the first row, which has no prior print
        assert (returns.index > last).sum() == 36
        assert ext.loc[returns.index > last].isna().all()
        off_calendar = prices.index[
            (prices.index >= returns.index[0]) & ~prices.index.isin(returns.index)
        ]
        assert list(off_calendar) == [pd.Timestamp("2016-10-10"), pd.Timestamp("2016-11-11")]
