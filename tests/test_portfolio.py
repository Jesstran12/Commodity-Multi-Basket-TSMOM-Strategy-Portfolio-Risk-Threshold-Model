"""Position sizing (Phase 3): vol estimators, per-asset scaling, rebalance schedule, portfolio
scaling and leverage cap, the sizing summary, and the §8 rebalance-schedule, vol-targeting
and no-lookahead integrity tests.

Synthetic frames use business days from 2020-01-01: d1..d12 =
01-01 01-02 01-03 01-06 01-07 01-08 01-09 01-10 01-13 01-14 01-15 01-16.
Every expected value is hand-computed; the arithmetic is in a comment next to it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from src.config import ANNUALISATION, SizingConfig, load_config
from src.data import clean_price_index, clean_returns
from src.portfolio import (
    SUMMARY_KEYS,
    SizingResult,
    asset_weights,
    hold,
    leverage_cap,
    portfolio_vol,
    realised_vol,
    rebalance_dates,
    size_positions,
    sizing_summary,
)
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


def sizing(window: int = 2, estimator: str = "rolling", cap: float = 3.0) -> SizingConfig:
    """A sizing config with the default targets and a short window for hand computation."""
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


# ---------------------------------------------------------------------- realised_vol


class TestRealisedVol:
    def test_rolling_hand_computed_window_three(self):
        r = frame(A=[0.01, 0.02, 0.03, 0.02, 0.01])
        out = realised_vol(r, window=3)
        # std(ddof=1) of [0.01, 0.02, 0.03] = 0.01; of [0.02, 0.03, 0.02]: mean 0.02333,
        # squared deviations 1.111e-5 + 4.444e-5 + 1.111e-5 = 6.667e-5, / 2 = 3.333e-5,
        # sqrt = 0.01 / sqrt(3); of [0.03, 0.02, 0.01] = 0.01. Each times sqrt(252).
        expected = pd.Series(
            [NA, NA, 0.01 * SQ, 0.01 / np.sqrt(3) * SQ, 0.01 * SQ], index=D[:5], name="A"
        )
        assert_series_equal(out["A"], expected)

    def test_rolling_uses_sample_std_ddof_one(self):
        r = frame(A=[0.01, 0.03])
        out = realised_vol(r, window=2)
        # std(ddof=1) of [0.01, 0.03] = |0.03 - 0.01| / sqrt(2) = 0.014142; ddof=0 would give 0.01.
        assert out["A"].iloc[1] == pytest.approx(0.02 / np.sqrt(2) * SQ)

    def test_rolling_warm_up_is_window_minus_one_rows(self):
        r = frame(A=alternating(8, 0.01))
        out = realised_vol(r, window=5)
        assert out["A"].isna().tolist() == [True] * 4 + [False] * 4

    def test_rolling_gap_costs_gap_plus_window_minus_one_rows(self):
        r = frame(A=[0.01, -0.01, 0.01, -0.01, NA, -0.01, 0.01, -0.01, 0.01, -0.01])
        out = realised_vol(r, window=3)
        # Warm-up rows 1-2; row 5 is the gap; rows 6-7 still have the gap in their window
        # (rows 4-6 and 5-7); row 8 (rows 6-8) is the first full window after it.
        assert (
            out["A"].isna().tolist() == [True, True, False, False, True, True, True] + [False] * 3
        )

    def test_ewma_hand_computed_span_three(self):
        r = frame(A=[0.01, 0.03, 0.02])
        out = realised_vol(r, window=3, estimator="ewma")
        # span 3 -> alpha = 0.5; adjust=True weights on the third row: 0.25, 0.5, 1 (sum 1.75,
        # sum of squares 1.3125). Weighted mean = (0.0025 + 0.015 + 0.02) / 1.75 = 0.0214286.
        # Weighted squared deviations: 0.25 * 0.0114286^2 + 0.5 * 0.0085714^2 + 0.0014286^2
        # = 3.2653e-5 + 3.6735e-5 + 2.0408e-6 = 7.1429e-5; / 1.75 = 4.0816e-5 (biased);
        # bias correction 1.75^2 / (1.75^2 - 1.3125) = 1.75 -> 7.1429e-5; sqrt = 0.0084515.
        assert out["A"].isna().tolist() == [True, True, False]
        assert out["A"].iloc[2] == pytest.approx(0.0084515 * SQ, rel=1e-4)

    def test_ewma_nan_row_has_no_vol_but_the_next_row_does(self):
        r = frame(A=[0.01, 0.03, 0.02, NA, 0.01])
        out = realised_vol(r, window=3, estimator="ewma")
        # Row 4 has no return -> NaN. Row 5 (ignore_na=False, positional weights 0.0625,
        # 0.125, 0.25, [0.5 on the NaN, dropped], 1; sum 1.4375): mean 0.019375 / 1.4375
        # = 0.013478; weighted squared deviations 5.7608e-5 / 1.4375 = 4.0075e-5; correction
        # 1.4375^2 / (1.4375^2 - 1.0820) = 2.0992 -> 8.4125e-5; sqrt = 0.009172.
        assert out["A"].isna().tolist() == [True, True, False, True, False]
        assert out["A"].iloc[4] == pytest.approx(0.009172 * SQ, rel=1e-3)

    def test_ewma_warm_up_counts_observations_not_rows(self):
        r = frame(A=[NA, NA, 0.01, 0.02, 0.03, 0.01])
        out = realised_vol(r, window=3, estimator="ewma")
        # Three observations are first available on row 5.
        assert out["A"].isna().tolist() == [True, True, True, True, False, False]

    def test_default_config_is_rolling_sixty(self):
        cfg = load_config().sizing
        assert (cfg.vol_window, cfg.vol_estimator) == (60, "rolling")
        idx = pd.DatetimeIndex(list(pd.bdate_range("2019-01-01", periods=100)), name="date")
        r = pd.DataFrame({"A": alternating(100, 0.01)}, index=idx)
        out = realised_vol(r, cfg.vol_window, cfg.vol_estimator)["A"]
        # 60 alternating +-0.01: mean 0, sum of squares 60e-4, / 59 -> std = 0.01 * sqrt(60/59).
        assert out.iloc[:59].isna().all()
        assert out.iloc[59:].to_numpy() == pytest.approx(0.01 * np.sqrt(60 / 59) * SQ)

    def test_shape_index_columns_and_dtype_are_preserved(self):
        r = frame(B=alternating(6, 0.02), A=alternating(6, 0.01))
        out = realised_vol(r, window=2)
        assert out.index.equals(r.index)
        assert list(out.columns) == ["B", "A"]
        assert all(pd.api.types.is_float_dtype(d) for d in out.dtypes)

    def test_all_nan_column_stays_all_nan(self):
        r = frame(A=alternating(5, 0.01), B=[NA] * 5)
        assert realised_vol(r, window=2)["B"].isna().all()
        assert realised_vol(r, window=2, estimator="ewma")["B"].isna().all()

    def test_input_is_not_modified(self):
        r = frame(A=[0.01, 0.03, NA, 0.02])
        before = r.copy()
        realised_vol(r, window=2)
        realised_vol(r, window=2, estimator="ewma")
        assert_frame_equal(r, before)

    def test_rejects_invalid_window_or_estimator(self):
        r = frame(A=alternating(6, 0.01))
        with pytest.raises(ValueError):
            realised_vol(r, window=1)
        with pytest.raises(ValueError):
            realised_vol(r, window=3, estimator="garch")


# --------------------------------------------------------------------- asset_weights


class TestAssetWeights:
    def test_hand_computed_with_a_late_starter(self):
        signal = frame(A=[1, 1, -1], B=[NA, 1, 1])
        vol = frame(A=[0.2, 0.2, 0.2], B=[0.1, 0.1, 0.1])
        raw, n_live = asset_weights(signal, vol, sigma_target=0.1)
        # d1: A alone, N_live 1: 1 * (0.1 / 0.2) / 1 = 0.5. d2: both, N_live 2: A 0.5 / 2
        # = 0.25, B 1 * (0.1 / 0.1) / 2 = 0.5. d3: A -0.25, B 0.5.
        assert_frame_equal(raw, frame(A=[0.5, 0.25, -0.25], B=[NA, 0.5, 0.5]))
        assert n_live.tolist() == [1, 2, 2]

    def test_flat_signal_is_live_with_weight_zero(self):
        raw, n_live = asset_weights(frame(A=[0.0], B=[1.0]), frame(A=[0.2], B=[0.2]), 0.1)
        # Both live -> N_live 2; A 0 * 0.5 / 2 = 0; B 0.5 / 2 = 0.25.
        assert raw.iloc[0].tolist() == [0.0, 0.25]
        assert n_live.tolist() == [2]

    @pytest.mark.parametrize("bad_vol", [NA, 0.0, np.inf])
    def test_signal_without_a_usable_vol_is_not_live(self, bad_vol):
        raw, n_live = asset_weights(frame(A=[1.0], B=[1.0]), frame(A=[bad_vol], B=[0.2]), 0.1)
        # Only B is live: N_live 1, B = 1 * 0.5 / 1 = 0.5; A NaN.
        assert np.isnan(raw.loc[D[0], "A"])
        assert raw.loc[D[0], "B"] == 0.5
        assert n_live.tolist() == [1]

    def test_ragged_panel_n_live_counts_who_is_live(self):
        signal = frame(A=[1, 1, 1, 1, 1], B=[NA, NA, -1, -1, -1], C=[NA, NA, NA, NA, 1])
        vol = frame(A=[0.1] * 5, B=[0.1] * 5, C=[0.1] * 5)
        raw, n_live = asset_weights(signal, vol, 0.1)
        assert n_live.tolist() == [1, 1, 2, 2, 3]
        # Each live weight is sign * 1 / N_live.
        assert raw["A"].tolist() == [1.0, 1.0, 0.5, 0.5, pytest.approx(1 / 3)]
        assert raw["B"].tolist()[2:] == [-0.5, -0.5, pytest.approx(-1 / 3)]
        assert raw["C"].iloc[4] == pytest.approx(1 / 3)

    def test_row_with_no_live_instrument(self):
        raw, n_live = asset_weights(frame(A=[NA, 1.0]), frame(A=[0.2, 0.2]), 0.1)
        assert np.isnan(raw.loc[D[0], "A"])
        assert n_live.tolist() == [0, 1]

    def test_n_live_is_an_integer_series_named_n_live(self):
        _, n_live = asset_weights(frame(A=[1, NA, 1]), frame(A=[0.2, 0.2, 0.2]), 0.1)
        assert pd.api.types.is_integer_dtype(n_live)
        assert n_live.name == "n_live"
        assert n_live.index.equals(D[:3])

    def test_shape_and_dtype_and_inputs_untouched(self):
        signal = frame(B=[1, -1, NA], A=[1, 1, 1])
        vol = frame(B=[0.1, 0.2, 0.3], A=[0.3, NA, 0.1])
        s0, v0 = signal.copy(), vol.copy()
        raw, _ = asset_weights(signal, vol, 0.1)
        assert list(raw.columns) == ["B", "A"]
        assert raw.index.equals(signal.index)
        assert all(pd.api.types.is_float_dtype(d) for d in raw.dtypes)
        assert_frame_equal(signal, s0)
        assert_frame_equal(vol, v0)

    def test_rejects_bad_target_or_misaligned_vol(self):
        signal, vol = frame(A=[1.0, 1.0]), frame(A=[0.2, 0.2])
        with pytest.raises(ValueError):
            asset_weights(signal, vol, 0.0)
        with pytest.raises(ValueError):
            asset_weights(signal, vol.iloc[:1], 0.1)
        with pytest.raises(ValueError):
            asset_weights(signal, vol.rename(columns={"A": "B"}), 0.1)


# ------------------------------------------------------------------- rebalance_dates


# Jan 2 to Mar 10 2020 business days with Jan 31 removed: Jan's last trading day is Jan 30.
M = pd.DatetimeIndex(
    [d for d in pd.bdate_range("2020-01-02", "2020-03-10") if d != pd.Timestamp("2020-01-31")],
    name="date",
)


class TestRebalanceDates:
    def test_monthly_is_the_last_trading_day_of_each_month(self):
        out = rebalance_dates(M, "monthly")
        expected = pd.DatetimeIndex(["2020-01-30", "2020-02-28", "2020-03-10"], name="date")
        assert_index_equal(out, expected)

    def test_partial_final_month_contributes_its_last_row(self):
        assert rebalance_dates(M, "monthly")[-1] == M[-1]

    def test_daily_is_every_row(self):
        assert_index_equal(rebalance_dates(M, "daily"), M)

    def test_multi_year_count_is_twelve_per_year(self):
        idx = pd.DatetimeIndex(list(pd.bdate_range("2018-01-01", "2020-12-31")), name="date")
        out = rebalance_dates(idx, "monthly")
        assert len(out) == 36
        assert out.isin(idx).all()
        assert out.is_monotonic_increasing
        # Each one is the largest index date of its (year, month).
        for d in out:
            assert d == idx[(idx.year == d.year) & (idx.month == d.month)].max()

    def test_single_row(self):
        assert_index_equal(rebalance_dates(M[:1], "monthly"), M[:1])

    def test_default_config_is_monthly(self):
        assert load_config().execution.rebalance == "monthly"

    def test_rejects_unknown_mode_or_bad_index(self):
        with pytest.raises(ValueError):
            rebalance_dates(M, "weekly")
        with pytest.raises(ValueError):
            rebalance_dates(M[::-1], "monthly")
        with pytest.raises(ValueError):
            rebalance_dates(M[[0, 0, 1]], "monthly")


# ------------------------------------------------------------------------------- hold


class TestHold:
    def test_holds_each_decision_until_the_next(self):
        v = frame(A=[1, 2, 3, 4, 5, 6], B=[10, 20, 30, 40, 50, 60])
        out = hold(v, D[[0, 3]])
        assert_frame_equal(out, frame(A=[1, 1, 1, 4, 4, 4], B=[10, 10, 10, 40, 40, 40]))

    def test_nan_before_the_first_date(self):
        out = hold(frame(A=[1, 2, 3, 4, 5, 6]), D[[2]])
        assert_frame_equal(out, frame(A=[NA, NA, 3, 3, 3, 3]))

    def test_nan_decided_on_a_rebalance_day_is_held_not_filled(self):
        out = hold(frame(A=[1, NA, 3, 4, 5, 6]), D[[0, 1, 3]])
        # d2's decision is NaN and stays NaN on d3; it is not filled with d1's 1.
        assert_frame_equal(out, frame(A=[1, NA, NA, 4, 4, 4]))

    def test_every_day_is_the_identity(self):
        v = frame(A=[1, NA, 3, 4], B=[5, 6, NA, 8])
        assert_frame_equal(hold(v, D[:4]), v)

    def test_series_input(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=D[:4], name="x")
        out = hold(s, D[[1, 3]])
        assert_series_equal(out, pd.Series([NA, 2.0, 2.0, 4.0], index=D[:4], name="x"))

    def test_bool_series_round_trips_through_float(self):
        b = pd.Series([True, False, True, False], index=D[:4])
        out = hold(b.astype(float), D[[0, 2]]).eq(1.0)
        assert out.tolist() == [True, True, True, True]
        assert out.dtype == bool

    def test_index_columns_and_input_preserved(self):
        v = frame(B=[1, 2, 3], A=[4, 5, 6])
        before = v.copy()
        out = hold(v, D[[0, 2]])
        assert out.index.equals(v.index)
        assert list(out.columns) == ["B", "A"]
        assert_frame_equal(v, before)

    def test_rejects_dates_outside_the_index(self):
        with pytest.raises(ValueError):
            hold(frame(A=[1, 2, 3]), pd.DatetimeIndex(["2020-01-01", "2021-01-01"]))


# ------------------------------------------------------------------- portfolio_vol


class TestPortfolioVol:
    def test_hand_computed_lagged_book(self):
        raw = frame(A=[0.5] * 5, B=[NA, NA, 0.5, 0.5, 0.5])
        r = frame(A=[0.01, 0.02, 0.03, 0.02, 0.01], B=[NA, NA, 0.01, 0.04, 0.02])
        out = portfolio_vol(raw, r, window=2)
        # Book return b(t) = sum_i raw_i(t-1) r_i(t): d1 NaN (no weight the row before);
        # d2 0.5 * 0.02 = 0.01 (B has no weight on d1); d3 0.5 * 0.03 = 0.015 (B's d2 weight
        # NaN); d4 0.5 * 0.02 + 0.5 * 0.04 = 0.03; d5 0.5 * 0.01 + 0.5 * 0.02 = 0.015.
        # Rolling 2 std: d2 NaN (d1 is NaN); d3 |0.015 - 0.01| / sqrt2 = 0.0035355;
        # d4 |0.03 - 0.015| / sqrt2 = 0.010607; d5 the same. Each times sqrt(252).
        expected = pd.Series(
            [NA, NA, 0.005 / np.sqrt(2) * SQ, 0.015 / np.sqrt(2) * SQ, 0.015 / np.sqrt(2) * SQ],
            index=D[:5],
            name="portfolio_vol",
        )
        assert_series_equal(out, expected)

    def test_row_with_no_defined_term_is_nan_not_zero(self):
        raw = frame(A=[0.5, 0.5, 0.5, 0.5])
        r = frame(A=[0.01, NA, 0.01, -0.01])
        out = portfolio_vol(raw, r, window=2)
        # Book: d1 NaN, d2 NaN (return missing), d3 0.005, d4 -0.005 -> vol only on d4.
        assert out.isna().tolist() == [True, True, True, False]
        assert out.iloc[3] == pytest.approx(0.01 / np.sqrt(2) * SQ)

    def test_estimator_is_passed_through(self):
        raw = frame(A=[0.5] * 6)
        r = frame(A=[0.01, 0.03, -0.02, 0.01, 0.02, -0.01])
        book = (raw.shift(1) * r).sum(axis=1, min_count=1).to_frame("portfolio_vol")
        out = portfolio_vol(raw, r, window=3, estimator="ewma")
        assert_series_equal(out, realised_vol(book, 3, "ewma")["portfolio_vol"])

    def test_name_index_and_inputs_untouched(self):
        raw, r = frame(A=[0.5] * 4), frame(A=[0.01, 0.02, -0.01, 0.03])
        raw0, r0 = raw.copy(), r.copy()
        out = portfolio_vol(raw, r, window=2)
        assert out.name == "portfolio_vol"
        assert out.index.equals(r.index)
        assert_frame_equal(raw, raw0)
        assert_frame_equal(r, r0)

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError):
            portfolio_vol(frame(A=[0.5] * 3), frame(A=[0.01] * 4), window=2)
        with pytest.raises(ValueError):
            portfolio_vol(frame(A=[0.5] * 3), frame(B=[0.01] * 3), window=2)


# -------------------------------------------------------------------- leverage_cap


class TestLeverageCap:
    def test_hand_computed(self):
        w = frame(A=[1, 2, -2], B=[1, 2, 2])
        capped, binds = leverage_cap(w, cap=3.0)
        # Gross 2, 4, 4: d1 untouched; d2 and d3 scaled by 3/4 -> 1.5 each, signs kept.
        assert_frame_equal(capped, frame(A=[1, 1.5, -1.5], B=[1, 1.5, 1.5]))
        assert binds.tolist() == [False, True, True]

    def test_gross_after_capping_is_exactly_the_cap(self):
        w = frame(A=[4.0, 0.25], B=[-6.0, 0.25], C=[2.0, NA])
        capped, binds = leverage_cap(w, cap=3.0)
        assert capped.abs().sum(axis=1).iloc[0] == pytest.approx(3.0)
        assert capped.iloc[1].tolist()[:2] == [0.25, 0.25]
        assert binds.tolist() == [True, False]

    def test_nan_cells_are_kept_and_skipped_in_the_gross(self):
        capped, binds = leverage_cap(frame(A=[NA, NA], B=[4.0, NA]), cap=3.0)
        # d1: gross 4 from B alone -> B 3, A stays NaN. d2: no weight at all -> untouched, False.
        assert np.isnan(capped.loc[D[0], "A"]) and capped.loc[D[0], "B"] == 3.0
        assert capped.iloc[1].isna().all()
        assert binds.tolist() == [True, False]

    def test_exactly_at_the_cap_does_not_bind(self):
        capped, binds = leverage_cap(frame(A=[1.5], B=[-1.5]), cap=3.0)
        assert_frame_equal(capped, frame(A=[1.5], B=[-1.5]))
        assert binds.tolist() == [False]

    def test_binds_is_a_bool_series_named_cap_binds(self):
        _, binds = leverage_cap(frame(A=[1.0, 5.0]), cap=3.0)
        assert binds.dtype == bool
        assert binds.name == "cap_binds"
        assert binds.index.equals(D[:2])

    def test_default_cap_is_three(self):
        assert load_config().sizing.leverage_cap == 3.0

    def test_input_untouched_and_bad_cap_rejected(self):
        w = frame(A=[4.0])
        before = w.copy()
        leverage_cap(w, cap=3.0)
        assert_frame_equal(w, before)
        with pytest.raises(ValueError):
            leverage_cap(w, cap=0.0)


# ------------------------------------------------------------------ size_positions


class TestSizePositions:
    """One instrument, returns alternating +-step, window 2, so every stage is a constant.

    vol = step * sqrt(2) * sqrt(252) from d2 (window 2). raw = 0.1 / vol from d2 (signal +1,
    N_live 1). Book b(t) = raw * (+-step) from d3; its rolling-2 std is raw * step * sqrt(2),
    annualised = 0.1 exactly, so the portfolio scale is 1 from d4 and the daily weight is
    raw from d4. With step 0.01: raw = 0.1 / 0.22450 = 0.44543 (below the cap). With step
    0.001: raw = 4.4543, capped to 3.0.
    """

    @pytest.mark.parametrize(
        ("step", "weight", "binds"),
        [(0.01, 0.1 / (0.01 * np.sqrt(2) * SQ), False), (0.001, 3.0, True)],
    )
    def test_hand_computed_daily(self, step, weight, binds):
        r = frame(A=alternating(8, step))
        s = frame(A=[1.0] * 8)
        res = size_positions(s, r, sizing(window=2), rebalance="daily")
        assert res.vol["A"].isna().tolist() == [True] + [False] * 7
        assert res.vol["A"].iloc[1] == pytest.approx(step * np.sqrt(2) * SQ)
        assert res.raw_weights["A"].iloc[1:].to_numpy() == pytest.approx(
            0.1 / (step * np.sqrt(2) * SQ)
        )
        assert res.portfolio_vol.isna().tolist() == [True] * 3 + [False] * 5
        assert res.portfolio_vol.iloc[3:].to_numpy() == pytest.approx(0.1)
        assert res.daily_weights["A"].isna().tolist() == [True] * 3 + [False] * 5
        assert res.daily_weights["A"].iloc[3:].to_numpy() == pytest.approx(weight)
        assert_frame_equal(res.weights, res.daily_weights)
        assert res.cap_binds.tolist() == [False] * 3 + [binds] * 5
        assert res.n_live.tolist() == [0] + [1] * 7
        assert_index_equal(res.rebalance_dates, D[:8])

    def test_monthly_holds_the_month_end_decision(self):
        # 60 business days from 2020-01-01: month-ends on the calendar are Jan 31 (row 22),
        # Feb 28 (row 42) and Mar 25 (row 59, the last row). Signal flips sign on row 30.
        idx = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=60)), name="date")
        r = pd.DataFrame({"A": alternating(60, 0.01)}, index=idx)
        s = pd.DataFrame({"A": [1.0] * 30 + [-1.0] * 30}, index=idx)
        res = size_positions(s, r, sizing(window=2), rebalance="monthly")
        w = 0.1 / (0.01 * np.sqrt(2) * SQ)
        assert_index_equal(res.rebalance_dates, idx[[22, 42, 59]])
        # NaN until the first rebalance day (Jan 31); +w through February; the daily
        # decision turns to -w on row 30 but the held weight only flips on Feb 28.
        assert res.weights["A"].iloc[:22].isna().all()
        assert res.weights["A"].iloc[22:42].to_numpy() == pytest.approx(w)
        assert res.weights["A"].iloc[42:].to_numpy() == pytest.approx(-w)
        assert res.daily_weights["A"].iloc[30] == pytest.approx(-w)

    def test_two_identical_instruments_split_the_book(self):
        r = frame(A=alternating(8, 0.01), B=alternating(8, 0.01))
        s = frame(A=[1.0] * 8, B=[1.0] * 8)
        res = size_positions(s, r, sizing(window=2), rebalance="daily")
        # Each raw weight is half of the single-instrument one; the book return is the same,
        # so the portfolio vol is 0.1 again and the weights are 0.44543 / 2 each.
        w = 0.1 / (0.01 * np.sqrt(2) * SQ) / 2
        assert res.n_live.iloc[1:].tolist() == [2] * 7
        assert res.weights[["A", "B"]].iloc[3:].to_numpy() == pytest.approx(w)

    def test_perfectly_hedged_book_has_no_finite_scale(self):
        r = frame(A=alternating(8, 0.01), B=alternating(8, -0.01))
        s = frame(A=[1.0] * 8, B=[1.0] * 8)
        res = size_positions(s, r, sizing(window=2), rebalance="daily")
        # Book return is identically zero, portfolio vol 0: no weight can be sized.
        assert (res.portfolio_vol.dropna() == 0).all()
        assert res.weights.isna().all().all()
        assert not res.cap_binds.any()

    def test_no_weight_before_the_first_live_rebalance_day(self):
        r = frame(A=alternating(8, 0.01))
        s = frame(A=[NA, NA, NA, 1.0, 1.0, 1.0, 1.0, 1.0])
        res = size_positions(s, r, sizing(window=2), rebalance="daily")
        # Live from d4; book from d5; portfolio vol from d6; first weight d6.
        assert res.n_live.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
        assert res.weights["A"].first_valid_index() == D[5]

    def test_result_fields_shapes_and_types(self):
        r = frame(B=alternating(8, 0.01), A=alternating(8, 0.02))
        s = frame(B=[1.0] * 8, A=[-1.0] * 8)
        res = size_positions(s, r, sizing(window=3), rebalance="monthly")
        assert isinstance(res, SizingResult)
        for f in (res.vol, res.raw_weights, res.daily_weights, res.weights):
            assert f.index.equals(s.index) and list(f.columns) == ["B", "A"]
            assert all(pd.api.types.is_float_dtype(d) for d in f.dtypes)
        assert pd.api.types.is_integer_dtype(res.n_live) and res.n_live.name == "n_live"
        assert res.portfolio_vol.name == "portfolio_vol"
        assert res.cap_binds.dtype == bool and res.cap_binds.name == "cap_binds"
        assert_index_equal(res.rebalance_dates, rebalance_dates(s.index, "monthly"))

    def test_weights_equal_the_daily_decision_on_rebalance_days(self):
        r = frame(A=alternating(8, 0.01))
        s = frame(A=[1.0] * 8)
        res = size_positions(s, r, sizing(window=2), rebalance="daily")
        on = res.rebalance_dates
        assert_frame_equal(res.weights.loc[on], res.daily_weights.loc[on])

    def test_inputs_untouched_and_misaligned_rejected(self):
        r, s = frame(A=alternating(8, 0.01)), frame(A=[1.0] * 8)
        r0, s0 = r.copy(), s.copy()
        size_positions(s, r, sizing(window=2), rebalance="daily")
        assert_frame_equal(r, r0)
        assert_frame_equal(s, s0)
        with pytest.raises(ValueError):
            size_positions(s, r.iloc[:7], sizing(window=2), rebalance="daily")
        with pytest.raises(ValueError):
            size_positions(s, r.rename(columns={"A": "B"}), sizing(window=2), rebalance="daily")


# ----------------------------------------------------------------------- §8 integrity


def iid_returns(n_assets: int, n_days: int, vol: float, seed: int) -> pd.DataFrame:
    """i.i.d. normal daily returns with the given annualised vol, on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(list(pd.bdate_range("2005-01-03", periods=n_days)), name="date")
    cols = [f"A{k}" for k in range(n_assets)]
    return pd.DataFrame(rng.normal(0.0, vol / SQ, size=(n_days, n_assets)), index=idx, columns=cols)


def realised(book: pd.Series) -> float:
    """Annualised sample vol of a daily return series, NaNs dropped."""
    return float(book.dropna().std() * SQ)


class TestIntegrityVolTargeting:
    """PROJECT_LOG.md §8 "vol targeting": on i.i.d. returns with known vol, realised portfolio
    vol is within tolerance of the target. Positions lag weights by one row (§2.3)."""

    @pytest.mark.parametrize("estimator", ["rolling", "ewma"])
    @pytest.mark.parametrize("rebalance", ["daily", "monthly"])
    def test_portfolio_vol_lands_on_the_target(self, estimator, rebalance):
        r = iid_returns(6, 3000, vol=0.20, seed=11)
        s = pd.DataFrame(1.0, index=r.index, columns=r.columns)
        cfg = sizing(window=60, estimator=estimator)
        res = size_positions(s, r, cfg, rebalance)
        book = (res.weights.shift(1) * r).sum(axis=1, min_count=1)
        # A trailing 60-day estimate is noisy, so the realised vol of a targeted series sits
        # a little above the target; 10% either side is the tolerance.
        assert realised(book) == pytest.approx(cfg.sigma_target_portfolio, rel=0.10)
        assert not res.cap_binds.any()

    def test_per_asset_stage_lands_on_its_target(self):
        r = iid_returns(3, 3000, vol=0.20, seed=5)
        s = pd.DataFrame(1.0, index=r.index, columns=r.columns)
        vol = realised_vol(r, 60)
        raw, n_live = asset_weights(s, vol, 0.10)
        # Undo the 1 / N_live so each column is the instrument scaled to 10% on its own.
        scaled = raw.mul(n_live, axis=0)
        for c in r.columns:
            assert realised(scaled[c].shift(1) * r[c]) == pytest.approx(0.10, rel=0.10)

    def test_higher_vol_instrument_gets_the_smaller_weight(self):
        r = iid_returns(2, 500, vol=0.20, seed=2)
        r["A1"] = r["A1"] * 2  # 40% vol
        s = pd.DataFrame(1.0, index=r.index, columns=r.columns)
        raw, _ = asset_weights(s, realised_vol(r, 60), 0.10)
        ratio = (raw["A0"] / raw["A1"]).dropna()
        assert (ratio > 1).all()
        assert ratio.median() == pytest.approx(2.0, rel=0.05)


class TestIntegrityRebalanceSchedule:
    """PROJECT_LOG.md §8 "rebalance schedule": weights change only on rebalance dates in
    monthly mode."""

    def pipeline(self, rebalance: str) -> SizingResult:
        r = iid_returns(4, 400, vol=0.20, seed=3)
        rng = np.random.default_rng(4)
        s = pd.DataFrame(rng.choice([-1.0, 1.0], size=r.shape), index=r.index, columns=r.columns)
        s.iloc[:30, 1] = np.nan  # a late starter
        return size_positions(s, r, sizing(window=20), rebalance)

    @staticmethod
    def change_days(w: pd.DataFrame) -> pd.DatetimeIndex:
        prev = w.shift(1)
        changed = (w != prev) & ~(w.isna() & prev.isna())
        return w.index[changed.any(axis=1)]

    def test_monthly_weights_change_only_on_month_ends(self):
        res = self.pipeline("monthly")
        days = self.change_days(res.weights)
        assert len(days) > 5
        assert days.isin(res.rebalance_dates).all()

    def test_daily_weights_change_on_most_days(self):
        res = self.pipeline("daily")
        assert len(self.change_days(res.weights)) > 300

    def test_monthly_matches_the_daily_decision_on_month_ends(self):
        res = self.pipeline("monthly")
        daily = self.pipeline("daily")
        on = res.rebalance_dates
        assert_frame_equal(res.weights.loc[on], daily.weights.loc[on])
        assert res.cap_binds.loc[on].tolist() == daily.cap_binds.loc[on].tolist()


class TestIntegrityNoLookahead:
    """PROJECT_LOG.md §8 "no lookahead", applied to the vol estimate, the raw weights, the
    portfolio vol and the final weights through the shared harness."""

    IDX = pd.DatetimeIndex(list(pd.bdate_range("2018-01-01", periods=80)), name="date")

    def prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        steps = rng.normal(0.0, 0.02, size=(len(self.IDX), 6))
        p = pd.DataFrame(100 * np.exp(steps.cumsum(axis=0)), index=self.IDX, columns=list("ABCDEF"))
        p.iloc[:15, 1] = np.nan  # a late starter
        p.iloc[30:34, 2] = np.nan  # an unfilled gap
        return p

    @staticmethod
    def returns(p: pd.DataFrame) -> pd.DataFrame:
        return clean_returns(p, 0.5)[0]

    def sized(self, p: pd.DataFrame, estimator: str, rebalance: str) -> SizingResult:
        r = self.returns(p)
        s = tsmom(clean_price_index(r), lookback=10)
        return size_positions(s, r, sizing(window=5, estimator=estimator), rebalance)

    @pytest.mark.parametrize("estimator", ["rolling", "ewma"])
    @pytest.mark.parametrize("row", [20, 40, 60])
    def test_vol_has_no_lookahead(self, estimator, row):
        assert_no_lookahead(
            lambda p: realised_vol(self.returns(p), 5, estimator), self.prices(), self.IDX[row]
        )

    @pytest.mark.parametrize("row", [20, 40, 60])
    def test_raw_weights_have_no_lookahead(self, row):
        assert_no_lookahead(
            lambda p: self.sized(p, "rolling", "daily").raw_weights, self.prices(), self.IDX[row]
        )

    @pytest.mark.parametrize("row", [40, 60])
    def test_n_live_has_no_lookahead(self, row):
        assert_no_lookahead(
            lambda p: self.sized(p, "rolling", "daily").n_live, self.prices(), self.IDX[row]
        )

    @pytest.mark.parametrize("estimator", ["rolling", "ewma"])
    @pytest.mark.parametrize("row", [40, 60])
    def test_portfolio_vol_has_no_lookahead(self, estimator, row):
        assert_no_lookahead(
            lambda p: self.sized(p, estimator, "daily").portfolio_vol, self.prices(), self.IDX[row]
        )

    @pytest.mark.parametrize("estimator", ["rolling", "ewma"])
    @pytest.mark.parametrize("row", [40, 60])
    def test_daily_weights_have_no_lookahead(self, estimator, row):
        assert_no_lookahead(
            lambda p: self.sized(p, estimator, "daily").weights, self.prices(), self.IDX[row]
        )

    # Monthly cuts sit on month-ends (Feb 28 = row 42, Mar 30 = row 64): the schedule is
    # calendar knowledge, and a sample cut mid-month would make the cut-off day the last
    # row of its month, i.e. a rebalance day that the full sample does not have.
    @pytest.mark.parametrize("row", [42, 64])
    def test_monthly_weights_have_no_lookahead(self, row):
        assert_no_lookahead(
            lambda p: self.sized(p, "rolling", "monthly").weights, self.prices(), self.IDX[row]
        )
        assert_no_lookahead(
            lambda p: self.sized(p, "rolling", "monthly").cap_binds.astype(float),
            self.prices(),
            self.IDX[row],
        )


# ----------------------------------------------------------------------- sizing_summary


class TestSizingSummary:
    def result(self) -> SizingResult:
        idx = D[:6]
        weights = frame(A=[NA, NA, 0.5, 0.5, -1.0, -1.0], B=[NA, NA, NA, NA, 2.5, 2.5])
        raw = frame(A=[NA, 0.3, 0.3, NA, 0.3, 0.3], B=[NA, NA, NA, NA, 0.3, 0.3])
        return SizingResult(
            vol=frame(A=[0.2] * 6, B=[0.2] * 6),
            raw_weights=raw,
            n_live=pd.Series([0, 1, 1, 0, 2, 2], index=idx, name="n_live"),
            portfolio_vol=pd.Series([NA, NA, 0.1, 0.1, 0.1, 0.1], index=idx, name="portfolio_vol"),
            daily_weights=weights,
            weights=weights,
            cap_binds=pd.Series([False] * 4 + [True] * 2, index=idx, name="cap_binds"),
            rebalance_dates=D[[0, 2, 4]],
        )

    def test_hand_computed(self):
        out = sizing_summary(self.result())
        # Days with a position: d3-d6 (4), first d3. Gross: 0.5, 0.5, 3.5, 3.5 -> mean 2.0,
        # max 3.5. n_live on those days 1, 0, 2, 2 -> 1.25. Cap binds on 2 of 4 days.
        # Unsizable: A's span is d2-d6 with d4 undefined (1); B's span d5-d6 is complete.
        assert out["n_rebalance_dates"] == 3
        assert out["first_weight_date"] == D[2]
        assert out["n_days_with_position"] == 4
        assert out["mean_n_live"] == pytest.approx(1.25)
        assert out["mean_gross_leverage"] == pytest.approx(2.0)
        assert out["max_gross_leverage"] == pytest.approx(3.5)
        assert out["n_cap_bind_days"] == 2
        assert out["cap_bind_fraction"] == pytest.approx(0.5)
        assert out["n_unsizable"] == 1

    def test_keys_and_types(self):
        out = sizing_summary(self.result())
        assert list(out) == list(SUMMARY_KEYS)
        for k in ("n_rebalance_dates", "n_days_with_position", "n_cap_bind_days", "n_unsizable"):
            assert type(out[k]) is int, k
        for k in ("mean_n_live", "mean_gross_leverage", "max_gross_leverage", "cap_bind_fraction"):
            assert type(out[k]) is float, k
        assert isinstance(out["first_weight_date"], pd.Timestamp)

    def test_no_position_at_all(self):
        res = self.result()
        empty = res.weights * NA
        out = sizing_summary(
            SizingResult(
                res.vol,
                res.raw_weights,
                res.n_live,
                res.portfolio_vol,
                empty,
                empty,
                res.cap_binds & False,
                res.rebalance_dates,
            )
        )
        assert out["first_weight_date"] is None
        assert out["n_days_with_position"] == 0
        assert out["n_cap_bind_days"] == 0
        assert all(
            np.isnan(out[k]) for k in ("mean_n_live", "mean_gross_leverage", "cap_bind_fraction")
        )
        assert out["n_unsizable"] == 1

    def test_round_trip_from_size_positions(self):
        r = frame(A=alternating(8, 0.001))
        s = frame(A=[1.0] * 8)
        out = sizing_summary(size_positions(s, r, sizing(window=2), rebalance="daily"))
        # From TestSizePositions: weights 3.0 (capped) on d4-d8, cap binds on all five.
        assert out["n_rebalance_dates"] == 8
        assert out["first_weight_date"] == D[3]
        assert out["n_days_with_position"] == 5
        assert out["mean_gross_leverage"] == pytest.approx(3.0)
        assert out["cap_bind_fraction"] == 1.0
        assert out["n_unsizable"] == 0


# -------------------------------------------------------------------------- snapshot


def snapshot_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(signal, returns) on the committed panel, with the default config."""
    cfg = load_config()
    panel = pd.read_csv(ROOT / cfg.data.panel_path, index_col="date", parse_dates=True)
    returns, _ = clean_returns(panel, cfg.data.clip)
    signal = tsmom(clean_price_index(returns), cfg.signal.lookback, cfg.signal.skip)
    return signal, returns


def snapshot_result() -> SizingResult:
    cfg = load_config()
    signal, returns = snapshot_inputs()
    return size_positions(signal, returns, cfg.sizing, cfg.execution.rebalance)


class TestSnapshot:
    """Real-data checks on the committed snapshot in ``data/``. No network."""

    def test_monthly_schedule_has_one_date_per_calendar_month(self):
        res = snapshot_result()
        # 2000-01 to 2026-09 inclusive is 26 years and 9 months = 321 months; the last row of
        # the panel (2026-09-07) closes the partial final month.
        assert len(res.rebalance_dates) == 321
        assert res.rebalance_dates[0] == pd.Timestamp("2000-01-31")
        assert res.rebalance_dates[-1] == pd.Timestamp("2026-09-07")

    def test_shapes_follow_the_panel(self):
        signal, _ = snapshot_inputs()
        res = snapshot_result()
        assert res.weights.index.equals(signal.index)
        assert list(res.weights.columns) == list(signal.columns)
        assert res.n_live.max() == 13

    def test_gross_never_exceeds_the_cap(self):
        cfg = load_config().sizing
        res = snapshot_result()
        gross = res.weights.abs().sum(axis=1, min_count=1).dropna()
        assert (gross <= cfg.leverage_cap + 1e-9).all()
        # On this panel the two-stage targeting never reaches the 3x cap: no day binds.
        assert gross.max() < cfg.leverage_cap
        assert not res.cap_binds.any()

    def test_weights_change_only_on_rebalance_dates(self):
        res = snapshot_result()
        w, prev = res.weights, res.weights.shift(1)
        changed = (w != prev) & ~(w.isna() & prev.isna())
        days = w.index[changed.any(axis=1)]
        assert len(days) > 250
        assert days.isin(res.rebalance_dates).all()

    def test_first_weights_land_after_the_second_warm_up(self):
        signal, _ = snapshot_inputs()
        res = snapshot_result()
        # The first signal is 2001-01-03 (KC=F, CT=F), so the book's first return is on the
        # next row and its vol_window = 60 valid returns are complete 60 rows after the first
        # signal: the portfolio vol first exists on 2001-03-30, which is also a month-end, so
        # the first weights are decided that same day.
        first_signal = signal.notna().any(axis=1).idxmax()
        first_pvol = res.portfolio_vol.first_valid_index()
        assert first_pvol == signal.index[signal.index.get_loc(first_signal) + 60]
        expected = res.rebalance_dates[res.rebalance_dates >= first_pvol][0]
        assert res.weights.notna().any(axis=1).idxmax() == expected
        assert expected == pd.Timestamp("2001-03-30") == first_pvol

    def test_unsizable_days_are_the_gaps_their_echoes_and_the_vol_warm_up_after_them(self):
        res = snapshot_result()
        raw = res.raw_weights
        defined = raw.notna()
        span = defined.cummax() & defined[::-1].cummax()[::-1]
        unsizable = (span & ~defined).sum()
        # LE=F's 20-day gap (January 2006) is 21 NaN return rows: the signal is NaN on those
        # and on their echo 252 rows later (42, as in Phase 2); the rolling vol is NaN on
        # them and on the 59 rows after (80). Union: 80 + 21 = 101. ZC=F and ZW=F only have
        # their echoes (19 and 8): their gaps sit inside the signal warm-up, long before the
        # first raw weight. Everyone else is complete.
        assert unsizable["LE=F"] == 101
        assert unsizable["ZC=F"] == 19
        assert unsizable["ZW=F"] == 8
        assert (unsizable.drop(["LE=F", "ZC=F", "ZW=F"]) == 0).all()
        assert sizing_summary(res)["n_unsizable"] == 128

    def test_headline_numbers_are_plausible(self):
        out = sizing_summary(snapshot_result())
        assert 1.0 <= out["mean_gross_leverage"] <= 3.0
        assert 0.0 <= out["cap_bind_fraction"] <= 0.4
        assert 8 <= out["mean_n_live"] <= 13

    def test_last_row_has_a_weight_for_every_instrument(self):
        res = snapshot_result()
        assert res.weights.iloc[-1].notna().all()

    def test_weights_have_no_lookahead_on_the_snapshot(self):
        cfg = load_config()
        panel = pd.read_csv(ROOT / cfg.data.panel_path, index_col="date", parse_dates=True)

        def weights(p: pd.DataFrame) -> pd.DataFrame:
            r, _ = clean_returns(p, cfg.data.clip)
            s = tsmom(clean_price_index(r), cfg.signal.lookback, cfg.signal.skip)
            return size_positions(s, r, cfg.sizing, cfg.execution.rebalance).weights

        # A month-end cut (see TestIntegrityNoLookahead on the monthly schedule).
        assert_no_lookahead(weights, panel, "2015-12-31", n_trials=1)
