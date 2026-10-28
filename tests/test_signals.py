"""Signal (Phase 2): trailing return, TSMOM sign signal, signal summary, and the §8 synthetic-
trend and no-lookahead integrity tests.

Synthetic frames use business days from 2020-01-01: d1..d12 =
01-01 01-02 01-03 01-06 01-07 01-08 01-09 01-10 01-13 01-14 01-15 01-16.
Every expected value is hand-computed; the arithmetic is in a comment next to it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.config import load_config
from src.data import clean_price_index, clean_returns
from src.signals import SIGNAL_VALUES, SUMMARY_COLUMNS, signal_summary, trailing_return, tsmom
from tests.integrity import assert_no_lookahead

ROOT = Path(__file__).resolve().parents[1]
NA = np.nan
D = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=12)), name="date")  # d1..d12


def frame(**columns: list[float]) -> pd.DataFrame:
    """Wide frame on d1..dn (n = the common length of the lists); NaN where a value is NA."""
    n = len(next(iter(columns.values())))
    return pd.DataFrame({k: v for k, v in columns.items()}, index=D[:n], dtype=float)


def geometric(n: int, step: float, start: float = 100.0) -> list[float]:
    """start, start*(1+step), start*(1+step)^2, ..."""
    return [start * (1 + step) ** k for k in range(n)]


# ------------------------------------------------------------------ trailing_return


class TestTrailingReturn:
    def test_hand_computed_lookback_two(self):
        p = frame(A=[100, 110, 121, 108.9, 99, 99])
        out = trailing_return(p, lookback=2)
        # r3 = 121/100 - 1 = 0.21; r4 = 108.9/110 - 1 = -0.01; r5 = 99/121 - 1 = -0.181818...;
        # r6 = 99/108.9 - 1 = -0.090909...; r1, r2: no price two rows up.
        expected = pd.Series(
            [NA, NA, 0.21, -0.01, 99 / 121 - 1, 99 / 108.9 - 1], index=D[:6], name="A"
        )
        assert_series_equal(out["A"], expected)

    def test_warm_up_is_lookback_plus_skip_rows(self):
        p = frame(A=geometric(8, 0.01))
        out = trailing_return(p, lookback=3, skip=2)
        # The first valid row needs P(t - 2) and P(t - 5): row 6 (index 5) is the first.
        assert out["A"].isna().tolist() == [True] * 5 + [False] * 3

    def test_skip_moves_the_window_back(self):
        p = frame(A=[100, 110, 121, 133.1, 146.41])
        out = trailing_return(p, lookback=2, skip=1)
        # r4 = P(3)/P(1) - 1 = 121/100 - 1 = 0.21; r5 = P(4)/P(2) - 1 = 133.1/110 - 1 = 0.21.
        expected = pd.Series([NA, NA, NA, 0.21, 0.21], index=D[:5], name="A")
        assert_series_equal(out["A"], expected)

    def test_nan_endpoint_gives_nan_but_interior_nan_does_not(self):
        p = frame(A=[100, NA, NA, 121, 133.1, NA, 150])
        out = trailing_return(p, lookback=3)
        # r4 = 121/100 - 1 = 0.21 (rows 2-3 are inside the window and do not matter);
        # r5: base P(2) is NaN -> NaN; r6: P(6) is NaN -> NaN; r7 = 150/121 - 1 = 0.2396694...
        expected = pd.Series([NA, NA, NA, 0.21, NA, NA, 150 / 121 - 1], index=D[:7], name="A")
        assert_series_equal(out["A"], expected)

    def test_ragged_panel_each_column_has_its_own_warm_up(self):
        p = frame(A=[100, 110, 121, 133.1, 146.41, 161.051], B=[NA, NA, 50, 55, 60.5, 66.55])
        out = trailing_return(p, lookback=2)
        # A: every row from r3 is 1.1^2 - 1 = 0.21. B starts on d3, so its first return is
        # r5 = 60.5/50 - 1 = 0.21, then r6 = 66.55/55 - 1 = 0.21.
        expected = frame(A=[NA, NA, 0.21, 0.21, 0.21, 0.21], B=[NA, NA, NA, NA, 0.21, 0.21])
        assert_frame_equal(out, expected)

    def test_shape_index_columns_and_dtype_are_preserved(self):
        p = frame(B=geometric(6, 0.02), A=geometric(6, -0.02))
        out = trailing_return(p, lookback=2)
        assert out.index.equals(p.index)
        assert list(out.columns) == ["B", "A"]
        assert all(pd.api.types.is_float_dtype(d) for d in out.dtypes)

    def test_all_nan_column_stays_all_nan(self):
        p = frame(A=geometric(5, 0.01), B=[NA] * 5)
        assert trailing_return(p, lookback=2)["B"].isna().all()

    def test_input_is_not_modified(self):
        p = frame(A=geometric(6, 0.01))
        before = p.copy()
        trailing_return(p, lookback=2, skip=1)
        assert_frame_equal(p, before)

    def test_rejects_invalid_lookback_or_skip(self):
        p = frame(A=geometric(6, 0.01))
        with pytest.raises(ValueError):
            trailing_return(p, lookback=0)
        with pytest.raises(ValueError):
            trailing_return(p, lookback=2, skip=-1)


# ------------------------------------------------------------------------- tsmom (F2.1)


class TestTsmom:
    def test_sign_of_the_trailing_return(self):
        p = frame(A=[100, 110, 121, 108.9, 99, 99])
        out = tsmom(p, lookback=2)
        # trailing returns: NaN, NaN, +0.21, -0.01, -0.18, -0.09 -> NaN, NaN, +1, -1, -1, -1
        expected = pd.Series([NA, NA, 1.0, -1.0, -1.0, -1.0], index=D[:6], name="A")
        assert_series_equal(out["A"], expected)

    def test_exactly_zero_trailing_return_is_zero(self):
        p = frame(A=[100, 90, 100, 90, 100, 90])
        out = tsmom(p, lookback=2)
        # P(t) == P(t-2) on every row from r3: 100/100 - 1 = 0, 90/90 - 1 = 0 -> 0, not +1/-1.
        expected = pd.Series([NA, NA, 0.0, 0.0, 0.0, 0.0], index=D[:6], name="A")
        assert_series_equal(out["A"], expected)

    def test_values_are_float_and_in_the_signal_set(self):
        rng = np.random.default_rng(3)
        p = pd.DataFrame(
            100 * np.exp(rng.normal(0, 0.02, size=(12, 4)).cumsum(axis=0)),
            index=D,
            columns=list("ABCD"),
        )
        out = tsmom(p, lookback=3)
        assert all(pd.api.types.is_float_dtype(d) for d in out.dtypes)
        assert set(np.unique(out.to_numpy()[~np.isnan(out.to_numpy())])) <= set(SIGNAL_VALUES)

    def test_nan_exactly_where_the_trailing_return_is_nan(self):
        p = frame(A=[100, NA, NA, 121, 133.1, NA, 150], B=[NA, NA, 50, 55, 60.5, 66.55, 70])
        out = tsmom(p, lookback=3)
        assert_frame_equal(out.isna(), trailing_return(p, lookback=3).isna())

    def test_skip_is_passed_through(self):
        p = frame(A=[100, 110, 121, 108.9, 99, 99])
        out = tsmom(p, lookback=2, skip=1)
        # r4 = P(3)/P(1) - 1 = +0.21 -> +1; r5 = P(4)/P(2) - 1 = -0.01 -> -1;
        # r6 = P(5)/P(3) - 1 = 99/121 - 1 < 0 -> -1.
        expected = pd.Series([NA, NA, NA, 1.0, -1.0, -1.0], index=D[:6], name="A")
        assert_series_equal(out["A"], expected)

    def test_default_config_warm_up_is_252_rows(self):
        cfg = load_config().signal
        assert (cfg.lookback, cfg.skip) == (252, 0)
        idx = pd.DatetimeIndex(list(pd.bdate_range("2019-01-01", periods=300)), name="date")
        p = pd.DataFrame({"A": geometric(300, 0.001)}, index=idx)
        out = tsmom(p, cfg.lookback, cfg.skip)["A"]
        assert out.iloc[:252].isna().all()
        assert (out.iloc[252:] == 1.0).all()

    def test_input_is_not_modified(self):
        p = frame(A=geometric(6, 0.01))
        before = p.copy()
        tsmom(p, lookback=2)
        assert_frame_equal(p, before)


class TestIntegritySyntheticTrend:
    """PROJECT_LOG.md §8 "synthetic trend": rising -> +1, falling -> -1, flat -> 0 after warm-up."""

    def test_rising_price_is_long_after_warm_up(self):
        out = tsmom(frame(A=geometric(12, 0.001)), lookback=5)["A"]
        assert out.iloc[:5].isna().all()
        assert (out.iloc[5:] == 1.0).all()

    def test_falling_price_is_short_after_warm_up(self):
        out = tsmom(frame(A=geometric(12, -0.001)), lookback=5)["A"]
        assert out.iloc[:5].isna().all()
        assert (out.iloc[5:] == -1.0).all()

    def test_flat_price_is_zero_after_warm_up(self):
        out = tsmom(frame(A=[100.0] * 12), lookback=5)["A"]
        assert out.iloc[:5].isna().all()
        assert (out.iloc[5:] == 0.0).all()

    def test_three_regimes_side_by_side(self):
        p = frame(UP=geometric(12, 0.01), DOWN=geometric(12, -0.01), FLAT=[42.0] * 12)
        out = tsmom(p, lookback=4)
        expected = frame(
            UP=[NA] * 4 + [1.0] * 8, DOWN=[NA] * 4 + [-1.0] * 8, FLAT=[NA] * 4 + [0.0] * 8
        )
        assert_frame_equal(out, expected)

    def test_reversal_flips_the_sign_after_lookback_rows(self):
        p = frame(A=[100, 101, 102, 101, 100, 99])
        out = tsmom(p, lookback=2)
        # r3 = 102/100 > 0 -> +1; r4 = 101/101 = 0 -> 0; r5 = 100/102 < 0 -> -1; r6 = 99/101 -> -1
        expected = pd.Series([NA, NA, 1.0, 0.0, -1.0, -1.0], index=D[:6], name="A")
        assert_series_equal(out["A"], expected)


class TestIntegrityNoLookahead:
    """PROJECT_LOG.md §8 "no lookahead", applied to the signal through the shared harness."""

    IDX = pd.DatetimeIndex(list(pd.bdate_range("2018-01-01", periods=80)), name="date")

    def prices(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        steps = rng.normal(0.0, 0.02, size=(len(self.IDX), 6))
        p = pd.DataFrame(100 * np.exp(steps.cumsum(axis=0)), index=self.IDX, columns=list("ABCDEF"))
        p.iloc[:15, 1] = np.nan  # a late starter
        p.iloc[30:34, 2] = np.nan  # an unfilled gap
        return p

    @pytest.mark.parametrize("row", [20, 40, 60])
    def test_tsmom_has_no_lookahead(self, row):
        assert_no_lookahead(lambda p: tsmom(p, lookback=10), self.prices(), self.IDX[row])

    @pytest.mark.parametrize("row", [20, 40, 60])
    def test_tsmom_with_skip_has_no_lookahead(self, row):
        assert_no_lookahead(lambda p: tsmom(p, lookback=10, skip=3), self.prices(), self.IDX[row])

    def test_trailing_return_has_no_lookahead(self):
        assert_no_lookahead(
            lambda p: trailing_return(p, lookback=10, skip=2), self.prices(), self.IDX[40]
        )


# ---------------------------------------------------------------------- signal_summary


class TestSignalSummary:
    def signal(self) -> pd.DataFrame:
        return frame(
            A=[NA, NA, 1, 1, -1, 0, 1, NA],
            B=[NA, -1, -1, -1, -1, 1, 1, 1],
            C=[NA] * 8,
        )

    def test_hand_computed_table(self):
        out = signal_summary(self.signal())
        # A: first signal d3; 5 signal rows (d3-d7); 1 missing after d3 (d8); 3 long, 1 short,
        #    1 zero -> 0.6 / 0.2 / 0.2; last row NaN.
        # B: first d2; 7 rows; 0 missing; 3 long, 4 short, 0 zero -> 3/7, 4/7, 0; last +1.
        # C: never a signal -> NaT, 0, 0, NaN shares, NaN last.
        assert out.loc["A", "first_signal"] == D[2]
        assert out.loc["A", "n_signal"] == 5
        assert out.loc["A", "n_missing"] == 1
        assert out.loc["A", "share_long"] == pytest.approx(0.6)
        assert out.loc["A", "share_short"] == pytest.approx(0.2)
        assert out.loc["A", "share_zero"] == pytest.approx(0.2)
        assert np.isnan(out.loc["A", "last_signal"])
        assert out.loc["B", "first_signal"] == D[1]
        assert out.loc["B", "n_signal"] == 7
        assert out.loc["B", "n_missing"] == 0
        assert out.loc["B", "share_long"] == pytest.approx(3 / 7)
        assert out.loc["B", "share_short"] == pytest.approx(4 / 7)
        assert out.loc["B", "share_zero"] == 0.0
        assert out.loc["B", "last_signal"] == 1.0
        assert pd.isna(out.loc["C", "first_signal"])
        assert out.loc["C", "n_signal"] == 0
        assert out.loc["C", "n_missing"] == 0
        assert out.loc["C", ["share_long", "share_short", "share_zero", "last_signal"]].isna().all()

    def test_columns_index_and_dtypes(self):
        out = signal_summary(self.signal())
        assert list(out.columns) == list(SUMMARY_COLUMNS)
        assert out.index.name == "ticker"
        assert list(out.index) == ["A", "B", "C"]
        assert pd.api.types.is_integer_dtype(out["n_signal"])
        assert pd.api.types.is_integer_dtype(out["n_missing"])
        assert pd.api.types.is_datetime64_any_dtype(out["first_signal"])

    def test_shares_sum_to_one_where_a_signal_exists(self):
        out = signal_summary(self.signal())
        total = out[["share_long", "share_short", "share_zero"]].sum(axis=1)
        assert total.loc[["A", "B"]].tolist() == pytest.approx([1.0, 1.0])

    def test_rejects_values_outside_the_signal_set(self):
        bad = frame(A=[NA, 1, 0.5, -1])
        with pytest.raises(ValueError):
            signal_summary(bad)

    def test_round_trip_from_tsmom(self):
        p = frame(UP=geometric(12, 0.01), DOWN=geometric(12, -0.01))
        out = signal_summary(tsmom(p, lookback=4))
        # 8 signal rows each, all long for UP and all short for DOWN, none missing.
        assert out["n_signal"].tolist() == [8, 8]
        assert out["n_missing"].tolist() == [0, 0]
        assert out.loc["UP", "share_long"] == 1.0
        assert out.loc["DOWN", "share_short"] == 1.0
        assert out["last_signal"].tolist() == [1.0, -1.0]


# -------------------------------------------------------------------------- snapshot


def snapshot_prices() -> pd.DataFrame:
    """The committed panel as the clean price index the signal is computed on."""
    cfg = load_config()
    panel = pd.read_csv(ROOT / cfg.data.panel_path, index_col="date", parse_dates=True)
    returns, _ = clean_returns(panel, cfg.data.clip)
    return clean_price_index(returns)


class TestSnapshot:
    """Real-data checks on the committed snapshot in ``data/``. No network."""

    cfg = load_config()

    def test_signal_covers_the_panel(self):
        prices = snapshot_prices()
        out = tsmom(prices, self.cfg.signal.lookback, self.cfg.signal.skip)
        assert out.index.equals(prices.index)
        assert list(out.columns) == list(prices.columns)
        assert len(out.columns) == 13

    def test_first_signal_is_one_lookback_after_the_second_price(self):
        panel = pd.read_csv(ROOT / self.cfg.data.panel_path, index_col="date", parse_dates=True)
        out = tsmom(snapshot_prices(), self.cfg.signal.lookback, self.cfg.signal.skip)
        for ticker in panel.columns:
            # The clean price index starts one row after the first price (its first return
            # is NaN), and the signal needs `lookback` more rows on top of that.
            first_price = panel.index.get_loc(panel[ticker].first_valid_index())
            expected = panel.index[first_price + 1 + self.cfg.signal.lookback]
            assert out[ticker].first_valid_index() == expected, ticker

    def test_missing_signal_days_are_the_gaps_and_their_echoes(self):
        out = tsmom(snapshot_prices(), self.cfg.signal.lookback, self.cfg.signal.skip)
        missing = signal_summary(out)["n_missing"]
        # Unfilled price gaps in the panel (longer than ffill_limit = 5): LE=F 20 days from
        # 2006-01-03, ZC=F 9 days from 2000-12-05 and 8 from 2001-01-03, ZW=F 7 from 2000-12-06.
        # A gap of g price rows is g + 1 NaN rows in the clean price index (the day after a gap
        # has no return either). Each such row costs the signal a NaN twice: on the day itself,
        # and 252 rows later as the missing base price (the echo). ZC=F and ZW=F start on
        # 2000-07-17 (row 135 of the panel) and their gaps sit inside the 253-row warm-up, so
        # only the echoes are counted after the first signal.
        # LE=F: 2 * 21 = 42; ZC=F: 10 + 9 = 19; ZW=F: 8; everyone else 0.
        assert missing["LE=F"] == 42
        assert missing["ZC=F"] == 19
        assert missing["ZW=F"] == 8
        others = missing.drop(["LE=F", "ZC=F", "ZW=F"])
        assert (others == 0).all(), others[others != 0]

    def test_every_instrument_is_long_and_short_at_some_point(self):
        out = tsmom(snapshot_prices(), self.cfg.signal.lookback, self.cfg.signal.skip)
        summary = signal_summary(out)
        assert (summary["share_long"] > 0).all() and (summary["share_long"] < 1).all()
        assert (summary["share_short"] > 0).all()

    def test_last_row_has_a_signal_for_every_instrument(self):
        out = tsmom(snapshot_prices(), self.cfg.signal.lookback, self.cfg.signal.skip)
        assert out.iloc[-1].notna().all()

    def test_signal_has_no_lookahead_on_the_snapshot(self):
        cfg = self.cfg.signal
        assert_no_lookahead(
            lambda p: tsmom(p, cfg.lookback, cfg.skip), snapshot_prices(), "2015-12-31", n_trials=1
        )
