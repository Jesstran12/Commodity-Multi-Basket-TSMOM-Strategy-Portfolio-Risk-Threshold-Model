"""Sub-periods (Phase 6, F6.3): calendar blocks, calendar years, the rolling Sharpe ratio and
the calm/stress regime view (which uses hindsight by design, and a test says so).

Every expected value is hand-computed; the arithmetic is in a comment next to it. Where a
statistic is one of ``src.metrics``' (tested in ``tests/test_metrics.py``), the expected
value is that function applied to the slice the contract names.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ANNUALISATION
from src.metrics import annualised_return, annualised_vol, sharpe
from src.portfolio import realised_vol
from src.subperiods import (
    BLOCK_COLUMNS,
    REGIME_COLUMNS,
    REGIMES,
    ROLLING_SUMMARY_COLUMNS,
    block_label,
    block_stats,
    calendar_year_returns,
    regime_labels,
    regime_stats,
    rolling_sharpe,
    rolling_summary,
)
from tests.integrity import assert_no_lookahead

NA = np.nan
SQ = np.sqrt(ANNUALISATION)  # 15.8745...


def frame(index: list[str], **columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.DatetimeIndex(index, name="date"), dtype=float)


# Six days across the 2007/2008 boundary; "b" starts in 2008, "a" misses the last day.
BOUNDARY = frame(
    ["2007-12-27", "2007-12-28", "2007-12-31", "2008-01-02", "2008-01-03", "2008-01-04"],
    a=[0.01, -0.02, 0.03, 0.01, 0.02, NA],
    b=[NA, NA, NA, 0.01, -0.01, 0.02],
)
BLOCKS = (("2000-01-01", "2007-12-31"), ("2008-01-01", None), ("2020-01-01", None))


class TestBlockLabel:
    def test_labels(self):
        assert block_label("2000-01-01", "2007-12-31") == "2000-2007"
        assert block_label("2020-01-01", None) == "2020-present"
        assert block_label(None, "2007-12-31") == "start-2007"


class TestBlockStats:
    def test_layout_block_major(self):
        out = block_stats(BOUNDARY, BLOCKS)
        assert list(out.columns) == list(BLOCK_COLUMNS)
        assert isinstance(out.index, pd.RangeIndex) and len(out) == 6
        assert (
            out["block"].tolist() == ["2000-2007"] * 2 + ["2008-present"] * 2 + ["2020-present"] * 2
        )
        assert out["series"].tolist() == ["a", "b"] * 3

    def test_days_and_dates_ends_inclusive(self):
        out = block_stats(BOUNDARY, BLOCKS)
        # a in 2000-2007: 12-27, 12-28, 12-31 (the block's last day included).
        assert out.loc[0, "n_days"] == 3
        assert out.loc[0, "first_date"] == pd.Timestamp("2007-12-27")
        assert out.loc[0, "last_date"] == pd.Timestamp("2007-12-31")
        # a in 2008-present: 01-02, 01-03 (01-04 is NaN, dropped); b: all three 2008 days.
        assert out.loc[2, "n_days"] == 2 and out.loc[2, "last_date"] == pd.Timestamp("2008-01-03")
        assert out.loc[3, "n_days"] == 3 and out.loc[3, "first_date"] == pd.Timestamp("2008-01-02")

    def test_statistics_are_the_metrics_of_the_slice(self):
        out = block_stats(BOUNDARY, BLOCKS, rf=0.02)
        a07 = BOUNDARY["a"].iloc[:3]
        assert out.loc[0, "ann_return"] == pytest.approx(annualised_return(a07))
        assert out.loc[0, "ann_vol"] == pytest.approx(annualised_vol(a07))
        assert out.loc[0, "sharpe"] == pytest.approx(sharpe(a07, 0.02))
        b08 = BOUNDARY["b"].iloc[3:]
        assert out.loc[3, "sharpe"] == pytest.approx(sharpe(b08, 0.02))
        # By hand: a in 2007 compounds 1.01 * 0.98 * 1.03 = 1.019494 over 3 rows.
        assert out.loc[0, "ann_return"] == pytest.approx(1.019494 ** (252 / 3) - 1)

    def test_a_block_without_rows_is_zero_days_and_nan(self):
        out = block_stats(BOUNDARY, BLOCKS)
        for i in (1, 4, 5):  # b before 2008; both series from 2020
            assert out.loc[i, "n_days"] == 0
            assert pd.isna(out.loc[i, "first_date"]) and pd.isna(out.loc[i, "last_date"])
            assert np.isnan(out.loc[i, "sharpe"]) and np.isnan(out.loc[i, "ann_return"])

    def test_open_start(self):
        out = block_stats(BOUNDARY, ((None, "2007-12-28"),))
        assert out["block"].tolist() == ["start-2007", "start-2007"]
        assert out["n_days"].tolist() == [2, 0]

    def test_input_not_modified(self):
        before = BOUNDARY.copy()
        block_stats(BOUNDARY, BLOCKS)
        pd.testing.assert_frame_equal(BOUNDARY, before)


class TestCalendarYearReturns:
    def test_compounded_by_year(self):
        out = calendar_year_returns(BOUNDARY)
        assert out.index.tolist() == [2007, 2008] and out.index.name == "year"
        assert list(out.columns) == ["a", "b"]
        # a 2007: 1.01 * 0.98 * 1.03 - 1 = 0.019494; a 2008: 1.01 * 1.02 - 1 = 0.0302.
        assert out.loc[2007, "a"] == pytest.approx(0.019494)
        assert out.loc[2008, "a"] == pytest.approx(0.0302)
        # b 2008: 1.01 * 0.99 * 1.02 - 1 = 0.9999 * 1.02 - 1 = 0.019898; b 2007: no row.
        assert out.loc[2008, "b"] == pytest.approx(0.019898)
        assert np.isnan(out.loc[2007, "b"])

    def test_years_are_integers_and_a_gap_year_is_nan_not_zero(self):
        f = frame(["2019-12-31", "2021-01-04"], a=[0.05, NA], b=[NA, 0.02])
        out = calendar_year_returns(f)
        assert out.index.tolist() == [2019, 2021]
        assert out.index.dtype.kind == "i"
        assert out.loc[2019, "a"] == pytest.approx(0.05) and np.isnan(out.loc[2021, "a"])
        assert np.isnan(out.loc[2019, "b"]) and out.loc[2021, "b"] == pytest.approx(0.02)


# ------------------------------------------------------------------- rolling Sharpe

SIX = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
ROLL = frame(SIX, x=[0.01, 0.03, 0.02, NA, 0.05, 0.02], flat=[0.01] * 6)


class TestRollingSharpe:
    def test_known_answer_window_3_over_each_columns_own_rows(self):
        out = rolling_sharpe(ROLL, 3)
        assert out.index.equals(ROLL.index) and list(out.columns) == ["x", "flat"]
        x = out["x"]
        # Non-NaN rows of x: 0.01, 0.03, 0.02, 0.05, 0.02.
        # [0.01, 0.03, 0.02]: mean 0.02, std sqrt((1e-4 + 1e-4 + 0) / 2) = 0.01 -> 2 * sqrt(252)
        # [0.03, 0.02, 0.05]: mean 0.0333333, squares 1.11111e-5 + 1.77778e-4 + 2.77778e-4 =
        #   4.66667e-4, / 2 = 2.33333e-4, std 0.0152753 -> 2.182179 * sqrt(252) = 34.6410
        # [0.02, 0.05, 0.02]: mean 0.03, squares 1e-4 + 4e-4 + 1e-4 = 6e-4, / 2 = 3e-4,
        #   std 0.0173205 -> 1.732051 * sqrt(252) = 27.4955
        assert np.isnan(x.iloc[0]) and np.isnan(x.iloc[1])
        assert x.iloc[2] == pytest.approx(2 * SQ)
        assert np.isnan(x.iloc[3])  # the column itself is NaN on that row
        assert x.iloc[4] == pytest.approx(34.6410, abs=1e-4)
        assert x.iloc[5] == pytest.approx(27.4955, abs=1e-4)

    def test_zero_std_is_nan(self):
        assert rolling_sharpe(ROLL, 3)["flat"].isna().all()

    def test_risk_free_rate_is_per_row(self):
        # rf 2.52% a year = 0.0001 a day: the first window's excess mean is 0.0199.
        out = rolling_sharpe(ROLL, 3, rf=0.0252)
        assert out["x"].iloc[2] == pytest.approx(1.99 * SQ)

    def test_window_below_two_raises(self):
        with pytest.raises(ValueError):
            rolling_sharpe(ROLL, 1)

    def test_lookahead_rolling_sharpe(self):
        rng = np.random.default_rng(3)
        days = pd.bdate_range("2020-01-01", periods=300)
        prices = pd.DataFrame(
            np.exp(np.cumsum(rng.normal(0, 0.01, size=(300, 2)), axis=0)),
            index=days,
            columns=["p", "q"],
        )
        for t in (days[60], days[150], days[298]):
            assert_no_lookahead(
                lambda p: rolling_sharpe(p.pct_change(fill_method=None), 40), prices, t
            )


class TestRollingSummary:
    D5 = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]

    def test_values(self):
        r = frame(self.D5, s=[NA, 0.5, -0.2, 0.5, 0.1], empty=[NA] * 5)
        out = rolling_summary(r)
        assert list(out.columns) == list(ROLLING_SUMMARY_COLUMNS)
        assert out.index.tolist() == ["s", "empty"]
        s = out.loc["s"]
        assert s["n_windows"] == 4
        assert s["share_positive"] == pytest.approx(3 / 4)
        assert s["min"] == pytest.approx(-0.2) and s["min_date"] == pd.Timestamp("2020-01-03")
        # Two windows at 0.5: the first one's date.
        assert s["max"] == pytest.approx(0.5) and s["max_date"] == pd.Timestamp("2020-01-02")
        assert s["last"] == pytest.approx(0.1) and s["last_date"] == pd.Timestamp("2020-01-07")

    def test_a_column_without_values(self):
        out = rolling_summary(frame(self.D5, s=[NA, 0.5, -0.2, 0.5, 0.1], empty=[NA] * 5))
        e = out.loc["empty"]
        assert e["n_windows"] == 0
        assert np.isnan(e["min"]) and np.isnan(e["last"]) and pd.isna(e["max_date"])


# ------------------------------------------------------------------------- regimes

EIGHT = [
    "2020-01-01",
    "2020-01-02",
    "2020-01-03",
    "2020-01-06",
    "2020-01-07",
    "2020-01-08",
    "2020-01-09",
    "2020-01-10",
]
BASKET = pd.Series(
    [0.0, 0.01, -0.01, 0.0, 0.05, -0.05, 0.0, 0.0],
    index=pd.DatetimeIndex(EIGHT, name="date"),
    name="basket",
)


class TestRegimeLabels:
    def test_known_answer(self):
        # Window 2, std of each pair times sqrt(252):
        #   rows 1..7: (0, .01) 0.0070711, (.01, -.01) 0.0141421, (-.01, 0) 0.0070711,
        #   (0, .05) 0.0353553, (.05, -.05) 0.0707107, (-.05, 0) 0.0353553, (0, 0) 0
        #   -> x 15.8745: 0.1122, 0.2245, 0.1122, 0.5612, 1.1225, 0.5612, 0
        # Median (quantile 0.5) of the seven: sorted 0, .1122, .1122, .2245, .5612, .5612,
        # 1.1225 -> position 3 = 0.2245 (row 2's own value, so row 2 is "at" the threshold).
        out = regime_labels(BASKET, 2, 0.5)
        assert out.name == "regime" and out.index.equals(BASKET.index)
        assert pd.isna(out.iloc[0])
        assert out.iloc[1:].tolist() == [
            "calm",
            "calm",
            "calm",
            "stress",
            "stress",
            "stress",
            "calm",
        ]

    def test_vol_is_the_portfolio_rolling_estimator(self):
        vol = realised_vol(BASKET.to_frame(), 2, "rolling").iloc[:, 0]
        threshold = vol.quantile(0.75)
        out = regime_labels(BASKET, 2, 0.75)
        stress = vol > threshold
        assert (out[vol.notna()] == np.where(stress[vol.notna()], "stress", "calm")).all()

    def test_labels_depend_on_the_future_by_design(self):
        # The hindsight the report's caveat is about: append 20 violent days (+-20%,
        # alternating, every pair's vol about 4.49) and the median of the 28 vols lands
        # among them, so 2020-01-07 (1.12), "stress" before, becomes "calm".
        later = pd.bdate_range("2020-01-13", periods=20)
        extended = pd.concat([BASKET, pd.Series([0.2, -0.2] * 10, index=later)])
        before = regime_labels(BASKET, 2, 0.5)
        after = regime_labels(extended, 2, 0.5).loc[BASKET.index]
        assert before.loc["2020-01-07"] == "stress" and after.loc["2020-01-07"] == "calm"

    @pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
    def test_quantile_out_of_range_raises(self, q):
        with pytest.raises(ValueError):
            regime_labels(BASKET, 2, q)


class TestRegimeStats:
    R6 = frame(
        SIX,
        a=[0.01, 0.02, -0.03, 0.04, -0.01, 0.00],
        b=[NA, 0.01, 0.02, 0.0, NA, 0.03],
    )
    LABELS = pd.Series(
        ["calm", "calm", "stress", None, "stress", "calm"],
        index=pd.DatetimeIndex(SIX, name="date"),
        name="regime",
    )

    def test_layout(self):
        out = regime_stats(self.R6, self.LABELS)
        assert list(out.columns) == list(REGIME_COLUMNS)
        assert REGIMES == ("calm", "stress")
        assert out["regime"].tolist() == ["calm", "calm", "stress", "stress"]
        assert out["series"].tolist() == ["a", "b", "a", "b"]
        assert isinstance(out.index, pd.RangeIndex)

    def test_counts_and_shares(self):
        out = regime_stats(self.R6, self.LABELS)
        # a: labelled non-NaN rows 0, 1, 2, 4, 5 (row 3 has no label): calm 0, 1, 5 (3 of 5),
        # stress 2, 4 (2 of 5). b: labelled non-NaN rows 1, 2, 5: calm 1, 5 (2 of 3), stress 2.
        assert out["n_days"].tolist() == [3, 2, 2, 1]
        assert out["share_of_days"].tolist() == pytest.approx([0.6, 2 / 3, 0.4, 1 / 3])

    def test_statistics_are_the_metrics_of_the_regimes_days(self):
        out = regime_stats(self.R6, self.LABELS, rf=0.01)
        calm_a = self.R6["a"].iloc[[0, 1, 5]]
        assert out.loc[0, "ann_return"] == pytest.approx(annualised_return(calm_a))
        assert out.loc[0, "ann_vol"] == pytest.approx(annualised_vol(calm_a))
        assert out.loc[0, "sharpe"] == pytest.approx(sharpe(calm_a, 0.01))
        assert np.isnan(out.loc[3, "sharpe"])  # one stress day for b: no std

    def test_labels_are_aligned_to_the_returns(self):
        shuffled = self.LABELS.iloc[::-1]
        extra = pd.concat([shuffled, pd.Series(["stress"], index=pd.DatetimeIndex(["2021-01-04"]))])
        pd.testing.assert_frame_equal(
            regime_stats(self.R6, extra), regime_stats(self.R6, self.LABELS)
        )
