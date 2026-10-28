"""Data panel (Phase 1): fetch layer, alignment and gap handling, cleaned returns, quality
gate and report, snapshot manifest, external benchmark.

Synthetic fixtures use business days from 2020-01-01: d1..d10 =
01-01 01-02 01-03 01-06 01-07 01-08 01-09 01-10 01-13 01-14 (Jan 4/5 and 11/12 are weekends).
Every expected value is hand-computed; the arithmetic is in a comment next to it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from src.config import QualityGateConfig, load_config
from src.data import (
    CSVSource,
    FetchError,
    PanelBuild,
    YFinanceSource,
    align_prices,
    build_manifest,
    clean_price_index,
    clean_returns,
    data_quality_report,
    extract_price_field,
    fetch_all,
    fetch_benchmark,
    load_raw,
    quality_gate,
    read_raw,
    run_panel,
    sha256_of,
    universe_dict,
    verify_manifest,
    write_raw,
)

ROOT = Path(__file__).resolve().parents[1]
D = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=10)), name="date")  # d1..d10


def series(ticker: str, values: dict[int, float]) -> pd.Series:
    """Price series on the days {k: value} of D (1-based day numbers)."""
    idx = pd.DatetimeIndex([D[k - 1] for k in values], name="date")
    return pd.Series(list(values.values()), index=idx, name=ticker, dtype=float)


def frame(columns: dict[str, list[float | None]], index: pd.DatetimeIndex = D) -> pd.DataFrame:
    return pd.DataFrame({t: pd.array(v, dtype=float) for t, v in columns.items()}, index=index)


class DictSource:
    """PriceSource over an in-memory dict; unknown tickers raise FetchError."""

    def __init__(self, data: dict[str, pd.Series]) -> None:
        self.data = data
        self.calls: list[str] = []

    def fetch(self, ticker: str) -> pd.Series:
        self.calls.append(ticker)
        if ticker not in self.data:
            raise FetchError(f"{ticker}: not available")
        return self.data[ticker]


def yf_frame(ticker: str, adj: list[float], close: list[float], index: pd.Index) -> pd.DataFrame:
    """A frame shaped like ``yfinance.download(auto_adjust=False)``: MultiIndex columns."""
    cols = pd.MultiIndex.from_tuples(
        [("Adj Close", ticker), ("Close", ticker), ("Volume", ticker)], names=["Price", "Ticker"]
    )
    data = np.column_stack([adj, close, np.ones(len(adj))])
    return pd.DataFrame(data, index=index, columns=cols)


# ------------------------------------------------------------------ F1.1 fetch layer


class TestExtractPriceField:
    def test_multiindex_frame_returns_requested_field_named_by_ticker(self):
        fr = yf_frame("GC=F", [1.0, 2.0, 3.0], [9.0, 9.0, 9.0], D[:3])
        s = extract_price_field(fr, "GC=F", "Adj Close")
        assert_series_equal(s, series("GC=F", {1: 1.0, 2: 2.0, 3: 3.0}))

    def test_falls_back_to_close_when_field_absent(self):
        fr = yf_frame("GC=F", [1.0, 2.0], [9.0, 8.0], D[:2]).drop(columns="Adj Close", level=0)
        s = extract_price_field(fr, "GC=F", "Adj Close")
        assert s.tolist() == [9.0, 8.0]

    def test_flat_columns_are_accepted(self):
        fr = pd.DataFrame({"Adj Close": [5.0, 6.0], "Close": [1.0, 1.0]}, index=D[:2])
        assert extract_price_field(fr, "X", "Adj Close").tolist() == [5.0, 6.0]

    def test_nan_rows_are_dropped(self):
        fr = yf_frame("X", [1.0, np.nan, 3.0], [1.0, 1.0, 1.0], D[:3])
        s = extract_price_field(fr, "X", "Adj Close")
        assert s.index.tolist() == [D[0], D[2]]
        assert s.tolist() == [1.0, 3.0]

    def test_timezone_aware_index_becomes_naive_dates(self):
        idx = pd.DatetimeIndex(D[:2]).tz_localize("America/New_York")
        s = extract_price_field(yf_frame("X", [1.0, 2.0], [1.0, 2.0], idx), "X", "Adj Close")
        assert s.index.tz is None
        assert_index_equal(s.index, D[:2])

    def test_output_conventions(self):
        s = extract_price_field(yf_frame("X", [1.0, 2.0], [1.0, 2.0], D[:2]), "X", "Adj Close")
        assert s.name == "X"
        assert s.index.name == "date"
        assert s.dtype.kind == "f"
        assert s.index.is_monotonic_increasing

    def test_neither_field_present_raises(self):
        fr = pd.DataFrame({"Open": [1.0]}, index=D[:1])
        with pytest.raises(FetchError, match="X"):
            extract_price_field(fr, "X", "Adj Close")

    def test_empty_frame_raises(self):
        with pytest.raises(FetchError, match="X"):
            extract_price_field(pd.DataFrame(), "X", "Adj Close")


class FlakyDownloader:
    """Raises ``fail_times`` times, then returns ``result`` (None = an empty frame)."""

    def __init__(self, fail_times: int, result: pd.DataFrame | None) -> None:
        self.fail_times = fail_times
        self.result = result if result is not None else pd.DataFrame()
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, ticker: str, **kwargs: object) -> pd.DataFrame:
        self.calls.append((ticker, kwargs))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("network down")
        return self.result


class TestYFinanceSource:
    def test_succeeds_on_third_attempt_and_sleeps_between_attempts(self):
        dl = FlakyDownloader(2, yf_frame("CL=F", [1.0, 2.0], [1.0, 2.0], D[:2]))
        sleeps: list[float] = []
        src = YFinanceSource("Adj Close", retries=3, downloader=dl, sleep=sleeps.append)
        s = src.fetch("CL=F")
        assert s.tolist() == [1.0, 2.0]
        assert [t for t, _ in dl.calls] == ["CL=F", "CL=F", "CL=F"]
        assert len(sleeps) == 2

    def test_gives_up_after_retries_with_a_clear_error(self):
        dl = FlakyDownloader(5, None)
        src = YFinanceSource("Adj Close", retries=2, downloader=dl, sleep=lambda _s: None)
        with pytest.raises(FetchError, match="CL=F"):
            src.fetch("CL=F")
        assert len(dl.calls) == 2

    def test_empty_download_counts_as_a_failure(self):
        dl = FlakyDownloader(0, None)  # never raises, always empty
        src = YFinanceSource("Adj Close", retries=3, downloader=dl, sleep=lambda _s: None)
        with pytest.raises(FetchError, match="CL=F"):
            src.fetch("CL=F")
        assert len(dl.calls) == 3

    def test_start_date_is_forwarded_to_the_downloader(self):
        dl = FlakyDownloader(0, yf_frame("X", [1.0], [1.0], D[:1]))
        YFinanceSource("Adj Close", retries=1, start="2005-01-01", downloader=dl).fetch("X")
        assert dl.calls[0][1]["start"] == "2005-01-01"
        assert "period" not in dl.calls[0][1]

    def test_no_start_date_requests_the_maximum_history(self):
        # yfinance's default window is one month; a backtest needs period="max".
        dl = FlakyDownloader(0, yf_frame("X", [1.0], [1.0], D[:1]))
        YFinanceSource("Adj Close", retries=1, start=None, downloader=dl).fetch("X")
        assert dl.calls[0][1]["period"] == "max"
        assert dl.calls[0][1]["auto_adjust"] is False


class TestCSVSource:
    def test_fetches_one_column_of_a_wide_csv(self, tmp_path):
        p = tmp_path / "wide.csv"
        p.write_text("date,A,B\n2020-01-01,1.0,2.0\n2020-01-02,1.5,\n")
        src = CSVSource(p)
        assert_series_equal(src.fetch("A"), series("A", {1: 1.0, 2: 1.5}))
        assert_series_equal(src.fetch("B"), series("B", {1: 2.0}))  # NaN row dropped

    def test_unknown_ticker_raises(self, tmp_path):
        p = tmp_path / "wide.csv"
        p.write_text("date,A\n2020-01-01,1.0\n")
        with pytest.raises(FetchError, match="C"):
            CSVSource(p).fetch("C")


class TestRawFiles:
    def test_write_then_read_round_trips(self, tmp_path):
        s = series("CL=F", {1: 10.0, 2: -37.63, 3: 10.01})
        path = write_raw(s, tmp_path)
        assert path == tmp_path / "CL=F.csv"
        assert_series_equal(read_raw(tmp_path, "CL=F"), s)

    def test_load_raw_keeps_ticker_order(self, tmp_path):
        for t in ("B", "A"):
            write_raw(series(t, {1: 1.0}), tmp_path)
        assert list(load_raw(tmp_path, ("A", "B"))) == ["A", "B"]


class TestFetchAll:
    def test_writes_every_ticker_and_returns_the_series(self, tmp_path):
        data = {"A": series("A", {1: 1.0, 2: 2.0}), "B": series("B", {2: 5.0})}
        out = fetch_all(DictSource(data), ("A", "B"), tmp_path)
        assert list(out) == ["A", "B"]
        assert_series_equal(out["A"], data["A"])
        assert_series_equal(read_raw(tmp_path, "B"), data["B"])

    def test_one_failure_names_the_ticker_after_trying_all(self, tmp_path):
        data = {"A": series("A", {1: 1.0}), "C": series("C", {1: 3.0})}
        src = DictSource(data)
        with pytest.raises(FetchError, match="B"):
            fetch_all(src, ("A", "B", "C"), tmp_path)
        assert src.calls == ["A", "B", "C"]  # did not stop at the first failure
        assert (tmp_path / "C.csv").exists()  # the good ones were still written


# ------------------------------------------------- F1.2 alignment and gap handling


@pytest.fixture
def ragged() -> dict[str, pd.Series]:
    return {
        "A": series("A", {k: float(k) for k in range(1, 11)}),  # all ten days, price = day no.
        "B": series("B", {2: 20.0, 3: 21.0, 6: 22.0, 7: 23.0, 8: 24.0, 9: 25.0, 10: 26.0}),
        # B: missing d1 (leading) and d4, d5 (a 2-day gap)
        "C": series("C", {1: 30.0, 2: 31.0, 7: 32.0, 8: 33.0, 9: 34.0, 10: 35.0}),
        # C: missing d3, d4, d5, d6 (a 4-day gap)
        "Dd": series("Dd", {k: 40.0 + k for k in range(1, 9)}),
        # Dd: missing d9, d10 (a trailing 2-day run)
    }


class TestAlignPrices:
    def test_calendar_is_sorted_union_of_valid_dates(self):
        prices, _ = align_prices(
            {"X": series("X", {1: 1.0, 3: 3.0}), "Y": series("Y", {2: 2.0})}, 5
        )
        assert_index_equal(prices.index, D[:3])
        assert prices.index.name == "date"
        assert prices.index.tz is None
        assert list(prices.columns) == ["X", "Y"]  # input order

    def test_present_values_are_unchanged(self, ragged):
        prices, _ = align_prices(ragged, 5)
        assert prices["A"].tolist() == [float(k) for k in range(1, 11)]
        assert prices.loc[D[1], "B"] == 20.0 and prices.loc[D[9], "C"] == 35.0

    def test_leading_missing_days_are_never_filled(self, ragged):
        prices, filled = align_prices(ragged, 5)
        assert np.isnan(prices.loc[D[0], "B"])
        assert not filled.loc[D[0], "B"]

    def test_short_gap_is_filled_with_last_price_and_flagged(self, ragged):
        prices, filled = align_prices(ragged, 5)
        assert prices.loc[D[3], "B"] == 21.0 and prices.loc[D[4], "B"] == 21.0  # d4, d5 <- d3
        assert filled["B"].tolist() == [False, False, False, True, True] + [False] * 5

    def test_gap_longer_than_limit_stays_entirely_nan(self, ragged):
        prices, filled = align_prices(ragged, 3)  # C's gap is 4 days > 3
        assert prices["C"].iloc[2:6].isna().all()
        assert not filled["C"].any()

    def test_gap_equal_to_limit_is_filled_in_full(self, ragged):
        prices, filled = align_prices(ragged, 4)  # C's gap is 4 days == 4
        assert prices["C"].iloc[2:6].tolist() == [31.0] * 4
        assert filled["C"].iloc[2:6].all() and filled["C"].sum() == 4

    def test_trailing_short_run_is_filled(self, ragged):
        prices, filled = align_prices(ragged, 5)
        assert prices["Dd"].iloc[8:].tolist() == [48.0, 48.0]
        assert filled["Dd"].iloc[8:].all()

    def test_zero_limit_fills_nothing(self, ragged):
        prices, filled = align_prices(ragged, 0)
        assert prices["B"].isna().sum() == 3 and prices["C"].isna().sum() == 4
        assert not filled.to_numpy().any()

    def test_filled_mask_shape_and_dtype(self, ragged):
        prices, filled = align_prices(ragged, 5)
        assert_index_equal(filled.index, prices.index)
        assert list(filled.columns) == list(prices.columns)
        assert all(dt.kind == "b" for dt in filled.dtypes)


# ------------------------------------------------------------- F1.3 cleaned returns


class TestCleanReturns:
    def test_hand_computed_returns_with_nan_gap(self):
        # A: 10, 11, 11, 5.5, NaN, NaN, 8, 9
        # r:  -, .1,  0, -.5,   -,   -, - (P(t-1) NaN), (9-8)/8 = .125
        prices = frame({"A": [10, 11, 11, 5.5, None, None, 8, 9]}, D[:8])
        r, flagged = clean_returns(prices, 0.5)
        expected = frame({"A": [None, 0.1, 0.0, -0.5, None, None, None, 0.125]}, D[:8])
        assert_frame_equal(r, expected)
        assert flagged.empty  # -0.5 is inside [-0.5, 0.5]: not clipped, not flagged

    def test_integrity_negative_price_finite_clipped_flagged(self):
        # N: 10, 5, -2, 4, -1
        # raw: -, (5-10)/10 = -0.5, (-2-5)/5 = -1.4, (4+2)/|-2| = 3.0, (-1-4)/4 = -1.25
        # clipped at 0.5: -, -0.5, -0.5, 0.5, -0.5
        prices = frame({"N": [10, 5, -2, 4, -1]}, D[:5])
        r, flagged = clean_returns(prices, 0.5)
        assert r["N"].tolist()[1:] == [-0.5, -0.5, 0.5, -0.5]
        assert np.isfinite(r["N"].iloc[1:]).all()
        assert flagged["date"].tolist() == [D[2], D[3], D[4]]
        assert flagged["ticker"].tolist() == ["N", "N", "N"]
        assert flagged["raw_return"].tolist() == pytest.approx([-1.4, 3.0, -1.25])
        assert flagged["clipped_return"].tolist() == [-0.5, 0.5, -0.5]

    def test_flagged_log_columns_and_order(self):
        # Two tickers clipped on the same day: rows sorted by date, then ticker.
        prices = frame({"Z": [1, 3, 3], "M": [1, 1, 4]}, D[:3])  # Z: +200% on d2; M: +300% on d3
        _, flagged = clean_returns(prices, 0.5)
        assert list(flagged.columns) == ["date", "ticker", "raw_return", "clipped_return"]
        assert flagged["date"].tolist() == [D[1], D[2]]
        assert flagged["ticker"].tolist() == ["Z", "M"]
        assert flagged["raw_return"].tolist() == [2.0, 3.0]

    def test_empty_log_still_has_the_columns(self):
        _, flagged = clean_returns(frame({"A": [1, 1.1]}, D[:2]), 0.5)
        assert list(flagged.columns) == ["date", "ticker", "raw_return", "clipped_return"]
        assert len(flagged) == 0

    def test_clip_parameter_is_respected(self):
        prices = frame({"A": [10, 11, 5.5]}, D[:3])  # +10%, -50%
        r, flagged = clean_returns(prices, 0.2)
        assert r["A"].tolist()[1:] == [0.1, -0.2]
        assert len(flagged) == 1 and flagged.loc[0, "raw_return"] == -0.5

    def test_price_through_exactly_zero_stays_finite(self):
        # P(t-1) = 0 gives division by zero: the raw value is infinite, the clipped value
        # is +clip, and the row is flagged with the raw (infinite) value.
        prices = frame({"A": [1, 0, 2]}, D[:3])
        r, flagged = clean_returns(prices, 0.5)
        assert r["A"].tolist()[1:] == [-0.5, 0.5]  # (0-1)/1 = -1 -> -0.5; (2-0)/0 = inf -> 0.5
        assert np.isfinite(r["A"].iloc[1:]).all()
        assert len(flagged) == 2 and np.isinf(flagged.loc[1, "raw_return"])


class TestIntegrityGapFill:
    """§8: a gap of ffill_limit days is filled with zero returns; ffill_limit + 1 stays NaN."""

    LIMIT = 5
    CAL = pd.DatetimeIndex(list(pd.bdate_range("2020-01-01", periods=14)), name="date")

    def _panel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cal = self.CAL
        base = pd.Series(np.arange(1.0, 15.0), index=cal, name="BASE")  # defines the calendar
        five = pd.Series(10.0, index=cal.delete(range(2, 7)), name="G5")  # d3..d7 missing (5)
        six = pd.Series(10.0, index=cal.delete(range(2, 8)), name="G6")  # d3..d8 missing (6)
        prices, filled = align_prices({"BASE": base, "G5": five, "G6": six}, self.LIMIT)
        return prices, filled

    def test_gap_of_limit_days_is_filled_and_returns_are_zero(self):
        prices, filled = self._panel()
        r, _ = clean_returns(prices, 0.5)
        assert filled["G5"].sum() == 5
        assert r["G5"].iloc[2:7].tolist() == [0.0] * 5
        assert r["G5"].iloc[7] == 0.0  # d8 is a real observation at the same price

    def test_gap_of_limit_plus_one_days_stays_nan_through_resumption(self):
        prices, filled = self._panel()
        r, _ = clean_returns(prices, 0.5)
        assert filled["G6"].sum() == 0
        assert r["G6"].iloc[2:8].isna().all()  # the six gap days
        assert np.isnan(r["G6"].iloc[8])  # d9: data resumes but P(t-1) is NaN
        assert r["G6"].iloc[9] == 0.0  # d10: the first return after resumption


class TestCleanPriceIndex:
    def test_hand_computed_cumulative_product(self):
        # r:   -, .1, 0, -.5, -, .25
        # 1+r: -, 1.1, 1, .5, -, 1.25  ->  cumprod: -, 1.1, 1.1, .55, -, .6875
        r = frame({"A": [None, 0.1, 0.0, -0.5, None, 0.25]}, D[:6])
        idx = clean_price_index(r)
        expected = frame({"A": [None, 1.1, 1.1, 0.55, None, 0.6875]}, D[:6])
        assert_frame_equal(idx, expected)

    def test_gap_day_is_nan_and_the_level_carries_across_it(self):
        r = frame({"A": [None, 0.1, None, 0.1]}, D[:4])
        assert clean_price_index(r)["A"].tolist()[1:] == pytest.approx(
            [1.1, np.nan, 1.21], nan_ok=True
        )

    def test_shape_and_index_are_preserved(self):
        r = frame({"A": [None, 0.1], "B": [None, -0.1]}, D[:2])
        idx = clean_price_index(r)
        assert_index_equal(idx.index, r.index)
        assert list(idx.columns) == ["A", "B"]


# ------------------------------------------------ F1.4 quality gate and data report


GATE = QualityGateConfig(min_years=10, max_fill_fraction=0.05, max_gap_days=20)
CAL12 = pd.DatetimeIndex(list(pd.bdate_range("2010-01-01", "2021-12-31")), name="date")
# CAL12: business days 2010-01-01 .. 2021-12-31; the snapshot end is 2021-12-31


def long_series(ticker: str, index: pd.DatetimeIndex, value: float = 1.0) -> pd.Series:
    return pd.Series(value, index=pd.DatetimeIndex(index, name="date"), name=ticker, dtype=float)


def gate_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seven synthetic instruments exercising every gate rule (a helper, not a fixture, so
    that an unimplemented align_prices shows up as a test failure rather than an error)."""
    full = long_series("FULL", CAL12)
    # SHORT starts 2014-01-01: 2014-01-01 -> 2021-12-31 is 2557 + 364 = 2921 days
    # (7 years incl. leap 2016 and 2020 = 2557; plus Jan 1 -> Dec 31 2021 = 364);
    # 2921 / 365.25 = 7.997 -> "history 8.0y < 10y".
    short = long_series("SHORT", CAL12[CAL12 >= "2014-01-01"])
    # FILLS drops every 10th day (isolated 1-day gaps, all filled): fills = len(dropped);
    # span = whole calendar since the first and last days are present.
    dropped = CAL12[9::10]
    fills = long_series("FILLS", CAL12.drop(dropped))
    # GAP25 is missing 25 consecutive business days starting at position 1000.
    gap25 = long_series("GAP25", CAL12.delete(range(1000, 1025)))
    # GAP20 is missing exactly 20: allowed (the rule is "> 20").
    gap20 = long_series("GAP20", CAL12.delete(range(1000, 1020)))
    # STALE stops 30 business days before the end: a trailing gap of 30.
    stale = long_series("STALE", CAL12[:-30])
    # BOTH starts 2014 and has the 25-day gap: two reasons, history first.
    both = long_series("BOTH", CAL12[CAL12 >= "2014-01-01"].delete(range(1000, 1025)))
    raw = {s.name: s for s in (full, short, fills, gap25, gap20, stale, both)}
    return align_prices(raw, 5)


class TestQualityGate:
    def test_columns_and_index(self):
        gate_panel_ = gate_panel()
        gate = quality_gate(*gate_panel_, GATE)
        assert list(gate.index) == ["FULL", "SHORT", "FILLS", "GAP25", "GAP20", "STALE", "BOTH"]
        assert list(gate.columns) == [
            "first_date", "last_date", "rows", "n_filled", "fill_fraction",
            "longest_gap", "years", "passed", "reasons",
        ]  # fmt: skip

    def test_clean_series_passes_with_no_reasons(self):
        gate_panel_ = gate_panel()
        g = quality_gate(*gate_panel_, GATE).loc["FULL"]
        assert bool(g["passed"]) and g["reasons"] == ""
        assert g["first_date"] == CAL12[0] and g["last_date"] == CAL12[-1]
        assert g["rows"] == len(CAL12) and g["n_filled"] == 0 and g["fill_fraction"] == 0.0
        assert g["longest_gap"] == 0
        # (2021-12-31 - 2010-01-01) = 4018 + 364 = 4382 days -> 4382 / 365.25 = 11.997
        assert g["years"] == pytest.approx(4382 / 365.25)

    def test_short_history_fails_with_reason(self):
        gate_panel_ = gate_panel()
        g = quality_gate(*gate_panel_, GATE).loc["SHORT"]
        assert not g["passed"]
        assert g["years"] == pytest.approx(2921 / 365.25)
        assert g["reasons"] == "history 8.0y < 10y"

    def test_too_many_fills_fails_with_reason(self):
        gate_panel_ = gate_panel()
        g = quality_gate(*gate_panel_, GATE).loc["FILLS"]
        n_dropped = len(CAL12[9::10])
        assert g["n_filled"] == n_dropped
        assert g["rows"] == len(CAL12) - n_dropped
        assert g["fill_fraction"] == pytest.approx(n_dropped / len(CAL12))
        assert g["reasons"] == f"filled {n_dropped / len(CAL12):.1%} > 5%"
        assert not g["passed"]

    def test_long_gap_fails_and_gap_at_limit_passes(self):
        gate_panel_ = gate_panel()
        gate = quality_gate(*gate_panel_, GATE)
        assert gate.loc["GAP25", "longest_gap"] == 25
        assert gate.loc["GAP25", "reasons"] == "gap 25d > 20d"
        assert gate.loc["GAP20", "longest_gap"] == 20
        assert bool(gate.loc["GAP20", "passed"]) and gate.loc["GAP20", "reasons"] == ""

    def test_trailing_gap_counts(self):
        gate_panel_ = gate_panel()
        g = quality_gate(*gate_panel_, GATE).loc["STALE"]
        assert g["longest_gap"] == 30 and g["last_date"] == CAL12[-31]
        assert g["reasons"] == "gap 30d > 20d"

    def test_multiple_reasons_are_joined_in_order(self):
        gate_panel_ = gate_panel()
        g = quality_gate(*gate_panel_, GATE).loc["BOTH"]
        assert g["reasons"] == "history 8.0y < 10y; gap 25d > 20d"

    def test_years_measured_to_snapshot_end_not_last_valid(self):
        gate_panel_ = gate_panel()
        # STALE ends early but its history is measured to the calendar's last day.
        g = quality_gate(*gate_panel_, GATE).loc["STALE"]
        assert g["years"] == pytest.approx(4382 / 365.25)

    def test_rows_exclude_filled_days(self):
        raw = {"BASE": series("BASE", {k: 1.0 for k in range(1, 5)})}  # defines d1..d4
        raw["A"] = series("A", {1: 1.0, 2: 2.0, 4: 4.0})  # d3 missing, filled
        prices, filled = align_prices(raw, 5)
        g = quality_gate(prices, filled, QualityGateConfig(0, 1.0, 100)).loc["A"]
        assert g["rows"] == 3 and g["n_filled"] == 1
        assert g["fill_fraction"] == pytest.approx(1 / 4)  # span d1..d4 = 4 calendar days


class TestUniverseDict:
    def test_lists_passers_and_excluded_with_reasons(self):
        gate_panel_ = gate_panel()
        gate = quality_gate(*gate_panel_, GATE)
        u = universe_dict(gate)
        assert u["universe"] == ["FULL", "GAP20"]
        assert u["excluded"] == {
            "SHORT": "history 8.0y < 10y",
            "FILLS": gate.loc["FILLS", "reasons"],
            "GAP25": "gap 25d > 20d",
            "STALE": "gap 30d > 20d",
            "BOTH": "history 8.0y < 10y; gap 25d > 20d",
        }
        assert u["n_candidates"] == 7


class TestDataQualityReport:
    def test_report_mentions_every_instrument_verdict_and_clip(self):
        gate_panel_ = gate_panel()
        prices, filled = gate_panel_
        gate = quality_gate(prices, filled, GATE)
        flagged = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-04-20")],
                "ticker": ["FULL"],
                "raw_return": [-3.06],
                "clipped_return": [-0.5],
            }
        )
        md = data_quality_report(gate, flagged, prices)
        assert md.startswith("# Data quality report")
        assert "2010-01-01" in md and "2021-12-31" in md and str(len(CAL12)) in md
        for t in gate.index:
            assert f"| {t} |" in md
        assert "| FULL |" in md and "pass" in md.lower() and "fail" in md.lower()
        assert "history 8.0y < 10y" in md and "gap 25d > 20d" in md
        assert "2020-04-20" in md and "-3.06" in md
        assert "FULL, GAP20" in md  # the passing universe, listed

    def test_report_with_no_clipped_days_says_so(self):
        gate_panel_ = gate_panel()
        prices, filled = gate_panel_
        gate = quality_gate(prices, filled, GATE)
        empty = pd.DataFrame(columns=["date", "ticker", "raw_return", "clipped_return"])
        md = data_quality_report(gate, empty, prices)
        assert "no clipped" in md.lower()


# ------------------------------------------------------ F1.6 external benchmark


class TestFetchBenchmark:
    def test_first_available_candidate_wins(self):
        src = DictSource({"DBC": series("DBC", {1: 1.0})})
        ticker, s = fetch_benchmark(src, ("^BCOM", "DBC"))
        assert ticker == "DBC" and s.tolist() == [1.0]
        assert src.calls == ["^BCOM", "DBC"]

    def test_first_candidate_is_used_when_it_loads(self):
        src = DictSource({"^BCOM": series("^BCOM", {1: 2.0}), "DBC": series("DBC", {1: 1.0})})
        assert fetch_benchmark(src, ("^BCOM", "DBC"))[0] == "^BCOM"
        assert src.calls == ["^BCOM"]

    def test_all_candidates_failing_raises_naming_them(self):
        with pytest.raises(FetchError, match=r"\^BCOM.*DBC"):
            fetch_benchmark(DictSource({}), ("^BCOM", "DBC"))


# ------------------------------------------- F1.5 snapshot manifest and panel build


def make_root(tmp_path: Path) -> Path:
    """A repo-like root with raw CSVs, a panel and a benchmark for two tickers."""
    (tmp_path / "data").mkdir()
    write_raw(series("A", {1: 1.0, 2: 2.0, 3: 3.0}), tmp_path / "data" / "raw")
    write_raw(series("B", {2: 5.0, 3: 6.0}), tmp_path / "data" / "raw")
    panel = frame({"A": [1.0, 2.0, 3.0], "B": [None, 5.0, 6.0]}, D[:3])
    panel.to_csv(tmp_path / "data" / "futures_panel.csv")
    series("DBC", {1: 9.0, 2: 9.5}).to_frame().to_csv(tmp_path / "data" / "benchmark.csv")
    return tmp_path


AB = load_config(overrides={"data.tickers": ["A", "B"]}).data  # the two tickers of make_root


class TestManifest:
    def test_sha256_of_matches_hashlib(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("abc")
        assert sha256_of(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_build_records_rows_dates_and_hashes(self, tmp_path):
        root = make_root(tmp_path)
        m = build_manifest(AB, root, benchmark_ticker="DBC", pulled_at="2026-09-04T00:00:00Z",
                           yfinance_version="1.7.0")  # fmt: skip
        assert m["pulled_at"] == "2026-09-04T00:00:00Z" and m["yfinance_version"] == "1.7.0"
        assert m["price_field"] == "Adj Close"
        a = m["tickers"]["A"]
        assert a["path"] == "data/raw/A.csv" and a["rows"] == 3
        assert a["first"] == "2020-01-01" and a["last"] == "2020-01-03"
        assert a["sha256"] == sha256_of(root / "data" / "raw" / "A.csv")
        assert m["tickers"]["B"]["rows"] == 2 and m["tickers"]["B"]["first"] == "2020-01-02"
        p = m["panel"]
        assert p["path"] == "data/futures_panel.csv" and p["rows"] == 3
        assert (
            p["columns"] == ["A", "B"] and p["first"] == "2020-01-01" and p["last"] == "2020-01-03"
        )
        assert p["sha256"] == sha256_of(root / "data" / "futures_panel.csv")
        b = m["benchmark"]
        assert b["ticker"] == "DBC" and b["path"] == "data/benchmark.csv" and b["rows"] == 2
        assert b["sha256"] == sha256_of(root / "data" / "benchmark.csv")

    def test_build_without_benchmark(self, tmp_path):
        root = make_root(tmp_path)
        m = build_manifest(AB, root, None, "t", "v")
        assert m["benchmark"] is None

    def test_verify_passes_then_detects_a_changed_and_a_missing_file(self, tmp_path):
        root = make_root(tmp_path)
        m = build_manifest(AB, root, "DBC", "t", "v")
        assert verify_manifest(m, root) == []
        (root / "data" / "raw" / "A.csv").write_text("date,A\n2020-01-01,1.0\n")
        (root / "data" / "benchmark.csv").unlink()
        problems = verify_manifest(m, root)
        assert len(problems) == 2
        assert any("data/raw/A.csv" in p for p in problems)
        assert any("data/benchmark.csv" in p for p in problems)


class TestRunPanel:
    def _cfg(self):
        return load_config(
            overrides={
                "data.tickers": ["FULL", "SHORT", "GAP25"],
                "data.raw_dir": "raw",
                "data.panel_path": "panel.csv",
                "paths.results_dir": "out",
            }
        )

    def _root(self, tmp_path: Path) -> Path:
        cal = CAL12
        write_raw(long_series("FULL", cal, 100.0), tmp_path / "raw")
        write_raw(long_series("SHORT", cal[cal >= "2014-01-01"], 50.0), tmp_path / "raw")
        write_raw(long_series("GAP25", cal.delete(range(1000, 1025)), 20.0), tmp_path / "raw")
        return tmp_path

    def test_writes_panel_of_passers_report_and_universe(self, tmp_path):
        root = self._root(tmp_path)
        build = run_panel(self._cfg(), root)
        assert isinstance(build, PanelBuild)
        assert build.universe == ["FULL"]
        assert list(build.prices.columns) == ["FULL", "SHORT", "GAP25"]  # all candidates kept
        panel = pd.read_csv(root / "panel.csv", index_col="date", parse_dates=True)
        assert list(panel.columns) == ["FULL"]  # only the gate's passers are written
        assert len(panel) == len(CAL12)
        u = json.loads((root / "out" / "universe.json").read_text())
        assert u["universe"] == ["FULL"] and set(u["excluded"]) == {"SHORT", "GAP25"}
        assert u["snapshot_end"] == "2021-12-31"
        md = (root / "out" / "data_quality.md").read_text()
        assert "| GAP25 |" in md and "gap 25d > 20d" in md

    def test_build_carries_every_intermediate(self, tmp_path):
        build = run_panel(self._cfg(), self._root(tmp_path))
        assert_index_equal(build.returns.index, build.prices.index)
        assert list(build.filled.columns) == list(build.prices.columns)
        assert list(build.flagged.columns) == ["date", "ticker", "raw_return", "clipped_return"]
        assert list(build.gate.index) == ["FULL", "SHORT", "GAP25"]


# ------------------------------------------------------- the committed snapshot


class TestSnapshot:
    """Real-data checks on the committed snapshot in ``data/``. No network."""

    cfg = load_config()

    @property
    def manifest(self) -> dict:
        return json.loads((ROOT / self.cfg.data.manifest_path).read_text())

    def test_manifest_hashes_match_the_committed_files(self):
        assert verify_manifest(self.manifest, ROOT) == []

    def test_committed_panel_is_reproduced_from_the_raw_files(self):
        raw = load_raw(ROOT / self.cfg.data.raw_dir, self.cfg.data.tickers)
        prices, filled = align_prices(raw, self.cfg.data.ffill_limit)
        gate = quality_gate(prices, filled, self.cfg.quality_gate)
        rebuilt = prices[universe_dict(gate)["universe"]].dropna(how="all")
        committed = pd.read_csv(ROOT / self.cfg.data.panel_path, index_col="date", parse_dates=True)
        assert_frame_equal(rebuilt, committed, check_exact=True)
        assert list(committed.columns) == self.manifest["panel"]["columns"]

    def test_wti_negative_settlement_days_are_clipped_and_flagged(self):
        raw = load_raw(ROOT / self.cfg.data.raw_dir, ("CL=F",))
        prices, _ = align_prices(raw, self.cfg.data.ffill_limit)
        _, flagged = clean_returns(prices, self.cfg.data.clip)
        by_day = flagged.set_index("date")
        assert by_day.loc[pd.Timestamp("2020-04-20"), "raw_return"] < -0.5
        assert by_day.loc[pd.Timestamp("2020-04-21"), "raw_return"] > 0.5
        assert by_day.loc[pd.Timestamp("2020-04-20"), "clipped_return"] == -0.5
        assert by_day.loc[pd.Timestamp("2020-04-21"), "clipped_return"] == 0.5

    def test_panel_covers_the_expected_range(self):
        committed = pd.read_csv(ROOT / self.cfg.data.panel_path, index_col="date", parse_dates=True)
        assert len(committed) == self.manifest["panel"]["rows"]
        assert committed.index[0].year <= 2001
        assert committed.index.is_monotonic_increasing and committed.index.is_unique
