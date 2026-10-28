"""Data layer: price sources, alignment and gap handling, cleaned returns, quality gate,
data-quality report, snapshot manifest, external benchmark (PROJECT_LOG.md §3, §4.3).

Conventions used throughout ``src/``:

* A **price series** is a ``pd.Series`` of floats named by its ticker, indexed by a tz-naive
  daily ``DatetimeIndex`` named ``"date"``, sorted, no NaNs, no duplicate dates.
* A **wide frame** is a ``pd.DataFrame`` indexed by such a ``DatetimeIndex`` with one column
  per ticker. Prices and returns are always wide frames.
* This module is the only place (with ``cli.py``, ``plots.py`` and ``report.py``) that does
  file or network I/O. Everything between ``load_raw`` and the writers is a pure function.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.config import Config, DataConfig, QualityGateConfig

FLAGGED_COLUMNS = ("date", "ticker", "raw_return", "clipped_return")
"""Columns of the clipped-returns log, in order."""

GATE_COLUMNS = (
    "first_date",
    "last_date",
    "rows",
    "n_filled",
    "fill_fraction",
    "longest_gap",
    "years",
    "passed",
    "reasons",
)
"""Columns of the quality-gate table, in order."""

DAYS_PER_YEAR = 365.25
"""Calendar days per year, used to turn a date span into years of history."""


class FetchError(RuntimeError):
    """A price source could not deliver a ticker. The message names the ticker."""


class PriceSource(Protocol):
    """Anything that can deliver one ticker's daily price series."""

    def fetch(self, ticker: str) -> pd.Series:
        """Return the ticker's price series (see module conventions) or raise FetchError."""
        ...


# ------------------------------------------------------------------ F1.1 fetch layer


def extract_price_field(frame: pd.DataFrame, ticker: str, price_field: str) -> pd.Series:
    """Pull one price column out of a ``yfinance.download`` result and normalise it.

    Parameters
    ----------
    frame:
        What ``yfinance.download(ticker, auto_adjust=False)`` returns: either MultiIndex
        columns ``(field, ticker)`` (current yfinance) or flat columns ``field`` (older
        versions and single-ticker frames). May be empty.
    ticker:
        Used to name the output and to select the column when the frame is MultiIndexed.
    price_field:
        The field to prefer (config ``data.price_field``, normally ``"Adj Close"``). When it
        is absent, fall back to ``"Close"``.

    Returns
    -------
    A price series in module convention: float dtype, name = ``ticker``, tz-naive daily
    ``DatetimeIndex`` named ``"date"`` (a tz-aware index is converted by dropping the zone,
    keeping the local calendar date), sorted, NaN rows dropped, duplicate dates keep the last.

    Raises
    ------
    FetchError
        If the frame is empty, if neither field is present, or if nothing is left after
        dropping NaNs. The message contains ``ticker``.
    """
    # TODO 1: implement extract_price_field.
    #   1. Empty frame (or None) -> raise FetchError(f"{ticker}: ...").
    #   2. Find the column. If ``frame.columns`` is a ``pd.MultiIndex``, level 0 holds the
    #      field and level 1 the ticker: check ``price_field in columns.get_level_values(0)``,
    #      then ``frame[price_field]`` gives a one-column frame -> pick ``[ticker]`` if present
    #      else its first column. Flat columns: ``frame[price_field]`` if present. Try
    #      ``price_field`` first, then "Close"; neither -> FetchError naming the ticker.
    #   3. ``pd.to_numeric(..., errors="coerce").dropna().astype(float)``; empty -> FetchError.
    #   4. Index: ``pd.DatetimeIndex(s.index)``; if ``idx.tz is not None`` use
    #      ``idx.tz_localize(None)`` (keeps the wall-clock date); then ``.normalize()``. Build
    #      the final index as ``pd.DatetimeIndex(idx.to_numpy(), name="date")`` so no
    #      frequency is attached. Drop duplicate dates (``~s.index.duplicated(keep="last")``),
    #      ``sort_index()``, set ``s.name = ticker``.
    #   Verified by tests/test_data.py::TestExtractPriceField.
    if frame is None or frame.empty:
        raise FetchError(f"{ticker}: no data returned")

    # Prefer the configured field, fall back to "Close"; cope with either column shape.
    col: pd.Series | None = None
    for field in (price_field, "Close"):
        if isinstance(frame.columns, pd.MultiIndex):
            if field in frame.columns.get_level_values(0):
                sub = frame[field]
                col = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
        elif field in frame.columns:
            col = frame[field]
        if col is not None:
            break
    if col is None:
        raise FetchError(f"{ticker}: no '{price_field}' or 'Close' column in the download")

    prices = pd.to_numeric(col, errors="coerce").dropna().astype(float)
    if prices.empty:
        raise FetchError(f"{ticker}: no usable prices after dropping NaNs")

    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # drop the zone, keep the wall-clock calendar date
    idx = idx.normalize()
    # Rebuild from raw values so no inferred ``freq`` is attached to the index.
    prices.index = pd.DatetimeIndex(idx.to_numpy(), name="date")
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices.name = ticker
    return prices


class YFinanceSource:
    """Daily prices from Yahoo Finance via ``yfinance.download``, with retries.

    Parameters
    ----------
    price_field:
        Passed to :func:`extract_price_field`.
    retries:
        Total attempts per ticker (config ``data.fetch_retries``); at least 1.
    start:
        Optional ``YYYY-MM-DD`` start date forwarded to the downloader; ``None`` means the
        maximum available history.
    downloader, sleep:
        Injection points for tests. ``downloader(ticker, **kwargs)`` must return a frame;
        it is called with ``auto_adjust=False, progress=False`` plus either ``start=<start>``
        or, when ``start`` is None, ``period="max"`` (yfinance's own default window is one
        month, which is not what a backtest wants). ``sleep(seconds)`` is called between
        attempts. Defaults are ``yfinance.download`` and ``time.sleep``.
    """

    def __init__(
        self,
        price_field: str,
        retries: int,
        start: str | None = None,
        downloader: Callable[..., pd.DataFrame] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.price_field = price_field
        self.retries = max(1, int(retries))
        self.start = start
        self._download = downloader
        self._sleep = sleep if sleep is not None else time.sleep

    def fetch(self, ticker: str) -> pd.Series:
        """Download ``ticker`` and return its price series.

        Tries up to ``retries`` times. An exception from the downloader or a result that
        :func:`extract_price_field` rejects (empty, no price column) both count as a failed
        attempt; ``sleep`` is called between attempts (a short, growing pause). After the
        last failed attempt raise ``FetchError`` naming the ticker and the last error.
        """
        # TODO 2: implement YFinanceSource.fetch.
        #   - ``download = self._download if self._download is not None else _yfinance_download``.
        #   - Loop ``attempt`` from 1 to ``self.retries``: call
        #     ``download(ticker, **self._download_kwargs())`` (the helper below builds the
        #     right keyword arguments; the tests check them), then return
        #     ``extract_price_field(frame, ticker, self.price_field)``.
        #   - Catch ``Exception`` (network errors and FetchError alike), remember it, and if
        #     this was not the last attempt call ``self._sleep(...)`` with a small number of
        #     seconds (e.g. ``2.0 * attempt``).
        #   - After the loop: ``raise FetchError(f"{ticker}: failed after ... : {last_error}")``.
        download = self._download if self._download is not None else _yfinance_download

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                frame = download(ticker, **self._download_kwargs())
                return extract_price_field(frame, ticker, self.price_field)
            except Exception as exc:  # network errors and FetchError alike
                last_error = exc
                if attempt < self.retries:
                    self._sleep(2.0 * attempt)

        raise FetchError(f"{ticker}: failed after {self.retries} attempt(s): {last_error}")

    def _download_kwargs(self) -> dict[str, Any]:
        """``start=`` when configured, else ``period="max"`` (yfinance's default is one month)."""
        window = {"start": self.start} if self.start else {"period": "max"}
        return {**window, "auto_adjust": False, "progress": False}


def _yfinance_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the module loads without network libraries

    return yf.download(ticker, **kwargs)


class CSVSource:
    """Prices from one wide CSV (``date`` column plus one column per ticker).

    This is the drop-in for a vendor's back-adjusted continuous series (config
    ``data.source: csv`` with ``data.csv_path``); the rest of the pipeline is unchanged.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, ticker: str) -> pd.Series:
        """Return the ``ticker`` column as a price series (NaN rows dropped).

        Raises ``FetchError`` naming the ticker when the column does not exist.
        """
        # TODO 3: implement CSVSource.fetch.
        #   ``pd.read_csv(self.path, index_col="date", parse_dates=True)``; missing column ->
        #   FetchError naming the ticker; otherwise ``pd.to_numeric(..., errors="coerce")``,
        #   ``.dropna().astype(float)``, index -> ``pd.DatetimeIndex(...).normalize()`` named
        #   "date", ``s.name = ticker``, ``sort_index()``. Compare with ``read_raw`` below.
        #   Verified by tests/test_data.py::TestCSVSource.
        wide = pd.read_csv(self.path, index_col="date", parse_dates=True)
        if ticker not in wide.columns:
            raise FetchError(f"{ticker}: no such column in {self.path}")

        # Same shape as ``read_raw``, plus coercion: this is a vendor's file, not one we wrote.
        s = pd.to_numeric(wide[ticker], errors="coerce").dropna().astype(float)
        s.index = pd.DatetimeIndex(s.index).normalize()
        s.index.name = "date"
        s.name = ticker
        return s.sort_index()


def write_raw(prices: pd.Series, raw_dir: str | Path) -> Path:
    """Write one price series to ``raw_dir/<ticker>.csv`` (columns ``date,<ticker>``)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{prices.name}.csv"
    prices.to_frame().to_csv(path, index_label="date")
    return path


def read_raw(raw_dir: str | Path, ticker: str) -> pd.Series:
    """Read ``raw_dir/<ticker>.csv`` back into a price series."""
    wide = pd.read_csv(Path(raw_dir) / f"{ticker}.csv", index_col="date", parse_dates=True)
    s = wide[ticker].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    s.index.name = "date"
    s.name = ticker
    return s


def load_raw(raw_dir: str | Path, tickers: Iterable[str]) -> dict[str, pd.Series]:
    """Read every ticker's raw CSV, in the given order."""
    return {t: read_raw(raw_dir, t) for t in tickers}


def fetch_all(
    source: PriceSource, tickers: Iterable[str], raw_dir: str | Path
) -> dict[str, pd.Series]:
    """Fetch every ticker from ``source`` and write each to ``raw_dir`` as it arrives.

    Returns ``{ticker: series}`` in the order given. Every ticker is attempted even after
    one fails, so a single run reports all problems; at the end, if any ticker failed,
    raise ``FetchError`` whose message lists the failed tickers and their errors. The
    tickers that succeeded are already on disk by then.
    """
    # TODO 4: implement fetch_all.
    #   Loop over the tickers; ``try: source.fetch(t)`` / ``except FetchError as e``: record
    #   ``str(e)`` and continue. On success ``write_raw(series, raw_dir)`` and store it in
    #   the result dict. After the loop, if anything failed raise one FetchError whose
    #   message joins the recorded errors (each already names its ticker).
    #   Verified by tests/test_data.py::TestFetchAll.
    result: dict[str, pd.Series] = {}
    errors: list[str] = []
    for t in tickers:
        try:
            series = source.fetch(t)
        except FetchError as e:  # a bad ticker is handled here; anything else is a bug
            errors.append(str(e))
            continue
        write_raw(series, raw_dir)  # on disk as it arrives, so a later failure loses nothing
        result[t] = series

    if errors:
        raise FetchError(f"{len(errors)} ticker(s) failed: " + "; ".join(errors))
    return result


# ------------------------------------------------- F1.2 alignment and gap handling


def align_prices(raw: dict[str, pd.Series], ffill_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put every price series on the union calendar and fill short gaps.

    Parameters
    ----------
    raw:
        ``{ticker: price series}``; column order of the output follows dict order.
    ffill_limit:
        A run of consecutive missing days is forward-filled with the last price **only if
        the whole run is at most this long**; longer runs stay NaN in full (all-or-nothing,
        PROJECT_LOG.md §3.2). ``0`` fills nothing.

    Returns
    -------
    ``(prices, filled)``, two wide frames with identical index and columns:

    * ``prices``: index = sorted union of every series' dates, tz-naive, named ``"date"``;
      NaN wherever a ticker has no price on a calendar day and the gap was not filled.
      Missing days before a ticker's first observation are never filled. A run at the end
      of the calendar (after the last observation) is treated like any other run.
    * ``filled``: boolean, True exactly where ``prices`` holds a forward-filled value.
    """
    # TODO 5: implement align_prices.
    #   1. Calendar: the sorted union of the (NaN-dropped) series' indexes, as a
    #      ``pd.DatetimeIndex`` named "date".
    #   2. ``prices = pd.DataFrame({t: s.reindex(calendar) ...}, index=calendar).astype(float)``
    #      and ``filled = pd.DataFrame(False, index=calendar, columns=prices.columns)``.
    #   3. Per column: ``missing = col.isna()``; ``first = col.first_valid_index()`` (skip the
    #      column if None). The length of the missing run each day belongs to:
    #      ``run_len = missing.groupby(col.notna().cumsum()).transform("sum")`` (every valid
    #      day starts a new group, so the missing days after it share one group whose sum is
    #      the run length). Fill where ``missing & (calendar > first) & (run_len <= limit)``
    #      using ``col.ffill()`` at those positions; write the same mask into ``filled``.
    #   Note: pandas' own ``ffill(limit=n)`` fills the first n days of a longer run, which is
    #   NOT the rule here (see the "gap longer than limit stays entirely NaN" test).
    #   Verified by tests/test_data.py::TestAlignPrices and ::TestIntegrityGapFill.
    # 1. Union calendar: every date any series has a price on, sorted, no freq attached.
    dates = pd.DatetimeIndex([])
    for s in raw.values():
        dates = dates.union(s.dropna().index)
    calendar = pd.DatetimeIndex(dates.to_numpy(), name="date")

    # 2. Every series on that calendar (NaN where it has no price) plus an all-False mask.
    prices = pd.DataFrame({t: s.reindex(calendar) for t, s in raw.items()}, index=calendar)
    prices = prices.astype(float)
    filled = pd.DataFrame(False, index=calendar, columns=prices.columns)

    # 3. Per column: measure each missing run, then fill whole runs that are short enough.
    for t in prices.columns:
        col = prices[t]
        first = col.first_valid_index()
        if first is None:  # a column with no prices at all: nothing to fill
            continue
        missing = col.isna()
        # Each real day starts a new group; the missing days after it inherit its id, so the
        # group's count of missing days is the length of the run every one of them belongs to.
        run_len = missing.groupby(col.notna().cumsum()).transform("sum")
        fill = missing & (calendar > first) & (run_len <= ffill_limit)
        prices.loc[fill, t] = col.ffill()[fill]
        filled.loc[fill, t] = True

    return prices, filled


# ------------------------------------------------------------- F1.3 cleaned returns


def clean_returns(prices: pd.DataFrame, clip: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily returns that survive a price through zero, clipped, with a log of every clip.

    ``r(t) = (P(t) - P(t-1)) / |P(t-1)|`` (PROJECT_LOG.md §3.3), then clipped to
    ``[-clip, +clip]``. A forward-filled day has ``P(t) == P(t-1)`` and so a return of 0.
    Wherever ``P(t)`` or ``P(t-1)`` is NaN the return is NaN (so the first day of every
    series, every unfilled gap day, and the first day after a gap are NaN). ``P(t-1) == 0``
    gives an infinite raw value, which clips to ``±clip``.

    Returns
    -------
    ``(returns, flagged)``:

    * ``returns``: wide frame, same index and columns as ``prices``, clipped, every non-NaN
      value finite.
    * ``flagged``: one row per clipped observation (``|raw| > clip``; a raw value exactly at
      the bound is not clipped), columns exactly ``FLAGGED_COLUMNS`` =
      ``("date", "ticker", "raw_return", "clipped_return")``, sorted by date then ticker,
      integer index from 0. Empty (but with those columns) when nothing was clipped.
    """
    # TODO 6: implement clean_returns.
    #   ``prev = prices.shift(1)``; ``raw = (prices - prev) / prev.abs()`` (NOT pct_change:
    #   it breaks on the negative WTI settlement); ``returns = raw.clip(-clip, clip)``.
    #   Log: ``mask = (raw.abs() > clip).fillna(False)``; the row/column positions of the
    #   True cells (``np.nonzero(mask.to_numpy())``) give the dates, tickers, raw and
    #   clipped values. Build the frame with the four columns, ``sort_values(["date",
    #   "ticker"])`` and ``reset_index(drop=True)``. Make the "date" column a proper
    #   datetime column even when empty (``pd.DatetimeIndex(prices.index[rows])``).
    #   Verified by tests/test_data.py::TestCleanReturns, ::TestIntegrityGapFill and
    #   ::TestSnapshot (the two WTI days in April 2020 must be flagged).
    # 1. Yesterday's price on today's row, then the dollar move over the absolute base.
    #    The absolute value is what keeps the sign right through the negative WTI print.
    prev = prices.shift(1)
    raw = (prices - prev) / prev.abs()
    returns = raw.clip(-clip, clip)

    # 2. Which cells were clipped: strictly beyond the bound (a value exactly at it is not).
    #    NaN compares False, so gap days are never flagged; inf compares True, so a zero
    #    previous price is.
    mask = raw.abs() > clip
    rows, cols = mask.to_numpy().nonzero()

    # 3. One log row per clipped cell, in the fixed four-column order, sorted by date then
    #    ticker. Indexing by position keeps the frame valid (and typed) when nothing clipped.
    flagged = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(prices.index[rows]),
            "ticker": prices.columns[cols],
            "raw_return": raw.to_numpy()[rows, cols],
            "clipped_return": returns.to_numpy()[rows, cols],
        }
    )
    flagged = flagged.sort_values(["date", "ticker"]).reset_index(drop=True)
    return returns, flagged


def clean_price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative price index rebuilt from cleaned returns: ``prod(1 + r)`` per column.

    NaN returns leave a NaN in the index on that day and do not reset the level: the
    product carries on from the last non-NaN value (pandas ``cumprod`` semantics). The
    first day of each series is therefore NaN and the level is 1 + r on the second day.
    Same index and columns as ``returns``.
    """
    # TODO 7: implement clean_price_index. One expression; ``DataFrame.cumprod`` already
    #   skips NaNs the way the docstring describes.
    #   Verified by tests/test_data.py::TestCleanPriceIndex.
    return (1 + returns).cumprod()


# ------------------------------------------------ F1.4 quality gate and data report


def quality_gate(
    prices: pd.DataFrame, filled: pd.DataFrame, gate: QualityGateConfig
) -> pd.DataFrame:
    """Per-instrument data statistics and the pass/fail verdict of PROJECT_LOG.md §3.4.

    Parameters
    ----------
    prices, filled:
        The two frames from :func:`align_prices`.
    gate:
        Thresholds: ``min_years``, ``max_fill_fraction``, ``max_gap_days``.

    Returns
    -------
    A frame indexed by ticker (in column order of ``prices``) with columns exactly
    ``GATE_COLUMNS``:

    * ``first_date``, ``last_date``: first and last day with a real (not filled) price.
    * ``rows``: number of real observations (filled days excluded).
    * ``n_filled``: number of forward-filled days.
    * ``fill_fraction``: ``n_filled / span`` where ``span`` is the number of calendar days
      from ``first_date`` to ``last_date`` inclusive.
    * ``longest_gap``: longest run of consecutive calendar days without a real price after
      ``first_date``, in calendar days, counting a run that reaches the end of the calendar.
      Filled days count as missing here.
    * ``years``: ``(calendar end - first_date).days / 365.25``, history at snapshot date.
    * ``passed``: ``years >= min_years`` and ``fill_fraction <= max_fill_fraction`` and
      ``longest_gap <= max_gap_days``.
    * ``reasons``: ``""`` when passed; otherwise the failing checks joined by ``"; "``, in
      this order and format: ``"history {years:.1f}y < {min_years:g}y"``,
      ``"filled {fill_fraction:.1%} > {max_fill_fraction:.0%}"``,
      ``"gap {longest_gap}d > {max_gap_days}d"``.

    A ticker with no real price at all gets ``rows = 0``, NaT dates, ``passed = False`` and
    ``reasons = "no data"``.
    """
    # TODO 8: implement quality_gate.
    #   Per column: ``real = prices[t].notna() & ~filled[t]``. ``first = real.idxmax()``,
    #   ``last = real[::-1].idxmax()`` (idxmax on a boolean Series is the first True).
    #   ``span = ((calendar >= first) & (calendar <= last)).sum()``.
    #   Longest gap: ``missing_after = ~real & (calendar > first)``; walk the boolean array
    #   keeping a running count that resets on False and track the maximum (a small helper
    #   is fine). ``years = (calendar[-1] - first).days / DAYS_PER_YEAR``.
    #   Build ``reasons`` as a list in the documented order and join with "; ".
    #   Assemble with ``pd.DataFrame.from_records(records, columns=list(GATE_COLUMNS),
    #   index=list(prices.columns))`` and name the index "ticker".
    #   Verified by tests/test_data.py::TestQualityGate (and used by the report tests).
    calendar = prices.index
    records = []
    for t in prices.columns:
        # 1. A "real" day has a price that was not forward-filled. Everything below hangs
        #    off this mask: a filled day is neither an observation nor a gap-breaker.
        real = prices[t].notna() & ~filled[t]
        n_filled = int(filled[t].sum())
        if not real.any():
            nan = float("nan")
            records.append((pd.NaT, pd.NaT, 0, n_filled, nan, 0, nan, False, "no data"))
            continue

        # 2. First and last real day. idxmax on a boolean Series returns the label of the
        #    first True; reversing the Series makes the same trick find the last one.
        first = real.idxmax()
        last = real[::-1].idxmax()
        rows = int(real.sum())

        # 3. Fill share over the span of calendar rows between first and last, inclusive.
        span = int(((calendar >= first) & (calendar <= last)).sum())
        fill_fraction = n_filled / span

        # 4. Longest run of calendar rows without a real price after the first one. A run
        #    that reaches the end of the calendar counts (a stale series fails here).
        missing_after = (~real & (calendar > first)).to_numpy()
        longest_gap = run = 0
        for is_missing in missing_after:
            run = run + 1 if is_missing else 0
            longest_gap = max(longest_gap, run)

        # 5. History measured to the snapshot date, not to the series' own last day.
        years = (calendar[-1] - first).days / DAYS_PER_YEAR

        # 6. Verdict: each rule adds one phrase, in the documented order; empty means pass.
        reasons = []
        if years < gate.min_years:
            reasons.append(f"history {years:.1f}y < {gate.min_years:g}y")
        if fill_fraction > gate.max_fill_fraction:
            reasons.append(f"filled {fill_fraction:.1%} > {gate.max_fill_fraction:.0%}")
        if longest_gap > gate.max_gap_days:
            reasons.append(f"gap {longest_gap}d > {gate.max_gap_days}d")
        passed = not reasons
        reason_text = "; ".join(reasons)
        records.append(
            (first, last, rows, n_filled, fill_fraction, longest_gap, years, passed, reason_text)
        )

    out = pd.DataFrame.from_records(records, columns=list(GATE_COLUMNS), index=list(prices.columns))
    out.index.name = "ticker"
    return out


def universe_dict(gate: pd.DataFrame) -> dict[str, Any]:
    """The JSON-ready universe record written to ``results/universe.json``."""
    passed = gate[gate["passed"].astype(bool)]
    failed = gate[~gate["passed"].astype(bool)]
    last = gate["last_date"].max()
    return {
        "universe": list(passed.index),
        "excluded": {t: r for t, r in failed["reasons"].items()},
        "n_candidates": int(len(gate)),
        "snapshot_end": None if pd.isna(last) else pd.Timestamp(last).strftime("%Y-%m-%d"),
    }


def data_quality_report(gate: pd.DataFrame, flagged: pd.DataFrame, prices: pd.DataFrame) -> str:
    """Render ``results/data_quality.md`` (PROJECT_LOG.md §3.2) as Markdown.

    Must contain, in this order: a first line ``# Data quality report``; the calendar's
    first date, last date and row count; one table row per instrument starting with
    ``| <ticker> |`` giving first/last date, rows, filled count and fraction, longest gap,
    years, clipped-day count, ``pass``/``fail`` and the reasons; a clipped-returns section
    listing every row of ``flagged`` (date, ticker, raw and clipped value) or the words
    ``No clipped returns``; and a universe section listing the passing tickers joined by
    ``", "`` and the excluded ones with their reasons.
    """
    # TODO 9: implement data_quality_report.
    #   Build a list of lines and ``"\n".join`` them (end with a newline). Per-ticker clipped
    #   counts: ``flagged.groupby("ticker").size()`` (guard the empty case). Dates print as
    #   ``%Y-%m-%d`` (``_fmt_date`` below handles NaT). ``universe_dict(gate)`` gives the
    #   passing list and the excluded reasons for the last section.
    #   Verified by tests/test_data.py::TestDataQualityReport.
    lines = ["# Data quality report", ""]

    # 1. The calendar, the only thing read from the prices frame.
    cal = prices.index
    if len(cal):
        lines.append(f"Calendar: {_fmt_date(cal[0])} to {_fmt_date(cal[-1])}, {len(cal)} rows.")
    else:
        lines.append("Calendar: empty.")

    # 2. One table row per instrument: the gate's nine columns plus the clipped-day count,
    #    which comes from the flagged log (a ticker with no clips is not in the grouped
    #    count, hence the default of 0).
    clipped_counts = {} if flagged.empty else flagged.groupby("ticker").size().to_dict()
    lines += ["", "## Instruments", ""]
    lines.append(
        "| Ticker | First | Last | Rows | Filled | Fill % | Longest gap | Years | "
        "Clipped days | Verdict | Reasons |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, g in gate.iterrows():
        verdict = "pass" if bool(g["passed"]) else "fail"
        fill = "" if pd.isna(g["fill_fraction"]) else f"{g['fill_fraction']:.2%}"
        years = "" if pd.isna(g["years"]) else f"{g['years']:.1f}"
        lines.append(
            f"| {t} | {_fmt_date(g['first_date'])} | {_fmt_date(g['last_date'])} | "
            f"{g['rows']} | {g['n_filled']} | {fill} | {g['longest_gap']} | {years} | "
            f"{clipped_counts.get(t, 0)} | {verdict} | {g['reasons']} |"
        )

    # 3. Every clipped observation, or the words the docstring asks for.
    lines += ["", "## Clipped returns", ""]
    if flagged.empty:
        lines.append("No clipped returns.")
    else:
        lines.append("| Date | Ticker | Raw return | Clipped return |")
        lines.append("|---|---|---|---|")
        for _, r in flagged.iterrows():
            lines.append(
                f"| {_fmt_date(r['date'])} | {r['ticker']} | {r['raw_return']:+.4f} | "
                f"{r['clipped_return']:+.2f} |"
            )

    # 4. The universe, formatted from what universe_dict already computed.
    u = universe_dict(gate)
    lines += ["", "## Universe", ""]
    lines.append(
        f"{len(u['universe'])} of {u['n_candidates']} candidates pass: {', '.join(u['universe'])}"
    )
    if u["excluded"]:
        lines += ["", "Excluded:"]
        for t, reason in u["excluded"].items():
            lines.append(f"- {t}: {reason}")

    lines.append("")
    return "\n".join(lines)


def _fmt_date(value: Any) -> str:
    return "" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


# ------------------------------------------------------ F1.6 external benchmark


def fetch_benchmark(source: PriceSource, candidates: Iterable[str]) -> tuple[str, pd.Series]:
    """Return ``(ticker, series)`` for the first candidate the source can deliver.

    Candidates are tried in order (config ``benchmarks.external_tickers``); a ``FetchError``
    moves on to the next. If none loads, raise ``FetchError`` naming every candidate.
    """
    # TODO 10: implement fetch_benchmark. A loop with try/except FetchError; collect the
    #   error messages and raise one FetchError joining them if the loop finishes.
    #   Verified by tests/test_data.py::TestFetchBenchmark.
    # First success wins; only the contract's own failure type moves us to the next
    # candidate, and every message is kept so the final error tells the whole story.
    errors: list[str] = []
    for ticker in candidates:
        try:
            return ticker, source.fetch(ticker)
        except FetchError as err:
            errors.append(str(err))
    raise FetchError("no benchmark candidate could be fetched: " + "; ".join(errors))


def write_benchmark(prices: pd.Series, path: str | Path) -> Path:
    """Write the benchmark series as a one-column wide CSV (``date,<ticker>``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_frame().to_csv(path, index_label="date")
    return path


def load_benchmark(path: str | Path) -> tuple[str, pd.Series]:
    """Read the benchmark CSV back; returns ``(ticker, series)``."""
    wide = pd.read_csv(path, index_col="date", parse_dates=True)
    ticker = str(wide.columns[0])
    s = wide[ticker].astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    s.index.name = "date"
    s.name = ticker
    return ticker, s


# ------------------------------------------- F1.5 snapshot manifest and panel build


def sha256_of(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_entry(root: Path, rel: str) -> dict[str, Any]:
    """Rows, first/last date, columns and hash of one CSV under ``root``."""
    wide = pd.read_csv(root / rel, index_col="date", parse_dates=True)
    return {
        "path": rel,
        "rows": int(len(wide)),
        "columns": [str(c) for c in wide.columns],
        "first": _fmt_date(wide.index[0]) if len(wide) else None,
        "last": _fmt_date(wide.index[-1]) if len(wide) else None,
        "sha256": sha256_of(root / rel),
    }


def build_manifest(
    cfg: DataConfig,
    root: str | Path,
    benchmark_ticker: str | None,
    pulled_at: str,
    yfinance_version: str,
) -> dict[str, Any]:
    """Describe the snapshot on disk (PROJECT_LOG.md §3.5) as a JSON-ready dict.

    Reads the files named by ``cfg`` (``raw_dir/<ticker>.csv`` for each of
    ``cfg.tickers``, ``panel_path``, and ``benchmark_path`` when ``benchmark_ticker`` is
    given), all relative to ``root``. :func:`_file_entry` produces each file's record.

    Returns::

        {
          "pulled_at": pulled_at, "yfinance_version": yfinance_version,
          "price_field": cfg.price_field,
          "tickers": {ticker: {"path", "rows", "first", "last", "sha256"}, ...},
          "panel": {"path", "rows", "columns", "first", "last", "sha256"},
          "benchmark": {"ticker", "path", "rows", "first", "last", "sha256"} or None,
        }

    Paths are the relative strings from ``cfg`` (e.g. ``"data/raw/CL=F.csv"``).
    """
    # TODO 11: implement build_manifest.
    #   ``_file_entry(root, f"{cfg.raw_dir}/{t}.csv")`` per ticker (drop its "columns" key),
    #   ``_file_entry(root, cfg.panel_path)`` for the panel (keep "columns"), and for the
    #   benchmark ``{"ticker": benchmark_ticker, **_file_entry(root, cfg.benchmark_path)}``
    #   minus "columns", or None. Assemble the dict in the documented key order.
    #   Verified by tests/test_data.py::TestManifest.
    root = Path(root)

    # 1. One record per raw file. A raw CSV has one column, the ticker itself, so the
    #    helper's "columns" key says nothing new and is dropped.
    tickers: dict[str, Any] = {}
    for t in cfg.tickers:
        entry = _file_entry(root, f"{cfg.raw_dir}/{t}.csv")
        entry.pop("columns")
        tickers[t] = entry

    # 2. The panel keeps its column list: which instruments made it in is a fact worth
    #    recording.
    panel = _file_entry(root, cfg.panel_path)

    # 3. The benchmark record leads with its ticker, or is None when nothing loaded.
    benchmark: dict[str, Any] | None = None
    if benchmark_ticker is not None:
        entry = _file_entry(root, cfg.benchmark_path)
        entry.pop("columns")
        benchmark = {"ticker": benchmark_ticker, **entry}

    # 4. Provenance first, then the files, in the documented key order. pulled_at and the
    #    yfinance version come from the caller that did the pull; this function is offline.
    return {
        "pulled_at": pulled_at,
        "yfinance_version": yfinance_version,
        "price_field": cfg.price_field,
        "tickers": tickers,
        "panel": panel,
        "benchmark": benchmark,
    }


def verify_manifest(manifest: dict[str, Any], root: str | Path) -> list[str]:
    """Check every file the manifest names against its recorded SHA-256.

    Looks at each entry under ``"tickers"``, the ``"panel"`` entry and the ``"benchmark"``
    entry (when not None). Returns one message per problem, each starting with the file's
    relative path: ``"<path>: missing"`` or ``"<path>: sha256 mismatch"``. An empty list
    means the snapshot on disk is exactly the one the manifest describes.
    """
    # TODO 12: implement verify_manifest. Gather the entries, then for each:
    #   ``Path(root) / entry["path"]`` exists? ``sha256_of(...) == entry["sha256"]``?
    #   Verified by tests/test_data.py::TestManifest and ::TestSnapshot (the committed
    #   snapshot must verify clean).
    root = Path(root)

    # 1. Every file record the manifest holds: each ticker, the panel, the benchmark if any.
    entries: list[dict[str, Any]] = list(manifest["tickers"].values())
    entries.append(manifest["panel"])
    if manifest.get("benchmark") is not None:
        entries.append(manifest["benchmark"])

    # 2. Existence before hashing: read_bytes on a missing path would raise instead of
    #    letting us report it. One message per problem, all problems reported.
    problems: list[str] = []
    for entry in entries:
        path = root / entry["path"]
        if not path.exists():
            problems.append(f"{entry['path']}: missing")
        elif sha256_of(path) != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 mismatch")
    return problems


def manifest_diff(old: dict[str, Any] | None, new: dict[str, Any]) -> str:
    """Plain-text summary of what changed between two manifests (for ``make data``)."""
    if old is None:
        return "No previous manifest: first snapshot."
    lines = []
    for t, entry in new["tickers"].items():
        before = old.get("tickers", {}).get(t)
        if before is None:
            lines.append(f"{t}: new ({entry['rows']} rows, {entry['first']} to {entry['last']})")
        elif before["sha256"] != entry["sha256"]:
            lines.append(
                f"{t}: {before['rows']} -> {entry['rows']} rows, "
                f"{before['first']} -> {entry['first']}, {before['last']} -> {entry['last']}"
            )
    for t in old.get("tickers", {}):
        if t not in new["tickers"]:
            lines.append(f"{t}: dropped")
    if old.get("panel", {}).get("sha256") != new["panel"]["sha256"]:
        lines.append(f"panel: {old.get('panel', {}).get('rows')} -> {new['panel']['rows']} rows")
    return "\n".join(lines) if lines else "Snapshot unchanged."


@dataclass(frozen=True)
class PanelBuild:
    """Everything ``run_panel`` computes, for the numbers script and later stages."""

    prices: pd.DataFrame
    filled: pd.DataFrame
    returns: pd.DataFrame
    flagged: pd.DataFrame
    gate: pd.DataFrame
    universe: list[str]


def run_panel(cfg: Config, root: str | Path) -> PanelBuild:
    """Build the aligned panel from the raw CSVs and write the snapshot files. No network.

    Steps, all relative to ``root``:

    1. :func:`load_raw` for ``cfg.data.tickers`` from ``cfg.data.raw_dir``;
    2. :func:`align_prices` with ``cfg.data.ffill_limit``;
    3. :func:`clean_returns` with ``cfg.data.clip``;
    4. :func:`quality_gate` with ``cfg.quality_gate`` and :func:`universe_dict`;
    5. write ``cfg.data.panel_path``: the aligned prices of the **passing** tickers only,
       dropping calendar days on which none of them has a price (so the panel's calendar
       is the union of the universe's trading days, not the candidates'), as a wide CSV
       with ``date`` as the index label (``DataFrame.to_csv`` defaults);
    6. write ``<results_dir>/universe.json`` (the universe dict, indented JSON) and
       ``<results_dir>/data_quality.md`` (:func:`data_quality_report`), creating the
       directory if needed.

    The manifest is **not** touched: it is the pull's record (``run_data``), and the test
    suite checks that this function reproduces the panel the manifest describes.
    Returns a :class:`PanelBuild` holding every intermediate for all candidates.
    """
    # TODO 13: implement run_panel. Follow the six steps literally; ``root / cfg.data.raw_dir``
    #   etc. for paths, ``mkdir(parents=True, exist_ok=True)`` before writing,
    #   ``prices[universe].dropna(how="all").to_csv(panel_path)`` for the panel,
    #   ``json.dumps(universe, indent=2) + "\n"`` for universe.json. Return the PanelBuild.
    #   Verified by tests/test_data.py::TestRunPanel; ``make panel`` then reproduces
    #   ``data/futures_panel.csv`` byte for byte (::TestSnapshot).
    root = Path(root)
    data = cfg.data

    # 1-4. The pure chain, on every candidate: load, align, clean, gate.
    raw = load_raw(root / data.raw_dir, data.tickers)
    prices, filled = align_prices(raw, data.ffill_limit)
    returns, flagged = clean_returns(prices, data.clip)
    gate = quality_gate(prices, filled, cfg.quality_gate)
    universe = universe_dict(gate)

    # 5. The panel: passers only, and only the days on which at least one of them traded.
    #    Default to_csv settings, so the rebuild matches the committed file byte for byte.
    panel_path = root / data.panel_path
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    prices[universe["universe"]].dropna(how="all").to_csv(panel_path)

    # 6. The two reports. The manifest is deliberately not written here.
    results_dir = root / cfg.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "universe.json").write_text(json.dumps(universe, indent=2) + "\n")
    (results_dir / "data_quality.md").write_text(data_quality_report(gate, flagged, prices))

    return PanelBuild(
        prices=prices,
        filled=filled,
        returns=returns,
        flagged=flagged,
        gate=gate,
        universe=universe["universe"],
    )


def make_source(cfg: Config) -> PriceSource:
    """The ``PriceSource`` named by ``cfg.data.source``."""
    d = cfg.data
    if d.source == "csv":
        return CSVSource(d.csv_path or "")
    return YFinanceSource(d.price_field, retries=d.fetch_retries, start=d.start)


def run_data(cfg: Config, root: str | Path) -> dict[str, Any]:
    """``make data``: the one network stage. Re-pull everything, rebuild, write the manifest.

    Fetches every candidate ticker and the external benchmark, runs :func:`run_panel`,
    writes ``cfg.data.manifest_path``, and returns the new manifest after printing a diff
    against the previous one. A benchmark that cannot be fetched is recorded as ``None``
    with a warning rather than failing the pull.
    """
    root = Path(root)
    source = make_source(cfg)
    fetch_all(source, cfg.data.tickers, root / cfg.data.raw_dir)
    benchmark_ticker: str | None
    try:
        benchmark_ticker, bench = fetch_benchmark(source, cfg.benchmarks.external_tickers)
        write_benchmark(bench, root / cfg.data.benchmark_path)
    except FetchError as e:
        print(f"warning: {e}")
        benchmark_ticker = None
    run_panel(cfg, root)
    manifest_path = root / cfg.data.manifest_path
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    try:
        import yfinance

        version = str(yfinance.__version__)
    except ImportError:  # pragma: no cover - yfinance is a hard dependency
        version = "unknown"
    pulled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(cfg.data, root, benchmark_ticker, pulled_at, version)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_diff(old, manifest))
    return manifest
