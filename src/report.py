"""Report helpers: the metrics table as Markdown, ``metrics.json``, the README injection,
and the robustness reports.

Phase 5 delivers the three pure building blocks (F5.3); the ``metrics`` stage in
``src/cli.py`` writes their output to ``results/``, and Phase 7's ``make report`` (F7.6)
uses :func:`inject_between_markers` to paste the table into ``README.md`` between the
``<!-- METRICS:START -->`` and ``<!-- METRICS:END -->`` markers. Phase 6 adds
:func:`format_frame_md`, a general frame-to-Markdown table, and the three report
composers built on it (``results/is_oos.md``, ``subperiods.md``, ``sweeps.md``). Nothing
here reads or writes a file: inputs in, strings out.

The input shared by the first two functions is ``metrics``: a dict mapping a series name
(``"Strategy (net)"``, ``"^BCOM"``, ...) to that series' dict of statistics (the keys of
``METRIC_ROWS``). Dict order is the column order in the table and the key order in the JSON.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.config import ANNUALISATION

METRICS_START = "<!-- METRICS:START -->"
METRICS_END = "<!-- METRICS:END -->"
"""The README markers between which the metrics table is injected."""

METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("ann_return", "Annualised return", "pct"),
    ("ann_vol", "Annualised volatility", "pct"),
    ("sharpe", "Sharpe ratio", "ratio"),
    ("sortino", "Sortino ratio", "ratio"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("max_drawdown_duration", "Max drawdown duration (days)", "int"),
    ("calmar", "Calmar ratio", "ratio"),
    ("hit_rate_daily", "Hit rate (daily)", "pct"),
    ("hit_rate_monthly", "Hit rate (monthly)", "pct"),
    ("skew", "Skew", "ratio"),
    ("kurtosis", "Excess kurtosis", "ratio"),
    ("best_month", "Best month", "pct"),
    ("worst_month", "Worst month", "pct"),
    ("ann_turnover", "Turnover (x per year)", "ratio"),
    ("mean_gross_leverage", "Average gross leverage", "ratio"),
    ("cap_bind_fraction", "Leverage cap binds (share of days)", "pct"),
    ("mean_n_live", "Average instruments live", "ratio"),
    ("sharpe_t_stat", "Sharpe t-statistic", "ratio"),
    ("sharpe_ci_low", "Sharpe CI low (bootstrap)", "ratio"),
    ("sharpe_ci_high", "Sharpe CI high (bootstrap)", "ratio"),
    ("psr", "Probabilistic Sharpe (vs 0)", "prob"),
    ("n_trials", "Trials counted", "int"),
    ("sharpe_hurdle", "Sharpe hurdle (DSR)", "ratio"),
    ("dsr", "Deflated Sharpe", "prob"),
    ("n_days", "Days", "int"),
    ("first_date", "First day", "date"),
    ("last_date", "Last day", "date"),
)
"""The table's rows, in order: ``(key, label, kind)``. ``kind`` picks the format:

* ``"pct"``: a percentage with two decimals, ``f"{x:.2%}"`` (``0.0317`` -> ``3.17%``);
* ``"ratio"``: two decimals, ``f"{x:.2f}"``;
* ``"prob"``: three decimals, ``f"{x:.3f}"``;
* ``"int"``: an integer, ``str(int(x))``;
* ``"date"``: ``f"{x:%Y-%m-%d}"``.

A missing key, ``None`` or a non-finite number renders as ``n/a``.
"""


def format_metrics_md(
    metrics: dict[str, dict[str, object]],
    rows: tuple[tuple[str, str, str], ...] = METRIC_ROWS,
) -> str:
    """The metrics as one Markdown table: a ``Metric`` column of labels, then one column per
    series in ``metrics`` order, one line per entry of ``rows`` in order.

    Exactly::

        | Metric | <name 1> | <name 2> | ... |
        |---|---:|---:|...|
        | <label> | <value> | <value> | ... |
        ...

    one line per row, each line ending in ``\\n`` (the last one included); a header cell is
    the series name as given, the separator has ``---`` for the label column and ``---:``
    (right-aligned) for each series, and each value is formatted by its row's kind (see
    ``METRIC_ROWS``), ``n/a`` when missing, ``None`` or not finite. numpy scalars format like
    the Python numbers they hold.
    """

    # TODO 56: implement format_metrics_md.
    #   A small helper ``cell(value, kind) -> str`` does the work: ``n/a`` if ``value is
    #   None``, or a number (``isinstance(value, int | float | np.integer | np.floating)``,
    #   but not a ``bool``) that is not finite (``math.isfinite(float(value))``); else by
    #   kind as listed in ``METRIC_ROWS``. Build the lines: ``"| Metric | " + " | ".join(names)
    #   + " |"``, ``"|---|" + "---:|" * len(names)``, then for each ``(key, label, kind)`` in
    #   ``rows``: ``"| " + " | ".join([label] + [cell(metrics[n].get(key), kind) for n in
    #   names]) + " |"``. Join with ``"\n"`` and add the final ``"\n"``.
    #   The stub imports nothing beyond ``__future__``. Imports to add as you go: ``math`` and
    #   ``numpy as np`` here; ``json`` and ``pandas as pd`` for TODO 57.
    #   Verified by tests/test_report.py::TestFormatMetricsMd.
    def cell(value: object, kind: str) -> str:
        if value is None:
            return "n/a"
        is_number = isinstance(value, int | float | np.integer | np.floating)
        if is_number and not isinstance(value, bool) and not math.isfinite(float(value)):
            return "n/a"
        if kind == "pct":
            return f"{value:.2%}"
        if kind == "ratio":
            return f"{value:.2f}"
        if kind == "prob":
            return f"{value:.3f}"
        if kind == "int":
            return str(int(value))
        if kind == "date":
            return f"{value:%Y-%m-%d}"
        raise ValueError(f"unknown kind: {kind!r}")

    names = list(metrics)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, label, kind in rows:
        cells = [cell(metrics[n].get(key), kind) for n in names]
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines) + "\n"


def metrics_json(metrics: dict[str, dict[str, object]]) -> str:
    """``metrics`` as strict JSON text, for ``results/metrics.json``.

    Values are converted first, recursively and without touching the input: a NaN or
    infinite float becomes ``null``; a ``pd.Timestamp`` becomes its date string
    ``"YYYY-MM-DD"``; numpy integers and floats become Python ``int`` and ``float``; ``None``,
    ``bool``, ``int``, ``float`` and ``str`` stay as they are; dicts keep their key order.
    The text is ``json.dumps(converted, indent=2, allow_nan=False)`` plus a final ``"\\n"``,
    so two calls on equal inputs return identical bytes (the §8 determinism test compares
    the file).
    """

    # TODO 57: implement metrics_json.
    #   A recursive ``convert(obj)``: dict -> ``{k: convert(v) for k, v in obj.items()}``;
    #   ``pd.Timestamp`` -> ``f"{obj:%Y-%m-%d}"``; ``np.integer`` -> ``int(obj)``;
    #   ``float`` or ``np.floating`` -> ``float(obj)`` if ``math.isfinite`` else ``None``
    #   (check ``bool`` and ``int`` before ``float``: a Python ``bool`` is an ``int``);
    #   anything else unchanged. Then ``json.dumps(convert(metrics), indent=2,
    #   allow_nan=False) + "\n"``.
    #   Verified by tests/test_report.py::TestMetricsJson.
    def convert(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, pd.Timestamp):
            return f"{obj:%Y-%m-%d}"
        if isinstance(obj, bool | int):
            return obj
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, float | np.floating):
            return float(obj) if math.isfinite(obj) else None
        return obj

    return json.dumps(convert(metrics), indent=2, allow_nan=False) + "\n"


def inject_between_markers(
    text: str, block: str, start: str = METRICS_START, end: str = METRICS_END
) -> str:
    """Replace whatever sits between the ``start`` and ``end`` markers of ``text`` by ``block``.

    The result is ``text`` up to and including ``start``, a newline, ``block`` with its
    leading and trailing newlines stripped, a newline, then ``text`` from ``end`` onwards:
    everything outside the markers, the markers themselves included, is unchanged byte for
    byte. Injecting twice gives the same text as injecting once.

    Raises ``ValueError`` unless ``text`` contains each marker exactly once, ``start`` before
    ``end``.
    """
    # TODO 58: implement inject_between_markers.
    #   ``text.count(start) == 1 and text.count(end) == 1`` or ``ValueError``; ``i =
    #   text.index(start) + len(start)``; ``j = text.index(end)``; ``ValueError`` if ``j <
    #   i``; ``text[:i] + "\n" + block.strip("\n") + "\n" + text[j:]``.
    #   Verified by tests/test_report.py::TestInjectBetweenMarkers.
    if not (text.count(start) == 1 and text.count(end) == 1):
        raise ValueError(f"text must contain {start!r} and {end!r} exactly once each")
    i = text.index(start) + len(start)
    j = text.index(end)
    if j < i:
        raise ValueError(f"{end!r} comes before {start!r}")
    return text[:i] + "\n" + block.strip("\n") + "\n" + text[j:]


# ------------------------------------------------------------ F6 robustness reports

FRAME_KINDS = ("pct", "ratio", "prob", "int", "date", "text", "flag")
"""The cell formats of :func:`format_frame_md`: the five of ``METRIC_ROWS`` plus ``"text"``
(``str(value)``) and ``"flag"`` (``"yes"`` for a true value, ``"-"`` otherwise)."""

NUMERIC_KINDS = ("pct", "ratio", "prob", "int")
"""The kinds whose column is right-aligned (``---:``); the others are left-aligned."""

HINDSIGHT_CAVEAT = (
    "**Hindsight caveat.** A day is labelled *stress* when the equal-weight basket's "
    "trailing volatility is above a quantile of its full-sample distribution. That "
    "threshold was only known once the sample had ended, so this view describes where the "
    "returns came from; it is not a rule anyone could have traded."
)
"""The caption printed above the regime table of ``results/subperiods.md`` (§5.3)."""


def format_frame_md(frame: pd.DataFrame, kinds: dict[str, str]) -> str:
    """``frame`` as one Markdown table: one column per frame column, one line per row.

    Exactly::

        | <column 1> | <column 2> | ... |
        |---|---:|...|
        | <value> | <value> | ... |
        ...

    the header cells are the column names (``str``), the separator has ``---:`` for a column
    of a numeric kind (``NUMERIC_KINDS``) and ``---`` otherwise, every line ends in ``\n``
    (the last one included), and the index is not printed (``reset_index`` first to show
    it). ``kinds`` maps **every** column to one of ``FRAME_KINDS``: ``"pct"``, ``"ratio"``,
    ``"prob"``, ``"int"`` and ``"date"`` format as in ``METRIC_ROWS``, ``"text"`` as
    ``str(value)``, ``"flag"`` as ``"yes"`` when the value is true and ``"-"`` otherwise. A
    cell that is ``None``, NaN, ``NaT`` or a non-finite number renders as ``n/a`` whatever
    its kind (a ``"flag"`` cell included). A frame without rows gives the two header lines.

    Raises ``ValueError`` if a column has no kind or a kind is not in ``FRAME_KINDS``.
    """
    # TODO 75: implement format_frame_md.
    #   Check first: every column in ``kinds`` and every kind in ``FRAME_KINDS``, or
    #   ``ValueError``. A ``cell(value, kind)`` helper: ``n/a`` when ``value is None``, or
    #   ``pd.isna(value)`` is true (NaN and ``NaT``), or the value is a number (not a
    #   ``bool``) that is not finite; else by kind (the five of ``METRIC_ROWS``, as in
    #   TODO 56, then ``"text"`` -> ``str(value)``, ``"flag"`` -> ``"yes" if value else
    #   "-"``). Header ``"| " + " | ".join(str(c) for c in frame.columns) + " |"``;
    #   separator ``"|" + "".join("---:|" if kinds[c] in NUMERIC_KINDS else "---|" for c in
    #   frame.columns)``; one line per row (``frame.itertuples(index=False)`` gives the values
    #   in column order). Join with ``"\n"`` and add the final ``"\n"``.
    #   Verified by tests/test_report.py::TestFormatFrameMd and ::TestRobustnessReports.
    for column in frame.columns:
        if column not in kinds:
            raise ValueError(f"column {column!r} has no kind")
    for column, kind in kinds.items():
        if kind not in FRAME_KINDS:
            raise ValueError(f"column {column!r} has unknown kind {kind!r}")

    def cell(value: object, kind: str) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        is_number = isinstance(value, int | float | np.integer | np.floating)
        if is_number and not isinstance(value, bool) and not math.isfinite(float(value)):
            return "n/a"
        if kind == "pct":
            return f"{value:.2%}"
        if kind == "ratio":
            return f"{value:.2f}"
        if kind == "prob":
            return f"{value:.3f}"
        if kind == "int":
            return str(int(value))
        if kind == "date":
            return f"{value:%Y-%m-%d}"
        if kind == "text":
            return str(value)
        return "yes" if value else "-"

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "|" + "".join("---:|" if kinds[c] in NUMERIC_KINDS else "---|" for c in columns),
    ]
    for row in frame.itertuples(index=False):
        cells = [cell(value, kinds[c]) for value, c in zip(row, columns, strict=True)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _summary_lines(values: list[tuple[str, str]]) -> list[str]:
    lines = ["| Quantity | Value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in values]
    return lines


def _num(value: object, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return format(value, spec)


def is_oos_md(
    table: pd.DataFrame, summary: dict[str, object], is_end: str, oos_start: str, cost_bps: float
) -> str:
    """``results/is_oos.md``: the in-sample / out-of-sample selection of the lookback.

    ``table`` and ``summary`` are ``robustness.is_oos``'s output. Provided complete (not a
    TODO); the table goes through :func:`format_frame_md`.
    """
    shown = table.reset_index().rename(
        columns={
            "lookback": "Lookback",
            "sharpe_is": "Sharpe IS",
            "sharpe_oos": "Sharpe OOS",
            "degradation": "IS - OOS",
            "n_days_is": "Days IS",
            "n_days_oos": "Days OOS",
            "selected": "Selected",
            "default": "Default",
        }
    )
    kinds = {
        "Lookback": "int",
        "Sharpe IS": "ratio",
        "Sharpe OOS": "ratio",
        "IS - OOS": "ratio",
        "Days IS": "int",
        "Days OOS": "int",
        "Selected": "flag",
        "Default": "flag",
    }
    s = summary
    lines = [
        "# In-sample / out-of-sample selection",
        "",
        f"The lookback `L` is selected on the in-sample net Sharpe ratio (each trial's life up "
        f"to {is_end}, at {cost_bps:g} bps), every other parameter at its a-priori default, "
        f"and then measured out of sample ({oos_start} to the end of the data). Only `L` is "
        "selected. The headline result stays the a-priori default; this table shows what "
        "selecting would have done.",
        "",
        format_frame_md(shown, kinds).rstrip("\n"),
        "",
        "## Summary",
        "",
        *_summary_lines(
            [
                ("Selected lookback", str(s["selected_lookback"])),
                ("Selected: Sharpe in sample", _num(s["selected_is"], ".2f")),
                ("Selected: Sharpe out of sample", _num(s["selected_oos"], ".2f")),
                ("Selected: degradation (IS - OOS)", _num(s["selected_degradation"], ".2f")),
                ("A-priori lookback", str(s["default_lookback"])),
                ("A-priori: Sharpe in sample", _num(s["default_is"], ".2f")),
                ("A-priori: Sharpe out of sample", _num(s["default_oos"], ".2f")),
                ("A-priori: degradation (IS - OOS)", _num(s["default_degradation"], ".2f")),
                ("Out of sample, selected minus a-priori", _num(s["oos_gap"], "+.2f")),
            ]
        ),
    ]
    return "\n".join(lines) + "\n"


def subperiods_md(
    blocks: pd.DataFrame,
    years: pd.DataFrame,
    rolling: pd.DataFrame,
    regimes: pd.DataFrame,
    rolling_years: int,
    regime_quantile: float,
    regime_window: int,
) -> str:
    """``results/subperiods.md``: calendar blocks, calendar years, the rolling Sharpe ratio
    and the regime view with its hindsight caveat.

    The four frames are ``subperiods.block_stats``, ``calendar_year_returns``,
    ``rolling_summary`` and ``regime_stats`` output. Provided complete (not a TODO).
    """
    block_kinds = {
        "Block": "text",
        "Series": "text",
        "First day": "date",
        "Last day": "date",
        "Days": "int",
        "Ann. return": "pct",
        "Ann. vol": "pct",
        "Sharpe": "ratio",
    }
    shown_blocks = blocks.rename(
        columns={
            "block": "Block",
            "series": "Series",
            "first_date": "First day",
            "last_date": "Last day",
            "n_days": "Days",
            "ann_return": "Ann. return",
            "ann_vol": "Ann. vol",
            "sharpe": "Sharpe",
        }
    )
    shown_years = years.reset_index().rename(columns={"year": "Year"})
    year_kinds = {c: "pct" for c in shown_years.columns}
    year_kinds["Year"] = "text"
    shown_rolling = (
        rolling.rename_axis("Series")
        .reset_index()
        .rename(
            columns={
                "n_windows": "Windows",
                "share_positive": "Share > 0",
                "min": "Min",
                "min_date": "Min ends",
                "max": "Max",
                "max_date": "Max ends",
                "last": "Last",
                "last_date": "Last ends",
            }
        )
    )
    rolling_kinds = {
        "Series": "text",
        "Windows": "int",
        "Share > 0": "pct",
        "Min": "ratio",
        "Min ends": "date",
        "Max": "ratio",
        "Max ends": "date",
        "Last": "ratio",
        "Last ends": "date",
    }
    shown_regimes = regimes.rename(
        columns={
            "regime": "Regime",
            "series": "Series",
            "n_days": "Days",
            "share_of_days": "Share of days",
            "ann_return": "Ann. return",
            "ann_vol": "Ann. vol",
            "sharpe": "Sharpe",
        }
    )
    regime_kinds = {
        "Regime": "text",
        "Series": "text",
        "Days": "int",
        "Share of days": "pct",
        "Ann. return": "pct",
        "Ann. vol": "pct",
        "Sharpe": "ratio",
    }
    lines = [
        "# Sub-periods",
        "",
        "## Calendar blocks (fixed a priori)",
        "",
        format_frame_md(shown_blocks, block_kinds).rstrip("\n"),
        "",
        "## Calendar years (compounded return)",
        "",
        format_frame_md(shown_years, year_kinds).rstrip("\n"),
        "",
        f"## Rolling {rolling_years}-year Sharpe ratio",
        "",
        f"Trailing {rolling_years * ANNUALISATION} observations of each series; a window is "
        "dated by its last day.",
        "",
        format_frame_md(shown_rolling, rolling_kinds).rstrip("\n"),
        "",
        "## Regimes: calm versus stress (hindsight)",
        "",
        HINDSIGHT_CAVEAT,
        f"Here: the equal-weight long-only basket's trailing {regime_window}-day volatility "
        f"above its full-sample {regime_quantile:.0%} quantile.",
        "",
        format_frame_md(shown_regimes, regime_kinds).rstrip("\n"),
    ]
    return "\n".join(lines) + "\n"


def sweeps_md(
    trials: dict[str, object],
    distribution: dict[str, object],
    costs: pd.DataFrame,
    lags: pd.DataFrame,
    cost_bps: float,
) -> str:
    """``results/sweeps.md``: the trial count and the Deflated Sharpe Ratio, where the default
    sits in the grid, and the cost and lag sweeps.

    ``trials`` holds ``n_trials``, ``sharpe_std``, ``sharpe_hurdle``, ``psr`` and ``dsr`` (the
    default's net series); ``distribution`` is ``robustness.grid_distribution``; ``costs``
    and ``lags`` are ``cost_sweep`` and ``lag_sweep``. Provided complete (not a TODO).
    """
    t, d = trials, distribution
    shown_costs = costs.reset_index().rename(
        columns={
            "cost_bps": "Cost (bps)",
            "ann_return": "Ann. return",
            "sharpe": "Sharpe",
            "max_drawdown": "Max drawdown",
            "grid_median_sharpe": "Grid median Sharpe",
            "grid_share_positive": "Grid share Sharpe > 0",
        }
    )
    cost_kinds = {
        "Cost (bps)": "int",
        "Ann. return": "pct",
        "Sharpe": "ratio",
        "Max drawdown": "pct",
        "Grid median Sharpe": "ratio",
        "Grid share Sharpe > 0": "pct",
    }
    shown_lags = lags.reset_index().rename(
        columns={
            "lag": "Lag (days)",
            "sharpe_gross": "Sharpe gross",
            "sharpe_net": "Sharpe net",
            "ann_return_net": "Ann. return net",
            "grid_median_net": "Grid median Sharpe net",
        }
    )
    lag_kinds = {
        "Lag (days)": "int",
        "Sharpe gross": "ratio",
        "Sharpe net": "ratio",
        "Ann. return net": "pct",
        "Grid median Sharpe net": "ratio",
    }
    lines = [
        "# Robustness: trials, grid and sweeps",
        "",
        "## Trials and the Deflated Sharpe Ratio",
        "",
        *_summary_lines(
            [
                ("Trials (configurations evaluated)", str(t["n_trials"])),
                (f"Std of the trials' net Sharpe ({cost_bps:g} bps)", _num(t["sharpe_std"], ".3f")),
                ("Sharpe hurdle (expected max of the trials)", _num(t["sharpe_hurdle"], ".3f")),
                ("Probabilistic Sharpe of the default (net, vs 0)", _num(t["psr"], ".3f")),
                ("Deflated Sharpe of the default (net)", _num(t["dsr"], ".3f")),
            ]
        ),
        "",
        f"## The default among the trials (net Sharpe at {cost_bps:g} bps)",
        "",
        *_summary_lines(
            [
                ("Minimum", _num(d["sharpe_min"], ".2f")),
                ("Lower quartile", _num(d["sharpe_q25"], ".2f")),
                ("Median", _num(d["sharpe_median"], ".2f")),
                ("Upper quartile", _num(d["sharpe_q75"], ".2f")),
                ("Maximum", _num(d["sharpe_max"], ".2f")),
                ("Share of trials with Sharpe > 0", _num(d["share_positive"], ".1%")),
                ("Default", _num(d["default_sharpe"], ".2f")),
                ("Default rank (1 = best)", f"{d['default_rank']} of {d['n_trials']}"),
                ("Best trial", str(d["best_trial"])),
                ("Worst trial", str(d["worst_trial"])),
            ]
        ),
        "",
        "## Cost sweep (default configuration; 0 bps is the gross result)",
        "",
        format_frame_md(shown_costs, cost_kinds).rstrip("\n"),
        "",
        "## Lag sweep (default configuration)",
        "",
        format_frame_md(shown_lags, lag_kinds).rstrip("\n"),
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "FRAME_KINDS",
    "HINDSIGHT_CAVEAT",
    "METRICS_END",
    "METRICS_START",
    "METRIC_ROWS",
    "NUMERIC_KINDS",
    "format_frame_md",
    "format_metrics_md",
    "inject_between_markers",
    "is_oos_md",
    "metrics_json",
    "subperiods_md",
    "sweeps_md",
]
