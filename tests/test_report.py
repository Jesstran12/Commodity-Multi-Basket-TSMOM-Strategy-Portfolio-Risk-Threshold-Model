"""Report helpers (Phase 5, F5.3): the metrics table as Markdown, ``metrics.json``, and the
README marker injection that ``make report`` will use (PROJECT_LOG.md §6, F7.6); and
(Phase 6) the general frame-to-Markdown table and the three robustness report composers.

All are pure functions over Python objects and strings; nothing here touches a file except
one read-only check that the repository's README carries the markers exactly once.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.report import (
    FRAME_KINDS,
    HINDSIGHT_CAVEAT,
    METRIC_ROWS,
    METRICS_END,
    METRICS_START,
    NUMERIC_KINDS,
    format_frame_md,
    format_metrics_md,
    inject_between_markers,
    is_oos_md,
    metrics_json,
    subperiods_md,
    sweeps_md,
)

ROOT = Path(__file__).resolve().parents[1]


def entry(**values) -> dict[str, object]:
    return dict(values)


# ---------------------------------------------------------------------- METRIC_ROWS


class TestMetricRows:
    def test_rows_cover_every_metric_book_and_inference_key_once(self):
        from src.cli import BOOK_KEYS
        from src.inference import INFERENCE_KEYS
        from src.metrics import METRIC_KEYS

        keys = [k for k, _, _ in METRIC_ROWS]
        assert len(keys) == len(set(keys))
        assert set(keys) == set(METRIC_KEYS) | set(BOOK_KEYS) | set(INFERENCE_KEYS)

    def test_kinds_are_known(self):
        assert {kind for _, _, kind in METRIC_ROWS} <= {"int", "date", "pct", "ratio", "prob"}

    def test_the_headline_rows_come_first(self):
        keys = [k for k, _, _ in METRIC_ROWS]
        assert keys[:5] == ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"]


# ---------------------------------------------------------------- format_metrics_md


ROWS_SMALL = (
    ("ann_return", "Annualised return", "pct"),
    ("sharpe", "Sharpe ratio", "ratio"),
    ("psr", "Probabilistic Sharpe", "prob"),
    ("n_days", "Days", "int"),
    ("first_date", "First day", "date"),
)


class TestFormatMetricsMd:
    def test_exact_rendering(self):
        metrics = {
            "Strategy (net)": entry(
                ann_return=0.031662,
                sharpe=0.34216,
                psr=0.9567,
                n_days=6410,
                first_date=pd.Timestamp("2001-04-02"),
            ),
            "^BCOM": entry(
                ann_return=-0.4042,
                sharpe=float("nan"),
                psr=None,
                n_days=6356,
                first_date=pd.Timestamp("2001-04-02"),
            ),
        }
        expected = (
            "| Metric | Strategy (net) | ^BCOM |\n"
            "|---|---:|---:|\n"
            "| Annualised return | 3.17% | -40.42% |\n"
            "| Sharpe ratio | 0.34 | n/a |\n"
            "| Probabilistic Sharpe | 0.957 | n/a |\n"
            "| Days | 6410 | 6356 |\n"
            "| First day | 2001-04-02 | 2001-04-02 |\n"
        )
        assert format_metrics_md(metrics, ROWS_SMALL) == expected

    def test_missing_keys_render_as_na(self):
        md = format_metrics_md({"A": {}}, ROWS_SMALL)
        assert md.splitlines()[2] == "| Annualised return | n/a |"
        assert md.count("n/a") == len(ROWS_SMALL)

    def test_numpy_scalars_render_like_python_ones(self):
        a = format_metrics_md({"A": entry(sharpe=np.float64(0.5), n_days=np.int64(3))}, ROWS_SMALL)
        b = format_metrics_md({"A": entry(sharpe=0.5, n_days=3)}, ROWS_SMALL)
        assert a == b

    def test_default_rows_and_column_order(self):
        metrics = {"B": {"sharpe": 1.0}, "A": {"sharpe": 2.0}}
        md = format_metrics_md(metrics)
        lines = md.splitlines()
        assert lines[0] == "| Metric | B | A |"
        assert len(lines) == 2 + len(METRIC_ROWS)
        assert "| Sharpe ratio | 1.00 | 2.00 |" in lines
        assert md.endswith("\n")

    def test_every_line_has_the_same_number_of_cells(self):
        metrics = {f"S{i}": {} for i in range(5)}
        lines = format_metrics_md(metrics).splitlines()
        assert {line.count("|") for line in lines} == {7}


# --------------------------------------------------------------------- metrics_json


class TestMetricsJson:
    METRICS = {
        "Strategy (net)": {
            "n_days": np.int64(6410),
            "first_date": pd.Timestamp("2001-04-02"),
            "last_date": None,
            "sharpe": np.float64(0.34216),
            "calmar": float("nan"),
            "skew": float("inf"),
            "n_trials": 1,
        },
        "^BCOM": {"sharpe": 0.135},
    }

    def test_round_trips_with_nan_as_null_and_dates_as_iso_strings(self):
        out = json.loads(metrics_json(self.METRICS))
        assert list(out) == ["Strategy (net)", "^BCOM"]
        s = out["Strategy (net)"]
        assert s == {
            "n_days": 6410,
            "first_date": "2001-04-02",
            "last_date": None,
            "sharpe": 0.34216,
            "calmar": None,
            "skew": None,
            "n_trials": 1,
        }
        assert type(s["n_days"]) is int

    def test_strict_json_no_nan_literals(self):
        text = metrics_json(self.METRICS)
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard constant {c}"))

    def test_layout_is_fixed(self):
        text = metrics_json({"A": {"x": 1.5, "y": 2}})
        assert text == '{\n  "A": {\n    "x": 1.5,\n    "y": 2\n  }\n}\n'

    def test_deterministic_and_input_untouched(self):
        before = repr(self.METRICS)
        assert metrics_json(self.METRICS) == metrics_json(self.METRICS)
        assert repr(self.METRICS) == before


# ------------------------------------------------------------ inject_between_markers


DOC = (
    "# Title\n\nIntro.\n\n## Results\n\n"
    f"{METRICS_START}\n_Not yet generated._\n{METRICS_END}\n\n## After\n\nTail.\n"
)


class TestInjectBetweenMarkers:
    def test_markers_are_the_readme_ones(self):
        assert METRICS_START == "<!-- METRICS:START -->"
        assert METRICS_END == "<!-- METRICS:END -->"

    def test_replaces_only_the_block(self):
        out = inject_between_markers(DOC, "| a | b |\n|---|---|\n")
        assert out == (
            "# Title\n\nIntro.\n\n## Results\n\n"
            f"{METRICS_START}\n| a | b |\n|---|---|\n{METRICS_END}\n\n## After\n\nTail.\n"
        )

    def test_idempotent_and_replaceable(self):
        once = inject_between_markers(DOC, "X\n")
        assert inject_between_markers(once, "X\n") == once
        assert inject_between_markers(once, "Y") == inject_between_markers(DOC, "Y")

    def test_block_newlines_are_normalised(self):
        assert inject_between_markers(DOC, "\n\nX\n\n") == inject_between_markers(DOC, "X")

    def test_custom_markers(self):
        out = inject_between_markers("a<s>old<e>b", "new", start="<s>", end="<e>")
        assert out == "a<s>\nnew\n<e>b"

    @pytest.mark.parametrize(
        "text",
        [
            "no markers at all\n",
            f"{METRICS_START}\nonly the start\n",
            f"only the end\n{METRICS_END}\n",
            f"{METRICS_END}\nreversed\n{METRICS_START}\n",
            f"{METRICS_START}\n{METRICS_START}\ntwice\n{METRICS_END}\n",
            f"{METRICS_START}\n{METRICS_END}\n{METRICS_END}\n",
        ],
    )
    def test_malformed_markers_raise(self, text):
        with pytest.raises(ValueError):
            inject_between_markers(text, "X")

    def test_the_repository_readme_takes_an_injection(self):
        readme = (ROOT / "README.md").read_text()
        out = inject_between_markers(readme, "TABLE")
        before, after = readme.split(METRICS_START)[0], readme.split(METRICS_END)[1]
        assert out.startswith(before + METRICS_START + "\nTABLE\n" + METRICS_END)
        assert out.endswith(METRICS_END + after)


# ------------------------------------------------------------------ format_frame_md


class TestFormatFrameMd:
    def test_kinds(self):
        assert FRAME_KINDS == ("pct", "ratio", "prob", "int", "date", "text", "flag")
        assert NUMERIC_KINDS == ("pct", "ratio", "prob", "int")

    def test_exact_rendering(self):
        f = pd.DataFrame(
            {
                "Series": ["Strategy (net)", "^BCOM"],
                "Return": [0.031662, -0.4042],
                "Sharpe": [0.34216, float("nan")],
                "PSR": [0.95761, None],
                "Days": [6410, 6356],
                "First": [pd.Timestamp("2001-04-02"), pd.NaT],
                "Picked": [True, False],
            }
        )
        kinds = {
            "Series": "text",
            "Return": "pct",
            "Sharpe": "ratio",
            "PSR": "prob",
            "Days": "int",
            "First": "date",
            "Picked": "flag",
        }
        assert format_frame_md(f, kinds) == (
            "| Series | Return | Sharpe | PSR | Days | First | Picked |\n"
            "|---|---:|---:|---:|---:|---|---|\n"
            "| Strategy (net) | 3.17% | 0.34 | 0.958 | 6410 | 2001-04-02 | yes |\n"
            "| ^BCOM | -40.42% | n/a | n/a | 6356 | n/a | - |\n"
        )

    def test_numpy_scalars_and_infinity(self):
        f = pd.DataFrame({"x": np.array([1.5, np.inf]), "n": np.array([3, 4], dtype=np.int64)})
        assert format_frame_md(f, {"x": "ratio", "n": "int"}) == (
            "| x | n |\n|---:|---:|\n| 1.50 | 3 |\n| n/a | 4 |\n"
        )

    def test_index_is_not_printed_and_column_names_are_strings(self):
        f = pd.DataFrame({2007: [0.1], 2008: [0.2]}, index=["ignored"])
        assert format_frame_md(f, {2007: "pct", 2008: "pct"}) == (
            "| 2007 | 2008 |\n|---:|---:|\n| 10.00% | 20.00% |\n"
        )

    def test_empty_frame_gives_the_header(self):
        f = pd.DataFrame({"a": pd.Series([], dtype=float)})
        assert format_frame_md(f, {"a": "ratio"}) == "| a |\n|---:|\n"

    def test_missing_or_unknown_kind_raises(self):
        f = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError):
            format_frame_md(f, {"a": "ratio"})
        with pytest.raises(ValueError):
            format_frame_md(f, {"a": "ratio", "b": "money"})

    def test_input_not_modified(self):
        f = pd.DataFrame({"a": [0.5, float("nan")]})
        before = f.copy()
        format_frame_md(f, {"a": "pct"})
        pd.testing.assert_frame_equal(f, before)


# --------------------------------------------------------- robustness report composers


def tiny_is_oos():
    table = pd.DataFrame(
        {
            "sharpe_is": [0.2, 0.6],
            "sharpe_oos": [0.3, 0.1],
            "degradation": [-0.1, 0.5],
            "n_days_is": [2900, 2850],
            "n_days_oos": [3450, 3450],
            "selected": [False, True],
            "default": [True, False],
        },
        index=pd.Index([126, 252], name="lookback"),
    )
    summary = {
        "selected_lookback": 252,
        "selected_is": 0.6,
        "selected_oos": 0.1,
        "selected_degradation": 0.5,
        "default_lookback": 126,
        "default_is": 0.2,
        "default_oos": 0.3,
        "default_degradation": -0.1,
        "oos_gap": -0.2,
    }
    return table, summary


class TestRobustnessReports:
    def test_is_oos_md(self):
        table, summary = tiny_is_oos()
        text = is_oos_md(table, summary, "2012-12-31", "2013-01-01", 5.0)
        assert text.startswith("# In-sample / out-of-sample selection\n")
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert "| Lookback | Sharpe IS | Sharpe OOS | IS - OOS |" in text
        assert "| 252 | 0.60 | 0.10 | 0.50 | 2850 | 3450 | yes | - |" in text
        assert "| Out of sample, selected minus a-priori | -0.20 |" in text
        assert "2012-12-31" in text and "2013-01-01" in text and "5 bps" in text

    def test_subperiods_md_carries_the_hindsight_caveat(self):
        blocks = pd.DataFrame(
            {
                "block": ["2000-2007"],
                "series": ["Strategy (net)"],
                "first_date": [pd.Timestamp("2001-04-02")],
                "last_date": [pd.Timestamp("2007-12-31")],
                "n_days": [1700],
                "ann_return": [0.05],
                "ann_vol": [0.1],
                "sharpe": [0.5],
            }
        )
        years = pd.DataFrame(
            {"Strategy (net)": [0.1, -0.02]}, index=pd.Index([2001, 2002], name="year")
        )
        rolling = pd.DataFrame(
            {
                "n_windows": [10],
                "share_positive": [0.7],
                "min": [-0.4],
                "min_date": [pd.Timestamp("2019-12-31")],
                "max": [1.2],
                "max_date": [pd.Timestamp("2008-12-31")],
                "last": [0.3],
                "last_date": [pd.Timestamp("2026-09-07")],
            },
            index=pd.Index(["Strategy (net)"]),
        )
        regimes = pd.DataFrame(
            {
                "regime": ["calm", "stress"],
                "series": ["Strategy (net)"] * 2,
                "n_days": [4800, 1600],
                "share_of_days": [0.75, 0.25],
                "ann_return": [0.02, 0.07],
                "ann_vol": [0.1, 0.12],
                "sharpe": [0.2, 0.6],
            }
        )
        text = subperiods_md(blocks, years, rolling, regimes, 3, 0.75, 60)
        assert text.startswith("# Sub-periods\n")
        assert HINDSIGHT_CAVEAT in text
        assert "hindsight" in HINDSIGHT_CAVEAT.lower()
        assert "| 2000-2007 | Strategy (net) | 2001-04-02 | 2007-12-31 | 1700 | 5.00% |" in text
        assert "| 2001 | 10.00% |" in text and "| 2002 | -2.00% |" in text
        assert "| Strategy (net) | 10 | 70.00% | -0.40 | 2019-12-31 |" in text
        assert "| stress | Strategy (net) | 1600 | 25.00% |" in text
        assert "trailing 60-day volatility" in text and "75% quantile" in text
        assert "Trailing 756 observations" in text

    def test_sweeps_md(self):
        trials = {
            "n_trials": 192,
            "sharpe_std": 0.12,
            "sharpe_hurdle": 0.33,
            "psr": 0.958,
            "dsr": 0.51,
        }
        dist = {
            "n_trials": 192,
            "sharpe_std": 0.12,
            "sharpe_min": -0.2,
            "sharpe_q25": 0.1,
            "sharpe_median": 0.2,
            "sharpe_q75": 0.3,
            "sharpe_max": 0.5,
            "share_positive": 0.9,
            "default_sharpe": 0.34,
            "default_rank": 20,
            "best_trial": "L 126 / vol 20 rolling / daily / skip 0 / lag 1",
            "worst_trial": "L 63 / vol 120 ewma / daily / skip 21 / lag 2",
        }
        costs = pd.DataFrame(
            {
                "ann_return": [0.0345, 0.0317],
                "sharpe": [0.37, 0.34],
                "max_drawdown": [-0.39, -0.40],
                "grid_median_sharpe": [0.25, 0.2],
                "grid_share_positive": [0.95, 0.9],
            },
            index=pd.Index([0.0, 5.0], name="cost_bps"),
        )
        lags = pd.DataFrame(
            {
                "sharpe_gross": [0.37, 0.35],
                "sharpe_net": [0.34, 0.32],
                "ann_return_net": [0.0317, 0.03],
                "grid_median_net": [0.2, 0.19],
            },
            index=pd.Index([1, 2], name="lag"),
        )
        text = sweeps_md(trials, dist, costs, lags, 5.0)
        assert text.startswith("# Robustness: trials, grid and sweeps\n")
        assert "| Trials (configurations evaluated) | 192 |" in text
        assert "| Deflated Sharpe of the default (net) | 0.510 |" in text
        assert "| Default rank (1 = best) | 20 of 192 |" in text
        assert "| 0 | 3.45% | 0.37 | -39.00% | 0.25 | 95.00% |" in text
        assert "| 2 | 0.35 | 0.32 | 3.00% | 0.19 |" in text
