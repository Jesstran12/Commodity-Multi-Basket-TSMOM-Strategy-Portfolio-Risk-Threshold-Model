# Systematic Commodity Futures Strategy — Spec & Feature Log

> **How we use this file.** This is the single source of truth for what we are building and
> where we are. Every working session starts by reading "Current State" below, then works the
> phase it names, one phase per session. Each phase is delivered as a self-contained
> assignment: the tests and function contracts are written and validated first
> (`PHASE<N>_INSTRUCTIONS.md`), the marked TODOs are implemented against them, the phase's
> numbers are reported back and cross-checked, and only then is the phase marked done.
> Statuses flip here, Notes are filled in, and a Session Log line is appended.

## Current State

| | |
|---|---|
| **Status** | **Phase 6 `SCAFFOLDED`** 2026-10-15: the assignment is on `main` (`PHASE6_INSTRUCTIONS.md`; TODOs 60–77 across the new `src/robustness.py` and `src/subperiods.py`, `src/report.py` and `src/cli.py`; Decisions 58–61); starting state `129 failed / 526 passed`, complete at `655 passed`, validated against a reference implementation. Phase 5 `DONE` 2026-10-15 (closed on `ba1d940`, `results/` in `5c70ec2`; Decision 57). Phases 0–4 `DONE` (Phase 4 on `a280750`, tagged `v0.2`; Phase 3 on `fa431cb`; Phase 2 on `2eb0d1c`; Phases 0–1 `v0.1`). `N_trials` = 0 until the grid lands. |
| **Next up** | Implement Phase 6 on `phase-6`, one pull request, cross-check, merge, tag `v0.3`. |
| **Last session** | 2026-10-15 — Phase 6 assignment prepared |
| **Blockers** | None. |

---

## 1. Objective

Build, backtest, and evaluate a rules-based long/short time-series-momentum strategy over a
basket of liquid commodity futures, using free daily data, and present the result the way a
systematic desk would review it: gross and net of costs, vol-targeted, walk-forward, benchmarked,
with parameter sensitivity and an explicit overfitting discussion.

**What it proves:** the ability to research a signal, engineer a lookahead-free backtest, account
for costs, and reason honestly about robustness and data limitations.

**What it is not:** a search for the best-performing strategy. Few parameters, chosen a priori,
with the sensitivity of results to each one shown rather than hidden.

## 2. Strategy

### 2.1 Signal: time-series momentum (TSMOM)

- For each instrument `i` on date `t`, compute trailing return over `L` trading days:
  `r_i(t-L, t) = P_i(t) / P_i(t-L) - 1` (on the cleaned price series, see §3.3).
- Signal `s_i(t) = sign(r_i(t-L, t))`, taking values `{+1, -1}`; `0` if the trailing return is
  exactly zero; `NaN` if fewer than `L+1` valid observations exist.
- **Default `L = 252`** (~12 months). No skip-month by default (matches Moskowitz, Ooi &
  Pedersen 2012 futures TSMOM). Skip-month (`skip = 21`) is a robustness variant.
- **Economic rationale (for README):** slow diffusion of information, investor under- and
  over-reaction, hedger-driven flows in commodity markets, and risk-premium persistence. TSMOM
  is documented across asset classes and decades; it is not a pattern discovered in this dataset.

### 2.2 Position sizing: two-stage volatility targeting

1. **Per-asset scaling.** `w_raw_i(t) = s_i(t) * (sigma_target_asset / sigma_hat_i(t)) / N_live(t)`
   where `sigma_hat_i` is the annualised realised vol estimate, `N_live(t)` is the number of
   instruments with a valid signal that day, and `sigma_target_asset = 0.10`.
2. **Portfolio scaling.** Compute the trailing realised vol of the pre-scaled portfolio return
   `sum_i w_raw_i * r_i` over the same vol window, using only past data, and scale all weights by
   `sigma_target_portfolio / sigma_hat_p(t)` with `sigma_target_portfolio = 0.10`.
3. **Leverage cap.** Clip so that `sum_i |w_i(t)| <= 3.0` (gross). Cap is a config parameter and
   the fraction of days it binds is reported.

Vol estimators (both implemented, `rolling` is default):
- `rolling`: std of daily returns over `vol_window = 60` days, annualised by `sqrt(252)`.
- `ewma`: exponentially weighted std with `span = 60`.
Sweep `vol_window in {20, 60, 120}` and estimator choice in robustness.

### 2.3 Rebalancing and execution timing

- **Default: monthly.** On the last trading day of each month, compute signal and vol from that
  close; hold weights unchanged until the next rebalance date. Daily rebalancing is a robustness
  variant.
- **Execution lag = 1 day (default).** Weights computed from close `t` earn the return from close
  `t` to close `t+1`. In code: `positions = weights.shift(lag)`. This assumes trading at the same
  close the signal was computed on, which is slightly optimistic; `lag = 2` is a robustness variant.
- Weights between rebalances are held **constant in weight terms** (not drifted by price). This is
  a simplification; document it.

### 2.4 Costs

- Turnover `TO(t) = sum_i |pos_i(t) - pos_i(t-1)|`.
- Cost `c(t) = TO(t) * cost_bps / 1e4`, **default `cost_bps = 5`** (per unit of notional traded,
  one-way). Sweep `{0, 5, 10, 20}`.
- Net return `r_net(t) = r_gross(t) - c(t)`. **Every table and figure shows gross and net.**
- Roll costs are not modelled explicitly (roll dates are unobservable in the data used). Stated
  as a limitation; a per-instrument cost table is an optional extension (F9.4).

## 3. Data

### 3.1 Universe (candidates, subject to the quality gate in §3.4)

| Sector    | Ticker | Name         |
|-----------|--------|--------------|
| Energy    | CL=F   | WTI crude    |
| Energy    | NG=F   | Natural gas  |
| Energy    | HO=F   | Heating oil  |
| Metals    | GC=F   | Gold         |
| Metals    | SI=F   | Silver       |
| Metals    | HG=F   | Copper       |
| Metals    | PL=F   | Platinum     |
| Grains    | ZC=F   | Corn         |
| Grains    | ZW=F   | Wheat        |
| Grains    | ZS=F   | Soybeans     |
| Softs     | KC=F   | Coffee       |
| Softs     | SB=F   | Sugar #11    |
| Softs     | CT=F   | Cotton       |
| Livestock | LE=F   | Live cattle  |

Source: yfinance continuous front-month series, daily, `Adj Close` (falls back to `Close`).
History: maximum available (~2000 onward for most).

### 3.2 Panel construction

- Calendar: **union** of all instruments' trading days.
- Gaps: forward-fill prices up to `ffill_limit = 5` consecutive days; a filled day has zero return.
  Longer gaps stay `NaN` and the instrument is excluded from the portfolio until data resumes.
- Ragged start: each instrument enters when it has `L+1` valid observations. No truncation to a
  common start date. Early years have fewer live names; `N_live(t)` is reported over time.
- A **data-quality report** (`results/data_quality.md`) lists per instrument: first/last date,
  row count, fill count and fraction, number of clipped return days (§3.3), and gate outcome.

### 3.3 Returns and outlier handling

- WTI settled at -$37.63 on 2020-04-20; `pct_change` through zero is meaningless. Daily return is
  therefore defined as `r(t) = (P(t) - P(t-1)) / |P(t-1)|`, then **clipped to
  `[-clip, +clip]` with `clip = 0.50`**. Every clipped observation is logged (date, ticker, raw
  value) in the data-quality report. This one rule also tames roll gaps and bad prints.
- Cumulative price index for the signal is rebuilt from cleaned returns:
  `P_clean(t) = prod(1 + r(t))`, so the momentum signal is computed on the same series the
  P&L is computed on.

### 3.4 Universe quality gate (applied automatically, reported)

Include an instrument in the default universe only if: `>= 10` years of history at snapshot date;
`<= 5%` of days forward-filled; no single gap `> 20` trading days after its first valid date.
Failures are listed with reasons; the final universe is written to `results/universe.json`.

### 3.5 Reproducibility

- `data/futures_panel.csv` (cleaned aligned prices) and `data/raw/<ticker>.csv` are **committed**,
  with `data/manifest.json` recording pull timestamp, yfinance version, per-ticker row counts,
  date ranges, and SHA-256 of each file.
- The pipeline reads the snapshot by default. `make data` re-pulls, rewrites the snapshot and
  manifest, and prints a diff summary. README results are tied to the committed snapshot.
- **Pluggable loader:** `data.py` exposes a `PriceSource` protocol with `YFinanceSource` and
  `CSVSource`. A back-adjusted continuous series (e.g. from a paid vendor) can be dropped in via
  config without touching the engine.

### 3.6 Known data limitations (to state in README)

Front-month continuous series contain roll gaps (unadjusted). No survivorship issue within the
basket because we fix the universe a priori, but the basket itself is chosen with knowledge that
these markets exist and are liquid today. No intraday data, no bid/ask, no margin modelling.

## 4. Benchmarks

1. **EW long-only, monthly rebalanced, same universe, same cost model** — the primary benchmark.
2. **EW long-only, vol-targeted to 10%** with the same portfolio scaling as the strategy, so the
   comparison is not just about leverage.
3. **External index:** `DBC` (Invesco DB Commodity Index Tracking ETF, from 2006-02) via yfinance;
   try `^BCOM` first if available. Aligned to the strategy's dates; shorter history noted.

## 5. Evaluation

### 5.1 Metrics (all computed on daily returns; rf = 0, stated)

Annualised return (geometric), annualised vol, Sharpe, Sortino, max drawdown, max drawdown
duration, Calmar, hit rate (daily and monthly), skew, kurtosis, worst month, best month,
annualised turnover, average gross leverage, fraction of days leverage cap binds, average
`N_live`.

### 5.2 Statistical inference

- **Sharpe t-stat** `SR * sqrt(T / 252)` and **95% bootstrap CI** on Sharpe via stationary block
  bootstrap (block length 20 days, 2000 resamples, fixed seed).
- **Probabilistic Sharpe Ratio** (Bailey & López de Prado) against benchmark SR = 0.
- **Deflated Sharpe Ratio** using `N_trials` = number of distinct strategy configurations
  evaluated in the robustness grid (§5.3), logged automatically by the grid runner. The README
  states the number and the Sharpe hurdle it implies.

### 5.3 Robustness

- **Parameter grid:** `L in {63, 126, 189, 252}` x `vol_window in {20, 60, 120}` x
  `estimator in {rolling, ewma}` x `rebalance in {monthly, daily}` x `skip in {0, 21}` x
  `lag in {1, 2}`; costs `{0, 5, 10, 20}` bps applied to each. Results cached to
  `results/grid.parquet`.
- **In-sample / out-of-sample:** IS = start to 2012-12-31, OOS = 2013-01-01 to end. Select the
  best `L` (and only `L`) on IS by net Sharpe, report its OOS Sharpe next to the a-priori
  default's OOS Sharpe. Report the IS-to-OOS degradation. The main headline result remains the
  a-priori config; the IS/OOS exercise is there to show what selection would have cost.
- **Sub-periods (fixed a priori):** calendar blocks 2000-2007, 2008-2013, 2014-2019,
  2020-present; per-calendar-year bars; rolling 3-year Sharpe. A regime-labelled view
  (e.g. crisis vs calm) is included **with an explicit hindsight caveat** in the caption.
  Defined as calm versus stress: a day is stress when the equal-weight long-only basket's
  trailing `vol_window`-day vol is above its full-sample `robustness.regime_vol_quantile`
  (0.75) quantile; the full-sample threshold is the hindsight (Decision 61).

## 6. Deliverables

- `README.md`: rationale, method, gross/net headline table vs benchmarks, robustness story,
  inference, limitations, how to reproduce. Metrics table auto-injected between markers.
- Figures in `figures/`: `equity_curve.png` (log scale; gross, net, three benchmarks),
  `drawdown.png`, `param_sensitivity.png` (heatmap `L` x `vol_window` of net Sharpe, plus net
  Sharpe vs cost bps lines), `returns_by_subperiod.png` (blocks + per-year bars). Extras:
  `rolling_sharpe.png`, `is_oos.png`, `exposure.png`.
- `results/`: `metrics.json`, `metrics_table.md`, `returns_daily.csv`, `positions.csv`,
  `grid.parquet`, `data_quality.md`, `universe.json`, `resolved_config.yaml` per run.
- `notebooks/strategy.ipynb`: thin narrative that imports `src/` and loads `results/`;
  executed headless in CI. No logic lives only in the notebook.
- Tests in `tests/` with the integrity tests in §8; CI green badge.

## 7. Architecture

```
commodity-tsmom/                 # working name
├── README.md  PROJECT_LOG.md  WORKLOG.md  SETUP.md  LICENSE (MIT)
├── PHASE<N>_INSTRUCTIONS.md   # one per handed-off phase: contracts, TODO map, expected numbers
├── pyproject.toml  uv.lock  requirements.txt (exported)  Makefile  ruff.toml
├── config/default.yaml
├── data/  raw/<ticker>.csv  futures_panel.csv  benchmark.csv  manifest.json
├── src/
│   ├── __init__.py
│   ├── config.py      # dataclasses + YAML loader + override + resolved dump
│   ├── data.py        # PriceSource protocol, YFinanceSource, CSVSource, align, clean, quality report
│   ├── signals.py     # tsmom(prices, lookback, skip) -> DataFrame of {-1,0,1,NaN}
│   ├── portfolio.py   # vol estimators, per-asset scaling, rebalance schedule, portfolio scaling, cap
│   ├── backtest.py    # run(weights, returns, lag, cost_bps) -> BacktestResult
│   ├── benchmark.py   # EW long-only, EW vol-targeted, external index
│   ├── metrics.py     # §5.1 summary stats on a return series
│   ├── inference.py   # §5.2 t-stat, stationary block bootstrap CI, PSR/DSR
│   ├── robustness.py  # grid, trial evaluation, IS/OOS selection, cost and lag sweeps
│   ├── subperiods.py  # calendar blocks and years, rolling Sharpe, regime view
│   ├── plots.py       # figure functions, one per file
│   ├── report.py      # metrics table md/json, README marker injection
│   └── cli.py         # `python -m src.cli <stage>`; Makefile calls this
├── tests/
├── notebooks/strategy.ipynb
├── figures/
├── results/
└── .github/workflows/ci.yml
```

Data flow: `data` -> `signals` + `portfolio` -> `backtest` -> `metrics` / `robustness` ->
`plots` + `report`. Each module is pure functions over pandas objects, no global state, no I/O
except in `data.py`, `report.py`, `plots.py`, and `cli.py`.

Make targets: `setup`, `data` (re-pull), `panel`, `backtest`, `benchmarks`, `metrics`, `robustness`,
`figures`, `report`, `notebook`, `test`, `lint`, `all` (everything from snapshot), `clean`.

## 8. Integrity tests (non-negotiable, must exist before Phase 4 is marked done)

- **No lookahead:** perturb all prices after date `t` by random noise; assert positions on and
  before `t` are bit-identical. Applied to signal, vol estimate, weights, and positions.
- **Lag:** positions at `t` equal weights at `t - lag`.
- **Zero-turnover, zero-cost:** constant weights produce cost identically zero.
- **Cost accounting:** a single full flip of one unit weight costs exactly `2 * bps / 1e4`.
- **Synthetic trend:** on a synthetic monotonically rising price, signal is +1 after warm-up;
  on falling, -1; on a flat price, 0.
- **Vol targeting:** on synthetic i.i.d. returns with known vol, realised portfolio vol is within
  tolerance of target.
- **Negative price:** a synthetic series through zero yields finite, clipped returns and flagged
  rows.
- **Gap fill:** a gap of `ffill_limit` days is filled with zero returns; `ffill_limit + 1` stays NaN.
- **Rebalance schedule:** weights change only on rebalance dates in monthly mode.
- **Metrics known-answer:** max drawdown, Sharpe, Sortino on hand-computed tiny series.
- **Determinism:** two full pipeline runs on the snapshot produce identical `metrics.json`.

## 9. Feature Log

Status values: `TODO` | `SCAFFOLDED` | `IN PROGRESS` | `DONE` | `BLOCKED` | `DROPPED`.
A phase goes `TODO` → `SCAFFOLDED` (its `PHASE<N>_INSTRUCTIONS.md`, tests and stubs are on
`main`) → `DONE` (the implementation is verified and merged). Each feature lists acceptance
criteria; the tests shipped with the handoff are those criteria made executable. Notes are
filled in when the status changes: a scaffold note when handed off, a completion note when
verified.

### Phase 0 — Setup

- [x] **F0.1 Repo scaffold** — `DONE`
  uv project, Python 3.12, `pyproject.toml` with deps (pandas, numpy, yfinance, pyyaml, matplotlib,
  pyarrow, pytest, ruff, jupyter, nbconvert), ruff config, pytest config, `.gitignore`, MIT LICENSE,
  Makefile with all targets stubbed, README stub stating signal and economic rationale,
  `SETUP.md` (install uv, clone, `make setup`, `make lint test`, day-to-day commands, the
  no-refetch rule), `requirements.txt` exported. Accept: `make lint test` runs green on an
  empty test; `SETUP.md` gets a fresh machine to a green suite. Notes: shipped 2026-09-04; `make lint test` green with 34 tests.
- [x] **F0.2 CI** — `DONE`
  GitHub Actions: ruff + pytest on push to `main` and on every PR, `uv sync --locked` so a stale
  lockfile fails; later stages add notebook execution. Accept: workflow file present and valid
  (a test parses it and checks the triggers and commands); README badge points at this
  repository; first run on GitHub is green. Notes: shipped 2026-09-04; first run on `main` green (run 33843275538).
- [x] **F0.3 Config system** — `DONE`
  `config/default.yaml` covering every parameter in §2-§5; `src/config.py` typed dataclasses,
  loader, dotted-key overrides, `resolved_config.yaml` dump. Accept: unit tests for load,
  override, validation errors. Notes: shipped 2026-09-04; 29 config tests.

### Phase 1 — Data panel

- [x] **F1.1 Fetch layer** — `DONE`
  `PriceSource` protocol; `YFinanceSource` pulls each ticker, writes `data/raw/<ticker>.csv`,
  writes `manifest.json`. Retries and clear errors on failure. Accept: tests with a fake source;
  live pull done once manually. Notes: scaffolded 2026-09-04 (TODOs 1–4); the one live pull was run 2026-09-04 with yfinance 1.7.0 and is committed.
  Done 2026-09-14: TODOs 1–4, `TestExtractPriceField`, `TestYFinanceSource`, `TestCSVSource`
  and `TestFetchAll` green; the committed manifest records the pull as 2026-09-08T15:07:58Z,
  yfinance 1.7.0 (a re-pull after the 2026-09-04 one; Decision 38).
- [x] **F1.2 Alignment and gap handling** — `DONE`
  Union calendar, ffill with limit, fill counts. Accept: gap-fill tests in §8. Notes: scaffolded
  2026-09-04 (TODO 5); all-or-nothing run filling (Decision 33). Done 2026-09-14: `TestAlignPrices`
  and `TestIntegrityGapFill` green; fill shares 0.21%–0.54% per instrument, none above 1%.
- [x] **F1.3 Cleaned returns** — `DONE`
  `dP/|P_{t-1}|`, clip, flagged-rows log, rebuilt clean price index. Accept: negative-price test,
  WTI 2020-04-20/21 appear in the flagged log on real data. Notes: scaffolded 2026-09-04
  (TODOs 6–7). Done 2026-09-14: `TestCleanReturns` and `TestCleanPriceIndex` green; on real data
  exactly 2 clipped days, both `CL=F`: 2020-04-20 raw −3.0597 → −0.50, 2020-04-21 raw +1.2660 → +0.50.
- [x] **F1.4 Quality gate and data-quality report** — `DONE`
  Apply §3.4, write `results/universe.json` and `results/data_quality.md`. Accept: report lists
  all 14 candidates with pass/fail and reasons. Notes: scaffolded 2026-09-04 (TODOs 8–9);
  gate definitions in Decision 34. Done 2026-09-14: `TestQualityGate`, `TestUniverseDict` and
  `TestDataQualityReport` green; `results/data_quality.md` lists all 14, 13 pass, `PL=F` fails
  `gap 53d > 20d`, `LE=F` passes at exactly 20.
- [x] **F1.5 Snapshot commit and CSVSource** — `DONE`
  Commit `futures_panel.csv`, raw CSVs, manifest. `CSVSource` loads an arbitrary wide CSV.
  Accept: `make panel` from snapshot needs no network; manifest hashes verified by a test.
  Notes: scaffolded 2026-09-04 (TODOs 11–13); snapshot committed with the scaffold (Decision 32):
  panel = gate passers on their own calendar (Decision 35). 2026-09-14: TODOs 11–13 written,
  `TestManifest`, `TestRunPanel` and `TestSnapshot` green; `make panel` runs offline and reproduces
  the panel byte for byte; panel and manifest rebuilt from the committed raw files (Decision 38).
  Closed 2026-09-14: the rebuilt panel, manifest and `results/` are committed on `main`, CI is
  green, and the panel was reproduced by a reference implementation on the committed raw files.
- [x] **F1.6 External benchmark series** — `DONE`
  Fetch `^BCOM` else `DBC`; store alongside. Accept: aligned series available to `benchmark.py`.
  Notes: scaffolded 2026-09-04 (TODO 10); stored as `data/benchmark.csv` (Decision 36). Done
  2026-09-14: `TestFetchBenchmark` green; the file holds `^BCOM`, 8,923 rows, 1991-01-02 to
  2026-07-17; alignment to the strategy's dates is the consumer's job in a later phase.

### Phase 2 — Signal

- [x] **F2.1 TSMOM signal** — `DONE`
  `tsmom(prices, lookback, skip)`; NaN warm-up; sign semantics per §2.1. Accept: synthetic-trend
  tests; no-lookahead perturbation test. Notes: scaffolded 2026-09-15 (TODOs 14–15
  `trailing_return` and `tsmom`, plus TODO 16 `signal_summary` for the phase's report);
  window and NaN rule in Decision 40, input series in Decision 41; the expected real-data
  numbers are in `PHASE2_INSTRUCTIONS.md` §5, validated against a reference implementation.
  Closed 2026-09-20: implementation `2eb0d1c` (TODOs 14–16) verified against the reference,
  same construction; `147 passed`, 18 lookahead tests green; §5 output identical to the
  expected tables; CI green on `main`.
- [x] **F2.2 Lookahead test harness** — `DONE`
  Reusable `assert_no_lookahead(fn, prices, t)` utility in `tests/integrity.py`, applied to every
  downstream stage as it lands. Accept: used by F2.1, F3.x, F4.1 tests. Notes: delivered
  complete with the 2026-09-15 scaffold (test infrastructure, not a TODO; design in
  Decision 42) with 10 tests of its own in `tests/test_integrity.py`, green in the shipped
  state; used by 8 signal tests.
  Closed 2026-09-20 with F2.1: harness unchanged by the implementation (`tests/` diff
  empty), 10 harness tests and 8 signal-side uses green.

### Phase 3 — Position sizing

- [x] **F3.1 Vol estimators** — `DONE`
  `rolling` and `ewma`, annualised, NaN warm-up. Accept: known-answer tests; no-lookahead.
  Notes: scaffolded 2026-09-21 (TODO 17 `realised_vol`; contracts in Decision 43); the
  expected real-data numbers are in `PHASE3_INSTRUCTIONS.md` §5, validated against a
  reference implementation. 2026-09-28: TODO 17 written; `TestRealisedVol` passing. Open:
  the no-lookahead tests (need TODOs 21 and 23).
  Closed 2026-10-01: TODO 17 (`fa431cb`) matches the reference implementation in substance
  (rolling `min_periods = window`, EWMA masked on NaN-return rows, `sqrt(ANNUALISATION)`);
  the per-instrument median vols identical to the expected table.
- [x] **F3.2 Per-asset vol scaling** — `DONE`
  §2.2 step 1 with `N_live`. Accept: unit tests incl. ragged panel. Notes: scaffolded
  2026-09-21 (TODO 18 `asset_weights`; the "live" rule in Decision 44); the unsizable
  instrument-days are the phase's "what binds" count. 2026-09-28: TODO 18 written;
  `TestAssetWeights` passing. Open: `TestIntegrityVolTargeting` and the no-lookahead tests
  (need TODOs 21–23).
  Closed 2026-10-01: TODO 18 is the same construction as the reference (live mask, per-asset
  target, row-wise divide by `N_live`); mean `N_live` 12.80 and 128 unsizable instrument-days
  (LE=F 101, ZC=F 19, ZW=F 8) reproduced.
- [x] **F3.3 Rebalance schedule** — `DONE`
  Monthly (last trading day) and daily; hold weights between rebalances. Accept: schedule test in
  §8. Notes: scaffolded 2026-09-21 (TODOs 19–20 `rebalance_dates`, `hold`; Decision 46);
  the §8 rebalance-schedule test ships with it. 2026-09-28: TODOs 19–20 written;
  `TestRebalanceDates`, `TestHold` and the §8 `TestIntegrityRebalanceSchedule` passing.
  Left unticked: the full suite is not green, nothing is committed and CI has not run.
  Closed 2026-10-01: TODOs 19–20 are the same construction as the reference (`to_period("M")`
  neighbour test with the last row kept; `loc[dates].reindex(method="ffill")` copying whole
  rows); 321 rebalance dates, first weights 2001-03-30 and 6,411 days with a position
  reproduced; the §8 schedule tests green; `248 passed`.
- [x] **F3.4 Portfolio vol scaling and leverage cap** — `DONE`
  §2.2 steps 2-3; cap-binding fraction exposed. Accept: vol-targeting synthetic test; no-lookahead.
  Notes: scaffolded 2026-09-21 (TODOs 21–23 `portfolio_vol`, `leverage_cap`,
  `size_positions`, plus TODO 24 `sizing_summary` for the report; Decisions 45 and 47); the
  §8 vol-targeting test and the no-lookahead tests on vol, weights and the book vol ship
  with it. 2026-09-30: TODOs 21–23 written; `TestPortfolioVol`, `TestLeverageCap`,
  `TestSizePositions` and the `TestIntegrity*` classes (vol-targeting, rebalance schedule,
  no-lookahead) passing. Open: TODO 24, full suite, §5 numbers, CI.
  Closed 2026-10-01: TODOs 21–24 are the same construction as the reference, including the
  fixed one-row shift in the book and the float round trip that holds the cap flag; every
  headline, per-year (26 rows) and per-instrument (13 rows) number identical to the expected
  tables, compared mechanically, zero mismatches; the cap binds on 0 of 6,411 days; `248
  passed`, 40 lookahead tests green, `tests/`, `data/`, `config/` untouched. Merged by squash
  without a pull request (Decision 48).

### Phase 4 — Backtest engine and costs

- [x] **F4.1 Engine** — `DONE`
  `run(weights, returns, lag) -> BacktestResult` (positions, gross returns, turnover, leverage,
  N_live). Accept: lag test; determinism test. Notes: scaffolded 2026-10-02 (TODOs 25
  `positions_from_weights`, 28 `gross_returns`, 29 `run` in `src/backtest.py`; contracts in
  Decision 49: `positions = weights.shift(lag)` with `lag >= 1` enforced, gross return NaN
  before the first position and `0` when flat after it, unpriced position-days earn nothing
  and are counted). Lag tests at 1, 2 and 5 rows; 11 new no-lookahead tests on positions,
  gross and net returns, turnover and leverage. Expected numbers in `PHASE4_INSTRUCTIONS.md`
  §5, validated against a reference implementation.
  Progress 2026-10-05: TODOs 25, 28, 29 implemented on `phase-4`; `TestPositionsFromWeights`,
  `TestGrossReturns`, `TestRun`, `TestIntegrityLag` passing. Left open: the determinism test
  runs through the CLI stage (TODO 40), and the §5 numbers are not yet cross-checked.
  Progress 2026-10-08: all of Phase 4 written; `make lint test` ruff clean, **395 passed**
  (lag and determinism tests green). Left open: the §5 numbers cross-check at close-out
  (`PHASE4_INSTRUCTIONS.md` §7 says statuses stay as they are until then).
  Closed 2026-10-09: TODOs 25, 28, 29 (`a280750`) are the same construction as the reference
  (`weights.shift(lag)` with `lag < 1` rejected; gross P&L masked before the first position by
  a running "started" flag; leverage and `N_live` from the positions; unpriced position-days
  counted); life 6,410 days, mean gross leverage 1.097, mean `N_live` 12.80 and 20 unpriced
  position-days reproduced; the lag and determinism tests green.
- [x] **F4.2 Costs** — `DONE`
  bps model, net returns, gross and net both in `BacktestResult`. Accept: zero-turnover and
  single-flip tests. Notes: scaffolded 2026-10-02 (TODOs 26 `turnover`, 27 `transaction_costs`,
  30 `backtest_summary`; `BASIS_POINT` constant added to `src/config.py`; a NaN position is
  zero exposure, so entry, exit and flip are traded, Decision 49). The unit flip is tested to
  cost exactly `2 * bps / 1e4` at 5, 10 and 20 bps. Also carries the first slice of
  `src/metrics.py` (TODOs 31–36: annualised return and vol, Sharpe, Sortino, max drawdown,
  `performance`; Decision 50) so the §8 metrics known-answer test exists at this phase.
  Progress 2026-10-05: TODOs 26, 27, 31 implemented; `TestTurnover`, `TestTransactionCosts`,
  `TestAnnualisedReturn`, `TestIntegrityZeroTurnover`, `TestIntegrityCostAccounting` passing.
  Left open: TODO 30 `backtest_summary` and TODOs 32–36 (metrics known-answer test).
  Progress 2026-10-08: TODOs 30 and 32–36 implemented; `make lint test` **395 passed**,
  zero-turnover, single-flip and metrics known-answer tests green. Left open: close-out
  cross-check only.
  Closed 2026-10-09: TODOs 26, 27 and 30–36 are the same construction as the reference (a NaN
  position is zero exposure before differencing; cost = turnover × bps × 1e-4; every metric
  drops NaN rows first; drawdown counts the starting wealth of 1 as a peak); net Sharpe 0.34,
  max drawdown −40.4% with the same peak, trough and recovery dates, turnover 5.59x/yr and
  total cost 7.10% reproduced; zero-turnover, single-flip, cost-accounting and metrics
  known-answer tests green.
- [x] **F4.3 Benchmarks** — `DONE`
  EW long-only (monthly, same costs), EW long-only vol-targeted, external index aligned. Accept:
  EW long-only reproduces a hand-computed 2-asset toy case. Notes: scaffolded 2026-10-02
  (TODOs 37 `ew_weights`, 38 `ew_vol_targeted_weights`, 39 `external_returns` in
  `src/benchmark.py`; contracts in Decision 51; the two-asset toy case is
  `tests/test_benchmark.py::TestEwLongOnlyToyCase`). On this panel the equal-weight basket's
  Sharpe comes out above the strategy's, a consequence of the unadjusted front-month series
  (§3.6) explained in the instructions doc; a finding, not a bug (Decision 52).
  Progress 2026-10-08: TODOs 37–39 implemented; `tests/test_benchmark.py` green including
  `TestEwLongOnlyToyCase`; `make lint test` **395 passed**. Left open: close-out cross-check.
  Closed 2026-10-09: TODOs 37–39 are the same construction as the reference (eligibility by a
  same-day return, held on the schedule; §2.2 steps 2–3 only; off-calendar prints dropped
  before `clean_returns`); every benchmark number in the headline and per-year tables
  identical; the toy case green.
- [x] **F4.4 End-to-end run on snapshot** — `DONE`
  `make backtest` writes `results/returns_daily.csv`, `positions.csv`, `resolved_config.yaml`.
  Accept: runs under 1 minute; determinism test passes. Notes: scaffolded 2026-10-02 (TODOs
  40 `run_backtest`, 41 `run_benchmarks` in `src/cli.py`, the I/O module; Decision 52).
  `make backtest` runs in about a second on the snapshot and writes `returns_daily.csv` (six
  columns over the strategy's life), `positions.csv`, `backtest.md`, `resolved_config.yaml`;
  `make benchmarks` writes `benchmarks_daily.csv` (five columns) and `benchmarks.md`. The
  determinism test runs the stage twice into scratch directories and compares the files byte
  for byte. The `results/` files arrive with the implementation's pull request.
  Progress 2026-10-08: TODOs 40–41 implemented; `tests/test_cli.py` green including
  `TestIntegrityDeterminism`; `make lint test` **395 passed**. Left open: `time make
  backtest` (< 1 min), `make benchmarks`, the §5 script, committing `results/`, the PR.
  Closed 2026-10-09: TODOs 40–41; `make backtest` 0.9 s; the `results/` files regenerated on
  the merged tree, byte-identical to the branch's own output, committed as `add3ab8`; every
  headline, per-year (26 rows) and per-instrument (13 rows) number identical to the expected
  tables, compared mechanically, zero mismatches; `395 passed`, 52 lookahead tests and every
  §8 integrity class green; `tests/`, `data/`, `config/` untouched; `N_trials` = 0. Merged by
  squash without a pull request (Decision 53); tagged `v0.2`.

### Phase 5 — Metrics and inference

- [x] **F5.1 Core metrics** — `DONE`
  §5.1 as pure functions plus `summarise(returns) -> dict`. Accept: known-answer tests. Notes:
  scaffolded 2026-10-09 (TODOs 42 `monthly_returns`, 43 `max_drawdown_duration`, 44 `calmar`,
  45 `hit_rate`, 46 `skew`, 47 `kurtosis`, 48 `summarise` in `src/metrics.py`; contracts in
  Decision 54). 30 new tests in `tests/test_metrics.py`, every expected value hand-computed
  (the `[0, 0, 0, 0.01]` series has skew exactly 2 and excess kurtosis exactly 4). The book
  statistics of §5.1 come from the backtest and sizing results and are assembled in the stage.
  Closed 2026-10-15: TODOs 42–48 the same construction as the reference (months by
  `to_period("M")`, spells by the cumulative count of rows at the peak, pandas' `G1`/`G2`);
  every metric row of `results/metrics_table.md` identical to the expected table.
- [x] **F5.2 Inference** — `DONE`
  t-stat, block bootstrap CI (seeded), PSR, DSR with `N_trials` input. Accept: bootstrap CI
  contains point estimate; PSR of a zero-mean series ~0.5; DSR decreases with trials. Notes:
  scaffolded 2026-10-09 in a new module, `src/inference.py` (TODOs 49 `sharpe_t_stat`, 50
  `stationary_bootstrap_indices`, 51 `bootstrap_sharpe_ci`, 52 `probabilistic_sharpe`, 53
  `expected_max_sharpe`, 54 `deflated_sharpe`, 55 `inference_summary`; Decision 55). 42 tests
  in `tests/test_inference.py`, the three acceptance criteria among them; the bootstrap's draw
  order is pinned by an independent loop re-statement of the documented recipe.
  Closed 2026-10-15: TODOs 49–55 the same construction as the reference (one seeded
  generator in the documented draw order, zero-std resamples dropped, `NormalDist`, the hurdle
  passed annualised); every interval, PSR and DSR identical to four decimals.
- [x] **F5.3 Metrics table** — `DONE`
  `results/metrics.json` and `metrics_table.md` for strategy gross, net, and three benchmarks.
  Accept: table renders; README marker injection tested. Notes: scaffolded 2026-10-09 (TODOs
  56 `format_metrics_md`, 57 `metrics_json`, 58 `inject_between_markers` in a new
  `src/report.py`; 59 `run_metrics` in `src/cli.py`, the new `metrics` stage and `make metrics`
  target; new config key `inference.ci_level`; Decision 56). 24 tests in
  `tests/test_report.py`, 21 new in `tests/test_cli.py` (the §8 determinism test extended to
  `metrics.json`). Shipped `107 failed / 407 passed`, complete at `514 passed`; expected
  numbers in `PHASE5_INSTRUCTIONS.md` §5. The `results/` files arrive with the
  implementation's pull request.
  Run on the snapshot (`make metrics` 2.5 s; 2,000 resamples, block 20, seed 0, one trial):
  strategy net Sharpe 0.342, t-stat **1.73**, 95% CI **[−0.034, 0.719]**, PSR **0.958** (= DSR
  at one trial); gross 0.368, t 1.86, CI [−0.009, 0.744], PSR 0.968. EW long-only net 0.713,
  CI [0.311, 1.131]; EW vol-targeted net 0.675, CI [0.253, 1.111]; `^BCOM` 0.135, CI
  [−0.272, 0.554], PSR 0.751. Net: 306 months, 52.3% up; 51.5% of days up; longest spell under
  water 2,442 days (2015-12-18 to 2025-09-04; gross 1,562 to 2022-03-07); `^BCOM` under water
  since 2008-07-03 (4,534 days, open). DSR hurdle per unit of trial-Sharpe std at N = 192:
  2.752. The strategy's 95% interval straddles zero while its PSR is 0.958; the long-only
  basket's excludes it.
  Closed 2026-10-15: TODOs 56–59; `make metrics` 2.5 s; `results/metrics.json`,
  `metrics_table.md` and `resolved_config.yaml` (now with `inference.ci_level`) regenerated on
  the merged tree, byte-identical to the branch's own output, committed as `5c70ec2`; the table
  identical to the expected one line for line (`diff`, zero lines), the §5 script's inference,
  under-water, month-count and hurdle lines all equal; `514 passed`, every §8 integrity class
  green; `tests/`, `data/`, `config/` untouched; `N_trials` = 0. Merged by squash without a
  pull request (Decision 57), `ba1d940`. No tag (the next milestone is `v0.3` at Phase 6).

### Phase 6 — Robustness

- [ ] **F6.1 Grid runner** — `SCAFFOLDED`
  Iterate §5.3 grid, cache to `results/grid.parquet`, record `N_trials`. Accept: resumable;
  under ~5 minutes on snapshot. Notes: scaffolded 2026-10-15 in a new module,
  `src/robustness.py` (TODOs 60 `grid_specs`, 61 `trial_config`, 62 `evaluate_trial`, 63
  `trial_fingerprint`, 64 `trial_stats`), and TODO 76 `load_or_run_grid` in `src/cli.py`, the
  resumable cache; contracts in Decision 58. The `metrics` stage now reads the grid and
  deflates the strategy columns by its trial count (Decision 59). 67 tests in
  `tests/test_robustness.py`, among them the §5.3 guarantee that the in-sample numbers cannot
  see the out-of-sample data; the resume, interruption and fingerprint behaviour in
  `tests/test_cli.py::TestLoadOrRunGrid`; the full 192-trial grid runs in the suite.
- [ ] **F6.2 IS/OOS analysis** — `SCAFFOLDED`
  Select `L` on IS, evaluate OOS, report degradation vs a-priori default. Accept: output table in
  `results/is_oos.md`. Notes: scaffolded 2026-10-15 (TODO 65 `is_oos`; the report through the
  provided `is_oos_md` and TODO 75 `format_frame_md` in `src/report.py`; Decision 60).
- [ ] **F6.3 Sub-period analysis** — `SCAFFOLDED`
  Calendar blocks, per-year, rolling 3y Sharpe, regime view with hindsight caveat. Accept:
  `results/subperiods.md`. Notes: scaffolded 2026-10-15 in a new module, `src/subperiods.py`
  (TODOs 69 `block_stats`, 70 `calendar_year_returns`, 71 `rolling_sharpe`, 72
  `rolling_summary`, 73 `regime_labels`, 74 `regime_stats`); new config key
  `robustness.regime_vol_quantile` (Decision 61). 27 tests in `tests/test_subperiods.py`,
  including a no-lookahead test on the rolling Sharpe and a test that shows the regime labels'
  hindsight.
- [ ] **F6.4 Cost and lag sweeps** — `SCAFFOLDED`
  Net Sharpe vs cost bps and vs lag. Accept: `results/sweeps.md`. Notes: scaffolded 2026-10-15
  (TODOs 66 `cost_sweep`, 67 `lag_sweep`, 68 `grid_distribution`; 77 `run_robustness` in
  `src/cli.py`, the `robustness` stage writing the three reports; Decision 60; the §8
  determinism test extended to the reports). Shipped `129 failed / 526 passed`, complete at
  `655 passed` (141 new tests); expected numbers in `PHASE6_INSTRUCTIONS.md` §5. The
  `results/` files arrive with the implementation's pull request.

### Phase 7 — Figures and report

- [ ] **F7.1 equity_curve.png** — `TODO`  Log scale; gross, net, 3 benchmarks; legend with net
  Sharpe. Notes: —
- [ ] **F7.2 drawdown.png** — `TODO`  Strategy net vs EW long-only. Notes: —
- [ ] **F7.3 param_sensitivity.png** — `TODO`  Heatmap + cost lines from grid. Notes: —
- [ ] **F7.4 returns_by_subperiod.png** — `TODO`  Blocks + per-year bars, strategy vs benchmark.
  Notes: —
- [ ] **F7.5 Extra figures** — `TODO`  rolling_sharpe, is_oos, exposure. Notes: —
- [ ] **F7.6 Report generation** — `TODO`  `make report` regenerates all `results/*.md` and
  injects the metrics table into README between `<!-- METRICS:START/END -->`. Notes: —

### Phase 8 — Write-up

- [ ] **F8.1 README** — `TODO`  Rationale, method, results, robustness, inference, limitations,
  reproduce. Reviewed against §10 checklist. Notes: —
- [ ] **F8.2 Notebook** — `TODO`  Narrative importing `src/`, loads `results/`; executed in CI.
  Notes: —
- [ ] **F8.3 Final rigor review** — `TODO`  Walk §10; every item cites the file/test that proves
  it. Tag `v1.0`. Notes: —

### Phase 9 — Optional extensions (each needs the core `DONE` first)

- [ ] **F9.1 Multi-lookback ensemble** — `TODO`  Average of 3/6/12m signals. Entry: Phase 8 done.
- [ ] **F9.2 Long-only momentum / trend filter** — `TODO`  Long when signal +1, flat otherwise.
- [ ] **F9.3 Carry signal** — `TODO`  Entry criterion: a free or already-licensed source of a
  second maturity per commodity is identified and documented. Candidates: CME settlement pages,
  Nasdaq Data Link (CHRIS discontinued; check alternatives), Stooq. Signal = annualised roll yield
  sign. Otherwise stays `BLOCKED`.
- [ ] **F9.4 Per-instrument cost table** — `TODO`  Sector-specific bps; re-run net results.

## 10. Rigor checklist (README must cite evidence for each)

- [ ] No lookahead: perturbation tests pass for signal, vol, weights, positions.
- [ ] Costs applied; every result shown gross and net.
- [ ] Explicit two-stage vol targeting with a leverage cap; cap-binding fraction reported.
- [ ] Walk-forward: a-priori parameters; IS/OOS selection exercise reported with degradation.
- [ ] Parameter sensitivity grid shown; `N_trials` disclosed; DSR reported.
- [ ] Benchmarked against EW long-only, EW vol-targeted, and an external index.
- [ ] Overfitting discussed honestly, including what the IS/OOS exercise cost.
- [ ] Data limitations stated: roll gaps, negative-price handling, gap fill, universe choice.

## 11. Decision Log (agreed at kickoff, 2026-09-03)

| # | Decision | Alternatives considered | Why |
|---|----------|-------------------------|-----|
| 1 | Audience: recruiters and code-reading interviewers equally | Either alone | README narrative for skim, tested modules for deep dive |
| 2 | yfinance default, pluggable `PriceSource` | Better paid source; heuristic roll detection | Free and reproducible now; upgrade path without engine changes |
| 3 | Max history (~2000+) | 2010+ | Includes 2008, 2014, 2020, 2022 regimes |
| 4 | Ragged panel, assets enter when warm | Common start; fixed long-history 8 | No history thrown away; no survivorship by dropping |
| 5 | Plain 12m TSMOM, no skip | 12-1; continuous z-score | Canonical, fewest parameters; skip is a variant |
| 6 | Monthly rebalance default | Daily | Realistic turnover; daily is a variant |
| 7 | Per-asset + portfolio vol target, 3x gross cap | Gross=1 normalise; per-asset only | Standard TSMOM construction; controls portfolio risk |
| 8 | "Walk-forward" = a-priori params + IS/OOS selection exercise | Expanding-window re-selection; none | Cheap, directly answers overfitting question |
| 9 | Flat 5 bps per unit turnover, swept | Per-instrument table; explicit roll cost | Transparent; extension F9.4 |
| 10 | Returns as dP/\|P\| clipped at 50%, flagged | Zero-out WTI Apr-2020; generic clip only | Keeps WTI; one rule handles negatives, rolls, bad prints |
| 11 | Benchmarks: EW long-only same costs, EW vol-targeted, DBC/BCOM | Never-rebalanced EW | Apples-to-apples plus recognised external reference |
| 12 | Union calendar, ffill limit 5, zero return on fill | Intersection; unlimited ffill | Standard; avoids stale prices |
| 13 | Committed snapshot + manifest with hashes | Live pull only | Results reproducible offline |
| 14 | Makefile stages calling `src/cli.py` | Notebook-driven | Testable, cacheable, diffable |
| 15 | Unit + integrity tests; TDD | Metrics-only tests; smoke test | Integrity is the deliverable |
| 16 | GitHub Actions: ruff + pytest (+ notebook later) | None | Green badge; enforces discipline |
| 17 | 14-name candidate universe, gated | 8 names | Sector breadth; gate keeps it honest |
| 18 | Rolling 60d vol default, EWMA implemented | EWMA default | Simplest to explain; robustness shows choice barely matters |
| 19 | Bootstrap CI, t-stat, PSR, DSR | Point estimates | "Not fooling yourself" centrepiece |
| 20 | Sub-periods: fixed calendar blocks, per-year, rolling 3y, regime-with-caveat | Regime only | A-priori cuts avoid hindsight; regime view flagged |
| 21 | uv + pyproject + ruff, Python 3.12 | pip/venv; conda | Lockfile reproducibility; `requirements.txt` exported |
| 22 | YAML config + typed dataclasses + resolved dump | CLI flags only | Traceable runs |
| 23 | Feature branch per feature, squash-merge via PR | Direct on main | PR description doubles as log entry |
| 24 | Thin narrative notebook, CI-executed | No notebook; notebook-first | Deliverable without logic drift |
| 25 | Execution lag 1 default, lag 2 variant | Lag 2 default | Standard; sensitivity shown |
| 26 | Local git only this session; GitHub repo in first coding session | Create now | Nothing public until there is code |
| 27 | Extensions specced as gated Phase 9 | Future-work note; carry as core | Clear entry criteria; core not held hostage |
| 28 | Phases opened and closed by a separate session from the one that built them | Builder closes its own phase | Every phase's tests and numbers are verified independently before the next phase starts. *Superseded by 29 on 2026-09-04.* |
| 29 | Phase-assignment model (2026-09-04): each phase ships as complete validated tests + stubs with numbered TODOs + `PHASE<N>_INSTRUCTIONS.md`; it is implemented on a branch, the numbers are reported back and cross-checked against a reference implementation, then one PR per phase is merged and the phase marked done | Per-feature PRs opened and closed by separate sessions (Decisions 23, 28) | Proven on the copper-hedge project; every line of strategy code is written against tests the implementer cannot edit, and numbers are cross-checked before anything lands |
| 30 | `PROJECT_LOG.md` (spec + state) and `WORKLOG.md` (session history) are the project's written record; reference solutions and local working notes are kept out of the repository (`.gitignore`) | Commit the reference solutions alongside | The repo shows the implementation; the reference solutions are the checking tool, not the deliverable |
| 31 | Phase 0 shipped directly as the initial code commit, plus `SETUP.md` (2026-09-04), rather than as an assignment | Ship F0.1–F0.2 directly and assign F0.3 as a first small phase | An assignment needs a running test suite to build on; the config loader is plumbing with little to learn from; this gets Phase 1 started a session sooner |
| 32 | The one live `make data` pull was run once (2026-09-04, yfinance 1.7.0, maximum history) and the snapshot committed with the Phase 1 assignment, so the implementation is offline end to end and its numbers can be checked exactly | Pull the data as part of the implementation | Exact expected numbers beat plausibility zones; one fewer moving part in the first assignment |
| 33 | Gap filling is all-or-nothing per run: a run of missing days no longer than `ffill_limit` is filled entirely, a longer run stays NaN in full; leading NaNs never filled; a trailing run is treated like any other | pandas `ffill(limit=n)` (fills the first n days of any run) | Matches §3.2 literally ("longer gaps stay NaN") and makes the §8 gap-fill test unambiguous |
| 34 | Quality-gate definitions: fill fraction = filled days / calendar days from first to last real observation inclusive; longest gap = longest run of days without a real price after the first one, counting a trailing run; years = (calendar end − first real date) / 365.25; reasons formatted `history {y:.1f}y < {min:g}y`, `filled {f:.1%} > {max:.0%}`, `gap {n}d > {max}d`, joined by `; ` in that order | Span to calendar end for fills; years to last valid date | Defined once so the report is reproducible and the tests can compare strings verbatim |
| 35 | `data/futures_panel.csv` holds only the gate passers, on the union of *their* trading days (all-NaN rows dropped); all 14 candidates stay in `data/raw/` and the quality report | Panel of all candidates on the full union calendar | Downstream stages read one file and get the universe; the excluded names remain inspectable |
| 36 | Config additions `data.csv_path` (null), `data.fetch_retries` (3), `data.benchmark_path` (`data/benchmark.csv`); the benchmark is a one-column wide CSV; none of the three enters the sensitivity grid (I/O settings, not strategy parameters) | Hard-code in `src/` | §13: no magic numbers in `src/`; §3.5's pluggable loader needs a path |
| 37 | `clean_returns` edge rules: a raw value exactly at ±`clip` is neither clipped nor logged; `P(t−1) = 0` gives an infinite raw value, clipped to ±`clip` and logged with the raw infinity; NaN on either side gives NaN | Treat the bound as clipped; drop zero-denominator days | Keeps the log to genuine outliers and every non-NaN return finite (§8 negative-price test) |
| 38 | (2026-09-14) The committed panel is rebuilt from the raw files in `data/raw/` (re-pull recorded in the manifest as 2026-09-08T15:07:58Z, yfinance 1.7.0, last row 2026-09-07): 6,724 rows to 2026-09-07, with the manifest's panel entry rebuilt offline using the recorded pull metadata. Supersedes the 2026-09-04 pull date in Decision 32 | Keep the earlier 6,723-row panel to 2026-09-04 and its manifest, one day shorter than the raw files it is supposed to derive from | The reproducibility claim (`TestSnapshot`) requires the panel to be derived from the raw files beside it. The differences are one extra row (Labor Day 2026-09-07: real prices for the six energy and metals contracts, forward-filled for the seven agriculturals) and a corrected 2026-09-04 row; no gate verdict, the `PL=F` exclusion, or the two clipped days change |
| 40 | (2026-09-15) Trailing return `r(t) = P(t−skip) / P(t−skip−L) − 1` is row-positional on the panel calendar and NaN only when an **endpoint** is NaN; a NaN strictly inside the window is tolerated because the clean price index carries its level across a gap. A gap of `g` price-index rows therefore costs `2g` signal days (the gap and its echo `L` rows later), and `signal_summary` reports the count as `n_missing` | Require all `L+1` observations in the window valid (drops LE=F for a year after its 20-day gap in January 2006); use the last valid price at or before `t−L` (hides the gap) | Literal §2.1 formula, one `shift` per endpoint, and what binds is counted rather than hidden |
| 41 | (2026-09-15) The signal is computed on `clean_price_index(clean_returns(panel))`, so an instrument's first signal lands `lookback + 1` rows after its first price (the index's first row is NaN). Phase 2 adds no config key, no CLI stage and no `results/` file: the signal is an intermediate that Phase 4's backtest stage writes; the phase's numbers come from `signal_summary` (TODO 16) through the instructions doc's script | Signal on raw panel prices; a `make signals` stage writing `results/signal_summary.csv` | §3.3 fixes the series; a stage for an intermediate is I/O for its own sake, and the summary function is testable without it |
| 42 | (2026-09-15) Harness design: `assert_no_lookahead(fn, prices, t, *, n_trials=3, seed=0)` attacks twice, log-normal noise (σ = 0.5) with 10% holes on every row after `t` (three seeded trials) and then outright truncation after `t`, and requires the output on and before `t` to match value for value and NaN for NaN. It lives in `tests/integrity.py` (not collected by pytest) with its own tests in `tests/test_integrity.py`; the lookahead tests are listed with `uv run pytest -k lookahead -vv` | Noise only; small noise (a sign signal would not flip); a pytest fixture | Two independent attacks catch both "reads the future" and "depends on how much future exists"; exact equality is the §8 wording |
| 43 | (2026-09-21) Vol estimator contracts: `rolling` = sample std (`ddof = 1`) over the full trailing window (`min_periods = window`), so a gap of `g` NaN return rows costs `g + window − 1` vol rows; `ewma` = pandas `ewm(span=window, min_periods=window)` with its defaults (`adjust=True`, `ignore_na=False`, `bias=False`), masked to NaN on rows whose own return is NaN; both times `sqrt(252)` | Partial windows (`min_periods < window`); `ignore_na=True`; `ddof = 0` | Literal §2.2 ("std over `vol_window` days"), no estimate from a partial window, and the two estimators differ as a robustness variant rather than in their NaN rules |
| 44 | (2026-09-21) "Live" = a non-NaN signal and a finite vol `> 0` on the same cell; `N_live` counts live instruments (a flat signal is live, with weight 0); the raw weight is NaN where not live. §2.2's "valid signal" is read as "valid signal that can be sized" | Count every valid signal (divides the sized names by a count that includes a name with no position) | `N_live` is the divisor of what is actually held; the signal-without-vol days are counted instead, in `n_unsizable` |
| 45 | (2026-09-21) The pre-scaled book's return is `sum_i w_raw_i(t−1) r_i(t)` (a fixed one-row lag, not `execution.lag`; NaN terms skipped, NaN when none is defined); its vol uses the same window and estimator as step 1; a book vol that is NaN or not `> 0` leaves the row's weights NaN. The portfolio vol therefore has its own `vol_window` warm-up after the first live instrument | Contemporaneous `w_raw(t) r(t)`; the held weights; `execution.lag` | The vol of a book somebody could have held; independent of the schedule and of the lag, so the grid's rebalance and lag variants compare the same sizing; no finite scale exists for a zero-vol book |
| 46 | (2026-09-21) Monthly rebalance dates = the last row of each calendar month present in the index, a partial final month included (a weight decided on the sample's last row never becomes a position); `hold` copies whole rows by label (a NaN decided on a rebalance day stays NaN until the next one; NaN before the first date) and the cap flag is held the same way. The no-lookahead test on held monthly weights cuts at month-ends only: the schedule is calendar knowledge, and a mid-month truncation creates a rebalance day the full sample lacks; every daily-frequency stage is cut at arbitrary dates | Month-ends from the business-day calendar (loses a month whose last business day is a holiday); skip the partial month; cell-wise `ffill` | Literal §2.3 on the panel calendar; the truncation attack cannot tell calendar knowledge from price knowledge, and the daily stages carry the full attack |
| 47 | (2026-09-21) Outputs: `size_positions(signal, returns, sizing, rebalance) -> SizingResult` carrying every intermediate (vol, raw weights, `n_live`, book vol, daily decision, held weights, cap flag, schedule); weights NaN where no position can be sized, 0 where flat; `sizing_summary` gives the headline numbers with `n_unsizable` (instrument-days inside the traded span without a raw weight) as the what-binds count. No config key, CLI stage or `results/` file (as Decision 41: the weights are an intermediate Phase 4 writes). | A `make weights` stage; a bare weights frame | Phase 4's report needs the intermediates; a stage for an intermediate is I/O for its own sake |
| 48 | (2026-10-01) Phase 3 merged by a squash of `phase-3` (`fa431cb`) without a pull request, since none was opened and the implementation was verified exact against the reference; the per-machine instructions file (`WORKING_AGREEMENT.md` §6) the branch tracked is left out of `main` (Decision 39 stands: it is per-machine and gitignored, see `WORKING_AGREEMENT.md` §6) and the remote `phase-3` branch is kept so that copy stays recoverable; CI runs on `main` at the push | Wait for the pull request; a merge commit as in Phase 1 | The numbers are exact and waiting adds nothing; a squash keeps `main` to conventional commits; deleting the branch would be the one way to lose the file |
| 49 | (2026-10-02) Engine contracts: `positions = weights.shift(lag)`, row-positional, `lag >= 1` enforced (`ValueError`; a zero lag is the lookahead the §8 lag test exists to catch); a NaN position is zero exposure for turnover, so entering, exiting and flipping are all traded and a constant book pays exactly its entry; `cost = turnover * cost_bps * BASIS_POINT` with the constant in `src/config.py`; the gross return is NaN before the strategy's first position and defined (zero when flat) from then on, so metrics run over the strategy's life; a held position whose instrument has no return that day earns nothing and is counted (`BacktestResult.unpriced`) | Entry not charged (NaN terms skipped in the turnover); zero returns from the panel's first day; NaN returns on flat days | The literal §2.4 formula with the only defensible reading of a missing previous position; a return series that measures the strategy only while it exists; what binds counted rather than hidden |
| 50 | (2026-10-02) Phase 4 ships the first slice of `src/metrics.py` (geometric annualised return, annualised vol with `ddof = 1`, Sharpe and Sortino on excess returns over `inference.risk_free_rate / 252` per row, Sortino's downside deviation over all rows, max drawdown from the running peak with the starting wealth of 1 counted as a peak and reported as a negative fraction, and `performance` bundling them); every function drops NaN rows first and returns `float("nan")` where undefined. Phase 5 extends the module with the rest of §5.1 and §5.2 | Leave all metrics to Phase 5 and report raw sums in Phase 4 | §8 and §13 require all eleven integrity tests, the metrics known-answer one included, before Phase 4 closes, and the phase's report needs Sharpe and drawdown gross and net |
| 51 | (2026-10-02) Benchmark contracts: an instrument is eligible for the equal-weight basket on a day when it has a return that day (no lookback), weight `1 / N_eligible`, decided on `execution.rebalance` and held; the vol-targeted basket applies §2.2 steps 2 and 3 only (portfolio vol of the daily equal weights, scale, cap, hold), not the per-asset step; the external index's returns are computed between consecutive prints that fall on the panel calendar (off-calendar prints dropped first, a missing print costs one NaN row and no return) with `clean_returns` and the §3.3 clip; the three series are aligned to the strategy's life by truncation in the stage, the baskets themselves being computed on the full panel. Under §2.3's constant-weight simplification the equal-weight basket trades only when its membership changes, so its net is almost exactly its gross | Eligibility by signal availability; per-asset vol scaling in the vol-targeted basket; reindexing the index's prices onto the calendar before differencing; restarting the baskets at the strategy's first day | Literal §4: the same universe, the same costs, the same portfolio scaling; nothing lost across holiday mismatches; a benchmark that exists on its own and is compared over the strategy's window |
| 52 | (2026-10-02) Stage design: `run_backtest(cfg, root) -> BacktestRun` and `run_benchmarks(cfg, root) -> BenchmarkRun` live in `src/cli.py` (one of §13's four I/O modules) so the engine, metrics and benchmarks stay pure; result-file names are module constants (`returns_daily.csv` with `gross, net, cost, turnover, gross_leverage, n_live` over the life, `positions.csv`, `backtest.md`, `benchmarks_daily.csv` with the two baskets gross and net plus `external`, `benchmarks.md`, `resolved_config.yaml`); no new config key; the §8 determinism test compares the CSVs byte for byte (Phase 5 adds `metrics.json`). On this panel the equal-weight basket's Sharpe is above the strategy's (see F4.3 and `PHASE4_INSTRUCTIONS.md` §5) | A `src/pipeline.py`; writing from `backtest.py`; config keys for file names | §7 and §13 as written; I/O settings are not strategy parameters (Decision 36) |
| 53 | (2026-10-09) Phase 4 merged by a squash of `phase-4` (`a280750`) without a pull request or the §7 report-back, since the implementation reproduces every §5 number exactly; the `results/` files were regenerated from the snapshot on the merged tree (byte-identical to the branch's own output) and committed as `add3ab8`; the per-machine instructions file (`WORKING_AGREEMENT.md` §6) the branch tracked is left out of `main` (Decisions 39 and 48) and the remote `phase-4` branch is kept so that copy stays recoverable | Wait for the pull request and the report-back | The numbers are exact and waiting adds nothing; the stage is deterministic from the snapshot, so the files do not depend on who runs it; deleting the branch would be the one way to lose the file |
| 54 | (2026-10-09) Metric contracts (F5.1): `monthly_returns` compounds the non-NaN rows within each calendar month (`index.to_period("M")`), labelled with the month's last row; partial first and last months kept, a month without rows absent (never a 0% month); `max_drawdown_duration` is the longest run of consecutive non-NaN rows strictly below the running peak (the starting wealth of 1 a peak, as Decision 50), counted in rows, an open spell counted to the last row, `0` when none; `calmar = ann_return / -max_drawdown`, NaN unless the drawdown is negative; `hit_rate` = share of rows strictly above zero (applied to monthly returns for the monthly rate); `skew` and `kurtosis` are pandas' bias-adjusted `G1` and excess `G2`, NaN below 3 / 4 rows or at zero variance; `summarise` = `performance` plus those eight keys (`METRIC_KEYS`). The book statistics of §5.1 (turnover, gross leverage, cap-binding share, `N_live`) are taken from the backtest and sizing results in the stage, not recomputed from returns. §5.2 lives in a new `src/inference.py`, since `src/metrics.py` with both would pass §13's ~300-line split point | Monthly returns by `resample("ME")`; duration from peak to recovery date in calendar days; population moments; one `metrics.py` | No fake zero months across the data's gaps; a duration in the same unit as every other count here; pandas' estimators are the ones a reviewer gets from `Series.skew()`; §13 asks for the split |
| 55 | (2026-10-09) Inference contracts (F5.2): Sharpe ratios passed in or out are annualised, the PSR working on `SR / sqrt(252)` internally; `sharpe_t_stat = SR * sqrt(T / 252)`; the stationary bootstrap's draw order is fixed (one `default_rng(seed)`; per resample `integers(0, n, size=n)` then `random(n)`; a new block where the draw is below `1 / block_length`, else the previous index plus one modulo `n`), and the interval is the `np.quantile` (linear) of the resample Sharpes at `(1 ∓ ci_level) / 2`, zero-std resamples dropped; PSR per Bailey and López de Prado with the bias-adjusted moments, `(kurtosis + 2) / 4` for `(gamma_4 - 1) / 4`, NaN below 4 rows or for a non-positive variance term; `expected_max_sharpe(N, std)` with the Euler-Mascheroni constant, `0` at `N = 1`; DSR = PSR at that hurdle, `std` being the spread of the trials' annualised Sharpes; normal CDF and inverse from `statistics.NormalDist` (no scipy) | Unfixed draw order (reproducible per implementation only); i.i.d. bootstrap; a basic or BCa interval; population moments; scipy | Exact reproducibility across implementations is the cross-check; blocks keep volatility clustering (§5.2); the percentile interval is the one the spec's acceptance criterion describes; no new dependency |
| 56 | (2026-10-09) Stage and table design (F5.3): a new `metrics` stage and `make metrics` target (in `make all` between `benchmarks` and `robustness`) runs `run_metrics(cfg, root, n_trials=1, trials_sharpe_std=0.0)` in `src/cli.py` and writes `results/metrics.json` (strict JSON, NaN as `null`) and `results/metrics_table.md`; five columns: strategy gross and net, the two baskets **net** and the external index; one entry per series of `summarise`, the four book statistics and `inference_summary`. `src/report.py` holds the pure formatters and `inject_between_markers`, tested now; `make report` wires the README injection in Phase 7 (F7.6). Until Phase 6's grid exists the DSR runs at one trial (hurdle 0, DSR = PSR); Phase 6 passes the grid's count and the spread of its Sharpes. New config key `inference.ci_level: 0.95`, validated in (0, 1), excluded from the grid as a reporting choice. The §8 determinism test now also compares `metrics.json` byte for byte | Write the files from `make report` now; inject into README in Phase 5; `inference.n_trials` in the config; a module constant for the level | Each stage one job; the README carries results only once `make report` regenerates everything (§13); §5.2 says the grid runner logs `N_trials`, so it is not a parameter; no magic number in `src/` |
| 57 | (2026-10-15) Phase 5 merged by a squash of `phase-5` (`ba1d940`) without a pull request or the §7 report-back, as for Phase 4 (Decision 53), since the implementation reproduces every §5 number exactly; the `results/` files were regenerated from the snapshot on the merged tree (byte-identical to the branch's own output) and committed as `5c70ec2`; the remote `phase-5` branch is kept | Wait for the pull request and the report-back | The numbers are exact and waiting adds nothing; the stage is deterministic from the snapshot, so the files do not depend on who runs it |
| 58 | (2026-10-15) Grid contracts (F6.1): a trial is one combination of the six §5.3 parameters (`GRID_KEYS`: lookback, vol window, estimator, rebalance, skip, lag), listed by `itertools.product` in config order, lookback slowest; costs are not trials: every trial is measured at each `robustness.cost_bps_sweep` cost and its 0-bps rows are its gross result, so `N_trials` = 192; each trial is measured over its own life (first position to the end), in sample on the life's rows up to `is_end`, out of sample from `oos_start`; the grid frame is long (one row per trial and cost, `GRID_COLUMNS` with `GRID_DTYPES`); `trial_stats` gives the row count at the default cost and the `ddof = 1` std of those rows' net Sharpe ratios; `results/grid.parquet` is the cache, keyed by the six parameters plus a SHA-256 fingerprint of the panel file's hash and the config values a row depends on (clip, vol targets, cap, cost sweep, IS/OOS dates, risk-free rate), rewritten after every computed trial so an interrupted run resumes, and holding exactly the current grid; a code change is not detected (delete the file). F6.3 lives in a new `src/subperiods.py` beside `src/robustness.py` (§13's split rule); the two I/O stages are in `src/cli.py` | Costs as trials (768); one common evaluation window for every trial; one row per trial with a column per cost; no fingerprint; a single `robustness.py` | §5.2 counts distinct configurations, and a cost is an assumption about the same positions, not a configuration; the default's rows reproducing Phase 4's headline exactly is the cross-check; tidy rows filter with one mask; a cache must not survive a change of inputs silently |
| 59 | (2026-10-15) DSR wiring: the `metrics` stage takes `N_trials` and the Sharpe spread from `load_or_run_grid` (building the grid first if the cache is missing, about 11 s), so `metrics.json` carries the grid's count whatever order the stages run in, and `make all` keeps metrics before robustness; `run_metrics` applies them to the two strategy columns only and measures the three benchmarks at one trial, a one-line change to the merged TODO 59 made with the assignment, its test updated and the change stated in `PHASE6_INSTRUCTIONS.md` | Reorder `make all` and fall back to one trial without a grid; report the DSR only in a robustness file; deflate all five columns | One answer from any stage order; the headline table shows the deflated number; nobody searched over the benchmarks |
| 60 | (2026-10-15) Selection and sweep contracts (F6.2, F6.4): the IS/OOS candidates are the grid rows at `costs.cost_bps` with the default's other five parameters; the selected `L` has the highest finite in-sample net Sharpe, a tie going to the shorter lookback, and the out-of-sample columns are never read (two integrity tests: every in-sample number bit-identical when the returns after `is_end` are replaced; the choice unmoved when the out-of-sample columns are scrambled); `oos_gap = selected_oos - default_oos`; the cost sweep is the default trial at each cost beside the grid's median and share positive, the lag sweep the default at each lag, gross (0 bps) and net, beside the median of the trials with that lag; `grid_distribution` ranks the default (one plus the trials strictly better) and names the best and worst trial; the three reports are laid out by provided composers in `src/report.py` over a new `format_frame_md` (a TODO) | Selecting over every grid parameter; re-running each trial on data truncated at `is_end` | §5.3 selects `L` only; truncation would move the warm-up, and the in-sample number would no longer be part of the trial's own life; report layout is presentation, the phase's substance is in the numbers |
| 61 | (2026-10-15) Sub-period contracts (F6.3): calendar blocks inclusive at both ends, `None` an open end; calendar-year returns compounded over each series' non-NaN rows, NaN (never 0) for a year without rows; the rolling Sharpe over each series' own last `rolling_sharpe_years * 252` observations, put back on the calendar afterwards (so the index's missing prints do not void every window spanning one), with a no-lookahead test; the regime view labels a day `stress` when the equal-weight long-only basket's trailing `sizing.vol_window`-day rolling vol (Phase 3's `realised_vol`, over the whole panel) is strictly above its full-sample quantile `robustness.regime_vol_quantile`, `calm` otherwise; the new key (0.75, validated in (0, 1)) stays out of the grid because it parameterises a hindsight description, not the strategy; the caveat is a constant printed above the table, and a test shows later data moving earlier labels | Named crisis windows; the sign of the basket's trailing 12-month return; windows of calendar rows | Mechanical and reproducible; the hindsight is explicit and tested rather than hidden; a regime by trailing trend would overlap the signal itself |

## 12. Session Log

- **2026-09-03** — Interview and spec. No code. Next: F0.1.
- **2026-09-04** — Added `.gitignore`. No code. Next: F0.1.
- **2026-09-04** — Phase 0 defined (F0.1-F0.3); Decision 28 added. No code. Next: F0.1.
- **2026-09-04** — Working model settled (Decisions 29–30); §13 ground rules added; work log
  opened. No code. Next: Phase 0.
- **2026-09-04** — Repository created on GitHub and `main` pushed. Phase 0 shipped
  (Decision 31): F0.1–F0.3 `DONE`, 34 tests, CI green on first run. Next: Phase 1.
- **2026-09-04** — Phase 1 assignment prepared (Decisions 32–37): 13 TODOs, 64 tests, starting
  state `60 failed / 38 passed`; data snapshot pulled and committed. Next: implement Phase 1.
- **2026-09-14** — Phase 1 implemented on `phase-1` (TODOs 7–13 this session, 1–6 in the
  sessions since 2026-09-11): `98 passed`, ruff clean; numbers cross-checked against
  `PHASE1_INSTRUCTIONS.md` §5; panel and manifest rebuilt from the raw files (Decision 38).
  F1.1–F1.4, F1.6 `DONE`; F1.5 pending commit and CI. Next: commit, push, PR, merge, tag `v0.1`.
- **2026-09-14** — Phase 1 implementation merged into `main`: `98 passed`, ruff clean after a
  one-line format fix; numbers reproduced by a reference implementation on the committed raw
  files. Working agreement renamed to `WORKING_AGREEMENT.md` (Decision 39). Next: CI green,
  final review, tag `v0.1`, then the Phase 2 assignment.
- **2026-09-14** — Phase 1 closed: F1.1–F1.6 `DONE`, CI green, `v0.1` tagged on `main`. Next:
  the Phase 2 assignment.
- **2026-09-15** — Phase 2 assignment prepared (Decisions 40–42): `PHASE2_INSTRUCTIONS.md`,
  `src/signals.py` with TODOs 14–16, 39 tests, the no-lookahead harness with 10 tests of its
  own; shipped at `39 failed, 108 passed`, complete at `147 passed`. Next: implement on
  `phase-2`, one pull request, report the §5 numbers.
- **2026-09-15** — Phase 2 assignment published to `main` (CI red on the unimplemented tests
  by design). Next: the `phase-2` branch and pull request.
- **2026-09-20** — Phase 2 implementation (`2eb0d1c`, on `main`) verified: ruff clean,
  `147 passed`, lookahead tests 18 green, §5 numbers identical to the expected tables,
  TODOs 14–16 match the reference. Statuses unchanged pending the close-out. Next: close
  Phase 2, then the Phase 3 assignment.
- **2026-09-20** — Phase 2 closed: F2.1–F2.2 `DONE`, no tag, nothing to merge; pushed to
  `main`. Next: the Phase 3 assignment.
- **2026-09-21** — Phase 3 assignment prepared (Decisions 43–47): `PHASE3_INSTRUCTIONS.md`,
  `src/portfolio.py` with TODOs 17–24, 101 tests; shipped at `99 failed, 149 passed`,
  complete at `248 passed`. Next: implement on `phase-3`, one pull request, report the §5
  numbers.
- **2026-09-21** — Phase 3 assignment published to `main` (CI red on the unimplemented tests
  by design). Next: the `phase-3` branch and pull request.
- **2026-09-28** — Phase 3 implementation started on `phase-3` (session ran into
  2026-09-29). Start: ruff clean, `99 failed, 149 passed`. TODOs 17 `realised_vol`,
  18 `asset_weights`, 19 `rebalance_dates`, 20 `hold` written; `TestRealisedVol`,
  `TestAssetWeights`, `TestRebalanceDates`, `TestHold`, `TestIntegrityRebalanceSchedule`
  passing. TODOs 21–24 open; full suite not re-run; nothing committed. Next: TODO 21.
- **2026-09-30** — TODOs 17–20 and the logs committed on `phase-3` (`35b31af`) and the branch pushed; not to `main`. TODO 21 `portfolio_vol` quizzed and
  written; `TestPortfolioVol` passing. TODO 22 `leverage_cap` quizzed and written;
  `TestLeverageCap` passing. TODO 23 `size_positions` quizzed and written;
  `TestSizePositions` and `TestIntegrity*` passing. TODO 24 `sizing_summary` quizzed and written, not yet
  tested. Next: `TestSizingSummary`, `TestSnapshot`, full `make lint test`.
- **2026-10-01** — Phase 3 implementation on `phase-3` (`4d38194`) verified: ruff clean,
  `248 passed`, 40 lookahead tests green, `tests/`, `data/` and `config/` unmodified, the
  §5 output identical to the expected tables (26 per-year and 13 per-instrument rows compared
  mechanically), TODOs 17–24 matching the reference in substance; `N_trials` = 0. No pull
  request and no §6 report-back on the branch.
- **2026-10-01** — Phase 3 closed: squash `fa431cb` (Decision 48), F3.1–F3.4 `DONE`, no tag.
  Next: push, then the Phase 4 assignment.
- **2026-10-02** — Phase 4 assignment prepared (Decisions 49–52): `PHASE4_INSTRUCTIONS.md`,
  TODOs 25–41 in `src/backtest.py`, `src/metrics.py`, `src/benchmark.py`, `src/cli.py`; 147
  tests; starting state `142 failed / 253 passed`, complete at `395 passed`; all eleven §8
  integrity tests exist (52 lookahead tests). One local commit. Next: push, implement
  Phase 4 on `phase-4`, one pull request, cross-check, merge, tag `v0.2`.
- **2026-10-02** — Pushed `5c87fd3..4479248` to `main`: the Phase 3 close-out and the Phase 4
  assignment. CI is red on the 142 unimplemented tests by design until the phase is
  implemented. Next: implement Phase 4 on `phase-4`, one pull request, cross-check, merge,
  tag `v0.2`.
- **2026-10-05** — Phase 4 implementation on `phase-4`: TODOs 25–29 and 31 written and
  quizzed; ruff clean, `79 failed / 316 passed` (from 142 / 253), every remaining failure a
  TODO not yet written (30, 32–41). No boxes ticked. Next: TODO 32 quiz, then 33–36, 30.
- **2026-10-08** — Phase 4 implementation on `phase-4`: TODOs 30 and 32–41 written (TODO 41
  ahead of its quiz, at the user's request); ruff clean, **395 passed / 0 failed**. No boxes
  ticked, statuses left `SCAFFOLDED` per `PHASE4_INSTRUCTIONS.md` §7 (flipped at the
  close-out cross-check). Next: §7 checklist — lookahead and Integrity listings, `make
  backtest` / `make benchmarks`, §5 script, commit `results/`, push, open the PR.
- **2026-10-09** — Phase 4 implementation on `phase-4` (`a82cb8a`) verified: ruff clean, `395
  passed`, 52 lookahead tests and every §8 integrity class green, `tests/`, `data/` and `config/`
  unmodified, the §5 output identical to the expected tables (26 per-year and 13 per-instrument rows
  compared mechanically), TODOs 25–41 matching the reference in substance; `N_trials` = 0. Closed:
  squash `a280750`, `results/` in `add3ab8` (Decision 53), F4.1–F4.4 `DONE`, tagged `v0.2`. Next:
  push, then the Phase 5 assignment.
- **2026-10-09** — Pushed `4479248..e632ba8` to `origin/main` and the `v0.2` tag after the
  pre-push checklist (no local-only file tracked; no private-name hit in the owner commits; the
  only such lines are in the collaborator's own logs inside `a280750`); `395 passed`. Next: the
  Phase 5 assignment.
- **2026-10-09** — Phase 5 scaffolded (Decisions 54–56): TODOs 42–59 across `src/metrics.py`,
  `src/inference.py` (new), `src/report.py` (new) and `src/cli.py`; 119 new tests, the §8
  determinism test extended to `metrics.json`; new `make metrics` stage and config key
  `inference.ci_level`; starting state `107 failed / 407 passed`, complete at `514 passed`;
  expected numbers in `PHASE5_INSTRUCTIONS.md` §5. Next: implement Phase 5 on `phase-5`, one
  pull request, cross-check, merge.
- **2026-10-09** — Phase 5 scaffold pushed to `origin/main` (`e632ba8..e2bd46c`); CI is red on
  the 107 unimplemented tests by design until the phase is implemented. Next: implement Phase 5
  on `phase-5`, one pull request, cross-check, merge.
- **2026-10-15** — Phase 5 cross-checked and closed: `phase-5` (`c8f7e88`) ruff clean, `514
  passed`, every §8 integrity class green, `tests/`, `data/`, `config/` untouched; the §5
  script reproduces `results/metrics_table.md` line for line and every expected number;
  TODOs 42–59 match the reference in substance; `N_trials` = 0. Merged by squash `ba1d940`
  without a pull request (Decision 57), `results/` in `5c70ec2`, F5.1–F5.3 `DONE`. Next: the
  Phase 6 assignment.
- **2026-10-15** — Phase 5 close-out pushed to `origin/main` (`e2bd46c..74bbc14`); `514
  passed`. Next: the Phase 6 assignment.
- **2026-10-15** — Phase 6 assignment prepared (Decisions 58–61): TODOs 60–77 across
  `src/robustness.py` (new), `src/subperiods.py` (new), `src/report.py` and `src/cli.py`; 141
  new tests; starting state `129 failed / 526 passed`, complete at `655 passed`; expected
  numbers in `PHASE6_INSTRUCTIONS.md` §5. Next: implement Phase 6 on `phase-6`, one pull
  request, cross-check, merge, tag `v0.3`.
- **2026-10-26** — Phase 6 TODOs 60–77 implemented on `phase-6` (uncommitted): ruff clean,
  `655 passed`; `make robustness` from an empty cache 8.1 s, 192 trials, std 0.157, hurdle
  0.433, PSR 0.958, DSR 0.323; §5 output and `results/metrics_table.md` match
  `PHASE6_INSTRUCTIONS.md` line for line. Feature Log and Current State left for the
  close-out, as the instructions say. Next: commit, push, one PR, cross-check, merge, `v0.3`.

## 13. Engineering ground rules

These bind every line under `src/` and `tests/`, whoever writes it. Phase instructions restate
the ones that matter for that phase; this is the full list.

### Rigor non-negotiables

- **No lookahead.** Every quantity used to form a position at `t` depends only on data at or
  before `t`. `positions = weights.shift(lag)`. Every rolling or expanding stage has a
  perturbation test (`tests/integrity.py::assert_no_lookahead`) before it is merged.
- **Gross and net always.** Any table or figure showing strategy returns shows both.
- **Parameters come from `config/default.yaml`.** No magic numbers in `src/`. Every run dumps
  `resolved_config.yaml` beside its outputs.
- **Snapshot first.** Backtests read `data/futures_panel.csv`. Nothing in tests or `make all`
  touches the network. Only `make data` pulls.
- **Report what binds.** Leverage cap hits, clipped returns, filled gaps, excluded instruments:
  all counted and written to `results/`.
- **Do not tune on the full sample.** The headline config is the a-priori default in §2. The
  IS/OOS exercise (§5.3) is the only place parameters are selected, and it is reported as such.
- **Count trials.** Every configuration evaluated in robustness increments `N_trials` for the
  Deflated Sharpe Ratio. No ad-hoc "let me just try" backtests without logging them.

### Tests

- Tests live in `tests/`, mirror module names (`tests/test_signals.py`), use synthetic data
  with known answers. Real-data tests use the committed snapshot only.
- The tests shipped with a phase handoff are its acceptance criteria and are not edited by the
  implementer; the same goes for `data/`, `config/default.yaml`, and `tests/integrity.py`.
- The integrity tests in §8 must all exist and pass before Phase 4 is marked `DONE`.

### Code conventions

- Python 3.12, `uv` for env and deps, `ruff` for lint and format (line length 100).
- Pure functions over pandas objects. I/O only in `data.py`, `plots.py`, `report.py`, `cli.py`.
- Type hints on all public functions. Docstrings state the shape and index of inputs and
  outputs (e.g. "DataFrame indexed by date, columns = tickers").
- Wide DataFrames: index = `DatetimeIndex` (tz-naive, daily), columns = tickers. Keep this
  shape everywhere; convert at the edges.
- Annualisation factor is `252`, defined once in `src/config.py`.
- No notebooks contain logic. `notebooks/strategy.ipynb` imports `src/` and loads `results/`.
- Figures: matplotlib, one function per figure in `plots.py`, saved at 150 dpi, no seaborn.
- Keep functions short. If a module exceeds ~300 lines, split it.

### Git

- Conventional commits: `feat(signals): add TSMOM sign signal`, `test(backtest): lag test`,
  `docs(log): ...`, `fix`, `refactor`, `chore`, `ci`. Messages describe the change and carry
  no trailers.
- One pull request per phase from a branch named `phase-<N>`; merged by squash after the
  numbers check out. Nothing goes to `main` directly.
- `data/` is committed intentionally (snapshot); `results/` and `figures/` are committed only
  when regenerated by `make all` on the current snapshot. Never commit secrets.
- Milestone tags: `v0.1` end of Phase 1, `v0.2` Phase 4, `v0.3` Phase 6, `v1.0` Phase 8.

### Commands

```
make setup        # uv sync
make lint         # ruff check + format --check
make test         # pytest -q
make data         # re-pull from yfinance (network); rewrites snapshot + manifest
make panel        # build aligned panel from snapshot
make backtest     # run default config
make benchmarks
make metrics      # results/metrics.json + metrics_table.md
make robustness   # grid + IS/OOS + sub-periods + sweeps
make figures
make report       # results/*.md + README metrics injection
make notebook     # execute notebook headless
make all          # panel -> backtest -> benchmarks -> metrics -> robustness -> figures -> report
make clean
```

### Do not

- Do not add a parameter without adding it to the config, this spec, and the sensitivity grid
  (or a written reason why it is excluded from the grid).
- Do not change the default config to improve results. Change it only for a documented bug or
  a spec change, and log it in the Decision Log.
- Do not use `pct_change()` on raw prices (negative WTI). Use `data.clean_returns`.
- Do not forward-fill without the limit. Do not drop NaNs silently.
- Do not write to `README.md` results sections by hand; use `make report`.
- Do not skip the `PROJECT_LOG.md` and `WORKLOG.md` updates. A session that ends without
  updating them is incomplete.
