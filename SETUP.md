# SETUP — from a fresh machine to a green test suite

Everything below is copy-paste. The whole thing takes a few minutes; nothing needs a network
connection after step 3 except `make data`, which you will almost never run.

## 1. Install uv

`uv` manages the Python version and the virtual environment. Nothing else is required, not
even a system Python.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(On Windows use PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.)

Open a new terminal afterwards and check:

```
uv --version
```

## 2. Clone the repository

```
git clone git@github.com:Jesstran12/Project-2-Multi-Basket-TSMOM-Strategy-Portfolio-Risk-Threshold-Model.git
cd Project-2-Multi-Basket-TSMOM-Strategy-Portfolio-Risk-Threshold-Model
```

## 3. Create the environment

```
make setup
```

This runs `uv sync`: it installs Python 3.12 (pinned in `.python-version`) and every
dependency at the exact versions locked in `uv.lock`. It creates a `.venv/` folder, which git
ignores. You never activate it by hand: every command goes through `uv run`, and the Makefile
already does that.

## 4. Check the suite is green

```
make lint test
```

Expected: `ruff` reports no issues, and `pytest` ends with all tests passed. If this is green
you are set up. If it is not, the problem is with the environment, not the code; re-run
`make setup` and try again, and if it still fails send the full output.

## Day to day

| Command | What it does |
|---|---|
| `make lint` | `ruff check` + `ruff format --check`. Run before every commit. |
| `make format` | Auto-fixes formatting and simple lint issues. |
| `make test` | `pytest` on the `tests/` folder. |
| `make lint test` | Both, the definition of "green". |
| `make all` | Runs the whole pipeline from the committed data snapshot (once those stages exist). |
| `make clean` | Removes caches. |
| `uv run python -m src.cli <stage>` | Runs one pipeline stage by hand. |
| `uv run python` | A Python shell with the project environment. |

Each phase arrives as a `PHASE<N>_INSTRUCTIONS.md` file at the top of the repository. It
names the branch to work on, the functions to fill in, and the exact `make test` output to
expect before and after. Work happens on a `phase-<N>` branch and lands through one pull
request; nothing is committed to `main` directly.

## The no-refetch rule

Backtests read the committed data snapshot in `data/`. Tests and `make all` never touch the
network, so results are reproducible by anyone who clones the repository. `make data` is the
one command that does go to the network: it re-downloads prices from Yahoo Finance and
rewrites the snapshot and its manifest. Do not run it unless a phase's instructions say to,
because a new snapshot changes every number downstream.

## Editor notes

- Any editor works. If yours has Python support, point it at `.venv/bin/python` so imports
  and type hints resolve.
- Line length is 100. `make format` applies the project style, so there is no need to
  configure it by hand.
