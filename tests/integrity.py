"""Integrity test harness (PROJECT_LOG.md §8, F2.2): the no-lookahead check.

Not a test module itself (pytest collects ``test_*.py`` only); every stage's tests import
:func:`assert_no_lookahead` from here. The harness is test infrastructure and is never a
TODO: it ships complete and is not edited by the implementer, like ``tests/`` as a whole.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Stage = Callable[[pd.DataFrame], pd.DataFrame | pd.Series]
"""A pipeline stage under test: one wide frame in, a wide frame or a series out."""

NOISE_SCALE = 0.5
"""Std of the log-normal noise applied to every future price (large, so signs flip)."""

HOLE_FRACTION = 0.1
"""Share of future cells set to NaN in each noise trial, on top of the noise."""


def assert_no_lookahead(
    fn: Stage,
    prices: pd.DataFrame,
    t: pd.Timestamp | str,
    *,
    n_trials: int = 3,
    seed: int = 0,
) -> None:
    """Assert that ``fn(prices)`` on and before ``t`` does not depend on anything after ``t``.

    Two attacks, each followed by the same check: ``fn`` is run on the attacked input and its
    output on the rows up to and including ``t`` must be identical, value for value and NaN
    for NaN, to the output on the untouched input.

    1. **Noise and holes**, ``n_trials`` times with fresh draws: every price strictly after
       ``t`` is multiplied by ``exp(N(0, NOISE_SCALE))`` and a random ``HOLE_FRACTION`` of
       those cells is set to NaN. Seeded, so a failure reproduces.
    2. **Truncation**: every row after ``t`` is deleted, so a function that peeks at the end
       of the sample (a full-sample mean, a centred window, ``bfill``) has nothing to peek at.

    Parameters
    ----------
    fn:
        The stage, as a one-argument callable (wrap extra arguments in a lambda).
    prices:
        Wide frame indexed by date. Not modified.
    t:
        A date in ``prices.index`` with at least one row after it; a string is accepted.

    Raises
    ------
    AssertionError
        Naming the attack, the trial and the first date on or before ``t`` where the output
        differs (or the shape / index mismatch).
    ValueError
        If ``t`` is not in the index or is its last row (the attack would be empty).
    """
    t = pd.Timestamp(t)
    if t not in prices.index:
        raise ValueError(f"t={t.date()} is not in the index")
    future = prices.index > t
    if not future.any():
        raise ValueError(f"t={t.date()} is the last row; nothing after it to perturb")

    base = _past(fn(prices), t)
    rng = np.random.default_rng(seed)
    n_future = int(future.sum())
    for trial in range(n_trials):
        attacked = prices.copy()
        noise = np.exp(rng.normal(0.0, NOISE_SCALE, size=(n_future, prices.shape[1])))
        values = attacked.loc[future].to_numpy(dtype=float) * noise
        values[rng.random(values.shape) < HOLE_FRACTION] = np.nan
        attacked.loc[future] = values
        _assert_same(base, _past(fn(attacked), t), f"noise trial {trial + 1}/{n_trials}")

    _assert_same(base, _past(fn(prices.loc[:t]), t), "truncation")


def _past(out: pd.DataFrame | pd.Series, t: pd.Timestamp) -> pd.DataFrame | pd.Series:
    if not isinstance(out, pd.DataFrame | pd.Series):
        raise TypeError(f"stage must return a DataFrame or Series, got {type(out).__name__}")
    return out.loc[:t]


def _assert_same(
    base: pd.DataFrame | pd.Series, other: pd.DataFrame | pd.Series, attack: str
) -> None:
    where = f"lookahead detected under the {attack} attack"
    if not base.index.equals(other.index):
        raise AssertionError(f"{where}: the output index on or before t changed")
    if isinstance(base, pd.DataFrame):
        if not isinstance(other, pd.DataFrame) or not base.columns.equals(other.columns):
            raise AssertionError(f"{where}: the output columns changed")
    a, b = np.asarray(base.to_numpy()), np.asarray(other.to_numpy())
    if a.shape != b.shape:
        raise AssertionError(f"{where}: output shape {a.shape} became {b.shape}")
    same = (a == b) | (pd.isna(a) & pd.isna(b))
    if not same.all():
        rows = np.nonzero(~same.reshape(len(base), -1).all(axis=1))[0]
        first = base.index[rows[0]]
        raise AssertionError(
            f"{where}: output on or before t differs, first at {first.date()} "
            f"({len(rows)} row(s) differ)"
        )
