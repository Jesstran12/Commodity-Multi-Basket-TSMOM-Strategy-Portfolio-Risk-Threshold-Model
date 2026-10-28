"""Typed configuration: YAML loader, dotted-key overrides, validation, resolved dump.

All strategy parameters live in ``config/default.yaml`` (PROJECT_LOG.md §2-§5). Code reads them
through the frozen dataclasses below; nothing in ``src/`` hard-codes a parameter.
"""

import dataclasses
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

ANNUALISATION = 252
"""Trading days per year. Defined once; every annualisation in ``src/`` uses this."""

BASIS_POINT = 1e-4
"""One basis point as a fraction. Defined once; the cost model converts ``cost_bps`` with it."""

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"

VOL_ESTIMATORS = ("rolling", "ewma")
REBALANCES = ("monthly", "daily")
SOURCES = ("yfinance", "csv")


class ConfigError(ValueError):
    """Raised for unknown keys, missing keys, bad types, or out-of-range values."""


@dataclass(frozen=True)
class SignalConfig:
    lookback: int
    skip: int


@dataclass(frozen=True)
class SizingConfig:
    sigma_target_asset: float
    sigma_target_portfolio: float
    leverage_cap: float
    vol_window: int
    vol_estimator: str


@dataclass(frozen=True)
class ExecutionConfig:
    rebalance: str
    lag: int


@dataclass(frozen=True)
class CostsConfig:
    cost_bps: float


@dataclass(frozen=True)
class DataConfig:
    tickers: tuple[str, ...]
    price_field: str
    start: str | None
    ffill_limit: int
    clip: float
    source: str
    csv_path: str | None
    fetch_retries: int
    raw_dir: str
    panel_path: str
    benchmark_path: str
    manifest_path: str


@dataclass(frozen=True)
class QualityGateConfig:
    min_years: float
    max_fill_fraction: float
    max_gap_days: int


@dataclass(frozen=True)
class BenchmarksConfig:
    external_tickers: tuple[str, ...]


@dataclass(frozen=True)
class InferenceConfig:
    risk_free_rate: float
    block_length: int
    n_resamples: int
    seed: int
    ci_level: float


@dataclass(frozen=True)
class RobustnessConfig:
    grid_lookbacks: tuple[int, ...]
    grid_vol_windows: tuple[int, ...]
    grid_estimators: tuple[str, ...]
    grid_rebalances: tuple[str, ...]
    grid_skips: tuple[int, ...]
    grid_lags: tuple[int, ...]
    cost_bps_sweep: tuple[float, ...]
    is_end: str
    oos_start: str
    subperiod_blocks: tuple[tuple[str | None, str | None], ...]
    rolling_sharpe_years: int
    regime_vol_quantile: float


@dataclass(frozen=True)
class PathsConfig:
    results_dir: str
    figures_dir: str


@dataclass(frozen=True)
class Config:
    signal: SignalConfig
    sizing: SizingConfig
    execution: ExecutionConfig
    costs: CostsConfig
    data: DataConfig
    quality_gate: QualityGateConfig
    benchmarks: BenchmarksConfig
    inference: InferenceConfig
    robustness: RobustnessConfig
    paths: PathsConfig

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (tuples become lists), suitable for YAML dumping."""
        return _to_plain(dataclasses.asdict(self))


# --------------------------------------------------------------------------- loading


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply ``{"section.key": value}`` overrides, validate, return.

    Override values may be strings (e.g. from a CLI) and are coerced to the field's type.
    """
    src = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(src.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{src}: top level must be a mapping")
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    cfg = _build(Config, raw, "")
    _validate(cfg)
    return cfg


def dump_resolved(cfg: Config, path: str | Path) -> Path:
    """Write the fully resolved config as YAML beside a run's outputs. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return out


def _set_dotted(raw: dict[str, Any], dotted: str, value: Any) -> None:
    section, _, key = dotted.partition(".")
    if not key or section not in raw or not isinstance(raw[section], dict):
        raise ConfigError(f"unknown config key '{dotted}'")
    if key not in raw[section]:
        raise ConfigError(f"unknown config key '{dotted}'")
    raw[section][key] = value


def _build(cls: type, data: Any, where: str) -> Any:
    """Recursively construct a dataclass from a dict, rejecting unknown and missing keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping")
    names = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise ConfigError(f"unknown config key(s) under '{where or 'root'}': {sorted(unknown)}")
    missing = set(names) - set(data)
    if missing:
        raise ConfigError(f"missing config key(s) under '{where or 'root'}': {sorted(missing)}")
    kwargs = {n: _coerce(names[n].type, data[n], f"{where}.{n}".strip(".")) for n in data}
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any, where: str) -> Any:
    """Coerce a YAML/CLI value to the annotated field type."""
    tp = annotation
    origin = get_origin(tp)
    if origin is types.UnionType:
        return _coerce_optional(tp, value, where)
    if is_dataclass(tp):
        return _build(tp, value, where)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: expected a list")
        (item_tp, _ellipsis) = get_args(tp)
        return tuple(_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))
    if tp is bool:
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return bool(value)
    if tp is int:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{where}: expected an integer, got {value!r}") from e
    if tp is float:
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{where}: expected a number, got {value!r}") from e
    if tp is str:
        if value is None:
            raise ConfigError(f"{where}: expected a string, got null")
        return str(value)
    return value


def _coerce_optional(tp: Any, value: Any, where: str) -> Any:
    if value is None or (isinstance(value, str) and value.lower() in ("", "null", "none")):
        return None
    inner = next(a for a in get_args(tp) if a is not type(None))
    return _coerce(inner, value, where)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    return obj


# ------------------------------------------------------------------------ validation


def _validate(cfg: Config) -> None:
    """Range and choice checks. Raises ConfigError naming the offending key."""
    s, z, e, c, d = cfg.signal, cfg.sizing, cfg.execution, cfg.costs, cfg.data
    _check(s.lookback >= 1, "signal.lookback must be >= 1")
    _check(0 <= s.skip < s.lookback, "signal.skip must be in [0, lookback)")
    _check(z.sigma_target_asset > 0, "sizing.sigma_target_asset must be > 0")
    _check(z.sigma_target_portfolio > 0, "sizing.sigma_target_portfolio must be > 0")
    _check(z.leverage_cap > 0, "sizing.leverage_cap must be > 0")
    _check(z.vol_window >= 2, "sizing.vol_window must be >= 2")
    _check(
        z.vol_estimator in VOL_ESTIMATORS, f"sizing.vol_estimator must be one of {VOL_ESTIMATORS}"
    )
    _check(e.rebalance in REBALANCES, f"execution.rebalance must be one of {REBALANCES}")
    _check(e.lag >= 1, "execution.lag must be >= 1")
    _check(c.cost_bps >= 0, "costs.cost_bps must be >= 0")
    _check(len(d.tickers) > 0, "data.tickers must not be empty")
    _check(len(set(d.tickers)) == len(d.tickers), "data.tickers must be unique")
    _check(d.ffill_limit >= 0, "data.ffill_limit must be >= 0")
    _check(0 < d.clip <= 1, "data.clip must be in (0, 1]")
    _check(d.source in SOURCES, f"data.source must be one of {SOURCES}")
    _check(d.source != "csv" or bool(d.csv_path), "data.csv_path is required when source = csv")
    _check(d.fetch_retries >= 1, "data.fetch_retries must be >= 1")
    _validate_gate_and_inference(cfg)
    _validate_robustness(cfg.robustness)


def _validate_gate_and_inference(cfg: Config) -> None:
    q, i = cfg.quality_gate, cfg.inference
    _check(q.min_years >= 0, "quality_gate.min_years must be >= 0")
    _check(0 <= q.max_fill_fraction <= 1, "quality_gate.max_fill_fraction must be in [0, 1]")
    _check(q.max_gap_days >= 0, "quality_gate.max_gap_days must be >= 0")
    _check(i.block_length >= 1, "inference.block_length must be >= 1")
    _check(i.n_resamples >= 1, "inference.n_resamples must be >= 1")
    _check(0 < i.ci_level < 1, "inference.ci_level must be in (0, 1)")


def _validate_robustness(r: RobustnessConfig) -> None:
    _check(all(x >= 1 for x in r.grid_lookbacks), "robustness.grid_lookbacks must be >= 1")
    _check(all(x >= 2 for x in r.grid_vol_windows), "robustness.grid_vol_windows must be >= 2")
    _check(set(r.grid_estimators) <= set(VOL_ESTIMATORS), "robustness.grid_estimators invalid")
    _check(set(r.grid_rebalances) <= set(REBALANCES), "robustness.grid_rebalances invalid")
    _check(all(x >= 0 for x in r.grid_skips), "robustness.grid_skips must be >= 0")
    _check(all(x >= 1 for x in r.grid_lags), "robustness.grid_lags must be >= 1")
    _check(all(x >= 0 for x in r.cost_bps_sweep), "robustness.cost_bps_sweep must be >= 0")
    _check(r.is_end < r.oos_start, "robustness.is_end must be before oos_start")
    _check(r.rolling_sharpe_years >= 1, "robustness.rolling_sharpe_years must be >= 1")
    _check(0 < r.regime_vol_quantile < 1, "robustness.regime_vol_quantile must be in (0, 1)")
    for start, end in r.subperiod_blocks:
        _check(start is None or end is None or start <= end, "robustness.subperiod_blocks order")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)
