"""Config system: YAML load, dotted-key overrides, validation, resolved dump."""

from pathlib import Path

import pytest
import yaml

from src import config as cfgmod
from src.config import ANNUALISATION, Config, ConfigError, dump_resolved, load_config

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_annualisation_factor_is_252():
    assert ANNUALISATION == 252


def test_load_default_returns_typed_config_with_spec_defaults():
    cfg = load_config(DEFAULT_YAML)
    assert isinstance(cfg, Config)
    assert cfg.signal.lookback == 252
    assert cfg.signal.skip == 0
    assert cfg.sizing.vol_window == 60
    assert cfg.sizing.vol_estimator == "rolling"
    assert cfg.sizing.sigma_target_asset == 0.10
    assert cfg.sizing.sigma_target_portfolio == 0.10
    assert cfg.sizing.leverage_cap == 3.0
    assert cfg.execution.rebalance == "monthly"
    assert cfg.execution.lag == 1
    assert cfg.costs.cost_bps == 5
    assert cfg.data.ffill_limit == 5
    assert cfg.data.clip == 0.5
    assert len(cfg.data.tickers) == 14
    assert "CL=F" in cfg.data.tickers


def test_load_without_path_uses_default_yaml():
    assert load_config() == load_config(DEFAULT_YAML)


def test_default_yaml_covers_every_config_field():
    raw = yaml.safe_load(DEFAULT_YAML.read_text())
    cfg = load_config(DEFAULT_YAML)
    assert set(raw) == set(cfg.to_dict())
    for section, values in cfg.to_dict().items():
        assert set(raw[section]) == set(values), section


def test_dotted_override_changes_nested_value():
    cfg = load_config(DEFAULT_YAML, overrides={"signal.lookback": 126})
    assert cfg.signal.lookback == 126
    assert cfg.sizing.vol_window == 60  # untouched


def test_override_from_string_is_coerced_to_field_type():
    cfg = load_config(DEFAULT_YAML, overrides={"costs.cost_bps": "10", "sizing.leverage_cap": "2"})
    assert cfg.costs.cost_bps == 10.0
    assert cfg.sizing.leverage_cap == 2.0


def test_override_unknown_key_raises():
    with pytest.raises(ConfigError, match="unknown"):
        load_config(DEFAULT_YAML, overrides={"signal.lookbak": 126})
    with pytest.raises(ConfigError, match="unknown"):
        load_config(DEFAULT_YAML, overrides={"nosuchsection.x": 1})


def test_unknown_key_in_yaml_raises(tmp_path):
    raw = yaml.safe_load(DEFAULT_YAML.read_text())
    raw["signal"]["mystery"] = 1
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="unknown"):
        load_config(p)


def test_missing_key_in_yaml_raises(tmp_path):
    raw = yaml.safe_load(DEFAULT_YAML.read_text())
    del raw["signal"]["lookback"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="missing"):
        load_config(p)


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("signal.lookback", 0, "lookback"),
        ("signal.skip", -1, "skip"),
        ("signal.skip", 252, "skip"),
        ("sizing.vol_window", 1, "vol_window"),
        ("sizing.vol_estimator", "garch", "vol_estimator"),
        ("sizing.sigma_target_asset", 0.0, "sigma_target_asset"),
        ("sizing.sigma_target_portfolio", -0.1, "sigma_target_portfolio"),
        ("sizing.leverage_cap", 0.0, "leverage_cap"),
        ("execution.rebalance", "weekly", "rebalance"),
        ("execution.lag", 0, "lag"),
        ("costs.cost_bps", -1, "cost_bps"),
        ("data.ffill_limit", -1, "ffill_limit"),
        ("data.clip", 0.0, "clip"),
        ("data.clip", 1.5, "clip"),
        ("inference.n_resamples", 0, "n_resamples"),
        ("inference.block_length", 0, "block_length"),
        ("inference.ci_level", 0.0, "ci_level"),
        ("inference.ci_level", 1.0, "ci_level"),
        ("robustness.is_end", "2013-06-30", "is_end"),
        ("robustness.regime_vol_quantile", 0.0, "regime_vol_quantile"),
        ("robustness.regime_vol_quantile", 1.0, "regime_vol_quantile"),
    ],
)
def test_validation_errors(key, value, match):
    with pytest.raises(ConfigError, match=match):
        load_config(DEFAULT_YAML, overrides={key: value})


def test_dump_resolved_round_trips(tmp_path):
    cfg = load_config(DEFAULT_YAML, overrides={"signal.lookback": 63})
    out = tmp_path / "resolved_config.yaml"
    dump_resolved(cfg, out)
    assert out.exists()
    assert load_config(out) == cfg
    assert yaml.safe_load(out.read_text())["signal"]["lookback"] == 63


def test_config_is_immutable():
    cfg = load_config(DEFAULT_YAML)
    with pytest.raises(AttributeError):
        cfg.signal.lookback = 10  # type: ignore[misc]


def test_module_has_no_io_beyond_config_files():
    # The loader may only read/write YAML config files; guard against network imports.
    assert not hasattr(cfgmod, "yf")
