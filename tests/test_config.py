from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from regimebot.config import (
    DEFAULT_CONFIG_PATH,
    config_hash,
    config_json,
    load_config,
    timeframe_minutes,
)
from regimebot.types import Regime, Side, side_sign


@pytest.fixture(autouse=True)
def _no_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_default_config_loads_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.database_url == "sqlite:///data/regimebot.db"
    assert set(cfg.playbook) == set(Regime)
    assert cfg.risk.max_gross_leverage == 1.5
    assert cfg.strategies.mean_reversion.trail_atr_mult is None
    assert cfg.backtest.start.utcoffset() is not None


def test_overrides_apply_and_unknown_path_raises() -> None:
    cfg = load_config(overrides={"risk.risk_per_trade": 0.01, "strategies.trend.min_rr": 3.0})
    assert cfg.risk.risk_per_trade == 0.01
    assert cfg.strategies.trend.min_rr == 3.0
    for bad in ("risk.nope", "nope.risk_per_trade", "risk.risk_per_trade.deeper"):
        with pytest.raises(ValueError, match="unknown config path"):
            load_config(overrides={bad: 1})


def test_override_is_validated() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides={"risk.max_gross_leverage": 1.6})


def test_missing_playbook_regime_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    del raw["playbook"]["range"]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="playbook missing regimes"):
        load_config(path)


def test_ema_fast_must_be_below_slow() -> None:
    with pytest.raises(ValidationError, match="ema_fast"):
        load_config(overrides={"features.ema_fast": 384})


def test_pctile_window_must_cover_rv_window() -> None:
    with pytest.raises(ValidationError, match="pctile_window"):
        load_config(overrides={"features.pctile_window": 10})


def test_fraction_and_timeframe_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides={"risk.daily_loss": 0.0})
    with pytest.raises(ValidationError):
        load_config(overrides={"exchange.decision_timeframe": "7m"})


def test_config_hash_stable_and_sensitive() -> None:
    base = config_hash(load_config())
    assert base == config_hash(load_config())
    assert len(base) == 16
    assert base != config_hash(load_config(overrides={"risk.risk_per_trade": 0.01}))


def test_database_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db/x")
    assert load_config().database_url == "postgresql+psycopg://u:p@db/x"
    monkeypatch.setenv("DATABASE_URL", "")
    assert load_config().database_url == "sqlite:///data/regimebot.db"


def test_timeframe_minutes() -> None:
    assert [timeframe_minutes(t) for t in ("1m", "5m", "15m", "1h", "4h")] == [1, 5, 15, 60, 240]
    with pytest.raises(ValueError):
        timeframe_minutes("2h")


def _write(tmp_path: Path, mutate) -> Path:  # type: ignore[no-untyped-def]  # test helper
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    mutate(raw)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_unknown_playbook_key_and_strategy_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, lambda r: r["playbook"].update(bogus=r["playbook"]["range"])))
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, lambda r: r["playbook"]["range"].update(strategies=["x"])))


def test_non_positive_window_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides={"features.atr_window": 0})


def test_percent_style_fractions_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(overrides={"strategies.breakout.min_target_pct": 5})
    with pytest.raises(ValidationError):
        load_config(overrides={"playbook.range.size_mult": 2.0})


def test_datetimes_normalised_to_utc_and_hash_equal() -> None:
    plus2 = load_config(overrides={"backtest.start": "2024-01-01T02:00:00+02:00"})
    zulu = load_config(overrides={"backtest.start": "2024-01-01T00:00:00Z"})
    assert plus2.backtest.start.utcoffset().total_seconds() == 0  # type: ignore[union-attr]  # aware
    assert config_hash(plus2) == config_hash(zulu)


def test_event_window_ordered_and_utc() -> None:
    ok = {"start": "2024-05-01T10:00:00+02:00", "end": "2024-05-01T12:00:00Z", "label": "fomc"}
    cfg = load_config(overrides={"regime.event_windows": [ok]})
    assert cfg.regime.event_windows[0].start.isoformat() == "2024-05-01T08:00:00+00:00"
    with pytest.raises(ValidationError, match="before end"):
        load_config(overrides={"regime.event_windows": [{**ok, "end": ok["start"]}]})


def test_config_json_and_hash_exclude_database_url() -> None:
    a = load_config(overrides={"database_url": "postgresql+psycopg://u:secret@db/x"})
    b = load_config()
    assert "database_url" not in config_json(a) and "secret" not in config_json(a)
    assert config_hash(a) == config_hash(b)


def test_empty_or_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    for text in ("", "- a\n- b\n"):
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ValueError, match="bad.yaml"):
            load_config(path)


def test_side_sign() -> None:
    assert side_sign(Side.LONG) == 1
    assert side_sign(Side.SHORT) == -1
    assert side_sign("long") == 1
    assert side_sign("short") == -1
    with pytest.raises(ValueError):
        side_sign("sideways")


def test_cost_bounds() -> None:
    load_config()  # default passes
    with pytest.raises(ValidationError, match="worst-case taker cost"):
        load_config(overrides={"costs.impact_bps_per_pct": 10.0})  # 1000 + 3 bps
    with pytest.raises(ValidationError):
        load_config(overrides={"costs.slippage_bps": 1001.0})
