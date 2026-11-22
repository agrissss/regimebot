"""Daily trend following: config, features, weights, risk caps, portfolio trades, the engine's
target-position step, backtest reproduction of the rule, halts and the time-shift baseline.

Hand-computed numbers use zero costs unless a test says otherwise; ``TREND_CFG`` is
``config/trend.yaml`` (BTC, ETH and SOL, 10 000 initial equity, 0.75 asset / 1.0 group / 1.5 gross
caps, band 0.01, min notional 10).
"""

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from synth import BTC, ETH, SOL, T0

from regimebot.config import (
    CONFIG_ENV,
    REPO_ROOT,
    AppConfig,
    decision_warmup_bars,
    load_config,
    timeframe_minutes,
)
from regimebot.strategy import DailyTrend, build_strategies
from regimebot.strategy.daily_trend import trend_features

TREND_YAML = REPO_ROOT / "config" / "trend.yaml"
DAY = timedelta(days=1)


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-12, abs=1e-9)


def trend_cfg(**overrides: object) -> AppConfig:
    return load_config(TREND_YAML, overrides=overrides)


def daily(closes: list[float]) -> pd.DataFrame:
    """Daily bars (close time index) from T0 + 1 day with the given closes; only ``close``
    matters."""
    idx = pd.date_range(T0 + DAY, periods=len(closes), freq="1D", name="ts")
    return pd.DataFrame({"close": closes}, index=idx, dtype=float)


# --- config ----------------------------------------------------------------------------------


def test_default_config_has_daily_trend_disabled_and_trend_yaml_enables_it_alone() -> None:
    default = load_config()
    assert default.strategies.daily_trend.enabled is False
    cfg = trend_cfg()
    trend = cfg.strategies.daily_trend
    assert (trend.enabled, trend.mode, trend.lookbacks, trend.vol_window) == (
        True,
        "long_short",
        [20, 50, 100],
        30,
    )
    assert (trend.target_vol, trend.rebalance_band) == (0.08, 0.01)
    assert cfg.exchange.symbols == [BTC, ETH, SOL] and cfg.exchange.decision_timeframe == "1d"
    assert not any(getattr(cfg.strategies, s).enabled for s in ("breakout", "trend"))
    assert not cfg.strategies.mean_reversion.enabled
    assert cfg.backtest.start.isoformat() == "2021-01-01T00:00:00+00:00"
    assert cfg.backtest.end.isoformat() == "2026-11-21T00:00:00+00:00"
    assert [type(s) for s in build_strategies(cfg)] == [DailyTrend]
    assert timeframe_minutes("1d") == 1440


def test_daily_trend_runs_alone_and_only_on_daily_decisions() -> None:
    with pytest.raises(ValidationError, match="disable breakout"):
        trend_cfg(**{"strategies.breakout.enabled": True})
    with pytest.raises(ValidationError, match="decision_timeframe: 1d"):
        trend_cfg(**{"exchange.decision_timeframe": "15m"})
    with pytest.raises(ValidationError):
        trend_cfg(**{"strategies.daily_trend.mode": "short_only"})
    with pytest.raises(ValidationError):
        trend_cfg(**{"strategies.daily_trend.lookbacks": []})


def test_the_trend_warm_up_replaces_the_regime_feature_warm_up() -> None:
    # max(lookbacks) + 1 = 101 and vol_window + 1 = 31 daily bars; not the 2,976 15m bars
    assert decision_warmup_bars(trend_cfg()) == 101
    long_vol = {"strategies.daily_trend.vol_window": 150, "paper.warmup_bars": 151}
    assert decision_warmup_bars(trend_cfg(**long_vol)) == 151
    assert decision_warmup_bars(load_config()) == 2976
    trend_cfg(**{"paper.warmup_bars": 101})
    with pytest.raises(ValidationError, match="raise paper.warmup_bars to at least 101"):
        trend_cfg(**{"paper.warmup_bars": 100})


def test_regimebot_config_selects_the_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV, str(TREND_YAML))
    assert load_config().strategies.daily_trend.enabled
    monkeypatch.delenv(CONFIG_ENV)
    assert not load_config().strategies.daily_trend.enabled


# --- features and weights --------------------------------------------------------------------

SHORT = {"strategies.daily_trend.lookbacks": [2, 3], "strategies.daily_trend.vol_window": 2}


def test_signal_is_the_mean_sign_of_close_minus_each_sma_including_the_current_close() -> None:
    cfg = trend_cfg(**SHORT).strategies.daily_trend
    f = trend_features(daily([10.0, 12.0, 11.0, 9.0, 9.0]), cfg)
    # row 2: SMA2 = 11.5 (11 below), SMA3 = 11 (11 equal: sign 0) -> (-1 + 0) / 2
    # row 3: SMA2 = 10 (9 below), SMA3 = 32/3 (below) -> -1
    # row 4: SMA2 = 9 (equal), SMA3 = 29/3 (below) -> -0.5
    assert f["signal"].iloc[:2].isna().all()  # SMA3 needs 3 closes
    assert f["signal"].tolist()[2:] == [-0.5, -1.0, -0.5]
    flat = trend_cfg(**SHORT, **{"strategies.daily_trend.mode": "long_flat"}).strategies
    assert trend_features(daily([10.0, 12.0, 11.0, 9.0, 13.0]), flat.daily_trend)[
        "signal"
    ].tolist()[2:] == [0.0, 0.0, 1.0]


def test_volatility_is_the_std_of_daily_returns_annualised_with_365_days() -> None:
    cfg = trend_cfg(**SHORT).strategies.daily_trend
    f = trend_features(daily([100.0, 110.0, 99.0, 99.0]), cfg)
    # returns +0.10, -0.10, 0.0; std(ddof=1) of the last two windows: 0.1414..., 0.0707...
    assert math.isnan(f["vol"].iloc[1])  # needs vol_window + 1 = 3 closes
    assert f["vol"].iloc[2] == approx(np.std([0.1, -0.1], ddof=1) * math.sqrt(365))
    assert f["vol"].iloc[3] == approx(np.std([-0.1, 0.0], ddof=1) * math.sqrt(365))


def test_features_are_causal_a_prefix_gives_the_same_rows() -> None:
    cfg = trend_cfg().strategies.daily_trend
    closes = list(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.03, 400))))
    full = trend_features(daily(closes), cfg)
    for cut in (101, 150, 399):
        pd.testing.assert_frame_equal(trend_features(daily(closes[:cut]), cfg), full.iloc[:cut])
    first = full.dropna().index[0]
    assert first == full.index[99]  # every SMA full at the 100th close, the 30-day vol long before


def rows(**signal_vol: tuple[float, float]) -> dict[str, dict[str, float]]:
    names = {"btc": BTC, "eth": ETH, "sol": SOL}
    return {names[k]: {"close": 100.0, "signal": s, "vol": v} for k, (s, v) in signal_vol.items()}


def test_weight_is_signal_times_the_capped_volatility_target_split_by_the_symbol_count() -> None:
    cfg = trend_cfg()  # target_vol 0.08, 3 symbols, asset cap 0.75, gross cap 1.5
    strategy = DailyTrend(cfg.strategies.daily_trend, cfg.risk, cfg.exchange.symbols)
    got = strategy.weights(T0, rows(btc=(1.0, 0.4), eth=(-1 / 3, 0.8), sol=(0.5, 0.01)))
    assert got == {
        BTC: approx(0.08 / 3 / 0.4),  # 0.0667
        ETH: approx(-1 / 3 * 0.08 / 3 / 0.8),
        SOL: approx(0.5 * 0.75),  # 0.08 / 3 / 0.01 = 2.67 is capped at 0.75
    }
    # a symbol without a full window has no weight (it is not traded at this decision)
    assert strategy.weights(T0, rows(btc=(math.nan, 0.4), eth=(1.0, math.nan))) == {}


def test_weights_above_the_gross_cap_are_scaled_down_together() -> None:
    cfg = trend_cfg(**{"strategies.daily_trend.target_vol": 10.0})
    strategy = DailyTrend(cfg.strategies.daily_trend, cfg.risk, cfg.exchange.symbols)
    got = strategy.weights(T0, rows(btc=(1.0, 0.5), eth=(-1.0, 0.5), sol=(0.5, 0.5)))
    # each capped at 0.75 (signal 1, -1, 0.5): gross 0.75 + 0.75 + 0.375 = 1.875 > 1.5
    assert got == {BTC: approx(0.6), ETH: approx(-0.6), SOL: approx(0.3)}
    assert sum(abs(w) for w in got.values()) == approx(1.5)
