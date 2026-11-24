"""The daily signals behind ``strategies.daily_trend.signal``: ``sma`` (the default), ``donchian``,
``momentum`` and ``hold``. Each is a per-symbol value in [-1, 1] that goes through the same
sizing, mode, band, fills and risk as the SMA rule; hand-computed values include the ties (sign
0) and the warm-up of each rule.
"""

import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from synth import BTC, ETH, T0
from test_daily_trend import daily
from test_daily_trend_engine import FREE, market, trend_cfg

from regimebot.backtest.runner import run_backtest
from regimebot.config import AppConfig, config_hash, config_json, decision_warmup_bars
from regimebot.market_state.features import warm_rows
from regimebot.strategy.daily_trend import trend_features

DAY = timedelta(days=1)
NAN = math.nan


def signal(closes: list[float], rule: str, lookbacks: list[int], mode: str = "long_short") -> list:
    cfg = trend_cfg(
        **{
            "strategies.daily_trend.signal": rule,
            "strategies.daily_trend.lookbacks": lookbacks,
            "strategies.daily_trend.vol_window": 2,
            "strategies.daily_trend.mode": mode,
        }
    ).strategies.daily_trend
    return trend_features(daily(closes), cfg)["signal"].tolist()


def same(got: list[float], expected: list[float]) -> bool:
    return np.array_equal(np.array(got), np.array(expected), equal_nan=True)


# --- the rules -------------------------------------------------------------------------------


def test_donchian_is_the_mean_sign_of_close_against_the_mid_of_its_channel() -> None:
    closes = [10.0, 12.0, 11.0, 9.0, 9.0, 10.5]
    # n=2 (current close included): -, mid 11 (12 above: +1), 11.5 (-1), 10 (-1), 9 (equal: 0),
    #   9.75 (+1)
    # n=3: -, -, mid 11 (11 equal: 0), 10.5 (9 below: -1), 10 (-1), 9.75 (+1)
    assert same(signal(closes, "donchian", [2, 3]), [NAN, NAN, -0.5, -1.0, -0.5, 1.0])
    assert same(signal(closes, "donchian", [2, 3], "long_flat"), [NAN, NAN, 0, 0, 0, 1.0])


def test_momentum_is_the_mean_sign_of_the_n_day_returns() -> None:
    closes = [10.0, 12.0, 11.0, 10.0, 9.0, 11.0, 11.0]
    # n=1: -, +1, -1, -1, -1, +1, 0 (11 = 11)
    # n=3 needs 4 closes: -, -, -, 0 (10 = 10), -1 (9 < 12), 0 (11 = 11), +1 (11 > 10)
    expected = [NAN, NAN, NAN, -0.5, -1.0, 0.5, 0.5]
    assert same(signal(closes, "momentum", [1, 3]), expected)
    assert same(signal(closes, "momentum", [1, 3], "long_flat"), [NAN, NAN, NAN, 0, 0, 0.5, 0.5])


def test_hold_is_always_long_whatever_the_closes_and_the_lookbacks() -> None:
    closes = [10.0, 8.0, 6.0, 4.0]
    assert signal(closes, "hold", [3]) == [1.0] * 4
    assert signal(closes, "hold", [3], "long_flat") == [1.0] * 4


def test_sma_is_the_default_and_unchanged() -> None:
    assert trend_cfg().strategies.daily_trend.signal == "sma"
    closes = [10.0, 12.0, 11.0, 9.0, 9.0]
    assert same(signal(closes, "sma", [2, 3]), [NAN, NAN, -0.5, -1.0, -0.5])  # test_daily_trend
    with pytest.raises(ValidationError):
        trend_cfg(**{"strategies.daily_trend.signal": "rsi"})


@pytest.mark.parametrize(
    ("rule", "need", "first_valid"),
    [("sma", 101, 100), ("donchian", 101, 100), ("momentum", 101, 101), ("hold", 31, 31)],
)
def test_each_rule_is_warm_after_its_warm_up_bars(rule: str, need: int, first_valid: int) -> None:
    # lookbacks [20, 50, 100] and vol_window 30: momentum needs n + 1 closes, hold only the
    # volatility's 31; sma and donchian keep the warm-up the SMA rule has always had (one spare)
    cfg = trend_cfg(**{"strategies.daily_trend.signal": rule})
    assert cfg.strategies.daily_trend.warmup_bars() == decision_warmup_bars(cfg) == need
    closes = list(100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.03, 150))))
    warm = warm_rows(trend_features(daily(closes), cfg.strategies.daily_trend))
    assert int(np.argmax(warm.to_numpy())) + 1 == first_valid  # closes at the first warm row
    assert warm.iloc[first_valid - 1 :].all() and first_valid <= need


def test_the_signal_is_left_out_of_the_stored_config_at_its_default() -> None:
    assert '"signal"' not in config_json(trend_cfg())
    donchian = trend_cfg(**{"strategies.daily_trend.signal": "donchian"})
    assert '"signal":"donchian"' in config_json(donchian)
    assert config_hash(donchian) != config_hash(trend_cfg())
    stored = json.loads(config_json(trend_cfg())) | {"database_url": "sqlite://"}
    assert AppConfig.model_validate(stored) == trend_cfg(**{"database_url": "sqlite://"})


# --- through the engine ----------------------------------------------------------------------


def donchian(closes: pd.DataFrame, n: int) -> pd.DataFrame:
    window = closes.rolling(n)
    return np.sign(closes - (window.max() + window.min()) / 2)


def momentum(closes: pd.DataFrame, n: int) -> pd.DataFrame:
    return np.sign(closes / closes.shift(n) - 1)


def vectorised_equity(
    candles: dict[str, pd.DataFrame], cfg: AppConfig, start: object, end: object
) -> pd.Series:
    """The rule written out directly, as ``test_daily_trend_engine.vectorised_equity`` does for
    the SMA: each day's weights held over the next day, no costs."""
    t = cfg.strategies.daily_trend
    at_midnight = {s: (df.index.hour == 0) & (df.index.minute == 0) for s, df in candles.items()}
    closes = pd.DataFrame({s: df["close"][at_midnight[s]] for s, df in candles.items()})
    if t.signal == "hold":
        signal = closes * 0 + 1.0
    else:
        rule = {"donchian": donchian, "momentum": momentum}[t.signal]
        signal = sum(rule(closes, n) for n in t.lookbacks) / len(t.lookbacks)
    vol = closes.pct_change().rolling(t.vol_window).std() * np.sqrt(365)
    w = signal * (t.target_vol / closes.shape[1] / vol).clip(upper=cfg.risk.max_asset_exposure)
    w = w.div((w.abs().sum(axis=1) / cfg.risk.max_gross_leverage).clip(lower=1), axis=0)
    w = w[(w.index >= start) & (w.index < end)]
    r = closes.pct_change().loc[w.index]
    daily_r = (w.shift(1).fillna(0) * r).sum(axis=1)
    return cfg.backtest.initial_equity * (1 + daily_r).cumprod()


@pytest.mark.parametrize("rule", ["donchian", "momentum", "hold"])
def test_the_engine_trades_each_rule_through_the_normal_path(rule: str, tmp_path: Path) -> None:
    days = 120
    cfg = trend_cfg(
        **FREE,
        **{
            "exchange.symbols": [BTC, ETH],
            "strategies.daily_trend.signal": rule,
            "strategies.daily_trend.lookbacks": [5, 10, 20],
            "strategies.daily_trend.vol_window": 10,
            "strategies.daily_trend.target_vol": 1.2,
            "strategies.daily_trend.rebalance_band": 0.0,
            "risk.min_notional": 1e-9,
            "risk.max_asset_exposure": 0.5,
            "risk.max_group_exposure": 1.5,
            "risk.daily_loss": 1.0,
            "risk.weekly_loss": 1.0,
            "risk.max_drawdown": 1.0,
            "costs.max_participation": 1.0,
            "paper.warmup_bars": 21,
        },
    )
    candles = market(days, seed=11)
    start, end = T0 + 25 * DAY, T0 + days * DAY
    result = run_backtest(cfg, candles, {}, start, end, tmp_path / "KILL")
    expected = vectorised_equity(candles, cfg, start, end)
    np.testing.assert_allclose(result.equity.to_numpy(), expected.to_numpy(), rtol=1e-9)
    sides = {t.side.value for t in result.trades}
    assert len(result.log.fills) > 4 and sides == (
        {"long"} if rule == "hold" else {"long", "short"}
    )
