"""Strategy tests on hand-built feature windows (compute_features is not used)."""

import math

import pandas as pd
import pytest

from regimebot.config import load_config
from regimebot.strategy import Breakout, MeanReversion, Trend, build_strategies
from regimebot.strategy.base import build_signal, target_is_plausible
from regimebot.types import Regime, Side

SYMBOL = "BTC/USDT:USDT"
CFG = load_config()


def make_window(n: int = 14, **last: float) -> pd.DataFrame:
    """Flat, NaN-free feature frame (price 100, ATR 1); ``last`` overrides the final row."""
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
            "atr": 1.0,
            "rv": 0.005,
            "ema_fast": 100.0,
            "volume_z": 2.0,
            "donchian_high": 110.0,
            "donchian_low": 90.0,
            "zscore": 0.0,
            "roll_mean": 100.0,
            "bars_since_compression": 3.0,
        },
        index=idx,
    )
    for col, val in last.items():
        df.loc[df.index[-1], col] = val
    return df


def mirror(df: pd.DataFrame, pivot: float = 200.0) -> pd.DataFrame:
    """Reflect prices about ``pivot``; high/low and channel bounds swap roles."""
    m = df.copy()
    for col in ("close", "ema_fast", "roll_mean"):
        m[col] = pivot - df[col]
    m["high"], m["low"] = pivot - df["low"], pivot - df["high"]
    m["donchian_high"], m["donchian_low"] = pivot - df["donchian_low"], pivot - df["donchian_high"]
    m["zscore"] = -df["zscore"]
    return m


def assert_geometry(sig, min_rr: float) -> None:
    sign = 1 if sig.side == Side.LONG else -1
    assert sign * (sig.stop - sig.entry) < 0
    assert sign * (sig.target - sig.entry) > 0
    rr = abs(sig.target - sig.entry) / abs(sig.entry - sig.stop)
    assert rr >= min_rr - 1e-9


# ---------- helpers ----------


def test_plausibility_boundary():
    limit = 1.5 * 0.01 * math.sqrt(100)
    assert target_is_plausible(limit, 0.01, 100, 1.5)
    assert not target_is_plausible(limit * 1.001, 0.01, 100, 1.5)


def test_build_signal_rejects_wrong_side_stop():
    w = make_window(close=100.0)
    kw = dict(name="x", symbol=SYMBOL, window=w, regime=Regime.TREND_UP, cfg=CFG.strategies.trend)
    assert build_signal(side=Side.LONG, stop=101.0, **kw) is None
    assert build_signal(side=Side.SHORT, stop=99.0, **kw) is None
    assert build_signal(side=Side.LONG, stop=100.0, **kw) is None


# ---------- breakout ----------


def breakout() -> Breakout:
    return Breakout(CFG.strategies.breakout)


def test_breakout_long_and_short_mirror():
    w = make_window(close=112.0)
    s = breakout().generate(SYMBOL, w, Regime.EXPANSION)
    assert s is not None and s.side == Side.LONG
    assert s.entry == 112.0 and s.stop == pytest.approx(112.0 - 1.5)
    assert s.target == pytest.approx(112.0 * 1.05)
    assert s.atr == 1.0 and s.regime == Regime.EXPANSION
    assert s.ts == w.index[-1].to_pydatetime() and s.ts.tzinfo is not None
    assert s.max_hold_bars == 288 and s.trail_atr_mult == 3.0
    assert_geometry(s, CFG.strategies.breakout.min_rr)

    # mirror about 200: entry 88, distances to stop/target preserved exactly
    ms = breakout().generate(SYMBOL, mirror(w), Regime.EXPANSION)
    assert ms is not None and ms.side == Side.SHORT
    assert ms.entry == pytest.approx(200 - s.entry)
    assert ms.stop - ms.entry == pytest.approx(s.entry - s.stop)
    assert ms.entry - ms.target == pytest.approx((s.target - s.entry) * 88 / 112)
    assert_geometry(ms, CFG.strategies.breakout.min_rr)


@pytest.mark.parametrize(
    "override",
    [
        {"close": 109.0},  # no breakout
        {"volume_z": 0.5},
        {"bars_since_compression": 25.0},
    ],
)
def test_breakout_each_condition_required(override):
    base = {"close": 112.0}
    assert (
        breakout().generate(SYMBOL, make_window(**{**base, **override}), Regime.EXPANSION) is None
    )


def test_breakout_boundaries_inclusive():
    w = make_window(close=112.0, volume_z=1.0, bars_since_compression=24.0)
    assert breakout().generate(SYMBOL, w, Regime.EXPANSION) is not None


def test_breakout_stop_too_wide_and_implausible():
    # atr 3 * 1.5 = 4.5 -> 4.0% > 3% max
    assert breakout().generate(SYMBOL, make_window(close=112.0, atr=3.0), Regime.EXPANSION) is None
    # 5% target needs rv >= ~0.00196
    assert breakout().generate(SYMBOL, make_window(close=112.0, rv=0.001), Regime.EXPANSION) is None


@pytest.mark.parametrize(
    "col",
    ["close", "atr", "donchian_high", "donchian_low", "volume_z", "bars_since_compression", "rv"],
)
def test_breakout_nan_inputs(col):
    w = make_window(**{"close": 112.0, col: float("nan")})
    assert breakout().generate(SYMBOL, w, Regime.EXPANSION) is None


def test_breakout_ignores_regime_and_short_window():
    assert breakout().generate(SYMBOL, make_window(close=112.0), Regime.NO_TRADE) is not None
    assert breakout().generate(SYMBOL, make_window(close=112.0).iloc[0:0], Regime.EXPANSION) is None


# ---------- trend ----------


def trend_window(**last: float) -> pd.DataFrame:
    """Long setup: bar -3 dips to ema_fast; current close breaks the previous high."""
    w = make_window(n=14, **{"close": 102.0, **last})
    w["low"], w["high"] = 100.6, 101.5  # lows just outside ema_fast + 0.5 * atr
    w.loc[w.index[-4], "low"] = 100.4  # the pullback bar; lowest low of the lookback
    return w


def trend() -> Trend:
    return Trend(CFG.strategies.trend)


def test_trend_long_and_short_mirror():
    w = trend_window()
    s = trend().generate(SYMBOL, w, Regime.TREND_UP)
    assert s is not None and s.side == Side.LONG
    assert s.entry == 102.0 and s.stop == pytest.approx(100.4 - 0.5)
    assert s.target == pytest.approx(102.0 * (1 + max(0.05, 2.5 * (102.0 - 99.9) / 102.0)))
    assert s.max_hold_bars == 384
    assert_geometry(s, CFG.strategies.trend.min_rr)

    ms = trend().generate(SYMBOL, mirror(w), Regime.TREND_DOWN)
    assert ms is not None and ms.side == Side.SHORT
    assert ms.stop - ms.entry == pytest.approx(s.entry - s.stop)
    assert ms.entry - ms.target == pytest.approx(s.target - s.entry)  # both rr-driven: 2.5 * 2.1
    assert_geometry(ms, CFG.strategies.trend.min_rr)


def test_trend_regime_must_match_side():
    w = trend_window()
    assert trend().generate(SYMBOL, w, Regime.TREND_DOWN) is None
    assert trend().generate(SYMBOL, mirror(w), Regime.TREND_UP) is None
    for r in (Regime.RANGE, Regime.COMPRESSION, Regime.EXPANSION, Regime.NO_TRADE):
        assert trend().generate(SYMBOL, w, r) is None


SIDES = [
    pytest.param(False, Regime.TREND_UP, id="long"),
    pytest.param(True, Regime.TREND_DOWN, id="short"),
]


def sided(w: pd.DataFrame, short: bool) -> pd.DataFrame:
    return mirror(w) if short else w


@pytest.mark.parametrize(("short", "regime"), SIDES)
def test_trend_each_condition_required(short, regime):
    # no pullback: every lookback low far above ema_fast + 0.5 atr
    w = trend_window()
    w["low"] = 100.9
    assert trend().generate(SYMBOL, sided(w, short), regime) is None
    # close not beyond the previous bar's extreme
    assert trend().generate(SYMBOL, sided(trend_window(close=100.5), short), regime) is None
    # volume filter
    assert trend().generate(SYMBOL, sided(trend_window(volume_z=-0.5), short), regime) is None


@pytest.mark.parametrize(("short", "regime"), SIDES)
def test_trend_pullback_uses_each_bars_own_ema(short, regime):
    w = trend_window()
    w["low"] = 101.0  # dip never reaches ema 100 + 0.5
    w.loc[w.index[-5], "ema_fast"] = 100.6  # that bar's own ema_fast brings it within range
    assert trend().generate(SYMBOL, sided(w, short), regime) is not None


@pytest.mark.parametrize(("short", "regime"), SIDES)
def test_trend_pullback_uses_that_bars_own_atr(short, regime):
    w = trend_window()
    w["low"] = 101.0
    assert trend().generate(SYMBOL, sided(w, short), regime) is None
    w.loc[w.index[-5], "atr"] = 2.0  # band 100 + 1.0 reaches 101; current-bar atr stays 1.0
    assert trend().generate(SYMBOL, sided(w, short), regime) is not None


@pytest.mark.parametrize(("short", "regime"), SIDES)
def test_trend_lookback_width_is_exactly_pullback_lookback(short, regime):
    def window(pull_at: int, extra_low_at: int | None = None) -> pd.DataFrame:
        w = trend_window()
        w["low"] = 100.9  # no pullback anywhere
        w.loc[w.index[pull_at], "low"] = 100.4
        if extra_low_at is not None:
            w.loc[w.index[extra_low_at], "low"] = 99.0
        return sided(w, short)

    oldest = trend().generate(SYMBOL, window(-13), regime)  # 12th bar back: inside
    assert oldest is not None
    assert trend().generate(SYMBOL, window(-14), regime) is None  # 13th bar back: outside
    # a lower extreme just outside the lookback does not move the stop
    with_outside = trend().generate(SYMBOL, window(-4, extra_low_at=-14), regime)
    inside_only = trend().generate(SYMBOL, window(-4), regime)
    assert with_outside is not None and inside_only is not None
    assert with_outside.stop == inside_only.stop
    # ... but at the edge of the lookback it does
    at_edge = trend().generate(SYMBOL, window(-4, extra_low_at=-13), regime)
    assert at_edge is None or at_edge.stop != inside_only.stop


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
@pytest.mark.parametrize("col", ["low", "high", "ema_fast", "atr"])
def test_trend_inf_in_lookback(col, bad):
    w = trend_window()
    w.loc[w.index[-6], col] = bad
    assert trend().generate(SYMBOL, w, Regime.TREND_UP) is None


def test_trend_current_bar_excluded_from_lookback():
    w = trend_window()
    w["low"] = 101.0
    w.loc[w.index[-1], "low"] = 100.0  # current bar dips, but only the prior bars count
    assert trend().generate(SYMBOL, w, Regime.TREND_UP) is None


def test_trend_short_window_returns_none():
    assert trend().generate(SYMBOL, trend_window().iloc[-12:], Regime.TREND_UP) is None
    assert trend().generate(SYMBOL, trend_window().iloc[-13:], Regime.TREND_UP) is not None


def test_trend_stop_too_wide_and_implausible():
    w = trend_window()
    w.loc[w.index[-6], "low"] = 96.0  # stop 95.5 -> 6.4% > 3%
    assert trend().generate(SYMBOL, w, Regime.TREND_UP) is None
    assert trend().generate(SYMBOL, trend_window(rv=0.001), Regime.TREND_UP) is None


@pytest.mark.parametrize("col", ["close", "atr", "volume_z", "rv"])
def test_trend_nan_current_row(col):
    assert trend().generate(SYMBOL, trend_window(**{col: float("nan")}), Regime.TREND_UP) is None


@pytest.mark.parametrize("col", ["low", "high", "ema_fast", "atr"])
def test_trend_nan_in_lookback(col):
    w = trend_window()
    w.loc[w.index[-6], col] = float("nan")
    assert trend().generate(SYMBOL, w, Regime.TREND_UP) is None


# ---------- mean reversion ----------


def mr() -> MeanReversion:
    return MeanReversion(CFG.strategies.mean_reversion)


def mr_window(**last: float) -> pd.DataFrame:
    return make_window(**{"close": 98.0, "zscore": -3.0, "roll_mean": 100.0, **last})


def test_mean_reversion_long_and_short_mirror():
    w = mr_window()
    s = mr().generate(SYMBOL, w, Regime.RANGE)
    assert s is not None and s.side == Side.LONG
    assert s.entry == 98.0 and s.stop == pytest.approx(97.0) and s.target == 100.0
    assert s.max_hold_bars == 48 and s.trail_atr_mult is None
    assert_geometry(s, CFG.strategies.mean_reversion.min_rr)

    ms = mr().generate(SYMBOL, mirror(w), Regime.RANGE)
    assert ms is not None and ms.side == Side.SHORT
    assert ms.entry == 102.0 and ms.stop == pytest.approx(103.0) and ms.target == 100.0
    assert_geometry(ms, CFG.strategies.mean_reversion.min_rr)


def test_mean_reversion_conditions_required():
    assert mr().generate(SYMBOL, mr_window(), Regime.TREND_UP) is None
    assert mr().generate(SYMBOL, mr_window(), Regime.EXPANSION) is None
    assert mr().generate(SYMBOL, mr_window(zscore=-2.4), Regime.RANGE) is None
    assert mr().generate(SYMBOL, mr_window(zscore=-2.5), Regime.RANGE) is not None


def test_mean_reversion_target_must_be_on_profit_side_with_min_rr():
    assert mr().generate(SYMBOL, mr_window(roll_mean=97.0), Regime.RANGE) is None  # below entry
    assert mr().generate(SYMBOL, mr_window(roll_mean=98.0), Regime.RANGE) is None  # at entry
    assert mr().generate(SYMBOL, mr_window(roll_mean=99.5), Regime.RANGE) is None  # rr 1.5 < 2
    assert mr().generate(SYMBOL, mr_window(roll_mean=100.0), Regime.RANGE) is not None  # rr 2.0
    short = mirror(mr_window(roll_mean=97.0))  # mean 103 above a short entry of 102
    assert mr().generate(SYMBOL, short, Regime.RANGE) is None


def test_mean_reversion_stop_too_wide():
    # ATR 1.0 -> stop 1.0 below 98 (1.02%) is ok; ATR 2.5 -> stop 2.5 (2.55%) exceeds the 2% max
    assert mr().generate(SYMBOL, mr_window(atr=2.5), Regime.RANGE) is None


def test_mean_reversion_skips_plausibility():
    # tiny rv would reject a 5% target but the mean target is exempt
    assert mr().generate(SYMBOL, mr_window(rv=0.0001), Regime.RANGE) is not None


@pytest.mark.parametrize("col", ["close", "atr", "zscore", "roll_mean"])
def test_mean_reversion_nan_inputs(col):
    assert mr().generate(SYMBOL, mr_window(**{col: float("nan")}), Regime.RANGE) is None


# ---------- registry ----------


def test_build_strategies_honours_enabled():
    assert [s.name for s in build_strategies(CFG)] == ["breakout", "trend", "mean_reversion"]
    cfg = load_config(
        overrides={"strategies.breakout.enabled": False, "strategies.trend.enabled": False}
    )
    assert [s.name for s in build_strategies(cfg)] == ["mean_reversion"]
