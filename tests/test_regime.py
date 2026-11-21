import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from regimebot.config import EventWindow, RegimeConfig, load_config
from regimebot.market_state.regime import REQUIRED_FEATURES, classify, classify_frame, playbook_for
from regimebot.market_state.study import regime_study
from regimebot.types import Regime

TS = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
WINDOW = EventWindow(
    start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    end=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
    label="fomc",
)


def cfg(windows: list[EventWindow] | None = None) -> RegimeConfig:
    return RegimeConfig(
        expansion_pctile=0.9,
        compression_pctile=0.2,
        adx_trend=25.0,
        adx_range=20.0,
        shock_z=6.0,
        illiq_pctile=0.99,
        max_spread_bps=10.0,
        event_windows=windows or [],
    )


def row(**kw: float) -> dict[str, float]:
    """A neutral row (mid vol, adx between range and trend): NO_TRADE by rule 8."""
    base = dict(
        close=100.0, atr_pct=0.01, vol_pctile=0.5, adx=22.0, ema_fast=100.0, ema_slow=100.0,
        volume_z=0.0, ret_z=0.0, illiq_pctile=0.5,
    )  # fmt: skip
    return {**base, **kw}


def outside() -> datetime:
    return datetime(2024, 2, 1, tzinfo=UTC)


UP = dict(adx=30.0, ema_fast=101.0, ema_slow=100.0, close=102.0)
DOWN = dict(adx=30.0, ema_fast=99.0, ema_slow=100.0, close=98.0)

CASES = [
    # (row overrides, ts, spread, expected)
    ({**UP, "adx": math.nan, "ret_z": 9.0}, TS, 99.0, Regime.NO_TRADE),  # NaN beats everything
    ({"ret_z": 9.0, "illiq_pctile": 1.0}, TS, None, Regime.EVENT_RISK),
    ({"illiq_pctile": 1.0}, TS, None, Regime.EVENT_RISK),  # event window beats liquidity stress
    ({"vol_pctile": 0.95, **UP}, outside(), None, Regime.EXPANSION),
    ({"ret_z": -6.0, "vol_pctile": 0.95}, outside(), None, Regime.EVENT_RISK),  # shock, equality
    ({"ret_z": 5.99}, outside(), None, Regime.NO_TRADE),
    ({"illiq_pctile": 0.99, "vol_pctile": 0.95}, outside(), None, Regime.LIQUIDITY_STRESS),
    ({"vol_pctile": 0.95}, outside(), 10.5, Regime.LIQUIDITY_STRESS),  # spread > max
    ({"vol_pctile": 0.95}, outside(), 10.0, Regime.EXPANSION),  # spread == max is fine
    ({"vol_pctile": 0.9}, outside(), None, Regime.EXPANSION),  # equality
    ({**UP, "vol_pctile": 0.1}, outside(), None, Regime.TREND_UP),  # trend beats compression
    ({**DOWN, "vol_pctile": 0.1}, outside(), None, Regime.TREND_DOWN),
    ({**UP, "adx": 25.0}, outside(), None, Regime.TREND_UP),  # equality
    ({**UP, "adx": 24.9}, outside(), None, Regime.NO_TRADE),
    ({**UP, "close": 101.0}, outside(), None, Regime.NO_TRADE),  # close == ema_fast: not trend
    ({"vol_pctile": 0.2, "adx": 10.0}, outside(), None, Regime.COMPRESSION),  # beats range
    ({"vol_pctile": 0.21, "adx": 20.0}, outside(), None, Regime.RANGE),  # equality
    ({"adx": 20.1}, outside(), None, Regime.NO_TRADE),
]


def one_row_frame(r: dict[str, float], ts: datetime) -> pd.DataFrame:
    return pd.DataFrame([r], index=pd.DatetimeIndex([ts], tz="UTC"))


@pytest.mark.parametrize(("over", "ts", "spread", "expected"), CASES)
def test_priority_and_thresholds(over, ts, spread, expected):
    assert classify(row(**over), cfg([WINDOW]), ts, spread) is expected
    if spread is None:  # classify_frame takes no spread
        frame = classify_frame(one_row_frame(row(**over), ts), cfg([WINDOW]))
        assert list(frame) == [expected.value]


def test_event_window_is_half_open():
    c = cfg([WINDOW])
    assert classify(row(), c, WINDOW.start) is Regime.EVENT_RISK
    assert classify(row(), c, WINDOW.end) is Regime.NO_TRADE
    assert list(classify_frame(one_row_frame(row(), WINDOW.start), c)) == ["event_risk"]
    assert list(classify_frame(one_row_frame(row(), WINDOW.end), c)) == ["no_trade"]


NAMES = [
    "atr_pct",
    "vol_pctile",
    "adx",
    "ema_fast",
    "ema_slow",
    "volume_z",
    "ret_z",
    "illiq_pctile",
]


def test_required_feature_list_is_pinned():
    assert list(REQUIRED_FEATURES) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_each_required_feature_nan_gives_no_trade(name):
    r = row(**{name: math.nan})
    assert classify(r, cfg(), outside()) is Regime.NO_TRADE
    assert list(classify_frame(one_row_frame(r, outside()), cfg())) == ["no_trade"]


def test_classify_frame_matches_classify_row_by_row():
    rng = np.random.default_rng(7)
    n = 3000
    idx = pd.date_range("2024-01-01 10:00", periods=n, freq="15min", tz="UTC")
    grid = [99.0, 100.0, 101.0]  # coarse so close/ema comparisons tie
    f = pd.DataFrame(
        {
            "close": rng.choice(grid, n),
            "atr_pct": rng.uniform(0, 0.02, n),
            "vol_pctile": rng.choice([0.0, 0.2, 0.5, 0.9, 1.0], n),
            "adx": rng.choice([10.0, 20.0, 22.0, 25.0, 40.0], n),
            "ema_fast": rng.choice(grid, n),
            "ema_slow": rng.choice(grid, n),
            "volume_z": rng.normal(0, 1, n),
            "ret_z": rng.choice([0.0, 1.0, 6.0, -6.0, 7.0], n),
            "illiq_pctile": rng.choice([0.1, 0.99, 1.0], n),
        },
        index=idx,
    )
    for col in REQUIRED_FEATURES:  # independent NaN masks: single NaN beside valid features
        f.loc[rng.random(n) < 0.03, col] = np.nan
    # window boundary rows with otherwise quiet features (no shock, no NaN, no illiquidity)
    quiet = row()
    for ts in (WINDOW.start, WINDOW.end, WINDOW.start - pd.Timedelta("15min")):
        f.loc[ts] = quiet
    f = f.sort_index()
    c = cfg([WINDOW])
    got = classify_frame(f, c)
    want = [classify(r, c, ts) for ts, r in zip(f.index, f.to_dict("records"), strict=True)]
    assert list(got) == [w.value for w in want]
    assert got.index.equals(f.index)
    assert set(got) == {r.value for r in Regime}


def test_playbook_for_returns_configured_entry():
    app_cfg = load_config()
    for regime in Regime:
        assert playbook_for(regime, app_cfg) == app_cfg.playbook[regime]


def test_regime_study_hand_built_series():
    idx = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
    close = [100.0, 110.0, 100.0, 100.0, 100.0, 90.0]
    features = pd.DataFrame({"close": close}, index=idx)
    a, b = Regime.RANGE.value, Regime.EXPANSION.value
    regimes = pd.Series([a, a, b, b, a, a], index=idx)
    before = features.copy()
    out = regime_study(features, regimes, horizons=(1, 2))

    assert list(features.columns) == ["close"]
    pd.testing.assert_frame_equal(features, before)
    assert len(out) == len(Regime)
    r = out.loc[a]
    assert r["share"] == pytest.approx(4 / 6)
    assert r["episodes"] == 2
    assert out.loc[b, "episodes"] == 1
    assert out.loc["trend_up", "episodes"] == 0
    # horizon 1, regime a at rows 0,1,4 (row 5 has no forward bar): log(1.1), log(100/110), log(.9)
    rets = np.log([110 / 100, 100 / 110, 90 / 100])
    assert r["n_1"] == 3
    assert r["mean_1"] == pytest.approx(rets.mean())
    assert r["std_1"] == pytest.approx(rets.std(ddof=1))
    assert r["p05_1"] == pytest.approx(np.quantile(rets, 0.05))
    assert r["p95_1"] == pytest.approx(np.quantile(rets, 0.95))
    assert r["big_move_1"] == 1.0  # +10%, -9.1%, -10%
    # horizon 2, regime a rows 0,1 only (rows 4,5 lack a full window)
    assert r["n_2"] == 2
    assert r["big_move_2"] == 1.0  # row 0 hits +10%, row 1 hits -9.09%
    # regime b horizon 2: rows 2,3 -> max excursion 0% and 10% (close 90 at t+2 for row 3 is -10%)
    assert out.loc[b, "n_2"] == 2
    assert out.loc[b, "big_move_2"] == pytest.approx(0.5)


def test_regimes_study_cli_writes_csv(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from regimebot.cli import app
    from regimebot.db.repo import upsert_candles
    from regimebot.db.session import make_engine, session_scope, upgrade_to_head

    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.chdir(tmp_path)
    upgrade_to_head(url)
    cfg_ = load_config()
    symbol = cfg_.exchange.symbols[0]
    n = 3 * 24 * 60
    idx = pd.date_range("2024-01-01 00:01", periods=n, freq="1min", tz="UTC", name="ts")
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(3).normal(0, 0.05, n)), index=idx)
    candles = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 5.0}
    )
    with session_scope(make_engine(url)) as session:
        upsert_candles(session, symbol, cfg_.exchange.base_timeframe, candles)

    result = CliRunner().invoke(app, ["regimes", "study"])
    assert result.exit_code == 0, result.output
    assert (
        tmp_path / "reports" / f"regime_study_{symbol.replace('/', '_').replace(':', '_')}.csv"
    ).exists()
