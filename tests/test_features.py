import time

import numpy as np
import pandas as pd
import pytest

from regimebot.config import FeaturesConfig
from regimebot.market_state.features import compute_features

COMPRESSION = 0.3


def make_cfg(**kw: int) -> FeaturesConfig:
    base = dict(
        atr_window=3,
        rv_window=5,
        pctile_window=20,
        adx_window=3,
        ema_fast=3,
        ema_slow=6,
        volume_window=3,
        donchian_window=2,
        zscore_window=3,
    )
    return FeaturesConfig(**{**base, **kw})


def make_candles(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    volume: list[float] | None = None,
) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": c.shift(1).fillna(c),
            "high": high if high is not None else c + 0.5,
            "low": low if low is not None else c - 0.5,
            "close": c,
            "volume": volume if volume is not None else [1.0] * n,
        },
        index=idx,
    )


def random_walk(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    spread = np.abs(rng.normal(0, 0.002, (2, n)))
    vol = rng.lognormal(3, 0.5, n)
    vol[rng.random(n) < 0.01] = 0.0  # a few zero-volume bars
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    c = pd.Series(close, index=idx)
    return pd.DataFrame(
        {
            "open": c.shift(1).fillna(c),
            "high": c * (1 + spread[0]),
            "low": c * (1 - spread[1]),
            "close": c,
            "volume": vol,
        },
        index=idx,
    )


def test_prefix_invariance_every_column() -> None:
    cfg = make_cfg(
        atr_window=14, rv_window=20, pctile_window=200, adx_window=14,
        ema_fast=10, ema_slow=30, volume_window=20, donchian_window=20, zscore_window=20,
    )  # fmt: skip
    candles = random_walk(4000)
    # Funding on and off the candle grid, including stamps after every cut point.
    stamps = candles.index[::32].append(candles.index[5::97] + pd.Timedelta(minutes=7))
    funding = pd.Series(np.random.default_rng(1).normal(0, 1e-4, len(stamps)), index=stamps)
    full = compute_features(candles, cfg, COMPRESSION, funding)
    assert full["funding"].notna().any()
    assert (full["bars_since_compression"] == 0).any()
    assert full["bars_since_compression"].isna().any()
    for k in (1, 5, 15, 30, 250, 1001, 2500, 3999):
        known = funding[funding.index <= candles.index[k - 1]]  # only what exists at the cut
        prefix = compute_features(candles.iloc[:k], cfg, COMPRESSION, known)
        pd.testing.assert_frame_equal(
            prefix, full.iloc[:k], check_exact=False, rtol=1e-9, atol=0, check_freq=False
        )


def test_atr_known_values_wilder_seed() -> None:
    # TRs (row 1..3): 2, 3, 1. Seed = first TR (2); alpha = 1/3; first output at row 3.
    c = make_candles([9, 10, 12, 11.5], high=[10, 11, 13, 12], low=[8, 9, 10, 11])
    atr = compute_features(c, make_cfg(), COMPRESSION)["atr"]
    assert atr.iloc[:3].isna().all()
    assert atr.iloc[3] == pytest.approx((2 + (3 - 2) / 3) * 2 / 3 + 1 / 3)


def test_ema_known_values() -> None:
    c = make_candles([9, 10, 12, 11.5])
    out = compute_features(c, make_cfg(ema_fast=3, ema_slow=4), COMPRESSION)
    # span 3 -> alpha 0.5, seeded with first close
    assert out["ema_fast"].iloc[:2].isna().all()
    assert out["ema_fast"].iloc[2] == pytest.approx(10.75)
    assert out["ema_fast"].iloc[3] == pytest.approx(11.125)


def test_donchian_excludes_current_bar() -> None:
    c = make_candles([9, 10, 12, 11.5], high=[10, 11, 13, 12], low=[8, 9, 10, 11])
    out = compute_features(c, make_cfg(donchian_window=2), COMPRESSION)
    assert out["donchian_high"].iloc[:2].isna().all()
    assert out["donchian_high"].iloc[2] == 11  # not 13 (current bar's high)
    assert out["donchian_high"].iloc[3] == 13
    assert out["donchian_low"].iloc[2] == 8
    assert out["donchian_low"].iloc[3] == 9


def test_volume_z_excludes_current_bar() -> None:
    c = make_candles([10.0] * 5, volume=[1, 2, 3, 4, 10])
    z = compute_features(c, make_cfg(volume_window=3), COMPRESSION)["volume_z"]
    assert z.iloc[:3].isna().all()
    assert z.iloc[3] == pytest.approx((4 - 2) / 1)  # baseline 1,2,3
    assert z.iloc[4] == pytest.approx((10 - 3) / 1)  # baseline 2,3,4


def test_vol_pctile_endpoints() -> None:
    n = 60
    mag = 0.001 * 1.1 ** np.arange(n)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    up = 100 * np.exp(np.cumsum(mag * sign))
    down = 100 * np.exp(np.cumsum(mag[::-1] * sign))
    cfg = make_cfg()
    rising = compute_features(make_candles(list(up)), cfg, COMPRESSION)["vol_pctile"]
    falling = compute_features(make_candles(list(down)), cfg, COMPRESSION)["vol_pctile"]
    assert rising.iloc[-1] == 1.0  # current maximum
    assert falling.iloc[-1] == pytest.approx(1 / cfg.pctile_window)  # current minimum
    # first rv at row rv_window; vol_pctile needs rv_window valid rv values
    assert rising.first_valid_index() == rising.index[2 * cfg.rv_window - 1]
    assert rising.iloc[: 2 * cfg.rv_window - 1].isna().all()


def test_adx_high_on_trend_low_on_noise() -> None:
    cfg = make_cfg(adx_window=5)
    trend = compute_features(make_candles([100.0 + i for i in range(60)]), cfg, COMPRESSION)
    noise = compute_features(make_candles([100.0 + (i % 2) for i in range(60)]), cfg, COMPRESSION)
    assert trend["adx"].iloc[-1] > 90
    assert trend["plus_di"].iloc[-1] > trend["minus_di"].iloc[-1]
    assert noise["adx"].iloc[-1] < 20


def test_warmup_nan_never_filled() -> None:
    cfg = make_cfg(atr_window=14, adx_window=14, ema_fast=10, ema_slow=30, zscore_window=20)
    out = compute_features(random_walk(200), cfg, COMPRESSION)
    assert out["atr"].iloc[:14].isna().all() and out["atr"].iloc[14:].notna().all()
    assert out["ema_slow"].iloc[:29].isna().all() and out["ema_slow"].iloc[29:].notna().all()
    assert out["adx"].iloc[:27].isna().all() and out["adx"].iloc[27:].notna().all()
    assert out["roll_mean"].iloc[:19].isna().all()
    assert np.isnan(out["ret"].iloc[0])
    assert out["funding"].isna().all()  # none supplied


def test_zero_volume_gives_nan_not_inf() -> None:
    c = random_walk(200)
    c.iloc[150, c.columns.get_loc("volume")] = 0.0
    out = compute_features(c, make_cfg(volume_window=5), COMPRESSION)
    assert np.isnan(out["illiq_pctile"].iloc[150])
    assert not np.isinf(out.to_numpy(dtype=float)).any()


def test_illiq_pctile_first_valid_index() -> None:
    cfg = make_cfg()
    out = compute_features(random_walk(100, seed=3).assign(volume=5.0), cfg, COMPRESSION)
    # Amihud is valid from row 1 (first ret); needs rv_window valid values
    assert out["illiq_pctile"].first_valid_index() == out.index[cfg.rv_window]


def test_ret_z_uses_previous_rv() -> None:
    out = compute_features(random_walk(100), make_cfg(), COMPRESSION)
    expected = out["ret"] / out["rv"].shift(1)
    pd.testing.assert_series_equal(out["ret_z"], expected, check_names=False)


def test_funding_asof_never_uses_future_stamp() -> None:
    c = make_candles([100.0] * 6)
    t = c.index
    funding = pd.Series([0.1, 0.2, 0.3], index=[t[2], t[4] + pd.Timedelta(minutes=1), t[5]])
    f = compute_features(c, make_cfg(), COMPRESSION, funding)["funding"]
    assert f.iloc[:2].isna().all()
    assert f.iloc[2:5].tolist() == [0.1, 0.1, 0.1]  # t[4]+1min stamp not yet visible at t[4]
    assert f.iloc[5] == 0.3  # at t[5] the 0.2 stamp (t[4]+1min) is superseded by 0.3


def test_bars_since_compression() -> None:
    out = compute_features(random_walk(500), make_cfg(), COMPRESSION)
    comp = (out["vol_pctile"] <= COMPRESSION).to_numpy()
    first = int(np.argmax(comp))
    assert out["bars_since_compression"].iloc[:first].isna().all()
    last = None
    for i, flag in enumerate(comp):  # test-only loop: independent reference implementation
        if flag:
            last = i
        if last is not None:
            assert out["bars_since_compression"].iloc[i] == i - last


@pytest.mark.parametrize("bad", ["naive", "unsorted", "columns"])
def test_input_validation(bad: str) -> None:
    c = random_walk(50)
    if bad == "naive":
        c.index = c.index.tz_localize(None)
    elif bad == "unsorted":
        c = c.iloc[::-1]
    else:
        c = c.drop(columns="volume")
    with pytest.raises(ValueError):
        compute_features(c, make_cfg(), COMPRESSION)


@pytest.mark.parametrize("tz", [None, "Europe/Riga"])
def test_non_utc_funding_rejected(tz: str | None) -> None:
    f = pd.Series([0.1], index=pd.DatetimeIndex(["2024-01-01"], tz=tz))
    with pytest.raises(ValueError, match="UTC"):
        compute_features(random_walk(50), make_cfg(), COMPRESSION, f)


@pytest.mark.parametrize(
    "empty",
    [
        pd.Series(dtype=float),
        pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")),
    ],
)
def test_empty_funding_behaves_like_none(empty: pd.Series) -> None:
    c = random_walk(50)
    out = compute_features(c, make_cfg(), COMPRESSION, empty)
    assert out["funding"].isna().all()
    assert out["funding"].index.equals(c.index)


def test_duplicate_funding_stamps_rejected() -> None:
    c = random_walk(50)
    f = pd.Series([0.1, 0.2], index=[c.index[3], c.index[3]])
    with pytest.raises(ValueError, match="duplicate"):
        compute_features(c, make_cfg(), COMPRESSION, f)


def test_performance_70k_rows() -> None:
    cfg = make_cfg(
        atr_window=14, rv_window=96, pctile_window=2880, adx_window=14,
        ema_fast=96, ema_slow=384, volume_window=96, donchian_window=96, zscore_window=96,
    )  # fmt: skip
    c = random_walk(70_000)
    start = time.perf_counter()
    compute_features(c, cfg, COMPRESSION)
    assert time.perf_counter() - start < 5.0
