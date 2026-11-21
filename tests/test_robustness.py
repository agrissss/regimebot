"""Sensitivity, cost stress and leverage comparison, on a few days of synthetic 1m data."""

import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from synth import BTC, T0, Scripted, regime_walk

from regimebot.backtest.robustness import cost_stress, leverage_compare, sensitivity
from regimebot.backtest.runner import run_backtest
from regimebot.config import AppConfig, load_config
from regimebot.reporting.metrics import compute_metrics
from regimebot.types import Regime

END = datetime(2024, 1, 4, tzinfo=UTC)
CANDLES = {BTC: regime_walk(3 * 1440, seed=5)}


def open_cfg() -> AppConfig:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    return load_config(
        overrides={"exchange.symbols": [BTC], **{f"playbook.{r.value}": entry for r in Regime}}
    )


def stub() -> list[Scripted]:
    return [Scripted("breakout", stop_dist=0.3, target_dist=0.6)]


def test_sensitivity_shape_and_base_row_equals_plain_run() -> None:
    cfg = open_cfg()
    params = ["risk.risk_per_trade", "strategies.breakout.compression_lookback"]
    df = sensitivity(cfg, CANDLES, {}, T0, END, params, strategies=stub())
    assert list(df.columns) == [
        "param",
        "variant",
        "value",
        "net_pnl",
        "profit_factor",
        "n_trades",
        "note",
    ]
    assert list(df.variant) == ["-20%", "base", "+20%"] * 2
    assert list(df.param) == [p for p in params for _ in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        plain = run_backtest(cfg, CANDLES, {}, T0, END, Path(tmp) / "KILL", stub())
    m = compute_metrics(plain.trades, plain.equity, cfg.backtest.initial_equity)
    for _, r in df[df.variant == "base"].iterrows():
        assert (r.net_pnl, r.n_trades) == (m.net_pnl, m.n_trades)
    risk = df[df.param == "risk.risk_per_trade"].value.tolist()
    assert risk == pytest.approx([0.004, 0.005, 0.006])
    ints = df[df.param.str.endswith("compression_lookback")].value.tolist()
    base = cfg.strategies.breakout.compression_lookback
    assert ints == [max(1, round(base * 0.8)), base, round(base * 1.2)]


def test_sensitivity_invalid_value_is_a_nan_row_with_note_and_non_numeric_raises() -> None:
    # max_gross_leverage 1.5 * 1.2 = 1.8 exceeds the hard cap of 1.5
    cfg = load_config(overrides={"exchange.symbols": [BTC], "risk.max_gross_leverage": 1.5})
    df = sensitivity(cfg, CANDLES, {}, T0, END, ["risk.max_gross_leverage"], strategies=stub())
    bad = df[df.variant == "+20%"].iloc[0]
    assert math.isnan(bad.net_pnl) and "invalid" in bad.note
    assert df[df.variant != "+20%"].note.eq("").all()
    with pytest.raises(ValueError, match="not a numeric"):
        sensitivity(cfg, CANDLES, {}, T0, END, ["exchange.base_timeframe"], strategies=stub())


def test_cost_stress_never_increases_net_pnl_for_the_same_trades() -> None:
    df = cost_stress(open_cfg(), CANDLES, {}, T0, END, mult=2.0, strategies=stub())
    base, stressed = df.iloc[0], df.iloc[1]
    assert list(df.variant) == ["base", "stressed"]
    assert base.n_trades == stressed.n_trades > 0  # stub entries do not depend on costs
    assert stressed.net_pnl <= base.net_pnl
    assert stressed.fees > base.fees


def test_leverage_compare_returns_both_rows() -> None:
    df = leverage_compare(open_cfg(), CANDLES, {}, T0, END, strategies=stub())
    assert df.max_gross_leverage.tolist() == [1.0, 1.5]
    assert {"net_pnl", "max_drawdown", "n_trades"} <= set(df.columns)
    assert df.n_trades.gt(0).all() and df.max_drawdown.ge(0).all()
