"""Daily trend on the money path: trade aggregation in the portfolio, the risk caps of a
rebalance, the engine's target-position step (band, close, flip, dust, expiry), halts, the
loss-streak exemption, and the backtest reproducing a direct vectorised computation of the rule.

Unless a test says otherwise costs are zero (``FREE``), so every fill is at the quoted price and
fees are 0; ``config/trend.yaml`` gives 10 000 equity, caps 0.75 asset / 1.0 group / 1.5 gross,
band 0.01 x equity, min notional 10.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth import BTC, ETH, MINUTE, SOL, T0, candles_from_closes

from regimebot.backtest.runner import run_backtest
from regimebot.config import REPO_ROOT, AppConfig, load_config
from regimebot.engine.loop import Engine, RunLog
from regimebot.engine.portfolio import Portfolio, Position
from regimebot.risk.manager import RiskManager
from regimebot.types import Bar, ExitReason, Fill, OrderSide, Regime, Side

TREND_YAML = REPO_ROOT / "config" / "trend.yaml"
DAY = timedelta(days=1)
D1, D2, D3, D4, D5, D6 = (T0 + k * DAY for k in range(1, 7))
FREE = {
    "costs.taker_fee_bps": 0.0,
    "costs.maker_fee_bps": 0.0,
    "costs.half_spread_bps": 0.0,
    "costs.slippage_bps": 0.0,
    "costs.impact_bps_per_pct": 0.0,
}
BUY, SELL = OrderSide.BUY, OrderSide.SELL


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-12, abs=1e-9)


def trend_cfg(**overrides: object) -> AppConfig:
    return load_config(TREND_YAML, overrides=overrides)


def fill(side: OrderSide, qty: float, price: float, fee: float = 0.0, minute: int = 1) -> Fill:
    return Fill(BTC, T0 + minute * MINUTE, side, qty, price, price, fee, 0.0, "taker")


# --- portfolio: one trade from flat to flat --------------------------------------------------


def test_a_trade_from_flat_to_flat_aggregates_its_fills_fees_and_funding() -> None:
    p = Portfolio(10_000.0)
    assert p.rebalance(fill(BUY, 10.0, 100.0, 0.5, 1), "daily_trend", ExitReason.SIGNAL) is None
    p.apply_funding(BTC, 105.0, 0.001)  # long pays 10 x 105 x 0.001 = 1.05
    p.rebalance(fill(BUY, 5.0, 110.0, 0.25, 2), "daily_trend", ExitReason.SIGNAL)
    position = p.positions[BTC]
    assert (position.qty, position.entry_price) == (15.0, approx(1550.0 / 15))  # average cost
    p.rebalance(fill(SELL, 6.0, 120.0, 0.36, 3), "daily_trend", ExitReason.SIGNAL)
    # partial exit: 6 x (120 - 103.33) = +100 realised, at the average cost
    assert (position.qty, position.realised) == (9.0, approx(100.0))
    assert p.cash == approx(10_000.0 - 0.5 - 1.05 - 0.25 + 100.0 - 0.36)
    p.apply_funding(BTC, 120.0, -0.002)  # long receives 9 x 120 x 0.002 = 2.16
    trade = p.rebalance(fill(SELL, 9.0, 90.0, 0.405, 4), "daily_trend", ExitReason.SIGNAL)
    assert trade is not None and p.positions == {}
    # entries 10 @ 100 + 5 @ 110 = 1550 over 15; exits 6 @ 120 + 9 @ 90 = 1530 over 15
    assert (trade.qty, trade.entry_price, trade.exit_price) == (
        15.0,
        approx(1550 / 15),
        approx(102),
    )
    assert trade.gross_pnl == approx(-20.0)  # 15 x (102 - 103.33) = 100 - 120
    assert trade.fees == approx(0.5 + 0.25 + 0.36 + 0.405)
    assert trade.funding == approx(1.05 - 2.16)
    assert trade.net_pnl == approx(-20.0 - 1.515 + 1.11)
    assert p.cash - 10_000.0 == approx(trade.net_pnl)  # the portfolio invariant holds
    assert (trade.side, trade.strategy, trade.exit_reason) == (Side.LONG, "daily_trend", "signal")
    assert (trade.entry_ts, trade.exit_ts) == (T0 + MINUTE, T0 + 4 * MINUTE)
    # no stop, target or regime: not zeros that could pass for real values
    assert math.isnan(trade.initial_stop) and math.isnan(trade.target)
    assert math.isnan(trade.r_multiple) and trade.regime is None


def test_a_flip_is_one_fill_that_closes_the_trade_and_opens_the_other_side() -> None:
    p = Portfolio(10_000.0)
    p.rebalance(fill(BUY, 10.0, 100.0, 0.5, 1), "daily_trend", ExitReason.SIGNAL)
    trade = p.rebalance(fill(SELL, 25.0, 110.0, 1.25, 2), "daily_trend", ExitReason.SIGNAL)
    assert trade is not None
    # the closing 10 of 25 carry 10/25 of the fee: 0.5
    assert (trade.qty, trade.gross_pnl, trade.fees, trade.net_pnl) == (10.0, 100.0, 1.0, 99.0)
    short = p.positions[BTC]
    assert (short.side, short.qty, short.entry_price, short.fees) == (Side.SHORT, 15.0, 110.0, 0.75)
    assert short.entry_ts == T0 + 2 * MINUTE
    assert p.cash == approx(10_000.0 - 0.5 + 100.0 - 0.5 - 0.75)
    # a short's partial exit realises at its cost: 5 x (110 - 100) = +50
    p.rebalance(fill(BUY, 5.0, 100.0, 0.0, 3), "daily_trend", ExitReason.SIGNAL)
    assert (short.qty, short.realised) == (10.0, approx(50.0))


def test_a_target_position_state_round_trips_and_an_older_state_still_loads() -> None:
    p = Portfolio(10_000.0)
    p.rebalance(fill(BUY, 10.0, 100.0), "daily_trend", ExitReason.SIGNAL)
    p.rebalance(fill(SELL, 4.0, 120.0, minute=2), "daily_trend", ExitReason.SIGNAL)
    position = p.positions[BTC]
    text = json.dumps(position.to_state(), sort_keys=True)  # NaN stop and target as NaN
    restored = Position.from_state(json.loads(text))
    assert json.dumps(restored.to_state(), sort_keys=True) == text
    assert restored.regime is None and math.isnan(restored.stop) and restored.realised == 80.0
    signal_position = Position(
        symbol=BTC,
        strategy="breakout",
        side=Side.LONG,
        regime=Regime.RANGE,
        qty=1.0,
        entry_price=100.0,
        entry_ts=T0,
        stop=95.0,
        initial_stop=95.0,
        target=110.0,
        atr_at_entry=2.0,
        max_hold_bars=48,
        trail_atr_mult=None,
        best_price=100.0,
    )
    old = signal_position.to_state()  # as saved before the target-position keys existed
    for key in ("entry_qty", "entry_notional", "realised"):
        del old[key]
    assert Position.from_state(old) == signal_position
    with pytest.raises(ValueError, match="missing keys"):
        Position.from_state({k: v for k, v in old.items() if k != "fees"})


# --- risk: caps on a rebalance ---------------------------------------------------------------


def risk_and_book(
    tmp_path: Path, cfg: AppConfig, btc_qty: float = 0.0
) -> tuple[RiskManager, Portfolio]:
    p = Portfolio(cfg.backtest.initial_equity)
    if btc_qty:
        p.rebalance(fill(BUY, btc_qty, 100.0), "daily_trend", ExitReason.SIGNAL)
    return RiskManager(cfg, tmp_path / "KILL"), p


def test_a_target_above_the_asset_cap_is_cut_to_it(tmp_path: Path) -> None:
    rm, p = risk_and_book(tmp_path, trend_cfg())
    decision, approved = rm.evaluate_target(BTC, 0.0, 100.0, 100.0, p, D1)  # 1.0 x equity
    assert (decision.approved, decision.rule, decision.qty, approved) == (
        True,
        "cap_asset",
        75.0,
        75.0,
    )
    decision, approved = rm.evaluate_target(BTC, 0.0, -100.0, 100.0, p, D1)
    assert (decision.rule, approved) == ("cap_asset", -75.0)
    assert rm.evaluate_target(BTC, 0.0, 50.0, 100.0, p, D1)[0].rule == "ok"


def test_the_group_cap_counts_the_other_members_but_not_the_symbol_itself(tmp_path: Path) -> None:
    rm, p = risk_and_book(tmp_path, trend_cfg(), btc_qty=70.0)  # BTC 7 000 of the 1.0 x 10 000
    decision, approved = rm.evaluate_target(ETH, 0.0, 50.0, 100.0, p, D1)
    assert (decision.rule, approved) == ("cap_group", approx(30.0))
    decision, approved = rm.evaluate_target(BTC, 70.0, 72.0, 100.0, p, D1)  # BTC's own 7 000
    assert (decision.rule, decision.qty, approved) == ("ok", approx(2.0), 72.0)


def test_pending_orders_count_toward_the_gross_cap(tmp_path: Path) -> None:
    rm, p = risk_and_book(tmp_path, trend_cfg(**{"risk.groups": {}}), btc_qty=70.0)
    decision, approved = rm.evaluate_target(SOL, 0.0, 50.0, 100.0, p, D1, {ETH: 5_000.0})
    assert (decision.rule, approved) == ("cap_gross", approx(30.0))  # 15 000 - 7 000 - 5 000


def test_a_close_is_never_cut_and_an_order_under_min_notional_is_too_small(
    tmp_path: Path,
) -> None:
    rm, p = risk_and_book(tmp_path, trend_cfg(**{"risk.max_group_exposure": 0.5}), btc_qty=70.0)
    assert rm.evaluate_target(BTC, 70.0, 0.0, 100.0, p, D1)[0].rule == "ok"  # group overfull
    assert rm.evaluate_target(BTC, 70.0, 60.0, 100.0, p, D1)[1] == approx(50.0)  # cut further
    rm, p = risk_and_book(tmp_path, trend_cfg(), btc_qty=70.0)
    decision, approved = rm.evaluate_target(BTC, 70.0, 70.05, 100.0, p, D1)  # order 5 < 10
    assert (decision.approved, decision.rule, approved) == (False, "too_small", 70.0)


def test_the_kill_switch_blocks_a_rebalance_but_the_loss_streak_rules_do_not(
    tmp_path: Path,
) -> None:
    rm, p = risk_and_book(tmp_path, trend_cfg())
    rm.loss_streak, rm.day_stopped = 9, D1.date()  # would stop and halve a playbook entry
    decision, approved = rm.evaluate_target(BTC, 0.0, 50.0, 100.0, p, D1)
    assert (decision.approved, decision.qty, approved) == (True, 50.0, 50.0)
    (tmp_path / "KILL").touch()
    decision, approved = rm.evaluate_target(BTC, 0.0, 50.0, 100.0, p, D1)
    assert (decision.rule, approved) == ("kill_switch", 0.0)


# --- engine: the target-position step --------------------------------------------------------


@dataclass
class Fixed:
    """Stub ``TargetStrategy``: the weights scripted per decision time (none: no view)."""

    by_ts: dict[datetime, dict[str, float]]
    name: str = "daily_trend"
    seen: list[datetime] = field(default_factory=list)

    def weights(self, ts: datetime, rows: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        self.seen.append(ts)
        return {s: w for s, w in self.by_ts.get(ts, {}).items() if s in rows}


def make(tmp_path: Path, weights: dict[datetime, dict[str, float]], **overrides: object) -> Engine:
    cfg = trend_cfg(**{"exchange.symbols": [BTC], **FREE, **overrides})
    return Engine(
        cfg,
        [Fixed(weights)],
        RiskManager(cfg, tmp_path / "KILL"),
        Portfolio(cfg.backtest.initial_equity),
        RunLog(),
    )


def decide(engine: Engine, ts: datetime, close: float) -> None:
    row = {"close": close, "signal": 1.0, "vol": 0.5}  # read by DailyTrend, not by the stub
    engine.on_decision(ts, {BTC: pd.DataFrame([row], index=pd.DatetimeIndex([ts]))})


def fill_at(engine: Engine, ts: datetime, price: float, volume: float = 1e9) -> None:
    engine.fill_queued(BTC, ts + MINUTE, price, price, volume)


def decisions(engine: Engine) -> list[tuple[datetime, str, float]]:
    return [(ts, d.rule, d.qty) for i, ts, d in engine.log.decisions if i is None]


def test_the_engine_trades_toward_the_target_only_past_the_band(tmp_path: Path) -> None:
    w = {D1: {BTC: 0.5}, D2: {BTC: 0.505}, D3: {BTC: 0.52}}
    engine = make(tmp_path, w)
    decide(engine, D1, 100.0)  # target 0.5 x 10 000 / 100 = 50
    fill_at(engine, D1, 100.0)
    decide(engine, D2, 100.0)  # 50.5: a 50 change is under the 100 band, no order
    decide(engine, D3, 100.0)  # 52: a 200 change
    fill_at(engine, D3, 100.0)
    assert decisions(engine) == [(D1, "ok", 50.0), (D3, "ok", approx(2.0))]
    assert [(f.side, f.qty) for f in engine.log.fills] == [(BUY, 50.0), (BUY, approx(2.0))]
    assert engine.portfolio.positions[BTC].qty == approx(52.0) and engine.log.trades == []


def test_a_sign_change_is_one_order_through_zero_and_a_zero_target_closes(
    tmp_path: Path,
) -> None:
    w = {D1: {BTC: 0.5}, D2: {BTC: -0.3}, D3: {BTC: -0.004}, D4: {BTC: 0.0}}
    engine = make(tmp_path, w)
    decide(engine, D1, 100.0)
    fill_at(engine, D1, 100.0)  # long 50
    decide(engine, D2, 110.0)  # equity 10 500: target -0.3 x 10 500 / 110 = -28.64
    fill_at(engine, D2, 110.0)
    (first,) = engine.log.trades
    assert (first.side, first.qty, first.gross_pnl) == (Side.LONG, 50.0, approx(500.0))
    flip = engine.log.fills[-1]
    assert (flip.side, flip.qty) == (SELL, approx(50 + 0.3 * 10_500 / 110))
    short = engine.portfolio.positions[BTC]
    assert (short.side, short.qty) == (Side.SHORT, approx(0.3 * 10_500 / 110))
    decide(engine, D3, 110.0)  # -0.004: a 2864 -> 42 change, past the band
    fill_at(engine, D3, 110.0)
    assert engine.portfolio.positions[BTC].qty == approx(0.004 * 10_500 / 110)  # 0.38, 42 notional
    decide(engine, D4, 110.0)  # zero target: the 42 position closes although under the band
    fill_at(engine, D4, 110.0)
    assert engine.portfolio.positions == {}
    assert [t.side for t in engine.log.trades] == [Side.LONG, Side.SHORT]
    assert engine.portfolio.cash - 10_000.0 == approx(sum(t.net_pnl for t in engine.log.trades))


def test_no_order_is_under_min_notional_so_a_position_under_it_stays_as_dust(
    tmp_path: Path,
) -> None:
    w = {D1: {BTC: 0.0005}, D2: {BTC: 0.5}, D3: {BTC: 0.0008}, D4: {BTC: 0.0}}
    engine = make(tmp_path, w, **{"strategies.daily_trend.rebalance_band": 0.0})
    decide(engine, D1, 100.0)  # 0.0005 x 10 000 = 5 < min notional 10: rejected
    assert not engine.has_queued_order(BTC)
    for ts in (D2, D3, D4):
        decide(engine, ts, 100.0)
        fill_at(engine, ts, 100.0)
    # D3 left 0.08 (8 notional); its zero target at D4 would be an order of 8: rejected too
    assert decisions(engine) == [
        (D1, "too_small", 0.0),
        (D2, "ok", 50.0),
        (D3, "ok", approx(49.92)),
        (D4, "too_small", 0.0),
    ]
    assert engine.portfolio.positions[BTC].qty == approx(0.08)


def test_an_unfilled_target_order_expires_at_the_next_decision(tmp_path: Path) -> None:
    engine = make(tmp_path, {D1: {BTC: 0.5}, D2: {BTC: 0.5}})
    decide(engine, D1, 100.0)
    decide(engine, D2, 100.0)  # no bar since D1: expired, then queued again
    assert decisions(engine) == [(D1, "ok", 50.0), (D2, "expired", 0.0), (D2, "ok", 50.0)]
    engine.fill_queued(BTC, D3 + MINUTE, 100.0, 100.0, 1e9)  # past D2 + one day: expired too
    assert decisions(engine)[-1] == (D3 + MINUTE, "expired", 0.0) and engine.log.fills == []


def test_an_order_that_adds_is_cut_by_participation_one_that_reduces_is_not(
    tmp_path: Path,
) -> None:
    engine = make(tmp_path, {D1: {BTC: 0.5}})
    decide(engine, D1, 100.0)
    fill_at(engine, D1, 100.0, volume=200.0)  # 5% of 200 = 10 of the 50
    assert engine.portfolio.positions[BTC].qty == 10.0
    engine2 = make(tmp_path, {D1: {BTC: 0.5}, D2: {BTC: 0.0}})
    decide(engine2, D1, 100.0)
    fill_at(engine2, D1, 100.0)
    decide(engine2, D2, 100.0)
    fill_at(engine2, D2, 100.0, volume=1.0)  # the 50 close fills in full on a thin bar
    assert engine2.portfolio.positions == {} and engine2.log.fills[-1].qty == 50.0


def test_minute_bars_are_not_needed_and_on_minute_never_exits_a_target_position(
    tmp_path: Path,
) -> None:
    engine = make(tmp_path, {D1: {BTC: 0.5}})
    assert engine.needs_minute_bars is False
    decide(engine, D1, 100.0)
    fill_at(engine, D1, 100.0)
    engine.on_minute(Bar(BTC, D1 + 2 * MINUTE, 100.0, 200.0, 1.0, 100.0, 1e6))
    assert engine.portfolio.positions[BTC].qty == 50.0


def test_a_halt_flattens_target_positions_and_blocks_rebalancing(tmp_path: Path) -> None:
    w = {ts: {BTC: 0.5} for ts in (D1, D2, D3)}
    engine = make(tmp_path, w)
    decide(engine, D1, 100.0)
    fill_at(engine, D1, 100.0)  # long 50
    decide(engine, D2, 80.0)  # -1 000: a 10% drawdown, the permanent halt
    assert engine.risk.halted_reason == "max_drawdown" and engine.queued_exits == {BTC: "kill"}
    fill_at(engine, D2, 80.0)
    decide(engine, D3, 80.0)
    assert [t.exit_reason for t in engine.log.trades] == [ExitReason.KILL]
    assert engine.portfolio.positions == {} and decisions(engine) == [(D1, "ok", 50.0)]


def test_the_loss_streak_rules_do_not_apply_to_target_positions(tmp_path: Path) -> None:
    # two losing trades in a row would stop a playbook strategy for the day (streak_stop 2)
    w = {}
    for k in range(12):
        w[T0 + k * DAY] = {BTC: 0.2 if k % 2 == 0 else 0.0}
    engine = make(tmp_path, w, **{"risk.streak_stop": 2, "risk.streak_half": 1})
    for k in range(12):
        ts = T0 + k * DAY
        decide(engine, ts, 100.0 if k % 2 == 0 else 99.0)
        fill_at(engine, ts, 100.0 if k % 2 == 0 else 99.0)
    assert len(engine.log.trades) == 6 and all(t.net_pnl < 0 for t in engine.log.trades)
    assert engine.risk.loss_streak == 0 and engine.risk.day_stopped is None
    assert [d.rule for _, _, d in engine.log.decisions] == ["ok"] * 12


def test_a_trend_config_takes_exactly_one_target_strategy(tmp_path: Path) -> None:
    from synth import Scripted

    cfg = trend_cfg()
    risk, p = RiskManager(cfg, tmp_path / "KILL"), Portfolio(10_000.0)
    for strategies in ([], [Fixed({}), Fixed({})], [Fixed({}), Scripted("breakout")]):
        with pytest.raises(ValueError, match="exactly one target-weight strategy"):
            Engine(cfg, strategies, risk, p, RunLog())
    with pytest.raises(ValueError, match="exactly one target-weight strategy"):
        Engine(load_config(), [Fixed({})], risk, p, RunLog())


# --- backtest: the engine reproduces the rule ------------------------------------------------


def market(days: int, seed: int) -> dict[str, pd.DataFrame]:
    """1m bars for BTC and ETH whose daily closes trend up and down; each minute bar opens at the
    previous close, so a fill at the first bar after a decision is at the decision close."""
    rng = np.random.default_rng(seed)
    out = {}
    for symbol, start in ((BTC, 100.0), (ETH, 50.0)):
        drift = np.repeat(rng.choice([-0.01, 0.01], days // 10 + 1), 10)[:days]
        closes = start * np.exp(np.cumsum(drift + rng.normal(0, 0.03, days)))
        path = np.concatenate([[start], closes])
        minutes = np.concatenate(
            [np.linspace(a, b, 1441)[1:] for a, b in zip(path[:-1], path[1:], strict=True)]
        )
        out[symbol] = candles_from_closes(minutes, volume=1e12)
    return out


def vectorised_equity(
    candles: dict[str, pd.DataFrame], cfg: AppConfig, start: datetime, end: datetime
) -> pd.Series:
    """The rule written out directly (as research/daily_trend.py does, with n not sqrt(n)): each
    day's weights held over the next day, no costs."""
    t = cfg.strategies.daily_trend
    at_midnight = {s: (df.index.hour == 0) & (df.index.minute == 0) for s, df in candles.items()}
    closes = pd.DataFrame({s: df["close"][at_midnight[s]] for s, df in candles.items()})
    signal = sum(np.sign(closes - closes.rolling(n).mean()) for n in t.lookbacks) / len(t.lookbacks)
    vol = closes.pct_change().rolling(t.vol_window).std() * np.sqrt(365)
    w = signal * (t.target_vol / closes.shape[1] / vol).clip(upper=cfg.risk.max_asset_exposure)
    w = w.div((w.abs().sum(axis=1) / cfg.risk.max_gross_leverage).clip(lower=1), axis=0)
    w = w[(w.index >= start) & (w.index < end)]
    r = closes.pct_change().loc[w.index]
    daily = (w.shift(1).fillna(0) * r).sum(axis=1)
    assert (w.abs() == cfg.risk.max_asset_exposure).any().all()  # the asset cap binds
    return cfg.backtest.initial_equity * (1 + daily).cumprod()


def test_the_engine_equity_matches_a_vectorised_computation_of_the_rule(tmp_path: Path) -> None:
    days = 120
    cfg = trend_cfg(
        **FREE,
        **{
            "exchange.symbols": [BTC, ETH],
            "strategies.daily_trend.lookbacks": [5, 10, 20],
            "strategies.daily_trend.vol_window": 10,
            "strategies.daily_trend.target_vol": 1.2,  # weights up to the asset cap
            "strategies.daily_trend.rebalance_band": 0.0,
            "risk.min_notional": 1e-9,
            # The rule caps each weight at 0.5, so gross stays under the 1.5 group and gross
            # caps: the risk caps never cut a target here (at the limit they would,
            # conservatively: a queued reduction frees room only once it fills).
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
    assert len(result.equity) == len(expected) == days - 25
    np.testing.assert_allclose(result.equity.to_numpy(), expected.to_numpy(), rtol=1e-9)
    assert len(result.trades) > 4 and {t.side for t in result.trades} == {Side.LONG, Side.SHORT}
    # warm at the start, no halt: only the absent funding is recorded
    assert {e.kind for e in result.log.risk_events} == {"funding_missing"}
