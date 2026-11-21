"""Engine event handling: entries, exits, trailing, kill switch, playbook, caps, funding.

Windows are hand-built one-row feature frames whose values pick the regime (see ``ROWS``), and
strategies are ``Scripted`` stubs, so every number below follows from the default config:
risk_per_trade 0.005 of 10 000 = 50 at the stop; taker 5 bps, maker 2 bps, half spread 1 bp,
slippage 2 bps, impact 1 bp per 1% of bar volume, max participation 5%.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from synth import BTC, ETH, MINUTE, SOL, T0, Scripted

from regimebot.config import load_config
from regimebot.engine.loop import REGIME_INPUTS, Engine, RunLog
from regimebot.engine.portfolio import Portfolio
from regimebot.risk.manager import RiskManager
from regimebot.types import Bar, ExitReason, OrderSide, Regime, Side, Signal

D = timedelta(minutes=15)
T1, T2, T3, T4, T5, T6 = (T0 + k * D for k in range(1, 7))
_CALM = {"atr_pct": 0.01, "vol_pctile": 0.5, "volume_z": 0.0, "ret_z": 0.0, "illiq_pctile": 0.5}
ROWS = {  # feature values that make ``classify`` return the key (closes are near 100)
    Regime.TREND_UP: {**_CALM, "adx": 30.0, "ema_fast": 2.0, "ema_slow": 1.0},
    Regime.RANGE: {**_CALM, "adx": 15.0, "ema_fast": 1.0, "ema_slow": 1.0},
    Regime.NO_TRADE: {**_CALM, "adx": 22.0, "ema_fast": 1.0, "ema_slow": 1.0},
    Regime.LIQUIDITY_STRESS: {**_CALM, "adx": 15.0, "ema_fast": 1.0, "ema_slow": 1.0}
    | {"illiq_pctile": 0.995},
}
FREE_FILLS = {"costs.slippage_bps": 0.0, "costs.impact_bps_per_pct": 0.0}  # fill at the quote


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-12, abs=1e-9)


def make(
    tmp_path: Path,
    *strategies: Scripted,
    symbols: tuple[str, ...] = (BTC,),
    overrides: dict[str, object] | None = None,
) -> Engine:
    cfg = load_config(overrides={"exchange.symbols": list(symbols), **(overrides or {})})
    portfolio = Portfolio(cfg.backtest.initial_equity)
    return Engine(cfg, strategies, RiskManager(cfg, tmp_path / "KILL"), portfolio, RunLog())


def window(
    ts: datetime, close: float, regime: Regime = Regime.TREND_UP, atr: float = 1.0
) -> pd.DataFrame:
    return pd.DataFrame(
        [{"close": close, "atr": atr, **ROWS[regime]}], index=pd.DatetimeIndex([ts])
    )


def decide(
    engine: Engine,
    ts: datetime,
    closes: dict[str, float],
    regime: Regime = Regime.TREND_UP,
    atr: float = 1.0,
) -> None:
    engine.on_decision(ts, {s: window(ts, c, regime, atr) for s, c in closes.items()})


def bar(ts: datetime, open_: float, high: float, low: float, symbol: str = BTC) -> Bar:
    return Bar(symbol, ts, open_, high, low, open_, 1e6)


def rules(engine: Engine) -> list[tuple[int, str]]:
    return [(i, d.rule) for i, _, d in engine.log.decisions]


# --- entries ---------------------------------------------------------------------------------


def test_signal_is_sized_queued_and_filled_at_the_quote_not_the_close(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: 100.0})
    # risk 50 / stop distance (100 - 98) = qty 25
    ((index, ts, decision),) = engine.log.decisions
    assert (index, ts, decision.approved, decision.rule) == (0, T1, True, "ok")
    assert decision.qty == approx(25.0)
    assert engine.portfolio.positions == {} and engine.is_active(BTC)
    assert engine.log.regimes == [
        (BTC, T1, Regime.TREND_UP, {k: window(T1, 100.0).iloc[-1][k] for k in REGIME_INPUTS})
    ]

    engine.fill_queued(BTC, T1, bid=99.99, ask=100.01, bar_volume=2500.0)
    # 25 / 2500 = 1% of the bar: slippage 2 + impact 1 = 3 bps on the ask
    (fill,) = engine.log.fills
    assert (fill.ts, fill.side, fill.ref_price) == (T1, OrderSide.BUY, 100.0)
    assert fill.qty == approx(25.0)
    assert fill.price == approx(100.01 * 1.0003)  # 100.040003
    assert fill.fee == approx(25 * 100.040003 * 0.0005)  # 1.2505000375
    position = engine.portfolio.positions[BTC]
    assert (position.qty, position.entry_price, position.stop) == (fill.qty, fill.price, 98.0)
    assert engine.portfolio.cash == approx(10_000 - 1.2505000375)


def test_participation_cap_gives_a_partial_entry(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: 100.0})
    engine.fill_queued(BTC, T1, 99.99, 100.01, bar_volume=100.0)
    # cut to 5% of 100 = 5; impact at 5% participation = 5 bps, plus 2 bps slippage
    (fill,) = engine.log.fills
    assert fill.qty == approx(5.0)
    assert fill.price == approx(100.01 * 1.0007)
    assert engine.portfolio.positions[BTC].qty == approx(5.0)


@pytest.mark.parametrize("volume", [0.0, 1.0, 5e-324], ids=["zero", "dust", "subnormal"])
def test_entry_without_liquidity_is_cancelled_and_the_symbol_is_free_again(
    tmp_path: Path, volume: float
) -> None:
    # zero volume: no fill; 1.0: 0.05 units ~ 5 quote < 10; subnormal: qty underflows to 0.0
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: 100.0})
    engine.fill_queued(BTC, T1, 99.99, 100.01, bar_volume=volume)
    assert rules(engine) == [(0, "ok"), (0, "no_liquidity")]
    assert engine.log.fills == [] and engine.portfolio.positions == {}
    assert not engine.is_active(BTC)
    decide(engine, T2, {BTC: 100.0})
    assert rules(engine)[-1] == (1, "ok")


def test_an_entry_not_filled_before_the_next_decision_expires(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: 100.0})  # queued, but no bar arrives (feed gap)
    decide(engine, T2, {BTC: 100.0})
    assert rules(engine) == [(0, "ok"), (0, "expired"), (1, "ok")]  # T2 signals afresh
    assert engine.log.decisions[1][1] == T2


@pytest.mark.parametrize(("delay", "fills"), [(D - MINUTE, True), (D, False), (2 * D, False)])
def test_a_fill_offered_at_or_after_the_next_decision_close_expires(
    tmp_path: Path, delay: timedelta, fills: bool
) -> None:
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: 100.0})
    engine.fill_queued(BTC, T1 + delay, 100.0, 100.0, 1e6)
    assert (BTC in engine.portfolio.positions) == fills
    assert rules(engine) == ([(0, "ok")] if fills else [(0, "ok"), (0, "expired")])
    assert engine.is_active(BTC) == fills  # an expired entry frees the symbol


def test_strategy_returning_another_bars_signal_is_an_error(tmp_path: Path) -> None:
    class Stale(Scripted):
        def generate(self, symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None:
            return Signal("trend", symbol, T0, Side.LONG, 100.0, 98.0, 104.0, 1.0, regime, 10, None)

    with pytest.raises(ValueError, match="expected"):
        decide(make(tmp_path, Stale("trend")), T1, {BTC: 100.0})


# --- exits -----------------------------------------------------------------------------------


def opened(
    tmp_path: Path, strategy: Scripted, overrides: dict[str, object] | None = None
) -> Engine:
    """Engine with one BTC position from a signal at T1 (close 100), filled at T1."""
    engine = make(tmp_path, strategy, overrides=overrides)
    decide(engine, T1, {BTC: 100.0})
    engine.fill_queued(BTC, T1, 100.0, 100.0, 1e6)
    assert BTC in engine.portfolio.positions
    return engine


def test_stop_is_assumed_first_when_one_bar_touches_stop_and_target(tmp_path: Path) -> None:
    engine = opened(tmp_path, Scripted("trend"))  # stop 98, target 104
    engine.on_minute(bar(T1 + MINUTE, 100.0, high=105.0, low=97.0))
    (trade,) = engine.log.trades
    assert trade.exit_reason == ExitReason.STOP
    assert engine.log.fills[-1].ref_price == 98.0 and trade.exit_price < 98.0


def test_target_fills_at_the_limit_with_maker_fee(tmp_path: Path) -> None:
    engine = opened(tmp_path, Scripted("trend"))
    engine.on_minute(bar(T1 + MINUTE, 100.0, high=104.0, low=99.0))  # a touch is not a fill
    assert engine.log.trades == []
    engine.on_minute(bar(T1 + 2 * MINUTE, 100.0, high=104.5, low=99.0))
    (trade,) = engine.log.trades
    assert (trade.exit_reason, trade.exit_price, trade.exit_ts) == (
        ExitReason.TARGET,
        104.0,
        T1 + 2 * MINUTE,
    )
    assert engine.log.fills[-1].fee == approx(25 * 104.0 * 0.0002)


def test_time_exit_after_max_hold_bars_at_the_next_quote(tmp_path: Path) -> None:
    engine = opened(tmp_path, Scripted("trend", max_hold_bars=3))
    for ts in (T2, T3):
        decide(engine, ts, {BTC: 100.5})
        engine.fill_queued(BTC, ts, 100.0, 100.0, 1e6)  # nothing queued yet: no-op
    assert engine.log.trades == []
    decide(engine, T4, {BTC: 101.0})  # third bar held
    engine.fill_queued(BTC, T4, 100.99, 101.01, 1e6)
    (trade,) = engine.log.trades
    assert (trade.exit_reason, trade.bars_held, trade.exit_ts) == (ExitReason.TIME, 3, T4)
    exit_fill = engine.log.fills[-1]
    assert exit_fill.ref_price == 101.0  # last decision close
    # SELL at the bid; 25 of 1e6 is 0.0025% of the bar: 2 + 0.0025 bps
    assert exit_fill.price == approx(100.99 * (1 - 2.0025e-4))
    assert [e[0] for e in engine.log.equity] == [T1, T2, T3, T4]


def test_force_close_regime_queues_a_regime_exit_and_no_entry(tmp_path: Path) -> None:
    engine = opened(tmp_path, Scripted("trend"))
    decide(engine, T2, {BTC: 100.0}, Regime.LIQUIDITY_STRESS)
    assert engine.log.regimes[-1][:3] == (BTC, T2, Regime.LIQUIDITY_STRESS)
    engine.fill_queued(BTC, T2, 100.0, 100.0, 1e6)
    assert [t.exit_reason for t in engine.log.trades] == [ExitReason.REGIME]
    assert len(engine.log.signals) == 1 and engine.portfolio.positions == {}


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_trailing_stop_starts_at_1r_tightens_and_never_loosens(tmp_path: Path, side: Side) -> None:
    # Fills at the quote: entry 100, initial stop 98 (long), 1R = 2, trail 3 ATR. Short mirrors
    # every price around 100.
    def px(x: float) -> float:
        return x if side == Side.LONG else 200.0 - x

    strategy = Scripted("trend", side=side, stop_dist=2.0, target_dist=10.0, trail_atr_mult=3.0)
    engine = opened(tmp_path, strategy, FREE_FILLS)
    assert engine.portfolio.positions[BTC].entry_price == 100.0
    steps = [  # (ts, close, atr, expected stop)
        (T2, 101.5, 0.5, 98.0),  # 1.5 in favour < 1R: no trailing
        (T3, 102.0, 0.5, 100.5),  # exactly 1R: trails to 102 - 3 * 0.5
        (T4, 102.5, 0.5, 101.0),  # 102.5 - 1.5
        (T5, 102.0, 0.25, 101.75),  # best stays 102.5: 102.5 - 0.75 tightens
        (T6, 103.0, 2.0, 101.75),  # 103 - 6 = 97 would loosen: kept
    ]
    for ts, close, atr, stop in steps:
        decide(engine, ts, {BTC: px(close)}, atr=atr)
        assert engine.portfolio.positions[BTC].stop == approx(px(stop))
    assert engine.portfolio.positions[BTC].best_price == px(103.0)
    engine.on_minute(
        bar(T6 + MINUTE, px(101.8), high=max(px(101.9), px(101.7)), low=min(px(101.9), px(101.7)))
    )
    (trade,) = engine.log.trades
    assert trade.exit_reason == ExitReason.TRAIL
    assert engine.log.fills[-1].ref_price == approx(px(101.75))


# --- kill switch, playbook, caps, stale data --------------------------------------------------


def test_kill_switch_flattens_every_position_and_blocks_entries(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("trend"), symbols=(BTC, ETH))
    decide(engine, T1, {BTC: 100.0, ETH: 100.0})
    for symbol in (BTC, ETH):
        engine.fill_queued(symbol, T1, 100.0, 100.0, 1e6)
    (tmp_path / "KILL").touch()
    decide(engine, T2, {BTC: 100.0, ETH: 100.0})
    assert [(e.kind, e.action) for e in engine.log.risk_events] == [("manual_kill", "halt")]
    for symbol in (BTC, ETH):
        engine.fill_queued(symbol, T2, 100.0, 100.0, 1e6)
    assert [(t.symbol, t.exit_reason) for t in engine.log.trades] == [
        (BTC, ExitReason.KILL),
        (ETH, ExitReason.KILL),
    ]
    (tmp_path / "KILL").unlink()  # manual_kill stays until clear_halt
    for ts in (T3, T4):
        decide(engine, ts, {BTC: 100.0, ETH: 100.0})
    assert len(engine.log.signals) == 2 and engine.portfolio.positions == {}
    assert not engine.is_active(BTC) and not engine.is_active(ETH)


def test_kill_switch_cancels_an_entry_still_queued(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: 100.0})
    (tmp_path / "KILL").touch()
    decide(engine, T2, {BTC: 100.0})
    assert rules(engine) == [(0, "ok"), (0, "kill_switch")]
    engine.fill_queued(BTC, T2, 100.0, 100.0, 1e6)
    assert engine.log.fills == [] and engine.portfolio.positions == {}


def test_playbook_decides_which_strategies_run(tmp_path: Path) -> None:
    always = [Scripted("mean_reversion"), Scripted("breakout"), Scripted("trend")]
    engine = make(tmp_path, *always, symbols=(BTC, ETH, SOL))
    engine.on_decision(
        T1,
        {
            BTC: window(T1, 100.0, Regime.TREND_UP),  # playbook [trend, breakout]
            ETH: window(T1, 100.0, Regime.RANGE),  # [mean_reversion]
            SOL: window(T1, 100.0, Regime.NO_TRADE),  # []
        },
    )
    # mean_reversion always fires but is not allowed in TREND_UP or NO_TRADE; of the two
    # allowed in TREND_UP, trend is first in the playbook (engine list order does not matter).
    assert [(s.symbol, s.strategy) for s in engine.log.signals] == [
        (BTC, "trend"),
        (ETH, "mean_reversion"),
    ]


@pytest.mark.parametrize(
    ("order", "winner"), [(["trend", "breakout"], "trend"), (["breakout", "trend"], "breakout")]
)
def test_the_playbooks_order_picks_between_strategies_that_both_fire(
    tmp_path: Path, order: list[str], winner: str
) -> None:
    playbook = {"strategies": order, "size_mult": 1.0, "allow_entries": True, "force_close": False}
    engine = make(
        tmp_path, Scripted("trend"), Scripted("breakout"), overrides={"playbook.trend_up": playbook}
    )
    decide(engine, T1, {BTC: 100.0})
    assert [s.strategy for s in engine.log.signals] == [winner]


def test_a_playbook_strategy_without_an_instance_is_skipped(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("breakout"))  # trend disabled: not built
    decide(engine, T1, {BTC: 100.0})  # TREND_UP: [trend, breakout]
    assert [s.strategy for s in engine.log.signals] == ["breakout"]


def test_entries_in_one_bar_share_the_group_and_gross_caps(tmp_path: Path) -> None:
    # stop 0.5 below 100: risk 50 / 0.5 = 100 units = 10 000 notional each. BTC is cut to the
    # asset cap 0.75 * 10 000 = 7500; ETH sees 7500 pending in group majors (cap 1.0 * 10 000)
    # and gets the 2500 left; SOL finds no group room.
    engine = make(tmp_path, Scripted("trend", stop_dist=0.5), symbols=(BTC, ETH, SOL))
    decide(engine, T1, {BTC: 100.0, ETH: 100.0, SOL: 100.0})
    got = [(engine.log.signals[i].symbol, d.rule, d.qty) for i, _, d in engine.log.decisions]
    assert got == [
        (BTC, "cap_asset", approx(75.0)),
        (ETH, "cap_group", approx(25.0)),
        (SOL, "cap_group", 0.0),
    ]
    approved = sum(d.qty * 100.0 for _, _, d in engine.log.decisions if d.approved)
    assert approved == approx(10_000.0)  # group cap; gross cap is 15 000


def test_stale_window_is_no_trade_with_an_event_and_keeps_the_position(tmp_path: Path) -> None:
    engine = make(tmp_path, Scripted("trend"), symbols=(BTC, ETH))
    decide(engine, T1, {BTC: 100.0})  # ETH has no window: stale, no signal
    engine.fill_queued(BTC, T1, 100.0, 100.0, 1e6)
    entry_mark = engine.portfolio.last_prices[BTC]
    engine.on_decision(T2, {BTC: window(T1, 90.0, Regime.LIQUIDITY_STRESS)})  # last row is T1
    stale = [(e.ts, e.detail, e.action) for e in engine.log.risk_events if e.kind == "stale_data"]
    assert stale == [
        (T1, f"{ETH}: no features", "skip_symbol"),
        (T2, f"{BTC}: last features row at {pd.Timestamp(T1)}", "skip_symbol"),
        (T2, f"{ETH}: no features", "skip_symbol"),
    ]
    assert [r for r in engine.log.regimes if r[1] == T2] == [
        (BTC, T2, Regime.NO_TRADE, {}),
        (ETH, T2, Regime.NO_TRADE, {}),
    ]
    engine.fill_queued(BTC, T2, 100.0, 100.0, 1e6)  # no forced exit was queued
    position = engine.portfolio.positions[BTC]
    assert (position.bars_held, engine.portfolio.last_prices[BTC]) == (1, entry_mark)
    assert engine.log.trades == [] and len(engine.log.signals) == 1


@pytest.mark.parametrize("close", [float("nan"), 0.0, float("inf")])
def test_window_with_an_unusable_close_is_stale(tmp_path: Path, close: float) -> None:
    engine = make(tmp_path, Scripted("trend"))
    decide(engine, T1, {BTC: close})
    (event,) = engine.log.risk_events
    assert (event.kind, event.detail) == ("stale_data", f"{BTC}: invalid close {close!r}")
    assert engine.log.signals == [] and BTC not in engine.portfolio.last_prices


# --- funding and flatten ---------------------------------------------------------------------


def test_funding_is_charged_only_to_a_position_held_at_the_funding_time(tmp_path: Path) -> None:
    engine = opened(tmp_path, Scripted("trend"), FREE_FILLS)  # long 25 entered at T1
    cash = engine.portfolio.cash
    engine.on_funding(BTC, T1, 0.01, 100.0)  # stamped at the entry: not held then
    engine.on_funding(ETH, T2, 0.01, 100.0)  # no position
    assert engine.portfolio.cash == cash
    engine.on_funding(BTC, T2, 0.0001, 100.0)  # long pays 25 * 100 * 0.0001
    assert engine.portfolio.positions[BTC].funding == approx(0.25)
    assert engine.portfolio.cash == approx(cash - 0.25)


def test_flatten_all_closes_at_worst_case_impact_and_cancels_queued_entries(
    tmp_path: Path,
) -> None:
    engine = make(tmp_path, Scripted("trend"), symbols=(BTC, ETH))
    decide(engine, T1, {BTC: 100.0, ETH: 100.0})
    engine.fill_queued(BTC, T1, 100.0, 100.0, 1e6)  # ETH stays queued
    engine.flatten_all(T1, {BTC: 100.0, ETH: 100.0}, ExitReason.END_OF_DATA)
    (trade,) = engine.log.trades
    assert trade.exit_reason == ExitReason.END_OF_DATA and trade.exit_ts == T1
    # bid 100 * (1 - 1 bp) = 99.99; full participation: 2 + 100 bps slippage
    assert trade.exit_price == approx(99.99 * (1 - 0.0102))
    assert rules(engine) == [(0, "ok"), (1, "ok"), (1, "cancelled")]
    assert engine.portfolio.positions == {} and not engine.is_active(ETH)
    assert engine.log.equity == [(T1, engine.portfolio.cash, 0.0, engine.portfolio.drawdown)]
    assert engine.portfolio.cash == approx(10_000 + trade.net_pnl)


def test_flatten_all_keeps_the_reason_of_an_exit_already_queued(tmp_path: Path) -> None:
    engine = opened(tmp_path, Scripted("trend", max_hold_bars=1))
    decide(engine, T2, {BTC: 100.0})  # TIME exit queued, flatten comes before it fills
    engine.flatten_all(T2, {BTC: 100.0}, ExitReason.END_OF_DATA)
    assert [t.exit_reason for t in engine.log.trades] == [ExitReason.TIME]
