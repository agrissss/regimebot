"""Portfolio accounting: hand-computed round trips, marks, drawdown and the cash invariant.

Linear USDT perpetual: gross pnl = qty * (exit - entry) * side_sign (+1 long, -1 short).
Fills are written out by hand (price chosen, fee shown as bps of notional) so only the
accounting is under test; the random-trades invariant drives real fills from the fill model.
"""

import json
import random
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from regimebot.config import CostsConfig
from regimebot.engine.portfolio import Portfolio, Position
from regimebot.execution.fill_model import market_fill, quote_from_bar_open, stop_fill, target_fill
from regimebot.types import Bar, ExitReason, Fill, OrderSide, Regime, Side, Signal, Trade

INITIAL = 10_000.0
BTC, ETH, SOL = "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"
T_OPEN = datetime(2024, 1, 1, 0, 16, tzinfo=UTC)
T_CLOSE = datetime(2024, 1, 1, 8, 1, tzinfo=UTC)
BUY, SELL = OrderSide.BUY, OrderSide.SELL


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def make_signal(side: Side, symbol: str = BTC, stop: float = 95.0, target: float = 110.0) -> Signal:
    return Signal(
        strategy="breakout",
        symbol=symbol,
        ts=T_OPEN - timedelta(minutes=1),
        side=side,
        entry=100.0,
        stop=stop,
        target=target,
        atr=2.5,
        regime=Regime.TREND_UP,
        max_hold_bars=48,
        trail_atr_mult=2.0,
    )


def make_fill(
    side: OrderSide, qty: float, price: float, fee: float, ts: datetime = T_OPEN, symbol: str = BTC
) -> Fill:
    return Fill(symbol, ts, side, qty, price, price, fee, 0.0, "taker")


def round_trip(
    signal: Signal, entry: Fill, funding_mark: float, funding_rate: float, exit_: Fill
) -> tuple[Portfolio, Trade]:
    p = Portfolio(INITIAL)
    p.open(signal, entry)
    p.apply_funding(signal.symbol, funding_mark, funding_rate)
    p.positions[signal.symbol].bars_held = 7  # the engine counts bars held
    return p, p.close(signal.symbol, exit_, ExitReason.STOP)


def check_money(
    p: Portfolio, trade: Trade, gross: float, fees: float, funding: float, net: float, r: float
) -> None:
    assert trade.gross_pnl == approx(gross)
    assert trade.fees == approx(fees)
    assert trade.funding == approx(funding)
    assert trade.net_pnl == approx(net)
    assert trade.r_multiple == approx(r)
    assert p.cash == approx(INITIAL + net)  # the invariant, per trade
    assert p.positions == {}
    assert p.equity == p.cash
    assert p.gross_exposure == 0.0


# --- open -----------------------------------------------------------------------------------


def test_open_takes_qty_and_price_from_fill_and_plan_from_signal() -> None:
    # Fill smaller than requested: 3 @ 100.2 ; fee = 3 * 100.2 * 5e-4 = 0.1503
    p = Portfolio(INITIAL)
    position = p.open(make_signal(Side.LONG), make_fill(BUY, 3.0, 100.2, 0.1503))
    assert position == Position(
        symbol=BTC,
        strategy="breakout",
        side=Side.LONG,
        regime=Regime.TREND_UP,
        qty=3.0,
        entry_price=100.2,
        entry_ts=T_OPEN,
        stop=95.0,
        initial_stop=95.0,
        target=110.0,
        atr_at_entry=2.5,
        max_hold_bars=48,
        trail_atr_mult=2.0,
        bars_held=0,
        best_price=100.2,
        fees=0.1503,
        funding=0.0,
    )
    assert p.positions == {BTC: position}
    assert p.cash == approx(9999.8497)  # 10000 - 0.1503


# --- round trips ----------------------------------------------------------------------------


def test_long_win_with_fees_and_funding() -> None:
    p = Portfolio(INITIAL)
    # Long 10 @ 100, entry fee 5 bps of 1000 = 0.5 ; cash = 10000 - 0.5 = 9999.5
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    assert p.cash == approx(9999.5)
    # Funding at mark 102, rate +0.0001: long pays 10 * 102 * 0.0001 = 0.102 ; cash = 9999.398
    p.apply_funding(BTC, 102.0, 0.0001)
    assert p.cash == approx(9999.398)
    assert p.positions[BTC].funding == approx(0.102)
    p.positions[BTC].bars_held = 7
    # Target exit SELL 10 @ 110, maker fee 2 bps of 1100 = 0.22
    # gross = 10 * (110 - 100) * (+1) = 100 ; cash = 9999.398 + 100 - 0.22 = 10099.178
    # fees = 0.5 + 0.22 = 0.72 ; net = 100 - 0.72 - 0.102 = 99.178
    # R = 99.178 / (10 * |100 - 95|) = 99.178 / 50 = 1.98356
    trade = p.close(BTC, make_fill(SELL, 10.0, 110.0, 0.22, ts=T_CLOSE), ExitReason.TARGET)
    check_money(p, trade, gross=100.0, fees=0.72, funding=0.102, net=99.178, r=1.98356)
    assert p.cash == approx(10099.178)
    assert (trade.symbol, trade.strategy, trade.side, trade.regime) == (
        BTC,
        "breakout",
        Side.LONG,
        Regime.TREND_UP,
    )
    assert (trade.qty, trade.entry_ts, trade.exit_ts) == (10.0, T_OPEN, T_CLOSE)
    assert (trade.entry_price, trade.exit_price, trade.initial_stop, trade.target) == (
        100.0,
        110.0,
        95.0,
        110.0,
    )
    assert (trade.exit_reason, trade.bars_held) == (ExitReason.TARGET, 7)


def test_long_loss_with_fees_and_funding_received() -> None:
    # Long 10 @ 100, fee 0.5 ; funding at mark 98, rate -0.0001: long receives,
    #   payment = 10 * 98 * -0.0001 = -0.098 (cash + 0.098)
    # Stop exit SELL 10 @ 95, taker fee 5 bps of 950 = 0.475
    # gross = 10 * (95 - 100) = -50 ; fees = 0.5 + 0.475 = 0.975
    # net = -50 - 0.975 - (-0.098) = -50.877 ; cash = 10000 - 0.5 + 0.098 - 50 - 0.475 = 9949.123
    # R = -50.877 / (10 * 5) = -1.01754
    p, trade = round_trip(
        make_signal(Side.LONG),
        make_fill(BUY, 10.0, 100.0, 0.5),
        98.0,
        -0.0001,
        make_fill(SELL, 10.0, 95.0, 0.475, ts=T_CLOSE),
    )
    check_money(p, trade, gross=-50.0, fees=0.975, funding=-0.098, net=-50.877, r=-1.01754)
    assert p.cash == approx(9949.123)
    assert trade.bars_held == 7


def test_short_win_with_fees_and_funding_received() -> None:
    # Short 4 @ 100 (stop 105, target 90), fee 5 bps of 400 = 0.2
    # funding at mark 99, rate +0.0001: short receives, payment = -1 * 4 * 99 * 0.0001 = -0.0396
    # Target exit BUY 4 @ 90, maker fee 2 bps of 360 = 0.072
    # gross = 4 * (90 - 100) * (-1) = +40 ; fees = 0.272 ; net = 40 - 0.272 + 0.0396 = 39.7676
    # cash = 10000 - 0.2 + 0.0396 + 40 - 0.072 = 10039.7676 ; R = 39.7676 / (4 * 5) = 1.98838
    p, trade = round_trip(
        make_signal(Side.SHORT, stop=105.0, target=90.0),
        make_fill(SELL, 4.0, 100.0, 0.2),
        99.0,
        0.0001,
        make_fill(BUY, 4.0, 90.0, 0.072, ts=T_CLOSE),
    )
    check_money(p, trade, gross=40.0, fees=0.272, funding=-0.0396, net=39.7676, r=1.98838)
    assert p.cash == approx(10039.7676)


def test_short_loss_with_fees_and_funding_paid() -> None:
    # Short 4 @ 100 (stop 105), fee 0.2 ; funding at mark 103, rate -0.0001: short pays,
    #   payment = -1 * 4 * 103 * -0.0001 = +0.0412
    # Stop exit BUY 4 @ 105, taker fee 5 bps of 420 = 0.21
    # gross = 4 * (105 - 100) * (-1) = -20 ; fees = 0.41 ; net = -20 - 0.41 - 0.0412 = -20.4512
    # cash = 10000 - 0.2 - 0.0412 - 20 - 0.21 = 9979.5488 ; R = -20.4512 / (4 * 5) = -1.02256
    p, trade = round_trip(
        make_signal(Side.SHORT, stop=105.0, target=90.0),
        make_fill(SELL, 4.0, 100.0, 0.2),
        103.0,
        -0.0001,
        make_fill(BUY, 4.0, 105.0, 0.21, ts=T_CLOSE),
    )
    check_money(p, trade, gross=-20.0, fees=0.41, funding=0.0412, net=-20.4512, r=-1.02256)
    assert p.cash == approx(9979.5488)


# --- marks, equity, drawdown, exposure ------------------------------------------------------


def test_equity_drawdown_and_exposure_follow_marks() -> None:
    p = Portfolio(INITIAL)
    assert (p.equity, p.peak_equity, p.drawdown) == (INITIAL, INITIAL, 0.0)
    # Long BTC 10 @ 100, fee 0.5 ; short ETH 2 @ 50, fee 5 bps of 100 = 0.05
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    p.open(
        make_signal(Side.SHORT, ETH, stop=52.5, target=45.0),
        make_fill(SELL, 2.0, 50.0, 0.05, symbol=ETH),
    )
    # cash = 10000 - 0.55 = 9999.45 ; both marked at their fill: equity = 9999.45
    # peak stays 10000 ; drawdown = 0.55 / 10000 = 0.000055
    assert p.cash == approx(9999.45)
    assert p.equity == approx(9999.45)
    assert p.peak_equity == INITIAL
    assert p.drawdown == approx(0.000055)

    p.mark({BTC: 110.0, ETH: 45.0})
    # unrealised BTC = 10 * (110 - 100) = +100 ; ETH = 2 * (45 - 50) * (-1) = +10
    # equity = 9999.45 + 110 = 10109.45 -> new peak, drawdown 0
    # exposure BTC = 10 * 110 = 1100 ; ETH = 2 * 45 = 90 ; gross = 1190
    assert p.equity == approx(10109.45)
    assert p.peak_equity == approx(10109.45)
    assert p.drawdown == 0.0
    assert (p.exposure(BTC), p.exposure(ETH), p.exposure(SOL)) == (
        approx(1100.0),
        approx(90.0),
        0.0,
    )
    assert p.gross_exposure == approx(1190.0)

    p.mark({BTC: 95.0, ETH: 55.0})
    # BTC = 10 * (95 - 100) = -50 ; ETH = 2 * (55 - 50) * (-1) = -10 ; equity = 9939.45
    # drawdown = (10109.45 - 9939.45) / 10109.45 = 170 / 10109.45 ; gross = 950 + 110 = 1060
    assert p.equity == approx(9939.45)
    assert p.peak_equity == approx(10109.45)
    assert p.drawdown == approx(170 / 10109.45)
    assert p.gross_exposure == approx(1060.0)

    p.mark({BTC: 100.0})  # partial mark: ETH keeps its last price 55
    # BTC = 0 ; ETH = -10 ; equity = 9999.45 - 10 = 9989.45
    assert p.equity == approx(9989.45)

    p.close(BTC, make_fill(SELL, 10.0, 100.0, 0.5, ts=T_CLOSE), ExitReason.TIME)
    # gross 0, exit fee 0.5: cash = 9998.95 ; equity = 9998.95 - 10 = 9988.95
    # drawdown = (10109.45 - 9988.95) / 10109.45 = 120.5 / 10109.45 ; gross exposure = 110
    assert p.cash == approx(9998.95)
    assert p.equity == approx(9988.95)
    assert p.drawdown == approx(120.5 / 10109.45)
    assert p.gross_exposure == approx(110.0)

    p.apply_funding(ETH, 55.0, 0.001)
    # short receives 2 * 55 * 0.001 = 0.11 ; cash = 9999.06 ; equity = 9989.06
    # drawdown = (10109.45 - 9989.06) / 10109.45 = 120.39 / 10109.45
    assert p.cash == approx(9999.06)
    assert p.drawdown == approx(120.39 / 10109.45)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_mark_rejects_invalid_prices_without_changing_state(bad: float) -> None:
    p = Portfolio(INITIAL)
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    with pytest.raises(ValueError):
        p.mark({BTC: 105.0, ETH: bad})
    assert p.equity == approx(9999.5)  # BTC still marked at its entry fill 100


def test_open_replaces_an_older_mark_with_the_fill_price() -> None:
    # BTC marked at 90 while flat, then opened 10 @ 100: unrealised must be 0, not 10 * -10
    p = Portfolio(INITIAL)
    p.mark({BTC: 90.0})
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    assert p.equity == approx(9999.5)  # 10000 - 0.5 fee
    assert p.exposure(BTC) == approx(1000.0)  # 10 * 100


def test_funding_without_position_is_a_no_op() -> None:
    p = Portfolio(INITIAL)
    p.apply_funding(BTC, 100.0, 0.01)
    assert p.cash == INITIAL


# --- invalid operations ---------------------------------------------------------------------


def test_second_position_on_a_symbol_is_rejected() -> None:
    p = Portfolio(INITIAL)
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    with pytest.raises(ValueError, match="already"):
        p.open(make_signal(Side.LONG), make_fill(BUY, 1.0, 100.0, 0.05))
    assert p.cash == approx(9999.5)
    assert p.positions[BTC].qty == 10.0


@pytest.mark.parametrize(
    ("signal", "fill"),
    [
        (make_signal(Side.LONG), make_fill(SELL, 1.0, 100.0, 0.05)),  # long opened by a SELL
        (make_signal(Side.SHORT, stop=105.0), make_fill(BUY, 1.0, 100.0, 0.05)),  # short by BUY
        (make_signal(Side.LONG), make_fill(BUY, 1.0, 100.0, 0.05, symbol=ETH)),  # other symbol
        (make_signal(Side.LONG, stop=100.0), make_fill(BUY, 1.0, 100.0, 0.05)),  # zero risk (1R)
    ],
)
def test_open_rejects_fill_that_does_not_match_signal(signal: Signal, fill: Fill) -> None:
    p = Portfolio(INITIAL)
    with pytest.raises(ValueError):
        p.open(signal, fill)
    assert p.positions == {}
    assert p.cash == INITIAL


@pytest.mark.parametrize(
    ("symbol", "fill"),
    [
        (ETH, make_fill(SELL, 10.0, 100.0, 0.5, ts=T_CLOSE, symbol=ETH)),  # no ETH position
        (BTC, make_fill(BUY, 10.0, 100.0, 0.5, ts=T_CLOSE)),  # long closed by a BUY
        (BTC, make_fill(SELL, 10.0, 100.0, 0.5, ts=T_CLOSE, symbol=ETH)),  # fill for ETH
        (BTC, make_fill(SELL, 9.99, 100.0, 0.5, ts=T_CLOSE)),  # partial exit
        (BTC, make_fill(SELL, 10.0, 100.0, 0.5, ts=T_OPEN - timedelta(minutes=1))),  # before entry
    ],
)
def test_close_rejects_invalid_fill_without_changing_state(symbol: str, fill: Fill) -> None:
    p = Portfolio(INITIAL)
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    with pytest.raises(ValueError):
        p.close(symbol, fill, ExitReason.TIME)
    assert p.cash == approx(9999.5)
    assert p.positions[BTC].qty == 10.0


def test_close_accepts_qty_within_float_tolerance() -> None:
    # 10 * (1 + 1e-12) is the same quantity after float noise: closes, gross = 0
    p = Portfolio(INITIAL)
    p.open(make_signal(Side.LONG), make_fill(BUY, 10.0, 100.0, 0.5))
    trade = p.close(BTC, make_fill(SELL, 10.0 * (1 + 1e-12), 100.0, 0.5), ExitReason.TIME)
    assert trade.qty == 10.0
    assert p.cash == approx(9999.0)  # 10000 - 0.5 - 0.5


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_portfolio_rejects_non_positive_initial_equity(bad: float) -> None:
    with pytest.raises(ValueError):
        Portfolio(bad)


# --- Position state round-trip --------------------------------------------------------------


@pytest.mark.parametrize("trail", [None, 2.0])
def test_position_state_round_trips_through_json(trail: float | None) -> None:
    position = Position(
        symbol=BTC,
        strategy="trend",
        side=Side.SHORT,
        regime=Regime.TREND_DOWN,
        qty=0.123456789,
        entry_price=43_210.987654321,
        entry_ts=T_OPEN,
        stop=44_000.1,
        initial_stop=44_500.2,
        target=41_000.3,
        atr_at_entry=321.0987,
        max_hold_bars=96,
        trail_atr_mult=trail,
        bars_held=5,
        best_price=42_900.5,
        fees=2.6672,
        funding=-0.0987,
    )
    state = position.to_state()
    assert state["entry_ts"] == "2024-01-01T00:16:00+00:00"
    assert type(state["side"]) is str and state["side"] == "short"
    assert type(state["regime"]) is str and state["regime"] == "trend_down"
    restored = Position.from_state(json.loads(json.dumps(state)))
    assert restored == position
    assert restored.entry_ts.utcoffset() == timedelta(0)


def test_position_from_state_rejects_naive_timestamp() -> None:
    p = Portfolio(INITIAL)
    state = p.open(make_signal(Side.LONG), make_fill(BUY, 1.0, 100.0, 0.05)).to_state()
    with pytest.raises(ValueError, match="timezone"):
        Position.from_state({**state, "entry_ts": "2024-01-01T00:16:00"})


# --- invariant: once flat, sum(net_pnl) == cash - initial_equity ----------------------------


def spanning_bar(symbol: str, ts: datetime, open_: float, level: float) -> Bar:
    """A bar that trades 1% beyond both its open and ``level``, so any stop or target fills."""
    low, high = min(open_, level) * 0.99, max(open_, level) * 1.01
    return Bar(symbol, ts, open_, high, low, open_, 1_000.0)


def test_random_trades_keep_cash_equal_to_sum_of_net_pnl() -> None:
    """3000 random opens, marks, funding events and exits (market, stop and target) through the
    real fill model; once flat, cash - initial_equity must equal the sum of net_pnl."""
    rng = random.Random(20240101)
    costs = CostsConfig(
        taker_fee_bps=5.0,
        maker_fee_bps=2.0,
        half_spread_bps=1.0,
        slippage_bps=2.0,
        impact_bps_per_pct=1.0,
        max_participation=0.05,
    )
    p = Portfolio(INITIAL)
    trades: list[Trade] = []
    fill_fees = 0.0  # independent ledger: every fee the fill model charged
    exits: Counter[str] = Counter()
    ts = T_OPEN

    def close(symbol: str, kind: str, price: float) -> None:
        nonlocal fill_fees
        position = p.positions[symbol]
        order = SELL if position.side is Side.LONG else BUY
        fill: Fill | None
        if kind == "market":
            bid, ask = quote_from_bar_open(price, costs)
            volume = 0.0 if rng.random() < 0.1 else rng.uniform(10.0, 500.0)
            fill = market_fill(
                symbol,
                ts,
                order,
                position.qty,
                bid,
                ask,
                price,
                volume,
                costs,
                cap_participation=False,
            )
        elif kind == "stop":
            bar = spanning_bar(symbol, ts, price, position.stop)
            fill = stop_fill(symbol, bar, order, position.qty, position.stop, costs)
        else:
            bar = spanning_bar(symbol, ts, price, position.target)
            fill = target_fill(symbol, bar, order, position.qty, position.target, costs)
        assert fill is not None
        reason = {"market": ExitReason.TIME, "stop": ExitReason.STOP}.get(kind, ExitReason.TARGET)
        trades.append(p.close(symbol, fill, reason))
        fill_fees += fill.fee
        exits[kind] += 1

    for _ in range(3000):
        ts += timedelta(minutes=1)
        symbol = rng.choice([BTC, ETH, SOL])
        price = rng.uniform(50.0, 150.0)
        position = p.positions.get(symbol)
        if position is None:
            side = rng.choice([Side.LONG, Side.SHORT])
            order = BUY if side is Side.LONG else SELL
            bid, ask = quote_from_bar_open(price, costs)
            qty, volume = rng.uniform(0.1, 20.0), rng.uniform(50.0, 500.0)
            fill = market_fill(
                symbol, ts, order, qty, bid, ask, price, volume, costs, cap_participation=True
            )
            assert fill is not None
            sign = 1 if side is Side.LONG else -1
            stop, target = fill.price * (1 - 0.05 * sign), fill.price * (1 + 0.1 * sign)
            signal = Signal(
                "breakout", symbol, ts, side, price, stop, target, 1.0, Regime.RANGE, 48, None
            )
            p.open(signal, fill)
            fill_fees += fill.fee
        elif (action := rng.random()) < 0.3:
            p.mark({symbol: price})
        elif action < 0.5:
            p.apply_funding(symbol, price, rng.uniform(-0.001, 0.001))
        else:
            close(symbol, rng.choice(["market", "stop", "target"]), price)
        assert p.drawdown >= 0.0
        assert p.peak_equity >= p.equity

    for symbol in list(p.positions):
        close(symbol, "market", rng.uniform(50.0, 150.0))

    assert p.positions == {}
    assert len(trades) > 500
    assert min(exits[k] for k in ("market", "stop", "target")) > 100
    assert any(t.funding != 0.0 for t in trades)
    assert p.equity == p.cash
    assert sum(t.net_pnl for t in trades) == pytest.approx(p.cash - INITIAL, rel=1e-9, abs=1e-6)
    assert sum(t.fees for t in trades) == pytest.approx(fill_fees, rel=1e-9)
    for t in trades:
        assert t.net_pnl == approx(t.gross_pnl - t.fees - t.funding)
