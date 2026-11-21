"""Fill model: every branch checked against hand-computed prices, fees and slippage.

Costs used throughout (1 bp = 1e-4): taker fee 5 bps, maker fee 2 bps, half spread 1 bp,
fixed slippage 2 bps, impact 1 bp per 1% of bar volume taken, max participation 5%.
Expected numbers are worked out by hand in the comments; tolerances are tight (rel 1e-9) so
even a convention change (e.g. adding spread and slippage instead of compounding) would fail.
"""

import math
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from regimebot.config import CostsConfig
from regimebot.execution.fill_model import (
    funding_payment,
    market_fill,
    quote_from_bar_open,
    stop_fill,
    target_fill,
)
from regimebot.types import Bar, Fill, OrderSide, Side

COSTS = CostsConfig(
    taker_fee_bps=5.0,
    maker_fee_bps=2.0,
    half_spread_bps=1.0,
    slippage_bps=2.0,
    impact_bps_per_pct=1.0,
    max_participation=0.05,
)
SYM = "BTC/USDT:USDT"
TS = datetime(2024, 1, 1, 0, 1, tzinfo=UTC)
BID, ASK = 99.99, 100.01  # quote around an open of 100 with a 1 bp half spread
BUY, SELL = OrderSide.BUY, OrderSide.SELL


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def make_bar(open_: float, high: float, low: float) -> Bar:
    return Bar(SYM, TS, open_, high, low, open_, 1000.0)


def check(
    fill: Fill | None,
    side: OrderSide,
    qty: float,
    price: float,
    fee: float,
    slippage_bps: float,
    liquidity: str,
    ref_price: float,
    ts: datetime = TS,
) -> None:
    assert fill is not None
    assert (fill.symbol, fill.ts, fill.side, fill.liquidity) == (SYM, ts, side, liquidity)
    assert fill.qty == approx(qty)
    assert fill.price == approx(price)
    assert fill.ref_price == ref_price
    assert fill.fee == approx(fee)
    assert fill.slippage_bps == approx(slippage_bps)


def test_quote_from_bar_open_puts_half_spread_each_side() -> None:
    # bid = 100 * (1 - 1e-4) = 99.99 ; ask = 100 * (1 + 1e-4) = 100.01
    bid, ask = quote_from_bar_open(100.0, COSTS)
    assert bid == approx(99.99)
    assert ask == approx(100.01)


# --- market_fill ----------------------------------------------------------------------------


def test_market_buy_pays_ask_plus_fixed_slippage_and_impact() -> None:
    # participation = 1 / 100 = 1% -> impact = 1 bp; slippage = 2 + 1 = 3 bps
    # price = 100.01 * 1.0003 = 100.040003
    # fee = 1 * 100.040003 * 5e-4 = 0.0500200015
    # slippage vs ref 100 = (100.040003 - 100) / 100 * 1e4 = +4.0003 bps (worse: paid more)
    fill = market_fill(SYM, TS, BUY, 1.0, BID, ASK, 100.0, 100.0, COSTS, cap_participation=True)
    check(fill, BUY, 1.0, 100.040003, 0.0500200015, 4.0003, "taker", 100.0)


def test_market_sell_receives_bid_minus_fixed_slippage_and_impact() -> None:
    # participation = 2 / 400 = 0.5% -> impact = 0.5 bp; slippage = 2.5 bps
    # price = 99.99 * (1 - 0.00025) = 99.9650025
    # fee = 2 * 99.9650025 * 5e-4 = 0.0999650025
    # slippage vs ref 100 = (100 - 99.9650025) / 100 * 1e4 = +3.49975 bps (worse: got less)
    fill = market_fill(SYM, TS, SELL, 2.0, BID, ASK, 100.0, 400.0, COSTS, cap_participation=False)
    check(fill, SELL, 2.0, 99.9650025, 0.0999650025, 3.49975, "taker", 100.0)


def test_market_fill_better_than_ref_has_negative_slippage() -> None:
    # Price fell from the decision close (ref 100) to the next open (98) before a BUY.
    # quote: ask = 98 * 1.0001 = 98.0098; participation 1% -> 3 bps
    # price = 98.0098 * 1.0003 = 98.03920294 ; fee = 98.03920294 * 5e-4 = 0.04901960147
    # slippage vs ref 100 = (98.03920294 - 100) / 100 * 1e4 = -196.079706 bps (better)
    bid, ask = quote_from_bar_open(98.0, COSTS)
    fill = market_fill(SYM, TS, BUY, 1.0, bid, ask, 100.0, 100.0, COSTS, cap_participation=True)
    check(fill, BUY, 1.0, 98.03920294, 0.04901960147, -196.079706, "taker", 100.0)


def test_entry_is_cut_to_max_participation_and_impact_uses_cut_qty() -> None:
    # Asked 10 on a 100-volume bar: cut to 0.05 * 100 = 5.
    # participation after the cut = 5 / 100 = 5% -> impact 5 bps (not 10); slippage = 7 bps
    # price = 100.01 * 1.0007 = 100.080007 ; fee = 5 * 100.080007 * 5e-4 = 0.2502000175
    # slippage vs ref 100 = 8.0007 bps
    fill = market_fill(SYM, TS, BUY, 10.0, BID, ASK, 100.0, 100.0, COSTS, cap_participation=True)
    check(fill, BUY, 5.0, 100.080007, 0.2502000175, 8.0007, "taker", 100.0)


def test_exit_is_never_cut_and_pays_full_impact() -> None:
    # SELL 10 on a 100-volume bar without the cap: qty stays 10, participation 10%
    # impact = 10 bps; slippage = 12 bps ; price = 99.99 * 0.9988 = 99.870012
    # fee = 10 * 99.870012 * 5e-4 = 0.49935006 ; slippage vs ref 100 = 12.9988 bps
    fill = market_fill(SYM, TS, SELL, 10.0, BID, ASK, 100.0, 100.0, COSTS, cap_participation=False)
    check(fill, SELL, 10.0, 99.870012, 0.49935006, 12.9988, "taker", 100.0)


def test_zero_volume_bar_gives_no_entry() -> None:
    fill = market_fill(SYM, TS, BUY, 1.0, BID, ASK, 100.0, 0.0, COSTS, cap_participation=True)
    assert fill is None


def test_zero_volume_exit_charges_impact_at_max_participation() -> None:
    # participation taken as max_participation = 5% -> impact 5 bps; slippage = 7 bps
    # price = 99.99 * 0.9993 = 99.920007 ; fee = 3 * 99.920007 * 5e-4 = 0.1498800105
    # slippage vs ref 100 = 7.9993 bps
    fill = market_fill(SYM, TS, SELL, 3.0, BID, ASK, 100.0, 0.0, COSTS, cap_participation=False)
    check(fill, SELL, 3.0, 99.920007, 0.1498800105, 7.9993, "taker", 100.0)


def test_impact_term_alone() -> None:
    # No fixed slippage, no spread (bid = ask = 100), impact 2 bps per 1% of volume.
    # participation = 2 / 100 = 2% -> impact = 2 * 2 = 4 bps ; price = 100 * 1.0004 = 100.04
    # fee = 2 * 100.04 * 5e-4 = 0.10004 ; slippage vs ref 100 = 4 bps
    costs = COSTS.model_copy(update={"slippage_bps": 0.0, "impact_bps_per_pct": 2.0})
    fill = market_fill(SYM, TS, BUY, 2.0, 100.0, 100.0, 100.0, 100.0, costs, cap_participation=True)
    check(fill, BUY, 2.0, 100.04, 0.10004, 4.0, "taker", 100.0)


@pytest.mark.parametrize(
    ("qty", "bid", "ask", "ref", "volume"),
    [
        (0.0, BID, ASK, 100.0, 100.0),  # zero qty
        (-1.0, BID, ASK, 100.0, 100.0),  # negative qty
        (math.nan, BID, ASK, 100.0, 100.0),  # NaN qty
        (1.0, 100.02, 100.01, 100.0, 100.0),  # crossed quote: bid > ask
        (1.0, 0.0, ASK, 100.0, 100.0),  # zero bid
        (1.0, BID, math.inf, 100.0, 100.0),  # infinite ask
        (1.0, BID, ASK, 0.0, 100.0),  # zero reference price
        (1.0, BID, ASK, 100.0, math.nan),  # NaN volume
    ],
)
def test_market_fill_rejects_invalid_inputs(
    qty: float, bid: float, ask: float, ref: float, volume: float
) -> None:
    with pytest.raises(ValueError):
        market_fill(SYM, TS, BUY, qty, bid, ask, ref, volume, COSTS, cap_participation=False)


def test_market_sell_with_slippage_beyond_100_percent_raises() -> None:
    # SELL 1000 on a 1-volume bar: participation 1000 -> impact 100000 bps -> price would be < 0
    with pytest.raises(ValueError, match="non-positive"):
        market_fill(SYM, TS, SELL, 1000.0, BID, ASK, 100.0, 1.0, COSTS, cap_participation=False)


# --- stop_fill: half spread + fixed slippage against the trader, multiplied ------------------
# SELL: trigger * (1 - 1e-4) * (1 - 2e-4) = trigger * 0.99970002
# BUY:  trigger * (1 + 1e-4) * (1 + 2e-4) = trigger * 1.00030002


def test_sell_stop_touched_exactly_fills_at_stop_less_costs() -> None:
    # low == stop == 100 triggers; open 105 is above, so trigger = min(100, 105) = 100
    # price = 100 * 0.99970002 = 99.970002 ; fee = 2 * 99.970002 * 5e-4 = 0.099970002
    # slippage vs stop 100 = (100 - 99.970002) / 100 * 1e4 = 2.9998 bps
    fill = stop_fill(SYM, make_bar(105.0, 106.0, 100.0), SELL, 2.0, 100.0, COSTS)
    check(fill, SELL, 2.0, 99.970002, 0.099970002, 2.9998, "taker", 100.0)


def test_sell_stop_gapped_through_fills_at_open() -> None:
    # bar opens at 97, below the 100 stop: trigger = min(100, 97) = 97
    # price = 97 * 0.99970002 = 96.97090194 ; fee = 96.97090194 * 5e-4 = 0.04848545097
    # slippage vs stop 100 = (100 - 96.97090194) / 100 * 1e4 = 302.909806 bps
    fill = stop_fill(SYM, make_bar(97.0, 98.0, 96.0), SELL, 1.0, 100.0, COSTS)
    check(fill, SELL, 1.0, 96.97090194, 0.04848545097, 302.909806, "taker", 100.0)


def test_sell_stop_not_touched() -> None:
    assert stop_fill(SYM, make_bar(102.0, 103.0, 100.5), SELL, 1.0, 100.0, COSTS) is None


def test_buy_stop_touched_exactly_fills_at_stop_plus_costs() -> None:
    # high == stop == 100 triggers; open 95 is below, so trigger = max(100, 95) = 100
    # price = 100 * 1.00030002 = 100.030002 ; fee = 2 * 100.030002 * 5e-4 = 0.100030002
    # slippage vs stop 100 = 3.0002 bps
    fill = stop_fill(SYM, make_bar(95.0, 100.0, 94.0), BUY, 2.0, 100.0, COSTS)
    check(fill, BUY, 2.0, 100.030002, 0.100030002, 3.0002, "taker", 100.0)


def test_buy_stop_gapped_through_fills_at_open() -> None:
    # bar opens at 103, above the 100 stop: trigger = max(100, 103) = 103
    # price = 103 * 1.00030002 = 103.03090206 ; fee = 103.03090206 * 5e-4 = 0.05151545103
    # slippage vs stop 100 = 303.090206 bps
    fill = stop_fill(SYM, make_bar(103.0, 104.0, 102.0), BUY, 1.0, 100.0, COSTS)
    check(fill, BUY, 1.0, 103.03090206, 0.05151545103, 303.090206, "taker", 100.0)


def test_buy_stop_not_touched() -> None:
    assert stop_fill(SYM, make_bar(98.0, 99.5, 97.0), BUY, 1.0, 100.0, COSTS) is None


# --- target_fill: resting limit, maker fee, no spread or slippage ---------------------------


def test_sell_target_touch_is_not_a_fill() -> None:
    assert target_fill(SYM, make_bar(105.0, 110.0, 104.0), SELL, 1.0, 110.0, COSTS) is None


def test_sell_target_traded_through_fills_at_target() -> None:
    # high 110.5 > 110: fills at exactly 110 ; fee = 2 * 110 * 2e-4 = 0.044 ; slippage 0
    fill = target_fill(SYM, make_bar(105.0, 110.5, 104.0), SELL, 2.0, 110.0, COSTS)
    check(fill, SELL, 2.0, 110.0, 0.044, 0.0, "maker", 110.0)


def test_sell_target_gap_open_beyond_fills_at_open() -> None:
    # opens at 112 above the 110 target: fills at 112 ; fee = 112 * 2e-4 = 0.0224
    # slippage vs 110 = (110 - 112) / 110 * 1e4 = -181.8181... bps (better than the target)
    fill = target_fill(SYM, make_bar(112.0, 113.0, 111.0), SELL, 1.0, 110.0, COSTS)
    check(fill, SELL, 1.0, 112.0, 0.0224, -2 / 110 * 1e4, "maker", 110.0)


def test_buy_target_touch_is_not_a_fill() -> None:
    assert target_fill(SYM, make_bar(95.0, 96.0, 90.0), BUY, 1.0, 90.0, COSTS) is None


def test_buy_target_traded_through_fills_at_target() -> None:
    # low 89.5 < 90: fills at exactly 90 ; fee = 2 * 90 * 2e-4 = 0.036 ; slippage 0
    fill = target_fill(SYM, make_bar(95.0, 96.0, 89.5), BUY, 2.0, 90.0, COSTS)
    check(fill, BUY, 2.0, 90.0, 0.036, 0.0, "maker", 90.0)


def test_buy_target_gap_open_beyond_fills_at_open() -> None:
    # opens at 88 below the 90 target: fills at 88 ; fee = 88 * 2e-4 = 0.0176
    # slippage vs 90 = (88 - 90) / 90 * 1e4 = -222.2222... bps (better than the target)
    fill = target_fill(SYM, make_bar(88.0, 89.0, 87.0), BUY, 1.0, 90.0, COSTS)
    check(fill, BUY, 1.0, 88.0, 0.0176, -2 / 90 * 1e4, "maker", 90.0)


@pytest.mark.parametrize("fill_fn", [stop_fill, target_fill])
@pytest.mark.parametrize(
    ("bar", "qty", "level"),
    [
        (make_bar(100.0, 101.0, 99.0), 0.0, 100.0),  # zero qty
        (make_bar(100.0, 101.0, 99.0), 1.0, 0.0),  # zero stop/target level
        (make_bar(100.0, 101.0, 99.0), 1.0, math.nan),  # NaN level
        (make_bar(102.0, 101.0, 99.0), 1.0, 100.0),  # open above high
        (make_bar(100.0, 101.0, 0.0), 1.0, 100.0),  # zero low
        (make_bar(math.nan, 101.0, 99.0), 1.0, 100.0),  # NaN open
    ],
)
def test_bar_fills_reject_invalid_inputs(
    fill_fn: Callable[..., Fill | None], bar: Bar, qty: float, level: float
) -> None:
    with pytest.raises(ValueError):
        fill_fn(SYM, bar, SELL, qty, level, COSTS)


# --- funding_payment ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "rate", "expected"),
    [
        # notional = 2 * 100 = 200 ; |payment| = 200 * 0.0001 = 0.02
        (Side.LONG, 0.0001, 0.02),  # longs pay a positive rate
        (Side.SHORT, 0.0001, -0.02),  # shorts receive a positive rate
        (Side.LONG, -0.0001, -0.02),  # longs receive a negative rate
        (Side.SHORT, -0.0001, 0.02),  # shorts pay a negative rate
    ],
)
def test_funding_payment_signs(side: Side, rate: float, expected: float) -> None:
    assert funding_payment(side, 2.0, 100.0, rate) == approx(expected)


@pytest.mark.parametrize(
    ("qty", "mark", "rate"), [(0.0, 100.0, 0.0001), (1.0, -1.0, 0.0001), (1.0, 100.0, math.nan)]
)
def test_funding_payment_rejects_invalid_inputs(qty: float, mark: float, rate: float) -> None:
    with pytest.raises(ValueError):
        funding_payment(Side.LONG, qty, mark, rate)
