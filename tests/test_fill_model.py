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


def make_bar(
    open_: float, high: float, low: float, volume: float = 1000.0, symbol: str = SYM
) -> Bar:
    return Bar(symbol, TS, open_, high, low, open_, volume)


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


# Exits of qty 5: participation for impact = min(qty / volume, 1), and 1 when volume <= 0.
#   volume 0    -> participation 1                 -> impact 100 bps -> slippage 102 bps
#   volume 0.05 -> 5 / 0.05 = 100, capped at 1     -> impact 100 bps -> slippage 102 bps
#   volume 10   -> 5 / 10 = 0.5                    -> impact  50 bps -> slippage  52 bps
# SELL @ 102: 99.99 * (1 - 0.0102) = 98.970102 ; fee 5 * 98.970102 * 5e-4 = 0.247425255
#             vs ref 100: (100 - 98.970102) / 100 * 1e4 = 102.9898 bps
# SELL @ 52:  99.99 * (1 - 0.0052) = 99.470052 ; fee 5 * 99.470052 * 5e-4 = 0.24867513
#             vs ref 100: 52.9948 bps
# BUY @ 102:  100.01 * 1.0102 = 101.030102 ; fee 5 * 101.030102 * 5e-4 = 0.252575255
#             vs ref 100: 103.0102 bps
# BUY @ 52:   100.01 * 1.0052 = 100.530052 ; fee 5 * 100.530052 * 5e-4 = 0.25132513
#             vs ref 100: 53.0052 bps
@pytest.mark.parametrize(
    ("side", "volume", "price", "fee", "slippage"),
    [
        (SELL, 0.0, 98.970102, 0.247425255, 102.9898),
        (SELL, 0.05, 98.970102, 0.247425255, 102.9898),
        (SELL, 10.0, 99.470052, 0.24867513, 52.9948),
        (BUY, 0.0, 101.030102, 0.252575255, 103.0102),
        (BUY, 0.05, 101.030102, 0.252575255, 103.0102),
        (BUY, 10.0, 100.530052, 0.25132513, 53.0052),
    ],
)
def test_thin_volume_exit_impact_is_bounded_at_full_participation(
    side: OrderSide, volume: float, price: float, fee: float, slippage: float
) -> None:
    fill = market_fill(SYM, TS, side, 5.0, BID, ASK, 100.0, volume, COSTS, cap_participation=False)
    check(fill, side, 5.0, price, fee, slippage, "taker", 100.0)


@pytest.mark.parametrize("side", [BUY, SELL])
def test_exit_never_gets_cheaper_as_volume_falls(side: OrderSide) -> None:
    volumes = [1e6, 1000.0, 100.0, 10.0, 5.0, 1.0, 1e-9, 0.0]  # falling
    slippages = []
    for volume in volumes:
        fill = market_fill(SYM, TS, side, 5.0, BID, ASK, 100.0, volume, COSTS, False)
        assert fill is not None
        slippages.append(fill.slippage_bps)
    assert slippages == sorted(slippages)  # non-decreasing cost as the bar thins


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


# --- stop_fill: trigger, then half spread, then slippage + bounded impact from bar.volume ----
# Participation 1% (qty 2 on volume 200, or qty 1 on volume 100) -> impact 1 bp -> 3 bps.
# SELL: trigger * (1 - 1e-4) * (1 - 3e-4) ; BUY: trigger * (1 + 1e-4) * (1 + 3e-4)


def test_sell_stop_touched_exactly_fills_at_stop_less_costs() -> None:
    # low == stop == 100 triggers; open 105 is above, so trigger = min(100, 105) = 100
    # bid = 100 * 0.9999 = 99.99 ; price = 99.99 * 0.9997 = 99.960003
    # fee = 2 * 99.960003 * 5e-4 = 0.099960003
    # slippage vs stop 100 = (100 - 99.960003) / 100 * 1e4 = 3.9997 bps
    fill = stop_fill(SYM, make_bar(105.0, 106.0, 100.0, volume=200.0), SELL, 2.0, 100.0, COSTS)
    check(fill, SELL, 2.0, 99.960003, 0.099960003, 3.9997, "taker", 100.0)


def test_sell_stop_gapped_through_fills_at_open() -> None:
    # bar opens at 97, below the 100 stop: trigger = min(100, 97) = 97
    # bid = 97 * 0.9999 = 96.9903 ; price = 96.9903 * 0.9997 = 96.96120291
    # fee = 96.96120291 * 5e-4 = 0.048480601455
    # slippage vs stop 100 = (100 - 96.96120291) / 100 * 1e4 = 303.879709 bps
    fill = stop_fill(SYM, make_bar(97.0, 98.0, 96.0, volume=100.0), SELL, 1.0, 100.0, COSTS)
    check(fill, SELL, 1.0, 96.96120291, 0.048480601455, 303.879709, "taker", 100.0)


def test_sell_stop_not_touched() -> None:
    assert stop_fill(SYM, make_bar(102.0, 103.0, 100.5), SELL, 1.0, 100.0, COSTS) is None


def test_buy_stop_touched_exactly_fills_at_stop_plus_costs() -> None:
    # high == stop == 100 triggers; open 95 is below, so trigger = max(100, 95) = 100
    # ask = 100 * 1.0001 = 100.01 ; price = 100.01 * 1.0003 = 100.040003
    # fee = 2 * 100.040003 * 5e-4 = 0.100040003 ; slippage vs stop 100 = 4.0003 bps
    fill = stop_fill(SYM, make_bar(95.0, 100.0, 94.0, volume=200.0), BUY, 2.0, 100.0, COSTS)
    check(fill, BUY, 2.0, 100.040003, 0.100040003, 4.0003, "taker", 100.0)


def test_buy_stop_gapped_through_fills_at_open() -> None:
    # bar opens at 103, above the 100 stop: trigger = max(100, 103) = 103
    # ask = 103 * 1.0001 = 103.0103 ; price = 103.0103 * 1.0003 = 103.04120309
    # fee = 103.04120309 * 5e-4 = 0.051520601545 ; slippage vs stop 100 = 304.120309 bps
    fill = stop_fill(SYM, make_bar(103.0, 104.0, 102.0, volume=100.0), BUY, 1.0, 100.0, COSTS)
    check(fill, BUY, 1.0, 103.04120309, 0.051520601545, 304.120309, "taker", 100.0)


# Stops of qty 2 at 100 (touched exactly) on thin bars:
#   volume 0 or 1e-6 -> participation 1 -> slippage 2 + 100 = 102 bps
#   volume 4         -> 2 / 4 = 0.5     -> slippage 2 + 50  =  52 bps
# SELL: 99.99 * 0.9898 = 98.970102 (fee 2 * 98.970102 * 5e-4 = 0.098970102, 102.9898 bps)
#       99.99 * 0.9948 = 99.470052 (fee 0.099470052, 52.9948 bps)
# BUY:  100.01 * 1.0102 = 101.030102 (fee 2 * 101.030102 * 5e-4 = 0.101030102, 103.0102 bps)
#       100.01 * 1.0052 = 100.530052 (fee 0.100530052, 53.0052 bps)
# These equal a market exit of the same qty and volume on the stop's quote: never cheaper.
@pytest.mark.parametrize(
    ("side", "bar_ohl", "volume", "price", "fee", "slippage"),
    [
        (SELL, (105.0, 106.0, 100.0), 0.0, 98.970102, 0.098970102, 102.9898),
        (SELL, (105.0, 106.0, 100.0), 1e-6, 98.970102, 0.098970102, 102.9898),
        (SELL, (105.0, 106.0, 100.0), 4.0, 99.470052, 0.099470052, 52.9948),
        (BUY, (95.0, 100.0, 94.0), 0.0, 101.030102, 0.101030102, 103.0102),
        (BUY, (95.0, 100.0, 94.0), 1e-6, 101.030102, 0.101030102, 103.0102),
        (BUY, (95.0, 100.0, 94.0), 4.0, 100.530052, 0.100530052, 53.0052),
    ],
)
def test_stop_on_thin_bar_pays_bounded_impact(
    side: OrderSide,
    bar_ohl: tuple[float, float, float],
    volume: float,
    price: float,
    fee: float,
    slippage: float,
) -> None:
    fill = stop_fill(SYM, make_bar(*bar_ohl, volume=volume), side, 2.0, 100.0, COSTS)
    check(fill, side, 2.0, price, fee, slippage, "taker", 100.0)


def test_buy_stop_not_touched() -> None:
    assert stop_fill(SYM, make_bar(98.0, 99.5, 97.0), BUY, 1.0, 100.0, COSTS) is None


# --- target_fill: resting limit, maker fee, no spread or slippage ---------------------------


def test_sell_target_touch_is_not_a_fill() -> None:
    assert target_fill(SYM, make_bar(105.0, 110.0, 104.0), SELL, 1.0, 110.0, COSTS) is None


def test_sell_target_traded_through_fills_at_target() -> None:
    # high 110.5 > 110: fills at exactly 110 ; fee = 2 * 110 * 2e-4 = 0.044 ; slippage 0
    fill = target_fill(SYM, make_bar(105.0, 110.5, 104.0), SELL, 2.0, 110.0, COSTS)
    check(fill, SELL, 2.0, 110.0, 0.044, 0.0, "maker", 110.0)


def test_sell_target_gap_open_beyond_still_fills_at_target() -> None:
    # opens at 112 above the 110 target: a resting limit trades at its own price, so 110
    # fee = 1 * 110 * 2e-4 = 0.022 ; slippage 0 (no price improvement assumed)
    fill = target_fill(SYM, make_bar(112.0, 113.0, 111.0), SELL, 1.0, 110.0, COSTS)
    check(fill, SELL, 1.0, 110.0, 0.022, 0.0, "maker", 110.0)


def test_buy_target_touch_is_not_a_fill() -> None:
    assert target_fill(SYM, make_bar(95.0, 96.0, 90.0), BUY, 1.0, 90.0, COSTS) is None


def test_buy_target_traded_through_fills_at_target() -> None:
    # low 89.5 < 90: fills at exactly 90 ; fee = 2 * 90 * 2e-4 = 0.036 ; slippage 0
    fill = target_fill(SYM, make_bar(95.0, 96.0, 89.5), BUY, 2.0, 90.0, COSTS)
    check(fill, BUY, 2.0, 90.0, 0.036, 0.0, "maker", 90.0)


def test_buy_target_gap_open_beyond_still_fills_at_target() -> None:
    # opens at 88 below the 90 target: fills at its own price 90
    # fee = 1 * 90 * 2e-4 = 0.018 ; slippage 0
    fill = target_fill(SYM, make_bar(88.0, 89.0, 87.0), BUY, 1.0, 90.0, COSTS)
    check(fill, BUY, 1.0, 90.0, 0.018, 0.0, "maker", 90.0)


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
        (make_bar(100.0, 101.0, 99.0, volume=math.nan), 1.0, 100.0),  # NaN volume
        (make_bar(100.0, 101.0, 99.0, symbol="ETH/USDT:USDT"), 1.0, 100.0),  # other symbol
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
