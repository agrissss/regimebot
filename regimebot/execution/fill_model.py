"""Simulated fills for linear USDT perpetuals: spread, slippage, market impact, fees, funding.

Prices are in quote currency (USDT), quantities in base units, 1 bp = 1e-4 and fees are bps of
executed notional (``qty * price``). Spread and slippage always move the price against the
trader. ``Fill.slippage_bps`` is measured against the fill's ``ref_price`` and is positive when
the fill is worse for the trader (BUY above the reference, SELL below it).

Every function raises ValueError on invalid input rather than return a wrong fill.
"""

import math
from datetime import datetime

from regimebot.config import CostsConfig
from regimebot.types import Bar, Fill, OrderSide, Side, side_sign


def _require_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def _check_bar(bar: Bar, symbol: str) -> None:
    """Stop and target fills read the bar; a malformed or foreign bar would price a wrong fill."""
    if bar.symbol != symbol:
        raise ValueError(f"bar is for {bar.symbol}, not {symbol}")
    for name, value in (("open", bar.open), ("high", bar.high), ("low", bar.low)):
        _require_positive(f"bar.{name}", value)
    if not bar.low <= bar.open <= bar.high:
        raise ValueError(f"bar open {bar.open} outside [low {bar.low}, high {bar.high}]")
    if not math.isfinite(bar.volume):
        raise ValueError(f"bar.volume must be finite, got {bar.volume!r}")


def _make_fill(
    symbol: str,
    ts: datetime,
    side: OrderSide,
    qty: float,
    price: float,
    ref_price: float,
    fee_bps: float,
    liquidity: str,
    quote: tuple[float, float] | None = None,
) -> Fill:
    worse = 1 if side == OrderSide.BUY else -1  # BUY: a higher price is worse
    bid, ask = (None, None) if quote is None else quote
    return Fill(
        symbol=symbol,
        ts=ts,
        side=side,
        qty=qty,
        price=price,
        ref_price=ref_price,
        fee=qty * price * fee_bps / 1e4,
        slippage_bps=worse * (price - ref_price) / ref_price * 1e4,
        liquidity=liquidity,
        bid=bid,
        ask=ask,
    )


def quote_from_bar_open(open_price: float, costs: CostsConfig) -> tuple[float, float]:
    """Synthetic (bid, ask) = ``open_price`` -/+ ``half_spread_bps``, used without a live quote."""
    _require_positive("open_price", open_price)
    half = costs.half_spread_bps / 1e4
    return open_price * (1 - half), open_price * (1 + half)


def market_fill(
    symbol: str,
    ts: datetime,
    side: OrderSide,
    qty: float,
    bid: float,
    ask: float,
    ref_price: float,
    bar_volume: float,
    costs: CostsConfig,
    cap_participation: bool,
) -> Fill | None:
    """Taker fill against a quote that already contains the spread.

    Slippage = ``slippage_bps`` + ``impact_bps_per_pct`` per 1% of ``bar_volume`` taken.
    Entries (``cap_participation``) are cut to ``max_participation * bar_volume`` and get no fill
    (None) on a bar without volume; impact uses the cut qty. Exits are never cut and always
    fill: their participation is ``min(qty / bar_volume, 1)``, and 1 on a bar without volume,
    so an exit never gets cheaper as the bar thins. The fill records ``bid`` and ``ask``.
    """
    side = OrderSide(side)  # a plain "buy"/"sell" is fine; anything else raises ValueError
    for name, value in (("qty", qty), ("bid", bid), ("ask", ask), ("ref_price", ref_price)):
        _require_positive(name, value)
    if bid > ask:
        raise ValueError(f"crossed quote: bid {bid} > ask {ask}")
    if not math.isfinite(bar_volume):
        raise ValueError(f"bar_volume must be finite, got {bar_volume!r}")
    if cap_participation:
        if bar_volume <= 0:
            return None
        qty = min(qty, costs.max_participation * bar_volume)
        participation = qty / bar_volume
    else:
        participation = 1.0 if bar_volume <= 0 else min(qty / bar_volume, 1.0)
    slip_bps = costs.slippage_bps + costs.impact_bps_per_pct * participation * 100
    if side == OrderSide.BUY:
        price = ask * (1 + slip_bps / 1e4)
    else:
        price = bid * (1 - slip_bps / 1e4)
    return _make_fill(
        symbol, ts, side, qty, price, ref_price, costs.taker_fee_bps, "taker", (bid, ask)
    )


def stop_fill(
    symbol: str, bar: Bar, side: OrderSide, qty: float, stop_price: float, costs: CostsConfig
) -> Fill | None:
    """Stop-market exit inside ``bar``, or None if the stop was not reached.

    A SELL stop triggers when ``bar.low <= stop_price`` at ``min(stop_price, bar.open)`` (a gap
    through the stop fills at the open); a BUY stop mirrors this with ``bar.high`` and ``max``.
    The trigger is then priced exactly as an exit ``market_fill`` on a quote built around it:
    half spread, then ``slippage_bps`` plus impact at participation ``min(qty / bar.volume, 1)``
    (1 on a bar without volume), so a stop is never cheaper than a market exit on the same bar.
    Taker fee; ``ref_price`` is the stop level, so a gap shows up as slippage.
    """
    _require_positive("qty", qty)
    _require_positive("stop_price", stop_price)
    _check_bar(bar, symbol)
    side = OrderSide(side)
    if side == OrderSide.SELL:
        if bar.low > stop_price:
            return None
        trigger = min(stop_price, bar.open)
    else:
        if bar.high < stop_price:
            return None
        trigger = max(stop_price, bar.open)
    bid, ask = quote_from_bar_open(trigger, costs)
    return market_fill(
        symbol, bar.ts, side, qty, bid, ask, stop_price, bar.volume, costs, cap_participation=False
    )


def target_fill(
    symbol: str, bar: Bar, side: OrderSide, qty: float, target_price: float, costs: CostsConfig
) -> Fill | None:
    """Resting limit exit inside ``bar`` at exactly ``target_price``, or None if not filled.

    A SELL limit fills when ``bar.high > target_price``; a BUY limit when ``bar.low <
    target_price``. A touch is not a fill (the queue ahead of us is unknown). A bar that opens
    beyond the target still fills at the target: a resting limit trades at its own price and
    no price improvement is assumed. Maker fee, no spread, zero slippage.
    """
    _require_positive("qty", qty)
    _require_positive("target_price", target_price)
    _check_bar(bar, symbol)
    side = OrderSide(side)
    traded_through = bar.high > target_price if side == OrderSide.SELL else bar.low < target_price
    if not traded_through:
        return None
    return _make_fill(
        symbol, bar.ts, side, qty, target_price, target_price, costs.maker_fee_bps, "maker"
    )


def funding_payment(side: Side, qty: float, mark_price: float, rate: float) -> float:
    """Cost of one funding event to a position, positive = paid.

    Longs pay ``qty * mark_price * rate`` when the rate is positive and shorts receive it; a
    negative rate reverses both.
    """
    _require_positive("qty", qty)
    _require_positive("mark_price", mark_price)
    if not math.isfinite(rate):
        raise ValueError(f"funding rate must be finite, got {rate!r}")
    return side_sign(side) * qty * mark_price * rate
