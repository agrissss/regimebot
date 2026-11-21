"""Risk manager: rule order, sizing, exposure caps, streaks, loss limits, kill switch, state.

Portfolios are built through the real API with fee-free fills, so equity stays exactly 10 000
until a mark moves it. Expected sizes are hand-computed in the comments. Default risk config:
risk_per_trade 0.005, max_gross_leverage 1.5, max_asset_exposure 0.75, max_group_exposure 1.0,
groups {majors: [BTC, ETH, SOL]}, daily_loss 0.015, weekly_loss 0.04, max_drawdown 0.10,
streak_half 3, streak_stop 5. 2024-01-01 is a Monday.
"""

import json
import random
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

import regimebot
from regimebot import cli
from regimebot.config import AppConfig, PlaybookEntry, RiskConfig, load_config
from regimebot.engine.portfolio import Portfolio
from regimebot.risk.manager import DEFAULT_KILL_FILE, MIN_NOTIONAL, RiskManager
from regimebot.types import (
    ExitReason,
    Fill,
    OrderSide,
    Regime,
    RiskDecision,
    RiskEvent,
    Side,
    Signal,
    Trade,
    side_sign,
)

INITIAL = 10_000.0
BTC, ETH, SOL, XRP, DOGE = "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP", "DOGE"
TWO_GROUPS = {"majors": [BTC, ETH, SOL], "alts": [SOL, XRP]}  # SOL in both, DOGE in none
MON = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)
BASE_CFG = load_config()
PLAY = PlaybookEntry(
    strategies=["breakout", "trend"], size_mult=1.0, allow_entries=True, force_close=False
)


def approx(x: float) -> object:
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def make_cfg(**risk: object) -> AppConfig:
    """Default config with some risk fields replaced (validated by RiskConfig)."""
    merged = RiskConfig.model_validate({**BASE_CFG.risk.model_dump(), **risk})
    return BASE_CFG.model_copy(update={"risk": merged})


def manager(tmp_path: Path, **risk: object) -> RiskManager:
    return RiskManager(make_cfg(**risk), tmp_path / "KILL")


def make_signal(
    symbol: str = BTC,
    side: Side = Side.LONG,
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 115.0,
    strategy: str = "breakout",
) -> Signal:
    return Signal(strategy, symbol, MON, side, entry, stop, target, 2.0, Regime.TREND_UP, 48, None)


def fill(symbol: str, order: OrderSide, qty: float, price: float) -> Fill:
    return Fill(symbol, MON, order, qty, price, price, 0.0, 0.0, "taker")  # at signal ts


def open_position(p: Portfolio, symbol: str, side: Side, qty: float, price: float) -> None:
    """Open a fee-free position through the real Portfolio API (marked at ``price``)."""
    sign = side_sign(side)
    signal = make_signal(symbol, side, price, price * (1 - 0.1 * sign), price * (1 + 0.2 * sign))
    p.open(signal, fill(symbol, OrderSide.BUY if sign > 0 else OrderSide.SELL, qty, price))


def book(*longs: tuple[str, float, float]) -> Portfolio:
    """Portfolio(10 000) holding fee-free longs given as (symbol, qty, price)."""
    p = Portfolio(INITIAL)
    for symbol, qty, price in longs:
        open_position(p, symbol, Side.LONG, qty, price)
    return p


def make_trade(net_pnl: float, exit_ts: datetime = MON) -> Trade:
    """Closed 1-unit long, fee-free: the risk manager reads only net_pnl and exit_ts."""
    return Trade(
        BTC,
        "breakout",
        Side.LONG,
        Regime.TREND_UP,
        1.0,
        exit_ts - timedelta(hours=1),
        exit_ts,
        100.0,
        100.0 + net_pnl,
        95.0,
        110.0,
        net_pnl,
        0.0,
        0.0,
        net_pnl,
        ExitReason.STOP,
        net_pnl / 5.0,
        4,
    )


def rejected(decision: RiskDecision) -> str:
    """Assert ``decision`` is a rejection with zero qty and return its rule."""
    assert not decision.approved
    assert decision.qty == 0.0
    return decision.rule


def approved(decision: RiskDecision) -> tuple[str, float]:
    assert decision.approved, decision
    return decision.rule, decision.qty


# --- rule order -----------------------------------------------------------------------------


def test_rules_apply_in_order_and_the_earliest_failure_is_reported(tmp_path: Path) -> None:
    """Start with every rule failing, fix them one at a time: each step reports the next rule."""
    rm = manager(tmp_path)
    kill = tmp_path / "KILL"
    kill.touch()  # kill_switch
    p = book((DOGE, 200.0, 100.0), (BTC, 1.0, 100.0))
    p.mark({DOGE: 40.0})  # no_equity: 10000 + 200 * (40 - 100) = -2000
    for minute in range(5):  # day_stopped: streak_stop = 5 losses on Monday
        rm.on_trade_closed(make_trade(-10.0, MON - timedelta(minutes=minute)))
    play = PLAY.model_copy(update={"allow_entries": False, "strategies": ["trend"]})
    signal = make_signal(BTC, stop=101.0)  # position_exists (BTC open) and invalid_stop
    now = MON

    assert rejected(rm.evaluate(signal, p, play, now)) == "kill_switch"
    kill.unlink()
    assert rejected(rm.evaluate(signal, p, play, now)) == "no_equity"
    p.close(DOGE, fill(DOGE, OrderSide.SELL, 200.0, 100.0), ExitReason.TIME)  # cash back 10000
    assert rejected(rm.evaluate(signal, p, play, now)) == "day_stopped"
    now += DAY
    assert rejected(rm.evaluate(signal, p, play, now)) == "regime_blocks_entries"
    play = play.model_copy(update={"allow_entries": True})
    assert rejected(rm.evaluate(signal, p, play, now)) == "strategy_not_allowed"
    play = play.model_copy(update={"strategies": ["trend", "breakout"]})
    assert rejected(rm.evaluate(signal, p, play, now)) == "position_exists"
    signal = replace(signal, symbol=ETH)
    assert rejected(rm.evaluate(signal, p, play, now)) == "invalid_stop"
    signal = replace(signal, stop=95.0)
    # risk 10000 * 0.005 = 50 ; qty 50 / 5 = 10 ; notional 1000 under every cap
    assert approved(rm.evaluate(signal, p, play, now)) == ("ok", approx(10.0))


@pytest.mark.parametrize(
    ("side", "entry", "stop", "target"),
    [
        (Side.LONG, 100.0, 101.0, 115.0),  # long stop above entry
        (Side.LONG, 100.0, 100.0, 115.0),  # zero stop distance
        (Side.SHORT, 100.0, 99.0, 85.0),  # short stop below entry
        (Side.SHORT, 100.0, 100.0, 85.0),
        (Side.LONG, float("nan"), 95.0, 115.0),
        (Side.LONG, 100.0, float("-inf"), 115.0),
        (Side.SHORT, 100.0, float("inf"), 85.0),
        (Side.LONG, 100.0, 95.0, float("nan")),
        (Side.LONG, -100.0, -105.0, 115.0),  # prices must be positive
        (Side.LONG, 100.0, 0.0, 115.0),
    ],
)
def test_invalid_stop_is_rejected(
    tmp_path: Path, side: Side, entry: float, stop: float, target: float
) -> None:
    rm = manager(tmp_path)
    decision = rm.evaluate(
        make_signal(side=side, entry=entry, stop=stop, target=target), book(), PLAY, MON
    )
    assert rejected(decision) == "invalid_stop"


@pytest.mark.parametrize("mark", [50.0, 40.0])  # equity 10000 + 200 * (mark - 100): 0, -2000
def test_non_positive_equity_is_rejected(tmp_path: Path, mark: float) -> None:
    p = book((DOGE, 200.0, 100.0))
    p.mark({DOGE: mark})
    assert rejected(manager(tmp_path).evaluate(make_signal(), p, PLAY, MON)) == "no_equity"


# --- sizing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("side", "stop", "qty", "exit_order"),
    [
        (Side.LONG, 95.0, 10.0, OrderSide.SELL),  # 50 / |100 - 95| = 10
        (Side.SHORT, 104.0, 12.5, OrderSide.BUY),  # 50 / |100 - 104| = 12.5
    ],
)
def test_stop_hit_loses_exactly_risk_per_trade_of_equity(
    tmp_path: Path, side: Side, stop: float, qty: float, exit_order: OrderSide
) -> None:
    # risk = 10000 * 0.005 = 50 ; notional 1000 / 1250 is under every cap -> "ok"
    p = book()
    signal = make_signal(side=side, stop=stop, target=100.0 - 15.0 * side_sign(side))
    decision = manager(tmp_path).evaluate(signal, p, PLAY, MON)
    assert approved(decision) == ("ok", approx(qty))
    # trade it through the portfolio, fee-free: the stop fill loses 50 = 0.5% of 10000
    p.open(signal, fill(BTC, OrderSide.BUY if side is Side.LONG else OrderSide.SELL, qty, 100.0))
    trade = p.close(BTC, fill(BTC, exit_order, qty, stop), ExitReason.STOP)
    assert trade.gross_pnl == approx(-0.005 * INITIAL)


def test_sizing_uses_marked_equity(tmp_path: Path) -> None:
    # ETH 20 @ 50 marked at 60: equity = 10000 + 20 * 10 = 10200 ; risk = 10200 * 0.005 = 51
    # BTC entry 100, stop 98: qty = 51 / 2 = 25.5 ; notional 2550
    # rooms: asset 0.75 * 10200 = 7650 ; majors 10200 - 1200 = 9000 ; gross 15300 - 1200 = 14100
    p = book((ETH, 20.0, 50.0))
    p.mark({ETH: 60.0})
    decision = manager(tmp_path).evaluate(make_signal(stop=98.0), p, PLAY, MON)
    assert approved(decision) == ("ok", approx(25.5))


def test_size_mult_and_streak_mult_scale_the_risk(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book()
    half = PLAY.model_copy(update={"size_mult": 0.5})
    # size_mult 0.5: risk = 10000 * 0.005 * 0.5 = 25 ; qty = 25 / 5 = 5
    assert approved(rm.evaluate(make_signal(), p, half, MON)) == ("ok", approx(5.0))
    for _ in range(2):
        rm.on_trade_closed(make_trade(-10.0))
    assert approved(rm.evaluate(make_signal(), p, PLAY, MON)) == ("ok", approx(10.0))  # 2 < 3
    rm.on_trade_closed(make_trade(-10.0))
    # 3 losses >= streak_half: risk = 50 * 0.5 = 25 -> qty 5 ; with size_mult 0.5 -> 2.5
    assert approved(rm.evaluate(make_signal(), p, PLAY, MON)) == ("ok", approx(5.0))
    assert approved(rm.evaluate(make_signal(), p, half, MON)) == ("ok", approx(2.5))


def test_cap_asset_binding_alone(tmp_path: Path) -> None:
    # risk qty = 50 / 0.6 = 83.33 -> notional 8333.33
    # asset room 0.75 * 10000 = 7500 binds ; majors room 10000 and gross room 15000 do not
    decision = manager(tmp_path).evaluate(make_signal(stop=99.4), book(), PLAY, MON)
    assert approved(decision) == ("cap_asset", approx(75.0))  # 7500 / 100


def test_cap_group_binding_alone_with_position_in_another_member(tmp_path: Path) -> None:
    # ETH 70 @ 100 marked at 110: exposure 7700 ; equity = 10000 + 70 * 10 = 10700
    # BTC entry 100, stop 99: risk = 10700 * 0.005 = 53.5 -> qty 53.5, notional 5350
    # rooms: asset 0.75 * 10700 = 8025 ; majors 10700 - 7700 = 3000 ; gross 16050 - 7700 = 8350
    p = book((ETH, 70.0, 100.0))
    p.mark({ETH: 110.0})
    decision = manager(tmp_path).evaluate(make_signal(stop=99.0), p, PLAY, MON)
    assert approved(decision) == ("cap_group", approx(30.0))  # 3000 / 100
    assert "majors" in decision.detail


def test_cap_gross_binding_alone(tmp_path: Path) -> None:
    # DOGE (no group) 100 @ 120: exposure 12000 ; gross room = 15000 - 12000 = 3000
    # BTC risk qty 50 / 1 = 50 -> notional 5000 ; asset room 7500 ; majors room 10000
    p = book((DOGE, 100.0, 120.0))
    decision = manager(tmp_path).evaluate(make_signal(stop=99.0), p, PLAY, MON)
    assert approved(decision) == ("cap_gross", approx(30.0))  # 3000 / 100


@pytest.mark.parametrize(
    ("longs", "rule", "qty"),
    [
        # BTC risk notional = 50 / 0.1 * 100 = 50000 ; every cap below reduces it
        # ETH 7000: asset 7500, majors 10000 - 7000 = 3000, gross 15000 - 7000 = 8000
        ([(ETH, 70.0, 100.0)], "cap_group", 30.0),
        # + DOGE 6000: majors 3000, gross 15000 - 13000 = 2000 is smallest
        ([(ETH, 70.0, 100.0), (DOGE, 60.0, 100.0)], "cap_gross", 20.0),
        # no positions: asset 7500 < majors 10000 < gross 15000
        ([], "cap_asset", 75.0),
    ],
)
def test_when_several_caps_bind_the_smallest_names_the_rule(
    tmp_path: Path, longs: list[tuple[str, float, float]], rule: str, qty: float
) -> None:
    decision = manager(tmp_path).evaluate(make_signal(stop=99.9), book(*longs), PLAY, MON)
    assert approved(decision) == (rule, approx(qty))


def test_every_group_containing_the_symbol_is_checked(tmp_path: Path) -> None:
    # groups majors [BTC, ETH, SOL] and alts [SOL, XRP] ; XRP 8000 (alts), ETH 1000 (majors)
    # signal entry 100, stop 99.5: risk qty 50 / 0.5 = 100 -> notional 10000
    # SOL: asset 7500, majors 10000 - 1000 = 9000, alts 10000 - 8000 = 2000, gross 6000
    rm = manager(tmp_path, groups=TWO_GROUPS)
    p = book((XRP, 80.0, 100.0), (ETH, 10.0, 100.0))
    sol = rm.evaluate(make_signal(SOL, stop=99.5), p, PLAY, MON)
    assert approved(sol) == ("cap_group", approx(20.0))
    assert "alts" in sol.detail
    # BTC is only in majors (room 9000): gross room 15000 - 9000 = 6000 binds
    btc = rm.evaluate(make_signal(BTC, stop=99.5), p, PLAY, MON)
    assert approved(btc) == ("cap_gross", approx(60.0))


def test_symbol_in_no_group_has_only_asset_and_gross_caps(tmp_path: Path) -> None:
    # ETH 9500 fills majors (room 500) ; entry 100, stop 99: risk qty 50 -> notional 5000
    p = book((ETH, 95.0, 100.0))
    rm = manager(tmp_path)
    # BTC: majors room 10000 - 9500 = 500 -> qty 5
    assert approved(rm.evaluate(make_signal(BTC, stop=99.0), p, PLAY, MON)) == (
        "cap_group",
        approx(5.0),
    )
    # DOGE: asset 7500, gross 15000 - 9500 = 5500 ; 5000 fits -> unchanged
    assert approved(rm.evaluate(make_signal(DOGE, stop=99.0), p, PLAY, MON)) == (
        "ok",
        approx(50.0),
    )


@pytest.mark.parametrize(
    ("longs", "rule"),
    [
        ([(ETH, 100.0, 100.0)], "cap_group"),  # majors room 10000 - 10000 = 0
        ([(ETH, 105.0, 100.0)], "cap_group"),  # majors room -500
        ([(DOGE, 150.0, 100.0)], "cap_gross"),  # gross room 15000 - 15000 = 0
        # majors room 10000 - 11000 = -1000 ; gross room 15000 - 17000 = -2000 is smaller
        ([(ETH, 110.0, 100.0), (DOGE, 60.0, 100.0)], "cap_gross"),
    ],
)
def test_no_room_under_a_cap_rejects_with_the_caps_rule(
    tmp_path: Path, longs: list[tuple[str, float, float]], rule: str
) -> None:
    decision = manager(tmp_path).evaluate(make_signal(), book(*longs), PLAY, MON)
    assert rejected(decision) == rule


def test_pending_entries_in_one_bar_cannot_jointly_breach_a_cap(tmp_path: Path) -> None:
    # empty 10000 portfolio ; each signal: entry 100, stop 99.5 -> risk qty 100, notional 10000
    rm, p = manager(tmp_path), book()
    pending: dict[str, float] = {}
    # BTC: asset 7500, majors 10000, gross 15000 -> asset binds
    btc = rm.evaluate(make_signal(BTC, stop=99.5), p, PLAY, MON, pending)
    assert approved(btc) == ("cap_asset", approx(75.0))
    pending[BTC] = btc.qty * 100.0  # 7500
    # ETH: asset 7500, majors 10000 - 7500 = 2500, gross 15000 - 7500 = 7500 -> group binds
    eth = rm.evaluate(make_signal(ETH, stop=99.5), p, PLAY, MON, pending)
    assert approved(eth) == ("cap_group", approx(25.0))
    pending[ETH] = eth.qty * 100.0  # 2500
    # SOL: majors 10000 - 10000 = 0 -> no room
    assert rejected(rm.evaluate(make_signal(SOL, stop=99.5), p, PLAY, MON, pending)) == "cap_group"
    assert sum(pending.values()) == approx(10_000.0)  # = majors cap, < gross cap 15000
    # without pending each would be judged alone (unchanged behaviour): 7500 again
    assert approved(rm.evaluate(make_signal(SOL, stop=99.5), p, PLAY, MON)) == (
        "cap_asset",
        approx(75.0),
    )


def test_pending_counts_toward_its_own_groups_and_gross(tmp_path: Path) -> None:
    # groups majors [BTC, ETH, SOL], alts [SOL, XRP] ; open ETH 1000 (majors), pending XRP 8000
    # signal entry 100, stop 99.5: risk notional 10000 ; gross used 1000 + 8000 = 9000
    rm = manager(tmp_path, groups=TWO_GROUPS)
    p = book((ETH, 10.0, 100.0))
    pending = {XRP: 8000.0}
    # BTC: asset 7500, majors 10000 - 1000 = 9000, gross 15000 - 9000 = 6000 -> gross binds
    btc = rm.evaluate(make_signal(BTC, stop=99.5), p, PLAY, MON, pending)
    assert approved(btc) == ("cap_gross", approx(60.0))
    # SOL: majors 9000, alts 10000 - 8000 = 2000, gross 6000 -> alts binds
    sol = rm.evaluate(make_signal(SOL, stop=99.5), p, PLAY, MON, pending)
    assert approved(sol) == ("cap_group", approx(20.0))
    assert "alts" in sol.detail
    # DOGE (no group): asset 7500, gross 6000 -> gross binds
    doge = rm.evaluate(make_signal(DOGE, stop=99.5), p, PLAY, MON, pending)
    assert approved(doge) == ("cap_gross", approx(60.0))


@pytest.mark.parametrize("notional", [7500.0, 0.0])
def test_symbol_already_pending_is_position_exists(tmp_path: Path, notional: float) -> None:
    decision = manager(tmp_path).evaluate(make_signal(BTC), book(), PLAY, MON, {BTC: notional})
    assert rejected(decision) == "position_exists"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_pending_notional_raises(tmp_path: Path, bad: float) -> None:
    rm = manager(tmp_path)
    state = rm.to_state()
    with pytest.raises(ValueError, match="pending"):
        rm.evaluate(make_signal(BTC), book(), PLAY, MON, {ETH: bad})
    assert rm.to_state() == state  # rejected before any rollover


def test_min_notional_boundary(tmp_path: Path) -> None:
    # risk_per_trade 0.0001: risk = 10000 * 0.0001 = 1
    rm = manager(tmp_path, risk_per_trade=0.0001)
    # stop 90: qty 1 / 10 = 0.1 -> notional 10 = MIN_NOTIONAL: allowed
    assert approved(rm.evaluate(make_signal(stop=90.0), book(), PLAY, MON)) == ("ok", approx(0.1))
    # stop 89: qty 1 / 11 -> notional 9.09 < 10
    assert rejected(rm.evaluate(make_signal(stop=89.0), book(), PLAY, MON)) == "too_small"
    assert MIN_NOTIONAL == 10.0


def test_cap_reduced_below_min_notional_is_too_small(tmp_path: Path) -> None:
    # DOGE 1 @ 14995: gross room = 15000 - 14995 = 5 > 0 but < 10
    p = book((DOGE, 1.0, 14_995.0))
    assert rejected(manager(tmp_path).evaluate(make_signal(), p, PLAY, MON)) == "too_small"


def test_zero_size_mult_is_too_small(tmp_path: Path) -> None:
    play = PLAY.model_copy(update={"size_mult": 0.0})
    assert rejected(manager(tmp_path).evaluate(make_signal(), book(), play, MON)) == "too_small"


# --- loss streak ----------------------------------------------------------------------------


def test_win_or_flat_trade_resets_the_streak(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    for pnl, streak in [(-1.0, 1), (-1.0, 2), (5.0, 0), (-1.0, 1), (0.0, 0)]:
        rm.on_trade_closed(make_trade(pnl))
        assert rm.loss_streak == streak


def test_streak_stop_blocks_the_rest_of_the_day_then_resets(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book()
    for i in range(4):
        rm.on_trade_closed(make_trade(-10.0, MON + timedelta(minutes=i)))
    assert rm.day_stopped is None  # 4 < streak_stop
    rm.on_trade_closed(make_trade(-10.0, MON + timedelta(minutes=4)))
    assert (rm.loss_streak, rm.day_stopped) == (5, date(2024, 1, 1))
    late = datetime(2024, 1, 1, 23, 59, 59, tzinfo=UTC)
    assert rejected(rm.evaluate(make_signal(), p, PLAY, late)) == "day_stopped"
    assert rm.must_flatten is False  # a streak stop blocks entries, it does not flatten
    # next UTC day: cleared, streak reset, full size again (not halved)
    decision = rm.evaluate(make_signal(), p, PLAY, datetime(2024, 1, 2, tzinfo=UTC))
    assert approved(decision) == ("ok", approx(10.0))
    assert (rm.loss_streak, rm.day_stopped) == (0, None)


def test_check_limits_on_the_next_day_also_clears_the_streak_stop(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    for _ in range(5):
        rm.on_trade_closed(make_trade(-10.0))
    assert rm.check_limits(book(), MON + DAY) == []
    assert (rm.loss_streak, rm.day_stopped) == (0, None)


def test_day_stopped_uses_the_exit_utc_date(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    plus2 = timezone(timedelta(hours=2))
    for _ in range(5):  # 2024-01-02 01:30 +02:00 is 2024-01-01 23:30 UTC
        rm.on_trade_closed(make_trade(-10.0, datetime(2024, 1, 2, 1, 30, tzinfo=plus2)))
    assert rm.day_stopped == date(2024, 1, 1)
    # now in +02:00 is converted too: 01:59 +02:00 is still Jan 1 UTC, 02:00 +02:00 is Jan 2
    still = datetime(2024, 1, 2, 1, 59, tzinfo=plus2)
    assert rejected(rm.evaluate(make_signal(), book(), PLAY, still)) == "day_stopped"
    next_day = datetime(2024, 1, 2, 2, 0, tzinfo=plus2)
    assert approved(rm.evaluate(make_signal(), book(), PLAY, next_day))[0] == "ok"


def test_loss_on_the_day_after_a_streak_stop_starts_a_new_streak(tmp_path: Path) -> None:
    # stopped Monday; a position closes at a loss Tuesday 00:01 before any Tuesday call:
    # the reset happens first, so the streak is 1, not 6, and Tuesday is not stopped
    rm = manager(tmp_path)
    for _ in range(5):
        rm.on_trade_closed(make_trade(-10.0))
    rm.on_trade_closed(make_trade(-10.0, datetime(2024, 1, 2, 0, 1, tzinfo=UTC)))
    assert (rm.loss_streak, rm.day_stopped) == (1, None)


def test_a_state_saved_with_the_old_pnl_books_still_loads(tmp_path: Path) -> None:
    # paper runs saved before pnl_by_day / pnl_by_week were dropped (never read) keep them
    rm = manager(tmp_path)
    for _ in range(3):
        rm.on_trade_closed(make_trade(-10.0))
    state = rm.to_state()
    assert "pnl_by_day" not in state and "pnl_by_week" not in state
    old = {**state, "pnl_by_day": {"2024-01-01": -30.0}, "pnl_by_week": {"2024-01-01": -30.0}}
    restored = manager(tmp_path)
    restored.load_state(old)
    assert restored.to_state() == state and restored.loss_streak == 3


def test_non_finite_trade_pnl_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="net_pnl"):
        manager(tmp_path).on_trade_closed(make_trade(float("nan")))


# --- loss limits ----------------------------------------------------------------------------


def test_daily_loss_halts_once_and_clears_on_the_next_utc_day(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))  # 10000 notional: 1% price move = 1% of equity
    assert rm.check_limits(p, datetime(2024, 1, 1, 0, 5, tzinfo=UTC)) == []  # day start 10000
    p.mark({DOGE: 98.51})  # equity 9851 > 10000 * 0.985 = 9850
    assert rm.check_limits(p, MON) == []
    p.mark({DOGE: 98.5})  # equity 9850 <= 9850: the boundary breaches
    events = rm.check_limits(p, MON)
    assert [(e.ts, e.kind, e.action) for e in events] == [(MON, "daily_loss", "halt")]
    assert (rm.halted_reason, rm.must_flatten) == ("daily_loss", True)
    assert rm.check_limits(p, MON + timedelta(minutes=15)) == []  # once per incident
    p.mark({DOGE: 100.0})  # recovery does not lift the halt within the day
    assert rm.check_limits(p, MON + timedelta(minutes=30)) == []
    assert rejected(rm.evaluate(make_signal(), p, PLAY, MON)) == "kill_switch"
    late = datetime(2024, 1, 1, 23, 59, tzinfo=UTC)
    assert rejected(rm.evaluate(make_signal(), p, PLAY, late)) == "kill_switch"

    tuesday = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    assert rm.check_limits(p, tuesday) == []  # new day start equity 10000
    assert (rm.halted_reason, rm.must_flatten) == (None, False)
    assert approved(rm.evaluate(make_signal(), p, PLAY, tuesday))[0] == "ok"
    p.mark({DOGE: 98.5})  # the condition recurs on Tuesday: a new incident, a new event
    assert [e.kind for e in rm.check_limits(p, tuesday + timedelta(hours=1))] == ["daily_loss"]


def test_day_start_equity_is_recorded_by_the_first_check_limits_call(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    assert approved(rm.evaluate(make_signal(), p, PLAY, MON))[0] == "ok"  # equity 10000
    p.mark({DOGE: 99.0})
    assert rm.check_limits(p, MON + timedelta(minutes=1)) == []  # day start = 9900
    # 9760 would breach a 10000 start (floor 9850) but not 9900 (floor 9900 * 0.985 = 9751.5)
    p.mark({DOGE: 97.6})
    assert rm.check_limits(p, MON + timedelta(minutes=2)) == []
    p.mark({DOGE: 97.5})  # 9750 <= 9751.5
    assert [e.kind for e in rm.check_limits(p, MON + timedelta(minutes=3))] == ["daily_loss"]


def test_weekly_loss_halts_until_monday(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    # each day loses less than 1.5% of its start equity; by Wednesday the week has lost 4.1%
    # Mon 10000 -> 9860 (floor 9850) ; Tue 9860 -> 9720 (floor 9712.1)
    for day, price in [(1, 98.6), (2, 97.2)]:
        start = datetime(2024, 1, day, 0, 5, tzinfo=UTC)
        assert rm.check_limits(p, start) == []  # records the day's (and Monday the week's) start
        p.mark({DOGE: price})
        assert rm.check_limits(p, start + timedelta(hours=1)) == []
    wednesday = datetime(2024, 1, 3, 0, 5, tzinfo=UTC)
    assert rm.check_limits(p, wednesday) == []  # day start 9720, floor 9574.2
    p.mark({DOGE: 95.9})  # 9590 > 9574.2 (no daily) but <= 10000 * 0.96 = 9600
    assert [e.kind for e in rm.check_limits(p, wednesday + timedelta(hours=1))] == ["weekly_loss"]
    sunday = datetime(2024, 1, 7, 23, 59, tzinfo=UTC)
    assert rm.check_limits(p, sunday) == []  # daily rolled over, weekly still active
    assert rm.halted_reason == "weekly_loss"
    assert rejected(rm.evaluate(make_signal(), p, PLAY, sunday)) == "kill_switch"
    monday = datetime(2024, 1, 8, 0, 0, tzinfo=UTC)  # ISO week starts Monday 00:00 UTC
    assert rm.check_limits(p, monday) == []
    assert rm.halted_reason is None
    assert approved(rm.evaluate(make_signal(), p, PLAY, monday))[0] == "ok"


def test_halt_reason_reports_the_most_severe_active_halt(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    assert rm.check_limits(p, MON) == []
    escalation = [
        (98.5, "daily_loss"),  # equity 9850
        (96.0, "weekly_loss"),  # 9600 <= 9600
        (90.0, "max_drawdown"),  # 9000: drawdown 1000 / 10000 = 0.10 >= 0.10
    ]
    for price, kind in escalation:
        p.mark({DOGE: price})
        assert [e.kind for e in rm.check_limits(p, MON)] == [kind]
        assert rm.halted_reason == kind
    (tmp_path / "KILL").touch()
    assert [e.kind for e in rm.check_limits(p, MON)] == ["manual_kill"]
    assert rm.halted_reason == "manual_kill"


def test_simultaneous_breaches_return_one_event_each_most_severe_first(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    assert rm.check_limits(p, MON) == []
    p.mark({DOGE: 89.0})
    (tmp_path / "KILL").touch()
    events = rm.check_limits(p, MON)
    assert [e.kind for e in events] == ["manual_kill", "max_drawdown", "weekly_loss", "daily_loss"]
    assert all(isinstance(e, RiskEvent) and e.action == "halt" and e.ts == MON for e in events)


def test_max_drawdown_halt_is_sticky_until_cleared(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    assert rm.check_limits(p, MON) == []
    p.mark({DOGE: 90.01})  # equity 9001: drawdown 0.0999 < 0.10 ; daily and weekly breach
    assert [e.kind for e in rm.check_limits(p, MON)] == ["weekly_loss", "daily_loss"]
    p.mark({DOGE: 90.0})  # drawdown 0.10: only the new kind is reported
    assert [e.kind for e in rm.check_limits(p, MON)] == ["max_drawdown"]
    next_week = datetime(2024, 1, 8, 0, 0, tzinfo=UTC)
    assert rm.check_limits(p, next_week) == []  # daily/weekly clear, drawdown stays
    assert (rm.halted_reason, rm.must_flatten) == ("max_drawdown", True)
    assert rejected(rm.evaluate(make_signal(), p, PLAY, next_week)) == "kill_switch"

    rm.clear_halt()
    assert (rm.halted_reason, rm.must_flatten) == (None, False)
    # still 10% below the peak: the next check is a new incident and halts again
    assert [e.kind for e in rm.check_limits(p, next_week)] == ["max_drawdown"]
    p.mark({DOGE: 95.0})  # drawdown 0.05
    rm.clear_halt()
    assert rm.check_limits(p, next_week) == []
    assert approved(rm.evaluate(make_signal(), p, PLAY, next_week))[0] == "ok"


def test_kill_file_blocks_entries_and_halts_until_removed_and_cleared(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    kill = tmp_path / "KILL"
    p = book()
    assert approved(rm.evaluate(make_signal(), p, PLAY, MON))[0] == "ok"
    kill.touch()
    # evaluate sees the file itself, even before check_limits has halted
    assert rejected(rm.evaluate(make_signal(), p, PLAY, MON)) == "kill_switch"
    assert rm.halted_reason is None
    events = rm.check_limits(p, MON)
    assert [(e.kind, e.action) for e in events] == [("manual_kill", "halt")]
    assert rm.check_limits(p, MON) == []
    assert rm.check_limits(p, datetime(2024, 1, 9, tzinfo=UTC)) == []  # survives day and week
    assert (rm.halted_reason, rm.must_flatten) == ("manual_kill", True)
    with pytest.raises(RuntimeError, match="kill file"):
        rm.clear_halt()
    assert rm.halted_reason == "manual_kill"
    kill.unlink()
    assert rm.halted_reason == "manual_kill"  # removing the file alone does not resume
    later = datetime(2024, 1, 9, 1, tzinfo=UTC)
    assert rejected(rm.evaluate(make_signal(), p, PLAY, later)) == "kill_switch"
    rm.clear_halt()
    assert (rm.halted_reason, rm.must_flatten) == (None, False)
    assert rm.check_limits(p, later) == []
    assert approved(rm.evaluate(make_signal(), p, PLAY, later))[0] == "ok"
    kill.touch()  # a new kill is a new incident
    assert [e.kind for e in rm.check_limits(p, later)] == ["manual_kill"]


# --- time and state -------------------------------------------------------------------------


def test_naive_datetimes_are_rejected(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    naive = datetime(2024, 1, 1, 12)  # noqa: DTZ001  # the naive input under test
    with pytest.raises(ValueError, match="naive"):
        rm.evaluate(make_signal(), book(), PLAY, naive)
    with pytest.raises(ValueError, match="naive"):
        rm.check_limits(book(), naive)
    with pytest.raises(ValueError, match="naive"):
        rm.on_trade_closed(make_trade(-1.0, naive))


@pytest.mark.parametrize(
    ("field", "bad"),
    [("cash", float("nan")), ("cash", float("inf")), ("peak_equity", float("nan"))],
)
def test_check_limits_raises_on_non_finite_equity_or_drawdown(
    tmp_path: Path, field: str, bad: float
) -> None:
    # a corrupted portfolio must not silently disable the limits (NaN compares False)
    rm = manager(tmp_path)
    p = book()
    assert rm.check_limits(p, MON) == []
    state = rm.to_state()
    setattr(p, field, bad)
    with pytest.raises(ValueError, match="finite"):
        rm.check_limits(p, MON + DAY)
    assert rm.to_state() == state  # nothing recorded, no rollover


def test_now_before_the_current_period_changes_nothing(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    wednesday = datetime(2024, 1, 10, 0, 5, tzinfo=UTC)  # second ISO week
    assert rm.check_limits(p, wednesday) == []
    p.mark({DOGE: 95.5})  # 9550: daily and weekly breach
    assert [e.kind for e in rm.check_limits(p, wednesday)] == ["weekly_loss", "daily_loss"]
    for _ in range(5):
        rm.on_trade_closed(make_trade(-1.0, wednesday))
    state = rm.to_state()
    for earlier in [wednesday - timedelta(minutes=10), wednesday - DAY, wednesday - 7 * DAY]:
        assert rm.check_limits(p, earlier) == []
        assert rejected(rm.evaluate(make_signal(), p, PLAY, earlier)) == "kill_switch"
    assert rm.to_state() == state  # no halt, start equity or streak stop was cleared


def test_weekly_halt_clears_across_a_year_boundary(tmp_path: Path) -> None:
    # ISO week 2025-W01 runs Mon 2024-12-30 .. Sun 2025-01-05 ; daily limit widened to isolate it
    rm = manager(tmp_path, daily_loss=0.5)
    p = book((DOGE, 100.0, 100.0))
    assert rm.check_limits(p, datetime(2024, 12, 30, 9, tzinfo=UTC)) == []  # week start 10000
    p.mark({DOGE: 96.0})  # 9600 <= 10000 * 0.96
    tuesday = datetime(2024, 12, 31, 9, tzinfo=UTC)
    assert [e.kind for e in rm.check_limits(p, tuesday)] == ["weekly_loss"]
    assert rm.check_limits(p, datetime(2025, 1, 5, 23, 59, tzinfo=UTC)) == []
    assert rm.halted_reason == "weekly_loss"  # same ISO week despite the new year
    monday = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    assert rm.check_limits(p, monday) == []
    assert rm.halted_reason is None
    assert approved(rm.evaluate(make_signal(), p, PLAY, monday))[0] == "ok"


def test_rollovers_apply_when_calls_skip_days(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    assert rm.check_limits(p, datetime(2024, 1, 1, 9, tzinfo=UTC)) == []  # Mon: start 10000
    p.mark({DOGE: 98.5})  # 9850: daily only
    assert [e.kind for e in rm.check_limits(p, datetime(2024, 1, 1, 10, tzinfo=UTC))] == [
        "daily_loss"
    ]
    for _ in range(5):
        rm.on_trade_closed(make_trade(-1.0, datetime(2024, 1, 1, 11, tzinfo=UTC)))
    # next call Thursday: daily halt and streak stop clear ; day start 9850, floor 9702.25
    thursday = datetime(2024, 1, 4, 9, tzinfo=UTC)
    assert rm.check_limits(p, thursday) == []
    assert (rm.halted_reason, rm.day_stopped, rm.loss_streak) == (None, None, 0)
    p.mark({DOGE: 96.0})  # 9600: daily (<= 9702.25) and weekly (<= 9600)
    assert [e.kind for e in rm.check_limits(p, thursday)] == ["weekly_loss", "daily_loss"]
    # next call two ISO weeks later: both clear and the new period starts at 9600
    later = datetime(2024, 1, 17, 9, tzinfo=UTC)
    assert rm.check_limits(p, later) == []
    assert rm.halted_reason is None
    assert approved(rm.evaluate(make_signal(), p, PLAY, later))[0] == "ok"


def drive(rm: RiskManager, p: Portfolio) -> list[object]:
    """A fixed sequence of calls; returns every decision and event list."""
    out: list[object] = []
    for ts in [
        datetime(2024, 1, 3, 23, 0, tzinfo=UTC),  # Wednesday: halted, day stopped
        datetime(2024, 1, 4, 0, 5, tzinfo=UTC),  # Thursday: daily and streak clear
        datetime(2024, 1, 8, 0, 5, tzinfo=UTC),  # next Monday: weekly clears
    ]:
        out.append(rm.evaluate(make_signal(), p, PLAY, ts))
        out.append(rm.check_limits(p, ts))
        out.append(rm.evaluate(make_signal(), p, PLAY, ts))
        p.mark({DOGE: p.last_prices[DOGE] - 1.0})
        out.append(rm.check_limits(p, ts + timedelta(hours=1)))
    out.append((rm.halted_reason, rm.loss_streak, rm.day_stopped))
    return out


def test_state_round_trips_through_json_and_reproduces_behaviour(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    p = book((DOGE, 100.0, 100.0))
    rm.on_trade_closed(make_trade(20.0, datetime(2023, 12, 29, tzinfo=UTC)))  # previous week
    assert rm.check_limits(p, datetime(2024, 1, 3, 0, 5, tzinfo=UTC)) == []  # Wed: start 10000
    p.mark({DOGE: 95.5})  # 9550: daily (floor 9850) and weekly (floor 9600)
    assert [e.kind for e in rm.check_limits(p, datetime(2024, 1, 3, 1, tzinfo=UTC))] == [
        "weekly_loss",
        "daily_loss",
    ]
    for _ in range(5):
        rm.on_trade_closed(make_trade(-3.0, datetime(2024, 1, 3, 2, tzinfo=UTC)))

    state = rm.to_state()
    assert json.loads(json.dumps(state)) == state  # JSON-safe and lossless
    assert state["day_stopped"] == "2024-01-03"
    restored = manager(tmp_path)
    restored.load_state(json.loads(json.dumps(state)))
    assert restored.to_state() == state
    assert (restored.halted_reason, restored.loss_streak, restored.day_stopped) == (
        "weekly_loss",
        5,
        date(2024, 1, 3),
    )

    twin = book((DOGE, 100.0, 100.0))
    twin.mark({DOGE: 95.5})
    original_run, restored_run = drive(rm, p), drive(restored, twin)
    assert original_run == restored_run
    assert rejected(original_run[0]) == "kill_switch"  # the sequence exercises real state
    assert restored.to_state() == rm.to_state()


def test_load_state_rejects_unknown_halt_kind(tmp_path: Path) -> None:
    rm = manager(tmp_path)
    state = rm.to_state()
    with pytest.raises(ValueError, match="halt"):
        rm.load_state({**state, "halts": ["bogus"]})
    assert rm.to_state() == state


# --- CLI ------------------------------------------------------------------------------------


def test_kill_and_unkill_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kill = tmp_path / "nested" / "data" / "KILL"
    monkeypatch.setattr(cli, "DEFAULT_KILL_FILE", kill)
    runner = CliRunner()
    for _ in range(2):  # idempotent; creates missing parent directories
        result = runner.invoke(cli.app, ["kill"])
        assert result.exit_code == 0
        assert len(result.output.splitlines()) == 1
        assert kill.exists()
    rm = RiskManager(make_cfg(), kill)
    assert rejected(rm.evaluate(make_signal(), book(), PLAY, MON)) == "kill_switch"
    for _ in range(2):  # removing an absent file is not an error
        result = runner.invoke(cli.app, ["unkill"])
        assert result.exit_code == 0
        assert len(result.output.splitlines()) == 1
        assert not kill.exists()
    assert approved(rm.evaluate(make_signal(), book(), PLAY, MON))[0] == "ok"


def test_default_kill_file_is_data_kill_at_the_repository_root() -> None:
    repo_root = Path(regimebot.__file__).resolve().parent.parent
    assert repo_root / "data" / "KILL" == DEFAULT_KILL_FILE


# --- property: approvals never breach a cap or the risk budget ------------------------------


def largest_allowed(
    r: RiskConfig, equity: float, symbol: str, used: dict[str, float], risk_notional: float
) -> float:
    """Independent oracle: the smallest of the risk-based notional and every cap's room, with
    ``used`` the notional already committed per symbol (open positions plus pending)."""
    rooms = [risk_notional, r.max_asset_exposure * equity]
    for members in TWO_GROUPS.values():
        if symbol in members:
            rooms.append(r.max_group_exposure * equity - sum(used.get(s, 0.0) for s in members))
    rooms.append(r.max_gross_leverage * equity - sum(used.values()))
    return min(rooms)


def test_random_approvals_never_breach_caps_or_the_risk_budget(tmp_path: Path) -> None:
    """3000 random configs and portfolios, each with a batch of signals for distinct symbols
    evaluated in sequence as one decision bar, approved notional accumulating in ``pending``.
    Every approval must keep asset, group and gross exposure (open positions + pending + the
    new entry) within their caps, risk at most risk_per_trade of equity at the stop, and be the
    largest size those limits allow."""
    rng = random.Random(8)
    tol = 1 + 1e-9
    rules: Counter[str] = Counter()
    for _ in range(3000):
        cfg = make_cfg(
            risk_per_trade=10 ** rng.uniform(-3.5, -1.3),  # 0.03% .. 5%
            max_gross_leverage=rng.uniform(0.1, 1.5),
            max_asset_exposure=rng.uniform(0.05, 1.0),
            max_group_exposure=rng.uniform(0.05, 1.0),
            groups=TWO_GROUPS,
        )
        r = cfg.risk
        rm = RiskManager(cfg, tmp_path / "KILL")
        losses = rng.randint(0, r.streak_stop - 1)
        for _ in range(losses):
            rm.on_trade_closed(make_trade(-1.0))
        streak_mult = 0.5 if losses >= r.streak_half else 1.0
        p = Portfolio(10 ** rng.uniform(2.5, 5.0))  # 316 .. 100 000
        for symbol in rng.sample([BTC, ETH, SOL, XRP, DOGE], rng.randint(0, 3)):
            price = rng.uniform(1.0, 1000.0)
            qty = p.initial_equity * rng.uniform(0.0, 0.6) / price
            open_position(p, symbol, rng.choice([Side.LONG, Side.SHORT]), qty, price)
            p.mark({symbol: price * rng.uniform(0.8, 1.2)})
        equity = p.equity
        positions = {s: abs(pos.qty) * p.last_prices[s] for s, pos in p.positions.items()}
        pending: dict[str, float] = {}
        for symbol in rng.sample([BTC, ETH, SOL, XRP, DOGE], rng.randint(1, 5)):
            side = rng.choice([Side.LONG, Side.SHORT])
            entry = rng.uniform(1.0, 1000.0)
            stop = entry - side_sign(side) * entry * 10 ** rng.uniform(-4.0, -0.5)
            if rng.random() < 0.03:
                stop = 2 * entry - stop  # wrong side
            size_mult = rng.choice([1.0, rng.uniform(0.0, 1.0)])
            play = PLAY.model_copy(update={"size_mult": size_mult})
            decision = rm.evaluate(make_signal(symbol, side, entry, stop), p, play, MON, pending)
            rules[f"{'approved' if decision.approved else 'rejected'} {decision.rule}"] += 1
            if not decision.approved:
                assert decision.qty == 0.0
                continue

            notional = decision.qty * entry
            used = {**positions, **pending}  # distinct keys: open symbols are never approved
            risk_budget = r.risk_per_trade * equity
            assert decision.qty > 0.0
            assert notional >= MIN_NOTIONAL
            assert decision.qty * abs(entry - stop) <= risk_budget * tol
            assert notional <= r.max_asset_exposure * equity * tol
            for members in TWO_GROUPS.values():
                if symbol in members:
                    group_used = sum(used.get(s, 0.0) for s in members)
                    assert group_used + notional <= r.max_group_exposure * equity * tol
            assert sum(used.values()) + notional <= r.max_gross_leverage * equity * tol
            # tight: the largest size the risk budget and every cap allow
            risk_notional = risk_budget * size_mult * streak_mult / abs(entry - stop) * entry
            largest = largest_allowed(r, equity, symbol, used, risk_notional)
            assert notional == pytest.approx(largest, rel=1e-9)
            if notional < largest_allowed(r, equity, symbol, positions, risk_notional) * (1 - 1e-9):
                rules["approved smaller because of pending"] += 1
            pending[symbol] = notional

    # the random cases must reach every branch, or the checks above prove little
    for rule in ["ok", "cap_asset", "cap_group", "cap_gross", "smaller because of pending"]:
        assert rules[f"approved {rule}"] > 100, rules
    for rule in ["too_small", "cap_group", "cap_gross", "position_exists", "invalid_stop"]:
        assert rules[f"rejected {rule}"] > 30, rules
