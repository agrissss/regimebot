"""Metrics on hand-built trades and equity. Expected values are worked out in the comments."""

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from regimebot.reporting.metrics import active_period, breakdown, compute_metrics, max_drawdown
from regimebot.types import ExitReason, Fill, OrderSide, Regime, RiskEvent, Side, Trade

D1 = datetime(2024, 1, 1, tzinfo=UTC)
INITIAL = 1000.0


def hm(h: int, m: int = 0) -> datetime:
    return D1 + timedelta(hours=h, minutes=m)


def trade(
    symbol: str, strategy: str, side: Side, regime: Regime, qty: float, entry_price: float,
    net: float, r: float, reason: ExitReason, entry: datetime, exit_: datetime,
    fees: float, funding: float, bars: int,
) -> Trade:  # fmt: skip
    return Trade(
        symbol, strategy, side, regime, qty, entry, exit_, entry_price, entry_price, 1.0, 2.0,
        net + fees + funding, fees, funding, net, reason, r, bars,
    )  # fmt: skip


TRADES = [
    trade("BTC", "breakout", Side.LONG, Regime.EXPANSION, 2, 100, 30, 1.5, ExitReason.TARGET,
          hm(0, 15), hm(1, 15), 1.0, 0.5, 4),
    trade("BTC", "trend", Side.SHORT, Regime.TREND_DOWN, 1, 200, -20, -1.0, ExitReason.STOP,
          hm(1), hm(2), 2.0, -0.5, 2),
    trade("ETH", "breakout", Side.LONG, Regime.RANGE, 4, 50, 10, 0.5, ExitReason.TRAIL,
          hm(3), hm(5), 1.0, 0.0, 6),
    trade("ETH", "mean_reversion", Side.LONG, Regime.RANGE, 1, 50, -10, -0.5, ExitReason.TRAIL,
          hm(6), hm(7), 1.0, 0.0, 4),
]  # fmt: skip

# Hourly points on day 1 (00:00..07:00, a dip to 950 at 04:00), then one point on each of days 2-4.
HOURLY = [1000, 1000, 1000, 1000, 950, 1000, 1000, 1000]
EQUITY = pd.Series(
    [*HOURLY, 1100, 990, 1089],
    index=pd.DatetimeIndex([hm(h) for h in range(8)] + [hm(36), hm(60), hm(84)]),
    name="equity",
    dtype=float,
)


def fill(symbol: str, ts: datetime, side: OrderSide, slip: float, liq: str = "taker") -> Fill:
    return Fill(symbol, ts, side, 1.0, 100.0, 100.0, 0.0, slip, liq)


FILLS = [
    fill("BTC", hm(0, 15), OrderSide.BUY, 2.0),  # T1 entry
    fill("BTC", hm(1, 15), OrderSide.SELL, 4.0),  # T1 exit
    fill("BTC", hm(1), OrderSide.SELL, 6.0),  # T2 entry (short)
    fill("BTC", hm(2), OrderSide.BUY, 10.0),  # T2 exit
    fill("BTC", hm(1, 15), OrderSide.SELL, 100.0, "maker"),  # maker: ignored
    fill("BTC", hm(1, 15), OrderSide.BUY, 999.0),  # matches no trade side: ignored
]


def test_headline_numbers_each_checked() -> None:
    m = compute_metrics(TRADES, EQUITY, INITIAL, FILLS)
    assert m.n_trades == 4
    assert m.net_pnl == 10  # 30 - 20 + 10 - 10
    assert m.return_pct == pytest.approx(0.089)  # 1089 / 1000 - 1
    assert m.profit_factor == pytest.approx(40 / 30)  # wins 30 + 10, losses 20 + 10
    assert m.win_rate == 0.5  # 2 of 4 have net_pnl > 0
    assert m.avg_win == 20  # (30 + 10) / 2
    assert m.avg_loss == -15  # (-20 - 10) / 2
    assert m.expectancy_r == pytest.approx(0.125)  # (1.5 - 1.0 + 0.5 - 0.5) / 4
    assert m.avg_bars_held == 4  # (4 + 2 + 6 + 4) / 4
    assert m.turnover == pytest.approx(0.65)  # (2*100 + 1*200 + 4*50 + 1*50) / 1000
    assert m.fees == 5  # 1 + 2 + 1 + 1
    assert m.funding == 0  # 0.5 - 0.5
    assert m.target_hit_rate == 0.25  # only T1
    assert m.stop_out_rate == 0.5  # T2 (STOP) and T4 (TRAIL below entry); T3 TRAIL won
    assert m.net_pnl_without_best_5pct == -20  # ceil(0.05 * 4) = 1: drop +30 -> -20 + 10 - 10
    assert m.avg_entry_slippage_bps == 4  # (2 + 6) / 2, taker fills of entries only
    assert m.avg_exit_slippage_bps == 7  # (4 + 10) / 2


def test_exposure_counts_decision_times_inside_trades_half_open() -> None:
    # 11 equity timestamps. In a position (entry <= ts < exit): 01:00 (T1, T2), 03:00, 04:00 (T3),
    # 06:00 (T4). Not: 02:00 (T2 exits), 05:00, 07:00 (exit times) and the three later days.
    assert compute_metrics(TRADES, EQUITY, INITIAL).exposure_pct == pytest.approx(4 / 11)


def test_sharpe_and_sortino_from_daily_equity() -> None:
    # Daily last equity: 1000, 1100, 990, 1089 -> returns +0.1, -0.1, +0.1.
    # mean = 0.1/3; sample variance = ((0.2/3)^2 * 2 + (0.4/3)^2) / 2 = 0.04/3
    # sharpe = (0.1/3) / sqrt(0.04/3) * sqrt(365)
    # downside deviation = sqrt((0 + 0.01 + 0) / 3); sortino = (0.1/3) / sqrt(0.01/3) * sqrt(365)
    m = compute_metrics(TRADES, EQUITY, INITIAL)
    assert m.sharpe == pytest.approx((0.1 / 3) / math.sqrt(0.04 / 3) * math.sqrt(365))
    assert m.sortino == pytest.approx((0.1 / 3) / math.sqrt(0.01 / 3) * math.sqrt(365))


def test_max_drawdown_peak_to_trough_including_the_initial_point() -> None:
    # peak 1100 -> trough 990: 110 / 1100 = 0.1 (beats the 950 dip from the initial 1000: 0.05)
    assert max_drawdown(EQUITY, INITIAL) == pytest.approx(0.1)
    # a series that only falls: the initial equity is the peak
    assert max_drawdown(pd.Series([900.0, 800.0], index=EQUITY.index[:2]), 1000) == pytest.approx(
        0.2
    )
    assert max_drawdown(pd.Series(dtype=float), 1000) == 0.0


def test_no_trades() -> None:
    flat = pd.Series([1000.0, 1000.0], index=EQUITY.index[:2])
    m = compute_metrics([], flat, INITIAL)
    assert m.n_trades == 0 and m.net_pnl == 0 and m.return_pct == 0 and m.max_drawdown == 0
    assert m.exposure_pct == 0.0 and m.turnover == 0 and m.net_pnl_without_best_5pct == 0
    for value in (m.profit_factor, m.win_rate, m.avg_win, m.avg_loss, m.expectancy_r,
                  m.avg_bars_held, m.target_hit_rate, m.stop_out_rate, m.sharpe, m.sortino,
                  m.avg_entry_slippage_bps, m.avg_exit_slippage_bps):  # fmt: skip
        assert math.isnan(value)
    empty = compute_metrics([], pd.Series(dtype=float), INITIAL)
    assert math.isnan(empty.exposure_pct) and empty.return_pct == 0


def test_only_winners_profit_factor_is_inf_and_avg_loss_nan() -> None:
    winners = [TRADES[0], TRADES[2]]
    m = compute_metrics(winners, EQUITY, INITIAL)
    assert m.profit_factor == math.inf and math.isnan(m.avg_loss)
    assert m.win_rate == 1.0 and m.avg_win == 20


def test_only_losers_profit_factor_is_zero_and_avg_win_nan() -> None:
    m = compute_metrics([TRADES[1], TRADES[3]], EQUITY, INITIAL)
    assert m.profit_factor == 0.0 and math.isnan(m.avg_win) and m.win_rate == 0.0


def test_single_day_has_no_ratios() -> None:
    one_day = EQUITY.iloc[:8]
    m = compute_metrics(TRADES, one_day, INITIAL)
    assert math.isnan(m.sharpe) and math.isnan(m.sortino)


def test_constant_equity_has_no_sharpe_and_no_losing_day_has_no_sortino() -> None:
    days = pd.DatetimeIndex([hm(0), hm(24), hm(48), hm(72)])
    assert math.isnan(compute_metrics([], pd.Series([1000.0] * 4, index=days), INITIAL).sharpe)
    rising = pd.Series([1000.0, 1010.0, 1030.0, 1040.0], index=days)
    m = compute_metrics([], rising, INITIAL)
    assert math.isnan(m.sortino) and not math.isnan(m.sharpe)


def test_best_5pct_dropped_is_ceil_with_exact_arithmetic() -> None:
    # 20 trades -> drop exactly 1 (0.05 * 20 must not round to 2); 21 trades -> drop 2
    def pnls(n: int) -> list[Trade]:
        return [
            trade("BTC", "breakout", Side.LONG, Regime.RANGE, 1, 100, float(i), 0.0,
                  ExitReason.TIME, hm(0), hm(1), 0.0, 0.0, 1)
            for i in range(1, n + 1)
        ]  # fmt: skip

    eq = EQUITY
    assert compute_metrics(pnls(20), eq, INITIAL).net_pnl_without_best_5pct == 210 - 20
    assert compute_metrics(pnls(21), eq, INITIAL).net_pnl_without_best_5pct == 231 - 21 - 20
    assert compute_metrics(pnls(1), eq, INITIAL).net_pnl_without_best_5pct == 0


@pytest.mark.parametrize("by", ["regime", "symbol", "strategy", "exit_reason"])
def test_breakdown_sums_match_totals(by: str) -> None:
    df = breakdown(TRADES, by)  # type: ignore[arg-type]  # parametrised literal
    assert df["n"].sum() == 4 and df["net_pnl"].sum() == 10
    total_r = sum(t.r_multiple for t in TRADES)
    assert (df["expectancy_r"] * df["n"]).sum() == pytest.approx(total_r)
    assert (df["win_rate"] * df["n"]).sum() == pytest.approx(2)


def test_breakdown_by_symbol_values_and_empty() -> None:
    df = breakdown(TRADES, "symbol")
    assert df.loc["BTC"].tolist() == [2, 0.5, 10, 0.25]  # 30 - 20; R (1.5 - 1) / 2
    assert df.loc["ETH"].tolist() == [2, 0.5, 0, 0.0]  # 10 - 10; R (0.5 - 0.5) / 2
    assert breakdown([], "regime").empty


def test_halted_run_keeps_the_snapshot_that_realises_the_kill_flatten() -> None:
    # Halt at 02:00; the kill trade exits at 02:00 but its loss first shows at the 03:00 snapshot.
    killed = trade("BTC", "breakout", Side.LONG, Regime.RANGE, 1, 100, -50, -1.0,
                   ExitReason.KILL, hm(1), hm(2), 0.0, 0.0, 4)  # fmt: skip
    late = trade("BTC", "breakout", Side.LONG, Regime.RANGE, 1, 100, 999, 1.0,
                 ExitReason.TARGET, hm(4), hm(5), 0.0, 0.0, 4)  # fmt: skip
    equity = pd.Series(
        [1000.0, 1000.0, 960.0, 950.0, 950.0, 950.0],  # 00:00 .. 05:00; 960 = mark before flatten
        index=pd.DatetimeIndex([hm(h) for h in range(6)]),
    )
    events = [RiskEvent(hm(2), "max_drawdown", "d", "halt")]
    trades, kept, halt = active_period([killed, late], equity, events)
    assert trades == [killed] and halt is not None  # the trade entered after the halt is dropped
    assert kept.index[-1] == hm(3) and len(kept) == 4  # includes 03:00, drops 04:00 and 05:00
    m = compute_metrics(trades, kept, INITIAL)
    assert m.net_pnl == -50
    assert m.return_pct == pytest.approx(m.net_pnl / INITIAL)  # 950 / 1000 - 1 = -0.05
    assert m.max_drawdown == pytest.approx(0.05)  # (1000 - 950) / 1000, not 40 / 1000
