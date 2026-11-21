"""Performance metrics from a backtest's trades and equity curve.

Every ratio is NaN (never 0 or a made-up number) when the data cannot support it, and the report
prints NaN as "n/a". Daily statistics use the last equity value of each UTC day.
"""

import math
import statistics
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from regimebot.risk.manager import PERMANENT_HALTS
from regimebot.types import (
    ExitReason,
    Fill,
    RiskDecision,
    RiskEvent,
    Trade,
    entry_order_side,
    exit_order_side,
)

DAYS_PER_YEAR = 365  # crypto trades every day
BEST_TRADES_DROPPED = 0.05
_FLAT_STD = 1e-12  # a deviation below this is rounding noise, not risk
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_MIN_TRADES = 20  # fewer trades: no interval (n/a)
RR_REL_TOL = 1e-9  # a trade filled exactly at its signal's reward-to-risk must not fail by rounding
# Rules of the second decision that cancels an approved entry before it fills (engine.loop).
CANCEL_RULES = ("no_liquidity", "expired", "gapped_through_stop", "cancelled", "kill_switch")


@dataclass(frozen=True, slots=True)
class Metrics:
    """Headline numbers of one run. Units: ``net_pnl``, ``avg_win``, ``avg_loss``, ``fees``,
    ``funding`` and ``net_pnl_without_best_5pct`` are currency; ``return_pct``, ``max_drawdown``,
    ``win_rate``, ``exposure_pct``, ``target_hit_rate`` and ``stop_out_rate`` are fractions
    (0.1 = 10%); ``avg_*_slippage_bps`` are basis points; ``expectancy_r`` is in R multiples;
    ``turnover`` and ``max_gross_leverage_seen`` (largest gross exposure / equity at a
    snapshot) are multiples of equity; ``sharpe``, ``sortino`` and ``profit_factor`` are plain
    ratios. NaN means "not computable from this data"."""

    net_pnl: float
    return_pct: float  # fraction of initial equity
    max_drawdown: float  # fraction, >= 0
    sharpe: float
    sortino: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float  # negative
    expectancy_r: float
    n_trades: int
    avg_bars_held: float
    exposure_pct: float
    turnover: float
    fees: float
    funding: float
    target_hit_rate: float
    stop_out_rate: float
    net_pnl_without_best_5pct: float
    avg_entry_slippage_bps: float
    avg_exit_slippage_bps: float
    max_gross_leverage_seen: float


def daily_equity(equity: pd.Series) -> pd.Series:
    """Last equity value of each UTC day."""
    if equity.empty:
        return pd.Series(dtype=float)
    days = pd.DatetimeIndex(equity.index).tz_convert("UTC").normalize()
    return equity.groupby(days).last()


def daily_pnl(equity: pd.Series, initial_equity: float) -> pd.Series:
    """Change of the last equity value of each UTC day from the day before (the first day from
    ``initial_equity``)."""
    daily = daily_equity(equity)
    pnl = daily.diff()
    if len(daily):
        pnl.iloc[0] = daily.iloc[0] - initial_equity
    return pnl


def daily_returns(equity: pd.Series) -> pd.Series:
    """Percentage change of the last equity value of each UTC day."""
    return daily_equity(equity).pct_change().dropna()


def sharpe_ratio(equity: pd.Series) -> float:
    """mean / std (ddof=1) of daily returns * sqrt(365); NaN with under 2 returns or no spread."""
    r = daily_returns(equity)
    if len(r) < 2:
        return math.nan
    std = float(r.std(ddof=1))
    return math.nan if std < _FLAT_STD else float(r.mean()) / std * math.sqrt(DAYS_PER_YEAR)


def sortino_ratio(equity: pd.Series) -> float:
    """mean / downside deviation * sqrt(365), downside deviation = sqrt(mean(min(r, 0)^2));
    NaN with under 2 returns or no losing day."""
    r = daily_returns(equity)
    if len(r) < 2:
        return math.nan
    downside = math.sqrt(float(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    return (
        math.nan if downside < _FLAT_STD else float(r.mean()) / downside * math.sqrt(DAYS_PER_YEAR)
    )


def max_drawdown(equity: pd.Series, initial_equity: float) -> float:
    """Largest peak-to-trough fall as a fraction of the peak, starting from ``initial_equity``."""
    values = np.concatenate([[initial_equity], equity.to_numpy(dtype=float)])
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))


def best_trades_dropped(n_trades: int) -> int:
    """ceil(5% of n) in integer arithmetic (0.05 * n can round the wrong way)."""
    return -(-n_trades * 5 // 100)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _profit_factor(pnls: Sequence[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses > 0:
        return wins / losses
    return math.inf if wins > 0 else math.nan


def _exposure(trades: Sequence[Trade], equity: pd.Series) -> float:
    """Share of decision timestamps inside any trade's [entry_ts, exit_ts)."""
    stamps = np.unique(pd.DatetimeIndex(equity.index).as_unit("ns").asi8)
    if len(stamps) == 0:
        return math.nan
    entries = np.sort(pd.DatetimeIndex([t.entry_ts for t in trades]).as_unit("ns").asi8)
    exits = np.sort(pd.DatetimeIndex([t.exit_ts for t in trades]).as_unit("ns").asi8)
    # trades open at ts = entries at or before ts minus exits at or before ts
    open_count = np.searchsorted(entries, stamps, "right") - np.searchsorted(exits, stamps, "right")
    return float(np.mean(open_count > 0))


def _slippage(trades: Sequence[Trade], fills: Sequence[Fill]) -> tuple[float, float]:
    """Mean taker-fill slippage (bps) of entry fills and of exit fills, matched to the trades by
    (symbol, ts, side)."""
    entry_keys = {(t.symbol, t.entry_ts, entry_order_side(t.side)) for t in trades}
    exit_keys = {(t.symbol, t.exit_ts, exit_order_side(t.side)) for t in trades}
    taker = [f for f in fills if f.liquidity == "taker"]
    entries = [f.slippage_bps for f in taker if (f.symbol, f.ts, f.side) in entry_keys]
    exits = [f.slippage_bps for f in taker if (f.symbol, f.ts, f.side) in exit_keys]
    return _mean(entries), _mean(exits)


def _max_leverage(equity: pd.Series, gross_exposure: pd.Series | None) -> float:
    if gross_exposure is None or equity.empty:
        return math.nan
    leverage = gross_exposure.reindex(equity.index) / equity
    return float(leverage.max()) if leverage.notna().any() else math.nan


def compute_metrics(
    trades: Sequence[Trade],
    equity: pd.Series,
    initial_equity: float,
    fills: Sequence[Fill] = (),
    gross_exposure: pd.Series | None = None,
) -> Metrics:
    """All headline metrics. ``equity`` is the decision-time equity curve on a UTC index;
    ``gross_exposure`` is the gross exposure at each equity snapshot (only the stamps in
    ``equity`` count; without it ``max_gross_leverage_seen`` is NaN)."""
    pnls = [t.net_pnl for t in trades]
    n = len(trades)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    drop = best_trades_dropped(n)
    kept = sorted(pnls, reverse=True)[drop:]
    stopped = sum(
        t.exit_reason == ExitReason.STOP or (t.exit_reason == ExitReason.TRAIL and t.net_pnl < 0)
        for t in trades
    )
    net_pnl = float(sum(pnls))
    final = float(equity.iloc[-1]) if len(equity) else initial_equity + net_pnl
    entry_slip, exit_slip = _slippage(trades, fills)
    return Metrics(
        net_pnl=net_pnl,
        return_pct=final / initial_equity - 1.0,
        max_drawdown=max_drawdown(equity, initial_equity),
        sharpe=sharpe_ratio(equity),
        sortino=sortino_ratio(equity),
        profit_factor=_profit_factor(pnls),
        win_rate=len(winners) / n if n else math.nan,
        avg_win=_mean(winners),
        avg_loss=_mean(losers),
        expectancy_r=_mean([t.r_multiple for t in trades]),
        n_trades=n,
        avg_bars_held=_mean([t.bars_held for t in trades]),
        exposure_pct=_exposure(trades, equity) if n else (0.0 if len(equity) else math.nan),
        turnover=sum(t.qty * t.entry_price for t in trades) / initial_equity,
        fees=float(sum(t.fees for t in trades)),
        funding=float(sum(t.funding for t in trades)),
        target_hit_rate=(
            sum(t.exit_reason == ExitReason.TARGET for t in trades) / n if n else math.nan
        ),
        stop_out_rate=stopped / n if n else math.nan,
        net_pnl_without_best_5pct=float(sum(kept)),
        avg_entry_slippage_bps=entry_slip,
        avg_exit_slippage_bps=exit_slip,
        max_gross_leverage_seen=_max_leverage(equity, gross_exposure),
    )


def expectancy_ci(r_multiples: Sequence[float], seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean R per trade: the 2.5th and 97.5th percentiles of the
    means of ``BOOTSTRAP_RESAMPLES`` resamples of the trades with replacement (seeded, so a
    report is reproducible). (NaN, NaN) under ``BOOTSTRAP_MIN_TRADES`` trades."""
    r = np.asarray(r_multiples, dtype=float)
    if len(r) < BOOTSTRAP_MIN_TRADES:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(r, len(r)).mean() for _ in range(BOOTSTRAP_RESAMPLES)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def reward_to_risk_at_fill(trade: Trade) -> float:
    """|target - entry_price| / |entry_price - initial_stop|: the reward-to-risk the trade had at
    its actual entry fill, not at the decision close its signal was built on."""
    return abs(trade.target - trade.entry_price) / abs(trade.entry_price - trade.initial_stop)


def meets_min_rr(trade: Trade, min_rr: float) -> bool:
    """True when the reward-to-risk at the entry fill reaches ``min_rr`` (within rounding)."""
    return reward_to_risk_at_fill(trade) >= min_rr * (1 - RR_REL_TOL)


def strategy_quality(trades: Sequence[Trade], min_rr: Mapping[str, float]) -> pd.DataFrame:
    """Per strategy: ``target_hit_rate`` (share of exits at the target; mean reversion's target
    is not a 5% move, so one number over all strategies misleads) and ``met_min_rr`` (share of
    trades that met the strategy's ``min_rr`` at the actual entry fill; NaN for a strategy not in
    ``min_rr``). Indexed by strategy, sorted."""
    rows: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        rows[t.strategy].append(t)
    out = pd.DataFrame(
        {
            "target_hit_rate": {
                s: float(np.mean([t.exit_reason == ExitReason.TARGET for t in ts]))
                for s, ts in rows.items()
            },
            "met_min_rr": {
                s: float(np.mean([meets_min_rr(t, min_rr[s]) for t in ts]))
                if s in min_rr
                else math.nan
                for s, ts in rows.items()
            },
        },
        columns=["target_hit_rate", "met_min_rr"],
        dtype=float,
    )
    out.index.name = "strategy"
    return out.sort_index()


def largest_market_moves(
    closes: pd.DataFrame, equity: pd.Series, initial_equity: float, n: int = 10
) -> pd.DataFrame:
    """The ``n`` UTC days of ``equity`` (the active period) with the largest absolute
    equal-weight average daily return of the symbols, largest first: ``market_return`` (mean
    over the symbols of the change of each one's last decision close of the day from the day
    before; ``closes`` has one column per symbol and should start before the period, so its
    first day has a return) and ``strategy_pnl`` (``daily_pnl``). Indexed by date."""
    pnl = daily_pnl(equity, initial_equity)
    days = pd.DatetimeIndex(closes.index).tz_convert("UTC").normalize()
    market = closes.groupby(days).last().pct_change().mean(axis=1).reindex(pnl.index).dropna()
    top = market.abs().sort_values(ascending=False, kind="stable").index[:n]
    out = pd.DataFrame({"market_return": market[top], "strategy_pnl": pnl[top]})
    out.index.name = "date"
    return out


@dataclass(frozen=True, slots=True)
class ExecutionQuality:
    """How a run's simulated fills went. Lags are seconds from the signal (decision close) to
    the entry fill; slippage is bps against the fill's reference price (positive = worse) by
    fill type; the spread is (ask - bid) / mid in bps over the fills that recorded a quote."""

    n_fills: int
    entry_lag_median_s: float
    entry_lag_p95_s: float
    slippage_bps: dict[str, tuple[int, float]]  # fill type -> (fills, mean slippage)
    avg_spread_bps: float
    cancelled: dict[str, int]  # every rule in CANCEL_RULES -> approved entries it cancelled


FILL_TYPES = ("entry", "stop exit", "target exit", "market exit")
_EXIT_TYPES = {
    ExitReason.STOP: "stop exit",
    ExitReason.TRAIL: "stop exit",
    ExitReason.TARGET: "target exit",
}  # every other reason (time, regime, kill, end of data) exits at market


def _fill_types(trades: Sequence[Trade]) -> dict[tuple[str, datetime, str], deque[str]]:
    """The type of each closed trade's fills, queued per (symbol, ts, order side) in event order:
    at one timestamp an exit comes before an entry (a stop in the bar closing at T, then the fill
    of the decision at T), so a key both share is told apart by the order fills were logged."""
    events = []
    for t in trades:
        exit_type = _EXIT_TYPES.get(t.exit_reason, "market exit")
        events.append(
            (t.exit_ts, 0, (t.symbol, t.exit_ts, str(exit_order_side(t.side))), exit_type)
        )
        events.append(
            (t.entry_ts, 1, (t.symbol, t.entry_ts, str(entry_order_side(t.side))), "entry")
        )
    queues: dict[tuple[str, datetime, str], deque[str]] = defaultdict(deque)
    for _, _, key, kind in sorted(events, key=lambda e: (e[0], e[1])):
        queues[key].append(kind)
    return queues


def execution_quality(
    fills: Sequence[Fill],
    trades: Sequence[Trade],
    signals: Mapping[int, tuple[str, datetime]],
    decisions: Sequence[tuple[int | None, datetime, RiskDecision]],
) -> ExecutionQuality:
    """``ExecutionQuality`` of a run from its fills (in time order), closed trades, signals (row
    id -> symbol, ts) and risk decisions (signal id, ts, decision) in stored order. A fill that
    is no closed trade's is an open position's entry (every exit closes a trade). An entry's
    signal is the latest approved signal of its symbol at or before the fill (one order per
    symbol at a time, so it is the one that queued it). A cancellation is a rejection, with a
    ``CANCEL_RULES`` rule, of a signal approved before."""
    approved_ids = [sid for sid, _, d in decisions if d.approved and sid is not None]
    approved: dict[str, list[datetime]] = defaultdict(list)
    for sid in approved_ids:
        symbol, ts = signals[sid]
        approved[symbol].append(ts)
    queues, lags = _fill_types(trades), []
    by_type: dict[str, list[float]] = defaultdict(list)
    for f in fills:
        queue = queues.get((f.symbol, f.ts, str(f.side)))
        kind = queue.popleft() if queue else "entry"
        by_type[kind].append(f.slippage_bps)
        if kind == "entry":
            before = [ts for ts in approved[f.symbol] if ts <= f.ts]
            if before:
                lags.append((f.ts - max(before)).total_seconds())
    spreads = [
        (f.ask - f.bid) / ((f.ask + f.bid) / 2) * 1e4
        for f in fills
        if f.bid is not None and f.ask is not None
    ]
    seen_approved, cancelled = set(approved_ids), Counter[str]()
    for signal_id, _, d in decisions:
        if not d.approved and signal_id in seen_approved and d.rule in CANCEL_RULES:
            cancelled[d.rule] += 1
    return ExecutionQuality(
        n_fills=len(fills),
        entry_lag_median_s=float(np.median(lags)) if lags else math.nan,
        entry_lag_p95_s=float(np.percentile(lags, 95)) if lags else math.nan,
        slippage_bps={k: (len(v), _mean(v)) for k in FILL_TYPES if (v := by_type.get(k))},
        avg_spread_bps=_mean(spreads),
        cancelled={rule: cancelled[rule] for rule in CANCEL_RULES},
    )


def breakdown(
    trades: Sequence[Trade], by: Literal["regime", "symbol", "strategy", "exit_reason", "side"]
) -> pd.DataFrame:
    """Per-group trade count, win rate, net pnl and mean R, indexed by the group value."""
    columns = ["n", "win_rate", "net_pnl", "expectancy_r"]
    if not trades:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=by), dtype=float)
    frame = pd.DataFrame(
        {
            by: [str(getattr(t, by)) for t in trades],
            "net_pnl": [t.net_pnl for t in trades],
            "r": [t.r_multiple for t in trades],
        }
    )
    frame["win"] = frame["net_pnl"] > 0
    grouped = frame.groupby(by)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "win_rate": grouped["win"].mean(),
            "net_pnl": grouped["net_pnl"].sum(),
            "expectancy_r": grouped["r"].mean(),
        }
    )
    return out.sort_index()


def first_permanent_halt(events: Sequence[RiskEvent]) -> RiskEvent | None:
    """The earliest halt that ends trading (``max_drawdown`` or ``manual_kill``), if any."""
    halts = [e for e in events if e.kind in PERMANENT_HALTS and e.action == "halt"]
    return min(halts, key=lambda e: e.ts, default=None)


def active_period(
    trades: Sequence[Trade], equity: pd.Series, events: Sequence[RiskEvent]
) -> tuple[list[Trade], pd.Series, RiskEvent | None]:
    """Trades and equity up to the first permanent halt (everything when there is none).

    After such a halt the strategy sits flat, and averaging that into Sharpe or exposure would
    flatter it, so headline metrics use this window. The flatten that follows a halt fills at
    the next price, so its realised loss first shows in the first snapshot strictly after both
    the halt and the last kept trade's exit; equity is cut right after that snapshot, so return
    and drawdown agree with the trades' net P&L and the flat stretch after it is excluded.
    """
    halt = first_permanent_halt(events)
    if halt is None:
        return list(trades), equity, None
    kept = [t for t in trades if t.entry_ts <= halt.ts]
    after = max([halt.ts, *(t.exit_ts for t in kept)])
    later = equity.index[equity.index > after]
    cut = later[0] if len(later) else equity.index[-1] if len(equity) else after
    return kept, equity[equity.index <= cut], halt


def median_or_nan(values: Sequence[float]) -> float:
    """Median ignoring NaN; NaN when nothing is left."""
    kept = [v for v in values if not math.isnan(v)]
    return statistics.median(kept) if kept else math.nan
