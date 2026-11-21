"""Backtest runner: drives the shared ``Engine`` over stored history.

Per symbol, base candles are resampled to the decision timeframe and features are computed once
on the full history (causal, so later rows never change earlier ones; rows before ``start`` are
warm-up only). The runner then walks the decision timeline (the sorted union of every symbol's
decision bars in ``[start, end)``). Between decisions T(k-1) and T(k) it replays, for each symbol
with an open position or queued order, the base bars with ``T(k-1) < ts <= T(k)``: the first one
fills queued orders at its open, every one is checked for stops and targets, and funding events
in the interval are settled at their timestamps. Events of all symbols run in time order.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from regimebot.config import AppConfig, timeframe_minutes
from regimebot.data.resample import resample
from regimebot.db import repo
from regimebot.engine.loop import Engine, RunLog
from regimebot.engine.portfolio import Portfolio
from regimebot.execution.fill_model import quote_from_bar_open
from regimebot.market_state.features import compute_features
from regimebot.risk.manager import RiskManager
from regimebot.strategy import Strategy, build_strategies
from regimebot.types import Bar, ExitReason, Mode, Trade

WINDOW_ROWS = 400  # trailing feature rows handed to strategies at each decision
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# Event kinds, in their order at equal timestamps: a bar closing at t, then funding stamped t,
# then a queued order filling at the open t of the next bar (so an entry at t pays no funding
# stamped t, and a position stopped in the bar closing at t pays none either).
_BAR, _FUNDING, _FILL = 0, 1, 2


@dataclass(frozen=True)
class BacktestResult:
    log: RunLog
    trades: list[Trade]
    equity: pd.Series  # equity after each decision (and the final flatten), UTC index
    cfg: AppConfig
    start: datetime
    end: datetime


@dataclass(frozen=True)
class _Feed:
    """One symbol's pre-sliced arrays: the inner loop never indexes a DataFrame by row."""

    features: pd.DataFrame
    feature_pos: np.ndarray  # row of each timeline decision in ``features``, -1 if absent
    bar_ts: np.ndarray  # int64 ns close times of base bars
    bars: np.ndarray  # (n, 5) open, high, low, close, volume
    bar_end: np.ndarray  # bars[: bar_end[k]] are the base bars with ts <= timeline[k]
    funding_ts: np.ndarray  # int64 ns
    funding_rate: np.ndarray
    funding_end: np.ndarray  # same as bar_end for funding events


def _ns(index: pd.Index) -> np.ndarray:
    return pd.DatetimeIndex(index).as_unit("ns").asi8


def _dt(ns: int) -> datetime:
    return _EPOCH + timedelta(microseconds=ns // 1000)


def _features(cfg: AppConfig, candles: pd.DataFrame, funding: pd.Series | None) -> pd.DataFrame:
    decision = resample(candles, cfg.exchange.base_timeframe, cfg.exchange.decision_timeframe)
    features = compute_features(decision, cfg.features, cfg.regime.compression_pctile, funding)
    return features.copy()  # copy() consolidates the columns into one block: faster slices


def _feed(
    features: pd.DataFrame, candles: pd.DataFrame, funding: pd.Series | None, timeline: np.ndarray
) -> _Feed:
    candles = candles.sort_index()
    funding = (funding if funding is not None else pd.Series(dtype=float)).dropna().sort_index()
    bar_ts = _ns(candles.index)
    funding_ts = _ns(funding.index) if len(funding) else np.empty(0, dtype=np.int64)
    return _Feed(
        features=features,
        feature_pos=pd.Index(_ns(features.index)).get_indexer(timeline),
        bar_ts=bar_ts,
        bars=candles[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float),
        bar_end=np.searchsorted(bar_ts, timeline, side="right"),
        funding_ts=funding_ts,
        funding_rate=funding.to_numpy(dtype=float),
        funding_end=np.searchsorted(funding_ts, timeline, side="right"),
    )


def _timeline(indexes: Sequence[pd.Index], start: datetime, end: datetime) -> np.ndarray:
    """Sorted union of decision-bar close times in ``[start, end)``, as int64 ns."""
    stamps = [_ns(index) for index in indexes]
    union = np.unique(np.concatenate(stamps)) if stamps else np.empty(0, dtype=np.int64)
    lo, hi = _ns(pd.to_datetime([start, end], utc=True))
    return np.asarray(union[(union >= lo) & (union < hi)])


def _replay_interval(
    engine: Engine, feeds: Mapping[str, _Feed], k: int, base_width_ns: int
) -> None:
    """Feed the base bars and funding events between decisions k-1 and k, in time order, for
    every symbol that has an open position or a queued order."""
    events: list[tuple[int, int, int, str, int]] = []  # (ns, kind, symbol order, symbol, row)
    for order, (symbol, feed) in enumerate(feeds.items()):
        if not engine.is_active(symbol):
            continue
        first, stop = int(feed.bar_end[k - 1]), int(feed.bar_end[k])
        if first < stop:  # queued orders fill at the open of the first bar
            events.append((int(feed.bar_ts[first]) - base_width_ns, _FILL, order, symbol, first))
        events.extend((int(feed.bar_ts[i]), _BAR, order, symbol, i) for i in range(first, stop))
        funding_rows = range(int(feed.funding_end[k - 1]), int(feed.funding_end[k]))
        events.extend((int(feed.funding_ts[j]), _FUNDING, order, symbol, j) for j in funding_rows)
    events.sort()
    for ns, kind, _, symbol, row in events:
        feed = feeds[symbol]
        if kind == _FILL:
            bid, ask = quote_from_bar_open(float(feed.bars[row, 0]), engine.cfg.costs)
            engine.fill_queued(symbol, _dt(ns), bid, ask, float(feed.bars[row, 4]))
        elif kind == _BAR:
            open_, high, low, close, volume = feed.bars[row].tolist()
            engine.on_minute(Bar(symbol, _dt(ns), open_, high, low, close, volume))
        else:  # mark price: close of the last base bar at or before the funding time
            last_bar = int(np.searchsorted(feed.bar_ts, ns, side="right")) - 1
            if last_bar >= 0:
                mark = float(feed.bars[last_bar, 3])
                engine.on_funding(symbol, _dt(ns), float(feed.funding_rate[row]), mark)


def run_backtest(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    kill_file: Path,
    strategies: Sequence[Strategy] | None = None,
) -> BacktestResult:
    """Backtest ``cfg.exchange.symbols`` on base-timeframe ``candles`` for decisions in
    ``[start, end)``; open positions are flattened at the last decision (END_OF_DATA).

    ``strategies`` defaults to ``build_strategies(cfg)``. ``candles`` must hold exactly the
    configured symbols; ``funding`` may lack symbols (no funding charged).
    """
    for bound in (start, end):
        if bound.utcoffset() is None:
            raise ValueError(f"start/end must be timezone-aware UTC, got {bound!r}")
    symbols = list(cfg.exchange.symbols)
    if set(candles) != set(symbols):
        raise ValueError(f"candles cover {sorted(candles)}, config trades {sorted(symbols)}")
    features = {s: _features(cfg, candles[s], funding.get(s)) for s in symbols}
    timeline = _timeline([f.index for f in features.values()], start, end)
    feeds = {s: _feed(features[s], candles[s], funding.get(s), timeline) for s in symbols}
    base_width_ns = timeframe_minutes(cfg.exchange.base_timeframe) * 60 * 10**9

    portfolio = Portfolio(cfg.backtest.initial_equity)
    log = RunLog()
    risk = RiskManager(cfg, kill_file)
    strategies = build_strategies(cfg) if strategies is None else strategies
    engine = Engine(cfg, strategies, risk, portfolio, log)
    decision_times = [_dt(int(t)) for t in timeline]
    for k, ts in enumerate(decision_times):
        if k > 0:
            _replay_interval(engine, feeds, k, base_width_ns)
        windows = {}
        for symbol, feed in feeds.items():
            pos = int(feed.feature_pos[k])
            if pos >= 0:
                windows[symbol] = feed.features.iloc[max(0, pos - WINDOW_ROWS + 1) : pos + 1]
        engine.on_decision(ts, windows)
    if decision_times:
        # ponytail: base bars after the last decision (under one decision bar, before ``end``)
        # are not replayed, so a stop or target there is missed; replay them first if it matters.
        engine.flatten_all(decision_times[-1], dict(portfolio.last_prices), ExitReason.END_OF_DATA)

    equity = pd.Series(
        [e[1] for e in log.equity],
        index=pd.DatetimeIndex([e[0] for e in log.equity], tz=UTC, name="ts"),
        name="equity",
        dtype="float64",
    )
    return BacktestResult(log, log.trades, equity, cfg, start, end)


def persist_result(session: Session, result: BacktestResult, note: str = "") -> int:
    """Store a backtest as a new run with all its log rows; returns the run id."""
    log = result.log
    run_id = repo.create_run(session, Mode.BACKTEST, result.cfg, note)
    signal_ids = repo.save_signals(session, run_id, log.signals)
    repo.save_risk_decisions(
        session, run_id, [(signal_ids[i], ts, decision) for i, ts, decision in log.decisions]
    )
    repo.save_fills(session, run_id, log.fills)
    repo.save_trades(session, run_id, log.trades)
    repo.save_regimes(session, run_id, log.regimes)
    repo.save_risk_events(session, run_id, log.risk_events)
    repo.save_equity(session, run_id, log.equity)
    repo.end_run(session, run_id)
    return run_id
