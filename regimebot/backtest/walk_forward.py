"""Walk-forward optimisation: fit on a train span, evaluate on the span right after it.

Bookkeeping that keeps the result honest:

* parameters are chosen from train-span metrics only; the test span is run once, with them;
* every run (train or test) gets a fresh kill file path and fresh portfolio/risk state, so each
  test window starts from ``backtest.initial_equity``. Equity is NOT chained across windows:
  ``oos_net_pnl`` is the plain sum of the windows' net pnl, each measured on the same capital;
* ``n_variants_tried`` counts every distinct (window, grid combination) fitted, which is what a
  multiple-testing correction needs.
"""

import itertools
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import structlog

from regimebot.backtest.runner import run_backtest
from regimebot.config import AppConfig, with_overrides
from regimebot.reporting.metrics import Metrics, compute_metrics
from regimebot.strategy import Strategy
from regimebot.types import Trade

log = structlog.get_logger(__name__)

# ponytail: every run recomputes resampling and features from the same candles; precompute the
# features once per config (they depend only on ``cfg.features``/``cfg.regime``) if walk-forward
# runtime (about 40 runs of up to 60 s on 2 years x 3 symbols) becomes the bottleneck.


@dataclass(frozen=True)
class WalkForwardResult:
    """``windows``: one row per window. ``oos_trades``: the test windows' trades in time order.
    Test windows are independent runs from initial equity (equity is not chained)."""

    windows: pd.DataFrame
    oos_trades: list[Trade]
    n_variants_tried: int
    oos_net_pnl: float


def run_metrics(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    strategies: Sequence[Strategy] | None = None,
) -> tuple[Metrics, list[Trade]]:
    """One backtest on a fresh kill file path (it does not exist, so nothing is halted)."""
    with tempfile.TemporaryDirectory() as tmp:
        result = run_backtest(cfg, candles, funding, start, end, Path(tmp) / "KILL", strategies)
    metrics = compute_metrics(result.trades, result.equity, cfg.backtest.initial_equity)
    return metrics, result.trades


def _qualifies(m: Metrics, min_trades: int) -> bool:
    return m.n_trades >= min_trades and not math.isnan(m.profit_factor)


def walk_forward(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    strategies: Sequence[Strategy] | None = None,
) -> WalkForwardResult:
    """Window i: train ``[start + i*test_days, +train_days)``, test = the next ``test_days``;
    windows stop when a test span would pass ``end``. Raises ``ValueError`` if none fits.

    On each train span every combination of ``cfg.walk_forward.grid`` is run; the one with the
    highest profit factor among those with at least ``min_trades`` trades wins (NaN does not
    qualify, ``inf`` is highest, ties go to the first in grid order). With no qualifier the test
    span runs on the unmodified config and the row is flagged ``no_qualifying_params``.
    """
    wf = cfg.walk_forward
    train, test = timedelta(days=wf.train_days), timedelta(days=wf.test_days)
    keys = list(wf.grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*wf.grid.values())]
    # built once: an invalid combination fails here, before any backtest runs
    variants = [with_overrides(cfg, params) for params in combos]
    rows: list[dict[str, object]] = []
    oos: list[Trade] = []
    i = 0
    while (train_start := start + i * test) + train + test <= end:
        train_end = train_start + train
        test_end = train_end + test
        best: tuple[float, dict[str, float], AppConfig] | None = None
        best_m: Metrics | None = None
        for n, (params, variant) in enumerate(zip(combos, variants, strict=True), 1):
            m, _ = run_metrics(variant, candles, funding, train_start, train_end, strategies)
            log.info("walk_forward_train", window=i, combo=n, of=len(combos), n_trades=m.n_trades)
            if _qualifies(m, wf.min_trades) and (best is None or m.profit_factor > best[0]):
                best, best_m = (m.profit_factor, params, variant), m
        chosen, test_cfg = (best[1], best[2]) if best else ({}, cfg)
        if best_m is None:  # nothing qualified: report the unmodified config's train run
            log.warning("walk_forward_no_qualifying_params", window=i, min_trades=wf.min_trades)
            best_m, _ = run_metrics(cfg, candles, funding, train_start, train_end, strategies)
        tm, trades = run_metrics(test_cfg, candles, funding, train_end, test_end, strategies)
        log.info("walk_forward_test", window=i, n_trades=tm.n_trades, net_pnl=tm.net_pnl)
        oos.extend(trades)
        rows.append(
            {
                "window": i,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "params": json.dumps(chosen, sort_keys=True),
                "no_qualifying_params": best is None,
                "train_n_trades": best_m.n_trades,
                "train_profit_factor": best_m.profit_factor,
                "train_net_pnl": best_m.net_pnl,
                "test_n_trades": tm.n_trades,
                "test_profit_factor": tm.profit_factor,
                "test_net_pnl": tm.net_pnl,
            }
        )
        i += 1
    if not rows:
        raise ValueError(
            f"no walk-forward window fits: need {wf.train_days + wf.test_days} days between "
            f"{start} and {end}"
        )
    windows = pd.DataFrame(rows)
    return WalkForwardResult(
        windows=windows,
        oos_trades=sorted(oos, key=lambda t: t.entry_ts),
        n_variants_tried=len(rows) * len(combos),
        oos_net_pnl=float(windows["test_net_pnl"].sum()),
    )
