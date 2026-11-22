"""Daily trend reports: R- and stop/target-based metrics are n/a (not zeros), the time-shift
baseline reruns the engine with each symbol's daily target weights shifted in time, and the CLI
runs and reports a trend backtest selected with REGIMEBOT_CONFIG."""

import math
from datetime import timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from synth import BTC, ETH, SOL, T0, candles_from_closes, funding_every_8h
from test_daily_trend_engine import TREND_YAML
from typer.testing import CliRunner

from regimebot.backtest.baselines import Baselines, compute_baselines, weight_table
from regimebot.backtest.runner import BacktestResult, run_backtest
from regimebot.cli import app, history_start
from regimebot.config import CONFIG_ENV, AppConfig, load_config
from regimebot.db import repo
from regimebot.db.models import Run
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.reporting.metrics import breakdown, compute_metrics, strategy_quality
from regimebot.reporting.report import write_report

DAY = timedelta(days=1)
DAYS = 90
START = T0 + 9 * DAY


def cfg(**extra: object) -> AppConfig:
    return load_config(
        TREND_YAML,
        overrides={
            "exchange.symbols": [BTC, ETH],
            "strategies.daily_trend.lookbacks": [2, 3, 5, 8],
            "strategies.daily_trend.vol_window": 3,
            "strategies.daily_trend.target_vol": 1.0,
            "paper.warmup_bars": 9,
            **extra,
        },
    )


def market(days: int = DAYS) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    rng = np.random.default_rng(4)
    candles = {}
    for symbol, price in ((BTC, 100.0), (ETH, 50.0)):
        closes = price * np.exp(np.cumsum(rng.normal(0.0, 0.04, days)))
        path = np.concatenate([[price], closes])
        minutes = np.concatenate([np.linspace(a, b, 1441)[1:] for a, b in pairwise(path)])
        candles[symbol] = candles_from_closes(minutes)
    funding = {s: funding_every_8h(T0, 3 * days, seed=i) for i, s in enumerate((BTC, ETH))}
    return candles, funding


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[BacktestResult, Baselines]:
    candles, funding = market()
    end = T0 + DAYS * DAY
    kill = tmp_path_factory.mktemp("kill") / "KILL"
    result = run_backtest(cfg(), candles, funding, START, end, kill)
    return result, compute_baselines(result, candles, funding, 0, seeds=(0, 1, 2))


def test_r_and_stop_or_target_metrics_are_nan_for_target_trades(
    run: tuple[BacktestResult, Baselines],
) -> None:
    result, _ = run
    m = compute_metrics(result.trades, result.equity, 10_000.0)
    assert m.n_trades >= 5
    for name in ("expectancy_r", "target_hit_rate", "stop_out_rate"):
        assert math.isnan(getattr(m, name)), name
    assert math.isfinite(m.net_pnl) and m.fees > 0 and m.funding != 0
    assert m.net_pnl == pytest.approx(sum(t.net_pnl for t in result.trades))
    quality = strategy_quality(result.trades, {})
    assert quality.loc["daily_trend"].isna().all()
    assert breakdown(result.trades, "symbol")["expectancy_r"].isna().all()
    assert set(breakdown(result.trades, "side").index) == {"long", "short"}


def test_the_time_shift_baseline_shifts_each_symbols_weights_and_reruns_the_engine(
    run: tuple[BacktestResult, Baselines],
) -> None:
    result, baselines = run
    candles, funding = market()
    table = weight_table(result.cfg, candles, funding, result.start, result.end)
    assert list(table.columns) == [BTC, ETH] and table.index[0] == START
    assert len(table) == len(result.equity) and table.notna().all().all()
    runs = baselines.time_shift
    assert baselines.time_shift_seeds == (0, 1, 2) and len(runs) == 3
    assert baselines.random_entry == ()
    nets = [sum(t.net_pnl for t in r.trades) for r in runs]
    assert len(set(nets)) == 3 and sum(t.net_pnl for t in result.trades) not in nets
    # the same engine: shifted runs trade target positions with the same costs and caps
    assert all(r.trades and all(t.strategy == "daily_trend" for t in r.trades) for r in runs)
    again = compute_baselines(result, candles, funding, 0, seeds=(1,))
    assert [repr(t) for t in again.time_shift[0].trades] == [repr(t) for t in runs[1].trades]


def test_the_report_shows_na_for_r_stop_and_target_and_the_time_shift_baseline(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, baselines = run
    text = write_report(result, baselines, tmp_path).read_text("utf-8")
    for label in (
        "Expectancy (R per trade)",
        "Target hit rate (all strategies; split by strategy below)",
        "Stop-out rate",
    ):
        assert f"<tr><td>{label}</td><td>n/a</td>" in text, label
    assert "n/a (no stop: R is not defined)" in text
    assert ">Time-shift baseline</h2>" in text and ">Random-entry baseline</h2>" not in text
    assert "Time-shifted (median of runs, range)" in text
    assert "Strategy beat " in text and " of 3 time-shifted runs</strong> on net P&amp;L" in text
    assert "n/a: daily_trend uses no regimes" in text
    assert "<td>daily_trend</td>" in text and "Breakdown by side" in text


def test_history_start_covers_the_trend_warm_up() -> None:
    trend = load_config(TREND_YAML)
    assert history_start(trend) == trend.backtest.start - timedelta(days=101 + 1)


def test_cli_runs_and_reports_a_trend_backtest_selected_by_regimebot_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "REGIMEBOT_CONFIG" in CliRunner().invoke(app, ["--help"]).output
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    upgrade_to_head(url)
    db = make_engine(url)
    days = 30
    candles, funding = market(days)
    with session_scope(db) as s:
        for symbol in (BTC, ETH, SOL):
            source = symbol if symbol in candles else ETH
            repo.upsert_candles(s, symbol, "1m", candles[source])
            repo.upsert_funding(s, symbol, funding[source])
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv(CONFIG_ENV, str(TREND_YAML))
    monkeypatch.chdir(tmp_path)
    short = ["strategies.daily_trend.lookbacks=[2,3,5,8]", "strategies.daily_trend.vol_window=3"]
    args = [
        "backtest",
        "run",
        "--start",
        (START).isoformat(),
        "--end",
        (T0 + days * DAY).isoformat(),
    ]
    done = CliRunner().invoke(app, [*args, *(a for kv in short for a in ("--set", kv))])
    assert done.exit_code == 0, done.output
    assert "warning:" not in done.output and "trades=" in done.output
    with session_scope(db) as s:
        (run_id,) = s.scalars(select(Run.id)).all()
    reported = CliRunner().invoke(app, ["report", str(run_id), "--random-seeds", "2"])
    assert reported.exit_code == 0, reported.output
    html = (tmp_path / "reports" / f"run_{run_id}" / "report.html").read_text("utf-8")
    assert " of 2 time-shifted runs</strong>" in html
    db.dispose()
