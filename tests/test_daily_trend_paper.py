"""Daily trend in paper trading: the same fills as the backtest on the same scripted market, one
decision per daily close, a mid-day start acting at the next close, and a resume that does not
trade a decision twice.

The market: 1m bars for BTC and ETH whose daily closes swing (sine waves), so the short-lookback
rule opens, rebalances, flips and closes; funding every 8 h. As in ``test_paper``, polls run
``LAG`` after the minute and a live fill is priced from the forming bar's open, so the only
difference from the backtest is a live fill's timestamp (LAG later).
"""

import math
import re
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fakes import Clock, ReplayExchange
from synth import BTC, ETH, MINUTE, T0, candles_from_closes, funding_every_8h
from test_daily_trend_engine import TREND_YAML
from typer.testing import CliRunner

from regimebot.backtest.runner import run_backtest
from regimebot.cli import app
from regimebot.config import AppConfig, load_config
from regimebot.db import repo
from regimebot.db.models import Base
from regimebot.db.session import make_engine, session_scope
from regimebot.engine.loop import RunLog
from regimebot.paper import session as paper_session
from regimebot.paper.session import PaperSession
from regimebot.types import ExitReason, Fill, Side, Trade

DAY = timedelta(days=1)
LAG = timedelta(seconds=3)
DAYS = 30  # of market data; the last day only gives the final decision's orders a bar to fill
START = T0 + 9 * DAY  # first decision: 9 closes warm up lookbacks [2, 3, 5, 8] and vol_window 3
END = T0 + (DAYS - 1) * DAY  # last decision


def cfg() -> AppConfig:
    """``config/trend.yaml`` on BTC and ETH with short windows; paper loads all of the market's
    history (``warmup_bars`` 40 days), as the backtest does."""
    return load_config(
        TREND_YAML,
        overrides={
            "exchange.symbols": [BTC, ETH],
            "strategies.daily_trend.lookbacks": [2, 3, 5, 8],
            "strategies.daily_trend.vol_window": 3,
            "strategies.daily_trend.target_vol": 1.0,
            "paper.warmup_bars": 40,
        },
    )


def market() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    def minutes(closes: list[float]) -> list[float]:
        path = [closes[0], *closes]
        return [a + (b - a) * (i + 1) / 1440 for a, b in pairwise(path) for i in range(1440)]

    btc = [100 * (1 + 0.12 * math.sin(k / 2.2)) for k in range(DAYS)]
    eth = [50 * (1 + 0.15 * math.cos(k / 3.1)) for k in range(DAYS)]
    candles = {BTC: candles_from_closes(minutes(btc)), ETH: candles_from_closes(minutes(eth))}
    funding = {s: funding_every_8h(T0, 3 * DAYS, seed=i) for i, s in enumerate((BTC, ETH))}
    return candles, funding


@pytest.fixture(autouse=True)
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(paper_session, "alert", lambda kind, detail: sent.append((kind, detail)))
    return sent


class Rig:
    def __init__(self, tmp_path: Path, now: datetime) -> None:
        self.cfg = cfg()
        candles, funding = market()
        self.clock = Clock(now)
        self.exchange = ReplayExchange(candles, funding, self.cfg.costs, self.clock)
        self.db = make_engine(f"sqlite:///{tmp_path / 'paper.db'}")
        Base.metadata.create_all(self.db)
        self.kill_file = tmp_path / "KILL"

    def session(self) -> PaperSession:
        return PaperSession(self.cfg, self.exchange, self.db, self.kill_file, clock=self.clock)

    def poll_daily(self, session: PaperSession, until: datetime) -> None:
        """Poll now, then LAG after every daily close and every midday, up to ``until`` + LAG."""
        half = timedelta(hours=12)
        while self.clock.now <= until + LAG:
            session.run_once()
            self.clock.now = T0 + ((self.clock.now - LAG - T0) // half + 1) * half + LAG

    def stored(self, run_id: int) -> dict[str, list[Any]]:
        with session_scope(self.db) as s:
            return {
                "fills": [repr(f) for f in repo.load_fills(s, run_id)],
                "trades": [repr(t) for t in repo.load_trades(s, run_id)],
                "decisions": repo.load_risk_decisions(s, run_id),
                "equity": repo.load_equity_snapshots(s, run_id),
            }


def unlag(ts: datetime) -> datetime:
    return ts - LAG if (ts - T0) % MINUTE == LAG else ts


def unlag_fill(f: Fill) -> Fill:
    return replace(f, ts=unlag(f.ts))


def unlag_trade(t: Trade) -> Trade:
    return replace(t, entry_ts=unlag(t.entry_ts), exit_ts=unlag(t.exit_ts))


def before_end(log: RunLog) -> dict[str, Any]:
    """What a run did up to (not including) the last decision, whose orders the backtest
    cancels and flattens (END_OF_DATA) while paper trades them; NaN-safe (``repr``)."""
    n = sum(ts < END for _, ts, _ in log.decisions)
    return {
        "decisions": log.decisions[:n],
        "fills": [repr(f) for f in log.fills if f.ts < END],
        "trades": [repr(t) for t in log.trades if t.exit_ts < END],
        "equity": [e for e in log.equity if e[0] < END],
        "risk_events": log.risk_events,
    }


def kinds(log: RunLog) -> set[str]:
    """What the approved orders did to the position: open, add, reduce, close, flip."""
    out = set()
    for _, _, d in log.decisions:
        m = re.search(r"target ([-+][^ ]+) from ([-+][^ ]+)", d.detail)
        if d.approved and m:
            target, current = float(m[1]), float(m[2])
            if current == 0:
                out.add("open")
            elif target == 0:
                out.add("close")
            elif target * current < 0:
                out.add("flip")
            else:
                out.add("add" if abs(target) > abs(current) else "reduce")
    return out


def backtest(tmp_path: Path) -> RunLog:
    candles, funding = market()
    return run_backtest(cfg(), candles, funding, START, END + MINUTE, tmp_path / "BT_KILL").log


def test_paper_trades_the_daily_trend_exactly_like_the_backtest(tmp_path: Path) -> None:
    rig = Rig(tmp_path, START + LAG)
    session = rig.session()
    session.start()
    rig.poll_daily(session, END)
    ref = backtest(tmp_path)
    paper = session.log
    paper_unlagged = RunLog(
        decisions=paper.decisions,
        fills=[unlag_fill(f) for f in paper.fills],
        trades=[unlag_trade(t) for t in paper.trades],
        equity=paper.equity,
        risk_events=paper.risk_events,
    )
    assert before_end(paper_unlagged) == before_end(ref)
    assert paper.fills[0].ts == START + LAG  # live fills, priced at the same quote
    # the scripted market makes the rule do everything, and funding is charged and received
    assert kinds(ref) == {"open", "add", "reduce", "close", "flip"}
    sides = {t.side for t in ref.trades if t.exit_reason == ExitReason.SIGNAL}
    assert sides == {Side.LONG, Side.SHORT}
    assert any(t.funding > 0 for t in ref.trades) and any(t.funding < 0 for t in ref.trades)
    assert all(math.isnan(t.r_multiple) for t in ref.trades)
    assert ref.risk_events == [] and ref.signals == paper.signals == []
    # one decision per daily close, though two polls fall in each day
    assert [e[0] for e in paper.equity] == [START + k * DAY for k in range(DAYS - 9)]


def test_a_session_started_mid_day_acts_at_the_next_daily_close(
    tmp_path: Path, alerts: list[tuple[str, str]]
) -> None:
    rig = Rig(tmp_path, START + 9 * timedelta(hours=1))
    session = rig.session()
    session.start()
    rig.poll_daily(session, START + 3 * DAY)
    assert [e[0] for e in session.log.equity] == [START + k * DAY for k in (1, 2, 3)]
    assert alerts == []  # the START close was before the session began: routine, not alerted


def test_a_resume_after_a_decision_does_not_trade_it_again(tmp_path: Path) -> None:
    (tmp_path / "whole").mkdir()
    (tmp_path / "split").mkdir()
    whole = Rig(tmp_path / "whole", START + LAG)
    uninterrupted = whole.session()
    whole_id = uninterrupted.start()
    whole.poll_daily(uninterrupted, START + 6 * DAY)

    rig = Rig(tmp_path / "split", START + LAG)
    first = rig.session()
    run_id = first.start()
    rig.poll_daily(first, START + 3 * DAY)  # decides START + 3 days and fills at + LAG
    acted = START + 3 * DAY
    assert any(ts == acted for _, ts, _ in first.log.decisions)
    first.close()
    rig.clock.now = acted + 2 * timedelta(hours=1)  # restarted two hours later, same day
    resumed = rig.session()
    resumed.start(resume_run_id=run_id)
    rig.poll_daily(resumed, START + 6 * DAY)
    assert not any(ts == acted for _, ts, _ in resumed.log.decisions)  # not decided again
    assert rig.stored(run_id) == whole.stored(whole_id)


def test_report_and_status_of_a_paper_trend_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, START + LAG)
    session = rig.session()
    run_id = session.start()
    rig.poll_daily(session, START + 4 * DAY)
    session.close()
    assert session.portfolio.positions  # still holding target positions
    monkeypatch.setenv("DATABASE_URL", str(rig.db.url))
    monkeypatch.chdir(tmp_path)
    reported = CliRunner().invoke(app, ["report", str(run_id)])
    assert reported.exit_code == 0, reported.output
    html = (tmp_path / "reports" / f"run_{run_id}" / "report.html").read_text("utf-8")
    assert "PAPER TRADING RUN" in html and "Time-shift baseline: n/a: backtest only" in html
    status = CliRunner().invoke(app, ["paper", "status"])
    assert "stop=" not in status.output and "(daily_trend)" in status.output
