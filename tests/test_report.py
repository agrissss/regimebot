"""Report files, honesty content and HTML safety."""

import math
from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from synth import BTC, T0, regime_walk
from typer.testing import CliRunner

from regimebot.backtest.baselines import Baselines, compute_baselines, random_entry
from regimebot.backtest.runner import BacktestResult
from regimebot.cli import app
from regimebot.config import AppConfig, config_hash, load_config
from regimebot.db import repo
from regimebot.db.models import Run
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.reporting.metrics import active_period, compute_metrics, first_permanent_halt
from regimebot.reporting.report import RISK_WARNING, write_report
from regimebot.types import Mode, Regime, RiskEvent, Trade

HEADINGS = [
    "Risk warning",
    "Headline metrics",
    "Equity and drawdown",
    "Breakdown by regime",
    "Breakdown by symbol",
    "Breakdown by strategy",
    "Breakdown by exit reason",
    "Slippage: modelled versus realised",
    "Risk-limit events",
    "Result without the best 5% of trades",
]


def open_cfg(**extra: object) -> AppConfig:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    overrides: dict[str, object] = {f"playbook.{r.value}": entry for r in Regime}
    return load_config(overrides={"exchange.symbols": [BTC], **overrides, **extra})


@pytest.fixture(scope="module")
def run() -> tuple[BacktestResult, Baselines]:
    """A reference that halts permanently (tiny max drawdown), with baselines for two seeds."""
    cfg = open_cfg(**{"risk.max_drawdown": 0.003})
    candles, start, end = {BTC: regime_walk(3 * 1440, seed=5)}, T0, T0 + timedelta(days=3)
    result = random_entry(cfg, candles, {}, start, end, 7, p=0.3)
    assert result.trades and first_permanent_halt(result.log.risk_events) is not None
    halt = first_permanent_halt(result.log.risk_events)
    assert halt is not None and halt.ts < result.equity.index[-1]
    result.log.risk_events.append(RiskEvent(start, "<b>kind</b>", "<script>x</script>", "skip"))
    result.log.risk_events.extend(
        RiskEvent(halt.ts + timedelta(minutes=i), "stale_data", "d", "skip_symbol")
        for i in range(60)
    )
    baselines = compute_baselines(result, candles, {}, len(result.log.signals), seeds=(0, 1))
    return result, baselines


def test_files_are_created_and_html_is_self_contained(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, baselines = run
    path = write_report(result, baselines, tmp_path / "out", n_variants_tried=12)
    out = tmp_path / "out"
    assert path == out / "report.html"
    for name in ("trades.csv", "daily_pnl.csv", "equity.png", "drawdown.png"):
        assert (out / name).stat().st_size > 0
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<script" not in text and "http://" not in text and "https://" not in text
    assert 'src="data:image/png;base64,' in text
    csv_rows = (out / "trades.csv").read_text().strip().splitlines()[1:]
    assert len(csv_rows) == len(result.trades)


def test_html_has_warning_first_every_heading_and_honest_notes(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, baselines = run
    text = write_report(result, baselines, tmp_path, n_variants_tried=12).read_text("utf-8")
    assert text.index("Risk warning") < min(text.index(h) for h in HEADINGS[1:])
    for h in HEADINGS:
        assert f">{h}</h2>" in text
    for sentence in RISK_WARNING:
        assert sentence in text
    assert len(RISK_WARNING) == 6
    assert config_hash(result.cfg) in text
    assert "12 parameter variants" in text and "likely overstated" in text
    # BTC alone, in the default group capped at 1.0: the 1.5 gross cap cannot be reached
    assert "Leverage: configured gross cap 1.50x, effective cap 1.00x (one group" in text
    assert "Buy and hold" in text and "Random entry" in text
    assert "Parameter variants tried" in text
    assert "all modelled fees (taker and maker)" in text
    assert "gaps through stop prices" in text and "next open" in text
    assert "only asked in regimes where breakout is allowed" in text
    for seed_run in baselines.random_entry:  # the per-seed table shows each period end
        _, eq, _ = active_period(seed_run.trades, seed_run.equity, seed_run.log.risk_events)
        assert f"<td>{eq.index[-1].isoformat()}</td>" in text
    assert "Period end (UTC)" in text and "Decision bars" in text
    assert "Seeds used: 0, 1" in text and "Random seed 1" in text and "Strategy (reference)" in text


def test_halted_run_reports_the_active_period_and_truncated_baselines(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, baselines = run
    text = write_report(result, baselines, tmp_path).read_text("utf-8")
    halt = first_permanent_halt(result.log.risk_events)
    assert halt is not None
    sentence = f"Trading halted on {halt.ts.isoformat()} ({halt.kind}). No entries after that;"
    assert text.count(sentence) == 2  # headline block and Run section
    trades, equity, _ = active_period(result.trades, result.equity, result.log.risk_events)
    cover = f"results below cover {result.start.isoformat()} to {equity.index[-1].isoformat()} "
    assert cover + "(includes the kill-switch flatten)" in text
    assert equity.index[-1] > halt.ts  # the realising snapshot is the bar after the halt
    # metrics are those of the truncated trades and equity
    assert len(equity) < len(result.equity)
    initial = result.cfg.backtest.initial_equity
    m = compute_metrics(trades, equity, initial, result.log.fills, result.gross_exposure)
    assert f"<td>{m.net_pnl:,.2f}</td>" in text
    assert m.max_gross_leverage_seen > 0
    label = "Max gross leverage seen (gross exposure / equity)"
    assert f"<td>{label}</td><td>{m.max_gross_leverage_seen:.2f}</td>" in text
    assert (
        f"<td>{'n/a' if math.isnan(m.sharpe) else f'{m.sharpe:.2f}'}</td>" in text
        and f"<td>{m.exposure_pct * 100:.2f}%</td>" in text
    )
    # buy and hold: active period beside the strategy, the full period as a separate line
    bh, full = baselines.buy_and_hold, baselines.buy_and_hold_full
    assert bh is not None and full is not None
    assert bh.index[-1] == equity.index[-1] >= halt.ts and bh.index[-1] < full.index[-1]
    initial = result.cfg.backtest.initial_equity
    assert f"<td>{bh.iloc[-1] - initial:,.2f}</td>" in text
    assert "Buy and hold over the full period" in text and f"{full.iloc[-1] - initial:,.2f}" in text


def test_events_always_list_halts_summarise_counts_and_cap_the_rest(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, baselines = run
    text = write_report(result, baselines, tmp_path).read_text("utf-8")
    halt = first_permanent_halt(result.log.risk_events)
    assert halt is not None
    assert f"<td>{halt.ts.isoformat()}</td><td>max_drawdown</td><td>halt</td>" in text
    assert "<td>stale_data</td><td>skip_symbol</td><td>60</td>" in text  # count summary
    # three days of history cannot warm the default windows: one warmup_incomplete event first
    assert "<td>warmup_incomplete</td><td>no_trade_until_warm</td><td>1</td>" in text
    # no funding was given: one funding_missing event, shown, and not a halt
    assert "<td>funding_missing</td><td>funding_not_charged</td><td>1</td>" in text
    assert "<td>funding_missing</td><td>halt</td>" not in text
    assert "All 1 halt events are listed, plus the first 50 of 63 other events" in text
    assert (
        text.count("<td>stale_data</td><td>skip_symbol</td><td>d</td>") == 47
    )  # 50 others: the warm-up and funding events, the escaped event and 47 of these


def test_slippage_table_shows_floor_and_realised_values(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, baselines = run
    text = write_report(result, baselines, tmp_path).read_text("utf-8")
    m = compute_metrics(result.trades, result.equity, 1.0, result.log.fills)
    costs = result.cfg.costs
    floor = f"{costs.half_spread_bps + costs.slippage_bps:.2f} bps"
    assert f"<td>{floor}</td><td>{floor}</td>" in text
    entry, exit_ = m.avg_entry_slippage_bps, m.avg_exit_slippage_bps
    assert f"<td>{entry:.2f} bps</td><td>{exit_:.2f} bps</td>" in text


def test_variants_section_absent_when_not_supplied_and_missing_baselines_show_na(
    run: tuple[BacktestResult, Baselines], tmp_path: Path
) -> None:
    result, _ = run
    text = write_report(result, Baselines(), tmp_path).read_text("utf-8")
    assert "Parameter variants tried" not in text
    assert "n/a" in text and "Strategy" in text


def test_untrusted_text_is_escaped(run: tuple[BacktestResult, Baselines], tmp_path: Path) -> None:
    result, baselines = run
    text = write_report(result, baselines, tmp_path).read_text("utf-8")
    assert "<script>x</script>" not in text and "&lt;script&gt;x&lt;/script&gt;" in text
    assert "&lt;b&gt;kind&lt;/b&gt;" in text


def test_report_with_no_trades_still_renders(tmp_path: Path) -> None:
    candles = {BTC: regime_walk(1440, seed=3)}
    # windows short enough (48 bars) for the 12 h before start to warm them: no risk event
    short = {"features.pctile_window": 24, "features.rv_window": 24, "features.ema_slow": 48}
    short |= {f"features.{w}_window": 24 for w in ("volume", "donchian", "zscore")}
    cfg = open_cfg(**short, **{"features.ema_fast": 12})
    result = random_entry(
        cfg, candles, {}, T0 + timedelta(hours=12), T0 + timedelta(days=1), 1, p=0.0
    )
    assert not result.trades and not result.log.risk_events
    text = write_report(result, Baselines(), tmp_path).read_text("utf-8")
    assert "No risk-limit events" in text and "n/a" in text
    header = (tmp_path / "trades.csv").read_text().strip()
    assert header == ",".join(f.name for f in fields(Trade))


def test_cli_set_override_is_stored_and_report_rebuilds_from_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    upgrade_to_head(url)
    engine = make_engine(url)
    with session_scope(engine) as session:
        for i, symbol in enumerate((BTC, "ETH/USDT:USDT", "SOL/USDT:USDT")):
            repo.upsert_candles(session, symbol, "1m", regime_walk(3 * 1440, seed=i))
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = ["backtest", "run", "--start", "2024-01-02", "--end", "2024-01-04"]
    bad = runner.invoke(app, [*args, "--set", "costs.nope=1"])
    assert bad.exit_code != 0
    ok = runner.invoke(
        app, [*args, "--set", "costs.taker_fee_bps=7", "--set", "risk.risk_per_trade=0.005"]
    )
    assert ok.exit_code == 0, ok.output
    # no funding is stored: the run says it charged none, in one line
    (warning,) = [line for line in ok.output.splitlines() if line.startswith("warning: funding")]
    gap = "no funding stamp from 2024-01-02T00:00:00+00:00 to 2024-01-04T00:00:00+00:00"
    assert all(f"{s}: {gap}" in warning for s in (BTC, "ETH/USDT:USDT", "SOL/USDT:USDT"))
    assert warning.endswith("run `regimebot data download` for the funding history")
    with session_scope(engine) as session:
        (run_id,) = session.scalars(select(Run.id)).all()
        run = repo.load_run(session, run_id)
    assert '"taker_fee_bps":7' in run.config_json and '"risk_per_trade":0.005' in run.config_json

    done = runner.invoke(app, ["report", str(run.id), "--variants-tried", "3"])
    assert done.exit_code == 0, done.output
    html_text = (tmp_path / "reports" / f"run_{run.id}" / "report.html").read_text("utf-8")
    assert "3 parameter variants" in html_text and run.config_hash in html_text
    # gross exposure comes back from the stored equity snapshots
    leverage = "<td>Max gross leverage seen (gross exposure / equity)</td><td>"
    assert leverage in html_text and leverage + "n/a" not in html_text
    assert runner.invoke(app, ["report", "999"]).exit_code != 0
    with session_scope(engine) as session:
        paper_id = repo.create_run(session, Mode.PAPER, load_config())
    refused = runner.invoke(app, ["report", str(paper_id)])
    assert refused.exit_code != 0 and "paper run" in refused.output
    engine.dispose()
