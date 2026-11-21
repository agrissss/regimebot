"""Report files, honesty content and HTML safety."""

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from synth import BTC, T0, regime_walk
from typer.testing import CliRunner

from regimebot.backtest.baselines import Baselines, compute_baselines, random_entry
from regimebot.backtest.runner import BacktestResult
from regimebot.cli import app
from regimebot.config import config_hash, load_config
from regimebot.db import repo
from regimebot.db.models import Run
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.reporting.report import RISK_WARNING, write_report
from regimebot.types import Regime, RiskEvent

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


@pytest.fixture(scope="module")
def run() -> tuple[BacktestResult, Baselines]:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    overrides: dict[str, object] = {f"playbook.{r.value}": entry for r in Regime}
    cfg = load_config(overrides={"exchange.symbols": [BTC], **overrides})
    candles, start, end = {BTC: regime_walk(3 * 1440, seed=5)}, T0, T0 + timedelta(days=3)
    result = random_entry(
        cfg, candles, {}, start, end, 7, n_reference_signals=90, n_decision_bars=288
    )
    assert result.trades
    result.log.risk_events.append(RiskEvent(start, "<b>kind</b>", "<script>x</script>", "halt"))
    return result, compute_baselines(result, candles, {}, n_signals=len(result.log.signals))


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
        assert f"<h2>{h}</h2>" in text or f">{h}</h2>" in text
    for sentence in RISK_WARNING:
        assert sentence in text
    assert len(RISK_WARNING) == 6
    assert config_hash(result.cfg) in text
    assert "12 parameter variants" in text and "likely overstated" in text
    assert "Buy and hold" in text and "Random entry" in text
    assert "Parameter variants tried" in text


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
    cfg = load_config(overrides={"exchange.symbols": [BTC]})
    candles = {BTC: regime_walk(1440, seed=3)}
    result = random_entry(
        cfg, candles, {}, T0, T0 + timedelta(days=1), 1, n_reference_signals=0, n_decision_bars=96
    )
    assert not result.trades
    text = write_report(result, Baselines(), tmp_path).read_text("utf-8")
    assert "No risk-limit events" in text and "n/a" in text


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
    with session_scope(engine) as session:
        (run_id,) = session.scalars(select(Run.id)).all()
        run = repo.load_run(session, run_id)
    assert '"taker_fee_bps":7' in run.config_json and '"risk_per_trade":0.005' in run.config_json

    done = runner.invoke(app, ["report", str(run.id), "--variants-tried", "3"])
    assert done.exit_code == 0, done.output
    html_text = (tmp_path / "reports" / f"run_{run.id}" / "report.html").read_text("utf-8")
    assert "3 parameter variants" in html_text and run.config_hash in html_text
    assert runner.invoke(app, ["report", "999"]).exit_code != 0
    engine.dispose()
