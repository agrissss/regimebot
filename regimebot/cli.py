"""Command-line entry point."""

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from importlib.metadata import version as package_version
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml
from pydantic import ValidationError
from sqlalchemy import Engine

from regimebot.backtest.baselines import Baselines, compute_baselines
from regimebot.backtest.robustness import cost_stress, leverage_compare, sensitivity
from regimebot.backtest.runner import BacktestResult, load_result, persist_result, run_backtest
from regimebot.backtest.walk_forward import walk_forward
from regimebot.config import AppConfig, load_config, timeframe_minutes
from regimebot.data.exchange import CcxtExchange
from regimebot.data.history import download_candles, download_funding
from regimebot.data.resample import resample
from regimebot.data.validate import (
    FUNDING_GAP_TOLERANCE,
    ValidationReport,
    funding_gaps,
    validate_candles,
)
from regimebot.db import repo
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.logging import configure_logging
from regimebot.market_state.features import compute_features, warmup_bars
from regimebot.market_state.regime import classify_frame
from regimebot.market_state.study import regime_study
from regimebot.paper.session import EXCHANGE_MAX_RETRIES, EXCHANGE_TIMEOUT_MS, PaperSession
from regimebot.paper.state import (
    clear_halts,
    end_paper_run,
    latest_open_run,
    latest_status,
    lock_run,
)
from regimebot.reporting.metrics import compute_metrics, execution_quality, first_permanent_halt
from regimebot.reporting.report import write_report
from regimebot.risk.manager import DEFAULT_KILL_FILE
from regimebot.types import Mode

app = typer.Typer(help="Regime-aware crypto trading research system.", no_args_is_help=True)

db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@app.callback()
def main() -> None:
    """regimebot commands."""


@app.command()
def version() -> None:
    """Print the installed regimebot version."""
    typer.echo(package_version("regimebot"))


@app.command()
def kill() -> None:
    """Emergency stop: create the kill file; sessions halt, flatten and stop opening positions."""
    DEFAULT_KILL_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_KILL_FILE.touch()
    typer.echo(f"kill switch on: {DEFAULT_KILL_FILE}")


@app.command()
def unkill() -> None:
    """Remove the kill file. A paper run it halted stays halted (and flat) until
    `regimebot paper clear-halt RUN_ID`."""
    try:
        DEFAULT_KILL_FILE.unlink()
    except FileNotFoundError:
        typer.echo(f"kill switch was not on: no {DEFAULT_KILL_FILE}")
        return
    typer.echo(f"kill switch off: removed {DEFAULT_KILL_FILE}")


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply all database migrations to the configured database."""
    cfg = load_config()
    upgrade_to_head(cfg.database_url)
    typer.echo("database is up to date")


data_app = typer.Typer(help="Market data commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


MAX_GAPS_SHOWN = 10


def history_start(cfg: AppConfig) -> datetime:
    """Where stored history must begin so that decisions from ``backtest.start`` see warm
    features: ``warmup_bars`` decision bars earlier, rounded up to whole days, plus one day."""
    width = timedelta(minutes=timeframe_minutes(cfg.exchange.decision_timeframe))
    days = -(-warmup_bars(cfg.features) * width // timedelta(days=1)) + 1  # ceil, + 1
    return cfg.backtest.start - timedelta(days=days)


def _parse_utc(value: str) -> datetime:
    """ISO date/datetime; a value without a zone is taken as UTC (CLI input boundary)."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"not an ISO date/time: {value!r}") from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@data_app.command("download")
def data_download(
    start: Annotated[
        str | None,
        typer.Option(help="ISO start (default: backtest start minus the feature warm-up)."),
    ] = None,
    end: Annotated[str | None, typer.Option(help="ISO end (default: backtest end).")] = None,
) -> None:
    """Download base-timeframe candles and funding for all configured symbols, by default from
    the feature warm-up before the backtest start (``history_start``) to the backtest end."""
    cfg = load_config()
    start_ts = _parse_utc(start) if start else history_start(cfg)
    end_ts = _parse_utc(end) if end else cfg.backtest.end
    exchange = CcxtExchange(cfg.exchange.id)
    engine = make_engine(cfg.database_url)
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            tf = cfg.exchange.base_timeframe
            candles = download_candles(exchange, session, symbol, tf, start_ts, end_ts)
            funding = download_funding(exchange, session, symbol, start_ts, end_ts)
            typer.echo(f"{symbol}: {candles} candles, {funding} funding rows")


def _report_lines(report: ValidationReport, df: pd.DataFrame, short: bool) -> list[str]:
    if report.bars == 0:
        return ["no candles stored"]
    first, last = df.index[0].isoformat(), df.index[-1].isoformat()
    lines = [
        f"bars={report.bars} first={first} last={last}"
        + ("  (history shorter than the configured range)" if short else ""),
        f"duplicates={report.duplicates} bad_ohlc={report.bad_ohlc} "
        f"non_positive_prices={report.non_positive_prices} "
        f"negative_volume={report.negative_volume} "
        f"missing_bars={report.missing_bars}/{report.expected_bars}",
    ]
    lines += [f"gap {a.isoformat()} .. {b.isoformat()}" for a, b in report.gaps[:MAX_GAPS_SHOWN]]
    if len(report.gaps) > MAX_GAPS_SHOWN:
        lines.append(f"... and {len(report.gaps) - MAX_GAPS_SHOWN} more")
    return lines


def _hours(span: timedelta) -> str:
    return f"{span / timedelta(hours=1):.1f} h"


def _funding_lines(
    stamps: list[datetime], gaps: list[tuple[datetime, datetime]], start: datetime, end: datetime
) -> list[str]:
    """Row count, first and last stamp, the largest stretch without a stamp (range edges
    included), and every stretch over the tolerance (at most ``MAX_GAPS_SHOWN``)."""
    largest = max(b - a for a, b in pairwise([start, *stamps, end]))
    head = f"funding rows={len(stamps)}"
    if stamps:
        head += f" first={stamps[0].isoformat()} last={stamps[-1].isoformat()}"
    lines = [f"{head} largest_gap={_hours(largest)}"]
    lines += [
        f"funding gap {a.isoformat()} .. {b.isoformat()} ({_hours(b - a)})"
        for a, b in gaps[:MAX_GAPS_SHOWN]
    ]
    if len(gaps) > MAX_GAPS_SHOWN:
        lines.append(f"... and {len(gaps) - MAX_GAPS_SHOWN} more funding gaps")
    return lines


@data_app.command("check")
def data_check() -> None:
    """Validate stored candles and funding per symbol over the backtest range and the feature
    warm-up before it (``history_start`` to the backtest end); exit 1 if any symbol is not ok.
    Funding fails on any stretch without a stamp longer than ``FUNDING_GAP_TOLERANCE``."""
    cfg = load_config()
    tf = cfg.exchange.base_timeframe
    width = pd.Timedelta(minutes=timeframe_minutes(tf))
    start, end = history_start(cfg), cfg.backtest.end
    engine = make_engine(cfg.database_url)
    candles_ok = funding_ok = True
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            df = repo.load_candles(session, symbol, tf, start, end)
            report = validate_candles(df, tf)
            # exact: first bar closes at the first grid point >= start, last at the one before end
            short = report.bars == 0 or (
                df.index[0] > pd.Timestamp(start).ceil(width)
                or df.index[-1] < pd.Timestamp(end).ceil(width) - width
            )
            stamps = list(repo.funding_timestamps(session, symbol, start, end).to_pydatetime())
            gaps = funding_gaps(stamps, start, end)
            ok = report.ok and not short
            candles_ok &= ok
            funding_ok &= not gaps
            typer.echo(f"{symbol}: {'OK' if ok and not gaps else 'FAIL'}")
            for line in _report_lines(report, df, short) + _funding_lines(stamps, gaps, start, end):
                typer.echo(f"  {line}")
    if not candles_ok:
        typer.echo(
            f"the backtest needs {tf} history from {start.isoformat()} (feature warm-up) to "
            f"{end.isoformat()}: run `regimebot data download` to fetch it"
        )
    if not funding_ok:
        typer.echo(
            f"funding has stretches over {FUNDING_GAP_TOLERANCE / timedelta(hours=1):.0f} h "
            f"without a stamp between {start.isoformat()} and {end.isoformat()}: "
            "run `regimebot data download` to fetch them"
        )
    if not (candles_ok and funding_ok):
        raise typer.Exit(1)


regimes_app = typer.Typer(help="Regime commands.", no_args_is_help=True)
app.add_typer(regimes_app, name="regimes")


@regimes_app.command("study")
def regimes_study() -> None:
    """Print and save the regime study (regimes vs forward returns) per symbol, over the
    backtest range (history before it only warms the features up)."""
    cfg = load_config()
    base, decision = cfg.exchange.base_timeframe, cfg.exchange.decision_timeframe
    first, end = history_start(cfg), cfg.backtest.end
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    engine = make_engine(cfg.database_url)
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            candles = repo.load_candles(session, symbol, base, first, end)
            if candles.empty:
                typer.echo(f"{symbol}: no candles in the database, run `data download` first")
                continue
            candles = resample(candles, base, decision)
            funding = repo.load_funding(session, symbol, first, end)
            features = compute_features(
                candles, cfg.features, cfg.regime.compression_pctile, funding
            )
            features = features[features.index >= cfg.backtest.start]
            table = regime_study(features, classify_frame(features, cfg.regime))
            typer.echo(f"{symbol}\n{table.to_string(float_format=lambda x: f'{x:.4f}')}")
            safe = symbol.replace("/", "_").replace(":", "_")
            table.to_csv(out_dir / f"regime_study_{safe}.csv")


DEFAULT_ROBUSTNESS_PARAMS = [
    "strategies.breakout.stop_atr_mult",
    "strategies.breakout.volume_z_min",
    "strategies.trend.stop_atr_mult",
    "risk.risk_per_trade",
]
backtest_app = typer.Typer(help="Backtest commands.", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


_NOT_EVIDENCE = "These are historical simulations, not evidence of future performance."
_SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", help="Config override key=value (dotted path, YAML value); repeatable."),
]


def _research_inputs(
    start: str | None, end: str | None, set_: list[str] | None
) -> tuple[AppConfig, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Config (with --set overrides, start/end applied to ``backtest``) and the stored history
    from ``history_start`` to its end; history before start is feature warm-up."""
    overrides: dict[str, object] = {}
    for item in set_ or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"--set needs key=value, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    if start:
        overrides["backtest.start"] = _parse_utc(start)
    if end:
        overrides["backtest.end"] = _parse_utc(end)
    try:
        cfg = load_config(overrides=overrides)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    symbols, base = cfg.exchange.symbols, cfg.exchange.base_timeframe
    first, last = history_start(cfg), cfg.backtest.end
    with session_scope(make_engine(cfg.database_url)) as session:
        candles = {s: repo.load_candles(session, s, base, first, last) for s in symbols}
        funding = {s: repo.load_funding(session, s, first, last) for s in symbols}
    if missing := [s for s, df in candles.items() if df.empty]:
        typer.echo(f"no candles for {', '.join(missing)}: run `data download` first")
        raise typer.Exit(1)
    return cfg, candles, funding


@backtest_app.command("run")
def backtest_run(
    start: Annotated[str | None, typer.Option(help="ISO start (default: backtest start).")] = None,
    end: Annotated[str | None, typer.Option(help="ISO end (default: backtest end).")] = None,
    note: Annotated[str, typer.Option(help="Free-text note stored with the run.")] = "",
    set_: _SetOption = None,
) -> None:
    """Backtest the configured symbols on stored candles, store the run, print headline numbers.

    The kill file lives in a throwaway directory: a backtest is a reproducible research run, so
    `regimebot kill` (meant for paper sessions) neither halts it nor changes its result.
    """
    cfg, candles, funding = _research_inputs(start, end, set_)  # warm-up history before start
    bt = cfg.backtest
    with tempfile.TemporaryDirectory() as tmp:
        result = run_backtest(cfg, candles, funding, bt.start, bt.end, Path(tmp) / "KILL")
    with session_scope(make_engine(cfg.database_url)) as session:
        run_id = persist_result(session, result, note)
    m = compute_metrics(result.trades, result.equity, bt.initial_equity)  # the whole run
    final = float(result.equity.iloc[-1]) if len(result.equity) else bt.initial_equity
    typer.echo(
        f"trades={m.n_trades} net_pnl={m.net_pnl:.2f} fees={m.fees:.2f} funding={m.funding:.2f} "
        f"final_equity={final:.2f} return={m.return_pct:+.2%} max_drawdown={m.max_drawdown:.2%}"
    )
    events = result.log.risk_events
    if halt := first_permanent_halt(events):  # it clears only by hand: no entries after it
        typer.echo(
            f"halted at {halt.ts.isoformat()}: {halt.kind} ({halt.detail}), no entries after"
        )
    if cold := [e.detail for e in events if e.kind == "warmup_incomplete"]:
        typer.echo(
            f"warning: under-warmed start, no trades until warm ({'; '.join(cold)}): "
            "run `regimebot data download` for the warm-up history"
        )
    if holes := [e.detail for e in events if e.kind == "funding_missing"]:
        typer.echo(
            f"warning: funding missing, not charged there ({'; '.join(holes)}): "
            "run `regimebot data download` for the funding history"
        )
    typer.echo(f"run {run_id}")


def _show(df: pd.DataFrame, csv: Path, summary: str = "") -> None:
    """Print the table (and an optional summary line), then the disclaimer: no number follows it."""
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    typer.echo(df.to_string(index=False))
    if summary:
        typer.echo(summary)
    typer.echo(_NOT_EVIDENCE)
    typer.echo(f"wrote {csv}")


@backtest_app.command("walk-forward")
def backtest_walk_forward(
    start: Annotated[str | None, typer.Option(help="ISO start (default: backtest start).")] = None,
    end: Annotated[str | None, typer.Option(help="ISO end (default: backtest end).")] = None,
    set_: _SetOption = None,
) -> None:
    """Walk-forward: fit the config's grid on each train span, test on the span after it.

    Slow: every window runs one backtest per grid combination plus one test run.
    """
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)  # before the long computation, so it fails fast
    cfg, candles, funding = _research_inputs(start, end, set_)
    try:
        res = walk_forward(cfg, candles, funding, cfg.backtest.start, cfg.backtest.end)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    pd.DataFrame([asdict(t) for t in res.oos_trades]).to_csv(
        out / "walk_forward_oos_trades.csv", index=False
    )
    summary = (
        f"out-of-sample: windows={len(res.windows)} trades={len(res.oos_trades)} "
        f"net_pnl={res.oos_net_pnl:.2f} n_variants_tried={res.n_variants_tried}"
    )
    if res.n_test_windows_halted:
        summary += (
            f"\n{res.n_test_windows_halted} of {len(res.windows)} test windows hit the drawdown "
            "halt and stopped early; use --set risk.max_drawdown=1.0 for full windows."
        )
    _show(res.windows, out / "walk_forward_windows.csv", summary)
    typer.echo(f"wrote {out / 'walk_forward_oos_trades.csv'}")


@backtest_app.command("robustness")
def backtest_robustness(
    param: Annotated[
        list[str] | None,
        typer.Option(help="Numeric dotted config path to perturb by +/-20%; repeatable."),
    ] = None,
    start: Annotated[str | None, typer.Option(help="ISO start (default: backtest start).")] = None,
    end: Annotated[str | None, typer.Option(help="ISO end (default: backtest end).")] = None,
    set_: _SetOption = None,
) -> None:
    """Parameter sensitivity, costs x0.5 / x1 / x2, and effective leverage (gross and group caps)
    1.0 vs 1.5."""
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)  # before the long computation, so it fails fast
    cfg, candles, funding = _research_inputs(start, end, set_)
    a, b = cfg.backtest.start, cfg.backtest.end
    try:
        sens = sensitivity(cfg, candles, funding, a, b, param or DEFAULT_ROBUSTNESS_PARAMS)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    _show(sens, out / "sensitivity.csv")
    _show(cost_stress(cfg, candles, funding, a, b), out / "cost_stress.csv")
    _show(leverage_compare(cfg, candles, funding, a, b), out / "leverage_compare.csv")


MAX_RANDOM_SEEDS = 50


@app.command()
def report(
    run_id: Annotated[int, typer.Argument(help="Run id printed by `backtest run`.")],
    variants_tried: Annotated[
        int | None, typer.Option(help="Parameter variants tried before this one (shown).")
    ] = None,
    random_seeds: Annotated[
        int,
        typer.Option(
            min=0,
            max=MAX_RANDOM_SEEDS,
            help="Number of random-entry seeds (0, 1, ... N-1), at most 50; 0 skips. Backtests.",
        ),
    ] = 3,
) -> None:
    """Rebuild the HTML report for a stored run into reports/run_<id>/.

    Trades, fills, risk events and equity come from the database; the baselines are recomputed
    from stored candles with the run's stored config (the random-entry rerun takes a minute or two).
    A paper run, open or ended, gets the same report over its period (first to last equity
    snapshot) with buy and hold as its only baseline and an execution-quality section.
    """
    database_url = load_config().database_url
    engine = make_engine(database_url)
    with session_scope(engine) as session:
        try:
            run = repo.load_run(session, run_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
        try:
            cfg = AppConfig.model_validate(
                json.loads(run.config_json) | {"database_url": database_url}
            )
        except ValidationError as exc:
            typer.echo(f"run {run_id}: stored config no longer validates:\n{exc}", err=True)
            raise typer.Exit(1) from None
        result = load_result(session, run_id, cfg)
        n_signals = repo.count_signals(session, run_id)
        execution = None
        first, end = history_start(cfg), cfg.backtest.end
        if paper := run.mode == Mode.PAPER:
            signals = repo.load_signal_stamps(session, run_id)
            decisions = repo.load_risk_decisions(session, run_id)
            execution = execution_quality(result.log.fills, result.trades, signals, decisions)
            # a day before the period: its first decision bar, and its first day's market return
            first, end = result.start - timedelta(days=1), result.end
        base = cfg.exchange.base_timeframe
        candles = {s: repo.load_candles(session, s, base, first, end) for s in cfg.exchange.symbols}
        funding = {s: repo.load_funding(session, s, first, end) for s in cfg.exchange.symbols}
    if paper:
        baselines = _paper_baselines(result, candles, funding)
    else:
        typer.echo("recomputing baselines (buy and hold, random entry) ...")
        baselines = compute_baselines(
            result, candles, funding, n_signals, seeds=tuple(range(random_seeds))
        )
    out = Path("reports") / f"run_{run_id}"
    path = write_report(result, baselines, out, variants_tried, execution)
    typer.echo(f"wrote {path}")


def _paper_baselines(
    result: BacktestResult, candles: dict[str, pd.DataFrame], funding: dict[str, pd.Series]
) -> Baselines:
    """Buy and hold over a paper run's period (random entry is backtest-only); none, with a
    note, before its first decision or when a symbol has no stored decision bar in it."""
    if result.equity.empty:
        typer.echo("paper run: no decision yet, so no buy-and-hold baseline")
        return Baselines()
    try:
        return compute_baselines(result, candles, funding, 0, seeds=())
    except ValueError as exc:
        typer.echo(f"paper run: no buy-and-hold baseline ({exc})")
        return Baselines()


def _say(line: str) -> None:
    """Human-facing paper-session text goes to stderr: stdout carries the JSON log lines."""
    typer.echo(line, err=True)


paper_app = typer.Typer(
    help="Paper trading: live market data, simulated fills, no orders.", no_args_is_help=True
)
app.add_typer(paper_app, name="paper")


def _resume_target(resume: str | None, db: Engine) -> int | None:
    """The run ``paper run --resume`` continues: a run id, or for ``latest`` the most recent
    open paper run (None, a new run, when there is none)."""
    if resume is None:
        return None
    if resume == "latest":
        with session_scope(db) as session:
            run_id = latest_open_run(session)
        if run_id is None:
            _say("--resume latest: no open paper run, starting a new one")
        else:
            _say(f"--resume latest: resuming open paper run {run_id}")
        return run_id
    try:
        return int(resume)
    except ValueError:
        raise typer.BadParameter(
            f"expected a run id or `latest`, got {resume!r}", param_hint="--resume"
        ) from None


@paper_app.command("run")
def paper_run(
    resume: Annotated[
        str | None,
        typer.Option(
            help="Continue this open paper run (its id from the banner), or `latest`: the most "
            "recent open paper run, else a new one (safe for a supervisor to restart)."
        ),
    ] = None,
) -> None:
    """Paper-trade the configured symbols until Ctrl-C or SIGTERM. Simulated fills only: no
    order is ever sent. A stopped run stays open; `--resume RUN_ID` continues it,
    `--resume latest` continues the most recent open run (or starts one when none is open), and
    `paper end RUN_ID` ends it for good.

    Logs are JSON lines on stdout; the banner goes to stderr.
    """
    cfg = load_config()
    configure_logging()
    _say("PAPER TRADING: simulated fills only, no orders are sent to any exchange.")
    db = make_engine(cfg.database_url)
    resume_id = _resume_target(resume, db)
    _say(
        f"warming up {cfg.paper.warmup_bars} {cfg.exchange.decision_timeframe} bars of history ..."
    )
    # few retries, short timeouts: one hanging symbol must not stall a poll (or the kill switch)
    exchange = CcxtExchange(
        cfg.exchange.id, max_retries=EXCHANGE_MAX_RETRIES, timeout_ms=EXCHANGE_TIMEOUT_MS
    )
    session = PaperSession(cfg, exchange, db, DEFAULT_KILL_FILE)
    try:
        run_id = session.start(resume_id)
    except ValueError as exc:  # not a resumable paper run
        raise typer.BadParameter(str(exc), param_hint="--resume") from None
    try:
        _run_session(cfg, session, run_id, resumed=resume_id is not None)
    finally:
        session.close()  # release the run lock (process exit would too)


def _run_session(cfg: AppConfig, session: PaperSession, run_id: int, resumed: bool) -> None:
    """Print the banner, poll until stopped, then say how to resume or end the run."""
    _say(
        "\n".join(
            [
                "=" * 72,
                "PAPER TRADING - simulated fills only, no orders are sent to any exchange",
                f"run id:    {run_id} ({'resumed' if resumed else 'new'})",
                f"symbols:   {', '.join(cfg.exchange.symbols)}, polling every "
                f"{cfg.paper.poll_seconds}s",
                f"kill file: {DEFAULT_KILL_FILE} (`regimebot kill` flattens at the next poll)",
                f"halts:     the kill switch or a {cfg.risk.max_drawdown:.0%} drawdown halt "
                "flattens and stops this run's trading;",
                "           after a manual kill: `regimebot unkill`, then "
                f"`regimebot paper clear-halt {run_id}`;",
                "           a drawdown halt ends the run (start a new one); daily and weekly "
                "loss halts",
                "           also flatten, and clear at the next UTC day / ISO week",
                "stop:      Ctrl-C or SIGTERM (finishes the current poll, positions stay open)",
                f"resume:    regimebot paper run --resume {run_id}",
                "=" * 72,
            ]
        )
    )
    session.run_forever()
    _say(
        f"stopped; run {run_id} is open. Resume with: regimebot paper run --resume {run_id} "
        f"(or end it: regimebot paper end {run_id})"
    )


@paper_app.command("clear-halt")
def paper_clear_halt(
    run_id: Annotated[int, typer.Argument(help="The halted paper run.")],
) -> None:
    """Clear a paper run's recorded halts so its next decision may trade again.

    Refuses while the kill file exists (`regimebot unkill` first) or while the run is running.
    A max_drawdown halt re-triggers at the next check while drawdown is still past the limit: by
    design that run is over, start a new one.
    """
    cfg = load_config()
    try:
        lock = lock_run(DEFAULT_KILL_FILE.parent, run_id)
    except ValueError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from None
    try:
        with session_scope(make_engine(cfg.database_url)) as session:
            cleared = clear_halts(session, run_id, cfg, DEFAULT_KILL_FILE)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from None
    finally:
        os.close(lock)  # held until the cleared state is committed
    typer.echo(f"run {run_id}: cleared {', '.join(cleared) if cleared else 'nothing (no halt)'}")


@paper_app.command("end")
def paper_end(
    run_id: Annotated[int, typer.Argument(help="The stopped paper run to end.")],
    force: Annotated[
        bool, typer.Option(help="End it even though it holds open (simulated) positions.")
    ] = False,
) -> None:
    """End a stopped paper run for good: it records its end time and can no longer be resumed.

    Refuses while the run is running, and while it holds open positions unless --force (they
    are simulated: nothing is closed, they stay in its last saved state).
    """
    cfg = load_config()
    try:
        lock = lock_run(DEFAULT_KILL_FILE.parent, run_id)
    except ValueError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from None
    try:
        with session_scope(make_engine(cfg.database_url)) as session:
            left_open = end_paper_run(session, run_id, force)
    except ValueError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from None
    finally:
        os.close(lock)  # held until the end is committed
    kept = f"; open positions left as saved: {', '.join(left_open)}" if left_open else ""
    typer.echo(f"run {run_id}: ended{kept}")


@paper_app.command("status")
def paper_status(
    max_age: Annotated[
        int | None,
        typer.Option(
            help="Seconds since the last poll before an open run counts as hung "
            "(default 3 x poll_seconds + stale_after_seconds)."
        ),
    ] = None,
) -> None:
    """Show the latest paper run: liveness, equity, open positions, last regimes, halt, risk
    events. Exits 1 when an open run that was not stopped cleanly has not polled within --max-age
    (crashed or hung: for uptime checks); a run stopped with Ctrl-C or SIGTERM exits 0."""
    cfg = load_config()
    limit = (
        max_age
        if max_age is not None
        else (3 * cfg.paper.poll_seconds + cfg.paper.stale_after_seconds)
    )
    with session_scope(make_engine(cfg.database_url)) as session:
        status = latest_status(session)
    if status is None:
        typer.echo("no paper runs yet: start one with `regimebot paper run`")
        return
    resume = f"(resume with `regimebot paper run --resume {status.run_id}`)"
    if not status.is_open:
        state = "ended"
    elif status.stopped_at is not None:
        state = f"stopped at {status.stopped_at.isoformat()} {resume}"
    else:
        state = f"open {resume}"
    typer.echo(f"paper run {status.run_id}: {state}, started {status.started_at.isoformat()}")
    age = None if status.last_poll is None else (datetime.now(UTC) - status.last_poll)
    typer.echo("last poll: never" if age is None else f"last poll: {age.total_seconds():.0f} s ago")
    if status.equity is None:
        typer.echo("equity: no decision yet")
    else:
        ts, equity = status.equity
        typer.echo(f"equity: {equity:.2f} at {ts.isoformat()} (cash {status.cash or 0.0:.2f})")
    typer.echo(f"halted: {status.halted_reason or 'no'}")
    typer.echo(f"open positions: {len(status.positions)}")
    for p in status.positions:
        typer.echo(
            f"  {p.symbol} {p.side} qty={p.qty:g} entry={p.entry_price:g} at "
            f"{p.entry_ts.isoformat()} stop={p.stop:g} target={p.target:g} ({p.strategy})"
        )
    typer.echo("last regime:")
    for symbol, (ts, regime) in status.last_regimes.items():
        typer.echo(f"  {symbol} {regime} at {ts.isoformat()}")
    typer.echo(f"last {len(status.risk_events)} risk events:")
    for e in status.risk_events:
        typer.echo(f"  {e.ts.isoformat()} {e.kind} -> {e.action}: {e.detail}")
    hung = age is None or age.total_seconds() > limit
    if status.is_open and status.stopped_at is None and hung:
        typer.echo(f"NOT POLLING: no poll in the last {limit} s (crashed or hung?)", err=True)
        raise typer.Exit(1)
