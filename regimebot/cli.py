"""Command-line entry point."""

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml
from pydantic import ValidationError

from regimebot.backtest.baselines import compute_baselines
from regimebot.backtest.robustness import cost_stress, leverage_compare, sensitivity
from regimebot.backtest.runner import BacktestResult, persist_result, run_backtest
from regimebot.backtest.walk_forward import walk_forward
from regimebot.config import AppConfig, load_config, timeframe_minutes
from regimebot.data.exchange import CcxtExchange
from regimebot.data.history import download_candles, download_funding
from regimebot.data.resample import resample
from regimebot.data.validate import ValidationReport, validate_candles
from regimebot.db import repo
from regimebot.db.repo import load_candles, load_funding
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.engine.loop import RunLog
from regimebot.logging import configure_logging
from regimebot.market_state.features import compute_features
from regimebot.market_state.regime import classify_frame
from regimebot.market_state.study import regime_study
from regimebot.paper.session import PaperSession
from regimebot.paper.state import clear_halts, latest_status, lock_run
from regimebot.reporting.report import write_report
from regimebot.risk.manager import DEFAULT_KILL_FILE

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


def _parse_utc(value: str) -> datetime:
    """ISO date/datetime; a value without a zone is taken as UTC (CLI input boundary)."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(f"not an ISO date/time: {value!r}") from None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@data_app.command("download")
def data_download(
    start: Annotated[str | None, typer.Option(help="ISO start (default: backtest start).")] = None,
    end: Annotated[str | None, typer.Option(help="ISO end (default: backtest end).")] = None,
) -> None:
    """Download base-timeframe candles and funding for all configured symbols."""
    cfg = load_config()
    start_ts = _parse_utc(start) if start else cfg.backtest.start
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


@data_app.command("check")
def data_check() -> None:
    """Validate stored candles per symbol; exit 1 if any symbol is not ok."""
    cfg = load_config()
    tf = cfg.exchange.base_timeframe
    width = pd.Timedelta(minutes=timeframe_minutes(tf))
    engine = make_engine(cfg.database_url)
    all_ok = True
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            df = load_candles(session, symbol, tf, cfg.backtest.start, cfg.backtest.end)
            report = validate_candles(df, tf)
            # exact: first bar closes at the first grid point >= start, last at the one before end
            short = report.bars == 0 or (
                df.index[0] > pd.Timestamp(cfg.backtest.start).ceil(width)
                or df.index[-1] < pd.Timestamp(cfg.backtest.end).ceil(width) - width
            )
            all_ok &= report.ok and not short
            typer.echo(f"{symbol}: {'OK' if report.ok and not short else 'FAIL'}")
            for line in _report_lines(report, df, short):
                typer.echo(f"  {line}")
    if not all_ok:
        raise typer.Exit(1)


regimes_app = typer.Typer(help="Regime commands.", no_args_is_help=True)
app.add_typer(regimes_app, name="regimes")


@regimes_app.command("study")
def regimes_study() -> None:
    """Print and save the regime study (regimes vs forward returns) per symbol."""
    cfg = load_config()
    base, decision = cfg.exchange.base_timeframe, cfg.exchange.decision_timeframe
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    engine = make_engine(cfg.database_url)
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            candles = load_candles(session, symbol, base, cfg.backtest.start, cfg.backtest.end)
            if candles.empty:
                typer.echo(f"{symbol}: no candles in the database, run `data download` first")
                continue
            candles = resample(candles, base, decision)
            funding = load_funding(session, symbol, cfg.backtest.start, cfg.backtest.end)
            features = compute_features(
                candles, cfg.features, cfg.regime.compression_pctile, funding
            )
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
    """Config (with --set overrides, start/end applied to ``backtest``) and the stored history up
    to its end; history before start is feature warm-up."""
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
    with session_scope(make_engine(cfg.database_url)) as session:
        candles = {s: load_candles(session, s, base, None, cfg.backtest.end) for s in symbols}
        funding = {s: load_funding(session, s, None, cfg.backtest.end) for s in symbols}
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
    trades = result.trades
    final = float(result.equity.iloc[-1]) if len(result.equity) else bt.initial_equity
    max_dd = max((e[3] for e in result.log.equity), default=0.0)
    typer.echo(
        f"trades={len(trades)} net_pnl={sum(t.net_pnl for t in trades):.2f} "
        f"fees={sum(t.fees for t in trades):.2f} funding={sum(t.funding for t in trades):.2f} "
        f"final_equity={final:.2f} return={final / bt.initial_equity - 1:+.2%} "
        f"max_drawdown={max_dd:.2%}"
    )
    # these halts clear only by hand, so the run opened nothing after them
    sticky = ("max_drawdown", "manual_kill")
    events = result.log.risk_events
    if halt := next((e for e in events if e.action == "halt" and e.kind in sticky), None):
        typer.echo(
            f"halted at {halt.ts.isoformat()}: {halt.kind} ({halt.detail}), no entries after"
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
    """Parameter sensitivity, doubled costs, and max gross leverage 1.0 vs 1.5."""
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


@app.command()
def report(
    run_id: Annotated[int, typer.Argument(help="Run id printed by `backtest run`.")],
    variants_tried: Annotated[
        int | None, typer.Option(help="Parameter variants tried before this one (shown).")
    ] = None,
    random_seeds: Annotated[
        int, typer.Option(min=0, help="Number of random-entry seeds (0, 1, ... N-1); 0 skips.")
    ] = 3,
) -> None:
    """Rebuild the HTML report for a stored backtest into reports/run_<id>/.

    Trades, fills, risk events and equity come from the database; the baselines are recomputed
    from stored candles with the run's stored config (the random-entry rerun takes a minute or two).
    """
    database_url = load_config().database_url
    engine = make_engine(database_url)
    with session_scope(engine) as session:
        try:
            run = repo.load_run(session, run_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
        if run.mode != "backtest":
            raise typer.BadParameter(
                f"run {run_id} is a {run.mode} run; only backtests have reports"
            )
        try:
            cfg = AppConfig.model_validate(
                json.loads(run.config_json) | {"database_url": database_url}
            )
        except ValidationError as exc:
            typer.echo(f"run {run_id}: stored config no longer validates:\n{exc}", err=True)
            raise typer.Exit(1) from None
        bt, base = cfg.backtest, cfg.exchange.base_timeframe
        trades = repo.load_trades(session, run_id)
        log = RunLog(
            trades=trades,
            fills=repo.load_fills(session, run_id),
            risk_events=repo.load_risk_events(session, run_id),
        )
        result = BacktestResult(
            log, trades, repo.load_equity(session, run_id), cfg, bt.start, bt.end
        )
        n_signals = repo.count_signals(session, run_id)
        candles = {s: load_candles(session, s, base, None, bt.end) for s in cfg.exchange.symbols}
        funding = {s: load_funding(session, s, None, bt.end) for s in cfg.exchange.symbols}
    typer.echo("recomputing baselines (buy and hold, random entry) ...")
    baselines = compute_baselines(
        result, candles, funding, n_signals, seeds=tuple(range(random_seeds))
    )
    path = write_report(result, baselines, Path("reports") / f"run_{run_id}", variants_tried)
    typer.echo(f"wrote {path}")


def _say(line: str) -> None:
    """Human-facing paper-session text goes to stderr: stdout carries the JSON log lines."""
    typer.echo(line, err=True)


paper_app = typer.Typer(
    help="Paper trading: live market data, simulated fills, no orders.", no_args_is_help=True
)
app.add_typer(paper_app, name="paper")


@paper_app.command("run")
def paper_run(
    resume: Annotated[
        int | None, typer.Option(help="Continue this open paper run (its id from the banner).")
    ] = None,
) -> None:
    """Paper-trade the configured symbols until Ctrl-C or SIGTERM. Simulated fills only: no
    order is ever sent. A stopped run stays open; `--resume RUN_ID` continues it.

    Logs are JSON lines on stdout; the banner goes to stderr.
    """
    cfg = load_config()
    configure_logging()
    _say("PAPER TRADING: simulated fills only, no orders are sent to any exchange.")
    _say(
        f"warming up {cfg.paper.warmup_bars} {cfg.exchange.decision_timeframe} bars of history ..."
    )
    session = PaperSession(
        cfg, CcxtExchange(cfg.exchange.id), make_engine(cfg.database_url), DEFAULT_KILL_FILE
    )
    try:
        run_id = session.start(resume)
    except ValueError as exc:  # not a resumable paper run
        raise typer.BadParameter(str(exc), param_hint="--resume") from None
    _say(
        "\n".join(
            [
                "=" * 72,
                "PAPER TRADING - simulated fills only, no orders are sent to any exchange",
                f"run id:    {run_id} ({'resumed' if resume is not None else 'new'})",
                f"symbols:   {', '.join(cfg.exchange.symbols)}, polling every "
                f"{cfg.paper.poll_seconds}s",
                f"kill file: {DEFAULT_KILL_FILE} (`regimebot kill` flattens at the next poll)",
                f"halts:     the kill switch or a {cfg.risk.max_drawdown:.0%} drawdown halt "
                "flattens and stops this run's trading;",
                "           after a manual kill: `regimebot unkill`, then "
                f"`regimebot paper clear-halt {run_id}`;",
                "           a drawdown halt ends the run (start a new one)",
                "stop:      Ctrl-C or SIGTERM (finishes the current poll, positions stay open)",
                f"resume:    regimebot paper run --resume {run_id}",
                "=" * 72,
            ]
        )
    )
    session.run_forever()
    _say(f"stopped; run {run_id} is open. Resume with: regimebot paper run --resume {run_id}")


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
    events. Exits 1 when an open run has not polled within --max-age (for uptime checks)."""
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
    state = (
        f"open (resume with `regimebot paper run --resume {status.run_id}`)"
        if status.is_open
        else "ended"
    )
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
    if status.is_open and (age is None or age.total_seconds() > limit):
        typer.echo(f"NOT POLLING: no poll in the last {limit} s (hung or stopped?)", err=True)
        raise typer.Exit(1)
