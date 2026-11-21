"""Command-line entry point."""

from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import typer

from regimebot.config import load_config
from regimebot.data.exchange import CcxtExchange
from regimebot.data.history import download_candles, download_funding
from regimebot.data.resample import resample
from regimebot.data.validate import validate_candles
from regimebot.db.repo import load_candles, load_funding
from regimebot.db.session import make_engine, session_scope, upgrade_to_head
from regimebot.market_state.features import compute_features
from regimebot.market_state.regime import classify_frame
from regimebot.market_state.study import regime_study

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


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply all database migrations to the configured database."""
    cfg = load_config()
    upgrade_to_head(cfg.database_url)
    typer.echo("database is up to date")


data_app = typer.Typer(help="Market data commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


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


@data_app.command("check")
def data_check() -> None:
    """Validate stored candles per symbol; exit 1 if any symbol is not ok."""
    cfg = load_config()
    tf = cfg.exchange.base_timeframe
    engine = make_engine(cfg.database_url)
    all_ok = True
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            df = load_candles(session, symbol, tf, cfg.backtest.start, cfg.backtest.end)
            report = validate_candles(df, tf)
            all_ok &= report.ok
            typer.echo(f"{symbol}: {'OK' if report.ok else 'FAIL'}")
            typer.echo(f"  {report}")
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
