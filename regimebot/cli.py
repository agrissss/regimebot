"""Command-line entry point."""

from datetime import UTC, datetime, timedelta
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from regimebot.config import load_config, timeframe_minutes
from regimebot.data.exchange import CcxtExchange
from regimebot.data.history import download_candles, download_funding
from regimebot.data.resample import resample
from regimebot.data.validate import ValidationReport, validate_candles
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
    width = timedelta(minutes=timeframe_minutes(tf))
    engine = make_engine(cfg.database_url)
    all_ok = True
    with session_scope(engine) as session:
        for symbol in cfg.exchange.symbols:
            df = load_candles(session, symbol, tf, cfg.backtest.start, cfg.backtest.end)
            report = validate_candles(df, tf)
            short = report.bars == 0 or (
                df.index[0] > cfg.backtest.start + width or df.index[-1] < cfg.backtest.end - width
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
            # compute_features fails on an empty funding series (features.py); None means no funding
            funding_arg = funding if len(funding) else None
            features = compute_features(
                candles, cfg.features, cfg.regime.compression_pctile, funding_arg
            )
            table = regime_study(features, classify_frame(features, cfg.regime))
            typer.echo(f"{symbol}\n{table.to_string(float_format=lambda x: f'{x:.4f}')}")
            safe = symbol.replace("/", "_").replace(":", "_")
            table.to_csv(out_dir / f"regime_study_{safe}.csv")
