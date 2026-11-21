"""Self-contained HTML report (inline CSS, no JavaScript, no external requests), CSVs and PNGs.

The report is built to avoid flattering the strategy: costs are shown, the strategy sits beside
buy-and-hold and random entry, the result without its best 5% of trades is shown, and a risk
warning comes first.
"""

import base64
import html
import io
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib.figure import Figure  # Figure renders with Agg and needs no GUI backend

from regimebot.backtest.baselines import Baselines
from regimebot.backtest.runner import BacktestResult
from regimebot.config import config_hash
from regimebot.reporting.metrics import (
    Metrics,
    best_trades_dropped,
    breakdown,
    compute_metrics,
    daily_equity,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from regimebot.types import RiskEvent

RISK_WARNING = (
    "Crypto trading involves substantial risk, and you can lose all of the money you put in.",
    "Leveraged trading can lose money fast: a small adverse move can wipe out a position.",
    "Past performance does not guarantee future results.",
    "No system predicts sudden market changes, such as crashes, exchange outages or news shocks.",
    "Test extensively in paper trading before risking any real capital.",
    "A backtest is not proof that a strategy works. It is a simulation on past data with "
    "modelled costs, and it can be wrong in ways that flatter the strategy.",
)
MAX_EVENT_ROWS = 50
_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;"
    "color:#222}table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
    "th,td{border:1px solid #bbb;padding:.25rem .6rem;text-align:right}"
    "th:first-child,td:first-child{text-align:left}.warn{border:3px solid #b00;background:#fff3f3;"
    "padding:.5rem 1rem;margin:1rem 0}img{max-width:100%}"
)


def _num(x: float, fmt: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return format(x, fmt)


def money(x: float) -> str:
    return _num(x, ",.2f")


def pct(x: float) -> str:
    return "n/a" if math.isnan(x) else _num(x * 100) + "%"


def ratio(x: float) -> str:
    return _num(x)


def bps(x: float) -> str:
    return "n/a" if math.isnan(x) else _num(x) + " bps"


def count(x: float) -> str:
    return _num(x, ",.0f")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _png(fig: Figure, path: Path) -> str:
    """Save ``fig`` to ``path`` and return the base64 of the same PNG for embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    path.write_bytes(buf.getvalue())
    return base64.b64encode(buf.getvalue()).decode()


def _equity_chart(result: BacktestResult, baselines: Baselines, path: Path) -> str:
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.plot(result.equity.index, result.equity.to_numpy(), label="strategy")
    if baselines.buy_and_hold is not None:
        ax.plot(
            baselines.buy_and_hold.index, baselines.buy_and_hold.to_numpy(), label="buy and hold"
        )
    if baselines.random_entry is not None:
        rnd = baselines.random_entry.equity
        ax.plot(rnd.index, rnd.to_numpy(), label="random entry")
    ax.set_title("Equity (after all modelled costs)")
    ax.legend()
    fig.autofmt_xdate()
    return _png(fig, path)


def _drawdown_chart(result: BacktestResult, path: Path) -> str:
    initial = result.cfg.backtest.initial_equity
    values = result.equity.to_numpy(dtype=float)
    peak = pd.Series(values).cummax().clip(lower=initial)
    fig = Figure(figsize=(9, 3))
    ax = fig.subplots()
    ax.fill_between(result.equity.index, -(1 - values / peak.to_numpy()) * 100, 0, color="#b00")
    ax.set_title("Strategy drawdown (%)")
    fig.autofmt_xdate()
    return _png(fig, path)


def _daily_pnl(equity: pd.Series, initial_equity: float) -> pd.DataFrame:
    daily = daily_equity(equity)
    pnl = daily.diff()
    if len(daily):
        pnl.iloc[0] = daily.iloc[0] - initial_equity
    out = pd.DataFrame({"equity": daily, "pnl": pnl})
    out.index.name = "date"
    return out


def _bh_column(bh: pd.Series | None, initial: float) -> dict[str, str]:
    if bh is None or bh.empty:
        return {}
    final = float(bh.iloc[-1])
    return {
        "Net P&L": money(final - initial),
        "Return": pct(final / initial - 1),
        "Max drawdown": pct(max_drawdown(bh, initial)),
        "Sharpe (daily, annualised)": ratio(sharpe_ratio(bh)),
        "Sortino (daily, annualised)": ratio(sortino_ratio(bh)),
    }


def _metric_column(m: Metrics) -> dict[str, str]:
    return {
        "Net P&L": money(m.net_pnl),
        "Return": pct(m.return_pct),
        "Max drawdown": pct(m.max_drawdown),
        "Sharpe (daily, annualised)": ratio(m.sharpe),
        "Sortino (daily, annualised)": ratio(m.sortino),
        "Profit factor": ratio(m.profit_factor),
        "Win rate": pct(m.win_rate),
        "Average win": money(m.avg_win),
        "Average loss": money(m.avg_loss),
        "Expectancy (R per trade)": ratio(m.expectancy_r),
        "Trades": count(m.n_trades),
        "Average bars held": ratio(m.avg_bars_held),
        "Exposure (share of decision bars in a position)": pct(m.exposure_pct),
        "Turnover (entry notional / initial equity)": ratio(m.turnover),
        "Fees paid": money(m.fees),
        "Funding paid (negative = received)": money(m.funding),
        "Target hit rate": pct(m.target_hit_rate),
        "Stop-out rate": pct(m.stop_out_rate),
        "Net P&L without the best 5% of trades": money(m.net_pnl_without_best_5pct),
    }


def _headline(result: BacktestResult, baselines: Baselines, strategy: Metrics) -> str:
    initial = result.cfg.backtest.initial_equity
    cols = [
        _metric_column(strategy),
        _bh_column(baselines.buy_and_hold, initial),
        {} if baselines.random_entry is None else _metric_column(_metrics(baselines.random_entry)),
    ]
    rows = [[label, *(c.get(label, "n/a") for c in cols)] for label in cols[0]]
    return _table(["Metric", "Strategy", "Buy and hold", "Random entry (one seed)"], rows)


def _metrics(result: BacktestResult) -> Metrics:
    return compute_metrics(
        result.trades, result.equity, result.cfg.backtest.initial_equity, result.log.fills
    )


def _breakdown_html(
    result: BacktestResult, by: Literal["regime", "symbol", "strategy", "exit_reason"]
) -> str:
    df = breakdown(result.trades, by)
    rows = [
        [str(key), count(r.n), pct(r.win_rate), money(r.net_pnl), ratio(r.expectancy_r)]
        for key, r in df.iterrows()
    ]
    return _table(
        [by.replace("_", " ").title(), "Trades", "Win rate", "Net P&L", "Expectancy (R)"], rows
    )


def _slippage_html(result: BacktestResult, m: Metrics) -> str:
    costs = result.cfg.costs
    floor = costs.half_spread_bps + costs.slippage_bps
    rows = [
        ["Modelled floor (half spread + fixed slippage)", bps(floor), bps(floor)],
        [
            "Realised average slippage of taker fills",
            bps(m.avg_entry_slippage_bps),
            bps(m.avg_exit_slippage_bps),
        ],
    ]
    note = (
        "<p>The floor is what the cost model charges even on a tiny order. Realised slippage also "
        "includes the volume-impact term, and exits by stop pay it on top of the stop price. "
        f"Taker fee: {_num(costs.taker_fee_bps)} bps per side. All of it is already in the P&amp;L "
        "above.</p>"
    )
    return _table(["", "Entry fills", "Exit fills"], rows) + note


def _events_html(events: Sequence[RiskEvent]) -> str:
    if not events:
        return "<p>No risk-limit events were triggered.</p>"
    rows = [[e.ts.isoformat(), e.kind, e.action, e.detail] for e in events[:MAX_EVENT_ROWS]]
    note = (
        f"<p>{len(events)} events; showing the first {MAX_EVENT_ROWS}.</p>"
        if len(events) > MAX_EVENT_ROWS
        else ""
    )
    return note + _table(["Time (UTC)", "Kind", "Action", "Detail"], rows)


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>\n{body}\n"


def write_report(
    result: BacktestResult,
    baselines: Baselines,
    out_dir: Path,
    n_variants_tried: int | None = None,
) -> Path:
    """Write ``report.html``, ``trades.csv``, ``daily_pnl.csv``, ``equity.png`` and
    ``drawdown.png`` into ``out_dir`` and return the HTML path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.cfg
    metrics = _metrics(result)
    pd.DataFrame([asdict(t) for t in result.trades]).to_csv(out_dir / "trades.csv", index=False)
    _daily_pnl(result.equity, cfg.backtest.initial_equity).to_csv(out_dir / "daily_pnl.csv")
    equity_png = _equity_chart(result, baselines, out_dir / "equity.png")
    drawdown_png = _drawdown_chart(result, out_dir / "drawdown.png")

    warning = (
        '<div class="warn"><h2>Risk warning</h2><ul>'
        + "".join(f"<li>{html.escape(w)}</li>" for w in RISK_WARNING)
        + "</ul></div>"
    )
    variants = ""
    if n_variants_tried is not None:
        variants = _section(
            "Parameter variants tried",
            f"<p>{n_variants_tried} parameter variants were tried before this one was chosen. "
            "A result picked from many variants is likely overstated: the best of many "
            "random-looking outcomes looks good by luck.</p>",
        )
    symbols = ", ".join(cfg.exchange.symbols)
    info = (
        f"<p>Config hash: <code>{html.escape(config_hash(cfg))}</code><br>"
        f"Period: {result.start.isoformat()} to {result.end.isoformat()} (UTC)<br>"
        f"Symbols: {html.escape(symbols)}<br>"
        f"Initial equity: {money(cfg.backtest.initial_equity)}</p>"
        "<p>This is one in-sample backtest on past data. Every number is net of modelled taker "
        "fees, spread, slippage and funding. Buy and hold ignores funding and volume impact. "
        "The random-entry column is a single random seed, so treat it as one sample, not a "
        "distribution. If the strategy does not clearly beat both columns after costs, "
        "there is no evidence of an edge.</p>"
    )
    parts = [
        warning,
        _section("Run", info),
        _section("Headline metrics", _headline(result, baselines, metrics)),
        _section(
            "Equity and drawdown",
            f'<img alt="equity curve" src="data:image/png;base64,{equity_png}">'
            f'<img alt="drawdown" src="data:image/png;base64,{drawdown_png}">',
        ),
        _section("Breakdown by regime", _breakdown_html(result, "regime")),
        _section("Breakdown by symbol", _breakdown_html(result, "symbol")),
        _section("Breakdown by strategy", _breakdown_html(result, "strategy")),
        _section("Breakdown by exit reason", _breakdown_html(result, "exit_reason")),
        _section("Slippage: modelled versus realised", _slippage_html(result, metrics)),
        _section("Risk-limit events", _events_html(result.log.risk_events)),
        _section(
            "Result without the best 5% of trades",
            f"<p>Net P&amp;L with the best {best_trades_dropped(metrics.n_trades)} "
            "trade(s) removed: "
            f"{money(metrics.net_pnl_without_best_5pct)} (all trades: {money(metrics.net_pnl)}). "
            "If this is far lower or negative, the result depends on a few lucky trades.</p>",
        ),
        variants,
    ]
    path = out_dir / "report.html"
    path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>regimebot backtest report</title>"
        f"<style>{_CSS}</style></head><body><h1>regimebot backtest report</h1>"
        + "".join(parts)
        + "</body></html>",
        encoding="utf-8",
    )
    return path
