"""Self-contained HTML report (inline CSS, no JavaScript, no external requests), CSVs and PNGs.

The report is built to avoid flattering the strategy: costs are shown, the strategy sits beside
buy-and-hold and random entry, the result without its best 5% of trades is shown, and a risk
warning comes first.
"""

import base64
import html
import io
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib.figure import Figure  # Figure renders with Agg and needs no GUI backend

from regimebot.backtest.baselines import Baselines
from regimebot.backtest.runner import BacktestResult
from regimebot.config import AppConfig, config_hash, gross_cap_limits
from regimebot.reporting.metrics import (
    BOOTSTRAP_MIN_TRADES,
    ExecutionQuality,
    Metrics,
    active_period,
    best_trades_dropped,
    breakdown,
    compute_metrics,
    daily_equity,
    daily_pnl,
    expectancy_ci,
    first_permanent_halt,
    largest_market_moves,
    max_drawdown,
    median_or_nan,
    meets_min_rr,
    sharpe_ratio,
    sortino_ratio,
    strategy_quality,
)
from regimebot.types import RiskEvent, Trade

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
PAPER_BANNER = "PAPER TRADING RUN (simulated fills on live data)"
CI_LABEL = "Expectancy 95% CI (bootstrap, 2,000 resamples of trades, R)"
_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;"
    "color:#222}table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
    "th,td{border:1px solid #bbb;padding:.25rem .6rem;text-align:right}"
    "th:first-child,td:first-child{text-align:left}.warn{border:3px solid #b00;background:#fff3f3;"
    "padding:.5rem 1rem;margin:1rem 0}img{max-width:100%}"
    ".banner{border:3px solid #036;background:#eef4fb;padding:.5rem 1rem;font-weight:bold}"
)


def _num(x: float, fmt: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return format(x, fmt)


def _money(x: float) -> str:
    return _num(x, ",.2f")


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else _num(x * 100) + "%"


def _ratio(x: float) -> str:
    return _num(x)


def _bps(x: float) -> str:
    return "n/a" if math.isnan(x) else _num(x) + " bps"


def _count(x: float) -> str:
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
        bh = baselines.buy_and_hold
        ax.plot(bh.index, bh.to_numpy(), label="buy and hold (active period)")
    if (
        baselines.buy_and_hold_full is not None
        and baselines.buy_and_hold is not baselines.buy_and_hold_full
    ):
        full = baselines.buy_and_hold_full
        ax.plot(full.index, full.to_numpy(), label="buy and hold (full period)", linestyle=":")
    for i, rnd in enumerate(baselines.random_entry):
        ax.plot(
            rnd.equity.index,
            rnd.equity.to_numpy(),
            label="random entry (each seed)" if i == 0 else None,
            color="gray",
            alpha=0.5,
            linewidth=0.8,
        )
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
    if len(values):  # a paper run before its first decision has no equity yet
        ax.fill_between(result.equity.index, -(1 - values / peak.to_numpy()) * 100, 0, color="#b00")
    ax.set_title("Strategy drawdown (%)")
    fig.autofmt_xdate()
    return _png(fig, path)


def _daily_pnl(equity: pd.Series, initial_equity: float) -> pd.DataFrame:
    out = pd.DataFrame({"equity": daily_equity(equity), "pnl": daily_pnl(equity, initial_equity)})
    out.index.name = "date"
    return out


def _bh_column(bh: pd.Series | None, initial: float) -> dict[str, str]:
    if bh is None or bh.empty:
        return {}
    final = float(bh.iloc[-1])
    return {
        "Net P&L": _money(final - initial),
        "Return": _pct(final / initial - 1),
        "Max drawdown": _pct(max_drawdown(bh, initial)),
        "Sharpe (daily, annualised)": _ratio(sharpe_ratio(bh)),
        "Sortino (daily, annualised)": _ratio(sortino_ratio(bh)),
    }


def _metric_column(m: Metrics) -> dict[str, str]:
    return {
        "Net P&L": _money(m.net_pnl),
        "Return": _pct(m.return_pct),
        "Max drawdown": _pct(m.max_drawdown),
        "Sharpe (daily, annualised)": _ratio(m.sharpe),
        "Sortino (daily, annualised)": _ratio(m.sortino),
        "Profit factor": _ratio(m.profit_factor),
        "Win rate": _pct(m.win_rate),
        "Average win": _money(m.avg_win),
        "Average loss": _money(m.avg_loss),
        "Expectancy (R per trade)": _ratio(m.expectancy_r),
        "Trades": _count(m.n_trades),
        "Average bars held": _ratio(m.avg_bars_held),
        "Exposure (share of decision bars in a position)": _pct(m.exposure_pct),
        "Turnover (entry notional / initial equity)": _ratio(m.turnover),
        "Fees paid": _money(m.fees),
        "Funding paid (negative = received)": _money(m.funding),
        "Target hit rate (all strategies; split by strategy below)": _pct(m.target_hit_rate),
        "Stop-out rate": _pct(m.stop_out_rate),
        "Net P&L without the best 5% of trades": _money(m.net_pnl_without_best_5pct),
        "Max gross leverage seen (gross exposure / equity)": _ratio(m.max_gross_leverage_seen),
    }


def _active_metrics(result: BacktestResult) -> tuple[Metrics, RiskEvent | None]:
    """Metrics over the active period (up to a permanent halt) and that halt event, if any."""
    trades, equity, halt = active_period(result.trades, result.equity, result.log.risk_events)
    initial = result.cfg.backtest.initial_equity
    gross = result.gross_exposure
    return compute_metrics(trades, equity, initial, result.log.fills, gross), halt


def _period(result: BacktestResult) -> tuple[str, str]:
    """(last timestamp, number of decision bars) of a run's active period."""
    _, equity, _ = active_period(result.trades, result.equity, result.log.risk_events)
    if equity.empty:
        return "n/a", "0"
    return equity.index[-1].isoformat(), _count(len(equity.index.unique()))


def _median_metrics(all_metrics: Sequence[Metrics]) -> Metrics:
    """Field-wise median over seeds (NaN ignored)."""
    values = {
        f.name: median_or_nan([float(getattr(m, f.name)) for m in all_metrics])
        for f in fields(Metrics)
    }
    values["n_trades"] = round(values["n_trades"])
    return Metrics(**values)  # type: ignore[arg-type]  # n_trades was rounded to an int above


def _random_column(baselines: Baselines) -> dict[str, str]:
    if not baselines.random_entry:
        return {}
    metrics = [_active_metrics(r)[0] for r in baselines.random_entry]
    col = _metric_column(_median_metrics(metrics))
    pnls = [m.net_pnl for m in metrics]
    col["Net P&L"] = f"{_money(median_or_nan(pnls))} ({_money(min(pnls))} to {_money(max(pnls))})"
    return col


def _halt_sentence(result: BacktestResult, halt: RiskEvent | None) -> str:
    if halt is None:
        return ""
    _, equity, _ = active_period(result.trades, result.equity, result.log.risk_events)
    end = equity.index[-1] if len(equity) else halt.ts  # the snapshot that realises the flatten
    return (
        f"<p><strong>Trading halted on {html.escape(halt.ts.isoformat())} "
        f"({html.escape(halt.kind)}). No entries after that; results below cover "
        f"{result.start.isoformat()} to {end.isoformat()} "
        "(includes the kill-switch flatten).</strong></p>"
    )


def _ci_text(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if math.isnan(lo):
        return f"n/a (under {BOOTSTRAP_MIN_TRADES} trades)"
    return f"{_ratio(lo)} to {_ratio(hi)}"


def _headline(
    result: BacktestResult,
    baselines: Baselines,
    strategy: Metrics,
    ci: tuple[float, float],
    period_end: str,
    paper: bool,
) -> str:
    initial = result.cfg.backtest.initial_equity
    strategy_col = {}
    for label, value in _metric_column(strategy).items():
        strategy_col[label] = value
        if label == "Expectancy (R per trade)":
            strategy_col[CI_LABEL] = _ci_text(ci)
    cols = [
        strategy_col,
        _bh_column(baselines.buy_and_hold, initial),
        {} if paper else _random_column(baselines),
    ]
    missing = ["n/a", "n/a", "n/a: backtest only" if paper else "n/a"]
    rows = [
        [label, *(c.get(label, na) for c, na in zip(cols, missing, strict=True))]
        for label in strategy_col
    ]
    out = _table(
        ["Metric", "Strategy", "Buy and hold", "Random entry (median of seeds, range)"], rows
    )
    full = baselines.buy_and_hold_full
    if full is not None and full is not baselines.buy_and_hold:
        c = _bh_column(full, initial)
        out += (
            "<p>Buy and hold over the full period "
            f"{result.start.isoformat()} to {period_end} (not comparable with the "
            f"columns above): net P&amp;L {c['Net P&L']}, return {c['Return']}, "
            f"max drawdown {c['Max drawdown']}.</p>"
        )
    return out


def _seeds_beaten(result: BacktestResult, baselines: Baselines) -> str:
    """How many random seeds the strategy beat on net P&L, each over its active period."""
    ref = _active_metrics(result)[0].net_pnl
    pnls = [_active_metrics(run)[0].net_pnl for run in baselines.random_entry]
    beaten, higher = sum(p < ref for p in pnls), sum(p > ref for p in pnls)
    return (
        f"<p><strong>Strategy beat {beaten} of {len(pnls)} random seeds</strong> on net P&amp;L "
        f"({higher} had a higher net P&amp;L).</p>"
    )


def _random_html(result: BacktestResult, baselines: Baselines, paper: bool) -> str:
    if paper:
        return (
            "<p>Random entry: n/a: backtest only. It reruns the engine over stored history with "
            "random entries; a paper run is compared with buy and hold over its own period.</p>"
        )
    if not baselines.random_entry:
        return "<p>Random-entry baseline not computed (n/a).</p>"
    rows = []
    ref, _ = _active_metrics(result)
    rows.append(
        [
            "Strategy (reference)",
            _count(baselines.reference_signals)
            if baselines.reference_signals is not None
            else "n/a",
            _count(ref.n_trades),
            _money(ref.net_pnl),
            *_period(result),
            "yes" if first_permanent_halt(result.log.risk_events) else "no",
        ]
    )
    for seed, run in zip(baselines.random_seeds, baselines.random_entry, strict=True):
        m, halt = _active_metrics(run)
        rows.append(
            [f"Random seed {seed}", _count(len(run.log.signals)), _count(m.n_trades),
             _money(m.net_pnl), *_period(run), "yes" if halt else "no"]
        )  # fmt: skip
    p = baselines.random_p
    note = (
        "<p>Random entry reruns the real engine, playbook and risk limits with entries drawn at "
        f"random (calibrated probability {'n/a' if p is None else f'{p:.5f}'} per question, set "
        "so the number of signals matches the strategy's). It uses breakout's ATR stop and "
        "target geometry and is only asked in regimes where breakout is allowed, so it never "
        "trades regimes the playbook closes to breakout (such as range). Each seed runs under the "
        "same risk limits as the strategy, over the strategy's active period (start to the period "
        "end shown), and is reported over its own active period if it halts earlier. "
        f"Seeds used: {', '.join(str(x) for x in baselines.random_seeds)}.</p>"
    )
    return (
        _seeds_beaten(result, baselines)
        + _table(
            [
                "Run",
                "Signals",
                "Trades",
                "Net P&L",
                "Period end (UTC)",
                "Decision bars",
                "Permanent halt",
            ],
            rows,
        )
        + note
    )


def _min_rr(cfg: AppConfig) -> dict[str, float]:
    """Each configured strategy's minimum reward-to-risk."""
    return {
        name: getattr(cfg.strategies, name).min_rr for name in type(cfg.strategies).model_fields
    }


def _breakdown_html(
    result: BacktestResult, by: Literal["regime", "symbol", "strategy", "exit_reason", "side"]
) -> str:
    df = breakdown(result.trades, by)
    headers = [by.replace("_", " ").title(), "Trades", "Win rate", "Net P&L", "Expectancy (R)"]
    rows = [
        [str(key), _count(r.n), _pct(r.win_rate), _money(r.net_pnl), _ratio(r.expectancy_r)]
        for key, r in df.iterrows()
    ]
    if by != "strategy":
        return _table(headers, rows)
    min_rr = _min_rr(result.cfg)
    quality = strategy_quality(result.trades, min_rr)
    headers += ["Target hit rate", "Met min reward-to-risk at the entry fill"]
    for row, key in zip(rows, df.index, strict=True):
        hit, met = quality.loc[key, ["target_hit_rate", "met_min_rr"]].tolist()
        row += [_pct(hit), _pct(met)]
    known = [t for t in result.trades if t.strategy in min_rr]
    share = (
        _pct(sum(meets_min_rr(t, min_rr[t.strategy]) for t in known) / len(known))
        if known
        else "n/a"
    )
    note = (
        f"<p>{share} of trades met their strategy's minimum reward-to-risk (min_rr) at the actual "
        "entry fill price: |target - entry| / |entry - initial stop|. Signals meet it at the "
        "decision close; the fill moves the entry against the trade. The target hit rate is "
        "split by strategy because mean reversion's target is not a 5% move.</p>"
    )
    return _table(headers, rows) + note


def _market_moves_html(result: BacktestResult, baselines: Baselines) -> str:
    if baselines.closes is None:
        return "<p>Market data not loaded (n/a).</p>"
    _, equity, _ = active_period(result.trades, result.equity, result.log.risk_events)
    moves = largest_market_moves(baselines.closes, equity, result.cfg.backtest.initial_equity)
    if moves.empty:
        return "<p>No full market day in the active period (n/a).</p>"
    rows = [
        [day.date().isoformat(), _pct(ret), _money(pnl)]
        for day, ret, pnl in zip(
            pd.DatetimeIndex(moves.index),
            moves["market_return"],
            moves["strategy_pnl"],
            strict=True,
        )
    ]
    note = (
        f"<p>The {len(moves)} UTC days of the active period with the largest absolute "
        "equal-weight average daily return of the configured symbols (each symbol's last "
        "decision close of the day against the day before), largest first, with the strategy's "
        "P&amp;L on that day from its equity curve.</p>"
    )
    return note + _table(["Date (UTC)", "Market return", "Strategy P&L"], rows)


def _secs(x: float) -> str:
    return "n/a" if math.isnan(x) else _num(x) + " s"


def _execution_html(e: ExecutionQuality) -> str:
    summary = _table(
        ["", "Value"],
        [
            ["Fills", _count(e.n_fills)],
            ["Decision-to-fill lag of entries, median", _secs(e.entry_lag_median_s)],
            ["Decision-to-fill lag of entries, 95th percentile", _secs(e.entry_lag_p95_s)],
            [
                "Average quoted spread at fill (live quotes: entries and market exits)",
                _bps(e.avg_spread_bps),
            ],
        ],
    )
    slippage = _table(
        ["Fill type", "Fills", "Average slippage versus the reference price"],
        [[kind, _count(n), _bps(bps)] for kind, (n, bps) in e.slippage_bps.items()],
    )
    cancelled = _table(
        ["Rule", "Approved entries cancelled"],
        [[rule, _count(n)] for rule, n in e.cancelled.items()],
    )
    note = (
        "<p>Lag runs from the decision close to the simulated fill at the live quote. The "
        "reference price is the decision close for entries and market exits, the stop level for "
        "stops and the target for targets; positive slippage is worse for the trade.</p>"
    )
    return summary + slippage + cancelled + note


def _slippage_html(result: BacktestResult, m: Metrics) -> str:
    costs = result.cfg.costs
    floor = costs.half_spread_bps + costs.slippage_bps
    rows = [
        ["Modelled floor (half spread + fixed slippage)", _bps(floor), _bps(floor)],
        [
            "Realised average slippage of taker fills",
            _bps(m.avg_entry_slippage_bps),
            _bps(m.avg_exit_slippage_bps),
        ],
    ]
    note = (
        "<p>The floor is what the cost model charges even on a tiny order. Realised slippage also "
        "includes the volume-impact term, the move from the decision close to the next open "
        "(entries fill there), and gaps through stop prices. "
        f"Taker fee: {_num(costs.taker_fee_bps)} bps per side. All of it is already in the P&amp;L "
        "above.</p>"
    )
    return _table(["", "Entry fills", "Exit fills"], rows) + note


def _events_html(events: Sequence[RiskEvent]) -> str:
    """Counts by kind, every halt event, then at most ``MAX_EVENT_ROWS`` other events."""
    if not events:
        return "<p>No risk-limit events were triggered.</p>"
    counts = Counter((e.kind, e.action) for e in events)
    summary = _table(
        ["Kind", "Action", "Count"], [[k, a, _count(n)] for (k, a), n in sorted(counts.items())]
    )
    halts = [e for e in events if e.action == "halt"]
    others = [e for e in events if e.action != "halt"]
    shown = halts + others[:MAX_EVENT_ROWS]
    rows = [[e.ts.isoformat(), e.kind, e.action, e.detail] for e in shown]
    note = ""
    if len(others) > MAX_EVENT_ROWS:
        note = (
            f"<p>All {len(halts)} halt events are listed, plus the first {MAX_EVENT_ROWS} of "
            f"{len(others)} other events.</p>"
        )
    return summary + note + _table(["Time (UTC)", "Kind", "Action", "Detail"], rows)


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>\n{body}\n"


def write_report(
    result: BacktestResult,
    baselines: Baselines,
    out_dir: Path,
    n_variants_tried: int | None = None,
    execution: ExecutionQuality | None = None,
) -> Path:
    """Write ``report.html``, ``trades.csv``, ``daily_pnl.csv``, ``equity.png`` and
    ``drawdown.png`` into ``out_dir`` and return the HTML path.

    ``execution`` makes it a paper-trading report: a banner, the period from the first to the
    last equity snapshot, random entry shown as backtest-only, and an execution-quality section.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.cfg
    paper = execution is not None
    metrics, halt = _active_metrics(result)
    active_trades, _, _ = active_period(result.trades, result.equity, result.log.risk_events)
    ci = expectancy_ci([t.r_multiple for t in active_trades])
    if paper:
        last = result.equity.index[-1] if len(result.equity) else result.start
        period_end = last.isoformat()
    else:
        period_end = result.end.isoformat()
    pd.DataFrame(
        [asdict(t) for t in result.trades], columns=[f.name for f in fields(Trade)]
    ).to_csv(out_dir / "trades.csv", index=False)
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
    gross_cap = cfg.risk.max_gross_leverage
    limits = gross_cap_limits(cfg.risk, cfg.exchange.symbols)
    effective_cap = min(limits.values())
    n = len(set(cfg.exchange.symbols))
    reasons = {
        "group": "one group holds every symbol: its cap binds",
        "asset": f"{n} symbol{'' if n == 1 else 's'} x the "
        f"{_num(cfg.risk.max_asset_exposure)}x per-asset cap binds",
    }
    binding = [text for cap, text in reasons.items() if limits.get(cap) == effective_cap]
    why = f" ({'; '.join(binding)})" if effective_cap < gross_cap else ""
    if paper:
        period = (
            f"Paper run period: {result.start.isoformat()} to {period_end} (UTC; first to last "
            "equity snapshot)"
        )
        honesty = (
            "<p>This is a paper-trading run: live market data, simulated fills (no order was "
            "sent). Every number is net of all modelled fees (taker and maker), spread, slippage "
            "and funding. Buy and hold covers the same period and ignores funding and volume "
            "impact; random entry is a backtest-only baseline. A paper run shows whether the "
            "system trades as its backtest would, on a short sample; it is not evidence of an "
            "edge.</p>"
        )
    else:
        period = f"Backtest period: {result.start.isoformat()} to {period_end} (UTC)"
        honesty = (
            "<p>This is one in-sample backtest on past data. Every number is net of all modelled "
            "fees (taker and maker), spread, slippage and funding. Buy and hold ignores funding "
            "and volume impact. The random-entry column summarises a few seeds, which is still a "
            "small sample. If the strategy does not clearly beat both columns after costs, there "
            "is no evidence of an edge.</p>"
        )
    info = (
        f"<p>Config hash: <code>{html.escape(config_hash(cfg))}</code><br>"
        f"{period}<br>"
        f"Symbols: {html.escape(symbols)}<br>"
        f"Initial equity: {_money(cfg.backtest.initial_equity)}<br>"
        f"Leverage: configured gross cap {_num(gross_cap)}x, effective cap "
        f"{_num(effective_cap)}x{why}</p>"
        f"{_halt_sentence(result, halt)}"
        f"{honesty}"
    )
    headline = _headline(result, baselines, metrics, ci, period_end, paper)
    parts = [
        warning,
        _section("Run", info),
        _section("Headline metrics", _halt_sentence(result, halt) + headline),
        _section("Random-entry baseline", _random_html(result, baselines, paper)),
        _section(
            "Equity and drawdown",
            f'<img alt="equity curve" src="data:image/png;base64,{equity_png}">'
            f'<img alt="drawdown" src="data:image/png;base64,{drawdown_png}">',
        ),
        _section("Breakdown by regime", _breakdown_html(result, "regime")),
        _section("Breakdown by symbol", _breakdown_html(result, "symbol")),
        _section("Breakdown by strategy", _breakdown_html(result, "strategy")),
        _section("Breakdown by side", _breakdown_html(result, "side")),
        _section("Breakdown by exit reason", _breakdown_html(result, "exit_reason")),
        _section("Largest market moves", _market_moves_html(result, baselines)),
        _section("Slippage: modelled versus realised", _slippage_html(result, metrics)),
        _section("Execution quality", _execution_html(execution)) if execution else "",
        _section("Risk-limit events", _events_html(result.log.risk_events)),
        _section(
            "Result without the best 5% of trades",
            f"<p>Net P&amp;L with the best {best_trades_dropped(metrics.n_trades)} "
            "trade(s) removed: "
            f"{_money(metrics.net_pnl_without_best_5pct)} (all trades: {_money(metrics.net_pnl)}). "
            "If this is far lower or negative, the result depends on a few lucky trades.</p>",
        ),
        variants,
    ]
    title = "regimebot paper trading report" if paper else "regimebot backtest report"
    banner = f'<p class="banner">{PAPER_BANNER}</p>' if paper else ""
    path = out_dir / "report.html"
    path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"<style>{_CSS}</style></head><body><h1>{title}</h1>{banner}"
        + "".join(parts)
        + "</body></html>",
        encoding="utf-8",
    )
    return path
