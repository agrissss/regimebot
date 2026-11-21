"""Robustness checks: parameter sensitivity, doubled costs, lower leverage.

Each function re-runs the backtest on ``[start, end)`` with a modified config; all other
settings stay at ``cfg``. ``strategies`` is an optional pass-through to ``run_backtest``.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd
import structlog
from pydantic import BaseModel

from regimebot.backtest.walk_forward import run_metrics
from regimebot.config import AppConfig, with_overrides
from regimebot.reporting.metrics import Metrics
from regimebot.strategy import Strategy

log = structlog.get_logger(__name__)

COST_KEYS = (
    "taker_fee_bps",
    "maker_fee_bps",
    "half_spread_bps",
    "slippage_bps",
    "impact_bps_per_pct",
)


def _get(cfg: AppConfig, dotted: str) -> Any:
    """Value at a dotted config path (attributes, or keys of dict nodes such as ``playbook``)."""
    node: Any = cfg
    for key in dotted.split("."):
        if isinstance(node, Mapping):
            found = [k for k in node if str(getattr(k, "value", k)) == key]
            if not found:
                raise ValueError(f"unknown config path: {dotted!r}")
            node = node[found[0]]
        elif isinstance(node, BaseModel) and key in type(node).model_fields:
            node = getattr(node, key)
        else:
            raise ValueError(f"unknown config path: {dotted!r}")
    return node


def sensitivity(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    params: Sequence[str],
    rel: float = 0.2,
    strategies: Sequence[Strategy] | None = None,
) -> pd.DataFrame:
    """Each numeric parameter at ``(1 - rel)``, base and ``(1 + rel)`` of its value, others fixed.

    The base config is run once and reused. Ints are rounded and kept >= 1; a variant that
    equals the base after rounding reuses the base result and says so in ``note``. A value the
    config rejects gives a row of NaNs with the reason in ``note``. Unknown or non-numeric
    paths raise ``ValueError`` before any run. Columns: param, variant, value, net_pnl,
    profit_factor, n_trades, note.
    """
    bases = {}
    for param in params:
        bases[param] = _get(cfg, param)
        if isinstance(bases[param], bool) or not isinstance(bases[param], int | float):
            raise ValueError(f"{param} is not a numeric parameter (value {bases[param]!r})")
    base_m, _, base_halt = run_metrics(cfg, candles, funding, start, end, strategies)
    rows = []
    for param, base in bases.items():
        for label, factor in ((f"-{rel:.0%}", 1 - rel), ("base", 1.0), (f"+{rel:.0%}", 1 + rel)):
            value = base * factor
            if isinstance(base, int):
                value = max(1, round(value))
            row: dict[str, Any] = {"param": param, "variant": label, "value": value, "note": ""}
            m: Metrics | None = base_m
            halted = base_halt
            if value == base:
                if label != "base":
                    row["note"] = "unchanged after rounding"
            else:
                try:
                    variant = with_overrides(cfg, {param: value})
                except ValueError as exc:
                    m = None
                    row.update(
                        net_pnl=math.nan, profit_factor=math.nan, n_trades=math.nan, halted_at=""
                    )
                    row["note"] = f"invalid: {str(exc).splitlines()[0]}"
                else:
                    m, _, halted = run_metrics(variant, candles, funding, start, end, strategies)
                    log.info("sensitivity", param=param, variant=label, value=value)
            if m is not None:
                row.update(net_pnl=m.net_pnl, profit_factor=m.profit_factor, n_trades=m.n_trades)
                row["halted_at"] = halted
            rows.append(row)
    cols = [
        "param",
        "variant",
        "value",
        "net_pnl",
        "profit_factor",
        "n_trades",
        "halted_at",
        "note",
    ]
    return pd.DataFrame(rows, columns=cols)


def cost_stress(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    mult: float = 2.0,
    strategies: Sequence[Strategy] | None = None,
) -> pd.DataFrame:
    """Rows ``base`` and ``stressed`` (every cost bps field times ``mult``)."""
    stressed = with_overrides(cfg, {f"costs.{k}": getattr(cfg.costs, k) * mult for k in COST_KEYS})
    rows = []
    for label, c in (("base", cfg), ("stressed", stressed)):
        m, _, halted = run_metrics(c, candles, funding, start, end, strategies)
        log.info("cost_stress", variant=label)
        rows.append(
            {
                "variant": label,
                "net_pnl": m.net_pnl,
                "profit_factor": m.profit_factor,
                "n_trades": m.n_trades,
                "fees": m.fees,
                "halted_at": halted,
            }
        )
    return pd.DataFrame(rows)


def leverage_compare(
    cfg: AppConfig,
    candles: Mapping[str, pd.DataFrame],
    funding: Mapping[str, pd.Series],
    start: datetime,
    end: datetime,
    strategies: Sequence[Strategy] | None = None,
) -> pd.DataFrame:
    """Two rows for ``risk.max_gross_leverage`` 1.0 and 1.5."""
    rows = []
    for lev in (1.0, 1.5):
        m, _, halted = run_metrics(
            with_overrides(cfg, {"risk.max_gross_leverage": lev}),
            candles,
            funding,
            start,
            end,
            strategies,
        )
        log.info("leverage_compare", max_gross_leverage=lev)
        rows.append(
            {
                "max_gross_leverage": lev,
                "net_pnl": m.net_pnl,
                "max_drawdown": m.max_drawdown,
                "n_trades": m.n_trades,
                "halted_at": halted,
            }
        )
    return pd.DataFrame(rows)
