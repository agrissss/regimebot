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

from regimebot.backtest.walk_forward import run_metrics
from regimebot.config import AppConfig, with_overrides
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
    node: Any = cfg
    for key in dotted.split("."):
        node = getattr(node, key)
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

    Ints are rounded and kept >= 1. A value the config rejects gives a row of NaNs with the
    reason in ``note``. Columns: param, variant, value, net_pnl, profit_factor, n_trades, note.
    """
    rows = []
    for param in params:
        base = _get(cfg, param)
        if isinstance(base, bool) or not isinstance(base, int | float):
            raise ValueError(f"{param} is not a numeric parameter (value {base!r})")
        for label, factor in ((f"-{rel:.0%}", 1 - rel), ("base", 1.0), (f"+{rel:.0%}", 1 + rel)):
            value = base * factor
            if isinstance(base, int):
                value = max(1, round(value))
            row: dict[str, Any] = {"param": param, "variant": label, "value": value, "note": ""}
            try:
                variant = with_overrides(cfg, {param: value})
            except ValueError as exc:
                row.update(net_pnl=math.nan, profit_factor=math.nan, n_trades=math.nan)
                row["note"] = f"invalid: {str(exc).splitlines()[0]}"
            else:
                m, _ = run_metrics(variant, candles, funding, start, end, strategies)
                row.update(net_pnl=m.net_pnl, profit_factor=m.profit_factor, n_trades=m.n_trades)
                log.info("sensitivity", param=param, variant=label, value=value)
            rows.append(row)
    cols = ["param", "variant", "value", "net_pnl", "profit_factor", "n_trades", "note"]
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
        m, _ = run_metrics(c, candles, funding, start, end, strategies)
        log.info("cost_stress", variant=label)
        rows.append(
            {
                "variant": label,
                "net_pnl": m.net_pnl,
                "profit_factor": m.profit_factor,
                "n_trades": m.n_trades,
                "fees": m.fees,
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
        m, _ = run_metrics(
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
            }
        )
    return pd.DataFrame(rows)
