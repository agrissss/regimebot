"""Regime study: how regimes relate to what happened next (research reporting only).

Forward returns use future data by design; they must never be merged into the features frame.
"""

import numpy as np
import pandas as pd

from regimebot.types import Regime

BIG_MOVE = 0.05


def _episodes(regimes: pd.Series) -> pd.Series:
    """Number of maximal runs of consecutive equal labels, per label."""
    starts = regimes != regimes.shift(1)
    return regimes[starts].value_counts()


def _forward(close: pd.Series, h: int) -> tuple[pd.Series, pd.Series]:
    """Forward log return and max |close[t+k]/close[t]-1| (k=1..h); NaN without a full window."""
    c = close.to_numpy(dtype=float)
    n = len(c)
    ratios = np.full((h, n), np.nan)
    for k in range(1, h + 1):
        if k < n:
            ratios[k - 1, : n - k] = c[k:] / c[: n - k]
    ret = pd.Series(np.log(ratios[h - 1]), index=close.index)
    excursion = pd.Series(np.abs(ratios - 1.0).max(axis=0), index=close.index)  # NaN propagates
    return ret, excursion.where(ret.notna())


def regime_study(
    features: pd.DataFrame, regimes: pd.Series, horizons: tuple[int, ...] = (4, 16, 96)
) -> pd.DataFrame:
    """One row per regime: share of bars, episode count, and per horizon ``h`` the mean, std,
    5th and 95th percentile of the forward log return, the share of bars followed by an absolute
    close-to-close move of 5% or more within ``h`` bars, and the bar count used (``n_h``)."""
    regimes = regimes.reindex(features.index)
    rows: dict[str, dict[str, float | int]] = {}
    episodes = _episodes(regimes)
    fwd = {h: _forward(features["close"], h) for h in horizons}
    for regime in Regime:
        mask = regimes == regime.value
        row: dict[str, float | int] = {
            "share": float(mask.mean()) if len(mask) else float("nan"),
            "episodes": int(episodes.get(regime.value, 0)),
        }
        for h, (ret, excursion) in fwd.items():
            r = ret[mask].dropna()
            row[f"n_{h}"] = len(r)
            row[f"mean_{h}"] = float(r.mean()) if len(r) else float("nan")
            row[f"std_{h}"] = float(r.std()) if len(r) > 1 else float("nan")
            row[f"p05_{h}"] = float(r.quantile(0.05)) if len(r) else float("nan")
            row[f"p95_{h}"] = float(r.quantile(0.95)) if len(r) else float("nan")
            e = excursion[mask].dropna()
            row[f"big_move_{h}"] = float((e >= BIG_MOVE).mean()) if len(e) else float("nan")
        rows[regime.value] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "regime"
    return out
