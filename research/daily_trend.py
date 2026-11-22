"""Daily trend-following prototype (vectorised, outside the engine).

Rule: each day at 00:00 UTC, per coin, signal = mean of sign(close - SMA_n) for
n in (20, 50, 100), in [-1, 1]. Weight = signal * equal-risk volatility target,
capped per coin and in gross. Positions decided at a close are held over the
next day. Costs: COST per unit of turnover, plus actual funding on the position.

Usage: python research/daily_trend.py BTC,ETH,SOL [--mode ls|lf] [--cost-mult 2]
Daily bars and funding are fetched from Binance USDT perpetuals (public data)
and cached under data/daily/.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

START = "2020-09-01"  # 100-day SMA plus 30-day volatility need warm-up before 2021
TRADE_START = "2021-01-01"
END = "2026-11-21"
COST = 8e-4  # taker 5 bps + slippage 2 bps + half spread 1 bps, per unit turnover
TARGET_VOL = 0.15  # annualised, whole portfolio if coins were uncorrelated
ASSET_CAP = 0.75
GROSS_CAP = 1.5
LOOKBACKS = (20, 50, 100)
CACHE = Path("data/daily")


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


def load(coin: str) -> tuple[pd.Series, pd.Series]:
    """Daily close (indexed by close time) and daily summed funding rate."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{coin}.csv"
    if not path.exists():
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        sym, end = f"{coin}/USDT:USDT", _ms(END)
        bars: list[list[float]] = []
        since = _ms(START)
        while since < end:
            page = ex.fetch_ohlcv(sym, "1d", since=since, limit=1000)
            if not page:
                break
            bars += page
            since = int(page[-1][0]) + 86_400_000
        close = pd.Series(
            [b[4] for b in bars],
            index=pd.to_datetime([b[0] for b in bars], unit="ms", utc=True) + pd.Timedelta("1D"),
        )
        rates: dict[pd.Timestamp, float] = {}
        since = _ms(START)
        while since < end:
            page = ex.fetch_funding_rate_history(sym, since=since, limit=1000)
            if not page:
                break
            for row in page:
                rates[pd.to_datetime(row["timestamp"], unit="ms", utc=True)] = row["fundingRate"]
            since = int(page[-1]["timestamp"]) + 1
        funding = pd.Series(rates).resample("1D", label="right", closed="right").sum()
        frame = pd.DataFrame({"close": close, "funding": funding.reindex(close.index).fillna(0.0)})
        frame[frame.index <= pd.Timestamp(END, tz="UTC")].to_csv(path)
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame["close"], frame["funding"]


def weights(closes: pd.DataFrame, mode: str) -> pd.DataFrame:
    signal = sum(np.sign(closes - closes.rolling(n).mean()) for n in LOOKBACKS) / len(LOOKBACKS)
    if mode == "lf":
        signal = signal.clip(lower=0)
    vol = closes.pct_change().rolling(30).std() * np.sqrt(365)
    n = closes.shape[1]
    w = (signal * (TARGET_VOL / np.sqrt(n) / vol).clip(upper=ASSET_CAP)).dropna(how="all").fillna(0)
    return w.div((w.abs().sum(axis=1) / GROSS_CAP).clip(lower=1), axis=0)


def returns(w: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, cost: float) -> pd.Series:
    held = w.shift(1)
    daily = held * closes.pct_change() - w.diff().abs().shift(1) * cost - held * funding
    out = daily.sum(axis=1)
    return out[out.index >= pd.Timestamp(TRADE_START, tz="UTC")]


def stats(r: pd.Series) -> dict[str, float]:
    eq = (1 + r).cumprod()
    return {
        "total": eq.iloc[-1] - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(365),
        "max_dd": (eq / eq.cummax() - 1).min(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coins")
    ap.add_argument("--mode", default="ls", choices=["ls", "lf"])
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--shuffles", type=int, default=200)
    a = ap.parse_args()
    coins = a.coins.split(",")
    data = {c: load(c) for c in coins}
    closes = pd.DataFrame({c: d[0] for c, d in data.items()})
    funding = pd.DataFrame({c: d[1] for c, d in data.items()}).fillna(0.0)
    cost = COST * a.cost_mult
    w = weights(closes, a.mode)
    r = returns(w, closes, funding, cost)
    s = stats(r)
    print(f"coins={len(coins)} mode={a.mode} cost_mult={a.cost_mult}")
    print(f"total={s['total']:+.2%} sharpe={s['sharpe']:.2f} max_dd={s['max_dd']:.2%}")
    print("by year:", {y: f"{(1 + x).prod() - 1:+.1%}" for y, x in r.groupby(r.index.year)})
    held = w.shift(1)
    per_coin = (held * closes.pct_change() - w.diff().abs().shift(1) * cost - held * funding)[
        r.index[0] :
    ].sum()
    print("coins with positive contribution:", int((per_coin > 0).sum()), "of", len(coins))
    print("contribution:", {c: f"{v:+.2f}" for c, v in per_coin.items()})
    # Timing test: shift each coin's weights by a random offset (keeps turnover and
    # long/short mix, destroys timing) and compare Sharpe.
    rng = np.random.default_rng(0)
    sharpes = []
    for _ in range(a.shuffles):
        shifted = pd.DataFrame(
            {c: np.roll(w[c].to_numpy(), rng.integers(30, len(w) - 30)) for c in coins},
            index=w.index,
        )
        sharpes.append(stats(returns(shifted, closes, funding, cost))["sharpe"])
    beaten = int((s["sharpe"] > np.array(sharpes)).sum())
    print(f"timing test: Sharpe above {beaten} of {a.shuffles} shifted runs")
    bh = closes.pct_change().mean(axis=1)[r.index[0] :]
    print(
        "equal-weight buy and hold:",
        {k: round(v, 2) for k, v in stats(bh).items()},
    )


if __name__ == "__main__":
    main()
