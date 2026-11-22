"""4-hour trend-following prototype: research/daily_trend.py's rule on 4-hour bars.

Rule: at each 4-hour close, per coin, signal = mean of sign(close - SMA_n) for n
in LOOKBACKS bars; weight = signal * equal-risk volatility target (30 days of
4-hour returns), capped per coin and in gross. A coin is only traded when its
target differs from the held weight by more than BAND. Costs as in the daily
prototype: COST per unit of turnover plus actual funding.

Usage: python research/trend_4h.py BTC,ETH,SOL [--lookbacks 20,50,100] [--band 0.01]
       [--cost-mult 2] [--mode ls|lf]
Bars and funding come from Binance USDT perpetuals, cached under data/h4/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from daily_trend import ASSET_CAP, COST, END, GROSS_CAP, START, TARGET_VOL, TRADE_START, _ms

CACHE = Path("data/h4")
BAR = pd.Timedelta("4h")
PER_YEAR = 365 * 6


def load(coin: str) -> pd.DataFrame:
    """4-hour close (indexed by close time) and funding summed over each bar."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{coin}.csv"
    if not path.exists():
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        sym, end = f"{coin}/USDT:USDT", _ms(END)
        bars: list[list[float]] = []
        since = _ms(START)
        while since < end:
            page = ex.fetch_ohlcv(sym, "4h", since=since, limit=1000)
            if not page:
                break
            bars += page
            since = int(page[-1][0]) + 14_400_000
        close = pd.Series(
            [b[4] for b in bars],
            index=pd.to_datetime([b[0] for b in bars], unit="ms", utc=True) + BAR,
        )
        rates: dict[pd.Timestamp, float] = {}
        since = _ms(START)
        while since < end:
            page = ex.fetch_funding_rate_history(sym, since=since, limit=1000)
            if not page:
                break
            for row in page:
                # Binance stamps funding a few ms after the hour: floor to the second.
                ts = pd.to_datetime(row["timestamp"], unit="ms", utc=True).floor("s")
                rates[ts] = row["fundingRate"]
            since = int(page[-1]["timestamp"]) + 1
        funding = pd.Series(rates).resample("4h", label="right", closed="right").sum()
        frame = pd.DataFrame({"close": close, "funding": funding.reindex(close.index).fillna(0.0)})
        frame[frame.index <= pd.Timestamp(END, tz="UTC")].to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def weights(
    closes: pd.DataFrame, lookbacks: tuple[int, ...], band: float, mode: str
) -> pd.DataFrame:
    signal = sum(np.sign(closes - closes.rolling(n).mean()) for n in lookbacks) / len(lookbacks)
    if mode == "lf":
        signal = signal.clip(lower=0)
    vol = closes.pct_change().rolling(180).std() * np.sqrt(PER_YEAR)
    n = closes.shape[1]
    w = (signal * (TARGET_VOL / np.sqrt(n) / vol).clip(upper=ASSET_CAP)).dropna(how="all").fillna(0)
    target = w.div((w.abs().sum(axis=1) / GROSS_CAP).clip(lower=1), axis=0).to_numpy()
    held = np.zeros_like(target)
    for i in range(1, len(target)):  # trade a coin only when it is off target by more than band
        move = np.abs(target[i] - held[i - 1]) > band
        held[i] = np.where(move, target[i], held[i - 1])
    return pd.DataFrame(held, index=w.index, columns=w.columns)


def returns(
    w: pd.DataFrame, closes: pd.DataFrame, funding: pd.DataFrame, cost: float
) -> pd.DataFrame:
    held = w.shift(1)
    r = held * closes.pct_change() - w.diff().abs().shift(1) * cost - held * funding
    return r[r.index >= pd.Timestamp(TRADE_START, tz="UTC")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coins")
    ap.add_argument("--lookbacks", default="20,50,100")
    ap.add_argument("--band", type=float, default=0.01)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--mode", default="ls", choices=["ls", "lf"])
    ap.add_argument("--shuffles", type=int, default=200)
    a = ap.parse_args()
    coins = a.coins.split(",")
    lookbacks = tuple(int(x) for x in a.lookbacks.split(","))
    data = {c: load(c) for c in coins}
    closes = pd.DataFrame({c: d["close"] for c, d in data.items()})
    funding = pd.DataFrame({c: d["funding"] for c, d in data.items()}).fillna(0.0)
    cost = COST * a.cost_mult
    w = weights(closes, lookbacks, a.band, a.mode)
    per_coin = returns(w, closes, funding, cost)
    r = per_coin.sum(axis=1)
    eq = (1 + r).cumprod()
    sharpe = r.mean() / r.std() * np.sqrt(PER_YEAR)
    print(
        f"coins={len(coins)} lookbacks={lookbacks} band={a.band} mode={a.mode} "
        f"cost_mult={a.cost_mult}"
    )
    print(
        f"total={eq.iloc[-1] - 1:+.2%} sharpe={sharpe:.2f} "
        f"max_dd={(eq / eq.cummax() - 1).min():.2%}"
    )
    print("by year:", {y: f"{(1 + x).prod() - 1:+.1%}" for y, x in r.groupby(r.index.year)})
    for label, rule in (("weeks", "W"), ("months", "ME")):
        x = (1 + r).resample(rule).prod() - 1
        print(f"positive {label}: {(x > 0).mean():.0%} of {len(x)}, worst {x.min():+.2%}")
    years = len(r) / PER_YEAR
    turnover = w.diff().abs().sum().sum() / years
    print(f"turnover per year: {turnover:.1f}x equity, cost drag {turnover * cost:.2%} a year")
    print("coins with positive contribution:", int((per_coin.sum() > 0).sum()), "of", len(coins))
    rng = np.random.default_rng(0)
    beaten = 0
    for _ in range(a.shuffles):
        shifted = pd.DataFrame(
            {c: np.roll(w[c].to_numpy(), rng.integers(180, len(w) - 180)) for c in coins},
            index=w.index,
        )
        s = returns(shifted, closes, funding, cost).sum(axis=1)
        beaten += sharpe > s.mean() / s.std() * np.sqrt(PER_YEAR)
    print(f"timing test: Sharpe above {beaten} of {a.shuffles} shifted runs")


if __name__ == "__main__":
    main()
