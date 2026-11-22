"""Funding carry prototype (vectorised, outside the engine).

Rule: per coin, each day at 00:00 UTC, hold spot and short the same notional of
the perpetual while the trailing 7-day funding sum is positive; otherwise hold
cash. Half the coin's capital buys spot, half is perp margin, so the return on
capital is funding / 2. Cost: COST of notional per entry or exit (both legs).
Spot-perp basis moves are ignored (they net to zero over a full hold).

Usage: python research/funding_carry.py BTC,ETH,SOL [--cost-mult 2]
Reuses the daily cache of research/daily_trend.py.
"""

from __future__ import annotations

import argparse

import pandas as pd
from daily_trend import TRADE_START, load, stats

COST = 21e-4  # spot taker 10 bps + perp taker 5 bps + 3 bps slippage/spread per leg
LOOKBACK = 7


def returns(funding: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Per-coin daily return on that coin's capital slice."""
    on = (funding.rolling(LOOKBACK).sum() > 0).astype(float)
    held = on.shift(1).fillna(0.0)
    r = (held * funding - on.diff().abs().shift(1).fillna(0.0) * cost) / 2
    return r[r.index >= pd.Timestamp(TRADE_START, tz="UTC")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coins")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    a = ap.parse_args()
    coins = a.coins.split(",")
    funding = pd.DataFrame({c: load(c)[1] for c in coins}).fillna(0.0)
    per_coin = returns(funding, COST * a.cost_mult)
    r = per_coin.mean(axis=1)
    s = stats(r)
    print(f"coins={len(coins)} cost_mult={a.cost_mult}")
    print(f"total={s['total']:+.2%} max_dd={s['max_dd']:.2%}")
    print("by year:", {y: f"{(1 + x).prod() - 1:+.1%}" for y, x in r.groupby(r.index.year)})
    for label, rule in (("weeks", "W"), ("months", "ME")):
        x = (1 + r).resample(rule).prod() - 1
        print(f"positive {label}: {(x > 0).mean():.0%} of {len(x)}, worst {x.min():+.2%}")
    print("coins with positive contribution:", int((per_coin.sum() > 0).sum()), "of", len(coins))
    switches = (funding.rolling(LOOKBACK).sum() > 0).astype(float).diff().abs().sum()
    print("entries+exits per coin:", {c: int(v) for c, v in switches.items()})


if __name__ == "__main__":
    main()
