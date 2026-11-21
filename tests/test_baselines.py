"""Buy-and-hold on hand-computed series; random entry is deterministic and uses the real engine."""

from datetime import timedelta

import pandas as pd
import pytest
from synth import BTC, T0, regime_walk

from regimebot.backtest.baselines import buy_and_hold, random_entry
from regimebot.config import AppConfig, CostsConfig, load_config
from regimebot.types import Regime

FREE = CostsConfig(
    taker_fee_bps=0,
    maker_fee_bps=0,
    half_spread_bps=0,
    slippage_bps=0,
    impact_bps_per_pct=0,
    max_participation=1,
)
COSTS = FREE.model_copy(update={"taker_fee_bps": 5, "half_spread_bps": 1, "slippage_bps": 1})


def decision_closes(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range(T0 + timedelta(minutes=15), periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def test_flat_series_without_costs_stays_at_initial_equity() -> None:
    eq = buy_and_hold({BTC: decision_closes([100.0] * 4)}, 1000.0, FREE)
    assert eq.tolist() == [1000.0] * 4


def test_doubling_series_without_costs_doubles() -> None:
    eq = buy_and_hold({BTC: decision_closes([100.0, 150.0, 200.0])}, 1000.0, FREE)
    assert eq.tolist() == [1000.0, 1500.0, 2000.0]  # 10 units bought at 100


def test_costs_hit_the_first_mark_and_the_final_mark() -> None:
    eq = buy_and_hold({BTC: decision_closes([100.0, 100.0, 100.0])}, 1000.0, COSTS)
    buy = 1.0002 * 1.0005  # price * (1 + spread + slippage), then fee on the notional
    sell = 0.9998 * 0.9995
    # qty = 1000 / (100 * buy), so the start mark is 100 * qty = 1000 / buy
    assert eq.iloc[0] == pytest.approx(1000 / buy)
    assert eq.iloc[1] == pytest.approx(1000 / buy)
    # the last point also pays to sell: qty * 100 * sell
    assert eq.iloc[2] == pytest.approx(1000 * sell / buy)


def test_two_symbols_share_equally_and_cash_waits_for_a_late_symbol() -> None:
    early = decision_closes([100.0, 100.0, 200.0])  # 5 units from 500
    late = decision_closes([10.0, 20.0]).iloc[1:]  # first bar at index 1; 25 units from 500 at 20
    eq = buy_and_hold({"A": early, "B": late}, 1000.0, FREE)
    assert eq.index.equals(early.index)
    # t0: A 500 + B still cash 500; t1: A 500 + B 25*20; t2: A 1000 + B 25*20 (last close 20 held)
    assert eq.tolist() == [1000.0, 1000.0, 1500.0]


def walk_cfg() -> tuple[AppConfig, dict[str, pd.DataFrame]]:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    overrides: dict[str, object] = {f"playbook.{r.value}": entry for r in Regime}
    cfg = load_config(overrides={"exchange.symbols": [BTC], **overrides})
    return cfg, {BTC: regime_walk(3 * 1440, seed=5)}


def run_random(seed: int, signals: int = 60) -> list[tuple[object, ...]]:
    cfg, candles = walk_cfg()
    result = random_entry(
        cfg,
        candles,
        {},
        T0 + timedelta(days=1),
        T0 + timedelta(days=3),
        seed,
        n_reference_signals=signals,
        n_decision_bars=192,
    )
    return [(s.ts, s.side, s.stop, s.strategy) for s in result.log.signals]


def test_random_entry_is_deterministic_for_a_seed_and_differs_across_seeds() -> None:
    first, again, other = run_random(1), run_random(1), run_random(2)
    assert first == again
    assert first != other
    assert first and {row[3] for row in first} == {"breakout"}
    assert {row[1] for row in first} == {"long", "short"}


def test_random_entry_with_no_reference_signals_never_enters() -> None:
    assert run_random(1, signals=0) == []
