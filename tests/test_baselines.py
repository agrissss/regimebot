"""Buy-and-hold on hand-computed series; random entry is deterministic and uses the real engine."""

from datetime import timedelta

import pandas as pd
import pytest
from synth import BTC, T0, regime_walk

from regimebot.backtest.baselines import (
    _hold_until,
    buy_and_hold,
    calibrate_p,
    compute_baselines,
    random_entry,
)
from regimebot.config import AppConfig, CostsConfig, load_config
from regimebot.reporting.metrics import active_period, first_permanent_halt
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


def walk_cfg(**extra: object) -> tuple[AppConfig, dict[str, pd.DataFrame]]:
    entry = {
        "strategies": ["breakout"],
        "size_mult": 1.0,
        "allow_entries": True,
        "force_close": False,
    }
    overrides: dict[str, object] = {f"playbook.{r.value}": entry for r in Regime}
    cfg = load_config(overrides={"exchange.symbols": [BTC], **overrides, **extra})
    return cfg, {BTC: regime_walk(3 * 1440, seed=5)}


def run_random(seed: int, p: float = 0.2) -> list[tuple[object, ...]]:
    cfg, candles = walk_cfg()
    result = random_entry(
        cfg, candles, {}, T0 + timedelta(days=1), T0 + timedelta(days=3), seed, p=p
    )
    return [(s.ts, s.side, s.stop, s.strategy) for s in result.log.signals]


def test_random_entry_is_deterministic_for_a_seed_and_differs_across_seeds() -> None:
    first, again, other = run_random(1), run_random(1), run_random(2)
    assert first == again
    assert first != other
    assert first and {row[3] for row in first} == {"breakout"}
    assert {row[1] for row in first} == {"long", "short"}


def test_random_entry_with_zero_probability_never_enters() -> None:
    assert run_random(1, p=0.0) == []


def calibrate(n_reference_signals: int, n_decision_bars: int) -> float:
    cfg, candles = walk_cfg()
    return calibrate_p(
        cfg,
        candles,
        {},
        T0 + timedelta(days=1),
        T0 + timedelta(days=3),
        n_reference_signals=n_reference_signals,
        n_decision_bars=n_decision_bars,
    )


def test_calibration_matches_the_reference_signal_count() -> None:
    reference = len(run_random(9, p=0.15))
    assert reference > 5
    p = calibrate(reference, 192)
    counts = [len(run_random(seed, p=p)) for seed in (0, 1, 2)]
    mean = sum(counts) / len(counts)
    assert abs(mean - reference) <= max(0.35 * reference, 3), (counts, reference, p)


def test_calibration_with_no_realised_signals_keeps_the_naive_probability() -> None:
    p0 = 1 / 10**9  # one reference signal over 1e9 calls: the first pass signals nothing
    assert run_random(0, p=p0) == []  # N1 == 0
    assert calibrate(1, 10**9) == min(1.0, p0 * 1)  # p1 = p0 * reference / max(N1, 1)
    assert calibrate(0, 192) == 0.0


def test_compute_baselines_is_deterministic_for_a_seed_tuple() -> None:
    cfg, candles = walk_cfg()
    ref = random_entry(cfg, candles, {}, T0, T0 + timedelta(days=3), 4, p=0.1)
    a = compute_baselines(ref, candles, {}, len(ref.log.signals), seeds=(0, 1))
    b = compute_baselines(ref, candles, {}, len(ref.log.signals), seeds=(0, 1))
    assert a.random_p == b.random_p and a.random_seeds == (0, 1)
    assert [len(r.log.signals) for r in a.random_entry] == [
        len(r.log.signals) for r in b.random_entry
    ]
    assert [r.trades for r in a.random_entry] == [r.trades for r in b.random_entry]


def test_buy_and_hold_rejects_a_symbol_without_bars() -> None:
    with pytest.raises(ValueError, match="no decision bars.*B"):
        buy_and_hold(
            {"A": decision_closes([1.0]), "B": decision_closes([1.0]).iloc[:0]}, 100.0, FREE
        )


def test_a_symbol_without_bars_by_the_cut_stays_cash() -> None:
    early = decision_closes([100.0, 200.0, 300.0])
    late = decision_closes([10.0, 20.0, 30.0]).iloc[2:]  # first bar after the cut
    cut = early.index[1]
    cfg = load_config(
        overrides={
            "exchange.symbols": ["A", "B"],
            "backtest.initial_equity": 1000.0,
            "costs.taker_fee_bps": 0,
            "costs.half_spread_bps": 0,
            "costs.slippage_bps": 0,
        }
    )
    window = {"A": early, "B": late}
    # A: 500 buys 5 units, worth 1000 at 200; B: 500 stays cash
    assert _hold_until(window, cut, cfg).tolist() == [1000.0, 1500.0]


def test_seeds_run_only_inside_the_halted_reference_period() -> None:
    cfg, candles = walk_cfg(**{"risk.max_drawdown": 0.003})
    end = T0 + timedelta(days=3)
    reference = random_entry(cfg, candles, {}, T0, end, 7, p=0.3)
    assert first_permanent_halt(reference.log.risk_events) is not None
    cut = active_period(reference.trades, reference.equity, reference.log.risk_events)[1].index[-1]
    assert cut < reference.equity.index[-1]
    base = compute_baselines(reference, candles, {}, len(reference.log.signals), seeds=(0, 1))
    assert len(base.random_entry) == 2
    for run in base.random_entry:
        assert run.equity.index.max() <= cut
        assert all(f.ts <= cut for f in run.log.fills)
    assert base.buy_and_hold is not None and base.buy_and_hold.index[-1] == cut
