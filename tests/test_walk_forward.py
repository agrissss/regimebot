"""Walk-forward bookkeeping: windows, train-only selection, no-qualifier path, variant count.

Most tests replace ``run_backtest`` with a fake whose trades depend on (parameter, span), so the
selection logic is checked exactly; one test drives the real engine end to end.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from synth import BTC, T0, Scripted, regime_walk

from regimebot.backtest import walk_forward as wf
from regimebot.backtest.runner import BacktestResult
from regimebot.config import AppConfig, load_config, with_overrides
from regimebot.engine.loop import RunLog
from regimebot.strategy import Strategy
from regimebot.types import ExitReason, Regime, Side, Trade

DAY = timedelta(days=1)
GRID_KEY = "risk.risk_per_trade"


def cfg_with(
    grid: dict[str, list[float]], train: int = 2, test: int = 1, min_trades: int = 2
) -> AppConfig:
    return load_config(
        overrides={
            "exchange.symbols": [BTC],
            "walk_forward.train_days": train,
            "walk_forward.test_days": test,
            "walk_forward.min_trades": min_trades,
            "walk_forward.grid": grid,
        }
    )


def trade(ts: datetime, pnl: float) -> Trade:
    return Trade(BTC, "s", Side.LONG, Regime.RANGE, 1.0, ts, ts + timedelta(hours=1), 100.0, 100.0,
                 98.0, 104.0, pnl, 0.0, 0.0, pnl, ExitReason.TARGET, pnl / 2, 1)  # fmt: skip


class FakeRun:
    """``run_backtest`` stand-in: ``script(risk_per_trade, start)`` gives the trades' pnls."""

    def __init__(self, script: Callable[[float, datetime], list[float]]) -> None:
        self.script = script
        self.calls: list[tuple[float, datetime, datetime]] = []

    def __call__(
        self,
        cfg: AppConfig,
        candles: Mapping[str, pd.DataFrame],
        funding: Mapping[str, pd.Series],
        start: datetime,
        end: datetime,
        kill_file: Path,
        strategies: Sequence[Strategy] | None = None,
    ) -> BacktestResult:
        assert not Path(kill_file).exists()
        self.calls.append((cfg.risk.risk_per_trade, start, end))
        pnls = self.script(cfg.risk.risk_per_trade, start)
        trades = [trade(start + timedelta(minutes=i), p) for i, p in enumerate(pnls)]
        equity = pd.Series([cfg.backtest.initial_equity], index=pd.DatetimeIndex([start], tz="UTC"))
        return BacktestResult(RunLog(), trades, equity, cfg, start, end)


def test_windows_have_exact_boundaries_and_tests_never_touch_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRun(lambda p, s: [1.0, 1.0])
    monkeypatch.setattr(wf, "run_backtest", fake)
    cfg = cfg_with({GRID_KEY: [0.004, 0.005]})
    res = wf.walk_forward(cfg, {}, {}, T0, T0 + 6 * DAY)
    w = res.windows
    assert len(w) == 4  # window 3 ends exactly on day 6; a 5th would end on day 7
    for i, r in w.iterrows():
        assert r.train_start == T0 + i * DAY and r.train_end == T0 + (i + 2) * DAY
        assert r.test_start == r.train_end and r.test_end == r.test_start + DAY
    assert w.test_end.iloc[-1] == T0 + 6 * DAY
    # per window: 2 train runs [train_start, train_end), then 1 test run [train_end, test_end)
    for i in range(4):
        *train_calls, test_call = fake.calls[3 * i : 3 * i + 3]
        assert all((s, e) == (T0 + i * DAY, T0 + (i + 2) * DAY) for _, s, e in train_calls)
        assert (test_call[1], test_call[2]) == (T0 + (i + 2) * DAY, T0 + (i + 3) * DAY)


def test_last_window_may_end_exactly_at_end_and_none_fitting_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "run_backtest", FakeRun(lambda p, s: [1.0, 1.0]))
    cfg = cfg_with({GRID_KEY: [0.005]})
    assert len(wf.walk_forward(cfg, {}, {}, T0, T0 + 3 * DAY).windows) == 1
    with pytest.raises(ValueError, match="no walk-forward window"):
        wf.walk_forward(cfg, {}, {}, T0, T0 + 3 * DAY - timedelta(minutes=1))


def test_params_are_argmax_on_train_not_on_test(monkeypatch: pytest.MonkeyPatch) -> None:
    train_end = T0 + 2 * DAY

    def script(p: float, start: datetime) -> list[float]:
        # 0.004 is better on train (PF 4), 0.005 is better on test (PF inf): a leaking
        # selector would pick 0.005.
        if start < train_end:
            return [4.0, 4.0, -2.0] if p == 0.004 else [2.0, -2.0, 1.0]
        return [-1.0, 1.0] if p == 0.004 else [5.0, 5.0]

    fake = FakeRun(script)
    monkeypatch.setattr(wf, "run_backtest", fake)
    res = wf.walk_forward(cfg_with({GRID_KEY: [0.005, 0.004]}), {}, {}, T0, T0 + 3 * DAY)
    (row,) = res.windows.to_dict("records")
    assert json.loads(row["params"]) == {GRID_KEY: 0.004}
    assert not row["no_qualifying_params"]
    assert row["train_profit_factor"] == pytest.approx(4.0)
    assert (row["test_n_trades"], row["test_net_pnl"]) == (2, 0.0)  # the 0.004 test run
    assert fake.calls[-1][0] == 0.004


def test_selection_metric_is_profit_factor_not_net_pnl(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.005: PF 4, net 15; 0.004: PF inf, net 6. A net-pnl selector would pick 0.005.
    fake = FakeRun(lambda p, s: [10.0, 10.0, -5.0] if p == 0.005 else [3.0, 3.0])
    monkeypatch.setattr(wf, "run_backtest", fake)
    res = wf.walk_forward(cfg_with({GRID_KEY: [0.005, 0.004]}), {}, {}, T0, T0 + 3 * DAY)
    assert json.loads(res.windows.params[0]) == {GRID_KEY: 0.004}
    assert res.windows.train_net_pnl[0] == 6.0


def test_nan_profit_factor_does_not_qualify_even_with_enough_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "run_backtest", FakeRun(lambda p, s: [0.0, 0.0, 0.0]))  # PF is NaN
    res = wf.walk_forward(cfg_with({GRID_KEY: [0.004]}), {}, {}, T0, T0 + 3 * DAY)
    assert res.windows.no_qualifying_params[0] and res.windows.train_n_trades[0] == 3


def test_invalid_grid_combination_fails_before_any_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRun(lambda p, s: [1.0, 1.0])
    monkeypatch.setattr(wf, "run_backtest", fake)
    grid = {GRID_KEY: [0.004], "risk.max_gross_leverage": [1.0, 2.0]}  # 2.0 exceeds the hard cap
    with pytest.raises(ValueError):
        wf.walk_forward(cfg_with(grid), {}, {}, T0, T0 + 3 * DAY)
    assert fake.calls == []


def test_ties_go_to_first_in_grid_order_and_inf_beats_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "run_backtest", FakeRun(lambda p, s: [1.0, -1.0]))
    res = wf.walk_forward(cfg_with({GRID_KEY: [0.006, 0.004, 0.005]}), {}, {}, T0, T0 + 3 * DAY)
    assert json.loads(res.windows.params[0]) == {GRID_KEY: 0.006}
    monkeypatch.setattr(
        wf, "run_backtest", FakeRun(lambda p, s: [3.0, 3.0] if p == 0.005 else [3.0, -1.0])
    )
    res = wf.walk_forward(cfg_with({GRID_KEY: [0.006, 0.004, 0.005]}), {}, {}, T0, T0 + 3 * DAY)
    assert json.loads(res.windows.params[0]) == {GRID_KEY: 0.005}


def test_no_qualifying_params_runs_test_on_unmodified_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # every combination has too few trades (1 < min_trades 2); one has NaN profit factor (no trades)
    fake = FakeRun(lambda p, s: [] if p == 0.004 else [5.0])
    monkeypatch.setattr(wf, "run_backtest", fake)
    cfg = cfg_with({GRID_KEY: [0.004, 0.006]})
    res = wf.walk_forward(cfg, {}, {}, T0, T0 + 3 * DAY)
    (row,) = res.windows.to_dict("records")
    assert row["no_qualifying_params"] and json.loads(row["params"]) == {}
    assert fake.calls[-1][0] == cfg.risk.risk_per_trade  # the test run used the base value
    # 2 grid train runs + 1 unmodified-config train run (reported) + 1 test run
    assert len(fake.calls) == 4
    assert (row["train_n_trades"], row["train_net_pnl"]) == (1, 5.0)


def test_n_variants_tried_is_combinations_times_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun(lambda p, s: [1.0, 1.0])
    monkeypatch.setattr(wf, "run_backtest", fake)
    grid = {GRID_KEY: [0.004, 0.005], "risk.max_gross_leverage": [1.0, 1.5]}
    res = wf.walk_forward(cfg_with(grid), {}, {}, T0, T0 + 5 * DAY)
    assert len(res.windows) == 3 and res.n_variants_tried == 12
    assert len(fake.calls) == 12 + 3  # train runs + one test run per window


def test_oos_trades_are_the_test_trades_in_order_and_pnl_sums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "run_backtest", FakeRun(lambda p, s: [2.0, 3.0]))
    res = wf.walk_forward(cfg_with({GRID_KEY: [0.005]}), {}, {}, T0, T0 + 5 * DAY)
    starts = [t.entry_ts for t in res.oos_trades]
    assert starts == sorted(starts) and len(starts) == 6
    assert min(starts) >= T0 + 2 * DAY
    assert res.oos_net_pnl == 15.0 == res.windows.test_net_pnl.sum()


def test_real_engine_end_to_end(tmp_path: Path) -> None:
    candles = {BTC: regime_walk(5 * 1440, seed=3)}
    cfg = with_overrides(
        load_config(),
        {
            "exchange.symbols": [BTC],
            "walk_forward.train_days": 2,
            "walk_forward.test_days": 1,
            "walk_forward.min_trades": 1,
            "walk_forward.grid": {"costs.slippage_bps": [1.0, 4.0]},
        },
    )
    cfg = with_overrides(
        cfg,
        {
            f"playbook.{r.value}": {
                "strategies": ["breakout"],
                "size_mult": 1.0,
                "allow_entries": True,
                "force_close": False,
            }
            for r in Regime
        },
    )
    stub = [Scripted("breakout", stop_dist=0.3, target_dist=0.6)]
    end = datetime(2024, 1, 6, tzinfo=UTC)
    res = wf.walk_forward(cfg, candles, {}, T0, end, strategies=stub)
    assert len(res.windows) == 3 and res.n_variants_tried == 6
    # same signals, only costs differ: the cheaper slippage has the higher PF on every train span
    assert all(json.loads(p) == {"costs.slippage_bps": 1.0} for p in res.windows.params)
    assert all(t.entry_ts >= T0 + 2 * DAY for t in res.oos_trades)
