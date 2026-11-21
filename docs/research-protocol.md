# Research protocol

Rules for testing a strategy without fooling yourself. Most trading backtests that look good are wrong in one of a few known ways; each rule below closes one of them. Some rules are enforced by code, some are discipline you must follow, and this page says which.

## The rules

| Rule | What it prevents | Enforced by |
|---|---|---|
| Causality | Look-ahead: using data from after the decision | Code and tests |
| Walk-forward | Choosing parameters on the data you score them on | `backtest walk-forward` |
| One-touch holdout | Slowly tuning to the final test | Discipline only (see below) |
| Sensitivity | A result that exists only at one exact parameter value | `backtest robustness` |
| Cost stress | A result that disappears when costs are a little worse | `backtest robustness` |
| Remove the best 5% of trades | A result carried by a few lucky trades | Report |
| Baselines | Mistaking market drift or luck for skill | Report |
| Trial counting | Picking the best of many tries and calling it a discovery | `--variants-tried`, walk-forward output |

### 1. Causality

- Every timestamp is UTC and a bar's `ts` is its close time. A decision at a 15-minute close sees only data up to that close.
- Orders fill no earlier than the next 1-minute bar's open.
- The percentile features need a full 30-day window. The classifier answers `no_trade` until `warmup_bars` decision bars exist (2,976 with defaults, 31 days), so history is downloaded and loaded from 31 days before `backtest.start`, while decisions begin at `backtest.start`. A backtest without enough history records a `warmup_incomplete` event per symbol and the CLI warns.
- Longer horizons are longer windows on the 15-minute frame, not joined higher-timeframe candles (ADR 0001).
- A test (`tests/test_features.py`, prefix invariance) checks that every feature column computed on a truncated history equals the same rows computed on the full history. A feature that peeks forward fails it.
- Resampling drops incomplete buckets rather than filling them.
- The regime study uses future returns by design and is kept apart from the features.

If you add a feature, keep that test passing. A backtest that beats the real world by a lot should be checked for look-ahead first.

### 2. Walk-forward

```bash
.venv/bin/regimebot backtest walk-forward
```

For each window, the config's grid is fitted on a 365-day train span, and the combination with the highest train profit factor (among those with at least 20 trades) runs once on the next 90 days. Parameters are chosen from train data only. Each test window starts from the initial equity; windows are not chained, so the out-of-sample total is the plain sum of the windows' net P&L. The default grid has 9 combinations (breakout `volume_z_min` x `stop_atr_mult`), 4 windows, so 36 variants are tried. Output: `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv`.

Walk-forward uses the config's risk limits, including the 10% drawdown halt. A window that hits the halt stops trading early and its metrics cover only the period before it. The output reports per-window `train_halted_at` and `test_halted_at`, and the command prints how many test windows halted. Pass `--set risk.max_drawdown=1.0` for full windows. Report both runs when the halt matters, as the worked example does.

Read the train columns as well as the test columns. If the best in-training profit factor is below 1, no combination in the grid worked even on the data it was fitted to.

### 3. One-touch holdout

Keep a final stretch of history that no decision about the strategy has used, and score it once.

This is a rule you follow; the code has no holdout feature and no holdout period is configured. Practical form: run all development with `--end` before the holdout start, then run the final candidate once on the holdout (`--start`, `--end`), and report whatever comes out. If you look at it, change something and look again, it is no longer a holdout.

Be honest about the current state: the first results below used all of 2024-2026 as development data. There is no untouched holdout left in that range. Paper trading forward in time is the only clean test remaining.

### 4. Sensitivity

```bash
.venv/bin/regimebot backtest robustness
```

Each listed parameter is perturbed by +/-20% and the backtest rerun (default parameters: breakout `stop_atr_mult` and `volume_z_min`, trend `stop_atr_mult`, `risk.risk_per_trade`; override with `--param`). A real edge degrades smoothly. If neighbouring values flip a profit to a loss, the original result was an accident of that value. Output: `reports/sensitivity.csv`.

### 5. Cost stress

The same command doubles every cost (fees, spread, slippage, impact) and compares effective gross leverage 1.0 against 1.5. Outputs: `reports/cost_stress.csv`, `reports/leverage_compare.csv`. If doubling costs wipes out the result, the margin is thinner than the cost estimates are accurate. Fees dominate high-frequency results, as the first results show.

### 6. Remove the best 5%

The report (`regimebot report RUN_ID`) shows net P&L with the best 5% of trades (rounded up) removed. If it falls far or turns negative, a few trades produced the result.

### 7. Baselines

The report compares the strategy with buy-and-hold (equal notional of each symbol, same entry costs) and random entry (the real engine and risk manager with only the entry decision randomised, calibrated to similar trade frequency, a few seeds). If the strategy does not clearly beat both after costs, there is no evidence of an edge. Buy-and-hold ignores funding and volume impact, and the random-entry column rests on few seeds, so both are rough.

### 8. Trial counting

Every variant you try raises the chance that the best one looks good by luck. Count them: walk-forward prints `n_variants_tried`; for your own manual runs, pass the count to the report.

```bash
.venv/bin/regimebot report 1 --variants-tried 36
```

The report then states how many variants were tried before this one was chosen. Discard a result that needed many tries and does not survive the holdout.

## How to read a report honestly

1. Start with the risk warning, and check whether the run halted. A run that hit the 10% drawdown limit stopped trading at that date; its numbers describe only the period before it. `backtest run` prints the halt time and reason.
2. Look at net P&L, then fees. If gross P&L before fees is near zero or negative, there is nothing for cost tuning to rescue.
3. Compare with buy-and-hold and random entry, not with zero.
4. Check the breakdowns (regime, symbol, strategy, exit reason). One profitable cell out of many is what chance produces.
5. Check the result without the best 5%, and the trade count. Under about 100 trades, most metrics are noise.
6. Check modelled versus realised slippage.
7. Ask how many variants were tried.
8. Remember the limits: one history, one fill model, no market impact beyond the bounded model, no live-trading frictions.

Sharpe and similar ratios are computed on daily equity. They are context, not a verdict.

## Worked example: the first results

Data: 2024-01-01 to 2026-01-01, BTC, ETH and SOL perpetuals, 10,000 USDT. Nothing was tuned to improve these.

Data 2023-11-30 to 2026-01-01 (the first 31 days only warm up the features); trading 2024-01-01 to 2026-01-01; BTC, ETH and SOL; 10,000 USDT. Nothing was tuned to improve these numbers.

**An implementation detail moved a headline number.** An earlier version of this example, run before a fix to the feature warm-up, reported 60 trades and -9.72% for the default-limits run. The percentile features need a full 30-day window. The fix makes the classifier answer `no_trade` until that window exists and loads the 31 days of history before `backtest.start`, so the early bars of the period are classified differently. The same default-limits run now gives 76 trades and -7.16%. The conclusion did not change, but the number did. Treat any single headline number as a product of the exact code that made it, and rerun everything after changing the code.

**Default limits** (run 5). 76 trades, net -715.60 USDT (-7.16%), fees 276.89, max drawdown 10.03%, profit factor 0.66, win rate 26.32%, expectancy -0.28R. The 10% drawdown limit halted the run on 2024-02-07, so this describes five weeks, not two years. Over the same period buy and hold lost 35.53 and random entries (3 seeds, calibrated to the same trade frequency) had a median of -327.34 (range -911.82 to -315.30). The strategy is in the same range as random entries, but five weeks and three seeds say little.

**Drawdown halt off** (run 6; `--set risk.max_drawdown=1.0`, the documented way to study a full period):

```bash
.venv/bin/regimebot backtest run --set risk.max_drawdown=1.0
```

1,312 trades, net -4,546.86 USDT (-45.47%), profit factor 0.83, win rate 31.86%, expectancy -0.16R, max drawdown 48.64%, max gross leverage seen 1.02x. Fees were 4,031.04 and funding 58.07, so before fees the loss was about 516 (about 458 before fees and funding). Buy and hold over the two years: +5,302.82 (+53.03%, max drawdown 50.52%). Random entries with the same exits: median -6,433.63, range -6,883.03 to -4,263.12, profit factor 0.67. Reading: the strategies lose a little before costs and a lot after. Trading 1,312 times is expensive, and cost tuning cannot fix a gross result that is not positive. The entry rules do somewhat better than random entries with the same exits (though one of the three random seeds lost less than the strategy), nowhere near enough to cover costs, and the strategy is far behind buy and hold.

**Robustness** (two years, drawdown halt off). Moving each of four parameters by +/-20% keeps the net result between -4,278.77 and -5,207.16 (profit factor 0.81 to 0.85), so the loss is not an artefact of one parameter value. Doubling all costs gives -6,976.88 (profit factor 0.69). Effective leverage 1.0x versus 1.5x gives -4,546.86 (max drawdown 48.64%, max gross seen 1.02x) versus -5,122.05 (max drawdown 53.19%, max gross seen 1.52x): more leverage made the loss larger. With the default config every symbol is in one group capped at 1.0x, so the configured 1.5x gross cap is unreachable; the comparison raises the group cap together with the gross cap, and reports print both the configured and the effective cap. An earlier sentence in these docs said the cap never binds at 0.5% risk per trade. That was wrong: in the two-year run 243 of 1,312 approved entries were reduced by the asset or group cap.

**Walk-forward, default limits** (drawdown halt on; 9 combinations, 4 windows, 36 variants tried). The selected training run in every window hit the halt (2024-04-07, 2024-07-02, 2024-08-12, 2024-12-03), so training saw only part of each year. The best in-training profit factors were 0.94, 0.93, 0.54 and 0.80, on 180, 184, 82 and 106 trades up to the halt. Out-of-sample: 463 trades, net -770.05 USDT; window nets +1,438.60, -418.84, -858.99, -930.82. The last three test windows (CSV `window` 1 to 3) halted on 2025-04-28, 2025-09-08 and 2025-12-10, so they are short; the first, winning, window ran in full. Reading: windows that stop early cannot support a claim about a full year in either direction.

**Walk-forward, drawdown halt off** (`--set risk.max_drawdown=1.0`, full windows; same grid, 36 variants):

```bash
.venv/bin/regimebot backtest walk-forward --set risk.max_drawdown=1.0
```

Best in-training profit factors 0.84, 0.96, 0.87 and 0.91 on 677, 647, 627 and 623 trades; training nets -2,434.49, -513.23, -1,827.04 and -1,289.81. Out-of-sample: 636 trades, net -2,234.05 USDT; test profit factors 1.36, 0.65, 0.86, 0.59; window nets +1,329.28, -1,557.25, -508.65, -1,497.43. One quarter won and three lost. Reading: the best parameter set in the grid lost money in training in every window, and the winning quarter came from parameters that had lost in training, so it is not evidence of an edge.

**Regime study** (`regimebot regimes study`, fully warmed classifier). A 5% move within 24 hours followed 22% / 37% / 60% of high-volatility-expansion bars on BTC / ETH / SOL, against 6% / 16% / 30% after compression. The mean 24-hour return after a "trending up" bar on BTC was +0.27% against a standard deviation of 2.2%. Regimes tell you how big the next move may be, not which way it goes. The breakout and trend strategies need the direction.

**Artefacts.** `reports/run_5/report.html` (default limits), `reports/run_6/report.html` (halt off), `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv` (halt off, the last run), `reports/walk_forward_windows_default_limits.csv` and `reports/walk_forward_oos_trades_default_limits.csv` (copies of the default-limits run, made by hand), `reports/sensitivity.csv`, `reports/cost_stress.csv`, `reports/leverage_compare.csv`, `reports/regime_study_*.csv`. The `reports/` directory is git-ignored; the commands regenerate it. Runs 1 to 3 in the development database predate the warm-up fix and are superseded.

**Conclusion.** As specified, these strategies show no edge on this data. The system's job is to say so, and it does. Possible next steps, each of which would count as new trials: a strategy that uses regime for sizing or volatility rather than direction, lower trade frequency, or different data. None of these is a promise of an edge.
