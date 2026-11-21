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

The same command doubles every cost (fees, spread, slippage, impact) and compares gross leverage 1.0 against 1.5. Outputs: `reports/cost_stress.csv`, `reports/leverage_compare.csv`. If doubling costs wipes out the result, the margin is thinner than the cost estimates are accurate. Fees dominate high-frequency results, as the first results show.

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

**Default limits.** 60 trades, net -971.71 USDT (-9.72%). The 10% drawdown limit halted the run on 2024-02-05, so this describes five weeks, not two years.

Over the same five weeks, buy and hold lost 51.49 and random entries (3 seeds, calibrated to the same trade frequency) had a median of -359.25 (range -982.29 to +103.33). The strategy's -971.71 is at the bad end of the random range, but five weeks and three seeds say little.

**Drawdown halt off** (`--set risk.max_drawdown=1.0`, the documented way to study a full period):

```bash
.venv/bin/regimebot backtest run --set risk.max_drawdown=1.0
```

1,300 trades, net -4,726.92 USDT (-47.27%), profit factor 0.82, max drawdown 48.62%. Buy and hold over the two years: +5,213.78. Random entries with the same exits: median -6,107.72, range -6,471.48 to -4,814.97 (profit factor 0.69). Fees were 3,835.64, so before fees the loss was roughly 840. Reading: the strategies lose a little before costs and a lot after. Trading 1,300 times is expensive, and cost tuning cannot fix a gross result that is not positive. The entry rules do slightly better than random entries with the same exits, but nowhere near enough to cover costs, and the strategy is far behind buy and hold.

**Robustness** (two years, drawdown halt off). Moving each of four parameters by +/-20% keeps the net result between -4,279.65 and -5,302.76, so the loss is not an artefact of one parameter value. Doubling all costs gives -6,932.21. A 1.0x versus 1.5x gross leverage cap gives identical results: the cap never binds at 0.5% risk per trade.

**Walk-forward** (9 combinations, 4 windows, 36 variants tried). The best in-training profit factor was below 1 in every window: 0.95, 0.93, 0.54, 0.80. Out-of-sample: -554.29 USDT over 465 trades, one winning quarter and three losing. Reading: the grid contains no setting that worked even in-sample, and the one winning quarter is what a random process produces among four. The first window's out-of-sample profit (about +1,650) was chosen from parameters whose training profit factor was 0.95, which is a reason to distrust it, not to build on it.

**Regime study** (`regimebot regimes study`). A 5% move within 24 hours followed 23% / 37% / 59% of high-volatility-expansion bars on BTC / ETH / SOL, against 6% / 16% / 29% after compression. Regimes tell you how big the next move may be, not which way it goes. The breakout and trend strategies need the direction.

**Artefacts.** `reports/run_2/report.html`, `reports/run_3/report.html`, `reports/walk_forward_windows.csv`, `reports/walk_forward_oos_trades.csv`, `reports/sensitivity.csv`, `reports/cost_stress.csv`, `reports/leverage_compare.csv`, `reports/regime_study_*.csv`. The `reports/` directory is git-ignored; the commands regenerate it.

**Conclusion.** As specified, these strategies show no edge on this data. The system's job is to say so, and it does. Possible next steps, each of which would count as new trials: a strategy that uses regime for sizing or volatility rather than direction, lower trade frequency, or different data. None of these is a promise of an edge.
