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

Be honest about the current state: the first results below used 2024-01-01 to 2026-01-01 as development data, so that range is not a holdout. The backtests end at 2026-01-01, so 2026-01-01 onwards has never been used for any parameter or design choice and is a clean holdout. It is also a falling market, unlike the 2024-2025 test period. That holdout was then run once against the criteria below; the result follows them. Paper trading forward in time is a further clean test.

#### Pre-registered criteria for the 2026 holdout

Written before the run, dated 2026-11-21.

- Configuration: `config/default.yaml` at the commit that runs it. No parameter changes.
- The strategy passes only if, with the drawdown halt off, all of these hold:
  - net P&L after all costs is above 0;
  - profit factor is at least 1.15;
  - there are at least 100 trades;
  - net P&L is above the median band of the 30 random-entry seeds, meaning the strategy beats at least 27 of the 30 seeds;
  - maximum drawdown is at most 20%.
- Whether the result stays positive with costs doubled is not required, but it will be reported.
- Anything else is a fail, and a fail closes the question for this rule set as specified.
- One secondary check, stated in advance: breakout alone has net P&L above 0 on the holdout. It tests the per-strategy observation in the README (breakout was the least bad strategy, positive before fees) and is reported as exploratory whatever the outcome.

#### Result of the 2026 holdout (run once, 2026-11-21): FAIL

Frozen `config/default.yaml`, 2026-01-01 to 2026-11-21, 10,000 USDT, 30 random-entry seeds. Runs 7 (default limits) and 8 (drawdown halt off).

| Criterion (halt off, run 8) | Required | Result | |
|---|---|---|---|
| Net P&L after all costs | above 0 | -3,983.88 (-39.84%) | fail |
| Profit factor | at least 1.15 | 0.60 | fail |
| Trades | at least 100 | 503 | pass |
| Random-entry seeds beaten | at least 27 of 30 | 2 of 30 (random median -2,693.29, range -4,534.47 to -1,010.98) | fail |
| Maximum drawdown | at most 20% | 45.04% | fail |

Buy and hold over the same period: -4,097.71 (max drawdown 51.13%). Fees 1,559.12, funding 23.34. Expectancy -0.31 R per trade (95% bootstrap interval -0.43 to -0.17). Costs x0.5: -3,110.72; costs x2: -5,247.55. By strategy: breakout 74 trades, -344.04; mean reversion 141, -1,429.66; trend 288, -2,210.19. By side: longs -2,312.09, shorts -1,671.79.

Secondary check (exploratory): breakout alone net P&L above 0. Result -344.04: not met. The after-the-fact observation that breakout was the least bad strategy did not turn into a profit out of sample.

With the default limits (run 7) the drawdown halt stopped trading on 2026-03-16 after 124 trades at -369.37 (-3.69%, profit factor 0.88, max drawdown 10.16%); buy and hold lost 2,230.69 over those weeks, and the strategy beat 22 of 30 random seeds over that shortened period. The limits did what they are for: they capped the loss.

Per the criteria above, this closes the question for this rule set as specified: it has no edge after costs, in a rising market (2024-2025) or a falling one (2026). The planned 2021-2023 run is therefore not needed to decide and has not been run. Any new idea is a new rule set and needs its own development data and its own untouched holdout; 2026 can no longer serve as one.

Commands:

```bash
.venv/bin/regimebot data download --end 2026-11-21T00:00:00+00:00
.venv/bin/regimebot backtest run --start 2026-01-01T00:00:00+00:00 --end 2026-11-21T00:00:00+00:00
.venv/bin/regimebot backtest run --start 2026-01-01T00:00:00+00:00 --end 2026-11-21T00:00:00+00:00 --set risk.max_drawdown=1.0
.venv/bin/regimebot report RUN_ID --random-seeds 30
```

#### Second rule set: 1-hour breakout only (search on 2024-2026, 2026-11-22)

After the fail above, 16 variants were tried on the already-used 2024-01-01 to 2026-11-21 data (drawdown halt off): each strategy alone and in combinations at 15 minutes, and breakout/trend combinations at 1-hour and 4-hour decisions. Every 15-minute variant lost. The best was **breakout only at a 1-hour decision timeframe** (`exchange.decision_timeframe=1h`, `features.pctile_window=720`, trend and mean reversion disabled, everything else default): 345 trades, net +2,466.97, profit factor 1.22, max drawdown 12.16%, positive in 2024 (+193), 2025 (+1,795) and 2026 (+479), ahead of 29 of 30 random-entry seeds. Parameters moved by +/-20% gave +2,062 to +2,715; costs x2 gave +474. Weak points: expectancy 0.18 R with a 95% bootstrap interval of -0.02 to 0.40 (zero is inside it), shorts made nothing (+1.66), and without the best 18 trades the result is -2,402. It is the best of 16 tries on used data, so by itself it proves nothing.

##### Pre-registered criteria for 2021-2023 (written before the data was downloaded, 2026-11-22)

- Data: 2021-01-01 to 2023-11-30, never downloaded or examined before this commit. Configuration frozen as above; no changes after seeing any result.
- Pass only if, with the drawdown halt off, ALL hold: net P&L after all costs above 0; profit factor at least 1.10; at least 150 trades; ahead of at least 24 of 30 random-entry seeds; maximum drawdown at most 25%; net P&L positive in at least two of the three calendar years.
- The run with default limits (10% drawdown halt) is reported alongside but does not decide.
- A fail means the 2024-2026 result was selection luck. A pass means the rule set has survived one honest test; it is then a candidate for paper trading forward, not proof of future profit.

##### Result on 2021-2023 (run once, 2026-11-22): FAIL

Runs 23 (halt off) and 24 (default limits), 2021-01-01 to 2023-11-30, 30 random-entry seeds.

| Criterion (halt off, run 23) | Required | Result | |
|---|---|---|---|
| Net P&L after all costs | above 0 | -912.38 (-9.12%) | fail |
| Profit factor | at least 1.10 | 0.91 | fail |
| Trades | at least 150 | 329 | pass |
| Random-entry seeds beaten | at least 24 of 30 | 19 of 30 (random median -1,151.10) | fail |
| Maximum drawdown | at most 25% | 16.25% | pass |
| Calendar years with positive net P&L | at least 2 of 3 | 2 (2021 +101, 2022 -1,114, 2023 +101) | pass |

Fees 860.70, funding 198.58. Expectancy -0.06 R per trade (95% bootstrap interval -0.22 to 0.12). With default limits (run 24) the drawdown halt stopped trading at -606.64 (-6.07%) after 208 trades.

Conclusion: the +2,467 on 2024-2026 was selection luck, as the criteria said a fail would mean. Across all six years the 1-hour breakout rule set is roughly flat before its drawdowns (+2,467 and -912) and is not distinguishable from random entries with the same risk management. No rule set built from this brief's ingredients (price-and-volume regimes, breakout, trend, mean reversion) has shown an edge after costs. All data from 2021 to 2026-11-21 has now been examined; the only clean test left for any further idea is paper trading forward.

#### Third rule set: daily trend following (2026-11-22)

A different kind of strategy from the brief's: one decision per coin per day, no regimes, no fixed 5% target. Prototype: `research/daily_trend.py` (vectorised, outside the engine). Rule: at each 00:00 UTC close the signal is the average of sign(close - SMA) over 20, 50 and 100 days, between -1 and 1; position size is the signal times an equal-risk volatility target (15% annualised), capped at 0.75x per coin and 1.5x gross; costs 8 bps per unit of turnover plus actual funding. The reasons to expect an edge are known ones (trend persistence in crypto is documented; costs are negligible at about 25x turnover a year), not a pattern found in this data.

Development on BTC, ETH and SOL, 2021-01-01 to 2026-11-21 (data already examined): 24 single-lookback variants (SMA, momentum, Donchian; long/flat, long/short, funding filter) were all profitable, Sharpe 0.2 to 1.1. The combined long/short rule above: +129.90%, Sharpe 0.79, max drawdown -28.54%, positive every year (2021 +27.4%, 2022 +2.2%, 2023 +11.6%, 2024 +6.0%, 2025 +4.5%, 2026 +42.8%), Sharpe above 197 of 200 runs with randomly time-shifted positions. Long/flat: +119.74%, Sharpe 1.03, max drawdown -22.28%, one losing year (2022, -13.8%). Buy and hold: +894% with an 85% drawdown.

##### Pre-registered criteria for ten other coins (written before their data was fetched, 2026-11-22)

- Universe, fixed here: BNB, XRP, DOGE, ADA, AVAX, LINK, LTC, DOT, BCH, TRX (Binance USDT perpetuals listed before 2021; none ever downloaded in this project). Same script, same parameters, long/short, 2021-01-01 to 2026-11-21. One run.
- Pass only if ALL hold: total return above 0; Sharpe at least 0.40; positive in at least 4 of the 6 calendar years; max drawdown no worse than -40%; at least 6 of the 10 coins contribute positively; Sharpe above at least 190 of 200 time-shifted runs; total return still above 0 with costs doubled.
- Long/flat is reported alongside but does not decide.
- Limits of this test, stated in advance: these coins move with BTC, so this is a test that the rule is not fitted to three price series, not a fully independent sample; it is a daily-close simulation, not the engine; a pass makes the rule a candidate to build into the engine and paper trade, nothing more.

##### Result on the ten other coins (run once, 2026-11-22): FAIL on two of seven criteria

| Criterion (long/short) | Required | Result | |
|---|---|---|---|
| Total return | above 0 | +112.22% | pass |
| Sharpe | at least 0.40 | 0.56 | pass |
| Positive calendar years | at least 4 of 6 | 3 of 6 (2021 +64.4%, 2022 -11.3%, 2023 -3.5%, 2024 +43.4%, 2025 -21.4%, 2026 +33.8%) | fail |
| Max drawdown | no worse than -40% | -46.86% | fail |
| Coins contributing positively | at least 6 of 10 | 8 of 10 | pass |
| Sharpe above time-shifted runs | at least 190 of 200 | 198 of 200 | pass |
| Total return with costs doubled | above 0 | +70.01% | pass |

Long/flat on the same coins (reported, not deciding): +227.68%, Sharpe 1.05, max drawdown -28.20%, positive in 4 of 6 years. Buy and hold: +438% with a 79% drawdown.

Reading it honestly: by the criteria written in advance this is a fail, and it is recorded as one. What the test does show is that the timing of the rule carries information on coins it was not developed on (198 of 200, profitable at double costs, 8 of 10 coins), which none of the brief's rule sets managed. What it also shows is that the ride is rough: half the years lose and the drawdown was 47% at this sizing. Part of that drawdown is a sizing error in the prototype (the volatility target assumes uncorrelated coins, and these ten move together), but the losing years are not. This is the normal profile of trend following, and it is not compatible with a 10% drawdown limit unless positions, and returns, are cut to roughly a quarter. All daily data for these 13 coins has now been examined; the remaining clean test is paper trading forward.

##### Daily trend in the engine (2026-11-22)

The rule was then built into the engine as `strategies.daily_trend` (`config/trend.yaml`), with sizing changed to `target_vol / n` per coin (the prototype's `/ sqrt(n)` assumed uncorrelated coins) and `target_vol` 0.08, chosen so the historical drawdown stays under the 10% limit. Through the engine's fill model on 1-minute data, BTC, ETH and SOL, 2021-01-01 to 2026-11-21: default limits (run 26) +2,890.68 (+28.91%), max drawdown 9.22%, 440 trades, all six years positive, ahead of 30 of 30 time-shifted runs on net P&L and 29 of 30 on Sharpe; all limits off (run 25) +29.38%, max drawdown 9.15%. The prototype at the same sizing gives +32.6% and -9.35%, so the engine reproduces it; the small gap is the later start (100 daily bars from 2020-11-01), the rebalance band and the fill model. This is in-sample for these three coins. The remaining clean test is paper trading forward.

### 4. Sensitivity

```bash
.venv/bin/regimebot backtest robustness
```

Each listed parameter is perturbed by +/-20% and the backtest rerun (default parameters: breakout `stop_atr_mult` and `volume_z_min`, trend `stop_atr_mult`, `risk.risk_per_trade`; override with `--param`). A real edge degrades smoothly. If neighbouring values flip a profit to a loss, the original result was an accident of that value. Output: `reports/sensitivity.csv`.

### 5. Cost stress

The same command scales every cost (x0.5 and x2 against the config as x1) (fees, spread, slippage, impact) and compares effective gross leverage 1.0 against 1.5. Outputs: `reports/cost_stress.csv`, `reports/leverage_compare.csv`. If doubling costs wipes out the result, the margin is thinner than the cost estimates are accurate. Fees dominate high-frequency results, as the first results show. The command runs costs at x0.5, x1 and x2 (`cost_stress.csv` has a `cost_mult` column).

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
2. Look at net P&L, then total modelled costs (fees, funding, spread and slippage). If gross P&L before all costs is near zero or negative, there is nothing for cost tuning to rescue.
3. Compare with buy-and-hold and random entry, not with zero.
4. Check the breakdowns (regime, symbol, strategy, side, exit reason), the share of trades that met the minimum reward-to-risk at the actual fill, and the largest market moves. The report also gives expectancy with a bootstrap interval and states how many random seeds the strategy beat (`--random-seeds`, up to 50). One profitable cell out of many is what chance produces.
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

1,312 trades, net -4,546.86 USDT (-45.47%), profit factor 0.83, win rate 31.86%, expectancy -0.16R, max drawdown 48.64%, max gross leverage seen 1.02x. Fees were 4,031.04 and funding 58.07, and spread and slippage inside the fill prices were about 2,374, so total modelled costs were about 6,460. The earlier figure of about -516 before fees still included the spread and slippage; before all modelled costs the rules made a small positive gross result, roughly +1,900 over two years, several times smaller than the costs. Buy and hold over the two years: +5,302.82 (+53.03%, max drawdown 50.52%). Random entries with the same exits: median -6,433.63, range -6,883.03 to -4,263.12, profit factor 0.67. Reading: the rules make a small gross profit before costs and lose a lot after them. Trading 1,312 times is expensive, and the costs are several times the gross result, so there is no edge after costs as specified. The entry rules do somewhat better than random entries with the same exits (though one of the three random seeds lost less than the strategy), nowhere near enough to cover costs, and the strategy is far behind buy and hold.

**Robustness** (two years, drawdown halt off). Moving each of four parameters by +/-20% keeps the net result between -4,278.77 and -5,207.16 (profit factor 0.81 to 0.85), so the loss is not an artefact of one parameter value. Doubling all costs gives -6,976.88 (profit factor 0.69). Effective leverage 1.0x versus 1.5x gives -4,546.86 (max drawdown 48.64%, max gross seen 1.02x) versus -5,122.05 (max drawdown 53.19%, max gross seen 1.52x): more leverage made the loss larger. With the default config every symbol is in one group capped at 1.0x, so the configured 1.5x gross cap is unreachable; the comparison raises the group cap together with the gross cap, and reports print both the configured and the effective cap. An earlier sentence in these docs said the cap never binds at 0.5% risk per trade. That was wrong: in the two-year run 243 of 1,312 approved entries were reduced by the asset or group cap.

**Walk-forward, default limits** (drawdown halt on; 9 combinations, 4 windows, 36 variants tried). The selected training run in every window hit the halt (2024-04-07, 2024-07-02, 2024-08-12, 2024-12-03), so training saw only part of each year. The best in-training profit factors were 0.94, 0.93, 0.54 and 0.80, on 180, 184, 82 and 106 trades up to the halt. Out-of-sample: 463 trades, net -770.05 USDT; window nets +1,438.60, -418.84, -858.99, -930.82. The last three test windows (CSV `window` 1 to 3) halted on 2025-04-28, 2025-09-08 and 2025-12-10, so they are short; the first, winning, window ran in full. Reading: windows that stop early cannot support a claim about a full year in either direction.

**Walk-forward, drawdown halt off** (`--set risk.max_drawdown=1.0`, full windows; same grid, 36 variants):

```bash
.venv/bin/regimebot backtest walk-forward --set risk.max_drawdown=1.0
```

Best in-training profit factors 0.84, 0.96, 0.87 and 0.91 on 677, 647, 627 and 623 trades; training nets -2,434.49, -513.23, -1,827.04 and -1,289.81. Out-of-sample: 636 trades, net -2,234.05 USDT; test profit factors 1.36, 0.65, 0.86, 0.59; window nets +1,329.28, -1,557.25, -508.65, -1,497.43. One quarter won and three lost. Reading: the best parameter set in the grid lost money in training in every window, and the winning quarter came from parameters that had lost in training, so it is not evidence of an edge.

**After-the-fact slices** (halt-off run; hypotheses, not findings). Mean reversion: 292 trades, net -2,404 (before fees -1,464). Trend: 666 trades, net -1,678 (before fees -62). Breakout: 354 trades, net -465 (before fees +1,069). By side: longs -4,196, shorts -351. These were looked at after the result was known; the breakout observation is tested once, in advance, in the holdout criteria above. Cost model: the modelled 3 bps per side of spread plus slippage is pessimistic for a small account on these markets; at about 0.5 bps the two-year halt-off loss would be roughly -2,600 instead of -4,547, still a loss.

**Regime study** (`regimebot regimes study`, fully warmed classifier). A 5% move within 24 hours followed 22% / 37% / 60% of high-volatility-expansion bars on BTC / ETH / SOL, against 6% / 16% / 30% after compression. The mean 24-hour return after a "trending up" bar on BTC was +0.27% against a standard deviation of 2.2%. Regimes tell you how big the next move may be, not which way it goes. The breakout and trend strategies need the direction.

**Artefacts.** `reports/run_5/report.html` (default limits), `reports/run_6/report.html` (halt off), `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv` (halt off, the last run), `reports/walk_forward_windows_default_limits.csv` and `reports/walk_forward_oos_trades_default_limits.csv` (copies of the default-limits run, made by hand), `reports/sensitivity.csv`, `reports/cost_stress.csv`, `reports/leverage_compare.csv`, `reports/regime_study_*.csv`. The `reports/` directory is git-ignored; the commands regenerate it. Runs 1 to 3 in the development database predate the warm-up fix and are superseded.

**Conclusion.** As specified, these strategies show no edge on this data. The system's job is to say so, and it does. Possible next steps, each of which would count as new trials: a strategy that uses regime for sizing or volatility rather than direction, lower trade frequency, or different data. None of these is a promise of an edge.

## Funding carry check (2026-11-22)

Question: can a market-neutral funding carry (hold spot, short the same notional of the perpetual) add a stream of small, frequent gains next to the daily trend strategy? Prototype: `research/funding_carry.py`. One rule, chosen before looking and not tuned: per coin, hold the carry while the trailing 7-day funding sum is positive. Cost 21 bps of notional per entry or exit; return on capital is half the funding because half the capital is perp margin. Basis moves are ignored. Data: BTC, ETH, SOL, 2021-01-01 to 2026-11-21, already used for other work, so this is in-sample.

```bash
.venv/bin/python research/funding_carry.py BTC,ETH,SOL
```

| | Always on, no trading costs | 7-day rule, with costs | 7-day rule, 2x costs |
|---|---|---|---|
| Total | +25.1% | +28.10% | +19.44% |
| Max drawdown | -6.9% | -1.03% | -3.50% |
| 2021 / 2022 / 2023 | +17.5% / -5.3% / +2.8% | +17.6% / -0.3% / +3.0% | +16.7% / -2.1% / +2.1% |
| 2024 / 2025 / 2026 | +6.7% / +1.8% / +0.8% | +6.1% / +0.3% / -0.3% | +5.6% / -1.3% / -1.8% |
| Positive months | 83% of 71 | 75% of 71 | 56% of 71 |

Reading: the carry is real but almost all of it was earned in 2021 and 2024. At 2025-2026 funding levels it earns about 1% a year always-on and about zero with the on/off rule after costs. It is not worth building a spot leg into the engine at these levels. It was not run on the ten other coins, because the result on the first three already answers the question. Revisit if funding stays high for weeks (as a guide, 2024 averaged about 13% a year on notional).

## 4-hour trend check (2026-11-22)

Question: does the daily trend rule work on 4-hour bars, to give results on a shorter horizon? Prototype: `research/trend_4h.py` (same rule, lookbacks of 20, 50 and 100 four-hour bars, volatility from 30 days of 4-hour returns, same sizing and 8 bps cost as `research/daily_trend.py`). Data: BTC, ETH, SOL, 2021-01-01 to 2026-11-21, in-sample. Four runs were made in total: three rebalance bands, and one at doubled costs.

| Band | Total | Sharpe | Max drawdown | Years positive | Positive weeks | Turnover a year |
|---|---|---|---|---|---|---|
| 0.01 | +86.39% | 0.62 | -21.90% | 5 of 6 (2025: -15.5%) | 44% | 146x equity |
| 0.03 | +79.51% | 0.59 | -22.07% | 5 of 6 | 44% | 145x |
| 0.05 | +97.06% | 0.67 | -21.94% | 5 of 6 | 45% | 140x |
| 0.03, 2x costs | -7.13% | 0.04 | -38.67% | 5 of 6 (2025: -26.2%) | n/a | 145x |

The timing test put the 0.01 run above 199 of 200 time-shifted runs, so the signal is real before costs. But the rule turns the book over about 145 times a year and pays about 11.6% of equity a year in costs; at doubled costs the whole result is gone. The daily rule on the same coins made +129.90% (Sharpe 0.79) and stayed profitable at doubled costs on the ten other coins. The 4-hour version is also no more regular: 44% of weeks and 48% of months are positive.

Decision: rejected on cost fragility, without spending fresh coins on a confirmation run. A wider band does not help because the turnover comes from sign flips, not from small resizing.
