# regimebot

A research system that tests whether a rule-based, regime-aware strategy has an
edge on crypto perpetual futures after costs. It runs backtests on stored
history, walk-forward and robustness studies, and paper trading (live market
data, simulated fills).

It is a research tool, not a profit machine. Its job is to say plainly when a
strategy has no edge. As specified, the strategies in this repository show no
edge on the data tested (see [First results](#first-results)).

## What it is and is not

| It is | It is not |
|---|---|
| A backtester with a conservative fill model (spread, slippage, volume impact, fees, funding) | A trading bot: **no live order placement exists**; no code path can send an order |
| Three rule-based strategies (breakout, trend continuation, mean reversion) gated by a regime classifier | A source of signals or advice |
| Walk-forward, sensitivity, cost-stress and baseline comparisons | Proof that any strategy works |
| A paper-trading session on live public data | Connected to any account: no API key is needed for anything in this version |

## Risk statements

- Crypto trading involves substantial risk, including the loss of everything you put in.
- Leveraged trading can lose money fast.
- Past performance does not guarantee future results.
- No system predicts sudden market changes.
- A backtest is not proof. It is one simulation on one stretch of history, and its fill model is an approximation.
- Test extensively in paper trading before any real capital. Even then, paper results omit real-world effects (queue position, outages, rejected orders).

## Requirements

- Python 3.11 or newer (developed on 3.12). The CLI imports on any platform, but paper trading's run lock needs `fcntl`, so `paper run` works on macOS and Linux only.
- Network access to Binance USDT-margined perpetuals public endpoints for `data download` and paper trading. No account or API key.
- About 500 MB of disk for the default database (two years of 1-minute candles for three symbols, plus the 31-day feature warm-up). Each stored two-year backtest adds roughly 80 MB (the regimes table); the database was about 780 MB after three stored runs, and the full default history download takes roughly 28 minutes and leaves a database of about 800 MB.
- The exchange's public API is geo-blocked in some countries. There the download fails with an HTTP 451 or 403. The exchange adapter is a single class (`regimebot/data/`), so it can be replaced with one for another venue.

## Quick start

All commands run from the repository root.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

On an Intel Mac the newest `ccxt` needs a `cryptography` version with no prebuilt wheel. Let pip choose a compatible `ccxt` instead:

```bash
.venv/bin/pip install --only-binary=cryptography -e ".[dev]"
```

Development used `ccxt` 4.5.64; `requirements.lock` records the exact environment.

Then:

```bash
.venv/bin/regimebot db upgrade          # create or migrate the database
.venv/bin/regimebot data download       # 2024-01-01 .. 2026-01-01, slow (resumable, safe to re-run)
.venv/bin/regimebot data check          # validate stored candles
.venv/bin/regimebot backtest run        # prints headline numbers and "run <id>"
.venv/bin/regimebot report 1            # HTML report in reports/run_1/ (use the printed run id)
```

A shorter first run (a few minutes instead of 28): download three months, starting 31 days before the backtest start so the features are warm, and backtest the same months. `--start` is used as given, so subtract the warm-up yourself:

```bash
.venv/bin/regimebot data download --start 2025-08-31 --end 2025-12-01
.venv/bin/regimebot backtest run --start 2025-10-01 --end 2025-12-01
```

To use one symbol, edit `exchange.symbols` in `config/default.yaml` first (`data download` reads the file; `--set` applies to `backtest` commands only).

After pulling new code, run `regimebot db upgrade` again: migration 0002 adds the bid and ask columns to fills.

Optionally activate the venv (`source .venv/bin/activate`) and drop the `.venv/bin/` prefix.

## Commands

| Command | What it does |
|---|---|
| `regimebot version` | Print the installed version. |
| `regimebot db upgrade` | Apply all migrations to the configured database. |
| `regimebot data download [--start ISO] [--end ISO]` | Download 1-minute candles and funding rates for all configured symbols. Default range: `backtest.start` minus the feature warm-up (31 days: 2023-11-30 for the default 2024-01-01 start) to `backtest.end`. Re-running resumes, and it also fills interior funding holes. Funding is only checked and filled inside the requested range, so include the period you need (for example `--end`). |
| `regimebot data check` | Validate stored candles per symbol over the warm-up plus backtest range (duplicates, bad OHLC, gaps, short history), and funding coverage (largest gap, and any stretch over 17 hours without a funding stamp fails). Exit code 1 if any symbol fails. |
| `regimebot regimes study` | Print and save how regimes relate to forward returns, per symbol, to files like `reports/regime_study_BTC_USDT_USDT.csv`. |
| `regimebot backtest run [--start ISO] [--end ISO] [--note TEXT] [--set key=value ...]` | Backtest on stored candles, store the run, print headline numbers and the run id. Warns when funding is missing in the range (`funding_missing` risk event; funding is not charged there). |
| `regimebot backtest walk-forward [--start] [--end] [--set ...]` | Fit the config's grid on each 365-day train span, test on the next 90 days. Slow (about 40 backtests). Uses the config's risk limits (use `--set risk.max_drawdown=1.0` for full windows) and prints how many test windows halted. Writes `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv`. |
| `regimebot backtest robustness [--param PATH ...] [--start] [--end] [--set ...]` | Perturb parameters by +/-20% (default: four parameters), run costs at x0.5, x1 and x2, compare effective gross leverage 1.0 vs 1.5 (the group cap is raised with the gross cap; output shows max gross seen and how many approvals each cap limited). Writes `reports/sensitivity.csv`, `cost_stress.csv`, `leverage_compare.csv`. |
| `regimebot report RUN_ID [--variants-tried N] [--random-seeds N]` | Rebuild the HTML report for a stored run into `reports/run_<id>/`. Recomputes baselines (random entry takes a minute or two; `--random-seeds 0` skips it, default 3, at most 50). Works for paper runs too: banner, buy and hold as the only baseline, and an Execution quality section. |
| `regimebot paper run [--resume RUN_ID\|latest]` | Paper trade until Ctrl-C or SIGTERM. `--resume latest` resumes the most recent open run or starts a new one. See [docs/operations.md](docs/operations.md). |
| `regimebot paper end RUN_ID [--force]` | End a stopped paper run for good (it cannot be resumed). Refuses while the run is running, or while it holds open positions unless `--force`. |
| `regimebot paper status [--max-age SECONDS]` | Show the latest paper run: liveness, equity, positions, regimes, halt, risk events, rejected polled bars. Exits 1 when an open run that was not stopped cleanly has not polled within `--max-age` (for uptime checks). |
| `regimebot paper clear-halt RUN_ID` | Clear a stopped paper run's recorded halts. Refuses while the kill file exists or the run is running. |
| `regimebot kill` / `regimebot unkill` | Create or remove the kill file (`data/KILL`). A running paper session flattens at its next poll. |

`--set` overrides any config value by dotted path with a YAML value and can be repeated, for example `--set risk.max_drawdown=1.0`. It applies to the `backtest` commands only. `paper run` reads the config file as is.

Walk-forward and robustness runs print a reminder that their numbers are historical simulations, not evidence of future performance.

## Configuration

Everything is in [`config/default.yaml`](config/default.yaml), validated by Pydantic at load time (invalid values fail with a clear message). Edit the file or override with `--set`. Environment variables:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Replaces `database_url`. Default is `sqlite:///data/regimebot.db`. Use `postgresql+psycopg://...` for Postgres (install the `postgres` extra). |
| `ALERT_WEBHOOK_URL` | Optional http(s) URL that receives a JSON POST for each alert. |
| `POSTGRES_PASSWORD` | Used only by `docker-compose.yml`. |

Copy `.env.example` to `.env` if you want a template; the program does not read `.env` itself, so export the variables in your shell or let Compose read them.

| Section | Controls |
|---|---|
| `exchange` | Venue (`binanceusdm`), symbols (BTC, ETH, SOL perpetuals), 1m base frame, 15m decision frame. |
| `costs` | Taker 5 bps, maker 2 bps, half spread 1 bp, slippage 2 bps, impact 1 bp per 1% of bar volume, entry participation cap 5%. |
| `features`, `regime` | Window lengths and thresholds for the causal features and the regime classifier. |
| `playbook` | Per regime: which strategies may trade (in priority order), size multiplier, whether entries are allowed, whether to force-close. |
| `strategies` | Per strategy: targets, minimum reward-to-risk, stop rules, hold limits, trailing stops. |
| `risk` | 0.5% risk per trade, minimum entry notional `min_notional` (10 USDT), 1.5x gross leverage cap, 75% per-asset and 100% per-group exposure (all three symbols are one group, so the effective gross cap is 1.0x), daily 1.5% / weekly 4% / max drawdown 10% limits, loss-streak rules. |
| `backtest`, `walk_forward` | Period (2024-01-01 to 2026-01-01), start equity 10,000 USDT, walk-forward windows and grid. |
| `paper` | Poll interval 10 s, stale after 180 s, 3400 warm-up decision bars (must cover the features' warm-up of 2,976). |

## How it works

Candles go through causal features to a rule-based regime label (trend up/down, range, compression, expansion, liquidity stress, event risk, no trade). The playbook maps each regime to the strategies allowed. Strategies emit signals on the close of a 15-minute decision bar; the risk manager sizes, caps or rejects them; the fill model executes at the next 1-minute bar. Backtest and paper trading share one engine. See [docs/architecture.md](docs/architecture.md) and the decisions in [docs/adr/](docs/adr/).

| Module | Responsibility |
|---|---|
| `regimebot/config.py` | Pydantic config, YAML loading, overrides, env |
| `regimebot/types.py` | Shared dataclasses and enums |
| `regimebot/db/` | SQLAlchemy models, repository functions; Alembic migrations in `regimebot/db/migrations/`, applied by `db upgrade` |
| `regimebot/data/` | Exchange adapter (ccxt), history download, validation, resampling |
| `regimebot/market_state/` | Features, regime classifier, playbook lookup, regime study |
| `regimebot/strategy/` | Breakout, trend continuation, mean reversion |
| `regimebot/risk/` | Sizing, exposure caps, loss limits, halts, kill switch |
| `regimebot/execution/` | Fill model and funding |
| `regimebot/engine/` | Portfolio accounting and the shared event loop |
| `regimebot/backtest/` | Runner, walk-forward, robustness, baselines |
| `regimebot/reporting/` | Metrics, HTML/CSV/PNG reports, alerts |
| `regimebot/paper/` | Polling session, stale-data guard, resumable state |
| `regimebot/cli.py` | Typer commands |

## First results

Data 2023-11-30 to 2026-01-01 (the first 31 days only warm up the features); trading 2024-01-01 to 2026-01-01; BTC, ETH and SOL perpetuals; 10,000 USDT start. Reported as measured, with no tuning to improve them.

| Run | Result |
|---|---|
| Default limits (run 5) | 76 trades, net -715.60 USDT (-7.16%), fees 276.89, max drawdown 10.03%, profit factor 0.66, win rate 26.32%, expectancy -0.28R. Halted by the 10% drawdown limit on 2024-02-07; no entries after that. Over the same active period buy and hold lost 35.53; random entries (3 seeds, calibrated trade frequency) had a median of -327.34, range -911.82 to -315.30. |
| Drawdown halt off (run 6; `--set risk.max_drawdown=1.0`) | 1,312 trades, net -4,546.86 USDT (-45.47%); fees 4,031.04 and funding 58.07, so total modelled costs were about 6,460 (fees 4,031, funding 58, spread and slippage about 2,374, the last inside the fill prices). Before all modelled costs the rules made a small positive gross result, roughly +1,900 over two years, several times smaller than the costs. Max drawdown 48.64%, profit factor 0.83, win rate 31.86%, expectancy -0.16R, max gross leverage seen 1.02x. Buy and hold: +5,302.82 (+53.03%, max drawdown 50.52%). Random entries with the same exits: median -6,433.63, range -6,883.03 to -4,263.12, profit factor 0.67. The entry rules do somewhat better than random entries (though one of three random seeds lost less than the strategy), and nowhere near enough to cover costs. |
| Robustness (two years, drawdown halt off) | Moving each of four parameters by +/-20% keeps the net result between -4,278.77 and -5,207.16 (profit factor 0.81 to 0.85). Doubling all costs gives -6,976.88 (profit factor 0.69). Effective leverage 1.0x versus 1.5x: -4,546.86 (max drawdown 48.64%, max gross seen 1.02x) versus -5,122.05 (max drawdown 53.19%, max gross seen 1.52x), so more leverage made the loss larger. |
| Walk-forward, default limits (drawdown halt on; 9-combination grid, 4 windows, 36 variants tried) | The selected training run in every window hit the halt (2024-04-07, 2024-07-02, 2024-08-12, 2024-12-03), so training saw only part of each year. Best in-training profit factors 0.94, 0.93, 0.54, 0.80 (on 180, 184, 82, 106 trades up to the halt). Out-of-sample 463 trades, net -770.05; window nets +1,438.60, -418.84, -858.99, -930.82. The last three test windows (CSV `window` 1 to 3) halted on 2025-04-28, 2025-09-08 and 2025-12-10; the first, winning, window ran in full. |
| Walk-forward, drawdown halt off (`--set risk.max_drawdown=1.0`, full windows; same grid, 36 variants) | Best in-training profit factors 0.84, 0.96, 0.87, 0.91 on 677, 647, 627, 623 trades; training nets -2,434.49, -513.23, -1,827.04, -1,289.81. Out-of-sample 636 trades, net -2,234.05; test profit factors 1.36, 0.65, 0.86, 0.59; window nets +1,329.28, -1,557.25, -508.65, -1,497.43 (one winning quarter, three losing). The best parameter set lost money in training in every window; the winning quarter came from parameters that had lost in training, so it is not evidence of an edge. |
| Regime study | Regimes separate the size of the next move, not its direction. A 5% move within 24 hours followed 22% / 37% / 60% of high-volatility-expansion bars on BTC / ETH / SOL, versus 6% / 16% / 30% after compression. The mean 24-hour return after a "trending up" bar on BTC was +0.27% against a standard deviation of 2.2%. |

Artefacts (the `reports/` directory is git-ignored; the commands regenerate them): `reports/run_5/report.html` (default limits), `reports/run_6/report.html` (halt off), `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv` (halt off, the last run), `reports/walk_forward_windows_default_limits.csv` and `reports/walk_forward_oos_trades_default_limits.csv` (copies of the default-limits run, made by hand), `reports/sensitivity.csv`, `reports/cost_stress.csv`, `reports/leverage_compare.csv`, `reports/regime_study_*.csv`. Runs 1 to 3 in the development database predate the warm-up fix and are superseded.

An earlier version of these results, before the feature warm-up fix, differed for the default-limits run (60 trades, -9.72%); see the worked example in [docs/research-protocol.md](docs/research-protocol.md).

After-the-fact slices of the halt-off run (run 6). These are hypotheses, not findings: they were looked at after the result was known, and slicing many ways always finds a winner.

| Slice | Trades | Net | Before fees |
|---|---|---|---|
| Mean reversion | 292 | -2,404 | -1,464 |
| Trend | 666 | -1,678 | -62 |
| Breakout | 354 | -465 | +1,069 |

By side: longs -4,196, shorts -351.

Cost-model note: the modelled 3 bps per side of spread plus slippage is pessimistic for a small account on these three markets, where live quoted spreads are far lower. With about 0.5 bps the two-year halt-off loss would be roughly -2,600 instead of -4,547, still a loss. `backtest robustness` now reports costs at x0.5, x1 and x2.

Holdout (run once, 2026-11-21): 2026-01-01 to 2026-11-21 had never been used for any parameter or design choice, and it was a falling market, unlike 2024-2025. The frozen default configuration was run on it once against pass criteria written down beforehand, and it failed: with the drawdown halt off, 503 trades, net -3,983.88 (-39.84%), profit factor 0.60, max drawdown 45.04%, better than only 2 of 30 random-entry seeds (buy and hold -4,097.71); with the default limits, trading halted on 2026-03-16 at -369.37 (-3.69%). Breakout alone also lost (-344.04). This closes the question for this rule set: no edge after costs in a rising or a falling market. Details and criteria: [docs/research-protocol.md](docs/research-protocol.md#pre-registered-criteria-for-the-2026-holdout).

Conclusion: as specified, these strategies show no edge on this data. The regimes carry information about how much price will move, but not which way, and the strategies depend on direction. See [docs/research-protocol.md](docs/research-protocol.md) for how to read these numbers.

## Tests and linters

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy regimebot
```

CI (`.github/workflows/ci.yml`) installs with `requirements.lock` as a constraint and runs the same four checks on every pull request. The checks run on Python 3.11 and 3.12. A second job runs the database tests (`pytest -m db`) against Postgres 16; it runs green in GitHub Actions on every push and pull request. A third job builds the Docker image and runs `version` and `db upgrade` in it. mypy runs in strict mode. Actions are pinned by commit SHA and Dependabot is enabled for pip and GitHub Actions. Tests use synthetic data and fake exchanges; they need no network.

## Docker

`Dockerfile` and `docker-compose.yml` (Postgres 16 plus the app) are provided. CI builds the image and runs `version` and `db upgrade` in it, but Docker has still not been run on the author's own machine, so the Compose file and the paper-trading container have not been exercised there. The image installs with `requirements.lock` as a constraint, has a `.dockerignore`, and records `GIT_SHA` (a build argument) as the run's git sha. Runs made from a dirty working tree are stamped `<sha>-dirty`. See [ADR 0002](docs/adr/0002-sqlite-default-postgres-optional.md) and the Compose note in [docs/operations.md](docs/operations.md).

## What the original brief asked for that is deliberately not built

- Live exchange adapter and real orders: research first, and no edge has been shown ([ADR 0003](docs/adr/0003-no-live-orders-in-v1.md)).
- Per-regime playbook fields beyond allowed strategies, size multiplier, entry permission and force-close (required stop distance, minimum reward-to-risk and maximum spread per regime): these are global per strategy and per config (`regime.max_spread_bps`).
- Strategy-specific forced-exit conditions: exits are stop, target, trailing stop, maximum hold time, a regime force-close (`liquidity_stress`) and the kill switch.
- Automatic deleveraging of already-open positions in high volatility: new entries are blocked or reduced (expansion halves the size) and open positions keep their stops.
- Regime inputs from funding extremes, order-book imbalance, open interest, cross-asset correlation and time of day: not used. Event windows exist as a config list (`regime.event_windows`, empty by default) and a return shock threshold also triggers `event_risk`.
- Exchange status and trading-pair metadata checks.
- An anomaly-detection module (bad polled bars are dropped and counted, nothing more).
- An explicit latency parameter: the next-1-minute-open fill rule is the latency model.
- Simple benchmark strategies other than buy and hold and random entry.
- A "comparable historical setups" model for the 5% target: strategies apply a volatility-based plausibility check (`plausibility_mult`), and the regime study answers the question at regime level.
- WebSocket data: REST polling is used.
- A web dashboard: there is a CLI and HTML reports.
- News, sentiment and on-chain data.
- A weekly research summary as a separate report.

Also not built: open-interest and liquidation history, implied volatility, separate staging and production environments and PR preview deploys (this is a local tool).

Dry-run mode is paper trading: no order is ever submitted anywhere.

## Not applicable

Authentication, tenancy and tenant isolation, SSO and SCIM, rate limiting on public endpoints, SLOs, uptime monitoring as a service, Sentry and GDPR data export and erasure paths do not apply. This is a single-user local research tool with no server, no users and no personal data; the database holds only public market data and simulated trades. If it were ever deployed for others, all of these would need to be done first.

## Known follow-ups

Recorded and not fixed:

- Paper trading has not yet completed a supervised 24-hour soak with a live decision, entry, exit, stop/resume and kill. Do that before leaving it unattended.
- Raising `features.pctile_window` above about 2,900 in a backtest also requires raising `paper.warmup_bars` (the default is 3400).
- Any live adapter needs a legal check for the user's jurisdiction first (see ADR 0003).
- In backtests the spread is modelled as a constant, so the `liquidity_stress` regime can only be triggered by the illiquidity percentile there, while paper uses the live quoted spread. Results by regime are therefore not strictly comparable between the two.

## Documentation

- [docs/architecture.md](docs/architecture.md): structure and conventions
- [docs/operations.md](docs/operations.md): paper trading runbook, kill switch, state, alerts
- [docs/research-protocol.md](docs/research-protocol.md): rules for not fooling yourself
- [docs/adr/](docs/adr/): decision records
- [SECURITY.md](SECURITY.md): reporting vulnerabilities

## Licence

No licence has been chosen yet. Until one is added, all rights are reserved: you may read the code, but not copy, modify or redistribute it.

To report a vulnerability, see [SECURITY.md](SECURITY.md).
