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
- About 500 MB of disk for the default database (two years of 1-minute candles for three symbols, plus the 31-day feature warm-up). Each stored two-year backtest adds roughly 80 MB (the regimes table); the database was about 780 MB after three stored runs.

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

Optionally activate the venv (`source .venv/bin/activate`) and drop the `.venv/bin/` prefix.

## Commands

| Command | What it does |
|---|---|
| `regimebot version` | Print the installed version. |
| `regimebot db upgrade` | Apply all migrations to the configured database. |
| `regimebot data download [--start ISO] [--end ISO]` | Download 1-minute candles and funding rates for all configured symbols. Default range: `backtest.start` minus the feature warm-up (31 days: 2023-11-30 for the default 2024-01-01 start) to `backtest.end`. Re-running resumes. |
| `regimebot data check` | Validate stored candles per symbol over the warm-up plus backtest range (duplicates, bad OHLC, gaps, short history). Exit code 1 if any symbol fails. |
| `regimebot regimes study` | Print and save how regimes relate to forward returns, per symbol, to files like `reports/regime_study_BTC_USDT_USDT.csv`. |
| `regimebot backtest run [--start ISO] [--end ISO] [--note TEXT] [--set key=value ...]` | Backtest on stored candles, store the run, print headline numbers and the run id. |
| `regimebot backtest walk-forward [--start] [--end] [--set ...]` | Fit the config's grid on each 365-day train span, test on the next 90 days. Slow (about 40 backtests). Uses the config's risk limits (use `--set risk.max_drawdown=1.0` for full windows) and prints how many test windows halted. Writes `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv`. |
| `regimebot backtest robustness [--param PATH ...] [--start] [--end] [--set ...]` | Perturb parameters by +/-20% (default: four parameters), double all costs, compare effective gross leverage 1.0 vs 1.5 (the group cap is raised with the gross cap; output shows max gross seen and how many approvals each cap limited). Writes `reports/sensitivity.csv`, `cost_stress.csv`, `leverage_compare.csv`. |
| `regimebot report RUN_ID [--variants-tried N] [--random-seeds N]` | Rebuild the HTML report for a stored backtest into `reports/run_<id>/`. Recomputes baselines (random entry takes a minute or two; `--random-seeds 0` skips it, default 3). |
| `regimebot paper run [--resume RUN_ID]` | Paper trade until Ctrl-C or SIGTERM. See [docs/operations.md](docs/operations.md). |
| `regimebot paper end RUN_ID [--force]` | End a stopped paper run for good (it cannot be resumed). Refuses while the run is running, or while it holds open positions unless `--force`. |
| `regimebot paper status [--max-age SECONDS]` | Show the latest paper run: liveness, equity, positions, regimes, halt, risk events. Exits 1 when an open run that was not stopped cleanly has not polled within `--max-age` (for uptime checks). |
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
| `risk` | 0.5% risk per trade, 1.5x gross leverage cap, 75% per-asset and 100% per-group exposure (all three symbols are one group, so the effective gross cap is 1.0x), daily 1.5% / weekly 4% / max drawdown 10% limits, loss-streak rules. |
| `backtest`, `walk_forward` | Period (2024-01-01 to 2026-01-01), start equity 10,000 USDT, walk-forward windows and grid. |
| `paper` | Poll interval 10 s, stale after 180 s, 3000 warm-up decision bars (must cover the features' warm-up of 2,976). |

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
| Drawdown halt off (run 6; `--set risk.max_drawdown=1.0`) | 1,312 trades, net -4,546.86 USDT (-45.47%); fees 4,031.04 and funding 58.07, so about -516 before fees and about -458 before fees and funding. Max drawdown 48.64%, profit factor 0.83, win rate 31.86%, expectancy -0.16R, max gross leverage seen 1.02x. Buy and hold: +5,302.82 (+53.03%, max drawdown 50.52%). Random entries with the same exits: median -6,433.63, range -6,883.03 to -4,263.12, profit factor 0.67. The entry rules do somewhat better than random entries (though one of three random seeds lost less than the strategy), and nowhere near enough to cover costs. |
| Robustness (two years, drawdown halt off) | Moving each of four parameters by +/-20% keeps the net result between -4,278.77 and -5,207.16 (profit factor 0.81 to 0.85). Doubling all costs gives -6,976.88 (profit factor 0.69). Effective leverage 1.0x versus 1.5x: -4,546.86 (max drawdown 48.64%, max gross seen 1.02x) versus -5,122.05 (max drawdown 53.19%, max gross seen 1.52x), so more leverage made the loss larger. |
| Walk-forward, default limits (drawdown halt on; 9-combination grid, 4 windows, 36 variants tried) | The selected training run in every window hit the halt (2024-04-07, 2024-07-02, 2024-08-12, 2024-12-03), so training saw only part of each year. Best in-training profit factors 0.94, 0.93, 0.54, 0.80 (on 180, 184, 82, 106 trades up to the halt). Out-of-sample 463 trades, net -770.05; window nets +1,438.60, -418.84, -858.99, -930.82. The last three test windows (CSV `window` 1 to 3) halted on 2025-04-28, 2025-09-08 and 2025-12-10; the first, winning, window ran in full. |
| Walk-forward, drawdown halt off (`--set risk.max_drawdown=1.0`, full windows; same grid, 36 variants) | Best in-training profit factors 0.84, 0.96, 0.87, 0.91 on 677, 647, 627, 623 trades; training nets -2,434.49, -513.23, -1,827.04, -1,289.81. Out-of-sample 636 trades, net -2,234.05; test profit factors 1.36, 0.65, 0.86, 0.59; window nets +1,329.28, -1,557.25, -508.65, -1,497.43 (one winning quarter, three losing). The best parameter set lost money in training in every window; the winning quarter came from parameters that had lost in training, so it is not evidence of an edge. |
| Regime study | Regimes separate the size of the next move, not its direction. A 5% move within 24 hours followed 22% / 37% / 60% of high-volatility-expansion bars on BTC / ETH / SOL, versus 6% / 16% / 30% after compression. The mean 24-hour return after a "trending up" bar on BTC was +0.27% against a standard deviation of 2.2%. |

Artefacts (the `reports/` directory is git-ignored; the commands regenerate them): `reports/run_5/report.html` (default limits), `reports/run_6/report.html` (halt off), `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv` (halt off, the last run), `reports/walk_forward_windows_default_limits.csv` and `reports/walk_forward_oos_trades_default_limits.csv` (copies of the default-limits run, made by hand), `reports/sensitivity.csv`, `reports/cost_stress.csv`, `reports/leverage_compare.csv`, `reports/regime_study_*.csv`. Runs 1 to 3 in the development database predate the warm-up fix and are superseded.

An earlier version of these results, before the feature warm-up fix, differed for the default-limits run (60 trades, -9.72%); see the worked example in [docs/research-protocol.md](docs/research-protocol.md).

Conclusion: as specified, these strategies show no edge on this data. The regimes carry information about how much price will move, but not which way, and the strategies depend on direction. See [docs/research-protocol.md](docs/research-protocol.md) for how to read these numbers.

## Tests and linters

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy regimebot
```

CI (`.github/workflows/ci.yml`) installs with `requirements.lock` as a constraint and runs the same four checks on every pull request. A second job runs the database tests (`pytest -m db`) against Postgres 16; it has never been executed. mypy runs in strict mode. Dependabot is enabled for pip and GitHub Actions. Tests use synthetic data and fake exchanges; they need no network.

## Docker

`Dockerfile` and `docker-compose.yml` (Postgres 16 plus the app) are provided and have not been tested on the development machine, where Docker is not installed. The image installs with `requirements.lock` as a constraint, has a `.dockerignore`, and records `GIT_SHA` (a build argument) as the run's git sha. Runs made from a dirty working tree are stamped `<sha>-dirty`. See [ADR 0002](docs/adr/0002-sqlite-default-postgres-optional.md) and the Compose note in [docs/operations.md](docs/operations.md).

## Not implemented

- Live order placement of any kind (see [ADR 0003](docs/adr/0003-no-live-orders-in-v1.md)).
- WebSocket feeds (REST polling is used), open-interest and liquidation history, implied volatility, news and sentiment.
- A web dashboard.
- Separate staging and production environments and PR preview deploys (this is a local tool).

Enterprise items that deliberately do not apply to a single-user local research tool:

- Multi-tenancy and tenant isolation.
- SSO and SCIM.
- GDPR data export and erasure paths (the database holds only public market data and simulated trades).
- Sentry error reporting. Add it before any real deployment.

## Known follow-ups

Recorded and not fixed:

- A 15-minute bar with zero volume would block that symbol for 30 days (none in the current data); fix before adding thinner symbols.
- Paper trading does not alert when its warm-up window is incomplete. It has only a 24-bar margin over the features' warm-up.
- The report's "effective cap" ignores the per-asset cap when only one symbol is configured.
- Raising `features.pctile_window` above about 2,900 in a backtest also requires raising `paper.warmup_bars`.
- `psycopg` is not pinned in `requirements.lock`.
- The Postgres CI job and all Docker steps have never been run.
- CI actions are pinned by tag, not by commit SHA.

## Documentation

- [docs/architecture.md](docs/architecture.md): structure and conventions
- [docs/operations.md](docs/operations.md): paper trading runbook, kill switch, state, alerts
- [docs/research-protocol.md](docs/research-protocol.md): rules for not fooling yourself
- [docs/adr/](docs/adr/): decision records
