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

- Python 3.11 or newer (developed on 3.12).
- Network access to Binance USDT-margined perpetuals public endpoints for `data download` and paper trading. No account or API key.
- About 500 MB of disk for the default database (two years of 1-minute candles for three symbols).

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
| `regimebot data download [--start ISO] [--end ISO]` | Download 1-minute candles and funding rates for all configured symbols (default range: the config's `backtest.start` to `backtest.end`). Re-running resumes. |
| `regimebot data check` | Validate stored candles per symbol (duplicates, bad OHLC, gaps, short history). Exit code 1 if any symbol fails. |
| `regimebot regimes study` | Print and save how regimes relate to forward returns, per symbol, to `reports/regime_study_<symbol>.csv`. |
| `regimebot backtest run [--start ISO] [--end ISO] [--note TEXT] [--set key=value ...]` | Backtest on stored candles, store the run, print headline numbers and the run id. |
| `regimebot backtest walk-forward [--start] [--end] [--set ...]` | Fit the config's grid on each 365-day train span, test on the next 90 days. Slow (about 40 backtests). Writes `reports/walk_forward_windows.csv` and `reports/walk_forward_oos_trades.csv`. |
| `regimebot backtest robustness [--param PATH ...] [--start] [--end] [--set ...]` | Perturb parameters by +/-20% (default: four parameters), double all costs, compare max gross leverage 1.0 vs 1.5. Writes `reports/sensitivity.csv`, `cost_stress.csv`, `leverage_compare.csv`. |
| `regimebot report RUN_ID [--variants-tried N] [--random-seeds N]` | Rebuild the HTML report for a stored backtest into `reports/run_<id>/`. Recomputes baselines (random entry takes a minute or two; `--random-seeds 0` skips it, default 3). |
| `regimebot paper run [--resume RUN_ID]` | Paper trade until Ctrl-C or SIGTERM. See [docs/operations.md](docs/operations.md). |
| `regimebot paper status [--max-age SECONDS]` | Show the latest paper run: liveness, equity, positions, regimes, halt, risk events. Exits 1 when an open run has not polled within `--max-age` (for uptime checks). |
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
| `risk` | 0.5% risk per trade, 1.5x gross leverage, 75% per-asset and 100% per-group exposure, daily 1.5% / weekly 4% / max drawdown 10% limits, loss-streak rules. |
| `backtest`, `walk_forward` | Period (2024-01-01 to 2026-01-01), start equity 10,000 USDT, walk-forward windows and grid. |
| `paper` | Poll interval 10 s, stale after 180 s, 3000 warm-up decision bars. |

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

2024-01-01 to 2026-01-01, BTC, ETH and SOL perpetuals, 10,000 USDT start. Reported as measured, with no tuning to improve them.

| Run | Result |
|---|---|
| Default limits | 60 trades, net -971.71 USDT (-9.72%). Halted by the 10% drawdown limit on 2024-02-05; no entries after that. Over the same five weeks buy and hold lost 51.49; random entries (3 seeds, same trade frequency) had a median of -359.25, range -982.29 to +103.33. |
| Drawdown halt off (`--set risk.max_drawdown=1.0`) | 1,300 trades, net -4,726.92 USDT (-47.27%), of which fees 3,835.64 (roughly -840 before fees). Max drawdown 48.62%, profit factor 0.82. Buy and hold over the two years: +5,213.78. Random entries with the same exits: median -6,107.72, range -6,471.48 to -4,814.97 (profit factor 0.69). The entry rules do slightly better than random entries, but nowhere near enough to cover costs. |
| Robustness (two years, drawdown halt off) | Moving each of four parameters by +/-20% keeps the net result between -4,279.65 and -5,302.76. Doubling all costs gives -6,932.21. A 1.0x versus 1.5x gross leverage cap gives identical results (the cap never binds at 0.5% risk per trade). |
| Walk-forward (9-combination grid, 4 windows, 36 variants tried) | Best in-training profit factor was below 1 in every window (0.95, 0.93, 0.54, 0.80). Out-of-sample total -554.29 over 465 trades; one winning quarter, three losing. |
| Regime study | Regimes separate the size of the next move, not its direction. A 5% move within 24 hours followed 23% / 37% / 59% of high-volatility-expansion bars on BTC / ETH / SOL, versus 6% / 16% / 29% after compression. |

Artefacts (the `reports/` directory is git-ignored; the commands regenerate them): `reports/run_2/report.html`, `reports/run_3/report.html`, `reports/walk_forward_windows.csv`, `reports/walk_forward_oos_trades.csv`, `reports/sensitivity.csv`, `reports/cost_stress.csv`, `reports/leverage_compare.csv`, `reports/regime_study_*.csv`.

Conclusion: as specified, these strategies show no edge on this data. The regimes carry information about how much price will move, but not which way, and the strategies depend on direction. See [docs/research-protocol.md](docs/research-protocol.md) for how to read these numbers.

## Tests and linters

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy regimebot
```

CI (`.github/workflows/ci.yml`) runs the same four checks on every pull request. mypy runs in strict mode. Dependabot is enabled for pip and GitHub Actions. Tests use synthetic data and fake exchanges; they need no network.

## Docker

`Dockerfile` and `docker-compose.yml` (Postgres 16 plus the app) are provided and have not been tested on the development machine, where Docker is not installed. See [ADR 0002](docs/adr/0002-sqlite-default-postgres-optional.md) and the Compose note in [docs/operations.md](docs/operations.md).

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

## Documentation

- [docs/architecture.md](docs/architecture.md): structure and conventions
- [docs/operations.md](docs/operations.md): paper trading runbook, kill switch, state, alerts
- [docs/research-protocol.md](docs/research-protocol.md): rules for not fooling yourself
- [docs/adr/](docs/adr/): decision records
