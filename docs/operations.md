# Operations: paper trading runbook

Paper trading polls live public market data, runs the same engine as the backtest and simulates every fill. No order is ever sent and no API key is used. All commands run from the repository root.

## Before you start

```bash
.venv/bin/regimebot db upgrade
.venv/bin/regimebot data check
```

`data check` validates the stored history for the configured backtest range, so it is only needed for backtests. A paper session fetches its own warm-up history (3000 decision bars, about 31 days) on start.

## Start a session

```bash
.venv/bin/regimebot paper run
```

The banner (stderr) shows the run id, symbols, poll interval, kill file path, how to stop and how to resume. Logs are JSON lines on stdout, so redirect them if you want a file:

```bash
.venv/bin/regimebot paper run >> data/paper.log
```

The session polls every 10 seconds (`paper.poll_seconds`) and decides once per closed 15-minute bar, when every symbol has that bar or is stale. Run it somewhere that stays on (a laptop that sleeps will go stale and skip decisions).

## Check on it

```bash
.venv/bin/regimebot paper status
```

Shows the latest paper run: `open` (running), or `stopped at <time>` after a clean stop (Ctrl-C or SIGTERM), with the resume command; the age of its last poll; equity and its timestamp, cash, the active halt (or `no`), open positions with stop and target, the last regime per symbol and the last 10 risk events. It reads the database and works while a session runs. It exits 0 for a running run and for a cleanly stopped run. It exits 1 (printing `NOT POLLING`) for an open run with no clean-stop marker whose last poll is older than `--max-age` seconds (default 3 x `poll_seconds` + `stale_after_seconds`, 210 s): the process crashed or hung. See Monitoring.

## Stop and resume

Stop with Ctrl-C or SIGTERM. The session finishes the current poll, saves its state, records a clean-stop marker (`stopped_at`) and exits. Positions stay open and the run stays resumable; nothing is flattened.

```bash
.venv/bin/regimebot paper run --resume 4
```

Use the run id from the banner or from `paper status`. Rules on resume:

- The run must be a paper run started with the same config (the config hash is checked; a changed config is refused). Change parameters by starting a new run.
- Positions, cash, peak equity, risk state and queued exits are restored.
- Base bars that closed while the process was down are replayed, so a stop or target hit during downtime is honoured at the bar where it happened.
- A decision whose wait (180 s after its bar close) has ended is skipped and never run late. If it was missed while the process was running, it is alerted as `decision_missed`. If it closed before this process started (a routine start or resume, including a fresh start more than 180 s into a bar), it is logged at info without an alert. The session continues at the next live decision.
- Queued entries are cancelled, with a recorded decision, on a clean shutdown. After a hard kill (SIGKILL) a leftover entry is dropped on resume.
- Each run has a lock file, `data/paper_run_<id>.lock`. A second session cannot resume a run that is already running.

## Kill switch

```bash
.venv/bin/regimebot kill      # create data/KILL
.venv/bin/regimebot unkill    # remove it
```

While the file exists, a running session acts on it at every poll (about every 10 seconds, `paper.poll_seconds`), not only at decisions, even when nothing is open: the halt is recorded, alerted and shown by `paper status` within about one poll. Open positions are flattened from fresh quotes immediately, queued entries are cancelled and nothing new opens. The halt is recorded as `manual_kill`.

The halt is sticky. `unkill` removes the file but the run stays halted and flat, and the halt is saved with the run, so `paper run --resume` restores it. To let a run trade again, stop the session, remove the kill file, then clear the halt:

```bash
.venv/bin/regimebot unkill
.venv/bin/regimebot paper clear-halt 4
.venv/bin/regimebot paper run --resume 4
```

`clear-halt` refuses while the kill file exists or the run is running. A `max_drawdown` halt re-triggers at the next check while drawdown is still past the limit; by design that run is over, so start a new one (`paper run` without `--resume`, which starts from `backtest.initial_equity` with fresh risk state). Daily and weekly loss halts clear on their own at the next UTC day or ISO week.

The kill file path is `<repo>/data/KILL`. A backtest ignores it: backtests use a throwaway kill file so they stay reproducible.

## Stale data

A symbol is stale when its last candle download failed or its newest bar is older than 180 seconds (`paper.stale_after_seconds`). For a stale symbol the engine records a `stale_data` risk event, trades nothing in it and opens no new entries; an open position keeps its stop and target, which are still checked on base bars. Each incident raises one alert, and another when it recovers. A missing bar therefore delays a decision by at most 180 seconds.

Timing limits:

- A fill after a decision can lag it by the quote fetches (each up to about 91 s while the exchange retries). An entry not filled within one decision bar expires.
- Exchange requests are retried per symbol in sequence (up to about 91 s per call). One symbol with hanging requests can therefore make decisions miss for all symbols while it lasts; each miss is alerted as `decision_missed`. This is a known limit.

## Alerts

Every alert is an ERROR log line. If `ALERT_WEBHOOK_URL` is set (http or https), the same alert is POSTed as JSON `{"kind", "detail", "ts"}` with a 5-second timeout:

```bash
export ALERT_WEBHOOK_URL="https://example.invalid/hook"   # replace with your endpoint
.venv/bin/regimebot paper run
```

A webhook failure is logged and never stops the session. The URL is never logged. Alert kinds raised by the paper session: `stale_data` and `stale_data_recovered` (per symbol), `poll_error` and `poll_error_recovered` (an exception during a poll; the next poll runs normally).

Every halt (`daily_loss`, `weekly_loss`, `max_drawdown`, `manual_kill`) sends an alert, and daily and weekly halts send a recovery alert when they clear. A skipped decision sends `decision_missed`. Halts are also visible in `paper status` and the logs (`risk_halt`).

## Where state lives

| What | Where |
|---|---|
| Candles, funding, runs, signals, decisions, fills, trades, equity, regimes, risk events | Database (`data/regimebot.db` by default, or `DATABASE_URL`), every row tagged with its `run_id` |
| Resumable paper state (positions, cash, risk state, cursors) | `open_positions` table, rewritten at the end of every poll |
| Kill file | `data/KILL` |
| Per-run lock | `data/paper_run_<id>.lock` |
| Reports and CSVs | `reports/` |
| Logs | stdout (JSON lines); `paper run` redirects are up to you |

`data/` and `reports/` are not in git. Back up the database file while no session is running (or use `sqlite3 data/regimebot.db ".backup copy.db"`).

The long SQLite writers are `data download`, the final save of `backtest run`, and a paper session (which writes every poll). A second writer waits only about 5 seconds (SQLite's default busy timeout) and then fails with "database is locked". Walk-forward reads candles once and writes CSVs, so it is not a long writer. Run writers one at a time, or use Postgres.

## Monitoring

`paper status` exits 1 when a run that was not stopped cleanly has not polled recently, so a cron job can watch for a hung or dead session. A generic example that runs every five minutes and sends mail on failure (use whatever notifier you have in place of `mail`):

```bash
*/5 * * * * cd /path/to/regimebot && .venv/bin/regimebot paper status >/dev/null || echo "regimebot paper run is stale or hung" | mail -s "regimebot" you@example.com
```

Exit code 0 means the latest run is polling or was stopped cleanly with Ctrl-C or SIGTERM. Exit code 1 means an open run with no clean-stop marker has not polled within `--max-age`: it crashed or hung. Pair it with `ALERT_WEBHOOK_URL` for in-session alerts.

## Docker Compose (untested on the development machine)

Docker was not available during development, so none of these steps has been run there.

```bash
export POSTGRES_PASSWORD=$(openssl rand -hex 24)   # keep it: the db volume stores it on first start
docker compose up -d db
docker compose run --rm app db upgrade
docker compose run --rm app data download
docker compose run -d app paper run
```

The image ships `alembic.ini` and creates writable `data/` and `reports/`. Compose mounts named volumes at `/app/data` and `/app/reports`, shared by every `app` container, so `docker compose run --rm app kill` reaches a running session through the shared kill file, and reports persist. The `pgdata` volume holds the Postgres data; back it up (and test a restore) before it holds anything you cannot re-download.

## API keys for any future live use

Not needed now. If live order placement is ever added (see ADR 0003):

- Create trade-only keys. Never enable withdrawals.
- Restrict the key to your IP address.
- Keep keys only in environment variables or a secret manager, never in git, config files or logs.
- Use the venue's testnet first.

## Known gaps

- Error reporting (Sentry) and uptime monitoring are not set up; add them before any real deployment.
- Structured logs have no request or trace IDs; they are single-process JSON lines.
