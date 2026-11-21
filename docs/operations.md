# Operations: paper trading runbook

Paper trading polls live public market data, runs the same engine as the backtest and simulates every fill. No order is ever sent and no API key is used. All commands run from the repository root.

## Before you start

```bash
.venv/bin/regimebot db upgrade
.venv/bin/regimebot data check
```

`data check` confirms the stored history is complete. A paper session fetches its own warm-up history (3000 decision bars, about 31 days) on start.

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

Shows the latest paper run: open or ended, the age of its last poll, equity and its timestamp, cash, the active halt (or `no`), open positions with stop and target, the last regime per symbol and the last 10 risk events. It reads the database and works while a session runs. With an open run it exits 1 when the last poll is older than `--max-age` seconds (default 3 x `poll_seconds` + `stale_after_seconds`, 210 s), which flags a hung process. See Monitoring.

## Stop and resume

Stop with Ctrl-C or SIGTERM. The session finishes the current poll, saves its state and exits. Positions stay open and the run stays open; nothing is flattened.

```bash
.venv/bin/regimebot paper run --resume 4
```

Use the run id from the banner or from `paper status`. Rules on resume:

- The run must be a paper run started with the same config (the config hash is checked; a changed config is refused). Change parameters by starting a new run.
- Positions, cash, peak equity, risk state and queued exits are restored.
- Base bars that closed while the process was down are replayed, so a stop or target hit during downtime is honoured at the bar where it happened.
- A decision whose wait (180 s after its bar close) has ended is skipped and alerted as `decision_missed`; it is never run late. This also applies to a fresh start more than 180 s into a bar. The session resumes at the next live decision.
- Queued entries are cancelled, with a recorded decision, on a clean shutdown. After a hard kill (SIGKILL) a leftover entry is dropped on resume.
- Each run has a lock file, `data/paper_run_<id>.lock`. A second session cannot resume a run that is already running.

## Kill switch

```bash
.venv/bin/regimebot kill      # create data/KILL
.venv/bin/regimebot unkill    # remove it
```

While the file exists, a running session acts on it at every poll (about every 10 seconds, `paper.poll_seconds`), not only at decisions: it halts, flattens open positions from fresh quotes immediately, cancels queued entries and opens nothing. The halt is recorded as `manual_kill` and alerted.

The halt is sticky. `unkill` removes the file but the run stays halted and flat, and the halt is saved with the run, so `paper run --resume` restores it. To let a run trade again, stop the session, remove the kill file, then clear the halt:

```bash
.venv/bin/regimebot unkill
.venv/bin/regimebot paper clear-halt 4
.venv/bin/regimebot paper run --resume 4
```

`clear-halt` refuses while the kill file exists or the run is running. A `max_drawdown` halt re-triggers at the next check while drawdown is still past the limit; by design that run is over, so start a new one (`paper run` without `--resume`, which starts from `backtest.initial_equity` with fresh risk state). Daily and weekly loss halts clear on their own at the next UTC day or ISO week.

The kill file path is `<repo>/data/KILL`. A backtest ignores it: backtests use a throwaway kill file so they stay reproducible.

## Stale data

A symbol is stale when its last candle download failed or its newest bar is older than 180 seconds (`paper.stale_after_seconds`). For a stale symbol the engine records a `stale_data` risk event, trades nothing in it and opens no new entries; an open position keeps only its stop. Each incident raises one alert, and another when it recovers. A missing bar therefore delays a decision by at most 180 seconds.

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

A long job (walk-forward, a download) and a paper session can contend for the SQLite file. Run them one at a time, or use Postgres.

## Monitoring

`paper status` exits 1 when an open run has not polled recently, so a cron job can watch for a hung or dead session. A generic example that runs every five minutes and sends mail on failure (use whatever notifier you have in place of `mail`):

```bash
*/5 * * * * cd /path/to/regimebot && .venv/bin/regimebot paper status >/dev/null || echo "regimebot paper run is stale or hung" | mail -s "regimebot" you@example.com
```

Exit code 0 means the latest run is healthy or has ended; non-zero means stale. Pair it with `ALERT_WEBHOOK_URL` for in-session alerts. Note: a run that has no open session but was never ended also counts as stale after `--max-age`.

## Docker Compose (untested on the development machine)

Docker was not available during development, so these steps are unverified.

```bash
export POSTGRES_PASSWORD=choose-one
docker compose up -d db
docker compose run --rm app db upgrade
docker compose run --rm app data download
docker compose run -d app paper run
```

Known gap: `docker-compose.yml` does not mount a shared `data/` volume. `docker compose run app kill` would create the kill file in a throwaway container and not reach the running session. With Compose, run `kill` through `docker compose exec` in the running session's container, or add a volume for `data/` to the `app` service. Reports written to `reports/` inside a container are also lost unless you mount that directory.

## API keys for any future live use

Not needed now. If live order placement is ever added (see ADR 0003):

- Create trade-only keys. Never enable withdrawals.
- Restrict the key to your IP address.
- Keep keys only in environment variables or a secret manager, never in git, config files or logs.
- Use the venue's testnet first.

## Known gaps

- Error reporting (Sentry) and uptime monitoring are not set up; add them before any real deployment.
- Structured logs have no request or trace IDs; they are single-process JSON lines.
