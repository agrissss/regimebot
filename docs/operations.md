# Operations: paper trading runbook

Paper trading polls live public market data, runs the same engine as the backtest and simulates every fill. No order is ever sent and no API key is used. All commands run from the repository root. Paper trading needs `fcntl` for its per-run lock, so `paper run` works on macOS and Linux (or in Docker) only.

## Before you start

```bash
.venv/bin/regimebot db upgrade
.venv/bin/regimebot data check
```

`data check` validates the stored history for the backtest range plus the 31-day feature warm-up before it, so it is only needed for backtests. A paper session fetches its own warm-up history (`paper.warmup_bars`, default 3400 decision bars, about 35 days) on start. That value must be at least the features' warm-up (2,976 with defaults); config validation enforces it. If a symbol's features are still not fully valid at a decision, the session records a `warmup_incomplete` risk event, trades nothing in that symbol and sends a `warmup_incomplete` alert once per incident.

After pulling new code, run `regimebot db upgrade` first (migration 0002 adds bid and ask columns to fills). The config hash changed with that code (`paper.warmup_bars` and `risk.min_notional`), so a paper run opened before it cannot be resumed with the new default config: end it with `regimebot paper end RUN_ID` and start a new one.

The host clock must be synced (NTP). Decisions are keyed to bar close times, and a clock that is off will skip or mistime them. macOS and most Linux distributions do this by default; check with `timedatectl` (Linux) or `sntp -d time.apple.com` (macOS).

## Start a session

```bash
.venv/bin/regimebot paper run
```

The banner (stderr) shows the run id, symbols, poll interval, kill file path, how to stop and how to resume. Logs are JSON lines on stdout, so redirect them if you want a file:

```bash
.venv/bin/regimebot paper run >> data/paper.log
```

The session polls every 10 seconds (`paper.poll_seconds`) and decides once per closed 15-minute bar, when every symbol has that bar or is stale. Run it somewhere that stays on (a laptop that sleeps will go stale and skip decisions).

Paper trading has not yet completed a supervised 24-hour soak with a live decision, entry, exit, stop/resume and kill. Do that before leaving it unattended.

## Check on it

```bash
.venv/bin/regimebot paper status
```

Shows the latest paper run: `open` (running, or crashed or hung when `NOT POLLING` also appears), `stopped at <time>` after a clean stop (Ctrl-C or SIGTERM), with the resume command, or `ended` after `paper end`; the age of its last poll; equity and its timestamp, cash, the active halt (or `no`), open positions with stop and target, the last regime per symbol, the count of rejected polled bars per symbol (`rejected bars: ...`, bars that broke the OHLC rules and were dropped) and the last 10 risk events. It reads the database and works while a session runs. It exits 0 for a running run, a cleanly stopped run and an ended run. It exits 1 (printing `NOT POLLING`) for an open run with no clean-stop marker whose last poll is older than `--max-age` seconds (default 3 x `poll_seconds` + `stale_after_seconds`, 210 s): the process crashed or hung. See Monitoring.

## Stop and resume

Stop with Ctrl-C or SIGTERM. The session finishes the current poll, saves its state, records a clean-stop marker (`stopped_at`) and exits. Positions stay open and the run stays resumable; nothing is flattened. To finish a run for good, use `regimebot paper end RUN_ID`: it refuses while the run is running or holds open positions (unless `--force`, which closes nothing; the simulated positions stay in the last saved state). An ended run cannot be resumed, and `paper status` shows `ended`.

```bash
.venv/bin/regimebot paper run --resume 4
```

Use the run id from the banner or from `paper status`. `--resume latest` resumes the most recent open paper run, or starts a new one if none is open, which suits a supervisor that restarts the process (see Running unattended). It refuses, like `--resume N`, if the open run has a different config hash. Rules on resume:

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

- A fill after a decision can lag it by the quote fetches (each bounded at about 11 s: the paper exchange client uses 1 retry and a 5 s timeout). An entry not filled within one decision bar expires.
- Exchange requests are made per symbol in sequence, but the paper client uses 1 retry and a 5 s timeout (about 11 s per call), so a hanging symbol cannot stall a poll for minutes. History download (`data download`) keeps 5 retries.

## Alerts

Every alert is an ERROR log line. If `ALERT_WEBHOOK_URL` is set (http or https), the same alert is POSTed as JSON `{"kind", "detail", "ts"}` with a 5-second timeout:

```bash
export ALERT_WEBHOOK_URL="https://example.invalid/hook"   # replace with your endpoint
.venv/bin/regimebot paper run
```

A webhook failure is logged and never stops the session. The URL is never logged. Alert kinds raised by the paper session: `stale_data` and `stale_data_recovered` (per symbol), `poll_error` and `poll_error_recovered` (an exception during a poll; the next poll runs normally), `warmup_incomplete` and `warmup_incomplete_recovered` (per symbol, while its features are not fully valid). A polled bar that breaks the OHLC rules is dropped and logged as a `paper_bar_rejected` warning.

Every halt (`daily_loss`, `weekly_loss`, `max_drawdown`, `manual_kill`) sends an alert, and daily and weekly halts send a recovery alert when they clear. A skipped decision sends `decision_missed`. Halts are also visible in `paper status` and the logs (`risk_halt`).

## Where state lives

| What | Where |
|---|---|
| Candles, funding, runs, signals, decisions, fills, trades, equity, regimes, risk events | Database (`data/regimebot.db` by default, or `DATABASE_URL`), every row tagged with its `run_id` |
| Resumable paper state (positions, cash, risk state, cursors) | `open_positions` table, rewritten at the end of every poll |
| Kill file | `data/KILL` |
| Per-run lock | `data/paper_run_<id>.lock` |
| Reports and CSVs | `reports/` |
| Logs | stdout (JSON lines); redirects and rotation are up to you (see Logs) |

`data/` and `reports/` are not in git. Back up the database file while no session is running (or use `sqlite3 data/regimebot.db ".backup copy.db"`).

The long SQLite writers are `data download`, the final save of `backtest run`, and a paper session (which writes every poll). A second writer waits only about 5 seconds (SQLite's default busy timeout) and then fails with "database is locked". Walk-forward reads candles once and writes CSVs, so it is not a long writer. Run writers one at a time, or use Postgres.

## Logs

Logs are JSON lines on stdout and stderr (the banner and warnings go to stderr). Redirect them to a file and rotate it with the system tool; the program never rotates its own logs.

```bash
.venv/bin/regimebot paper run --resume latest >> data/paper.log 2>> data/paper.err
```

macOS (`/etc/newsyslog.d/regimebot.conf`; fields: path, mode, count, size in KB, when, flags). The process keeps the file open, so after a rotation restart the session (the supervisor below does this) or it keeps writing to the rotated file:

```
/path/to/regimebot/data/paper.log   644  7  10240  *  NJ
```

Linux (`/etc/logrotate.d/regimebot`; `copytruncate` is needed because the process keeps the file open):

```
/path/to/regimebot/data/paper.log {
    weekly
    rotate 8
    compress
    missingok
    copytruncate
}
```

## Running unattended

Run `regimebot paper run --resume latest` under a supervisor that restarts it on exit. Each restart resumes the open run, or starts one if none is open. Set the working directory to the repository root. Replace `/path/to/regimebot` and the paths below with your own.

macOS launchd (`~/Library/LaunchAgents/com.example.regimebot.plist`; load with `launchctl load ~/Library/LaunchAgents/com.example.regimebot.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.example.regimebot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/regimebot/.venv/bin/regimebot</string>
    <string>paper</string><string>run</string><string>--resume</string><string>latest</string>
  </array>
  <key>WorkingDirectory</key><string>/path/to/regimebot</string>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>/path/to/regimebot/data/paper.log</string>
  <key>StandardErrorPath</key><string>/path/to/regimebot/data/paper.err</string>
</dict>
</plist>
```

Linux systemd (`/etc/systemd/system/regimebot-paper.service`; enable with `systemctl enable --now regimebot-paper`):

```ini
[Unit]
Description=regimebot paper trading
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/path/to/regimebot
ExecStart=/path/to/regimebot/.venv/bin/regimebot paper run --resume latest
Restart=always
RestartSec=30
StandardOutput=append:/path/to/regimebot/data/paper.log
StandardError=append:/path/to/regimebot/data/paper.err

[Install]
WantedBy=multi-user.target
```

A supervisor restarts the process whatever the reason it exited, including a clean stop. The kill file still works: while `data/KILL` exists, the restarted session halts, stays flat and opens nothing. Ending a run needs `regimebot paper end RUN_ID`; without it, `--resume latest` keeps resuming the same run. Add `User=` to the systemd unit to run as an unprivileged user.

## How to stop everything

1. Stop the supervisor so it does not restart the process: `launchctl unload ~/Library/LaunchAgents/com.example.regimebot.plist` on macOS, `systemctl stop regimebot-paper` on Linux. Without a supervisor, press Ctrl-C or send SIGTERM; the session saves its state and exits.
2. To flatten open positions first, run `regimebot kill` while the session is still running and wait for the next poll (about 10 seconds); check with `regimebot paper status`.
3. To finish the run for good: `regimebot paper end RUN_ID` (add `--force` if it still holds open positions; that closes nothing).
4. `regimebot unkill` removes the kill file when you want to trade again; the run's halt stays until `paper clear-halt`.

## Monitoring

`paper status` exits 1 when an open run that was not stopped cleanly has not polled recently, so a cron job can watch for a hung or dead session. A generic example that runs every five minutes and sends mail on failure (use whatever notifier you have in place of `mail`):

```bash
*/5 * * * * cd /path/to/regimebot && .venv/bin/regimebot paper status >/dev/null || echo "regimebot paper run is stale or hung" | mail -s "regimebot" you@example.com
```

Exit code 0 means the latest run is polling, was stopped cleanly with Ctrl-C or SIGTERM, or has ended. Exit code 1 means an open run with no clean-stop marker has not polled within `--max-age`: it crashed or hung. Pair it with `ALERT_WEBHOOK_URL` for in-session alerts.

## Docker Compose

CI builds the image and runs `version` and `db upgrade` in it. Docker has still not been run on the author's own machine, so the Compose steps below and a paper session in a container have not been exercised there.

```bash
export POSTGRES_PASSWORD=$(openssl rand -hex 24)   # keep it: the db volume stores it on first start
docker compose up -d db
docker compose run --rm app db upgrade
docker compose run --rm app data download
docker compose run -d app paper run
```

The image installs with `requirements.lock` as a constraint (`psycopg` is pinned in it), has a `.dockerignore`, records the `GIT_SHA` build argument as the run's git sha, and ships `alembic.ini` and creates writable `data/` and `reports/`. Compose mounts named volumes at `/app/data` and `/app/reports`, shared by every `app` container, so `docker compose run --rm app kill` reaches a running session through the shared kill file, and reports persist. The `pgdata` volume holds the Postgres data; back it up (and test a restore) before it holds anything you cannot re-download.

## API keys for any future live use

Not needed now. If live order placement is ever added (see ADR 0003):

- Create trade-only keys. Never enable withdrawals.
- Restrict the key to your IP address.
- Keep keys only in environment variables or a secret manager, never in git, config files or logs.
- Use the venue's testnet first.

## Known gaps

- Error reporting (Sentry) is not set up; add it before any real deployment. Uptime monitoring is not set up; see Monitoring for a cron recipe to start from.
- Structured logs have no request or trace IDs; they are single-process JSON lines.

## Liquidity stress in backtests and paper

In backtests the spread is modelled as a constant, so the `liquidity_stress` regime can only be triggered there by the illiquidity percentile. Paper trading uses the live quoted spread as well. Results by regime are therefore not strictly comparable between the two.

## Known follow-ups

- Paper trading has not yet completed a supervised 24-hour soak (see Before you start).
- Raising `features.pctile_window` above about 2,900 in a backtest also requires raising `paper.warmup_bars`.
