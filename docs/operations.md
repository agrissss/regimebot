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

## Running the daily trend strategy

Set `REGIMEBOT_CONFIG=config/trend.yaml` for every command (backtest, report, paper, kill). After pulling this version run `regimebot db upgrade` once (migration 0003 lets trades without a stop, target or R-multiple be stored as empty rather than zero).

Paper: `regimebot paper run --resume latest` fetches about 130 days of 1-minute history per coin on first start, then decides once a day just after 00:00 UTC. Between decisions it only watches the kill file and, every hour, the loss and drawdown limits. A session that was down over midnight acts at the next daily close it sees. A machine that sleeps misses decisions; keep it awake (`caffeinate -s` on macOS) or run it on an always-on host. `regimebot paper status`, `regimebot kill` and `regimebot paper end` work as for the other strategies.

### On Kraken data

`config/kraken.yaml` is `config/trend.yaml` with `exchange.id: krakenfutures` and the linear perpetuals `BTC/USD:USD`, `ETH/USD:USD`, `SOL/USD:USD`. Use `REGIMEBOT_CONFIG=config/kraken.yaml` for every command. Funding on Kraken is hourly; the engine applies each stored funding event as it comes, so nothing else changes. The modelled 5 bps taker fee equals Kraken's entry-tier futures fee.

History: `regimebot data download --start 2026-06-01` (about 250,000 one-minute bars per coin, a few minutes). Only data from 2026-06-01 was downloaded, so the long backtests remain on Binance data.

Check made on 2026-11-22, same rule and dates (2026-10-12 to 2026-11-22) on both venues: Kraken +383.74 (+3.84%, 9 trades, fees 4.77, funding -3.78), Binance +392.23 (+3.92%, 9 trades, fees 4.55, funding -5.21). Six weeks say nothing about an edge; the point is that the two venues give nearly the same result for a once-a-day rule. Daily closes differed by about 5 bps on average, 19 bps at most.

Kraken is thinner at the fill minute (00:00 to 00:01 UTC). Over 175 days the 5% participation cap was worth under 1,000 USD on 10 days for BTC, 63 for ETH and 98 for SOL, so some orders fill in part and the rest waits for the next daily decision. In the six-week check that cut turnover by about 6% (9,534 against 10,106 with a 25% cap) and changed the result by about 4 USD. The cap is left at 5%: it is the pessimistic choice.

Working an order over several minutes: `strategies.daily_trend.fill_minutes` (default 1, one fill per decision as before; at most 1440). Above 1, the part of a target order the cap cut keeps working: it fills at the open of each following 1-minute bar (in paper: at a fresh quote on the first poll after each new bar, with that bar's volume), each slice capped at `costs.max_participation` of its bar and priced and costed like the first fill, until the decision's approved target is reached or `fill_minutes` have passed since the decision. Then the rest is dropped with a `no_liquidity` decision row ("... left after the N-minute fill window"). A rest worth less than `risk.min_notional` ends the order as done. A halt or the kill file stops it (`kill_switch` row), and the next decision replaces an order still working (`expired`). Paper restart: working orders are not saved, so a stop or crash in the window drops the rest (a clean stop records `cancelled`, a hard kill records nothing); what was filled is in the restored position, so nothing fills twice, and the next decision trades from there. Check on Kraken, 2026-10-12 to 2026-11-22 (in-sample, the same history as above): `fill_minutes` 1 filled 9,534.39 of turnover, net +383.74; `fill_minutes` 30 filled 10,106.98, net +379.90 (fees 5.05, funding -3.89). One cut order was completed a minute later; nothing was left unfilled.

## Parallel paper sessions

Rule sets paper-trade side by side on Kraken Futures public data so they can be compared. Each config has its own SQLite database (one file per session, because two sessions writing one SQLite file fail with "database is locked") and its own log. Each starts from 10,000 with the default risk limits. The first four have run since 2026-11-23; the six daily rules below them were added on 2026-11-24 (same symbols, costs and risk limits as `config/kraken.yaml`, and `fill_minutes: 30`: see "On Kraken data").

| Config | Rule set | Database | Log | Run id |
|---|---|---|---|---|
| `config/kraken.yaml` | daily trend, long/short | `data/regimebot.db` | `data/paper_kraken.log` | 31 |
| `config/kraken_trend_longflat.yaml` | daily trend, long/flat | `data/paper_kraken_trend_longflat.db` | `data/paper_kraken_trend_longflat.log` | 1002 |
| `config/kraken_playbook_15m.yaml` | the original 15-minute playbook (mean reversion, trend, breakout) | `data/paper_kraken_playbook_15m.db` | `data/paper_kraken_playbook_15m.log` | 2002 |
| `config/kraken_breakout_1h.yaml` | breakout only, 1-hour decisions (`features.pctile_window` 720) | `data/paper_kraken_breakout_1h.db` | `data/paper_kraken_breakout_1h.log` | 3002 |
| `config/kraken_trend_ls_fill.yaml` | daily trend, long/short, exactly as `config/kraken.yaml` but with `fill_minutes: 30`: its twin, to show what working an order over 30 minutes changes | `data/paper_kraken_trend_ls_fill.db` | `data/paper_kraken_trend_ls_fill.log` | placeholder 4000 |
| `config/kraken_trend_fast.yaml` | daily trend, long/short, SMA lookbacks 10, 20 and 50 days | `data/paper_kraken_trend_fast.db` | `data/paper_kraken_trend_fast.log` | placeholder 5000 |
| `config/kraken_trend_slow.yaml` | daily trend, long/short, SMA lookbacks 50, 100 and 200 days (230 days of warm-up) | `data/paper_kraken_trend_slow.db` | `data/paper_kraken_trend_slow.log` | placeholder 6000 |
| `config/kraken_donchian.yaml` | daily Donchian, long/short: per lookback (20, 50, 100 days) +1 above the midpoint of the highest and lowest close, -1 below, averaged | `data/paper_kraken_donchian.db` | `data/paper_kraken_donchian.log` | placeholder 7000 |
| `config/kraken_momentum.yaml` | daily time-series momentum, long/short: the sign of the 20, 50 and 100-day returns, averaged | `data/paper_kraken_momentum.db` | `data/paper_kraken_momentum.log` | placeholder 8000 |
| `config/kraken_hold.yaml` | baseline: always long, with the same volatility-targeted size and band, long/flat | `data/paper_kraken_hold.db` | `data/paper_kraken_hold.log` | placeholder 9000 |

The first run of each new database gets the id after its placeholder (4001, 5001, ...), unless something else (a backtest) runs there first.

**Warning.** The playbook and the 1-hour breakout rule sets LOST money in backtests: the playbook lost 45% over 2024-2025 (drawdown halt off), and the 1-hour breakout failed its pre-registered 2021-2023 confirmation test (-9.12%). They run on paper for comparison only. The six daily rules added on 2026-11-24 are comparisons of how different rules behave on paper, not candidates with an edge: none of them is pre-registered or confirmed, and their backtests (docs/research-protocol.md, "More daily rules for paper comparison (2026-11-24)") are in-sample on history used many times before. The hold baseline stays long through every fall: in its backtest it reached the permanent 10% drawdown halt in May 2022 and traded no more, and its paper session will stop the same way. One night, one week or a few months of results says nothing about an edge, for any of the ten.

```bash
scripts/paper_all.sh start    # start each session that is not running; safe to re-run
scripts/paper_all.sh status   # paper status for each, with its process id; exits 1 if one is hung
```

`start` launches each missing session like a single one (a config whose database is not set up yet, see below, is skipped with a message, so the script stays safe to re-run): `REGIMEBOT_CONFIG=config/<name>.yaml nohup caffeinate -i .venv/bin/regimebot paper run --resume latest >> data/paper_<name>.log 2>&1 &`. It recognises a running session by the `REGIMEBOT_CONFIG` value in the process environment, so start sessions with the script or with exactly that value. Set the same `REGIMEBOT_CONFIG` for every other command about a session (`paper status`, `paper end`, `report`). A backtest with one of these configs writes to that session's database, so run it while the session is stopped.

Two things are shared by every session, whatever the config, because the paper code keys them to the repository rather than to the database:

- **The kill file.** `regimebot kill` creates `<repo>/data/KILL`, which halts and flattens every running session, run 31 included; each halted run then needs its own `regimebot paper clear-halt RUN_ID` under its own config. To stop one session, send it SIGTERM instead: `kill <pid>`, with the pid from `scripts/paper_all.sh status`. It stops cleanly and `start` resumes it.
- **The run locks.** A run's lock is `data/paper_run_<id>.lock` and run ids are numbered per database, so two running sessions with the same run id block each other (the second refuses to start, "already running"). Each new database therefore holds one placeholder row in `runs` (mode `reserved`, ids 1000, 2000 and 3000), and its runs are numbered after it.

Adding another session: copy a config, give it its own `database_url`, add its name to `CONFIGS` in `scripts/paper_all.sh`, then once:

```bash
export REGIMEBOT_CONFIG=config/<name>.yaml
.venv/bin/regimebot db upgrade
sqlite3 data/paper_<name>.db "INSERT INTO runs (id, mode, config_hash, config_json, git_sha, started_at, ended_at, note) VALUES (4000, 'reserved', '', '{}', '', datetime('now'), datetime('now'), 'run id offset: paper run locks are shared across databases')"
.venv/bin/regimebot data download --start 2026-06-01 --end "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"   # optional
```

The download is optional (the session fetches its own warm-up on first start) but is needed for a backtest.

Setting up the six daily-rule databases of 2026-11-24, once, with the run-id offsets of the table (4000 to 9000), before `scripts/paper_all.sh start`:

```bash
for pair in kraken_trend_ls_fill:4000 kraken_trend_fast:5000 kraken_trend_slow:6000 \
            kraken_donchian:7000 kraken_momentum:8000 kraken_hold:9000; do
  name=${pair%%:*}; id=${pair##*:}
  REGIMEBOT_CONFIG=config/$name.yaml .venv/bin/regimebot db upgrade
  sqlite3 data/paper_$name.db "INSERT INTO runs (id, mode, config_hash, config_json, git_sha, started_at, ended_at, note) VALUES ($id, 'reserved', '', '{}', '', datetime('now'), datetime('now'), 'run id offset: paper run locks are shared across databases')"
done
```

`config/kraken_trend_slow.yaml` needs 201 daily closes before its first decision, so its `paper.warmup_bars` is 230: on first start the session fetches 230 days of 1-minute history per coin itself (about 332 requests of 1000 bars per coin, several minutes; Kraken Futures served 1-minute bars from 2026-03-01 when checked on 2026-11-24). For a backtest of it on Kraken data, download from an earlier start than the other configs: `REGIMEBOT_CONFIG=config/kraken_trend_slow.yaml .venv/bin/regimebot data download --start 2026-04-01 --end "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"` (a backtest starting 2026-10-20 or later is then warm). The others need 110 days (`config/kraken_hold.yaml` only 31).
