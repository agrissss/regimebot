#!/usr/bin/env bash
# Parallel paper sessions, one per config, each with its own database and log
# (docs/operations.md, "Parallel paper sessions"). macOS: uses caffeinate and `ps -E`.
#
#   scripts/paper_all.sh start    start every session that is not already running (safe to re-run)
#   scripts/paper_all.sh status   `regimebot paper status` for each config; exits 1 if any is hung
#
# A config whose database is not set up yet (migrated, with its run-id placeholder row: see
# "Adding another session" in the docs) is skipped by both, so a new name can be listed early.
set -u
cd "$(dirname "$0")/.."

CONFIGS="kraken kraken_trend_longflat kraken_playbook_15m kraken_breakout_1h
  kraken_trend_ls_fill kraken_trend_fast kraken_trend_slow kraken_donchian kraken_momentum
  kraken_hold"

# True when config/<name>.yaml's SQLite database exists and holds at least one run (a placeholder
# or earlier runs), so a new run cannot take an id another session's lock already uses.
# Read-only: never creates the file.
ready() {
  db=$(sed -n 's|^database_url: sqlite:///||p' "config/$1.yaml")
  runs=$(sqlite3 -readonly "$db" "SELECT count(*) FROM runs" 2>/dev/null) || return 1
  [ "${runs:-0}" -gt 0 ]
}

# Pid of the `paper run` process started with REGIMEBOT_CONFIG=config/<name>.yaml, if any. The
# command lines are identical, so the config is read from the process environment.
running() {
  for pid in $(pgrep -f "regimebot paper run"); do
    if ps -E -p "$pid" -o command= | tr ' ' '\n' | grep -qx "REGIMEBOT_CONFIG=config/$1.yaml"; then
      echo "$pid"
      return 0
    fi
  done
  return 1
}

start() {
  for name in $CONFIGS; do
    if pid=$(running "$name"); then
      echo "$name: already running (pid $pid), skipped"
      continue
    fi
    if ! ready "$name"; then
      echo "$name: database not set up (docs/operations.md, \"Parallel paper sessions\"), skipped"
      continue
    fi
    REGIMEBOT_CONFIG="config/$name.yaml" nohup caffeinate -i \
      .venv/bin/regimebot paper run --resume latest >> "data/paper_$name.log" 2>&1 &
    echo "$name: started (pid $!), log data/paper_$name.log"
  done
}

status() {
  rc=0
  for name in $CONFIGS; do
    if pid=$(running "$name"); then proc="pid $pid"; else pid=""; proc="no process"; fi
    echo "== $name (config/$name.yaml, $proc)"
    if ! ready "$name"; then
      if [ -n "$pid" ]; then
        echo "session running, but its database fails the set-up check (no runs table or no rows)"
        rc=1
      else
        echo "database not set up yet: no session"
      fi
      echo
      continue
    fi
    REGIMEBOT_CONFIG="config/$name.yaml" .venv/bin/regimebot paper status || rc=1
    echo
  done
  return "$rc"
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  *) echo "usage: $0 start|status" >&2; exit 2 ;;
esac
