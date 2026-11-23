#!/usr/bin/env bash
# Parallel paper sessions, one per config, each with its own database and log
# (docs/operations.md, "Parallel paper sessions"). macOS: uses caffeinate and `ps -E`.
#
#   scripts/paper_all.sh start    start every session that is not already running (safe to re-run)
#   scripts/paper_all.sh status   `regimebot paper status` for each config; exits 1 if any is hung
set -u
cd "$(dirname "$0")/.."

CONFIGS="kraken kraken_trend_longflat kraken_playbook_15m kraken_breakout_1h"

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
    REGIMEBOT_CONFIG="config/$name.yaml" nohup caffeinate -i \
      .venv/bin/regimebot paper run --resume latest >> "data/paper_$name.log" 2>&1 &
    echo "$name: started (pid $!), log data/paper_$name.log"
  done
}

status() {
  rc=0
  for name in $CONFIGS; do
    if pid=$(running "$name"); then proc="pid $pid"; else proc="no process"; fi
    echo "== $name (config/$name.yaml, $proc)"
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
