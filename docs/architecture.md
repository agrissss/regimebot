# regimebot architecture

A research system that tests whether a rule-based, regime-aware strategy has an
edge in crypto perpetuals after costs. It is not a profit machine. The most
likely honest result is "no reliable edge", and the system is built to say so.

## Shape

One Python package, one process, one event loop. No message broker.

```
            +-------------+        +-----------+
 candles -> |  features   | -----> |  regime   | --+
            +-------------+        +-----------+   |
                                                   v
 +---------+   Signal   +--------+  RiskDecision  +----------+  Fill  +-----------+
 | strategy| ---------> |  risk  | -------------> | executor | -----> | portfolio |
 +---------+            +--------+                +----------+        +-----------+
                                                                           |
                                                             db (runs, signals, fills, ...)
```

The `Engine` owns the loop. Two parts are swappable:

| Part | Backtest | Paper | Live (later) |
|---|---|---|---|
| Bar source | candles from the database | polled from the exchange | same as paper |
| Fill pricing | `fill_model` on the next 1m bar | `fill_model` on the live quote | exchange |

Strategy, risk, portfolio and persistence code is identical in every mode.

## Modules

| Module | Responsibility |
|---|---|
| `config` | Pydantic models, YAML loading, env secrets |
| `types` | Frozen dataclasses and enums shared by all modules |
| `db` | SQLAlchemy models, Alembic migrations, small repository functions |
| `data` | Exchange protocol, ccxt adapter, history loader, validation, resampling |
| `market_state` | Causal features, rule-based regime classifier, playbook, regime study |
| `strategy` | Breakout, trend continuation, mean reversion; shared filters |
| `risk` | Sizing, exposure and loss limits, kill switch |
| `execution` | Fill model (spread, slippage, fees, partial fills), funding |
| `engine` | Portfolio accounting and the shared event loop |
| `backtest` | Runner, walk-forward, sensitivity and stress runs, baselines |
| `reporting` | Metrics, charts, HTML and CSV reports, alerts |
| `paper` | Live polling feed, stale-data guard, paper session, resumable state |
| `cli` | Typer commands |

## Time and causality

- Every timestamp is timezone-aware UTC. A bar's `ts` is its **close** time.
- Decisions happen on the close of a decision bar (15m). Fills happen no
  earlier than the open of the next base bar (1m).
- Features at row *t* use rows `<= t` only. A prefix-invariance test enforces
  this for every feature column.
- The percentile features need a full 30-day window: the classifier answers
  `no_trade` until `warmup_bars` decision bars exist (2,976 by default), and
  history is loaded from that span before `backtest.start`.
- Longer horizons are expressed as longer windows on the decision frame, not as
  joined higher-timeframe bars. This removes the most common look-ahead bug.
- Backtests model the spread as a constant; paper uses the live quoted spread, so the `liquidity_stress` regime is triggered only by the illiquidity percentile in backtests and results by regime are not strictly comparable between the two.
- Inside one 1m bar, if both stop and target are touched, the stop is assumed
  to fill first. Fill conventions are in ADR 0004.

## Storage

SQLAlchemy 2.0. The default URL is a local SQLite file; `DATABASE_URL` switches
to Postgres (the Compose file runs Postgres 16). Schema changes are Alembic
migrations. Every trading row carries a `run_id`, so backtest, paper and live
results never mix.

## Safety

- Risk decisions override strategy signals, and every decision is stored.
- Loss limits halt trading: daily and weekly halts clear at the next UTC day or
  ISO week; the 10% max-drawdown halt and the manual kill (`regimebot kill`)
  are sticky. A halt flattens positions and blocks new entries. `regimebot paper clear-halt`
  clears a stopped run's recorded halts (a drawdown halt re-triggers by
  design). A running session acts on the kill file at every poll. See ADR 0005.
- Entries approved in the same decision bar count toward exposure caps before
  they fill, and a queued entry expires if unfilled at the next decision.
- Stale or invalid data blocks new entries.
- In paper trading, a symbol whose features are not fully valid at a decision is recorded (`warmup_incomplete`), alerted once per incident and not traded.
- Live order submission is not implemented in this version. See ADR 0003.

## Not in this version

Live order adapter, WebSocket feeds (REST polling is used), open-interest and
liquidation history, implied volatility, news and sentiment, web dashboard.
The full list of what the original brief asked for and was deliberately left out,
with reasons, and the enterprise items that do not apply to a single-user local
tool, are in the README sections
[What the original brief asked for that is deliberately not built](../README.md#what-the-original-brief-asked-for-that-is-deliberately-not-built)
and [Not applicable](../README.md#not-applicable).

See also: `README.md`, `docs/operations.md`, `docs/research-protocol.md`, `docs/adr/`.
