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
| `market_state` | Causal features, rule-based regime classifier, playbook |
| `strategy` | Breakout, trend continuation, mean reversion; shared filters |
| `risk` | Sizing, exposure and loss limits, kill switch |
| `execution` | Fill model (spread, slippage, fees, partial fills), funding |
| `engine` | Portfolio accounting and the shared event loop |
| `backtest` | Runner, walk-forward, sensitivity and stress runs, baselines |
| `reporting` | Metrics, charts, HTML and CSV reports, alerts |
| `paper` | Live polling feed, stale-data guard, paper session |
| `cli` | Typer commands |

## Time and causality

- Every timestamp is timezone-aware UTC. A bar's `ts` is its **close** time.
- Decisions happen on the close of a decision bar (15m). Fills happen no
  earlier than the open of the next base bar (1m).
- Features at row *t* use rows `<= t` only. A prefix-invariance test enforces
  this for every feature column.
- Longer horizons are expressed as longer windows on the decision frame, not as
  joined higher-timeframe bars. This removes the most common look-ahead bug.
- Inside one 1m bar, if both stop and target are touched, the stop is assumed
  to fill first.

## Storage

SQLAlchemy 2.0. The default URL is a local SQLite file; `DATABASE_URL` switches
to Postgres (the Compose file runs Postgres 16). Schema changes are Alembic
migrations. Every trading row carries a `run_id`, so backtest, paper and live
results never mix.

## Safety

- Risk decisions override strategy signals, and every decision is stored.
- A kill switch (automatic on limit breach, manual via `regimebot kill`)
  flattens positions and blocks new entries until cleared by hand.
- Stale or invalid data blocks new entries.
- Live order submission is not implemented in this version. See ADR 0003.

## Not in this version

Live order adapter, WebSocket feeds (REST polling is used), open-interest and
liquidation history, implied volatility, news and sentiment, web dashboard.
