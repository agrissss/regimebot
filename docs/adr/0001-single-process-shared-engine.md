# ADR 0001: Single process, one shared engine for backtest and paper

Status: accepted

## Context

The system must answer one question: does a rule-based strategy have an edge after costs? A backtest that runs different code from the paper session cannot answer it, because any difference between the two (sizing, risk checks, fill order) becomes an untested gap. The system also has one user and no throughput requirement that a single process cannot meet.

## Decision

- One Python package, one process, one event loop. No message broker, no microservices.
- A single `Engine` (`regimebot/engine/loop.py`) owns the sequence decision, risk, queued orders, fills, stops and targets, funding. A feed drives it through `on_decision`, `fill_queued`, `on_minute` and `on_funding`.
- The backtest runner replays stored candles into the engine; the paper session replays polled candles into the same engine. Strategy, risk, portfolio and persistence code are identical in both modes.
- Longer horizons are longer windows on the 15-minute decision frame, not joined higher-timeframe candles. This removes the most common look-ahead bug (a higher-timeframe bar that is not yet closed).
- Market data comes from REST polling, not WebSockets. A 15-minute decision cadence does not need streaming, and polling needs no connection state, heartbeat handling or reconnect logic.
- Open interest, liquidations, implied volatility, news and sentiment are not used. The data is unavailable as long, clean history from public endpoints or is hard to align causally.

## Consequences

- Backtest results describe the code that paper trading runs. A fix to the engine changes both.
- The engine must stay free of mode-specific branches; the differences live in the feed (bar source and quotes).
- Polling adds up to one poll interval (10 s by default) of delay to a decision, and cannot see intra-bar events that a stream would.
- Scaling out is not possible without redesign. That is acceptable for a research tool; revisit only with a measured reason.
