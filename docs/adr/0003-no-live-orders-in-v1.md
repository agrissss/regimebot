# ADR 0003: No live orders in version 1

Status: accepted

## Context

Version 1 exists to find out whether the strategies have an edge. The first honest results say they do not (see the README). Live trading adds failure modes that research does not have: real money, exchange rejections and outages, partial fills, API keys that can be stolen, and legal and regulatory exposure that depends on where the user lives and which venue is used. Code that can send an order is a liability even when switched off.

## Decision

- There is no order-placement code in the repository. The `Exchange` protocol exposes market-data methods only (`fetch_candles`, `fetch_funding`, `fetch_quote`), and the paper session calls nothing else. No API key is read or needed.
- Paper trading simulates every fill with the same fill model as the backtest.
- Live order placement may be added only when all of these hold:
  1. Paper trading has run for weeks with results that match the backtest's expectations, including costs and stale-data behaviour, and a strategy has shown an edge in out-of-sample testing.
  2. The adapter has run against the venue's testnet first.
  3. Live mode requires an explicit confirmation flag on every start; it is never the default and never implied by configuration alone.
  4. The venue's legality for the user's jurisdiction has been checked.
  5. Keys are trade-only, withdrawals disabled, IP-allowlisted and held only in environment variables or a secret manager.
  6. The enterprise baseline gaps in the README (error reporting, deployment separation) are closed for that deployment.

## Consequences

- Nothing in this version can lose real money through an order.
- The `Exchange` protocol and `Engine` feed boundary are where a live adapter would attach; adding one will need its own ADR and design review.
- A reader should not assume paper results transfer to live trading. Real fills differ from the model.
