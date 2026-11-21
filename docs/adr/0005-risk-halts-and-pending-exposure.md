# ADR 0005: Risk halts, pending exposure and entry expiry

Status: accepted

## Context

Risk limits protect capital only if they cannot be bypassed by timing, and a research run must show what the strategy does when the limits bite. Two timing gaps were considered: several entries approved in the same decision bar before any of them fills, and an order queued at one decision that is still unfilled at the next.

## Decision

- **Pending exposure.** Entries approved in the same decision bar count toward the per-asset, per-group and gross exposure caps before they fill. Without this, several symbols could each be sized against the same free room and together breach the caps.
- **Entry expiry.** A queued entry that has not filled before the next decision is cancelled (`expired`). A stale order is never carried into a changed market and regime.
- **Strategy priority.** When two strategies fire on the same bar for a symbol, the playbook's list order for the current regime decides which is taken.
- **Loss halts.** Daily (1.5%) and weekly (4%) loss limits halt trading and clear at the next UTC day or ISO week (Monday 00:00 UTC).
- **Sticky drawdown halt.** The 10% max-drawdown halt, and a manual kill, are permanent until cleared by hand. A halt flattens positions and blocks new entries. `regimebot unkill` removes the kill file but does not clear a recorded halt; `regimebot paper clear-halt RUN_ID` does, for a stopped run. A `max_drawdown` halt re-triggers while drawdown is still past the limit, so that run is over. Every halt sends an alert (see `docs/operations.md`).
- **Backtests keep the drawdown halt on by default**, so a backtest shows the behaviour of the system with its safety limit. To study a full period, run with `--set risk.max_drawdown=1.0`, the documented way. The `backtest run` output states when and why a run halted.

## Consequences

- With the default limits the first results run stopped on 2024-02-05 after 60 trades. A full-period study needs the override, and its result must be read as "what happens without the limit", not as achievable performance.
- Exposure caps hold even when many signals arrive together.
- Because the drawdown halt is sticky, a halted paper session stays halted across restarts until someone runs `clear-halt`; this is deliberate, and an operator is expected to look before trading resumes.
