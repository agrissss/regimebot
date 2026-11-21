# ADR 0004: Fill model conventions

Status: accepted

## Context

A backtest flatters a strategy whenever its fills are better than reality. The fill model (`regimebot/execution/fill_model.py`, used by `regimebot/engine/loop.py`) must be pessimistic where information is missing, so a surviving result is not an artefact of generous fills. Rules below apply to backtest and paper alike. In paper trading, queued orders fill at a fresh live quote (live spread, session clock), and participation uses the last closed base bar's volume.

## Decision

| Situation | Rule |
|---|---|
| Decision timing | Decisions happen on the close of a 15-minute bar. Orders fill no earlier than the open of the next 1-minute bar, never at the decision close. |
| Stop and target in one bar | If one 1-minute bar touches both, the stop fills first. |
| Entry size | Entry fills are capped at 5% of the fill bar's volume (`costs.max_participation`). A bar without volume gives no fill. |
| Exits | Exits always fill; they are never cut by the participation cap. Volume impact is capped at one full bar's volume, so impact is bounded and an exit is never cheaper on a thin bar. |
| Stop fills | A stop is a market order once triggered. A gap through the stop fills at the open. It pays the same half spread, slippage and bounded impact as a market exit on that bar, so a stop is never cheaper than a market exit. |
| Target fills | A target is a resting limit. A touch is not a fill: price must trade through it. If a bar opens beyond the target it still fills at the target price; no price improvement is assumed. Maker fee, no spread, no slippage. |
| Gap through the stop | An entry whose fill is at or beyond its stop is cancelled (`gapped_through_stop`): the trade would start already stopped out. |
| Minimum entry | An entry whose filled notional is under 10 USDT is cancelled (`too_small` at sizing, `no_liquidity` at fill time). |
| Costs | Taker 5 bps, maker 2 bps, half spread 1 bp (backtest, when no live quote), 2 bps slippage plus 1 bp per 1% of bar volume taken. Funding is charged from stored funding rates. |

Spread and slippage always move the price against the trader.

## Consequences

- Backtests understate results relative to an ideal venue and overstate nothing deliberately. Real execution can still be worse (queue position, outages, adverse selection on limits).
- A target hit only by a touch is not credited, which lowers win rates relative to a generous model.
- Changing any of these rules changes stored results; compare runs only with the same rules (the stored config hash does not cover code changes).
