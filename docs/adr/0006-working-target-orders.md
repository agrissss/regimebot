# ADR 0006: A daily target order may keep working over several minutes

Status: accepted (2026-11-24)

## Context

ADR 0004 caps a fill at 5% of its bar's volume (`costs.max_participation`) and gives each order one fill; the daily trend strategy then trades the rest at the next daily decision. On Kraken Futures the minute after 00:00 UTC is thin: on paper a 400 USD SOL order filled 18 USD on day one and reached 85 USD on day two. A real order would simply keep working.

## Decision

`strategies.daily_trend.fill_minutes` (default 1: ADR 0004 unchanged). Above 1, the part of a target order the cap cut keeps working, in backtest and paper alike (one `Engine`):

- It is offered a fill at the open of each later 1-minute bar within `fill_minutes` of the decision (paper: a fresh quote on the first poll after each new bar, with that bar's volume). Each slice is for what is left of the target the risk manager approved at the decision, never more, capped at 5% of its own bar and priced and costed like the first fill. A bar that cannot fill it (no volume, a slice under `risk.min_notional`) is skipped.
- It ends when the rest is worth less than `risk.min_notional` (done), with the window's last bar or an offer after it (the rest dropped, a `no_liquidity` decision row), on a halt or while the kill file exists (`kill_switch`), or at the next decision (`expired`, which trades from the current position).
- The caps are not evaluated again per slice: the slices add up to at most the order approved at the decision, which a single fill could have taken whole.
- Working orders are not saved: a paper restart drops the rest. Their fills are in the restored portfolio, so nothing fills twice.

## Consequences

- Filled turnover gets closer to what a worked order gets on a thin venue, without a cheaper fill: every slice pays the spread, slippage and impact of its own bar.
- Fills come up to `fill_minutes` after the decision, so price moves inside the window are part of the result, for better or worse.
- Default configs, their hashes and their stored results are unchanged; a config with `fill_minutes` above 1 has a different hash.
