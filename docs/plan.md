# regimebot MVP implementation plan

Spec: `docs/architecture.md` (read it first; it is short).

## Global Constraints

These bind every task.

- Python 3.11+ syntax; the venv is `.venv` (Python 3.12). Run tools as
  `.venv/bin/pytest`, `.venv/bin/ruff check .`, `.venv/bin/ruff format .`,
  `.venv/bin/mypy regimebot`. All three must pass with zero errors and zero
  warnings before you commit. Do not add `# type: ignore` or `noqa` without a
  reason on the same line.
- Installed versions: pandas 3.1, numpy 2.5, SQLAlchemy 2.1, pydantic 2.14,
  ccxt 4.5.64. These are newer than most examples you remember. If an API call
  warns or fails, check the installed package, not memory. Do not add
  dependencies; if you believe one is needed, report NEEDS_CONTEXT.
- Tests never touch the network and never touch `data/`. Use `tmp_path` and
  in-memory or temp-file SQLite. Tests are deterministic (seeded RNG).
- Every timestamp is a timezone-aware UTC `datetime` (or a UTC
  `DatetimeIndex`). A bar's `ts` is its close time. Naive datetimes are a bug.
- No look-ahead: a value used for a decision at time *t* may depend only on
  data with `ts <= t`.
- Money path arithmetic is plain `float`; compare in tests with
  `pytest.approx`.
- Shared types live in `regimebot/types.py` and config models in
  `regimebot/config.py`. Do not redefine them elsewhere. If a task needs a
  field that does not exist, add it there and say so in your report.
- Logging: `structlog.get_logger(__name__)`, event name plus key-values. Never
  log secrets. No `print` outside `cli.py`.
- Readability first: small functions, clear names, docstrings on public
  functions that say what and why. No abstractions with a single
  implementation unless this plan names them. No speculative features.
- Deliberate simplifications that cut a real corner get a comment starting
  `# ponytail:` naming the ceiling and the upgrade path.
- Commit on the current branch (`build/mvp`) with a clear message. Do not
  push.

## Quality gate

Every task is scored 1-10 by an independent reviewer on: correctness, tests,
readability, safety (causality, risk, money handling) and fit to this plan.
A task is accepted at 8 or higher; tasks marked **money path** need 9. Ruff,
mypy and pytest passing is required regardless of score.

## Shared contracts

`regimebot/types.py` (Task 1 creates it exactly as below; all are
`@dataclass(frozen=True, slots=True)` unless stated):

```python
class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Regime(StrEnum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    LIQUIDITY_STRESS = "liquidity_stress"
    EVENT_RISK = "event_risk"
    NO_TRADE = "no_trade"


class Mode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"


class ExitReason(StrEnum):
    STOP = "stop"
    TARGET = "target"
    TRAIL = "trail"
    TIME = "time"
    REGIME = "regime"
    KILL = "kill"
    END_OF_DATA = "end_of_data"


class Bar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Quote:
    symbol: str
    ts: datetime
    bid: float
    ask: float


class Signal:
    strategy: str
    symbol: str
    ts: datetime
    side: Side
    entry: float  # reference price: decision-bar close
    stop: float
    target: float
    atr: float  # ATR at signal time, price units
    regime: Regime
    max_hold_bars: int  # decision bars
    trail_atr_mult: float | None  # None = no trailing stop


class RiskDecision:
    approved: bool
    qty: float  # qty in base units, 0.0 when rejected
    rule: str  # machine name of the rule that decided, "ok" when approved unchanged
    detail: str


class Fill:
    symbol: str
    ts: datetime
    side: OrderSide
    qty: float
    price: float  # executed price
    ref_price: float  # intended price (decision close, stop level or target level)
    fee: float  # quote currency, positive = cost
    slippage_bps: float  # signed, positive = worse than ref_price
    liquidity: str  # "taker" | "maker"


class Trade:  # one closed position
    symbol: str
    strategy: str
    side: Side
    regime: Regime
    qty: float
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    initial_stop: float
    target: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    exit_reason: ExitReason
    r_multiple: float  # net_pnl / (qty * |entry_price - initial_stop|)
    bars_held: int


class RiskEvent:
    ts: datetime
    kind: str
    detail: str
    action: str  # what the system did, e.g. "halt", "flatten", "skip_symbol"
```

`side_sign(side: Side) -> int` returns +1 for LONG and -1 for SHORT.

## Tasks

### Task 1: Config, shared types, logging, project tooling

Model tier: standard.

Create:

- `regimebot/types.py` exactly as in Shared contracts, plus `side_sign`.
- `regimebot/config.py`: Pydantic v2 models (`model_config =
  ConfigDict(extra="forbid", frozen=True)`) and
  `load_config(path: Path | None = None, overrides: Mapping[str, object] | None = None) -> AppConfig`.
  `path=None` loads `config/default.yaml`. `overrides` maps dotted paths
  (`"risk.risk_per_trade"`) to values and is applied before validation; an
  unknown path raises `ValueError`. The env var `DATABASE_URL`, when set,
  replaces `database_url`. Also `config_hash(cfg) -> str` (sha256 of the
  canonical JSON dump, first 16 hex chars).
- `config/default.yaml` with exactly these sections and defaults:

```yaml
database_url: sqlite:///data/regimebot.db
exchange:
  id: binanceusdm
  symbols: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
  base_timeframe: 1m
  decision_timeframe: 15m
costs:
  taker_fee_bps: 5.0
  maker_fee_bps: 2.0
  half_spread_bps: 1.0         # used when no live quote exists (backtest)
  slippage_bps: 2.0            # fixed part, applied to taker fills
  impact_bps_per_pct: 1.0      # extra bps per 1% of bar volume taken
  max_participation: 0.05      # entry fills capped at this share of bar volume
features:
  atr_window: 14
  rv_window: 96
  pctile_window: 2880
  adx_window: 14
  ema_fast: 96
  ema_slow: 384
  volume_window: 96
  donchian_window: 96
  zscore_window: 96
regime:
  expansion_pctile: 0.90
  compression_pctile: 0.20
  adx_trend: 25.0
  adx_range: 20.0
  shock_z: 6.0
  illiq_pctile: 0.99
  max_spread_bps: 10.0
  event_windows: []            # list of {start: ISO UTC, end: ISO UTC, label: str}
playbook:                      # one entry per Regime value
  trend_up:         {strategies: [trend, breakout], size_mult: 1.0, allow_entries: true,  force_close: false}
  trend_down:       {strategies: [trend, breakout], size_mult: 1.0, allow_entries: true,  force_close: false}
  range:            {strategies: [mean_reversion],  size_mult: 0.5, allow_entries: true,  force_close: false}
  compression:      {strategies: [breakout],        size_mult: 1.0, allow_entries: true,  force_close: false}
  expansion:        {strategies: [breakout],        size_mult: 0.5, allow_entries: true,  force_close: false}
  liquidity_stress: {strategies: [],                size_mult: 0.0, allow_entries: false, force_close: true}
  event_risk:       {strategies: [],                size_mult: 0.0, allow_entries: false, force_close: false}
  no_trade:         {strategies: [],                size_mult: 0.0, allow_entries: false, force_close: false}
strategies:
  breakout:
    enabled: true
    min_target_pct: 0.05
    min_rr: 2.5
    stop_atr_mult: 1.5
    max_stop_pct: 0.03
    volume_z_min: 1.0
    compression_lookback: 24   # bars; breakout must follow compression within this many bars
    max_hold_bars: 288
    trail_atr_mult: 3.0
    plausibility_mult: 1.5
  trend:
    enabled: true
    min_target_pct: 0.05
    min_rr: 2.5
    stop_atr_mult: 0.5         # buffer beyond the pullback extreme
    max_stop_pct: 0.03
    pullback_lookback: 12
    pullback_atr_dist: 0.5     # pullback must come within this many ATR of ema_fast
    volume_z_min: 0.0
    max_hold_bars: 384
    trail_atr_mult: 3.0
    plausibility_mult: 1.5
  mean_reversion:
    enabled: true
    min_target_pct: 0.0        # targets the rolling mean, not a 5% move
    min_rr: 2.0
    entry_z: 2.5
    stop_atr_mult: 1.0
    max_stop_pct: 0.02
    max_hold_bars: 48
    trail_atr_mult: null
    plausibility_mult: 1.5
risk:
  risk_per_trade: 0.005
  max_gross_leverage: 1.5
  max_asset_exposure: 0.75
  max_group_exposure: 1.0
  groups: {majors: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]}
  daily_loss: 0.015
  weekly_loss: 0.04
  max_drawdown: 0.10
  streak_half: 3
  streak_stop: 5
backtest:
  initial_equity: 10000.0
  start: "2024-01-01T00:00:00Z"
  end: "2026-01-01T00:00:00Z"
walk_forward:
  train_days: 365
  test_days: 90
  min_trades: 20
  grid:
    strategies.breakout.volume_z_min: [0.5, 1.0, 1.5]
    strategies.breakout.stop_atr_mult: [1.0, 1.5, 2.0]
paper:
  poll_seconds: 10
  stale_after_seconds: 180
  warmup_bars: 3000            # decision bars loaded before trading starts
```

  Validators: every playbook key is a `Regime` value and all eight are
  present; strategy names in playbooks are one of `breakout`, `trend`,
  `mean_reversion`; `max_gross_leverage <= 1.5` (hard cap, a larger value is a
  validation error); fractions are in (0, 1]; windows are positive ints;
  `ema_fast < ema_slow`; timeframes are one of `1m 5m 15m 1h 4h`.
  `timeframe_minutes(tf: str) -> int` lives here.
- `regimebot/logging.py`: `configure_logging(json: bool = True, level: str = "INFO")`
  setting structlog to emit JSON lines with ISO UTC timestamps.
- `regimebot/cli.py`: a Typer `app` with a `version` command only (later tasks
  add commands).
- `.env.example` (`DATABASE_URL=`, `ALERT_WEBHOOK_URL=`; no exchange keys are
  needed in this version, say so in a comment).
- `Dockerfile` (python:3.12-slim, installs the package, entrypoint
  `regimebot`) and `docker-compose.yml` (services `db`: postgres:16 with a
  named volume and healthcheck; `app`: the image with
  `DATABASE_URL=postgresql+psycopg://...@db/regimebot`, installs the
  `postgres` extra). These cannot be run on this machine (no Docker); keep
  them minimal and conventional.
- `.github/workflows/ci.yml`: on pull_request and push to main; Python 3.12;
  `pip install -e ".[dev]"`; ruff check, ruff format --check, mypy, pytest.
- `.github/dependabot.yml` for pip and github-actions, weekly.

Tests (`tests/test_config.py`): default config loads; overrides work and
unknown path raises; leverage above 1.5 is rejected; missing playbook regime
is rejected; `config_hash` is stable and changes with an override;
`DATABASE_URL` env override works (use `monkeypatch`).

### Task 2: Database layer

Model tier: standard.

Create `regimebot/db/models.py`, `regimebot/db/session.py`,
`regimebot/db/repo.py`, Alembic setup (`alembic.ini`, `regimebot/db/migrations/`)
with one initial migration, and a CLI command `regimebot db upgrade` that runs
Alembic to head against `cfg.database_url`.

Tables (SQLAlchemy 2.0 typed `Mapped[]` models; all timestamps
`DateTime(timezone=True)`; must work on SQLite and Postgres):

| Table | Columns |
|---|---|
| `runs` | id PK, mode, config_hash, config_json, git_sha, started_at, ended_at nullable, note |
| `candles` | symbol, timeframe, ts, open, high, low, close, volume; PK (symbol, timeframe, ts) |
| `funding_rates` | symbol, ts, rate; PK (symbol, ts) |
| `regimes` | run_id FK, symbol, ts, regime, inputs_json; PK (run_id, symbol, ts) |
| `signals` | id PK, run_id FK, strategy, symbol, ts, side, entry, stop, target, atr, regime |
| `risk_decisions` | id PK, run_id FK, signal_id FK nullable, ts, approved, qty, rule, detail |
| `fills` | id PK, run_id FK, symbol, ts, side, qty, price, ref_price, fee, slippage_bps, liquidity |
| `trades` | id PK, run_id FK, plus every field of `types.Trade` |
| `open_positions` | run_id FK, symbol, state_json, updated_at; PK (run_id, symbol) |
| `risk_events` | id PK, run_id FK, ts, kind, detail, action |
| `equity_snapshots` | run_id FK, ts, equity, gross_exposure, drawdown; PK (run_id, ts) |

Index every `run_id` FK and `(symbol, ts)` on signals, fills and trades.

`session.py`: `make_engine(url) -> Engine` (creates the parent directory for a
SQLite file URL; enables SQLite foreign keys and WAL via a connect event) and
`session_scope(engine)` context manager (commit on success, rollback on
error).

`repo.py` (plain functions taking a `Session`):

- `upsert_candles(session, symbol, timeframe, df) -> int`: `df` has a UTC
  `DatetimeIndex` named `ts` and columns open, high, low, close, volume.
  Idempotent (re-inserting the same rows changes nothing; a changed row is
  updated). Returns rows written. Must be efficient for 1M rows (bulk
  statements in chunks, not one statement per row).
- `load_candles(session, symbol, timeframe, start=None, end=None) -> pd.DataFrame`:
  same shape as above, sorted, float64 columns, UTC index, `start <= ts < end`.
- `upsert_funding(session, symbol, series) -> int` and
  `load_funding(session, symbol, start=None, end=None) -> pd.Series`.
- `create_run(session, mode, cfg, note="") -> int`, `end_run(session, run_id)`.
  `git_sha` comes from `git rev-parse --short HEAD`, or `"unknown"` on failure.
- `save_trades`, `save_fills`, `save_signals`, `save_risk_decisions`,
  `save_risk_events`, `save_regimes`, `save_equity` (bulk inserts from lists
  of the `types` dataclasses or simple tuples; choose the simplest consistent
  signature and document it), and `load_trades(session, run_id) -> list[Trade]`,
  `load_equity(session, run_id) -> pd.Series`.
- `save_open_position(session, run_id, symbol, state: dict | None)` (None
  deletes) and `load_open_positions(session, run_id) -> dict[str, dict]`.

Tests: round-trip for candles (including idempotent re-insert, update of a
changed row, range filtering, tz-awareness of the returned index), funding,
runs, trades (dataclass equality after round-trip), open positions;
the Alembic migration creates the same tables as the models
(`alembic upgrade head` on a temp SQLite file, then compare table and column
names to `Base.metadata`).

### Task 3: Market data: exchange adapter, history loader, validation

Model tier: standard.

Create `regimebot/data/exchange.py`, `regimebot/data/history.py`,
`regimebot/data/validate.py`, `regimebot/data/resample.py`, and CLI commands.

- `exchange.py`: `class Exchange(Protocol)` with
  `fetch_candles(symbol, timeframe, since: datetime, limit: int) -> list[Bar]`
  (closed bars only, ascending, `Bar.ts` = close time),
  `fetch_funding(symbol, since: datetime, limit: int) -> list[tuple[datetime, float]]`,
  `fetch_quote(symbol) -> Quote`. `class CcxtExchange` implements it over the
  synchronous ccxt client for `cfg.exchange.id` with `enableRateLimit=True`
  and no API keys. ccxt returns the bar **open** time in ms; convert to close
  time and drop the still-forming last bar (close time in the future).
  Network errors (`ccxt.NetworkError`) are retried up to 5 times with
  exponential backoff, then raised.
- `history.py`: `download_candles(exchange, session, symbol, timeframe, start, end) -> int`
  pages forward from `start` (or from the last stored bar when resuming) to
  `end`, upserting each page; safe to re-run and resumes where it stopped.
  `download_funding(...)` likewise. Log progress every page.
- `validate.py`: `validate_candles(df, timeframe) -> ValidationReport` with
  counts of duplicates, gaps (missing bars, listed as (start, end) ranges),
  bars violating `low <= min(open, close) <= max(open, close) <= high`,
  non-positive prices, and negative volume. `report.ok` is true when only gaps
  exist and they total under 0.1% of expected bars.
- `resample.py`: `resample(df, timeframe_from, timeframe_to) -> pd.DataFrame`
  aggregates base bars into larger bars labelled by **close** time; a target
  bar is emitted only when all of its base bars are present (incomplete
  buckets are dropped, never forward-filled).
- CLI: `regimebot data download [--start ISO] [--end ISO]` (defaults from
  `cfg.backtest`; all configured symbols; base timeframe plus funding) and
  `regimebot data check` (prints the validation report per symbol, exits 1 if
  any is not ok).

Tests use a `FakeExchange` (deterministic synthetic bars, configurable page
size, can inject a network error) living in `tests/fakes.py` so later tasks
can reuse it: paging covers the full range with no duplicates and no gap;
resume after an interruption downloads only the remainder; the forming bar is
dropped; ccxt open-time to close-time conversion (test `CcxtExchange` parsing
with a stubbed ccxt client object, no network); retry then success; retry
exhaustion raises; every validation rule; resample close-time labelling
(1m bars closing 00:01..00:15 make the 15m bar with ts 00:15) and
incomplete-bucket dropping.

### Task 4: Causal features

Model tier: standard. Reviewer attention: causality.

Create `regimebot/market_state/features.py`:

`compute_features(candles: pd.DataFrame, cfg: FeaturesConfig, compression_pctile: float, funding: pd.Series | None = None) -> pd.DataFrame`

Input: decision-timeframe candles (UTC close-time index). Output: same index,
columns below. Row *t* depends only on input rows `<= t`. Warm-up rows are
NaN (never back-filled).

| Column | Definition |
|---|---|
| `close`, `high`, `low`, `volume` | copied through |
| `ret` | log return of close |
| `atr` | Wilder ATR over `atr_window`, price units |
| `atr_pct` | `atr / close` |
| `rv` | rolling std of `ret` over `rv_window` (per bar, not annualised) |
| `vol_pctile` | percentile rank in [0, 1] of current `rv` within the trailing `pctile_window` values (inclusive of current) |
| `adx`, `plus_di`, `minus_di` | Wilder ADX over `adx_window` |
| `ema_fast`, `ema_slow` | EMAs of close (`adjust=False`), NaN until `span` bars exist |
| `volume_z` | z-score of volume over `volume_window`, using mean and std of the **previous** `volume_window` bars (current bar excluded) |
| `donchian_high`, `donchian_low` | max high / min low of the **previous** `donchian_window` bars (current bar excluded) |
| `zscore` | `(close - rolling mean) / rolling std` over `zscore_window` |
| `roll_mean` | the rolling mean used in `zscore` |
| `ret_z` | `ret` divided by the previous bar's `rv` |
| `illiq_pctile` | percentile rank over `pctile_window` of Amihud illiquidity `abs(ret) / (close * volume)`; NaN when volume is 0 |
| `funding` | last known funding rate at or before `t` (NaN if none supplied) |
| `bars_since_compression` | bars since `vol_pctile` was last below the compression threshold; takes the threshold as an explicit argument `compression_pctile: float` on `compute_features`; NaN if never |

Tests:

- **Prefix invariance** (the key test): for random-walk candles of length
  N = 4000 (seeded) and several cut points k, `compute_features(c.iloc[:k])`
  equals `compute_features(c).iloc[:k]` for every column (NaN-equal,
  `rtol=1e-9`). Use a reduced `pctile_window` so the test is fast.
- Known-value checks on tiny hand-computable inputs for ATR, EMA,
  donchian (verifies the current bar is excluded), volume_z (current bar
  excluded), vol_pctile endpoints.
- ADX rises on a steady trend and is low on alternating noise.
- Warm-up rows are NaN and never filled; zero volume does not produce inf.
- Performance: 70,000 rows computes in under 5 seconds (rolling rank must be
  vectorised, e.g. `rolling(...).rank(pct=True)`).

### Task 5: Regime classifier, playbook, regime study

Model tier: standard.

Create `regimebot/market_state/regime.py`:

- `classify(row: Mapping[str, float], cfg: RegimeConfig, ts: datetime, spread_bps: float | None = None) -> Regime`
  and a vectorised `classify_frame(features, cfg) -> pd.Series` (no spread)
  that gives identical results row by row (test this equivalence).
  Rules in priority order, first match wins:
  1. `NO_TRADE` if any of `atr_pct, vol_pctile, adx, ema_fast, ema_slow, volume_z, ret_z, illiq_pctile` is NaN.
  2. `EVENT_RISK` if `ts` is inside a configured event window or `abs(ret_z) >= shock_z`.
  3. `LIQUIDITY_STRESS` if `spread_bps` is given and `> max_spread_bps`, or `illiq_pctile >= cfg.illiq_pctile`.
  4. `EXPANSION` if `vol_pctile >= expansion_pctile`.
  5. `TREND_UP` if `adx >= adx_trend` and `ema_fast > ema_slow` and `close > ema_fast`;
     `TREND_DOWN` mirrored.
  6. `COMPRESSION` if `vol_pctile <= compression_pctile`.
  7. `RANGE` if `adx <= adx_range`.
  8. otherwise `NO_TRADE`.
- `playbook_for(regime, cfg: AppConfig) -> PlaybookEntry`.
- `regimebot/market_state/study.py`: `regime_study(features, regimes, horizons=(4, 16, 96)) -> pd.DataFrame`
  with one row per regime: share of time, count of episodes, and for each
  horizon the mean, std and 5th/95th percentile of forward log return plus the
  share of bars followed by an absolute move of 5% or more within the horizon
  (forward max excursion on close). Forward returns are for research
  reporting only and must never be merged into the features frame.
- CLI `regimebot regimes study` prints the table per symbol from stored
  candles and writes `reports/regime_study_<symbol>.csv`.

Tests: one constructed row per rule proving priority order (for example a row
that satisfies both EXPANSION and TREND_UP returns EXPANSION; NaN beats
everything; event window beats liquidity stress); `classify_frame` equals
row-by-row `classify`; study numbers on a tiny hand-built series.

### Task 6: Strategies

Model tier: standard. Reviewer attention: stop/target geometry.

Create `regimebot/strategy/base.py`, `breakout.py`, `trend.py`,
`mean_reversion.py`, `regimebot/strategy/__init__.py` exporting
`build_strategies(cfg: AppConfig) -> list[Strategy]` (enabled ones only).

`class Strategy(Protocol)`: `name: str`;
`generate(symbol: str, window: pd.DataFrame, regime: Regime) -> Signal | None`
where `window` is the features frame up to and including the decision bar
(last row = now). Strategies are pure: no state, no I/O. They do not check the
playbook (the engine does) and do not size (risk does).

Shared helpers in `base.py`:

- `build_signal(...)`: given side, entry (= last close), a raw stop price and
  the strategy config, compute `stop_pct = |entry - stop| / entry`; reject
  (return None) if `stop_pct <= 0` or `stop_pct > max_stop_pct`;
  `target_pct = max(min_target_pct, min_rr * stop_pct)`; reject if not
  `target_is_plausible`; target price = `entry * (1 +/- target_pct)`.
  The resulting reward-to-risk is therefore always `>= min_rr`.
- `target_is_plausible(target_pct, rv_per_bar, max_hold_bars, mult) -> bool`:
  `target_pct <= mult * rv_per_bar * sqrt(max_hold_bars)`.
- Mean reversion overrides the target: target = `roll_mean`; reject unless
  `|target - entry| / |entry - stop| >= min_rr`.

Rules (long side shown; short is the exact mirror):

- **Breakout**: `close > donchian_high`, `volume_z >= volume_z_min`,
  `bars_since_compression <= compression_lookback`. Stop = `close - stop_atr_mult * atr`.
- **Trend**: regime is `TREND_UP` (long only in TREND_UP, short only in
  TREND_DOWN); within the last `pullback_lookback` bars (excluding the current
  one) some bar's `low <= ema_fast + pullback_atr_dist * atr` (using that
  bar's own ema_fast and atr); current `close > previous bar's high`;
  `volume_z >= volume_z_min`. Stop = lowest low of the lookback window
  `- stop_atr_mult * atr`.
- **Mean reversion**: regime is `RANGE`; `zscore <= -entry_z` for long.
  Stop = `close - stop_atr_mult * atr`. Target = `roll_mean`.

Any NaN among the inputs a strategy reads returns None.

Tests per strategy: a constructed window that triggers long; its mirror
triggers short with mirrored stop/target; each condition removed in turn
yields None; stop wider than `max_stop_pct` yields None; implausible target
yields None; reward-to-risk of every emitted signal `>= min_rr`; stop and
target are on the correct sides of entry; NaN inputs yield None;
`build_strategies` honours `enabled`.

### Task 7: Fill model and portfolio accounting (money path)

Model tier: most capable.

Create `regimebot/execution/fill_model.py` and `regimebot/engine/portfolio.py`.

`fill_model.py` (pure functions, `costs: CostsConfig`):

- `market_fill(symbol, ts, side: OrderSide, qty, bid, ask, ref_price, bar_volume, costs, cap_participation: bool) -> Fill | None`
  Taker. Base price is `ask` for BUY, `bid` for SELL. Participation =
  `qty / bar_volume` (if `bar_volume <= 0`: return None when
  `cap_participation`). When `cap_participation` (entries), qty is cut to
  `max_participation * bar_volume`. Exits are never cut; their participation
  for the impact term is `min(qty / bar_volume, 1.0)`, and 1.0 when
  `bar_volume <= 0`, so an exit always fills and never gets cheaper as
  volume falls. Slippage bps =
  `slippage_bps + impact_bps_per_pct * participation * 100` (participation
  after the cut), applied against the trader. Fee = `taker_fee_bps` of
  notional. `Fill.slippage_bps` is measured against `ref_price`, signed so
  that positive is worse for the trader.
- `quote_from_bar_open(open_price, costs) -> tuple[float, float]`: bid/ask =
  open -/+ `half_spread_bps`.
- `stop_fill(symbol, bar: Bar, side: OrderSide, qty, stop_price, costs) -> Fill | None`
  A SELL stop triggers when `bar.low <= stop_price`; trigger price is
  `min(stop_price, bar.open)` (gap through the stop fills at the open). BUY
  stop mirrored with `bar.high` and `max`. Then half spread and slippage are
  applied against the trader as in an exit `market_fill` (same bounded
  participation from `bar.volume`), taker fee, `ref_price = stop_price`.
- `target_fill(symbol, bar: Bar, side: OrderSide, qty, target_price, costs) -> Fill | None`
  A SELL limit fills when `bar.high > target_price` (strictly through, a touch
  is not a fill) at exactly `target_price`, even when the bar opens beyond it
  (a resting limit trades at its own price; no price improvement is
  assumed), maker fee, zero slippage against `ref_price = target_price`.
  BUY mirrored.
- `funding_payment(side: Side, qty, mark_price, rate) -> float`: cost to the
  position, positive = paid. Longs pay when rate > 0, shorts receive.

`portfolio.py`:

- `@dataclass class Position` (mutable): symbol, strategy, side, regime, qty,
  entry_price, entry_ts, stop, initial_stop, target, atr_at_entry,
  max_hold_bars, trail_atr_mult, bars_held = 0, best_price, fees = 0.0,
  funding = 0.0; `to_state() -> dict` / `from_state(dict)` (JSON-safe
  round-trip for the `open_positions` table).
- `class Portfolio(initial_equity)`: `cash` (realised equity), `positions:
  dict[str, Position]` (at most one per symbol; opening a second raises),
  `open(signal, fill) -> Position`, `close(symbol, fill, reason) -> Trade`,
  `apply_funding(symbol, mark_price, rate)`, `mark(prices: Mapping[str, float])`
  storing last prices, `equity` (cash + unrealised at last marks),
  `gross_exposure` (sum of |qty * last price|), `exposure(symbol)`,
  `peak_equity`, `drawdown` (fraction below peak, >= 0).
  Accounting: entry fee is deducted from cash at open; at close, cash changes
  by gross pnl minus the exit fee; funding is deducted when applied.
  `Trade.fees` is entry + exit fees, `Trade.net_pnl = gross_pnl - fees - funding`,
  and the sum of `net_pnl` over all trades equals `cash - initial_equity` once
  flat (test this invariant with random trades).

Tests: every branch above with hand-computed numbers: BUY and SELL market
fills, participation cut, zero-volume bar, impact term; stop exact, stop gap,
stop not touched, both directions; target touch-is-not-fill, through, gap
open beyond target; funding signs for all four side/rate combinations;
portfolio long win, long loss, short win, short loss with fees and funding;
equity/drawdown/exposure after marks; the cash invariant; `Position` state
round-trip.

### Task 8: Risk manager (money path)

Model tier: most capable.

Create `regimebot/risk/manager.py` (sizing, limits, kill switch in one module;
split only if it passes ~300 lines).

`class RiskManager(cfg: AppConfig, kill_file: Path)` with state: realised pnl
by UTC day and ISO week, consecutive losing trades, `halted_reason: str | None`,
`day_stopped: date | None`.

- `evaluate(signal: Signal, portfolio: Portfolio, playbook: PlaybookEntry, now: datetime, pending: Mapping[str, float] | None = None) -> RiskDecision`
  `pending` maps symbol to the notional of entries already approved but not
  yet filled (the engine's queue). Pending notional counts as exposure for
  the asset, group and gross caps, and a symbol present in `pending` is
  rejected as `position_exists`, so several approvals in one decision bar
  can never jointly breach a cap.
  Checks in this order, first failure rejects with that rule name:
  1. `kill_switch` (halted, or `kill_file` exists)
  2. `day_stopped` (streak stop hit today)
  3. `regime_blocks_entries` (`not playbook.allow_entries`)
  4. `strategy_not_allowed` (signal.strategy not in playbook.strategies)
  5. `position_exists` (symbol already open)
  6. `invalid_stop` (stop distance <= 0 or stop on the wrong side)
  Then size: `risk_amount = equity * risk_per_trade * playbook.size_mult * streak_mult`
  where `streak_mult` is 0.5 once consecutive losses `>= streak_half`, else 1;
  `qty = risk_amount / |entry - stop|`. Then cap notional (`qty * entry`) by,
  in order: remaining per-asset room (`max_asset_exposure * equity`),
  remaining room in every group containing the symbol
  (`max_group_exposure * equity` minus current group exposure), remaining
  gross room (`max_gross_leverage * equity` minus current gross exposure).
  If a cap reduces qty the decision is approved with `rule` naming the binding
  cap (`cap_asset`, `cap_group`, `cap_gross`). If the final notional is under
  10 quote units, reject with `too_small`. Unreduced approvals use rule `ok`.
- `on_trade_closed(trade: Trade) -> None`: updates day/week pnl and the
  streak (a trade with `net_pnl < 0` extends it, otherwise resets it); when
  the streak reaches `streak_stop`, sets `day_stopped` to the trade's exit
  date (cleared automatically on the next UTC day, streak reset to 0 then).
- `check_limits(portfolio, now) -> list[RiskEvent]`: using equity at the
  start of the UTC day/ISO week (tracked by the manager from the equity it
  sees on the first call of each period): if
  `equity <= day_start_equity * (1 - daily_loss)` -> event `daily_loss`;
  weekly likewise -> `weekly_loss`; `portfolio.drawdown >= max_drawdown` ->
  `max_drawdown`; kill file present -> `manual_kill`. Any event sets
  `halted_reason`. Daily and weekly halts clear automatically at the next
  UTC day / ISO week; `max_drawdown` and `manual_kill` stay until
  `clear_halt()` is called (and the kill file is removed). Each event is
  returned once, not on every call.
- `must_flatten -> bool`: true while halted.
- `to_state() -> dict` / `load_state(dict)` for paper-session restarts.
- `RiskEvent` is defined in `types.py` (Task 1).

CLI: `regimebot kill` creates the kill file (`data/KILL`), `regimebot unkill`
removes it. This is the one-command emergency stop.

Tests: each rejection rule in order (a signal failing two rules reports the
earlier one); sizing formula (stop hit loses exactly `risk_per_trade` of
equity before costs, test with numbers); each cap binding alone; group cap
with an existing position in another group member; streak half and streak
stop, including reset next day; daily, weekly and drawdown halts with
auto-clear versus sticky behaviour; events emitted once; kill file; state
round-trip; a property-style test over random signals and portfolios that an
approved decision never breaches the gross, asset or group caps and never
risks more than `risk_per_trade * equity` at the stop.

### Task 9: Engine loop and backtest runner (money path)

Model tier: most capable.

Create `regimebot/engine/loop.py` and `regimebot/backtest/runner.py`.

`class Engine(cfg, strategies, risk: RiskManager, portfolio: Portfolio, log: RunLog)`

`RunLog` (in `loop.py`): plain dataclass of lists the engine appends to:
signals, decisions (signal index + RiskDecision + ts), fills, trades,
regimes (symbol, ts, regime), risk_events, equity (ts, equity,
gross_exposure, drawdown).

Engine methods (the engine is identical in backtest and paper):

- `on_decision(ts, windows: Mapping[str, pd.DataFrame], spreads: Mapping[str, float] | None = None) -> None`
  `windows[symbol]` is the features frame up to and including `ts` (missing
  symbol or last index != `ts` means no data: treat as `NO_TRADE` and log a
  `stale_data` risk event). Steps, in order:
  1. Mark portfolio at each symbol's last close.
  2. Classify regime per symbol; record it.
  3. For each open position: `bars_held += 1`; update `best_price`; if
     `trail_atr_mult` is set and price has moved at least 1R in favour, raise
     (long) the stop to `best_price - trail_atr_mult * atr` (current ATR),
     never loosening it; queue a market exit if `bars_held >= max_hold_bars`
     (reason TIME) or the regime's playbook has `force_close` (reason REGIME).
  4. `risk.check_limits`; record events; if `risk.must_flatten`, queue market
     exits for every open position (reason KILL) and skip entries.
  5. For each symbol without a position or queued order: run the strategies
     the playbook allows, in config order; first signal wins; record it;
     `risk.evaluate` (passing the notional of entries already queued in this
     decision as `pending`); record the decision; if approved queue a market
     entry.
- `fill_queued(symbol, ts, bid, ask, bar_volume) -> None`: executes queued
  market entries (participation-capped; a None fill, or a fill whose
  notional is under 10 quote units, cancels the entry and is recorded as a
  decision with rule `no_liquidity`) and queued exits via
  `fill_model.market_fill`, updating portfolio, risk (`on_trade_closed`) and
  the log. Entry `ref_price` is the signal's entry; exit `ref_price` is the
  last decision close.
- `on_minute(bar: Bar) -> None`: for an open position on `bar.symbol`, check
  the stop first (`stop_fill`), then the target (`target_fill`); a stop moved
  by trailing reports reason TRAIL when it is above (long) the initial stop,
  else STOP. A position opened by a fill at this bar's open is checked against
  this same bar.
- `on_funding(symbol, ts, rate, mark_price) -> None`.
- `flatten_all(ts, prices, reason) -> None` for end of data and shutdown.

`runner.py`:

`run_backtest(cfg, candles: Mapping[str, pd.DataFrame], funding: Mapping[str, pd.Series], start, end, kill_file) -> BacktestResult`
(`candles` are base-timeframe frames). It resamples to the decision
timeframe, computes features once per symbol on the full history, then walks
decision timestamps in order. Between consecutive decision times T(k-1) and
T(k), for each symbol that has an open position or a queued order, it feeds
the base bars with `T(k-1) < ts <= T(k)` in time order: the first of them
calls `fill_queued` with `quote_from_bar_open(bar.open)` and that bar's
volume, then every bar goes to `on_minute`; funding events in the interval
are applied at their timestamps. Symbols with nothing open skip their minute
bars (this is the main performance lever; a 2-year, 3-symbol run should take
well under two minutes). Then `on_decision(T(k), windows)` where
`windows[symbol]` is a slice ending at T(k) (pass a trailing slice of fixed
length, at least 400 rows, not a growing copy). At the end, `flatten_all`
with reason END_OF_DATA. Features for bars before `start` are warm-up only;
no decisions before `start`.

`BacktestResult`: `log: RunLog`, `trades: list[Trade]`, `equity: pd.Series`
(UTC index at decision times), `cfg`, `start`, `end`.

`persist_result(session, result, note="") -> int` writes a run and all log
rows via `db.repo`. CLI: `regimebot backtest run [--start] [--end] [--note]`
loads candles and funding from the database, runs, persists, prints the
headline numbers and the run id.

Tests (synthetic candles built by small helpers in `tests/synth.py`):

- No look-ahead, end to end: run on candles `C` to time T; run again on `C`
  with all bars after T replaced by different random data; the fills and
  trades with `exit_ts <= T` and the signals with `ts <= T` are identical.
- An entry signalled at decision close T fills at the open of the first base
  bar after T, not at the decision close.
- Stop-before-target when one base bar touches both.
- A scripted scenario with a known outcome: one long entered, target hit;
  check qty from the sizing formula, fill prices including spread and
  slippage, fees, net pnl and final equity to the cent by hand calculation.
- Time exit, regime force-close, trailing stop tightening and never
  loosening, kill switch flattening and blocking further entries.
- Playbook is enforced (a strategy not allowed in the regime never trades).
- Participation cap produces a partial entry; zero-volume bar cancels.
- Funding is charged to a position held across a funding timestamp.
- Final equity equals initial equity plus the sum of trade `net_pnl`.
- `persist_result` round-trips trades and equity.

Test strategies may be simple stubs implementing the `Strategy` protocol.

### Task 10: Metrics, baselines and reports

Model tier: standard.

Create `regimebot/reporting/metrics.py`, `regimebot/backtest/baselines.py`,
`regimebot/reporting/report.py`.

`metrics.py`: `compute_metrics(trades, equity, initial_equity) -> Metrics`
(a frozen dataclass) with: net_pnl, return_pct, max_drawdown,
sharpe and sortino (from daily equity returns, annualised with sqrt(365);
NaN if fewer than 2 days or zero deviation), profit_factor (inf-safe: NaN
when there are no losing trades and no winning trades, `inf` when only
winners), win_rate, avg_win, avg_loss, expectancy_r (mean r_multiple),
n_trades, avg_bars_held, exposure_pct (share of decision bars with a
position), turnover (sum of entry notionals / initial equity), fees,
funding, target_hit_rate (share with exit TARGET), stop_out_rate (STOP or
TRAIL below entry), net_pnl_without_best_5pct (drop the top
`ceil(0.05 * n)` trades by net_pnl), avg_entry_slippage_bps and
avg_exit_slippage_bps (from fills, taker only).
`breakdown(trades, by: Literal["regime", "symbol", "strategy", "exit_reason"]) -> pd.DataFrame`
(n, win_rate, net_pnl, expectancy_r per group).

`baselines.py`: `buy_and_hold(candles, initial_equity, costs) -> pd.Series`
(equal-weight across symbols at 1x, entry and exit costs included, equity on
the decision index); `random_entry(cfg, candles, funding, start, end, seed) -> BacktestResult`
which reruns the real engine with a stub strategy that enters at random with
the same average frequency as a supplied reference result and the same
stop/target geometry as breakout (ATR stop, `build_signal`), so entries are
the only difference.

`report.py`: `write_report(result, baselines, out_dir) -> Path` producing
`report.html` (single self-contained file, charts embedded as base64 PNG),
`trades.csv`, `daily_pnl.csv`, `equity.png`, `drawdown.png`. The HTML has:
a risk warning block (see Task 13 text requirements: substantial risk,
leverage can lose fast, past performance is no guarantee, backtest is not
proof); headline metrics table next to buy-and-hold and random-entry;
breakdowns by regime, symbol, strategy and exit reason; modelled versus
realised slippage; risk-limit events; net pnl without the best 5% of trades;
the number of parameter variants tried when supplied. Matplotlib uses the
`Agg` backend. CLI: `regimebot report RUN_ID` rebuilds the report from the
database into `reports/run_<id>/`.

Tests: metrics on hand-built trades and equity (each number checked);
edge cases (no trades, only winners, single day); breakdown sums match
totals; buy-and-hold on a flat and on a doubling series; report files are
created and the HTML contains the risk warning and every section heading;
random-entry is deterministic for a seed.

### Task 11: Walk-forward, sensitivity and stress runs

Model tier: standard.

Create `regimebot/backtest/walk_forward.py` and `regimebot/backtest/robustness.py`.

- `walk_forward(cfg, candles, funding, start, end) -> WalkForwardResult`:
  rolling windows of `train_days` then `test_days`, stepping by `test_days`.
  For each window run every combination of `cfg.walk_forward.grid` on the
  train span; pick the combination with the highest profit factor among those
  with at least `min_trades` trades (if none qualifies, the test span is run
  with the unmodified config and flagged `no_qualifying_params`); run the
  test span with the chosen parameters. Each test run starts from the initial
  equity with fresh risk state. Result: per-window table (dates, chosen
  params, train and test metrics), the concatenated out-of-sample trades, and
  `n_variants_tried` (combinations times windows). Feature warm-up data
  before each span start may be used, as in `run_backtest`.
- `robustness.py`:
  `sensitivity(cfg, candles, funding, start, end, params: Sequence[str], rel=0.2) -> pd.DataFrame`
  (each numeric dotted-path param at -20%, base, +20%, others fixed; net_pnl,
  profit_factor, n_trades per run);
  `cost_stress(..., mult=2.0)` (all cost bps doubled);
  `leverage_compare(...)` (`risk.max_gross_leverage` 1.0 versus 1.5).
- CLI: `regimebot backtest walk-forward` and `regimebot backtest robustness`
  print the tables and write CSVs under `reports/`.

Tests (small synthetic data and a tiny grid so they run in seconds): window
boundaries are correct and test spans never overlap train spans of the same
window; the chosen params are the argmax on train, not on test (construct a
case where they differ); `no_qualifying_params` path; `n_variants_tried`;
sensitivity table shape and that the base row equals a plain run; cost
stress never increases net pnl for the same trades; leverage compare returns
both rows.

### Task 12: Paper trading session (money path)

Model tier: most capable.

Create `regimebot/paper/session.py`, `regimebot/reporting/alerts.py`, CLI.

- `alerts.py`: `alert(kind: str, detail: str) -> None` logs at ERROR and, if
  `ALERT_WEBHOOK_URL` is set, POSTs `{"kind", "detail", "ts"}` as JSON with a
  5 s timeout using `urllib.request`; a webhook failure is logged and never
  raised.
- `PaperSession(cfg, exchange: Exchange, engine_db, kill_file, clock=..., sleep=...)`
  (clock and sleep injectable for tests):
  - `start(resume_run_id: int | None)`: new run (mode paper) or resume:
    restore positions from `open_positions`, cash/peak from the last equity
    snapshot plus trades, and risk state from a `risk_state` entry stored
    alongside (store it as the `open_positions` row with symbol `"__risk__"`).
    Then warm up: load the last `warmup_bars` decision bars' worth of base
    candles from the exchange into the database (reuse `download_candles`).
  - `run_once() -> None`: one poll. Fetch new closed base bars per symbol and
    upsert them; feed each new bar to `engine.on_minute` in time order; apply
    funding when a new funding record appears; when a decision bar has just
    closed for all symbols, rebuild features from the stored trailing window
    (same `compute_features`, same resample), fetch a quote per symbol, call
    `engine.on_decision(ts, windows, spreads)` and then
    `engine.fill_queued(symbol, ts, quote.bid, quote.ask, last_bar_volume)`
    using the live quote. After each decision, flush new log rows and the
    position and risk state to the database.
  - Stale-data guard: if a symbol's newest closed bar is older than
    `stale_after_seconds`, or a fetch fails, that symbol gets no window at
    the next decision (the engine then records `stale_data` and does not
    trade it) and `alert("stale_data", ...)` fires once per incident, with a
    recovery alert when data returns. Exchange errors never crash the loop.
  - `run_forever()`: loop `run_once` + `sleep(poll_seconds)`; on
    KeyboardInterrupt or SIGTERM, flush state and exit without flattening
    (positions resume next start). Kill file handling is the engine's
    (flatten + block).
- CLI: `regimebot paper run [--resume RUN_ID]`, `regimebot paper status`
  (latest paper run: equity, open positions, last regime per symbol, halted
  reason, last 10 risk events).

The paper session never sends orders. There is no code path that can place
an order on an exchange in this version.

Tests with `FakeExchange` and a fake clock: a scripted bar sequence produces
the same trades as `run_backtest` on the same bars with matching spread
(parity test; allow only the documented fill-price difference from live
quote versus bar-open quote by giving the fake quote equal to the next bar
open +/- half spread); stale data blocks entries and alerts once; an
exchange exception in one poll does not stop the next; resume restores
positions, equity and risk state exactly (stop mid-position, restart,
finish, compare to an uninterrupted run); kill file flattens; webhook
failure is swallowed; decision fires once per decision bar even with several
polls inside it.

### Task 13: Documentation

Model tier: standard.

Write `README.md`, `docs/adr/0001-single-process-shared-engine.md`,
`docs/adr/0002-sqlite-default-postgres-optional.md`,
`docs/adr/0003-no-live-orders-in-v1.md`, `docs/operations.md`,
`docs/research-protocol.md`. Read the code first; document what exists, with
commands that work (run each command you document, except Docker ones).

- README: what this is and is not; the risk statements (crypto trading
  involves substantial risk; leveraged trading can lose money fast; past
  performance does not guarantee future results; no system predicts sudden
  market changes; test extensively in paper trading before any real capital;
  a backtest is not proof); quick start (venv, install, `db upgrade`, `data
  download`, `backtest run`, `report`); configuration overview; module map;
  how to run tests and linters; what is not implemented.
- operations.md: paper trading runbook, kill switch, resume, alerts, stale
  data behaviour, where state lives, API key guidance for any future live use
  (trade-only keys, withdrawals never enabled, IP allowlist, keys only in env).
- research-protocol.md: the anti-overfitting rules: causality tests,
  walk-forward, one-touch holdout, sensitivity, cost stress, remove-best-5%,
  baselines, trial counting, and how to read the report honestly.
- ADRs: context, decision, consequences; under one page each. ADR 0002 notes
  Docker was unavailable on the development machine. ADR 0003 states the
  conditions for adding live orders (weeks of paper results, testnet first,
  explicit confirmation flag, legal check of the venue for the user's
  jurisdiction).

No tests. Markdown must render cleanly; no placeholder text.
