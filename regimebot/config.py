"""Pydantic config models and YAML loading.

``load_config`` reads ``config/default.yaml`` (or a given file), applies the
``DATABASE_URL`` env var and dotted-path overrides, then validates.
"""

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt, model_validator

from regimebot.types import Regime

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
MAX_GROSS_LEVERAGE = 1.5  # hard cap, not configurable above this

Timeframe = Literal["1m", "5m", "15m", "1h", "4h"]
StrategyName = Literal["breakout", "trend", "mean_reversion"]
Fraction = Annotated[float, Field(gt=0, le=1)]
NonNegative = Annotated[float, Field(ge=0)]
Bps = Annotated[float, Field(ge=0, le=1000)]  # 1000 bps = 10%
UnitFraction = Annotated[float, Field(ge=0, le=1)]  # 0 allowed, e.g. size_mult 0.0

_TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


def timeframe_minutes(tf: str) -> int:
    """Length of a timeframe string in minutes; raises ValueError if unsupported."""
    try:
        return _TIMEFRAME_MINUTES[tf]
    except KeyError:
        raise ValueError(f"unsupported timeframe {tf!r}") from None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExchangeConfig(_Model):
    id: str
    symbols: list[str]
    base_timeframe: Timeframe
    decision_timeframe: Timeframe


class CostsConfig(_Model):
    taker_fee_bps: Bps
    maker_fee_bps: Bps
    half_spread_bps: Bps
    slippage_bps: Bps
    impact_bps_per_pct: Bps
    max_participation: Fraction

    @model_validator(mode="after")
    def _worst_case_cost_bounded(self) -> "CostsConfig":
        # Worst case: spread + fixed slippage + impact at 100% of bar volume. Keeps a SELL
        # price (bid * (1 - bps / 1e4)) far from zero.
        worst = self.half_spread_bps + self.slippage_bps + 100 * self.impact_bps_per_pct
        if worst >= 1000:
            raise ValueError(
                f"worst-case taker cost {worst:.0f} bps (half_spread + slippage + "
                "100 * impact_bps_per_pct) must be below 1000 bps"
            )
        return self


class FeaturesConfig(_Model):
    atr_window: PositiveInt
    rv_window: PositiveInt
    pctile_window: PositiveInt
    adx_window: PositiveInt
    ema_fast: PositiveInt
    ema_slow: PositiveInt
    volume_window: PositiveInt
    donchian_window: PositiveInt
    zscore_window: PositiveInt

    @model_validator(mode="after")
    def _fast_below_slow(self) -> "FeaturesConfig":
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be smaller than ema_slow")
        if self.pctile_window < self.rv_window:
            raise ValueError("pctile_window must be at least rv_window")
        return self


class EventWindow(_Model):
    start: AwareDatetime
    end: AwareDatetime
    label: str

    @model_validator(mode="after")
    def _utc_and_ordered(self) -> "EventWindow":
        object.__setattr__(self, "start", self.start.astimezone(UTC))
        object.__setattr__(self, "end", self.end.astimezone(UTC))
        if self.start >= self.end:
            raise ValueError("event window start must be before end")
        return self


class RegimeConfig(_Model):
    expansion_pctile: Fraction
    compression_pctile: Fraction
    adx_trend: NonNegative
    adx_range: NonNegative
    shock_z: NonNegative
    illiq_pctile: Fraction
    max_spread_bps: NonNegative
    event_windows: list[EventWindow]


class PlaybookEntry(_Model):
    strategies: list[StrategyName]
    size_mult: UnitFraction
    allow_entries: bool
    force_close: bool


class BreakoutConfig(_Model):
    enabled: bool
    min_target_pct: UnitFraction
    min_rr: float = Field(gt=0)
    stop_atr_mult: float = Field(gt=0)
    max_stop_pct: Fraction
    volume_z_min: float
    compression_lookback: PositiveInt
    max_hold_bars: PositiveInt
    trail_atr_mult: float | None = Field(gt=0)
    plausibility_mult: float = Field(gt=0)


class TrendConfig(_Model):
    enabled: bool
    min_target_pct: UnitFraction
    min_rr: float = Field(gt=0)
    stop_atr_mult: float = Field(gt=0)
    max_stop_pct: Fraction
    pullback_lookback: PositiveInt
    pullback_atr_dist: float = Field(gt=0)
    volume_z_min: float
    max_hold_bars: PositiveInt
    trail_atr_mult: float | None = Field(gt=0)
    plausibility_mult: float = Field(gt=0)


class MeanReversionConfig(_Model):
    enabled: bool
    min_target_pct: UnitFraction
    min_rr: float = Field(gt=0)
    entry_z: float = Field(gt=0)
    stop_atr_mult: float = Field(gt=0)
    max_stop_pct: Fraction
    max_hold_bars: PositiveInt
    trail_atr_mult: float | None = Field(gt=0)
    plausibility_mult: float = Field(gt=0)


class StrategiesConfig(_Model):
    breakout: BreakoutConfig
    trend: TrendConfig
    mean_reversion: MeanReversionConfig


class RiskConfig(_Model):
    risk_per_trade: Fraction
    max_gross_leverage: float = Field(gt=0, le=MAX_GROSS_LEVERAGE)
    max_asset_exposure: Fraction
    max_group_exposure: float = Field(gt=0, le=MAX_GROSS_LEVERAGE)  # may exceed 1x, like gross
    groups: dict[str, list[str]]
    daily_loss: Fraction
    weekly_loss: Fraction
    max_drawdown: Fraction
    streak_half: PositiveInt
    streak_stop: PositiveInt
    # quote units: a smaller entry is rejected (too_small) or, cut at fill time, cancelled
    # (no_liquidity). Defaulted so configs stored before it existed still validate.
    min_notional: float = Field(default=10.0, gt=0)


def gross_cap_limits(risk: RiskConfig, symbols: Sequence[str]) -> dict[str, float]:
    """Each cap's bound on gross exposure (multiple of equity): ``gross``; ``group`` when one
    group holds every configured symbol (its exposure is then the gross exposure); ``asset``:
    the number of symbols times ``max_asset_exposure``."""
    limits = {"gross": risk.max_gross_leverage}
    if any(set(symbols) <= set(members) for members in risk.groups.values()):
        limits["group"] = risk.max_group_exposure
    limits["asset"] = len(set(symbols)) * risk.max_asset_exposure
    return limits


def effective_gross_cap(risk: RiskConfig, symbols: Sequence[str]) -> float:
    """The gross exposure (multiple of equity) the caps let a run reach: the lowest of
    ``gross_cap_limits``."""
    return min(gross_cap_limits(risk, symbols).values())


class BacktestConfig(_Model):
    initial_equity: float = Field(gt=0)
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _start_before_end(self) -> "BacktestConfig":
        object.__setattr__(self, "start", self.start.astimezone(UTC))
        object.__setattr__(self, "end", self.end.astimezone(UTC))
        if self.start >= self.end:
            raise ValueError("backtest start must be before end")
        return self


class WalkForwardConfig(_Model):
    train_days: PositiveInt
    test_days: PositiveInt
    min_trades: PositiveInt
    grid: dict[str, list[float]]  # dotted config path -> values to try


class PaperConfig(_Model):
    poll_seconds: PositiveInt
    stale_after_seconds: PositiveInt
    warmup_bars: PositiveInt


class AppConfig(_Model):
    database_url: str
    exchange: ExchangeConfig
    costs: CostsConfig
    features: FeaturesConfig
    regime: RegimeConfig
    playbook: dict[Regime, PlaybookEntry]
    strategies: StrategiesConfig
    risk: RiskConfig
    backtest: BacktestConfig
    walk_forward: WalkForwardConfig
    paper: PaperConfig

    @model_validator(mode="after")
    def _playbook_covers_all_regimes(self) -> "AppConfig":
        missing = set(Regime) - set(self.playbook)
        if missing:
            raise ValueError(f"playbook missing regimes: {sorted(r.value for r in missing)}")
        return self

    @model_validator(mode="after")
    def _paper_warmup_covers_features(self) -> "AppConfig":
        from regimebot.market_state.features import warmup_bars  # it imports this module

        if self.paper.warmup_bars < (need := warmup_bars(self.features)):
            raise ValueError(
                f"paper.warmup_bars {self.paper.warmup_bars} is below the {need} decision bars "
                f"the feature windows need before every feature is valid: raise "
                f"paper.warmup_bars to at least {need}"
            )
        return self


def _apply_override(raw: dict[str, Any], dotted: str, value: object) -> None:
    """Set ``raw[a][b][c] = value`` for ``dotted == "a.b.c"``; the path must already exist."""
    *parents, leaf = dotted.split(".")
    node: Any = raw
    for key in parents:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"unknown config path: {dotted!r}")
        node = node[key]
    if not isinstance(node, dict) or leaf not in node:
        raise ValueError(f"unknown config path: {dotted!r}")
    node[leaf] = value


def load_config(
    path: Path | None = None, overrides: Mapping[str, object] | None = None
) -> AppConfig:
    """Load and validate the config.

    ``path=None`` uses ``config/default.yaml`` at the repository root. A non-empty
    ``DATABASE_URL`` env var replaces ``database_url``; ``overrides`` (dotted path
    to value) are applied last, before validation, so an explicit override wins.
    """
    path = path or DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    if env_url := os.environ.get("DATABASE_URL"):
        raw["database_url"] = env_url
    for dotted, value in (overrides or {}).items():
        _apply_override(raw, dotted, value)
    return AppConfig.model_validate(raw)


def with_overrides(cfg: AppConfig, overrides: Mapping[str, object]) -> AppConfig:
    """A copy of ``cfg`` with dotted-path ``overrides`` applied and the result re-validated
    (unknown path or invalid value raises ``ValueError``; ``cfg`` is not modified)."""
    raw = cfg.model_dump(mode="json")
    for dotted, value in overrides.items():
        _apply_override(raw, dotted, value)
    return AppConfig.model_validate(raw)


def config_json(cfg: AppConfig) -> str:
    """Canonical JSON of the strategy config, without ``database_url`` (it may hold a password
    and must not change the identity of a run between SQLite and Postgres)."""
    return json.dumps(
        cfg.model_dump(mode="json", exclude={"database_url"}), sort_keys=True, separators=(",", ":")
    )


def config_hash(cfg: AppConfig) -> str:
    """First 16 hex chars of the sha256 of ``config_json`` (identifies a run's config)."""
    return hashlib.sha256(config_json(cfg).encode()).hexdigest()[:16]
