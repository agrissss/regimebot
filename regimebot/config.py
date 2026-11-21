"""Pydantic config models and YAML loading.

``load_config`` reads ``config/default.yaml`` (or a given file), applies the
``DATABASE_URL`` env var and dotted-path overrides, then validates.
"""

import copy
import hashlib
import json
import os
from collections.abc import Mapping
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
    taker_fee_bps: NonNegative
    maker_fee_bps: NonNegative
    half_spread_bps: NonNegative
    slippage_bps: NonNegative
    impact_bps_per_pct: NonNegative
    max_participation: Fraction


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
        return self


class EventWindow(_Model):
    start: AwareDatetime
    end: AwareDatetime
    label: str


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
    size_mult: NonNegative
    allow_entries: bool
    force_close: bool


class BreakoutConfig(_Model):
    enabled: bool
    min_target_pct: NonNegative
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
    min_target_pct: NonNegative
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
    min_target_pct: NonNegative
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
    max_group_exposure: Fraction
    groups: dict[str, list[str]]
    daily_loss: Fraction
    weekly_loss: Fraction
    max_drawdown: Fraction
    streak_half: PositiveInt
    streak_stop: PositiveInt


class BacktestConfig(_Model):
    initial_equity: float = Field(gt=0)
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _start_before_end(self) -> "BacktestConfig":
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
    with (path or DEFAULT_CONFIG_PATH).open(encoding="utf-8") as f:
        raw = copy.deepcopy(yaml.safe_load(f))
    if env_url := os.environ.get("DATABASE_URL"):
        raw["database_url"] = env_url
    for dotted, value in (overrides or {}).items():
        _apply_override(raw, dotted, value)
    return AppConfig.model_validate(raw)


def config_hash(cfg: AppConfig) -> str:
    """First 16 hex chars of the sha256 of the canonical JSON dump (identifies a run's config)."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
