from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
import pytest
from fakes import T0, FakeExchange
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from regimebot.cli import app, history_start
from regimebot.config import AppConfig, load_config
from regimebot.data.exchange import CLOSE_GRACE, CcxtExchange
from regimebot.data.history import download_candles, download_funding
from regimebot.data.resample import resample
from regimebot.data.validate import validate_candles
from regimebot.db.models import Base, Candle
from regimebot.db.repo import load_candles, load_funding, upsert_candles
from regimebot.db.session import make_engine, session_scope

SYM = "BTC/USDT:USDT"
MIN = timedelta(minutes=1)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with session_scope(engine) as s:
        yield s


def _frame(n: int = 10, start: datetime = T0 + MIN) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", name="ts")
    c = pd.Series(range(100, 100 + n), index=idx, dtype="float64")
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})


# --- history ------------------------------------------------------------------------------


def test_download_pages_cover_range_without_gap_or_duplicate(session: Session) -> None:
    ex = FakeExchange(n_bars=250, page_size=100)
    n = download_candles(ex, session, SYM, "1m", T0, T0 + 1000 * MIN)
    df = load_candles(session, SYM, "1m")
    assert n == len(df) == 250
    assert df.index[0] == T0 + MIN and df.index[-1] == T0 + 250 * MIN
    assert validate_candles(df, "1m").ok and validate_candles(df, "1m").missing_bars == 0
    assert len(ex.candle_calls) == 4  # pages of 100, 100, 50, then an empty page ends it


def test_download_end_is_inclusive_and_later_bars_not_stored(session: Session) -> None:
    ex = FakeExchange(n_bars=250, page_size=100)
    download_candles(ex, session, SYM, "1m", T0, T0 + 120 * MIN)
    df = load_candles(session, SYM, "1m")
    assert df.index[-1] == T0 + 120 * MIN and len(df) == 120


def test_resume_after_interruption_downloads_only_remainder(session: Session) -> None:
    first = FakeExchange(n_bars=250, page_size=100, fail_on_call=2)
    with pytest.raises(ccxt.NetworkError):
        download_candles(first, session, SYM, "1m", T0, T0 + 1000 * MIN)
    assert len(load_candles(session, SYM, "1m")) == 100  # page 1 survived

    second = FakeExchange(n_bars=250, page_size=100)
    n = download_candles(second, session, SYM, "1m", T0, T0 + 1000 * MIN)
    assert n == 150
    # the unfillable T0 hole (no bar closes at T0), then 101..250 and one empty page
    assert second.candle_calls == [T0, T0 + 101 * MIN, T0 + 201 * MIN, T0 + 251 * MIN]
    assert len(load_candles(session, SYM, "1m")) == 250


def test_rerun_is_idempotent_and_refetches_nothing_stored(session: Session) -> None:
    download_candles(FakeExchange(n_bars=50), session, SYM, "1m", T0, T0 + 50 * MIN)
    again = FakeExchange(n_bars=50)
    n = download_candles(again, session, SYM, "1m", T0, T0 + 50 * MIN)
    assert n == 0 and again.candle_calls == [T0]  # only the unfillable T0 hole is retried
    assert len(load_candles(session, SYM, "1m")) == 50


def _delete_candles(session: Session, lo: datetime, hi: datetime) -> None:
    session.execute(delete(Candle).where(Candle.ts >= lo, Candle.ts <= hi))
    session.commit()


def test_interrupted_backfill_is_repaired_by_same_arguments(session: Session) -> None:
    end = T0 + 500 * MIN
    download_candles(FakeExchange(n_bars=500), session, SYM, "1m", T0 + 300 * MIN, end)
    interrupted = FakeExchange(n_bars=500, page_size=100, fail_on_call=3)
    with pytest.raises(ccxt.NetworkError):
        download_candles(interrupted, session, SYM, "1m", T0 + MIN, end)
    assert len(load_candles(session, SYM, "1m")) < 500
    n = download_candles(FakeExchange(n_bars=500, page_size=100), session, SYM, "1m", T0 + MIN, end)
    df = load_candles(session, SYM, "1m")
    assert len(df) == 500 and validate_candles(df, "1m").missing_bars == 0
    assert n > 0


def test_interior_hole_is_filled_and_only_the_hole_is_fetched(session: Session) -> None:
    end = T0 + 100 * MIN
    download_candles(FakeExchange(n_bars=100), session, SYM, "1m", T0 + MIN, end)
    _delete_candles(session, T0 + 40 * MIN, T0 + 44 * MIN)
    ex = FakeExchange(n_bars=100)
    assert download_candles(ex, session, SYM, "1m", T0 + MIN, end) == 5
    assert ex.candle_calls[0] == T0 + 40 * MIN
    assert len(load_candles(session, SYM, "1m")) == 100


def test_unfillable_hole_costs_one_request_per_run_and_terminates(session: Session) -> None:
    end = T0 + 100 * MIN
    download_candles(FakeExchange(n_bars=100), session, SYM, "1m", T0 + MIN, end)
    _delete_candles(session, T0 + 40 * MIN, T0 + 44 * MIN)
    for _ in range(2):
        ex = FakeExchange(n_bars=39)  # the exchange has nothing for the hole or after it
        assert download_candles(ex, session, SYM, "1m", T0 + MIN, end) == 0
        assert len(ex.candle_calls) == 1


def test_fully_stored_range_makes_no_request(session: Session) -> None:
    download_candles(FakeExchange(n_bars=50), session, SYM, "1m", T0 + MIN, T0 + 50 * MIN)
    ex = FakeExchange(n_bars=50)
    assert download_candles(ex, session, SYM, "1m", T0 + MIN, T0 + 50 * MIN) == 0
    assert ex.candle_calls == []


def test_repeating_exchange_does_not_loop_forever(session: Session) -> None:
    ex = FakeExchange(n_bars=500, page_size=100, repeat_page=True)
    download_candles(ex, session, SYM, "1m", T0, T0 + 1000 * MIN)
    assert len(ex.candle_calls) <= 2
    assert len(load_candles(session, SYM, "1m")) == 100


def test_download_funding_pages_and_resumes(session: Session) -> None:
    ex = FakeExchange(n_bars=1, page_size=4, n_funding=10)
    end = T0 + timedelta(days=30)
    assert download_funding(ex, session, SYM, T0, end) == 10
    s = load_funding(session, SYM)
    assert len(s) == 10 and s.iloc[3] == pytest.approx(0.0003)
    again = FakeExchange(n_bars=1, page_size=4, n_funding=10)
    assert download_funding(again, session, SYM, T0, end) == 0
    resumed = FakeExchange(n_bars=1, page_size=4, n_funding=12)
    assert download_funding(resumed, session, SYM, T0, end) == 2
    assert len(load_funding(session, SYM)) == 12


def test_start_not_on_bar_boundary(session: Session) -> None:
    ex = FakeExchange(n_bars=20)
    download_candles(ex, session, SYM, "1m", T0 + timedelta(seconds=30), T0 + 10 * MIN)
    df = load_candles(session, SYM, "1m")
    assert df.index[0] == T0 + MIN and df.index[-1] == T0 + 10 * MIN and len(df) == 10


def test_earlier_start_backfills_range_before_stored_bars(session: Session) -> None:
    download_candles(FakeExchange(n_bars=500), session, SYM, "1m", T0 + 300 * MIN, T0 + 500 * MIN)
    n = download_candles(FakeExchange(n_bars=500), session, SYM, "1m", T0, T0 + 500 * MIN)
    df = load_candles(session, SYM, "1m")
    assert n == 299 and len(df) == 500  # bars 1..299 added, 300..500 not refetched
    assert validate_candles(df, "1m").missing_bars == 0


def test_interrupted_funding_backfill_rolls_back_and_reruns_cleanly(session: Session) -> None:
    end = T0 + timedelta(days=30)
    download_funding(
        FakeExchange(n_bars=1, page_size=4, n_funding=20),
        session,
        SYM,
        T0 + timedelta(hours=80),
        end,
    )
    assert len(load_funding(session, SYM)) == 10
    broken = FakeExchange(n_bars=1, page_size=4, n_funding=20, fail_on_call=2)
    with pytest.raises(ccxt.NetworkError):
        download_funding(broken, session, SYM, T0, end)
    assert len(load_funding(session, SYM)) == 10  # backfill all-or-nothing: nothing kept
    n = download_funding(FakeExchange(n_bars=1, page_size=4, n_funding=20), session, SYM, T0, end)
    assert n == 10 and len(load_funding(session, SYM)) == 20


def test_funding_hole_before_listing_costs_one_request_and_full_range_none(
    session: Session,
) -> None:
    end = T0 + timedelta(days=30)
    download_funding(FakeExchange(n_bars=1, n_funding=10), session, SYM, T0, end)
    full = FakeExchange(n_bars=1, n_funding=10)
    assert download_funding(full, session, SYM, T0, end) == 0
    assert len(full.funding_calls) == 1  # the forward resume probe only
    late = FakeExchange(n_bars=1, n_funding=10)  # nothing exists before the first stored stamp
    late_start = T0 - timedelta(days=5)
    assert download_funding(late, session, SYM, late_start, end) == 0
    assert len(late.funding_calls) == 2  # one backfill probe + one forward probe, both terminate


def test_earlier_start_backfills_funding(session: Session) -> None:
    end = T0 + timedelta(days=30)
    ex = FakeExchange(n_bars=1, n_funding=10)
    download_funding(ex, session, SYM, T0 + timedelta(hours=8 * 5), end)
    assert len(load_funding(session, SYM)) == 5
    assert download_funding(FakeExchange(n_bars=1, n_funding=10), session, SYM, T0, end) == 5
    assert len(load_funding(session, SYM)) == 10


# --- ccxt adapter ---------------------------------------------------------------------------


class StubClient:
    def __init__(self, ohlcv: list[list[float]] | None = None, fail: int = 0) -> None:
        self.ohlcv = ohlcv or []
        self.fail = fail
        self.calls: list[tuple[Any, ...]] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        self.calls.append((symbol, timeframe, since, limit))
        if self.fail:
            self.fail -= 1
            raise ccxt.RequestTimeout("boom")
        return self.ohlcv

    def fetch_funding_rate_history(self, symbol: str, since: int, limit: int) -> list[dict]:
        return [
            {"timestamp": int(T0.timestamp() * 1000), "fundingRate": 0.0001},
            {"timestamp": int(T0.timestamp() * 1000) + 1, "fundingRate": None},
        ]

    def fetch_order_book(self, symbol: str, limit: int) -> dict[str, Any]:
        return {"bids": [[99.5, 1]], "asks": [[100.5, 1]], "timestamp": _ms(T0)}


def _ms(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def test_ccxt_open_time_becomes_close_time_and_forming_bar_dropped() -> None:
    now = T0 + 3 * MIN + CLOSE_GRACE + timedelta(seconds=30)  # bar opening at +3m is still forming
    rows = [[_ms(T0 + i * MIN), 1, 2, 0.5, 1.5, 7] for i in range(4)]
    ex = CcxtExchange("binanceusdm", client=StubClient(rows), now=lambda: now)
    bars = ex.fetch_candles(SYM, "1m", T0 + MIN, 1000)
    assert [b.ts for b in bars] == [T0 + MIN, T0 + 2 * MIN, T0 + 3 * MIN]
    assert bars[0].open == 1 and bars[0].high == 2 and bars[0].low == 0.5
    assert bars[0].close == 1.5 and bars[0].volume == 7 and bars[0].symbol == SYM
    # since (close time) is passed to ccxt as the open time of the first wanted bar
    assert ex._client.calls[0][2] == _ms(T0)


def test_ccxt_bar_closing_exactly_at_the_cutoff_is_kept() -> None:
    rows = [[_ms(T0), 1, 2, 0.5, 1.5, 7]]
    ex = CcxtExchange("binanceusdm", client=StubClient(rows), now=lambda: T0 + MIN + CLOSE_GRACE)
    assert len(ex.fetch_candles(SYM, "1m", T0 + MIN, 10)) == 1


def test_bar_inside_grace_margin_is_not_closed() -> None:
    rows = [[_ms(T0), 1, 2, 0.5, 1.5, 7]]
    just_inside = T0 + MIN + CLOSE_GRACE - timedelta(milliseconds=1)
    ex = CcxtExchange("binanceusdm", client=StubClient(rows), now=lambda: just_inside)
    assert ex.fetch_candles(SYM, "1m", T0 + MIN, 10) == []


def test_cutoff_is_read_before_the_request() -> None:
    """The clock passes close + grace during the request; only a pre-request read drops the bar."""
    clock = [T0 + timedelta(seconds=59, milliseconds=950)]

    class SlowClient(StubClient):
        def fetch_ohlcv(self, *a: Any) -> list[list[float]]:
            clock[0] = T0 + timedelta(seconds=62, milliseconds=100)  # beyond CLOSE_GRACE
            return self.ohlcv

    rows = [[_ms(T0), 1, 2, 0.5, 1.5, 7]]  # open 00:00 -> ts 00:01
    ex = CcxtExchange("binanceusdm", client=SlowClient(rows), now=lambda: clock[0])
    assert ex.fetch_candles(SYM, "1m", T0 + MIN, 10) == []


def test_bars_before_since_are_filtered() -> None:
    rows = [[_ms(T0 + i * MIN), 1, 2, 0.5, 1.5, 7] for i in range(3)]  # ts +1m,+2m,+3m
    ex = CcxtExchange("binanceusdm", client=StubClient(rows), now=lambda: T0 + 10 * MIN)
    assert [b.ts for b in ex.fetch_candles(SYM, "1m", T0 + 2 * MIN, 10)] == [
        T0 + 2 * MIN,
        T0 + 3 * MIN,
    ]


def test_ccxt_funding_and_quote_parsing() -> None:
    ex = CcxtExchange("binanceusdm", client=StubClient(), now=lambda: T0)
    assert ex.fetch_funding(SYM, T0, 10) == [(T0, 0.0001)]  # null rate skipped
    q = ex.fetch_quote(SYM)
    assert (q.bid, q.ask, q.ts) == (99.5, 100.5, T0)  # exchange timestamp wins


def test_quote_falls_back_to_local_time_without_exchange_timestamp() -> None:
    client = StubClient()
    client.fetch_order_book = lambda s, n: {"bids": [[1, 1]], "asks": [[2, 1]], "timestamp": None}  # type: ignore[method-assign]  # stub override
    ex = CcxtExchange("binanceusdm", client=client, now=lambda: T0 + MIN)
    assert ex.fetch_quote(SYM).ts == T0 + MIN


def test_retry_then_success_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    client = StubClient([[_ms(T0), 1, 2, 0.5, 1.5, 7]], fail=3)
    ex = CcxtExchange(
        "binanceusdm", client=client, sleep=sleeps.append, now=lambda: T0 + MIN + CLOSE_GRACE
    )
    assert len(ex.fetch_candles(SYM, "1m", T0 + MIN, 10)) == 1
    assert sleeps == [1.0, 2.0, 4.0]


def test_retry_exhaustion_raises() -> None:
    sleeps: list[float] = []
    client = StubClient(fail=99)
    ex = CcxtExchange("binanceusdm", client=client, sleep=sleeps.append)
    with pytest.raises(ccxt.NetworkError):
        ex.fetch_candles(SYM, "1m", T0, 10)
    assert len(client.calls) == 6 and len(sleeps) == 5


# --- validation -----------------------------------------------------------------------------


def test_clean_frame_is_ok() -> None:
    r = validate_candles(_frame(), "1m")
    assert r.ok and r.bars == r.expected_bars == 10 and r.gaps == ()


def test_gaps_listed_as_ranges() -> None:
    df = _frame(20).drop(_frame(20).index[[3, 4, 5, 10]])
    r = validate_candles(df, "1m")
    ts = _frame(20).index
    assert r.gaps == ((ts[3].to_pydatetime(), ts[5].to_pydatetime()),) + (
        (ts[10].to_pydatetime(), ts[10].to_pydatetime()),
    )
    assert r.missing_bars == 4 and r.expected_bars == 20
    assert not r.ok  # 20% missing


def test_tiny_gap_share_is_still_ok() -> None:
    df = _frame(5000)
    r = validate_candles(df.drop(df.index[100]), "1m")  # 1 / 5000 = 0.02%
    assert r.missing_bars == 1 and r.ok


def test_duplicates_counted() -> None:
    df = _frame()
    r = validate_candles(pd.concat([df, df.iloc[[2]]]), "1m")
    assert r.duplicates == 1 and not r.ok


def test_bad_ohlc_counted() -> None:
    df = _frame()
    df.iloc[1, df.columns.get_loc("high")] = 0.0  # high below open/close
    df.iloc[2, df.columns.get_loc("low")] = 500.0  # low above open/close
    r = validate_candles(df, "1m")
    assert r.bad_ohlc == 2 and not r.ok


def test_non_positive_prices_counted() -> None:
    df = _frame()
    df.iloc[0, df.columns.get_loc("low")] = 0.0
    df.iloc[1, df.columns.get_loc("low")] = -1.0
    r = validate_candles(df, "1m")
    assert r.non_positive_prices == 2 and not r.ok


def test_negative_volume_counted() -> None:
    df = _frame()
    df.iloc[4, df.columns.get_loc("volume")] = -1.0
    r = validate_candles(df, "1m")
    assert r.negative_volume == 1 and not r.ok


def test_empty_frame_not_ok() -> None:
    assert not validate_candles(_frame(0), "1m").ok


# --- resample -------------------------------------------------------------------------------


def test_resample_labels_by_close_time() -> None:
    df = _frame(15, start=datetime(2024, 1, 1, 0, 1, tzinfo=UTC))  # closes 00:01..00:15
    out = resample(df, "1m", "15m")
    assert list(out.index) == [datetime(2024, 1, 1, 0, 15, tzinfo=UTC)]
    row = out.iloc[0]
    assert row["open"] == 100 and row["close"] == 114
    assert row["high"] == 115 and row["low"] == 99 and row["volume"] == 15


def test_resample_bar_closing_on_boundary_belongs_to_that_bucket() -> None:
    df = _frame(16, start=datetime(2024, 1, 1, 0, 1, tzinfo=UTC))  # 00:01..00:16
    out = resample(df, "1m", "15m")
    assert len(out) == 1  # 00:16 starts the next bucket, which is incomplete


def test_resample_drops_incomplete_buckets_without_filling() -> None:
    df = _frame(45, start=datetime(2024, 1, 1, 0, 1, tzinfo=UTC))  # 00:01..00:45
    df = df.drop(df.index[20])  # 00:21 missing: breaks the 00:30 bucket
    out = resample(df, "1m", "15m")
    assert list(out.index.strftime("%H:%M")) == ["00:15", "00:45"]


def test_resample_partial_leading_bucket_dropped() -> None:
    df = _frame(20, start=datetime(2024, 1, 1, 0, 6, tzinfo=UTC))  # 00:06..00:25
    assert list(resample(df, "1m", "15m").index.strftime("%H:%M")) == []


def test_resample_empty_frame() -> None:
    assert resample(_frame(0), "1m", "15m").empty


def test_resample_rejects_non_multiple() -> None:
    with pytest.raises(ValueError):
        resample(_frame(), "15m", "5m")


# --- CLI ------------------------------------------------------------------------------------


# Feature windows short enough that the warm-up span (48 decision bars, 12 h) rounds up to one
# day: history must start two days before the backtest.
SHORT_WINDOWS = {"features.pctile_window": 24, "features.rv_window": 24, "features.ema_slow": 48}
SHORT_WINDOWS |= {f"features.{w}_window": 24 for w in ("volume", "donchian", "zscore")}
SHORT_WINDOWS["features.ema_fast"] = 12


def _narrow_cfg(monkeypatch: pytest.MonkeyPatch, start: str, end: str) -> AppConfig:
    """Config whose backtest range is tiny, with short feature windows, patched into the CLI."""
    cfg = load_config(overrides={"backtest.start": start, "backtest.end": end, **SHORT_WINDOWS})
    monkeypatch.setattr("regimebot.cli.load_config", lambda: cfg)
    return cfg


def test_history_starts_the_feature_warm_up_before_the_backtest_rounded_to_days() -> None:
    # default: 2976 decision bars of 15m are exactly 31 days, plus one day of margin
    assert history_start(load_config()) == datetime(2023, 11, 30, tzinfo=UTC)
    # 48 bars = 12 h, rounded up to one day, plus one
    cfg = load_config(overrides={"backtest.start": "2024-03-10T06:00:00Z", **SHORT_WINDOWS})
    assert history_start(cfg) == datetime(2024, 3, 8, 6, tzinfo=UTC)


def _store_flat(eng: Engine, cfg: AppConfig, start: datetime, n: int) -> None:
    idx = pd.date_range(start, periods=n, freq="1min", name="ts")
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.0, "volume": 1.0}, index=idx
    )
    with session_scope(eng) as s:
        for sym in cfg.exchange.symbols:
            upsert_candles(s, sym, "1m", df)


def test_data_check_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    cfg = _narrow_cfg(monkeypatch, "2024-01-03T00:00:00Z", "2024-01-03T01:40:00Z")
    runner = CliRunner()
    assert runner.invoke(app, ["data", "check"]).exit_code == 1  # no data: not ok
    start = history_start(cfg)  # 2024-01-01
    _store_flat(eng, cfg, start, (cfg.backtest.end - start) // MIN)
    res = runner.invoke(app, ["data", "check"])
    assert res.exit_code == 0, res.output
    assert "OK" in res.output and f"first={start.isoformat()}" in res.output
    eng.dispose()


def test_data_check_fails_without_warm_up_history_and_says_what_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'warm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    cfg = _narrow_cfg(monkeypatch, "2024-01-03T00:00:00Z", "2024-01-03T01:40:00Z")
    _store_flat(eng, cfg, cfg.backtest.start, 100)  # the backtest range only
    res = CliRunner().invoke(app, ["data", "check"])
    assert res.exit_code == 1
    assert "shorter than the configured range" in res.output
    assert (
        "the backtest needs 1m history from 2024-01-01T00:00:00+00:00 (feature warm-up) to "
        "2024-01-03T01:40:00+00:00: run `regimebot data download` to fetch it"
    ) in res.output
    eng.dispose()


def test_data_download_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"sqlite:///{tmp_path / 'dl.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    fake = FakeExchange(n_bars=30, n_funding=2)
    monkeypatch.setattr("regimebot.cli.CcxtExchange", lambda exchange_id: fake)
    res = CliRunner().invoke(
        app, ["data", "download", "--start", "2024-01-01", "--end", "2024-01-01T00:10:00Z"]
    )
    assert res.exit_code == 0, res.output
    with session_scope(eng) as s:
        for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"):
            assert len(load_candles(s, sym, "1m")) == 10
            assert len(load_funding(s, sym)) == 1  # only the 00:00 event is <= end
    assert "BTC/USDT:USDT: 10 candles, 1 funding rows" in res.output
    eng.dispose()


def test_data_download_defaults_to_the_warm_up_before_the_backtest_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'dlw.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    cfg = _narrow_cfg(monkeypatch, "2024-01-03T00:00:00Z", "2024-01-03T00:10:00Z")
    assert history_start(cfg) == T0  # two days before the backtest start
    fake = FakeExchange(n_bars=2 * 1440 + 10, page_size=1000, n_funding=7)
    monkeypatch.setattr("regimebot.cli.CcxtExchange", lambda exchange_id: fake)
    res = CliRunner().invoke(app, ["data", "download"])
    assert res.exit_code == 0, res.output
    assert fake.candle_calls[0] == fake.funding_calls[0] == T0
    with session_scope(eng) as s:
        candles = load_candles(s, SYM, "1m")
        assert (candles.index[0], len(candles)) == (T0 + MIN, 2 * 1440 + 10)
        assert len(load_funding(s, SYM)) == 7  # 00:00 every 8 h from T0 to the start
    eng.dispose()


def test_data_check_fails_on_short_history_and_prints_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite:///{tmp_path / 'sh.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    cfg = _narrow_cfg(monkeypatch, "2024-01-03T00:00:00Z", "2024-01-03T03:00:00Z")
    df = _frame(100, start=history_start(cfg) + timedelta(hours=1))  # starts an hour late
    with session_scope(eng) as s:
        for sym in cfg.exchange.symbols:
            upsert_candles(s, sym, "1m", df)
    res = CliRunner().invoke(app, ["data", "check"])
    assert res.exit_code == 1
    assert "bars=100 first=" in res.output and "shorter" in res.output
    eng.dispose()


@pytest.mark.parametrize(
    ("first_missing", "last_missing", "expected_exit"),
    [(0, 0, 0), (1, 0, 1), (0, 1, 1)],  # complete, first bar missing, last bar missing
)
def test_data_check_edge_bars_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_missing: int,
    last_missing: int,
    expected_exit: int,
) -> None:
    url = f"sqlite:///{tmp_path / 'edge.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    cfg = _narrow_cfg(monkeypatch, "2024-01-03T00:00:00Z", "2024-01-03T01:40:00Z")
    start = history_start(cfg)  # the first bar closes here, the last one a minute before end
    total = (cfg.backtest.end - start) // MIN
    _store_flat(eng, cfg, start + first_missing * MIN, total - first_missing - last_missing)
    assert CliRunner().invoke(app, ["data", "check"]).exit_code == expected_exit
    eng.dispose()
