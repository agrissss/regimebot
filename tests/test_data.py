from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
import pytest
from fakes import T0, FakeExchange
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from regimebot.cli import app
from regimebot.data.exchange import CcxtExchange
from regimebot.data.history import download_candles, download_funding
from regimebot.data.resample import resample
from regimebot.data.validate import validate_candles
from regimebot.db.models import Base
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
    assert second.candle_calls[0] == T0 + 101 * MIN  # first bar not yet stored
    assert len(load_candles(session, SYM, "1m")) == 250


def test_rerun_is_idempotent_and_refetches_nothing_stored(session: Session) -> None:
    download_candles(FakeExchange(n_bars=50), session, SYM, "1m", T0, T0 + 50 * MIN)
    again = FakeExchange(n_bars=50)
    n = download_candles(again, session, SYM, "1m", T0, T0 + 50 * MIN)
    assert n == 0 and again.candle_calls == []
    assert len(load_candles(session, SYM, "1m")) == 50


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

    def fetch_bids_asks(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        return {symbols[0]: {"bid": 99.5, "ask": 100.5}}


def _ms(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def test_ccxt_open_time_becomes_close_time_and_forming_bar_dropped() -> None:
    now = T0 + 3 * MIN + timedelta(seconds=30)  # bar opening at +3m is still forming
    rows = [[_ms(T0 + i * MIN), 1, 2, 0.5, 1.5, 7] for i in range(4)]
    ex = CcxtExchange("binanceusdm", client=StubClient(rows), now=lambda: now)
    bars = ex.fetch_candles(SYM, "1m", T0 + MIN, 1000)
    assert [b.ts for b in bars] == [T0 + MIN, T0 + 2 * MIN, T0 + 3 * MIN]
    assert bars[0].open == 1 and bars[0].high == 2 and bars[0].low == 0.5
    assert bars[0].close == 1.5 and bars[0].volume == 7 and bars[0].symbol == SYM
    # since (close time) is passed to ccxt as the open time of the first wanted bar
    assert ex._client.calls[0][2] == _ms(T0)


def test_ccxt_bar_closing_exactly_now_is_kept() -> None:
    rows = [[_ms(T0), 1, 2, 0.5, 1.5, 7]]
    ex = CcxtExchange("binanceusdm", client=StubClient(rows), now=lambda: T0 + MIN)
    assert len(ex.fetch_candles(SYM, "1m", T0 + MIN, 10)) == 1


def test_ccxt_funding_and_quote_parsing() -> None:
    ex = CcxtExchange("binanceusdm", client=StubClient(), now=lambda: T0)
    assert ex.fetch_funding(SYM, T0, 10) == [(T0, 0.0001)]  # null rate skipped
    q = ex.fetch_quote(SYM)
    assert (q.bid, q.ask, q.ts) == (99.5, 100.5, T0)


def test_retry_then_success_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    client = StubClient([[_ms(T0), 1, 2, 0.5, 1.5, 7]], fail=3)
    ex = CcxtExchange("binanceusdm", client=client, sleep=sleeps.append, now=lambda: T0 + MIN)
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


def test_resample_rejects_non_multiple() -> None:
    with pytest.raises(ValueError):
        resample(_frame(), "15m", "5m")


# --- CLI ------------------------------------------------------------------------------------


def test_data_check_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    eng = make_engine(url)
    Base.metadata.create_all(eng)
    runner = CliRunner()
    assert runner.invoke(app, ["data", "check"]).exit_code == 1  # no data: not ok

    from regimebot.config import load_config

    cfg = load_config()
    n = 100
    idx = pd.date_range(cfg.backtest.start, periods=n, freq="1min", name="ts")
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.0, "volume": 1.0}, index=idx
    )
    with session_scope(eng) as s:
        for sym in cfg.exchange.symbols:
            upsert_candles(s, sym, "1m", df)
    res = runner.invoke(app, ["data", "check"])
    assert res.exit_code == 0, res.output
    assert "OK" in res.output
    eng.dispose()
