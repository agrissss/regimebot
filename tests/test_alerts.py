"""Alerts: always logged at ERROR, optionally POSTed to ALERT_WEBHOOK_URL, never raising."""

import json
import urllib.error
import urllib.request
from typing import Any

import pytest
import structlog.testing

from regimebot.reporting import alerts
from regimebot.reporting.alerts import alert

URL = "https://hooks.example.test/T000/abc"


class Recorder:
    """Stands in for ``urllib.request.urlopen``; raises ``error`` when one is set."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> Any:
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return Response()


class Response:
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def urlopen(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(alerts.urllib.request, "urlopen", recorder)
    return recorder


def test_alert_logs_at_error_without_a_webhook(
    urlopen: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    with structlog.testing.capture_logs() as logs:
        alert("stale_data", "BTC/USDT:USDT: fetch failed")
    assert logs == [
        {
            "event": "alert",
            "kind": "stale_data",
            "detail": "BTC/USDT:USDT: fetch failed",
            "log_level": "error",
        }
    ]
    assert urlopen.requests == []


def test_alert_posts_kind_detail_and_ts_as_json_with_a_5s_timeout(
    urlopen: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALERT_WEBHOOK_URL", URL)
    alert("stale_data", "ETH: no bars")
    ((request, timeout),) = urlopen.requests
    assert (request.full_url, request.get_method(), timeout) == (URL, "POST", 5)
    assert request.get_header("Content-type") == "application/json"
    assert isinstance(request.data, bytes)
    body = json.loads(request.data)
    assert set(body) == {"kind", "detail", "ts"}
    assert (body["kind"], body["detail"]) == ("stale_data", "ETH: no bars")
    assert body["ts"].endswith("+00:00")  # ISO 8601 UTC


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("unreachable"), TimeoutError("timed out"), ValueError("bad url")],
)
def test_a_webhook_failure_is_logged_and_never_raised(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(alerts.urllib.request, "urlopen", Recorder(error))
    monkeypatch.setenv("ALERT_WEBHOOK_URL", URL)
    with structlog.testing.capture_logs() as logs:
        alert("poll_error", "boom")  # does not raise
    assert [e["event"] for e in logs] == ["alert", "alert_webhook_failed"]
    assert logs[1]["log_level"] == "error" and logs[1]["error"] == repr(error)
    assert URL not in json.dumps(logs)  # the URL may carry a token: never logged


def test_only_http_webhooks_are_called(urlopen: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "file:///etc/passwd")
    with structlog.testing.capture_logs() as logs:
        alert("stale_data", "x")
    assert urlopen.requests == []
    assert logs[-1]["event"] == "alert_webhook_failed"
