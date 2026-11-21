"""Operator alerts: an ERROR log line, and a JSON POST to ``ALERT_WEBHOOK_URL`` when set."""

import json
import os
import urllib.request
from datetime import UTC, datetime

import structlog

log = structlog.get_logger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 5


def alert(kind: str, detail: str) -> None:
    """Log ``kind``/``detail`` at ERROR and POST ``{"kind", "detail", "ts"}`` to the webhook.

    A webhook failure (bad URL, timeout, HTTP error) is logged and never raised: an alert must not
    stop the session it reports on. The URL is never logged (it may carry a token), and only
    http(s) URLs are called.
    """
    log.error("alert", kind=kind, detail=detail)
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    if not url.startswith(("https://", "http://")):
        log.error("alert_webhook_failed", kind=kind, error="ALERT_WEBHOOK_URL is not http(s)")
        return
    body = json.dumps({"kind": kind, "detail": detail, "ts": datetime.now(UTC).isoformat()})
    try:
        request = urllib.request.Request(
            url, data=body.encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_SECONDS):
            pass
    except Exception as exc:  # any failure: an alert never raises
        log.error("alert_webhook_failed", kind=kind, error=repr(exc))
