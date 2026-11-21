import json
from collections.abc import Iterator
from importlib.metadata import version

import pytest
import structlog
from typer.testing import CliRunner

from regimebot.cli import app
from regimebot.logging import configure_logging


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == version("regimebot")


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def test_json_logging_emits_utc_iso_timestamp(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json=True, level="INFO")
    log = structlog.get_logger("my.module")
    log.debug("hidden")
    log.info("hello", answer=42)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "hello" and record["answer"] == 42 and record["level"] == "info"
    assert record["logger"] == "my.module"
    assert record["timestamp"].endswith("Z")
