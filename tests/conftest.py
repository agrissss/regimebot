"""Shared fixtures."""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@pytest.fixture(autouse=True)
def _default_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests load ``config/default.yaml`` (or the file they name), whatever the shell exports."""
    monkeypatch.delenv("REGIMEBOT_CONFIG", raising=False)


@pytest.fixture
def database_url(tmp_path: Path) -> Iterator[str]:
    """URL of a fresh, empty database for one test.

    Default: a SQLite file under ``tmp_path`` (in a directory that does not exist yet). With
    ``TEST_DATABASE_URL`` set (the CI Postgres job, tests marked ``db``): a new schema in that
    Postgres database, made the connection's ``search_path`` through the URL, so the code
    under test (``make_engine``, Alembic, the CLI via ``DATABASE_URL``) needs no change; the
    schema and everything in it is dropped afterwards, so the job can be re-run.
    """
    base = os.environ.get("TEST_DATABASE_URL")
    if not base:
        yield f"sqlite:///{tmp_path / 'db' / 'test.db'}"
        return
    # Postgres only from here on
    schema = f"test_{uuid.uuid4().hex}"
    admin = create_engine(base)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    url = make_url(base).update_query_dict({"options": f"-csearch_path={schema}"})
    try:
        yield url.render_as_string(hide_password=False)
    finally:
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
