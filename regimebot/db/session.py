"""Engine and session helpers, plus running migrations."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def make_engine(url: str) -> Engine:
    """Create an engine. For SQLite: create the file's directory, enforce FKs, use WAL."""
    parsed = make_url(url)
    is_sqlite = parsed.get_backend_name() == "sqlite"
    if is_sqlite and parsed.database not in (None, "", ":memory:"):
        Path(str(parsed.database)).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: object) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Session that commits on success, rolls back on error and always closes."""
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise


def upgrade_to_head(url: str) -> None:
    """Run Alembic migrations up to head against ``url``."""
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # ini interpolation
    command.upgrade(cfg, "head")
