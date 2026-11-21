"""Alembic environment. URL: programmatic ``sqlalchemy.url`` if set, else the app config."""

from alembic import context

from regimebot.config import load_config
from regimebot.db.models import Base
from regimebot.db.session import make_engine

config = context.config
url = config.get_main_option("sqlalchemy.url") or load_config().database_url
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
