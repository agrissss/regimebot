# ADR 0002: SQLite by default, Postgres optional

Status: accepted

## Context

The database holds candles (about 1.05 million 1-minute rows per symbol per two years), funding rates, and every run's signals, risk decisions, fills, trades, equity snapshots and paper-session state. One user reads and writes it from one machine. The system should install and run without a database server, but must not be locked into a file database.

## Decision

- The default database is a local SQLite file, `data/regimebot.db` (about 490 MB with two years of 1-minute candles for three symbols). SQLite is opened with foreign keys enforced and WAL mode.
- `DATABASE_URL` switches to Postgres. The code uses SQLAlchemy 2.0 and Alembic migrations only, with no SQLite-specific SQL, so one set of migrations serves both.
- `docker-compose.yml` runs Postgres 16 and the app (image built from `Dockerfile` with the `postgres` extra). `POSTGRES_PASSWORD` is required and is not stored in the file.
- Every trading row carries a `run_id`, so backtest, paper and any future live results never mix.
- The stored config for a run excludes `database_url`, because the URL may contain a password.

## Consequences

- A new user needs only Python and `db upgrade`.
- SQLite allows one writer. A long job holding the database (a walk-forward run, a paper session) can make a second writer wait; use Postgres to run several at once.
- Docker was not available on the development machine, so the Dockerfile and Compose file are untested there. Treat them as a starting point and verify before relying on them. In particular the Compose file does not mount a shared `data/` volume (see `docs/operations.md`).
- Postgres needs backups and a tested restore before it holds anything you cannot re-download. The SQLite default can be rebuilt by `data download`, except for stored runs and paper-session state.
