# ADR 0002: SQLite by default, Postgres optional

Status: accepted

## Context

The database holds candles (about 1.05 million 1-minute rows per symbol per two years), funding rates, and every run's signals, risk decisions, fills, trades, equity snapshots and paper-session state. One user reads and writes it from one machine. The system should install and run without a database server, but must not be locked into a file database.

## Decision

- The default database is a local SQLite file, `data/regimebot.db` (about 500 MB for two years of 1-minute candles for three symbols; each stored two-year backtest adds roughly 80 MB). SQLite is opened with foreign keys enforced and WAL mode.
- `DATABASE_URL` switches to Postgres. The code uses SQLAlchemy 2.0 and Alembic migrations only, with no SQLite-specific SQL, so one set of migrations serves both.
- `docker-compose.yml` runs Postgres 16 and the app (image built from `Dockerfile` with the `postgres` extra). `POSTGRES_PASSWORD` is required and is not stored in the file.
- Every trading row carries a `run_id`, so backtest, paper and any future live results never mix.
- The stored config for a run excludes `database_url`, because the URL may contain a password.

## Consequences

- A new user needs only Python and `db upgrade`.
- SQLite allows one writer. The long writers are `data download`, the final save of `backtest run`, and a paper session (which writes every poll). A second writer waits only for SQLite's default busy timeout (about 5 seconds) and then fails with "database is locked". Walk-forward reads candles once and writes CSVs, so it is not a long writer. Use Postgres to run several writers at once.
- The CI job that runs the database tests against Postgres 16 runs green in GitHub Actions on every push and pull request, and CI also builds the Docker image and runs `version` and `db upgrade` in it. Docker has still not been run on the author's own machine, so treat the Compose file and a paper session in a container as a starting point and verify before relying on them. `psycopg` is pinned in `requirements.lock`. The image ships `alembic.ini` and creates writable `data/` and `reports/`; Compose mounts named volumes at `/app/data` and `/app/reports`, shared by every `app` container, so the kill file and lock files are visible across containers.
- Postgres needs backups and a tested restore before it holds anything you cannot re-download. The SQLite default can be rebuilt by `data download`, except for stored runs and paper-session state.
