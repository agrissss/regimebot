FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml requirements.lock ./
COPY regimebot ./regimebot
COPY config ./config
COPY alembic.ini ./
ARG EXTRAS=postgres
# editable: the default config is located relative to the package source; the lock pins the
# versions CI tested (psycopg, from the postgres extra, is not in it)
RUN pip install -c requirements.lock -e ".[${EXTRAS}]"

# the image has no .git: runs record this as their git_sha (docker compose passes it through)
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

RUN useradd --create-home app && mkdir -p data reports && chown app data reports
USER app

ENTRYPOINT ["regimebot"]
