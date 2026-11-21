FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY regimebot ./regimebot
COPY config ./config
COPY alembic.ini ./
ARG EXTRAS=postgres
# editable: the default config is located relative to the package source
RUN pip install -e ".[${EXTRAS}]"

RUN useradd --create-home app && mkdir -p data reports && chown app data reports
USER app

ENTRYPOINT ["regimebot"]
