FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY regimebot ./regimebot
COPY config ./config
ARG EXTRAS=postgres
RUN pip install ".[${EXTRAS}]"

RUN useradd --create-home app && mkdir -p data && chown app data
USER app

ENTRYPOINT ["regimebot"]
