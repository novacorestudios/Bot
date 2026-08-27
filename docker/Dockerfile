# syntax=docker/dockerfile:1
# --------------------------------------------------------------------------- #
# Multi-stage build. The runtime image contains no build toolchain, runs as a
# non-root user, and carries no credentials — those arrive via the environment.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    TZ=UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin tradebot

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=tradebot:tradebot src/ ./src/
COPY --chown=tradebot:tradebot config/ ./config/
COPY --chown=tradebot:tradebot scripts/ ./scripts/
COPY --chown=tradebot:tradebot docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN mkdir -p /app/data /app/logs /app/reports \
 && chown -R tradebot:tradebot /app \
 && chmod +x /usr/local/bin/entrypoint.sh

USER tradebot

EXPOSE 8080

# The dashboard's health endpoint reports component status, not just liveness:
# a bot that is up but disconnected from Binance is NOT healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${DASHBOARD_PORT:-8080}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["run"]
