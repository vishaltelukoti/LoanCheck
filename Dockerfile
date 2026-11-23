
FROM python:3.12-slim AS builder

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/missing_value_stats.json ./models/missing_value_stats.json

USER 10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "loancheck.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
