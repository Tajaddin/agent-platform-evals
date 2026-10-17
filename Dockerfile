# syntax=docker/dockerfile:1.7
# Two stages: build the virtualenv and the demo warehouse with uv, then copy them into a slim runtime image
# that runs as a non-root user.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# dependencies first so the layer is cached across source changes
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project --no-dev

COPY README.md LICENSE ./
COPY src ./src
COPY evals ./evals
COPY scripts ./scripts
COPY data/sample ./data/sample
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev \
 && /app/.venv/bin/python scripts/build_demo_warehouse.py --out /app/data/demo_warehouse.duckdb

FROM python:3.12-slim AS runtime
RUN groupadd -r -g 10001 agent && useradd -r -u 10001 -g agent -d /app agent
WORKDIR /app
COPY --from=build --chown=agent:agent /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENT_WAREHOUSE_PATH=/app/data/demo_warehouse.duckdb
USER agent
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
CMD ["analyst-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
