FROM python:3.11-slim AS base

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

COPY . /app

FROM base
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
WORKDIR /app

# Set environment variables for streamable HTTP
ENV MCP_SERVER_HOST=0.0.0.0
ENV MCP_SERVER_PORT=8080

EXPOSE 8080

# Default command - can be overridden
ENTRYPOINT ["python", "-m", "graphistry_mcp_server.server", "--http"]