
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency definitions first
COPY pyproject.toml uv.lock ./

# Install dependencies into the project virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application and trained model
COPY app/ ./app/

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
