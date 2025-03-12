# =======================
# Base Image
# =======================
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# =======================
# Development
# =======================
FROM base AS dev
USER appuser
RUN pip install --no-cache-dir -r requirements.dev.txt

# =======================
# Production
# =======================
FROM base AS prod
USER appuser
COPY --chown=appuser:appuser . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]