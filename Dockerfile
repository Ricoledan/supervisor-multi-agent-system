# =======================
# Base Image
# =======================

FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app

# Copy requirements first for better caching
COPY requirements.txt requirements.txt

# Install Python dependencies with more verbose output for debugging
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt

# Change to app user
USER appuser

# =======================
# Development
# =======================

FROM base AS dev
USER appuser