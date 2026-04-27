# Multi-stage build for Python app
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Make pip more robust in container builds (slow/intermittent networks)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=15

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --timeout 180 --retries 15 -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["python", "--version"]