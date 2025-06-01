# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user (optional but good practice)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser


# Add this before pip install
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    libasound-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tools
# RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies (copy instead of bind mount for portability)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Run as non-root user
USER appuser

# No port exposed — not needed for script-only jobs
# EXPOSE 8000   ← REMOVE THIS LINE

# Run your script
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "main.handler" ]
