# PainCare AI Model - Ultra-minimal Production Dockerfile
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install essential system dependencies only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy essential requirements
COPY requirements-essential.txt ./

# Install Python dependencies in one go with cleanup
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-essential.txt && \
    rm -rf /tmp/* /var/tmp/* /root/.cache && \
    apt-get autoremove -y gcc libc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary application files
COPY src/ ./src/
COPY templates/ ./templates/
COPY start.py ./
COPY run_server.py ./

# Create non-root user
RUN adduser --disabled-password --gecos '' --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Health check (lightweight)
HEALTHCHECK --interval=60s --timeout=15s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/v1/health/liveness')" || exit 1

# Command to run the application
CMD ["python", "run_server.py"]