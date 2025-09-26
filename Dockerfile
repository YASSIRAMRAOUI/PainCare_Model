# PainCare AI Model - Production Dockerfile
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc-dev \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in stages to reduce space usage
COPY requirements-minimal.txt ./requirements.txt

# Split dependencies into smaller chunks and clean between installs
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir numpy>=1.24.0 pandas>=2.0.0 scipy>=1.11.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install ML frameworks
RUN pip install --no-cache-dir scikit-learn>=1.3.0 joblib>=1.3.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install deep learning frameworks (largest packages)
RUN pip install --no-cache-dir --no-deps tensorflow>=2.13.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

RUN pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install remaining dependencies
RUN pip install --no-cache-dir transformers>=4.30.0 lime>=0.2.0 shap>=0.42.0 \
    matplotlib>=3.7.0 plotly>=5.15.0 seaborn>=0.12.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install web framework dependencies
RUN pip install --no-cache-dir fastapi>=0.103.0 uvicorn>=0.23.0 pydantic>=2.3.0 \
    python-jose>=3.3.0 python-multipart>=0.0.6 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install Firebase and other dependencies
RUN pip install --no-cache-dir firebase-admin>=6.2.0 python-dateutil>=2.8.0 pytz>=2023.3 \
    cryptography>=41.0.0 bcrypt>=4.0.0 passlib>=1.7.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install remaining dependencies
RUN pip install --no-cache-dir requests>=2.31.0 httpx>=0.24.0 aiohttp>=3.8.0 aiofiles>=23.2.0 \
    python-dotenv>=1.0.0 pyyaml>=6.0.0 loguru>=0.7.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install development and notebook dependencies
RUN pip install --no-cache-dir pytest>=7.4.0 pytest-asyncio>=0.21.0 pytest-cov>=4.1.0 \
    black>=23.7.0 flake8>=6.0.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Install Flask and related dependencies
RUN pip install --no-cache-dir flask>=3.0.0 flask-socketio>=5.3.0 psutil>=5.9.0 \
    jinja2>=3.1.0 werkzeug>=3.0.0 eventlet>=0.33.0 gputil>=1.4.0 \
    && rm -rf /tmp/* /var/tmp/* /root/.cache

# Final cleanup
RUN apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache

# Copy project
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
