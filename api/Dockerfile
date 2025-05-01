FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    procps \
    net-tools \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY api_requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r api_requirements.txt

# Copy application code
COPY ./api ./api
COPY ./utils.py ./utils.py
COPY ./src ./src

# Create directories for data
RUN mkdir -p data/wrong_classified

# Set environment variables
ENV MODEL_PATH=outputs/models/best_model.pth
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health/ || exit 1

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]