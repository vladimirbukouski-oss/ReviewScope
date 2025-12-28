# ReviewScope Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY reviewscope_all.py .
COPY backend/ ./backend/

# Copy models (WARNING: ~2.2GB - will increase image size)
COPY models/ ./models/

# Create data directory
RUN mkdir -p /app/backend/data

# Expose port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8888/health')"

# Run the application
CMD ["python", "backend/main.py"]
