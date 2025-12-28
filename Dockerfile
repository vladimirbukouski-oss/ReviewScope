# ReviewScope Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
RUN pip install --prefer-binary -r requirements.txt

# Copy application code
COPY reviewscope_all.py .
COPY backend/ ./backend/

# Models are downloaded on startup (see backend/main.py) to keep image small.

# Create data directory
RUN mkdir -p /app/backend/data

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os,requests; requests.get(f\"http://localhost:{os.getenv('PORT','8080')}/health\")"

# Run the application
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
