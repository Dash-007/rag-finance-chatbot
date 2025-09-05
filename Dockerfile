FROM python:3.9-slim

WORKDIR /app

# Install system deps (only once, cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p evaluation_results logs

# Expose Streamlit port
EXPOSE 8501

# Simple health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]