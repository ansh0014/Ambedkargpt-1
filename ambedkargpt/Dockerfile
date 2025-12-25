FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/processed

# Set Python path
ENV PYTHONPATH=/app

# Expose port if needed (for API)
EXPOSE 8000

# Default command (adjust as needed)
CMD ["python", "src/pipeline/ambedkargpt.py"]