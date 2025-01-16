# Use Python 3.8 as base image
FROM python:3.8-slim

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY setup.py .
COPY README.md .
COPY whisper_jax ./whisper_jax

# Install Python dependencies
RUN pip install --no-cache-dir ".[endpoint]"

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"] 