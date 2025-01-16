FROM python:3.10-slim

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    cmake \
    curl \
    wget \
    sudo \
    gnupg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PJRT_DEVICE=TPU

# Set working directory
WORKDIR /app

# Install JAX TPU dependencies first
RUN pip install --upgrade pip && \
    pip install --upgrade jax && \
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Copy requirements first to leverage Docker cache
COPY setup.py .
COPY README.md .
COPY whisper_jax ./whisper_jax

# Install Python dependencies
RUN pip install --no-cache-dir ".[endpoint]"

# Install additional dependencies that might be helpful
RUN pip install --no-cache-dir \
    transformers==4.43.3 \
    flax \
    einops \
    optax

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"] 