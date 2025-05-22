FROM ubuntu:22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    pkg-config \
    libsdl2-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the whisper.cpp source code
COPY . .

# Build whisper.cpp with optimizations
RUN WHISPER_CFLAGS="-O3 -march=native" make -j$(nproc)

# Create models directory
RUN mkdir -p models

# Download base model
RUN bash ./models/download-ggml-model.sh base.en

# Make sure whisper-cli is executable
RUN chmod +x build/bin/whisper-cli

# Install Python dependencies
RUN pip3 install flask werkzeug

# Copy API server
COPY api.py .

# Expose the API port
EXPOSE 5000

# Set up entrypoint to run the API server
ENTRYPOINT ["python3", "api.py"]