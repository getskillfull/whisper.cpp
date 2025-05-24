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

# Create required directories
RUN mkdir -p /app/models \
    && mkdir -p /app/samples \
    && mkdir -p /opt/whisper/samples \
    && chmod -R 777 /app/models \
    && chmod -R 777 /app/samples \
    && chmod -R 777 /opt/whisper/samples

# Download base model
RUN bash ./models/download-ggml-model.sh base.en

# Make sure whisper-cli is executable
RUN chmod +x build/bin/whisper-cli

# Install Python dependencies
RUN pip3 install flask werkzeug boto3 flask-socketio eventlet numpy

# Copy API server
COPY api.py .

# Expose the API port
EXPOSE 5000

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Checking required directories..."\n\
if [ ! -d "/app/models" ]; then\n\
    echo "Creating /app/models directory"\n\
    mkdir -p /app/models\n\
    chmod 777 /app/models\n\
fi\n\
\n\
if [ ! -d "/app/samples" ]; then\n\
    echo "Creating /app/samples directory"\n\
    mkdir -p /app/samples\n\
    chmod 777 /app/samples\n\
fi\n\
\n\
if [ ! -d "/opt/whisper/samples" ]; then\n\
    echo "Creating /opt/whisper/samples directory"\n\
    mkdir -p /opt/whisper/samples\n\
    chmod 777 /opt/whisper/samples\n\
fi\n\
\n\
echo "Starting Flask application..."\n\
python3 api.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Use startup script
CMD ["/app/start.sh"]