FROM ubuntu:22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask \
    flask-socketio \
    eventlet \
    boto3 \
    numpy \
    scipy

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Whisper
RUN pip install --no-cache-dir openai-whisper

# Pre-download the Whisper model
RUN python3 -c "import whisper; whisper.load_model('base.en')"

# Create necessary directories
RUN mkdir -p /opt/whisper/samples

# Final stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and downloaded model from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/whisper /root/.cache/whisper
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
WORKDIR /app
COPY api.py /app/

# Create necessary directories
RUN mkdir -p /opt/whisper/samples

# Expose port
EXPOSE 5000

# Run the application
CMD ["python3", "api.py"]