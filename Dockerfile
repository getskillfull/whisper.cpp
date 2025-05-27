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
    fastapi \
    uvicorn[standard] \
    python-multipart \
    numpy \
    scipy \
    wave \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-multipart

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install faster-whisper
RUN pip install --no-cache-dir faster-whisper

# Pre-download the Whisper model
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

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
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
WORKDIR /app
COPY api.py /app/
COPY index.html /app/

# Create necessary directories
RUN mkdir -p /opt/whisper/samples

# Expose port
EXPOSE 5000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]