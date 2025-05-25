FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
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
    openai-whisper

# Create necessary directories
RUN mkdir -p /opt/whisper/samples

# Set working directory
WORKDIR /app

# Copy application files
COPY api.py /app/
COPY requirements.txt /app/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python3", "api.py"]