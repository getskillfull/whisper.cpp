#!/bin/bash
set -e

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
fi

# Login to ECR
aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin 819669852177.dkr.ecr.us-east-1.amazonaws.com

# Pull the latest image
sudo docker pull 819669852177.dkr.ecr.us-east-1.amazonaws.com/skillfull/whisper:latest

# Stop and remove existing container if it exists
sudo docker stop whisper-cpp || true
sudo docker rm whisper-cpp || true

# Create necessary directories
sudo mkdir -p /opt/whisper/models
sudo mkdir -p /opt/whisper/samples

# Run the new container
sudo docker run -d \
  --name whisper-cpp \
  --restart unless-stopped \
  -p 5000:5000 \
  -v /opt/whisper/models:/app/models \
  -v /opt/whisper/samples:/app/samples \
  819669852177.dkr.ecr.us-east-1.amazonaws.com/skillfull/whisper:latest

echo "Whisper.cpp deployed successfully!" 