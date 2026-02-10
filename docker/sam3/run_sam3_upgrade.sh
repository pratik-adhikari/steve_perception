#!/bin/bash
# Script to build and run SAM3 Docker with text-prompted segmentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  SAM3 Server Setup (Real SAM3!)"
echo "=========================================="

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HF_TOKEN not set!"
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "Get your token from:"
    echo "  https://huggingface.co/settings/tokens"
    echo ""
    echo "Ensure you have access to:"
    echo "  https://huggingface.co/facebook/sam3"
    echo ""
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Check for NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "Warning: NVIDIA Docker not available or GPU not detected"
    echo "SAM3 will run on CPU (very slow)"
fi

# Stop old container if running
echo "Stopping old containers..."
docker compose -f docker-compose.yml down 2>/dev/null || true
docker compose -f docker-compose-sam3.yml down 2>/dev/null || true

# Build SAM3 container
echo ""
echo "Building SAM3 container (this may take 10-15 minutes on first run)..."
docker compose -f docker-compose-sam3.yml build

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

# Start container
echo ""
echo "Starting SAM3 server..."
docker compose -f docker-compose-sam3.yml up -d

echo ""
echo "=========================================="
echo "SAM3 server is starting..."
echo "  - Health check: curl http://localhost:5005/health"
echo "  - Logs: docker compose -f docker-compose-sam3.yml logs -f sam3"
echo "  - Stop: docker compose -f docker-compose-sam3.yml down"
echo ""
echo "Wait ~2-3 minutes for model to load, then test with:"
echo "  python3 ~/steve_ros2_ws/sam3_text_query.py image.jpg 'bottle'"
echo "=========================================="
