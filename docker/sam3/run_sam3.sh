#!/bin/bash
# Script to run SAM3 Docker with text-prompted segmentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  SAM3 Server (Text-Prompted Segmentation)"
echo "=========================================="

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠️  HF_TOKEN not set!"
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Ensure access to: https://huggingface.co/facebook/sam3"
    echo ""
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Build and run
echo "Building SAM3 container..."
docker compose build

echo "Starting SAM3 server on port 5005..."
docker compose up -d

echo ""
echo "SAM3 server is starting..."
echo "  - Health: curl http://localhost:5005/health"
echo "  - Logs: docker compose logs -f sam3"
echo "  - Stop: docker compose down"
echo ""
