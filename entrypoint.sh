#!/bin/bash

# Assumes that user has Docker installed and running
# Also conda, git, and Python should be installed

set -e

# Function to check if a Docker container is running
is_container_running() {
    docker ps --filter "name=$1" --filter "status=running" | grep "$1" > /dev/null
}

# Pull the nlm-ingestor Docker image
docker pull ghcr.io/nlmatics/nlm-ingestor:latest

# Run the nlm-ingestor container if not already running
if ! is_container_running "nlm-ingestor"; then
    docker run -p 5010:5001 --name nlm-ingestor -d ghcr.io/nlmatics/nlm-ingestor:latest
else
    echo "nlm-ingestor container is already running."
fi

# Clone or update the figure-extractor repository
FIGURE_EXTRACTOR_DIR="src/parsers/figure_extractor"
if [ ! -d "$FIGURE_EXTRACTOR_DIR" ]; then
    git clone https://github.com/Huang-lab/figure-extractor "$FIGURE_EXTRACTOR_DIR"
    cd "$FIGURE_EXTRACTOR_DIR"
    docker build -t figure-extractor .
    docker run -p 5001:5001 --name figure-extractor -d figure-extractor
    cd -
else
    cd "$FIGURE_EXTRACTOR_DIR"
    git pull
    cd -
    # Restart the figure-extractor container
    if is_container_running "figure-extractor"; then
        docker restart figure-extractor
    else
        docker run -p 5001:5001 --name figure-extractor -d figure-extractor
    fi
fi

# Clone or update the llmsherpa repository
LLMSHERPA_DIR="src/parsers/llmsherpa"
if [ ! -d "$LLMSHERPA_DIR" ]; then
    git clone https://github.com/nlmatics/llmsherpa "$LLMSHERPA_DIR"
    cd "$LLMSHERPA_DIR"
    pip install -e .
    cd -
else
    cd "$LLMSHERPA_DIR"
    git pull
    pip install -e .
    cd -
fi

# Add CLIP weights download
# Add conda env / faiss download with conda 1.3.17 / pip install -r requirements

# Execute the command passed to the script
exec "$@"