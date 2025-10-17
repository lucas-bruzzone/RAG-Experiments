#!/bin/bash
set -e

echo "=========================================="
echo "RAG Development Environment Setup"
echo "=========================================="

# Install Python dependencies
echo -e "\n[1/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

# Install Ollama
echo -e "\n[2/4] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Ollama"
        exit 1
    fi
else
    echo "Ollama already installed"
fi

# Start Ollama service
echo -e "\n[3/4] Starting Ollama service..."
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Ollama failed to start. Check /tmp/ollama.log"
        exit 1
    fi
    sleep 1
done

# Download small model
echo -e "\n[4/4] Downloading qwen2.5:0.5b model (~352MB)..."
echo "This may take a few minutes..."
ollama pull qwen2.5:0.5b

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download model"
    exit 1
fi

# Index sample document
echo -e "\nCreating and indexing sample document..."
python scripts/index.py --create-sample

if [ $? -ne 0 ]; then
    echo "WARNING: Failed to create sample document (you can do this manually later)"
fi

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  python cli.py stats          # Show statistics"
echo "  python cli.py search 'RAG'   # Search documents"
echo "  python cli.py query 'What is RAG?'  # Ask questions"
echo "  python cli.py chat           # Interactive chat"
echo ""
echo "Ollama log: /tmp/ollama.log"
echo "=========================================="