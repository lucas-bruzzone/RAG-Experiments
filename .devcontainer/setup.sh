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
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
echo -e "\n[3/4] Starting Ollama service..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# Download small model
echo -e "\n[4/4] Downloading tinyllama model (~637MB)..."
echo "This may take a few minutes..."
ollama pull tinyllama

# Index sample document
echo -e "\nCreating and indexing sample document..."
python scripts/index.py --create-sample

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
