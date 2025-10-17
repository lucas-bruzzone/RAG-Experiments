#!/bin/bash
set -e

echo "=========================================="
echo "RAG Development Environment Setup"
echo "=========================================="

# [1/5] Install Poetry
echo -e "\n[1/5] Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "Poetry already installed."
fi

# Verify Poetry installation
if ! command -v poetry &> /dev/null; then
    echo "ERROR: Poetry installation failed"
    exit 1
fi
poetry --version

# [2/5] Install Python dependencies
echo -e "\n[2/5] Installing Python dependencies..."
poetry config virtualenvs.in-project true
poetry install

# [3/5] Install Ollama
echo -e "\n[3/5] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Ollama"
        exit 1
    fi
else
    echo "Ollama already installed."
fi

# [4/5] Start Ollama service
echo -e "\n[4/5] Starting Ollama service..."
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

# [5/5] Download small model
echo -e "\n[5/5] Downloading qwen2.5:0.5b model (~352MB)..."
echo "This may take a few minutes..."
ollama pull qwen2.5:0.5b
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download model"
    exit 1
fi

# Index sample document
echo -e "\nCreating and indexing sample document..."
poetry run python scripts/index.py --create-sample || \
echo "WARNING: Failed to create sample document (you can do this manually later)"

# Final output
echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Poetry environment created at: .venv/"
echo ""
echo "Quick Start:"
echo "  poetry shell                          # Activate environment"
echo "  python cli.py stats                   # Show statistics"
echo "  python cli.py search 'RAG'            # Search documents"
echo "  python cli.py query 'What is RAG?'    # Ask questions"
echo "  python cli.py chat                    # Interactive chat"
echo ""
echo "Or run directly:"
echo "  poetry run python cli.py stats"
echo ""
echo "Ollama log: /tmp/ollama.log"
echo "=========================================="
