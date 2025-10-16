#!/bin/bash
# Download Ollama models

echo "Available models:"
echo "  1) tinyllama (637MB, 1.1GB RAM) - Recommended for testing"
echo "  2) qwen2.5:0.5b (352MB, 800MB RAM) - Smallest"
echo "  3) llama2 (3.8GB, 6GB RAM) - Better quality"
echo "  4) custom - Enter model name"
echo ""

read -p "Choose option [1-4]: " choice

case $choice in
  1)
    MODEL="tinyllama"
    ;;
  2)
    MODEL="qwen2.5:0.5b"
    ;;
  3)
    MODEL="llama2"
    ;;
  4)
    read -p "Enter model name: " MODEL
    ;;
  *)
    echo "Invalid option"
    exit 1
    ;;
esac

echo ""
echo "Downloading $MODEL..."
ollama pull $MODEL

echo ""
echo "Done! Model available for use."
echo "Update config/settings.yaml if needed:"
echo "  llm:"
echo "    ollama:"
echo "      model: \"$MODEL\""
