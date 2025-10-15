#!/bin/bash
# Setup Ollama no devcontainer

echo "=== Instalando Ollama ==="

# Instala Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Inicia serviço Ollama em background
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Aguarda Ollama iniciar
sleep 5

# Baixa modelo llama2 (7B - ~4GB)
echo "Baixando modelo llama2 (pode demorar alguns minutos)..."
ollama pull llama2

echo "✓ Ollama configurado com modelo llama2"
echo "Log disponível em: /tmp/ollama.log"
