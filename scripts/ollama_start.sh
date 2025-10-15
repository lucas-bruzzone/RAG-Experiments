#!/bin/bash
# Helper para gerenciar Ollama

case "$1" in
  start)
    echo "Iniciando Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    echo "✓ Ollama iniciado"
    ;;
  stop)
    echo "Parando Ollama..."
    pkill ollama
    echo "✓ Ollama parado"
    ;;
  status)
    if pgrep -x ollama > /dev/null; then
      echo "✓ Ollama rodando"
      ollama list
    else
      echo "✗ Ollama não está rodando"
      echo "Execute: bash scripts/ollama_start.sh start"
    fi
    ;;
  pull)
    if [ -z "$2" ]; then
      echo "Uso: bash scripts/ollama_start.sh pull <modelo>"
      echo "Exemplo: bash scripts/ollama_start.sh pull llama2"
      exit 1
    fi
    ollama pull "$2"
    ;;
  *)
    echo "Uso: bash scripts/ollama_start.sh {start|stop|status|pull <modelo>}"
    exit 1
    ;;
esac
