# RAG System - Retrieval Augmented Generation

Complete RAG (Retrieval-Augmented Generation) system with modular architecture, supporting multiple LLM providers.

## Features

- Document processing (PDF, TXT)
- Vector storage with ChromaDB
- Semantic search with sentence-transformers
- Multiple LLM support (Ollama, OpenAI)
- CLI and programmatic interfaces
- Dev Container ready

## Quick Start

### Using DevContainer (Recommended)

1. Open in VS Code with Dev Containers extension
2. Wait for automatic setup (downloads tinyllama model)
3. Start using:

```bash
python cli.py stats
python cli.py chat
```

### Manual Setup

```bash
# Install dependencies
pip install -e .

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull tinyllama

# Index documents
python scripts/index.py --create-sample
# or
python scripts/index.py --dir data/raw

# Use CLI
python cli.py chat
```

## CLI Commands

```bash
# Index documents
python cli.py index --dir data/raw
python cli.py index --file document.pdf

# Search (retrieval only)
python cli.py search "your query"

# Query (retrieval + generation)
python cli.py query "What is RAG?" --show-context

# Interactive chat
python cli.py chat

# Statistics
python cli.py stats
```

## Configuration

Edit `config/settings.yaml` to customize:

- Model selection (tinyllama, llama2, qwen2.5:0.5b)
- Chunk size and overlap
- Top-k retrieval
- Vector store settings

## LLM Providers

### Ollama (Default)

```bash
# Models by memory requirement:
# - qwen2.5:0.5b (352MB, 800MB RAM)
# - tinyllama (637MB, 1.1GB RAM)
# - llama2 (3.8GB, 6GB RAM)

ollama pull tinyllama
```

### OpenAI

Create `.env`:
```
OPENAI_API_KEY=your_key_here
```

Set in `config/settings.yaml`:
```yaml
llm:
  openai:
    use_if_available: true
```

Or use CLI:
```bash
python cli.py query "question" --llm openai
```

## Programmatic Usage

```python
from src.utils.config import load_config
from src.rag.chain import RAGChain

config = load_config()
rag = RAGChain(config)

result = rag.query("What is RAG?")
print(result['answer'])
```

## Project Structure

```
RAG-Experiments/
├── config/              # Configuration
├── data/
│   ├── raw/            # Original documents
│   └── indexed/        # Indexing metadata
├── db/chroma/          # Vector database
├── src/
│   ├── core/           # Core components
│   ├── processing/     # Document processing
│   ├── rag/            # RAG logic
│   └── utils/          # Utilities
├── scripts/            # Helper scripts
├── cli.py              # CLI interface
└── setup.py            # Package setup
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Format code
black .

# Run tests
pytest
```

## Requirements

- Python 3.8+
- 2GB RAM minimum (for tinyllama)
- 6GB RAM for llama2

## License

Open source for educational purposes.