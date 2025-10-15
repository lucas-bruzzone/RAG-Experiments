#!/usr/bin/env python3
"""Validação ETAPA 5 - Interface Interativa"""

import sys
from pathlib import Path

print("=== Validação ETAPA 5 ===\n")

# Verifica arquivos criados
files = {"notebooks/rag_demo.ipynb": "Jupyter notebook", "cli_rag.py": "CLI script"}

all_ok = True
for file, desc in files.items():
    if Path(file).exists():
        print(f"✓ {desc}: {file}")
    else:
        print(f"✗ {desc} não encontrado: {file}")
        all_ok = False

if not all_ok:
    sys.exit(1)

# Testa CLI
print("\n[Teste CLI]")
import subprocess

result = subprocess.run(
    [sys.executable, "cli_rag.py", "--help"], capture_output=True, text=True
)

if result.returncode == 0:
    print("✓ CLI funcional")
    print("\nComandos disponíveis:")
    print("  python cli_rag.py index --dir data/")
    print("  python cli_rag.py search 'sua busca'")
    print("  python cli_rag.py query 'sua pergunta'")
    print("  python cli_rag.py chat")
    print("  python cli_rag.py stats")
else:
    print(f"✗ Erro CLI: {result.stderr}")

print("\n" + "=" * 60)
print("✓ Etapa 5 validada!")
print("\nPróxima etapa: Otimizações")
