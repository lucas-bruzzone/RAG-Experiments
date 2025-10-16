# Plano de Estudos: RAG Avançado em 4 Semanas

## Objetivo

Revisar e validar seu domínio prático dos conceitos de RAG avançado em 4 semanas, com ênfase em:

- Arquitetura de RAG híbrido (semântico + keyword)
- LlamaIndex para orquestração
- Qdrant e vetorização eficiente
- Reranking, query expansion e avaliação (RAGAS, TruLens)

---

## Estrutura do Programa

### Semana 1 — Fundamentos e Implementação Base de RAG

**Objetivo:** Garantir domínio sobre o pipeline RAG completo.

#### Conceitos

- O que é Retrieval-Augmented Generation e por que é usado
- Diferença entre retrieval semântico, keyword e híbrido
- Estrutura geral: Document Loader → Splitter → Embedding → Vector Store → Retriever → LLM → Response Synthesizer

#### Prática

1. Implementar um RAG básico em Python:
   - Use `LlamaIndex` com `OpenAI embeddings (text-embedding-3)` e um retriever simples
   - Fonte: [LlamaIndex Getting Started Guide](https://docs.llamaindex.ai/)
2. Testar chunking com diferentes tamanhos e comparar qualidade de resposta

#### Autoavaliação

- Consigo explicar cada componente do pipeline RAG?
- Entendo o que são embeddings e como o tamanho do chunk afeta o contexto?

---

### Semana 2 — Retrieval Híbrido e Reranking

**Objetivo:** Dominar a parte de precisão e relevância na recuperação.

#### Conceitos

- Retrieval híbrido (BM25 + semântico): como combinar pontuações
- Reranking e fusion retrieval (re-ranqueamento pós-busca)
- Introdução a query expansion e query rewriting

#### Prática

1. Implementar um retriever híbrido com:
   - `BM25` (usando `rank_bm25` ou `Elasticsearch`)
   - Embeddings (usando `Qdrant` ou `FAISS`)
   - Combinar resultados (fusion score ou interleaving)
2. Aplicar reranking com `cross-encoder` (modelo do Hugging Face: `cross-encoder/ms-marco-MiniLM-L-6-v2`)

#### Autoavaliação

- Sei ajustar o peso relativo entre BM25 e embeddings?
- Consigo medir o impacto de reranking nos resultados?

---

### Semana 3 — LlamaIndex Avançado e Qdrant

**Objetivo:** Consolidar a parte de infraestrutura e otimização.

#### Conceitos

- Hierarchical retrieval (retrieval em múltiplos níveis de contexto)
- Memory e context management no LlamaIndex
- Query planning e composition graphs
- Fundamentos de vector databases (Qdrant, Pinecone, Milvus)

#### Prática

1. Criar um pipeline com:
   - `LlamaIndex` + `Qdrant` como vector store
   - Implementar hierarchical retrieval com dois níveis de contexto
2. Explorar parâmetros do Qdrant (`cosine`, `dot`, `euclidean`) e otimizar a similaridade

#### Autoavaliação

- Sei explicar como o Qdrant armazena e busca embeddings?
- Sei configurar um retriever com múltiplos contextos e níveis hierárquicos?

---

### Semana 4 — Avaliação, Métricas e Projeto Final

**Objetivo:** Validar resultados e consolidar aprendizado prático.

#### Conceitos

- Avaliação de sistemas RAG: Precision@K, Recall@K, Context Relevance
- Ferramentas: RAGAS e TruLens

#### Prática

1. Rodar uma avaliação com `RAGAS` em um pequeno dataset (por exemplo, perguntas sobre jogos)
2. Analisar métricas e ajustar:
   - Chunk size
   - Número de documentos recuperados (K)
   - Balanceamento semântico vs keyword
3. Criar um mini-projeto:
   - "FAQ Bot para jogos" usando `LlamaIndex + Qdrant + OpenAI API`
   - Adicionar reranking e query expansion

#### Autoavaliação

- Sei medir a qualidade do meu RAG?
- Sei justificar ajustes técnicos com base em métricas?

---

## Recursos Recomendados

| Tema | Recurso |
|------|---------|
| Conceitos RAG | [Understanding RAG Systems (LlamaIndex)](https://docs.llamaindex.ai/) |
| Retrieval híbrido | [Dense vs Sparse Retrieval Explained (Pinecone)](https://www.pinecone.io/learn/) |
| Reranking | [Hugging Face Cross-Encoder Models](https://huggingface.co/cross-encoder) |
| LlamaIndex Avançado | [LlamaIndex Advanced Retrieval Cookbook](https://docs.llamaindex.ai/en/stable/examples/) |
| Qdrant | [Qdrant Docs — Tutorials](https://qdrant.tech/documentation/) |
| Avaliação RAG | [RAGAS GitHub](https://github.com/explodinggradients/ragas) / [TruLens.ai](https://www.trulens.org/) |

---

## Pré-requisitos

- Python 3.8+
- Conhecimento básico de Machine Learning e NLP
- API Key da OpenAI (para embeddings e LLM)
- Familiaridade com conceitos de embeddings e similaridade vetorial

---

## Como Usar Este Guia

1. Dedique pelo menos 5-10 horas por semana ao estudo
2. Siga a ordem das semanas para construir conhecimento progressivo
3. Complete as práticas antes de avançar para a próxima semana
4. Use as autoavaliações para identificar pontos fracos
5. Mantenha um repositório Git com seus experimentos

---

## Resultados Esperados

Ao final das 4 semanas, você será capaz de:

- Implementar um sistema RAG completo do zero
- Combinar múltiplas estratégias de retrieval
- Avaliar e otimizar performance de sistemas RAG
- Integrar componentes avançados como reranking e query expansion
- Tomar decisões arquiteturais baseadas em métricas

---

## Licença

Este material é de código aberto para fins educacionais.

## Contribuições

Sinta-se livre para adicionar recursos, exemplos de código ou melhorias ao plano de estudos.