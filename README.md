# RAG Question Answering over Wikipedia

**Assignment 3 — Text Retrieval and Search Engines**
*Reichman University (RUNI), Semester 1, 2026*

A Retrieval-Augmented Generation (RAG) pipeline for open-domain question answering over the full Wikipedia corpus. The system retrieves relevant passages using an ensemble of BM25 and Query Likelihood (QLD), reranks with a BGE cross-encoder, and generates concise answers via a fine-tuned LLM with few-shot prompting.

**Authors:** [eforus-overseer](https://github.com/eforus-overseer-Pecani) & Raz Cohen

---

## Pipeline

```
Question
  -> Entity extraction + query variant generation
  -> Ensemble retrieval: BM25 (weight=0.7) + QLD (weight=0.3) over Wikipedia KILT index
  -> BGE cross-encoder reranking (top-10 from 30 candidates)
  -> Context assembly (top-N passages, max 800 chars)
  -> Few-shot LLM generation (4 examples + system prompt)
  -> Smart retry: if answer is suspicious, retry with expanded context
  -> Final answer
```

## Key Design Decisions

### Multi-Query Ensemble Retrieval

Instead of searching with just the raw question, the system:
1. **Extracts entities** — capitalized phrases, quoted strings, regex patterns for "who is X", "what is X"
2. **Generates query variants** — entity-only query, cleaned question, domain-specific expansions (currency, capital, actor/role, author/book)
3. **Fuses results** — score normalization + weighted combination across all variants, with a bonus for documents found by multiple variants

### BGE Cross-Encoder Reranking

After ensemble retrieval returns 30 candidates, the BGE-reranker-base cross-encoder reranks them to surface the most semantically relevant passages. This bridges the lexical-semantic gap that BM25/QLD miss.

### Smart Retry System

If the initial LLM answer is suspicious (empty, "unknown", "I don't know", single stopword, etc.), the system retries with:
- More context passages (top-5 instead of default)
- Expanded character limit (900 chars)
- Same few-shot prompt structure

This catches cases where the initially retrieved context didn't contain the answer but nearby passages do.

### Few-Shot Prompting

4 carefully chosen examples covering different question types (currency, person, founder, geography) guide the LLM to produce concise, extractive answers in 1-9 words without repeating the question.

---

## Architecture

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Index** | Pyserini `wikipedia-kilt-doc` | Full Wikipedia, prebuilt Lucene index |
| **Retrieval** | BM25 (k1=0.9, b=0.2) + QLD (mu=2000) | Lexical matching with score fusion |
| **Reranking** | BAAI/bge-reranker-base | Semantic cross-encoder reranking |
| **Generation** | Fireworks API (deployed LLM) | Few-shot answer generation |
| **Retry** | Rule-based suspicious answer detection | Robustness against retrieval failures |

## Repository Structure

```
RAG-QA-Wikipedia/
├── notebooks/
│   ├── rag-q-a-v5-8.ipynb             # Main RAG pipeline (Kaggle notebook)
│   ├── rag_pipeline_databricks.py      # Databricks version with full pipeline
│   └── rag-analysis-notebook.ipynb     # Results analysis
└── README.md
```

## How to Run

### Databricks (recommended)

1. Import `rag_pipeline_databricks.py` as a Databricks notebook
2. Set up Fireworks API key in Databricks secrets:
   ```bash
   databricks secrets create-scope --scope fireworks
   databricks secrets put --scope fireworks --key api_key
   ```
3. Upload `rag_train.csv` and `rag_test.csv` to the working directory
4. Run all cells — the Wikipedia index downloads automatically (~5GB, first run only)

### Local / Kaggle

1. Open `rag-q-a-v5-8.ipynb` in Jupyter or Kaggle
2. Install dependencies: `pip install pyserini sentence-transformers openai nltk faiss-cpu`
3. Set your Fireworks API key
4. Run all cells

## Configuration

Key parameters (tuned for the competition):

```python
BM25_K1 = 0.9               # BM25 term frequency saturation
BM25_B = 0.2                # BM25 document length normalization
QLD_MU = 2000               # QLD Dirichlet smoothing parameter
BM25_WEIGHT = 0.7           # Ensemble weight for BM25
QLD_WEIGHT = 0.3            # Ensemble weight for QLD
RETRIEVAL_K = 30            # Initial retrieval candidates
RERANK_TOP_K = 10           # Cross-encoder reranking depth
CONTEXT_TOP_N = 20          # Passages for context assembly
LLM_TEMPERATURE = 0.3       # Generation temperature
```

## Tech Stack

- **Index:** [Pyserini](https://github.com/castorini/pyserini) (Wikipedia KILT prebuilt index)
- **Reranker:** [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- **LLM:** Fireworks AI API (deployed model)
- **NLP:** NLTK (stopwords, tokenization)
- **Platform:** Databricks with GPU / Kaggle

## License

Academic project — Reichman University, 2026.

---

<!-- demo-lab:start -->
## Explore this project

[Project page & walkthrough](https://eforus-overseer.github.io/demo-lab/projects/rag-qa-wikipedia/) — Try an interactive explanation and inspect the original source and results. Browser teaching examples are distinguished from trained models.
<!-- demo-lab:end -->


<!-- codex-pixel-art -->
## Pixel art

![Pixel art for RAG-QA-Wikipedia](assets/pixel-art.svg)

A small visual signature for this repository.
