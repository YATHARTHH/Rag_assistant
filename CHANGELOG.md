# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-16

### Added
- **Infrastructure & Storage**: migrated vector store from local Chroma to Qdrant client storage mode (`qdrant_db`) with payload indexing and scalar quantization.
- **Process Orchestration**: created `start.bat` to concurrently boot uvicorn FastAPI, Streamlit, and Celery tasks worker pool using Windows-safe execution (`-P solo`).
- **Background workers**: integrated Celery task queue with Memurai Redis for async layout-aware document ingestion.
- **Ingestion formats**: added support for parsing `.docx`, `.csv`, `.html`, and `.json` files.
- **Data protection**: added local AES-256 encryption-at-rest (Fernet) for sensitive payload text in Qdrant.
- **Security**: added JSON Web Token (JWT) session generation, password strength checks (digit, uppercase, 8 chars minimum), user role access checks, programmatic API Keys (`X-API-Key`), and rate limits (`slowapi`).
- **Retrieval Upgrades**:
  - Parent-Document retrieval (split sentences, return parent paragraph context).
  - Hypothetical Document Embeddings (HyDE) query expansion logic.
  - Reciprocal Rank Fusion (RRF) dense-sparse merging.
  - Maximal Marginal Relevance (MMR) vector candidate diversification.
  - Contextual compression utilizing LLM extraction filtering.
- **Performance**: added local SQLite embedding cache (`(text_hash -> embedding)`) and TTL-based semantic cache invalidation.
- **Observability**:
  - Added structured JSON logging formatting.
  - Added request correlation UUID headers (`X-Correlation-ID`).
  - Implemented Prometheus `/metrics` collector endpoints.
  - In-process tracing utilizing Arize Phoenix (port 6006).
- **Evaluation**: added BLEU and ROUGE-L verification calculations.
- **UX UI Frontend**:
  - RAG controller panel toggles for parent-document and HyDE.
  - Interactive file list manager with document deleting.
  - SQLite persistent session restorer and chats logger.
  - 👍/👎 feedback buttons below assistant answers.
  - Hallucination warnings banner for low faithfulness answers (< 0.70).
  - PDF Citations metadata display containing page levels.
  - Export chat history button (JSON download).
- **Test suite**: golden regression evaluator script (`test_rag.py`) running off static query test dataset.

### Changed
- Refactored `api.py` to route all chat streams over Server-Sent Events (SSE).
- Re-routed frontend app stream parser to cleanly decode SSE chunks.
- Refactored document ingestion to parse with `unstructured` layout layouts.

## [1.0.0] - 2026-07-14

### Added
- Decoupled FastAPI server and Streamlit user interface client.
- Asynchronous database ingestion background tasks.
- SQLite credentials registry and session token isolation filters.
- sentence distance semantic boundary chunking.
- Hybrid ensemble search (dense embeddings + BM25 keyword search).
- Sentence window retrieval expansion.
- Lazy-loaded Cross-Encoder reranking model.
- LLM fallback listwise reranking prompts.
- Input and Output safety guardrail classification filters.
- Real-time LLM-as-a-Judge evaluators dashboard.
