# 📘 Enterprise RAG Platform: Complete Technical Deep Dive & Tech Stack Rationale

Welcome to the beginner-to-advanced technical deep-dive guide for the **Enterprise RAG Platform**. This document explains the entire system architecture, algorithms, data flows, security mechanisms, mathematical foundations, code implementations, and **why every single technology was selected over alternatives**.

---

## 📌 Table of Contents

1. [Beginner's Guide: What is RAG & Why Do We Need It?](#1-beginners-guide-what-is-rag--why-do-we-need-it)
2. [Technology Choice Rationale Matrix ("Why Tech X over Y?")](#2-technology-choice-rationale-matrix-why-tech-x-over-y)
3. [High-Level System Architecture](#3-high-level-system-architecture)
4. [End-to-End Execution Flows & Diagrams](#4-end-to-end-execution-flows--diagrams)
   - [Document Ingestion & Indexing Pipeline](#document-ingestion--indexing-pipeline)
   - [Query Processing & Multi-Stage Retrieval Pipeline](#query-processing--multi-stage-retrieval-pipeline)
   - [Corrective RAG (CRAG) Web Search Fallback](#corrective-rag-crag-web-search-fallback)
   - [Bi-Directional PII Scrubbing & Stream Restoration](#bi-directional-pii-scrubbing--stream-restoration)
5. [Module-by-Module Code Architecture](#5-module-by-module-code-architecture)
   - [`api/` — API Gateway & Routing](#api--api-gateway--routing)
   - [`database/` — Vector & Relational Storage](#database--vector--relational-storage)
   - [`rag/` — Retrieval, Reranking & Evaluation](#rag--retrieval-reranking--evaluation)
   - [`security/` — Encryption, Guardrails & PII](#security--encryption-guardrails--pii)
   - [`app.py` — Streamlit Frontend UI](#apppy--streamlit-frontend-ui)
6. [Deep Dive: Algorithms Made Simple (With Analogies & Math)](#6-deep-dive-algorithms-made-simple-with-analogies--math)
   - [Paragraph-First & Parent-Child Chunking](#paragraph-first--parent-child-chunking)
   - [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
   - [BM25 Lexical Scoring](#bm25-lexical-scoring)
   - [Scalar Quantization (INT8) & HNSW Indexing](#scalar-quantization-int8--hnsw-indexing)
   - [Maximal Marginal Relevance (MMR)](#maximal-marginal-relevance-mmr)
   - [Cross-Encoder Reranking vs Bi-Encoders](#cross-encoder-reranking-vs-bi-encoders)
   - [Lost-in-the-Middle Chunk Re-ordering](#lost-in-the-middle-chunk-re-ordering)
   - [LLM-as-a-Judge Evaluation Metrics](#llm-as-a-judge-evaluation-metrics)
7. [Security, Multi-Tenancy & Data Privacy](#7-security-multi-tenancy--data-privacy)
8. [Observability, Telemetry & Cost Auditor](#8-observability-telemetry--cost-auditor)
9. [Production Deployment Checklist](#9-production-deployment-checklist)

---

## 1. Beginner's Guide: What is RAG & Why Do We Need It?

### 💡 Plain English Analogy
Imagine taking a **closed-book exam**. If the exam asks about proprietary company policies created yesterday, you will fail or guess (hallucinate) because your brain wasn't trained on those files.

Now imagine taking an **open-book exam**. Before answering a question, a smart assistant searches a library, finds the exact 3 pages containing the answer, places those 3 pages on your desk, and tells you: *"Answer the question using ONLY these 3 pages."* 

**That is Retrieval-Augmented Generation (RAG).**

```
+-------------------+       +--------------------+       +-------------------+
|  1. USER QUERY    | ----> | 2. RETRIEVE        | ----> | 3. GENERATE       |
| "What is our Q3   |       | Search DB & fetch  |       | LLM reads context |
|  revenue target?" |       | matching pages     |       | & returns answer  |
+-------------------+       +--------------------+       +-------------------+
```

---

## 2. Technology Choice Rationale Matrix ("Why Tech X over Y?")

When building production AI systems, picking the right tools is critical. Here is the explicit breakdown of why every core technology in this platform was chosen over popular alternatives:

| Component | Selected Technology | Alternative Considered | Why We Selected Our Choice (Detailed Rationale) |
| :--- | :--- | :--- | :--- |
| **API Framework** | **FastAPI** | Flask, Django | **FastAPI** offers native asynchronous concurrency (`async/await`), automatic OpenAPI (`/docs`) interactive documentation, and high-performance Pydantic request validation. It is up to 5x faster than Flask under heavy stream workloads. |
| **Vector DB** | **Qdrant** | Pinecone, ChromaDB, PGVector | **Qdrant** provides local file-system persistence (no cloud lock-in), built-in **INT8 Scalar Quantization** (reducing RAM by 75%), fast HNSW indexing, and payload filtering for multi-tenancy without needing a heavy database cluster. |
| **Embeddings** | **FastEmbed (`bge-small-en-v1.5`)** | OpenAI (`text-embedding-3`), HuggingFace Transformers | **FastEmbed** uses ONNX Runtime under the hood, running embeddings in CPU memory without PyTorch bloat. It computes 384-dimensional vectors in sub-10ms with **zero API costs** and zero network latency. |
| **LLM Inference** | **Groq (Llama-3.3-70B)** | OpenAI GPT-4o, Anthropic Claude | **Groq LPU (Language Processing Unit)** hardware yields stream speeds of 250–300 tokens/second for Llama-3.3-70B. It is ~$0.59 / $0.79 per million tokens (95% cheaper than GPT-4) while matching GPT-4 reasoning capability. |
| **Metadata DB** | **SQLite (WAL Mode)** | PostgreSQL, MongoDB | **SQLite** requires zero server configuration and runs in-process. In Write-Ahead Logging (WAL) mode, it handles thousands of read queries per second with sub-millisecond latency for session history, feedback, and semantic caching. |
| **Frontend UI** | **Streamlit** | React / Next.js | **Streamlit** allows rapid pure-Python UI development. It natively handles Server-Sent Events (SSE) streaming, interactive chat components, session state, and metric dashboards without JavaScript build overhead. |
| **Payload Security** | **Fernet AES-256** | Plaintext, RSA | **Fernet (AES-256-CBC + HMAC-SHA256)** provides symmetric authenticated encryption. It secures document text stored in Qdrant payloads at rest against unauthorized disk inspection, decrypting only in-memory during search. |
| **Reranker** | **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** | Bi-Encoder Cosine only | Bi-encoders embed queries and documents separately, missing word-pair interactions. Cross-Encoders pass query and chunk together through transformer attention layers, boosting precision by up to 30%. |
| **Web Fallback** | **DuckDuckGo (`ddgs`)** | SerpAPI, Google Search API | **`ddgs`** is a free, open-source python package requiring zero API keys, zero subscription fees, and no rate-limit hurdles for retrieving real-time web search fallback context. |
| **Authentication** | **PyJWT + Bcrypt** | Session Cookies, Auth0 | **PyJWT** provides stateless token authentication ideal for API microservices, paired with **Bcrypt** for secure password hashing and complexity validation. |

---

## 3. High-Level System Architecture

The platform follows a clean 4-tier modular layer:

```
+-----------------------------------------------------------------------+
|                         Streamlit Frontend UI                         |
|   (Chat Interface, File Manager, Token Cost Auditor, Eval Dashboard)  |
+-----------------------------------------------------------------------+
                                   | HTTP REST / SSE Stream
                                   v
+-----------------------------------------------------------------------+
|                           FastAPI API Layer                           |
|  (Auth JWT, Rate Limiting, Correlation ID, Prom Metrics, Routing)     |
+-----------------------------------------------------------------------+
            |                              |                   |
            v                              v                   v
+-----------------------+     +-----------------------+     +-----------+
|    Security Layer     |     |   Core RAG Pipeline   |     | Database  |
| - AES-256 Encryption  |     | - Chunking & Embed    |     | - Qdrant  |
| - PII Redactor Shield |     | - Hybrid RRF & BM25   |     |   Vector  |
| - Input/Output Safety |     | - Cross-Encoder       |     | - SQLite  |
+-----------------------+     | - CRAG Web Fallback   |     |   Relational
                              | - LLM-as-a-Judge Eval |     +-----------+
                              +-----------------------+
```

---

## 4. End-to-End Execution Flows & Diagrams

### Document Ingestion & Indexing Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit App
    participant API as FastAPI Router
    participant BG as BackgroundTasks / Celery
    participant Chunker as Paragraph Chunkers
    participant Embedder as FastEmbed Model
    participant Enc as Fernet AES-256
    participant Qdrant as Qdrant Vector DB

    User->>UI: Upload Document (.pdf, .docx, .txt, .csv, .json)
    UI->>API: POST /ingest (File + Auth Token)
    API->>API: Verify JWT & Validate File Extension
    alt Celery / Redis Available
        API->>BG: Dispatch Celery Task
    else Celery Offline (Fallback)
        API->>BG: Run via FastAPI BackgroundTasks
    end
    API-->>UI: Return Task ID & Status "Processing"
    
    BG->>Chunker: Extract Raw Text & Run Parent-Child Splitter
    Chunker-->>BG: Small Child Chunks (300 chars) + Large Parent Contexts (1200 chars)
    BG->>Embedder: Generate 384-d Dense Embeddings (bge-small-en-v1.5)
    Embedder-->>BG: Dense Vectors
    BG->>Enc: Encrypt Payload Content with Fernet AES-256
    Enc-->>BG: Encrypted Ciphertext
    BG->>Qdrant: Upsert Points into 'research_papers' Collection (with INT8 Quantization)
    Qdrant-->>BG: Points Saved
    BG-->>UI: Update Status to "Completed"
```

---

### Query Processing & Multi-Stage Retrieval Pipeline

```mermaid
flowchart TD
    A[👤 User Query] --> B[🔐 JWT Authentication & Rate Limit Check]
    B --> C[🛡️ PII Redactor: Extract PII & Build Mapping]
    C --> D[🚧 Safety Guardrails Classifier]
    
    D -- Unsafe --> E[⛔ Block Response with Safety Error]
    D -- Safe --> F{⚡ Semantic Cache Lookup}
    
    F -- Cache Hit >= 0.90 --> G[📤 Restore PII & Stream Cached Response]
    F -- Cache Miss --> H[🧭 Intent Classifier: conversational / general / rag]
    
    H -- 'general' --> I[🌐 CRAG Web Search Fallback via DuckDuckGo]
    H -- 'rag' --> J[📝 Query Rewriter & Memory Context Reformulator]
    
    J --> K[🔍 Dense Search: FastEmbed BGE]
    J --> L[🔤 Sparse Search: BM25 Lexical]
    
    K --> M[🔀 Reciprocal Rank Fusion RRF]
    L --> M
    
    M --> N[🔙 Step-Back Query & Multi-Hop Retrieval]
    N --> O{Max Similarity Score >= 0.30?}
    
    O -- No (Low Similarity) --> I
    O -- Yes --> P[🎯 Cross-Encoder Reranking: ms-marco-MiniLM-L-6-v2]
    
    P --> Q[📌 MMR Diversification & Lost-in-the-Middle Reordering]
    I --> R[📑 Prompt Construction]
    Q --> R
    
    R --> S[🤖 Groq LLM Inference: Llama-3.3 70B Stream]
    S --> T[🔓 PII Stream Restoration Buffer]
    T --> U[💬 Render Tokens in Streamlit UI]
    
    U --> V[📊 Async LLM-as-a-Judge Evaluation]
    V --> W[💾 Save History, Metrics & Feedback in SQLite]
```

---

### Corrective RAG (CRAG) Web Search Fallback

```mermaid
stateDiagram-v2
    [*] --> QueryReceived
    QueryReceived --> IntentClassification
    
    IntentClassification --> DirectGeneral: Intent = 'general'
    IntentClassification --> LocalRetrieval: Intent = 'rag'
    
    DirectGeneral --> TriggerCRAG: General Knowledge Question
    
    LocalRetrieval --> CheckSimilarity: RRF + BM25 Candidate Retrieval
    CheckSimilarity --> LocalRerank: Max Similarity Score >= 0.30
    CheckSimilarity --> TriggerCRAG: Max Similarity Score < 0.30 (Out-of-Domain)
    
    TriggerCRAG --> DuckDuckGoSearch: Execute ddgs text search (max_results=3)
    DuckDuckGoSearch --> FormatWebSources: Wrap web snippets into grounding context
    FormatWebSources --> LLMGeneration: Route Tag = '⚡ Route: Corrective Web Search'
    
    LocalRerank --> LLMGeneration: Route Tag = '⚡ Route: RAG Retrieval'
    
    LLMGeneration --> [*]
```

---

### Bi-Directional PII Scrubbing & Stream Restoration

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Redactor as PII Redactor
    participant Core as RAG Routing Core
    participant LLM as Groq Llama 3.3 70B
    participant Buffer as Sliding Stream Buffer
    participant Output as UI Stream

    User->>Redactor: "My email is test@domain.com. List strategies."
    Redactor->>Redactor: Scan Regex (Email, Phone, IP)
    Redactor-->>Core: Redacted Query: "My email is redacted_email_0..." <br/> Mapping: {"redacted_email_0": "test@domain.com"}
    
    Note over Core: Logs & Tracing record 'redacted_email_0' (Zero PII Leakage)
    
    Core->>Core: Re-inject mapping into LLM prompt text
    Core->>LLM: Send Prompt: "User email is test@domain.com. List strategies."
    
    LLM-->>Buffer: Stream Chunk 1: "Your email "
    Buffer->>Buffer: Check tail against max placeholder length
    Buffer-->>Output: Yield "Your email "
    
    LLM-->>Buffer: Stream Chunk 2: "test@domain.com is noted. Strategy 1: Few-Shot..."
    Buffer->>Buffer: Replace any residual placeholders
    Buffer-->>Output: Yield "test@domain.com is noted. Strategy 1: Few-Shot..."
```

---

## 5. Module-by-Module Code Architecture

### `api/` — API Gateway & Routing
- **`api/main.py`**: Entry point for FastAPI. Configures CORS middleware, mounts sub-routers, initializes Prometheus metrics, and manages lifespan startup events.
- **`api/auth.py`**: Handles user registration, password strength validation (using regex for uppercase, numbers, special characters), password hashing via `bcrypt`, and JWT token issuance/verification (`PyJWT`).
- **`api/middleware.py`**: Implements Slowapi rate limiting (`10 requests/minute` per user), correlation-ID injection for every HTTP request, and latency metrics tracking.
- **`api/routing.py`**: The core orchestration file. Contains `/chat`, `/ingest`, `/db/stats`, `/db/token_usage`, `/db/clear`, and `/chat/feedback` endpoints. Integrates multi-tenant checks, CRAG fallback logic, and SSE streaming.

### `database/` — Vector & Relational Storage
- **`database/qdrant.py`**: Establishes the Qdrant client connection (`./qdrant_db`). Defines collection parameters for `research_papers`:
  - **Vector Dimension**: `384` (matching `BAAI/bge-small-en-v1.5`)
  - **Distance Metric**: `Cosine`
  - **Scalar Quantization**: `INT8` (4x memory reduction)
  - **HNSW Index Config**: `m=16`, `ef_construct=100`
- **`database/sqlite.py`**: Manages the SQLite database (`users.db`). Contains tables:
  - `users`: Credentials, password hash, role (`admin`, `readonly`).
  - `chat_history`: Persistent session histories.
  - `feedback`: User ratings (+1 / -1) and review text.
  - `token_usage`: Prompt tokens, completion tokens, total tokens, timestamp.
  - `embedding_cache`: Hashes and cached vector blobs for FastEmbed.

### `rag/` — Retrieval, Reranking & Evaluation
- **`rag/chunking.py`**: `paragraph_first_chunking()` and `parent_child_chunking()`.
- **`rag/embedding.py`**: Lazily loads `FastEmbed` (`BAAI/bge-small-en-v1.5`) with SQLite vector caching.
- **`rag/search.py`**: Hybrid retrieval (Vector + BM25 + RRF + MMR) and semantic caching.
- **`rag/prompts.py`**: System prompt templates, intent routing, step-back expansion, HyDE, and query rewriting.
- **`rag/reranking.py`**: Local Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) with LLM fallback.
- **`rag/evaluation.py`**: LLM-as-a-Judge metric evaluators (Faithfulness, Relevance, Precision).

### `security/` — Encryption, Guardrails & PII
- **`security/encryption.py`**: Fernet AES-256 payload encryption at rest.
- **`security/guardrails.py`**: Safety classifier checking input/output alignment.
- **`security/pii_redactor.py`**: Regex PII scrubber supporting bi-directional mapping tuples.

### `app.py` — Streamlit Frontend UI
- Multi-page tab interface (`💬 Chat` and `📊 Evaluation Dashboard`).
- Live grounding source cards, feedback buttons, token cost auditor, and automated hallucination warning banners.

---

## 6. Deep Dive: Algorithms Made Simple (With Analogies & Math)

### Paragraph-First & Parent-Child Chunking
- **Analogy**: Imagine cutting a recipe book into individual lines vs keeping full recipes together. If you cut across a line, you get *"Add 2 tbsp of..."* without knowing what to add!
- **Paragraph-First**: Splits text at natural double-newlines `\n\n` so full thoughts stay together.
- **Parent-Child**: 
  - **Child Chunks** ($300$ chars): Small index cards used for high-precision search.
  - **Parent Chunks** ($1200$ chars): The full page passed to the LLM when a child card matches.

---

### Reciprocal Rank Fusion (RRF)
- **Analogy**: Imagine two movie critics (Vector Search & Keyword Search) ranking movies. Critic A ranks *Inception* #1. Critic B ranks *Inception* #2. RRF merges their ranking lists into a single master score without caring about their different rating scales!

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $k = 60$, and $r_m(d)$ is the 1-based rank of document $d$ in search system $m$.

---

### BM25 Lexical Scoring
Measures exact word match frequency adjusted for document length:

$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

---

### Scalar Quantization (INT8) & HNSW Indexing
- **Analogy**: Converting a massive 4K raw image into a compressed WebP image. You save 75% file size while the human eye can't spot the difference!
- **Quantization**: Converts 32-bit float vectors (`FP32`) into 8-bit integers (`INT8`).
- **HNSW**: Builds a multi-layer graph skip-list for $O(\log N)$ nearest-neighbor search speed.

---

### Maximal Marginal Relevance (MMR)
- **Analogy**: If you ask for 3 news articles about space, MMR ensures you don't get 3 articles covering the exact same launch details.

$$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, Q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$

Setting $\lambda = 0.7$ balances query relevance with context diversity.

---

### Cross-Encoder Reranking vs Bi-Encoders
- **Bi-Encoder**: Compares query vector and chunk vector independently via dot product ($<5\text{ms}$).
- **Cross-Encoder**: Feeds query AND chunk together into a joint attention transformer ($~50\text{ms}$). High accuracy.

Our pipeline uses Bi-Encoder + BM25 for top-20 candidate retrieval, and Cross-Encoder to rerank down to top-3.

---

### Lost-in-the-Middle Chunk Re-ordering
- **Analogy**: When reading a long grocery list, you remember the first item ("Milk") and the last item ("Eggs"), but forget the item in the middle ("Butter").

Given top reranked chunks $[C_1, C_2, C_3, C_4, C_5]$:
- Reordered context sent to LLM: $[C_1, C_3, C_5, C_4, C_2]$.

---

### LLM-as-a-Judge Evaluation Metrics

1. **Faithfulness**:
   $$\text{Faithfulness} = \frac{\text{Supported Claims in Answer}}{\text{Total Claims in Answer}}$$
   *(If score $< 0.70$, Streamlit displays an automated Hallucination Warning Banner).*

2. **Answer Relevance**: Cosine similarity between original query vector and a query regenerated from the LLM response.

3. **Context Precision**: Evaluates whether relevant chunks were ranked higher than irrelevant chunks in the context list.

---

## 7. Security, Multi-Tenancy & Data Privacy

- **Data Isolation**: Multi-tenancy payload filter `user_id == username` in Qdrant ensures zero cross-tenant data visibility.
- **Fernet AES-256 Encryption at Rest**: Encrypts document chunk payloads at rest in Qdrant; decrypts in-memory only during search.
- **Bi-Directional PII Scrubbing**: Redacts sensitive emails, phones, and IPs before logging/tracing, while dynamically restoring values in LLM prompt generation and output token streaming.

---

## 8. Observability, Telemetry & Cost Auditor

- **Prometheus `/metrics`**: Counters and latency histograms for API health monitoring.
- **Arize Phoenix Tracing (`http://localhost:6006`)**: OpenTelemetry visualization for retrieval, reranking, and generation spans.
- **Token Cost Auditor**: Tracks prompt/completion tokens and calculates real-time USD costs based on Llama-3.3-70B pricing ($0.59 / $0.79 per million tokens).

---

## 9. ML/DL Models Used in the Platform

This platform uses **4 distinct ML/DL models**, each selected for a specific role in the pipeline. Here is a complete breakdown:

---

### Model 1: FastEmbed — `BAAI/bge-small-en-v1.5` (Dense Embedding Model)

| Property | Details |
| :--- | :--- |
| **Model Type** | Bi-Encoder Sentence Transformer (BERT architecture) |
| **Dimensions** | 384-dimensional dense vectors |
| **Runtime** | ONNX Runtime (CPU, no GPU required) |
| **Size** | ~130 MB |
| **Used For** | Converting document chunks and user queries into vector embeddings for similarity search |

**Why this model over OpenAI `text-embedding-3-small`?**
- Runs **100% locally** — zero API cost, zero network latency.
- ONNX Runtime is 3–5x faster than PyTorch for CPU inference.
- Achieves top rankings on MTEB benchmarks despite being 130MB (vs 1500MB+ for larger models).
- Produces **compact 384-d vectors** (vs OpenAI's 1536-d), requiring 75% less Qdrant storage.

**Pipeline role**: Documents are embedded at ingestion time. User queries are embedded at runtime. Cosine similarity between these vectors is the core of retrieval.

---

### Model 2: Sentence Transformers — `ms-marco-MiniLM-L-6-v2` (Cross-Encoder Reranker)

| Property | Details |
| :--- | :--- |
| **Model Type** | Cross-Encoder Transformer (joint query+doc attention) |
| **Architecture** | MiniLM-L-6 (6 attention layers, distilled from BERT-Large) |
| **Size** | ~90 MB |
| **Latency** | ~50ms for 20 candidate chunks |
| **Used For** | Re-scoring and reranking top-20 candidate chunks retrieved by FastEmbed + BM25 |

**Why Cross-Encoder instead of just Bi-Encoder similarity?**
- A Bi-Encoder computes `cosine_sim(embed(query), embed(doc))` independently. It **cannot model word-level interactions** between query and document.
- A Cross-Encoder feeds `[CLS] + query + [SEP] + document` through **joint self-attention layers**. Every word in the query attends to every word in the document.
- This produces up to **30% higher precision** at the cost of slower inference — perfectly acceptable after Bi-Encoder already narrows candidates to 20.

**Pipeline role**: After RRF fusion produces top-20 candidates, this model reranks them to top-3 for the LLM prompt.

---

### Model 3: Groq Llama-3.3-70B (`llama-3.3-70b-versatile`)

| Property | Details |
| :--- | :--- |
| **Model Type** | Decoder-only Large Language Model (LLM) |
| **Parameter Count** | 70 Billion |
| **Context Window** | 128K tokens |
| **Inference Speed** | 250–300 tokens/second on Groq LPU |
| **Used For** | RAG answer generation, intent classification, query rewriting, safety guardrails, step-back generation, LLM-as-a-Judge evaluation |

**Why Groq + Llama-3.3-70B over OpenAI GPT-4?**
- Groq's **LPU (Language Processing Unit)** hardware is purpose-built for transformer inference, delivering 250–300 tokens/second — 10x faster than standard GPU clusters.
- Cost: **$0.59 per million prompt tokens** and **$0.79 per million completion tokens** vs GPT-4o at $2.50 / $10.00 per million — saving 80–95% on API spend.
- Llama-3.3-70B is an **open-weights model** — if needed, it can be self-hosted on private GPUs (vLLM) for complete air-gapped operation.
- Instruction-following quality matches GPT-4 on RAG, summarization, and structured JSON output tasks.

**Pipeline role**: Used for 6 distinct tasks: answer generation, intent routing, conversational query rewriting, step-back query expansion, safety guardrail classification, and LLM-as-a-Judge evaluation.

---

### Model 4: BM25 (Statistical Sparse Retrieval — No Neural Weights)

| Property | Details |
| :--- | :--- |
| **Model Type** | Statistical keyword scoring algorithm (not a neural network) |
| **Library** | `rank_bm25` Python package |
| **Size** | No model weights — computed at runtime over candidate pool |
| **Latency** | $<1\text{ms}$ for 20 candidates |
| **Used For** | Lexical keyword matching on the candidate chunk pool |

**Why BM25 alongside neural embeddings?**
- Neural embeddings capture **semantic meaning** but miss **exact keyword matches** (product codes, acronyms, names, version numbers).
- BM25 catches **exact keyword** occurrences that embeddings overlook.
- Combining both via RRF fusion achieves the best of both worlds: semantic + lexical coverage.

---

## 10. Enterprise Cloud Production Deployment — AWS

> **Why AWS?** AWS has the largest managed ML/AI service ecosystem, most mature Kubernetes (EKS) tooling, broadest global region coverage (33 regions), and native integrations between services like S3, ElastiCache, and RDS. It is the most widely adopted enterprise cloud platform.

### AWS Architecture Diagram

```mermaid
flowchart TD
    Client([👤 Internet Users]) --> CF[🌐 AWS CloudFront CDN + Route 53 DNS]
    CF --> WAF[🛡️ AWS WAF + AWS Shield DDoS Protection]
    WAF --> ALB[⚖️ AWS Application Load Balancer - HTTPS + ACM SSL Certificate]

    subgraph EKS ["☸️ AWS EKS Kubernetes Cluster (Private VPC Subnets)"]
        ALB -->|app.domain.com| StreamlitPods[🖥️ Streamlit UI Pods x3]
        ALB -->|api.domain.com| FastAPIPods[⚡ FastAPI Gateway Pods x5]
        FastAPIPods --> CeleryPods[⚙️ Celery Worker Pods x3]
        StreamlitPods -.->|Internal API| FastAPIPods
    end

    subgraph PrivateData ["🔒 Private Subnet - Managed Data Tier"]
        FastAPIPods --> Qdrant[(🧠 Qdrant Cloud / EC2 i3en NVMe)]
        FastAPIPods --> RDS[(💾 AWS RDS PostgreSQL Multi-AZ)]
        FastAPIPods --> Redis[(⚡ AWS ElastiCache Redis Cluster)]
        CeleryPods --> S3[(📁 AWS S3 - SSE-KMS Encrypted)]
    end

    FastAPIPods --> Groq[🤖 Groq LPU API / Private vLLM on g5 EC2]
    FastAPIPods --> Secrets[🔑 AWS Secrets Manager]
    FastAPIPods --> CloudWatch[📈 AWS CloudWatch + Datadog APM]
```

### AWS Service Breakdown — What & Why

| AWS Service | Replaces (Local) | What It Does | Why AWS Chose This |
| :--- | :--- | :--- | :--- |
| **AWS EKS** | Local Terminal | Runs Docker containers of FastAPI, Streamlit, Celery as Kubernetes pods. HPA auto-scales from 3 to 50 replicas. | Kubernetes is the industry standard for container orchestration. EKS gives managed control plane without self-managing Kubernetes master nodes. |
| **AWS ECR** | Local Docker images | Private Docker image registry. Stores versioned FastAPI and Streamlit images. | Tight IAM integration with EKS — pods pull images without manual authentication. |
| **AWS ALB** | `localhost:8000` | Routes HTTPS traffic to correct pod groups. Terminates SSL/TLS. | Layer 7 routing allows path-based (`/api/*`) and host-based (`api.domain.com`) rules. Health checks automatically remove unhealthy pods from rotation. |
| **AWS WAF** | None | Blocks SQL injection, XSS, bot scrapers, and custom IP blacklists at the CDN edge. | Protects before traffic reaches any application pod, saving compute. |
| **AWS CloudFront** | None | Global CDN that caches static UI assets. Reduces Streamlit pod load. | 450+ edge PoPs globally reduces latency from 200ms to $<30\text{ms}$ for international users. |
| **AWS RDS PostgreSQL (Multi-AZ)** | SQLite `users.db` | Stores users, sessions, feedback, token logs, semantic cache. Multi-AZ ensures automatic failover in $<30\text{s}$. | SQLite cannot handle thousands of concurrent writes. RDS Multi-AZ provides zero-downtime failover across two availability zones. |
| **AWS ElastiCache Redis** | Local Redis | Celery task broker queue for async ingestion. Also stores distributed semantic cache entries. | Managed cluster auto-failover. Removes Redis as a single-point-of-failure in document ingestion. |
| **AWS S3 + KMS** | Local `./docs` folder | Stores uploaded raw PDF, DOCX, TXT files with AES-256 server-side encryption and lifecycle policies. | 11 9's data durability. Built-in versioning for document rollback. KMS manages encryption keys with automatic rotation. |
| **AWS Secrets Manager** | `.env` file | Stores `GROQ_API_KEY`, `JWT_SECRET_KEY`, `DB_PASSWORD`, `FERNET_KEY` with 90-day automatic rotation. | `.env` files on disk are a security liability. Secrets Manager uses IAM roles — pods retrieve secrets at runtime without storing them on disk. |
| **AWS CloudWatch + Datadog** | Prometheus / Arize Phoenix | Centralized logging, metric dashboards, APM traces, and PagerDuty alerting on error rate spikes. | CloudWatch integrates natively with all AWS services. Datadog provides distributed tracing across FastAPI, Qdrant, PostgreSQL, and Redis in one unified view. |
| **Qdrant Cloud (NVMe EC2)** | Local `./qdrant_db` | Multi-node vector database cluster with read replicas. | EC2 `i3en` instances provide NVMe SSDs (3.3 TB) optimized for Qdrant's write-heavy ingestion workloads. |

---

## 11. Enterprise Cloud Production Deployment — GCP

> **Why GCP?** Google Cloud Platform has a strong ML ecosystem (Vertex AI, TPU pods), superior BigQuery analytics integration, and is often preferred for AI-first organizations. GCP's **Cloud Run** offers serverless auto-scaling to zero — perfect for cost-efficient RAG workloads with variable traffic patterns.

### GCP Architecture Diagram

```mermaid
flowchart TD
    Client([👤 Internet Users]) --> GCPDNS[🌐 GCP Cloud DNS]
    GCPDNS --> CloudArmor[🛡️ GCP Cloud Armor WAF + DDoS]
    CloudArmor --> GCPLB[⚖️ GCP Global HTTP(S) Load Balancer + Managed SSL]

    subgraph GKE ["☸️ GCP GKE Autopilot Cluster (Private VPC)"]
        GCPLB -->|app.domain.com| StreamlitPods[🖥️ Streamlit UI Cloud Run / GKE Pods]
        GCPLB -->|api.domain.com| FastAPIPods[⚡ FastAPI Gateway GKE Pods]
        FastAPIPods --> CeleryPods[⚙️ Celery Worker Pods]
    end

    subgraph PrivateVPC ["🔒 Private VPC - Managed Data Services"]
        FastAPIPods --> Qdrant[(🧠 Qdrant Cloud on GCP)]
        FastAPIPods --> CloudSQL[(💾 GCP Cloud SQL PostgreSQL HA)]
        FastAPIPods --> Memorystore[(⚡ GCP Memorystore for Redis)]
        CeleryPods --> GCS[(📁 GCP Cloud Storage - CMEK)]
    end

    FastAPIPods --> Groq[🤖 Groq API / Vertex AI Model Garden]
    FastAPIPods --> SecretMgr[🔑 GCP Secret Manager]
    FastAPIPods --> CloudTrace[📈 GCP Cloud Monitoring + Cloud Trace]
```

### GCP Service Breakdown — What & Why

| GCP Service | Replaces (Local) | What It Does | Why GCP Chose This |
| :--- | :--- | :--- | :--- |
| **GCP GKE Autopilot** | Local Terminal | Fully managed Kubernetes where Google automatically provisions, scales, and patches nodes. No worker node management needed. | Autopilot eliminates the need to manually configure node pools, instance types, and node autoscaling — Google handles it automatically based on pod resource requests. |
| **GCP Cloud Run** (alternative) | Local FastAPI server | Serverless container platform that scales FastAPI from 0 to 1000 instances in seconds. Charges only per request (pay-per-use). | For RAG workloads with variable traffic, Cloud Run saves 60–80% cost over always-on EKS/GKE worker nodes. Scales to zero during off-hours. |
| **GCP Artifact Registry** | Local Docker images | Stores Docker images. Replaces Docker Hub with IAM-integrated private repositories per region. | Native integration with GKE — containers pull images via Workload Identity without credentials stored in pods. |
| **GCP Global Load Balancer + Cloud Armor** | `localhost:8000` | Anycast Layer 7 load balancing across all GCP regions. Cloud Armor implements OWASP Top 10 WAF rules. | GCP's single anycast IP routes users to the nearest region automatically, reducing latency compared to regional ALBs. |
| **GCP Cloud SQL PostgreSQL (HA)** | SQLite `users.db` | Fully managed PostgreSQL with synchronous replication to a hot standby. Automatic failover in $<60\text{s}$. | Cloud SQL HA provides a read replica in a second zone for zero-data-loss failover. Supports `pg_vector` extension if migrating embedding cache to Postgres. |
| **GCP Memorystore for Redis** | Local Redis | Fully managed Redis cluster for Celery task queue and semantic cache. Supports 99.9% SLA. | Memorystore supports Redis Cluster mode with automatic sharding, allowing semantic cache to horizontally scale across multiple nodes. |
| **GCP Cloud Storage (GCS) + CMEK** | Local `./docs` folder | Object storage for uploaded documents with Customer-Managed Encryption Keys (CMEK) via GCP KMS. | CMEK gives organizations direct cryptographic control over document encryption keys — required for HIPAA and GDPR compliance. |
| **GCP Secret Manager** | `.env` file | Stores API keys and credentials with version history and IAM-controlled access via Workload Identity Federation. | Workload Identity allows GKE pods to access secrets via identity binding without any secret credentials stored inside container images or env files. |
| **GCP Cloud Monitoring + Cloud Trace** | Prometheus / Arize Phoenix | Centralized metrics dashboards, distributed tracing, log-based alerting, and uptime checks. | Native OpenTelemetry support traces requests from GCP Load Balancer through GKE pods to Cloud SQL and Memorystore, providing end-to-end latency breakdown with zero config. |
| **Vertex AI Model Garden** (optional) | Groq API | GCP's managed ML model hosting platform. Can host Llama-3.3-70B or Gemini on dedicated TPU/GPU infrastructure. | For organizations requiring data residency compliance, Vertex AI hosts LLMs within the same GCP region as the application, ensuring query data never leaves the compliance boundary. |

---

## 12. Production Deployment Checklist

- [x] Set 64-character JWT `SECRET_KEY` in AWS Secrets Manager or GCP Secret Manager.
- [x] Persist Qdrant on NVMe SSD storage (AWS EC2 i3en or dedicated Qdrant Cloud cluster).
- [x] Enable SQLite WAL mode locally (`PRAGMA journal_mode=WAL;`) or migrate to RDS/Cloud SQL in production.
- [x] Deploy behind AWS ALB or GCP Load Balancer with TLS HTTPS.
- [x] Run Celery workers on dedicated worker pods with ElastiCache/Memorystore Redis as broker.
- [x] Enable HPA (Horizontal Pod Autoscaler) for FastAPI pods based on CPU/request queue metrics.
- [x] Configure VPC private subnets for all databases — zero public IP exposure.
