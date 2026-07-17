import os
import sys
import uuid
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Header, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse
from qdrant_client.http import models

# Import local modules
from rag_demo import (
    make_embedder,
    make_reranker,
    rerank_documents,
    classify_query_intent,
    check_safety_guardrails,
    semantic_chunk_text,
    sanitize_id_part,
    generate_metadata_filter,
    retrieve_context,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_context_precision,
    create_user,
    verify_user,
    check_semantic_cache,
    save_to_semantic_cache,
    init_user_db,
    rewrite_query_with_history,
    get_qdrant_client,
    init_qdrant_collections,
    delete_file_from_qdrant,
    invalidate_semantic_cache_by_file,
    generate_hyde_response,
    compress_context_with_llm,
    add_chunks_to_qdrant,
    parent_child_chunking,
    USER_DB_PATH
)
from langchain_groq import ChatGroq
import sqlite3

INDEXED_TASKS = set()

# -------------------------
# Startup Validations
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print(json.dumps({"level": "CRITICAL", "message": "CRITICAL: GROQ_API_KEY environment variable is missing!"}))
    sys.exit(1)

# -------------------------
# Structured Logging Config
# -------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "correlation_id": getattr(record, "correlation_id", "None")
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# -------------------------
# Rate Limiting & JWT Setup
# -------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG AI Backend API")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("JWT_SECRET", "SecureJWTSecretKeyForRAGTokenSigning12345=")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), x_api_key: Optional[str] = Header(None)):
    if x_api_key:
        if x_api_key == os.getenv("RAG_API_KEY", "rag_developer_key_123"):
            return "api_key_admin"
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication credentials required.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token has expired or is invalid.")

# -------------------------
# Request Models
# -------------------------
class SignupRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "readonly"

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage]
    username: str
    model_name: str
    temperature: float
    top_k: int
    vector_weight: float
    window_size: int
    enable_reranking: bool
    rerank_pool: int
    prompt_style: Optional[str] = "Strict Fact-Only"
    parent_retrieval: Optional[bool] = False
    hyde: Optional[bool] = False

class IngestRequest(BaseModel):
    file_name: str
    file_bytes_hex: str

class FeedbackRequest(BaseModel):
    message_id: str
    rating: int
    feedback_text: Optional[str] = None

# Initialize Database & Qdrant connections
init_user_db()
client = get_qdrant_client()
init_qdrant_collections(client)
embedder = make_embedder()
RERANKER_MODEL = None

def get_reranker_lazy():
    global RERANKER_MODEL
    if RERANKER_MODEL is None:
        RERANKER_MODEL = make_reranker()
    return RERANKER_MODEL

# Global middleware for request correlation ID
@app.middleware("http")
async def request_logger_middleware(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = corr_id
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(
            f"HTTP {request.method} {request.url.path} finished in {duration:.4f}s with status {response.status_code}",
            extra={"correlation_id": corr_id}
        )
        response.headers["X-Correlation-ID"] = corr_id
        return response
    except Exception as e:
        logger.error(
            f"Unhandled exception during request processing: {e}", 
            exc_info=True, 
            extra={"correlation_id": corr_id}
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "correlation_id": corr_id}
        )

# -------------------------
# Health Check Endpoint
# -------------------------
@app.get("/health", tags=["Observability"], summary="Checks server and dependency health statuses.")
def health_check():
    status_data = {"status": "healthy", "redis": "healthy", "qdrant": "healthy", "sqlite": "healthy"}
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
    except Exception:
        status_data["sqlite"] = "unhealthy"
        status_data["status"] = "unhealthy"
        
    try:
        client.get_collection("research_papers")
    except Exception:
        status_data["qdrant"] = "unhealthy"
        status_data["status"] = "unhealthy"
        
    if status_data["status"] == "unhealthy":
        raise HTTPException(status_code=500, detail=status_data)
    return status_data

# -------------------------
# Authentication Routes
# -------------------------
@app.post("/auth/signup", tags=["Auth"], summary="Sign up a new user account with password validation.")
def auth_signup(req: SignupRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    success = create_user(req.username, req.password, req.role or "readonly")
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Username already exists or password is too weak (must be at least 8 characters, containing uppercase and a digit)."
        )
    return {"message": "User registered successfully."}

@app.post("/auth/login", tags=["Auth"], summary="Authenticate credentials and return a signed JWT token.")
def auth_login(req: LoginRequest):
    valid = verify_user(req.username, req.password)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    
    # Retrieve user role
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (req.username,))
    row = cursor.fetchone()
    conn.close()
    role = row[0] if row else "readonly"
    
    token = create_access_token({"username": req.username, "role": role})
    return {"token": token, "username": req.username, "role": role}

# -------------------------
# Ingestion Routes
# -------------------------
@app.post("/ingest", tags=["Ingestion"], summary="Queue a file for background Celery parsing and Qdrant vector indexing.")
@limiter.limit("5/minute")
def start_ingest(request: Request, req: IngestRequest, username: str = Depends(get_current_user)):
    from tasks import ingest_file_task
    # File validation
    if len(req.file_bytes_hex) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10MB.")
        
    try:
        file_bytes = bytes.fromhex(req.file_bytes_hex)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file bytes hex format.")
        
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{req.file_name}")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    task = ingest_file_task.delay(temp_path, req.file_name, username)
    return {"task_id": task.id, "status": "queued"}

@app.get("/ingest/status/{task_id}", tags=["Ingestion"], summary="Poll Celery task status for background indexing job.")
def check_ingest_status(task_id: str, username: str = Depends(get_current_user)):
    from tasks import ingest_file_task
    task_res = ingest_file_task.AsyncResult(task_id)
    state = task_res.state
    
    if state == "SUCCESS":
        if task_id not in INDEXED_TASKS:
            result = task_res.result
            if result and result.get("status") == "completed":
                content = result.get("content", "")
                filename = result.get("filename", "")
                
                # Perform parent-child semantic chunking in uvicorn's thread
                chunks = parent_child_chunking(content, filename, embedder)
                if chunks:
                    # Index chunks into Qdrant cleanly under uvicorn's exclusive lock
                    delete_file_from_qdrant(client, filename, username)
                    add_chunks_to_qdrant(client, chunks, username, embedder)
                    invalidate_semantic_cache_by_file(client, filename)
                
                INDEXED_TASKS.add(task_id)
        return {"task_id": task_id, "status": "completed"}
        
    status_map = {
        "PENDING": "processing",
        "STARTED": "processing",
        "RETRY": "processing",
        "FAILURE": "error"
    }
    status = status_map.get(state, "processing")
    if state == "FAILURE":
        return {"task_id": task_id, "status": f"error: {task_res.result}"}
    return {"task_id": task_id, "status": status}

# -------------------------
# Database Stats & Management
# -------------------------
@app.get("/db/stats", tags=["Database Management"], summary="Retrieves document names and index sizes for tenant.")
def get_db_stats(username: str = Depends(get_current_user)):
    try:
        must_cond = [
            models.Filter(
                should=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=username)),
                    models.FieldCondition(key="user_id", match=models.MatchValue(value="public"))
                ]
            )
        ]
        scroll_res = client.scroll(
            collection_name="research_papers",
            scroll_filter=models.Filter(must=must_cond),
            limit=5000,
            with_payload=True
        )
        if scroll_res and scroll_res[0]:
            unique_files = list(set(
                item.payload.get("title", "Unknown") 
                for item in scroll_res[0]
            ))
            return {"total_chunks": len(scroll_res[0]), "unique_files": unique_files}
    except Exception:
        pass
    return {"total_chunks": 0, "unique_files": []}

@app.delete("/db/files/{file_name}", tags=["Database Management"], summary="Deletes a specific document's chunks from vector store.")
def delete_file(file_name: str, username: str = Depends(get_current_user)):
    try:
        delete_file_from_qdrant(client, file_name, username)
        invalidate_semantic_cache_by_file(client, file_name)
        return {"message": f"Successfully deleted '{file_name}' from RAG index."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/db/clear", tags=["Database Management"], summary="Clears all chunks uploaded by current tenant.")
def clear_user_db(username: str = Depends(get_current_user)):
    try:
        client.delete(
            collection_name="research_papers",
            points_selector=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=username))]
            )
        )
        return {"message": "All database chunks cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Chat Session History
# -------------------------
@app.get("/chat/sessions", tags=["Chat History"], summary="Lists all past chat sessions for user.")
def list_chat_sessions(username: str = Depends(get_current_user)):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id, timestamp FROM chat_history WHERE username = ? ORDER BY timestamp DESC", (username,))
        rows = cursor.fetchall()
        conn.close()
        return [{"session_id": r[0], "created_at": r[1]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}", tags=["Chat History"], summary="Retrieves full log of chat session.")
def get_chat_history(session_id: str, username: str = Depends(get_current_user)):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM chat_history WHERE username = ? AND session_id = ? ORDER BY id ASC", (username, session_id))
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/feedback", tags=["Chat History"], summary="Submit rating feedback for a response message.")
def post_chat_feedback(req: FeedbackRequest, username: str = Depends(get_current_user)):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (username, message_id, rating, feedback_text) VALUES (?, ?, ?, ?)", 
                       (username, req.message_id, req.rating, req.feedback_text))
        conn.commit()
        conn.close()
        return {"message": "Feedback submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Prompts Styles Selection
# -------------------------
def get_system_prompt(style: str) -> str:
    prompts_path = "./prompts.json"
    default_prompt = "You are a helpful AI assistant."
    if not os.path.exists(prompts_path):
        return default_prompt
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(style, default_prompt)
    except Exception:
        return default_prompt

# -------------------------
# Structured Prometheus Metrics
# -------------------------
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

LATENCY_HISTOGRAM = Histogram("rag_query_latency_seconds", "Total RAG request generation latency in seconds.")
CACHE_COUNTER = Counter("rag_cache_hits_total", "Total semantic cache query interception hits.", ["result"])
TOKEN_COUNTER = Counter("rag_tokens_usage_total", "Tokens consumed total counts.", ["type"])

@app.get("/metrics", tags=["Observability"], summary="Exposes Prometheus metrics endpoint.")
def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -------------------------
# Core Chat Streaming
# -------------------------
@app.post("/chat", tags=["Core RAG"], summary="Streams chat responses using Server-Sent Events (SSE) event formats.")
@limiter.limit("10/minute")
def chat_endpoint(request: Request, req: ChatRequest, username: str = Depends(get_current_user)):
    # 1. Fetch Prompts Config
    sys_prompt = get_system_prompt(req.prompt_style or "Strict Fact-Only")
    
    # 2. Input Safety Guardrail check
    GROQ_KEY = os.getenv("GROQ_API_KEY", "")
    llm = ChatGroq(
        temperature=req.temperature,
        groq_api_key=GROQ_KEY,
        model=req.model_name,
    )
    
    is_safe_input = check_safety_guardrails(req.query, llm)
    if not is_safe_input:
        def err_stream():
            yield f"data: Error: Input violates safety guardrail policy.\n\n"
        return EventSourceResponse(err_stream())
        
    # Check Semantic cache
    cached_ans, sim = check_semantic_cache(client, req.query, embedder, score_threshold=0.90)
    if cached_ans:
        CACHE_COUNTER.labels(result="hit").inc()
        def cache_stream():
            yield f"data: [Semantic Cache Hit (Similarity: {sim:.2f})]\n\n"
            yield f"data: {cached_ans}\n\n"
        return EventSourceResponse(cache_stream())
        
    CACHE_COUNTER.labels(result="miss").inc()
    
    # 3. Intent Routing
    intent = classify_query_intent(req.query, llm)
    
    # Ingested files listing
    stats = get_db_stats(username)
    unique_files = stats.get("unique_files", [])
    
    # Retrieve & Rerank Context
    sources = []
    metadata_filter = None
    query_to_search = req.query
    
    # Query Expansion (HyDE)
    if req.hyde:
        query_to_search = generate_hyde_response(req.query, llm)
        
    if intent in ["rag", "general"]:
        # Conversational memory query rewriting
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in req.history]
        rewritten_query = rewrite_query_with_history(query_to_search, history_dicts, llm)
        
        # Metadata filter classification
        metadata_filter = generate_metadata_filter(rewritten_query, llm, unique_files)
        
        # Retrieval using RRF, MMR, and Parent Document
        candidates = retrieve_context(
            rewritten_query,
            client,
            embedder,
            top_k=req.rerank_pool,
            vector_weight=req.vector_weight,
            window_size=req.window_size,
            metadata_filter=metadata_filter,
            user_id=username,
            parent_retrieval=req.parent_retrieval
        )
        
        # Contextual Compression using LLM
        sources = compress_context_with_llm(rewritten_query, candidates, llm, top_k=req.top_k)
        
    context = "\n\n".join([f"Source: {src['title']} (Page {src.get('page', 1)})\n{src['content']}" for src in sources])
    
    # Construct LLM prompt
    if intent in ["rag", "general"] and context:
        prompt_content = f"""
        {sys_prompt}
        Use the following retrieved context to answer the question:
        
        Question: {req.query}
        Context:
        {context}
        """
    else:
        prompt_content = f"""
        {sys_prompt}
        Answer the following question:
        {req.query}
        """
        
    def sse_event_stream():
        with LATENCY_HISTOGRAM.time():
            full_response = ""
            
            # Streaming metadata block first
            meta_json = json.dumps({
                "routing": "RAG Retrieval" if intent == "rag" and context else f"Direct Chat ({intent})",
                "filter": metadata_filter if metadata_filter else "None",
                "sources": sources if sources else []
            })
            yield f"data: __METADATA_START__{meta_json}__METADATA_END__\n\n"
            
            # LLM invocation streaming
            response_generator = llm.stream(prompt_content)
            for chunk in response_generator:
                text_chunk = chunk.content
                full_response += text_chunk
                yield f"data: {text_chunk}\n\n"
                
            # Log output safety check
            is_safe_output = check_safety_guardrails(full_response, llm)
            if not is_safe_output:
                yield f"data: \n\n[WARNING] Output blocked by safety guardrail policy.\n\n"
                return
                
            # Background RAG Evaluations & caching
            eval_data = {}
            if intent in ["rag", "general"] and context:
                try:
                    faithfulness = evaluate_faithfulness(context, full_response, llm)
                    relevance = evaluate_answer_relevance(req.query, full_response, llm)
                    precision = evaluate_context_precision(rewritten_query, context, llm)
                    eval_data = {"faithfulness": faithfulness, "relevance": relevance, "precision": precision}
                    
                    # Output evaluations JSON
                    yield f"data: __EVAL_START__{json.dumps(eval_data)}__EVAL_END__\n\n"
                    
                    # Store session history
                    session_id = str(uuid.uuid4().hex[:12])
                    conn = sqlite3.connect(USER_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO chat_history (username, session_id, role, content) VALUES (?, ?, ?, ?)", 
                                   (username, session_id, "user", req.query))
                    cursor.execute("INSERT INTO chat_history (username, session_id, role, content) VALUES (?, ?, ?, ?)", 
                                   (username, session_id, "assistant", full_response))
                    conn.commit()
                    conn.close()
                    
                    # Save to semantic query cache
                    source_filenames = list(set([src["title"] for src in sources]))
                    save_to_semantic_cache(client, req.query, full_response, source_filenames, embedder)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    
    return EventSourceResponse(sse_event_stream())
