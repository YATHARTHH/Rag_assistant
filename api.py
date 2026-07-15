import os
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
import uuid

# Import backend functions from rag_demo
from rag_demo import (
    make_embedder,
    get_collection,
    get_cache_collection,
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
    load_uploaded_file,
    init_user_db,
    rewrite_query_with_history
)
from langchain_groq import ChatGroq

# Initialize FastAPI
app = FastAPI(title="RAG AI Backend API")

# Initialize models once
init_user_db()
embedder = make_embedder()
collection = get_collection(embedder)
cache_collection = get_cache_collection(embedder)
RERANKER_MODEL = None

def get_reranker_lazy():
    global RERANKER_MODEL
    if RERANKER_MODEL is None:
        RERANKER_MODEL = make_reranker()
    return RERANKER_MODEL

# Task tracker for async ingestion
INGEST_TASKS: Dict[str, str] = {}


# -------------------------
# Request Models
# -------------------------
class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict]] = None
    routing: Optional[str] = None
    filter: Optional[str] = None
    eval: Optional[Dict] = None


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


class IngestRequest(BaseModel):
    file_name: str
    file_bytes_hex: str
    username: str


# -------------------------
# Authentication Routes
# -------------------------
@app.post("/auth/signup")
def auth_signup(req: SignupRequest):
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    success = create_user(req.username, req.password)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists.")
    return {"message": "User registered successfully."}


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    valid = verify_user(req.username, req.password)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    # Return simple session token (username itself acts as session identifier locally)
    return {"token": req.username, "username": req.username}


# -------------------------
# Asynchronous Ingestion Background Task
# -------------------------
def background_ingest_task(task_id: str, file_name: str, file_bytes: bytes, username: str):
    INGEST_TASKS[task_id] = "processing"
    try:
        # Create a mock uploaded file object for compatibility
        class MockUploadedFile:
            def __init__(self, name, data):
                self.name = name
                self.data = data
            def read(self):
                return self.data
                
        mock_file = MockUploadedFile(file_name, file_bytes)
        
        # Parse document
        content, fname = load_uploaded_file(mock_file)
        
        # Semantic Chunking (using preloaded embedder)
        chunks = semantic_chunk_text(content, fname, embedder)
        
        if chunks:
            collection.add_texts(
                texts=[c["content"] for c in chunks],
                metadatas=[{
                    "title": c["title"],
                    "sent_index": c["sent_index"],
                    "user_id": username
                } for c in chunks],
                ids=[f"{sanitize_id_part(username)}_{sanitize_id_part(c['title'])}_sent_{c['sent_index']}" for c in chunks]
            )
            # Try persist
            try:
                collection.persist()
            except Exception:
                pass
                
        INGEST_TASKS[task_id] = "completed"
    except Exception as e:
        INGEST_TASKS[task_id] = f"error: {str(e)}"


@app.post("/ingest")
def start_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    try:
        file_bytes = bytes.fromhex(req.file_bytes_hex)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file bytes format.")
        
    task_id = f"{req.username}_{uuid.uuid4().hex[:8]}"
    background_tasks.add_task(
        background_ingest_task, 
        task_id, 
        req.file_name, 
        file_bytes, 
        req.username
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/ingest/status/{task_id}")
def check_ingest_status(task_id: str):
    status = INGEST_TASKS.get(task_id, "unknown")
    return {"task_id": task_id, "status": status}


@app.get("/db/stats")
def get_db_stats(username: str):
    try:
        db_data = collection.get()
        if db_data and db_data.get("documents"):
            unique_files = list(set(
                meta.get("title", "Unknown") 
                for meta in db_data.get("metadatas", []) 
                if meta and (meta.get("user_id") == username or meta.get("user_id") == "public")
            ))
            total_chunks = len([
                meta for meta in db_data.get("metadatas", [])
                if meta and (meta.get("user_id") == username or meta.get("user_id") == "public")
            ])
            return {"total_chunks": total_chunks, "unique_files": unique_files}
    except Exception:
        pass
    return {"total_chunks": 0, "unique_files": []}


@app.post("/db/clear")
def clear_user_db(username: str):
    try:
        db_data = collection.get()
        if db_data and db_data.get("ids"):
            user_ids = [
                doc_id for doc_id, meta in zip(db_data["ids"], db_data["metadatas"])
                if meta and meta.get("user_id") == username
            ]
            if user_ids:
                collection.delete(ids=user_ids)
                try:
                    collection.persist()
                except Exception:
                    pass
                return {"message": f"Successfully cleared {len(user_ids)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "No chunks found to clear."}


def rerank_documents_with_llm(query: str, sources: list, llm, top_k=3):
    """
    Reranks document chunks using Groq LLM instead of a local PyTorch model to save local memory.
    """
    if not sources:
        return []
        
    passages_str = ""
    for idx, src in enumerate(sources):
        passages_str += f"[{idx}] Source: {src['title']}\n{src['content']}\n\n"
        
    prompt = f"""
    You are an expert search ranker.
    Given the following query and a list of retrieved passages, re-rank the passages from most relevant to least relevant to answer the query.

    Query: "{query}"

    Passages:
    {passages_str}

    Respond with ONLY a comma-separated list of indices representing the re-ranked order, from most relevant to least relevant (e.g. "2,0,1").
    Do NOT include any text, explanations, or quotes.
    """
    try:
        response = llm.invoke(prompt)
        order_str = response.content.strip()
        indices = [int(x.strip()) for x in order_str.split(",") if x.strip().isdigit()]
        ranked_sources = []
        for idx in indices:
            if 0 <= idx < len(sources):
                ranked_sources.append(sources[idx])
        for idx, src in enumerate(sources):
            if src not in ranked_sources:
                ranked_sources.append(src)
        return ranked_sources[:top_k]
    except Exception as e:
        print(f"[WARNING] LLM Reranker failed: {e}. Falling back to default order.")
        return sources[:top_k]


# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # Initialize Groq client
    GROQ_KEY = os.getenv("GROQ_API_KEY", "")
    if not GROQ_KEY:
        raise HTTPException(status_code=400, detail="Groq API key not configured on backend.")
        
    try:
        llm = ChatGroq(
            temperature=req.temperature,
            groq_api_key=GROQ_KEY,
            model=req.model_name,
            streaming=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM: {str(e)}")

    # 1. Input Guardrails
    safety_in = check_safety_guardrails(req.query, llm, stage="input")
    if safety_in == "unsafe":
        def generate_block():
            yield "⚠️ **Blocked**: Prompt violated safety guardrail checks."
        return StreamingResponse(generate_block(), media_type="text/plain")

    # 2. Semantic Query Cache Check
    cached_response, similarity = check_semantic_cache(req.query, cache_collection)
    if cached_response:
        def generate_cache_hit():
            yield f"[⚡ Semantic Cache Hit (Similarity: {similarity:.2f})]\n" + cached_response
        return StreamingResponse(generate_cache_hit(), media_type="text/plain")

    # 3. Intent Routing
    intent = classify_query_intent(req.query, llm)
    
    # 4. Ingested unique files list for multi-tenant metadata filter
    try:
        db_data = collection.get()
        if db_data and db_data.get("documents"):
            unique_files = list(set(
                meta.get("title", "Unknown") 
                for meta in db_data.get("metadatas", []) 
                if meta and meta.get("user_id") == req.username
            ))
        else:
            unique_files = []
    except Exception:
        unique_files = []

    sources = []
    prompt = ""
    metadata_filter = None
    
    if intent in ["conversational", "general"]:
        # Conversational / General direct chat route
        prompt = f"""
        You are an advanced AI assistant. Answer the user prompt directly. 
        If it is conversational, respond warmly. If it is a coding or general question, answer it thoroughly.
        
        User: {req.query}
        Assistant:
        """
        route_info = f"Direct Chat ({intent})"
    else:
        # RAG retrieval route
        # Formulate query history
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in req.history]
        rewritten_query = rewrite_query_with_history(req.query, history_dicts, llm)
        
        # Self-Querying Metadata Filter
        metadata_filter = generate_metadata_filter(rewritten_query, llm, unique_files)
        
        # Fetch Context from Chroma (restricted strictly by tenant user_id)
        candidates = retrieve_context(
            rewritten_query, 
            collection, 
            top_k=req.rerank_pool, 
            vector_weight=req.vector_weight,
            window_size=req.window_size,
            metadata_filter=metadata_filter,
            user_id=req.username
        )
        
        if req.enable_reranking and candidates:
            lazy_reranker = get_reranker_lazy()
            if lazy_reranker is not None:
                sources = rerank_documents(rewritten_query, candidates, lazy_reranker, top_k=req.top_k)
            else:
                sources = rerank_documents_with_llm(rewritten_query, candidates, llm, top_k=req.top_k)
        else:
            sources = candidates[:req.top_k]
            
        context = "\n\n".join([f"Source: {src['title']}\n{src['content']}" for src in sources])
        
        prompt = f"""
        You are an advanced AI research assistant.
        Use the following retrieved context to answer the question.
        If the context does not contain the answer, say "I cannot find the answer in the provided documents." - do not make up an answer.
        Keep the answer factual, clear, and refer to sources if applicable.

        Question: {req.query}

        Context:
        {context}

        Answer:
        """
        route_info = "Document Retrieval (RAG)"

    # Stream Generation Function
    def stream_generator():
        # Let's send headers as metadata in the first yield chunk
        # We can format it as a json-like string or metadata tag so the client parses it
        meta_header = {
            "routing": route_info,
            "filter": metadata_filter["title"] if metadata_filter else "None",
            "sources": sources
        }
        
        # Output metadata tag
        yield f"__METADATA_START__{str(meta_header)}__METADATA_END__"
        
        full_response = ""
        for chunk in llm.stream(prompt):
            full_response += chunk.content
            yield chunk.content
            
        # 5. Output Guardrails check on complete answer
        safety_out = check_safety_guardrails(full_response, llm, stage="output")
        if safety_out == "unsafe":
            yield "\n\n⚠️ **Warning**: Generated response was blocked by output safety guardrails."
            return
            
        # 6. Run background evaluation metrics
        if intent == "rag" and sources:
            context_str = "\n\n".join([src["content"] for src in sources])
            faithfulness = evaluate_faithfulness(context_str, full_response, llm)
            relevance = evaluate_answer_relevance(req.query, full_response, llm)
            precision = evaluate_context_precision(req.query, context_str, llm)
            
            eval_scores = {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "precision": precision
            }
            yield f"__EVAL_START__{str(eval_scores)}__EVAL_END__"
            
            # 7. Save to semantic cache
            save_to_semantic_cache(req.query, full_response, cache_collection)

    return StreamingResponse(stream_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
