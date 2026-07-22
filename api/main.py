import os
import sys
import json
import logging
import atexit
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Import local packages
from api.middleware import limiter, request_logger_middleware, logger
from api.routing import router

# -------------------------
# Startup Validations
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print(json.dumps({"level": "CRITICAL", "message": "CRITICAL: GROQ_API_KEY environment variable is missing!"}))
    sys.exit(1)

# Initialize FastAPI App
app = FastAPI(
    title="RAG AI Backend API",
    description="Enterprise-grade modular RAG API with hybrid retrieval, step-back prompting, evaluation, and security guardrails.",
    version="1.0.0"
)

# Wire Limiter & Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Correlation ID / Latency Request Middleware
app.middleware("http")(request_logger_middleware)

# Include API Routes Router
app.include_router(router)

# -------------------------
# Arize Phoenix Tracing (In-Process Startup)
# -------------------------
try:
    import phoenix as px
    from openinference.instrumentation.langchain import LangChainInstrumentor
    _phoenix_session = px.launch_app()
    LangChainInstrumentor().instrument()
    logger.info("Arize Phoenix tracing active at http://localhost:6006")
except ImportError:
    logger.warning("Arize Phoenix not installed — skipping tracing. pip install arize-phoenix openinference-instrumentation-langchain")
except Exception as _phoenix_err:
    logger.warning(f"Arize Phoenix failed to launch: {_phoenix_err}")

# -------------------------
# Graceful Shutdown Handler
# -------------------------
def _shutdown_handler():
    """Flushes pending Celery tasks and logs orderly shutdown."""
    logger.info("[SHUTDOWN] Graceful shutdown initiated. Flushing pending Celery tasks...")
    try:
        from tasks import celery_app as _celery_app
        _celery_app.control.broadcast("shutdown", reply=False)
    except Exception:
        pass
    logger.info("[SHUTDOWN] Shutdown complete.")

atexit.register(_shutdown_handler)
