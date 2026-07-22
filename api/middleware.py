import sys
import time
import uuid
import json
import logging
from typing import Optional
from fastapi import Request
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram
from slowapi import Limiter
from slowapi.util import get_remote_address

# Prometheus metrics definition
LATENCY_HISTOGRAM = Histogram("rag_query_latency_seconds", "Total RAG request generation latency in seconds.")
CACHE_COUNTER = Counter("rag_cache_hits_total", "Total semantic cache query interception hits.", ["result"])
TOKEN_COUNTER = Counter("rag_tokens_usage_total", "Tokens consumed total counts.", ["type"])

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Logger Formatter
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
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

async def request_logger_middleware(request: Request, call_next):
    """
    Middleware to inject X-Correlation-ID headers, track latency, and log all requests.
    """
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
