import re
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag_api")

def make_reranker() -> CrossEncoder:
    """
    Creates a local Cross-Encoder model to score and re-rank document relevance.
    """
    try:
        return CrossEncoder("BAAI/bge-reranker-base")
    except Exception as e:
        logger.warning(f"[RERANKER] Failed to load CrossEncoder due to memory/setup: {e}")
        return None

def llm_rerank_fallback(query: str, sources: list, llm, top_k=3) -> list:
    """
    Reranks documents using LLM prompt-based scoring (fallback if Cross-Encoder is unavailable).
    """
    if not sources or not llm:
        return sources[:top_k]
    
    scored_sources = []
    for src in sources:
        prompt = f"""
        You are a search relevance evaluator. Rate the relevance of the following document chunk to the user's query.
        Rate it on a scale from 1 (completely irrelevant) to 10 (perfectly answers the query).

        Query: "{query}"
        Document Chunk: "{src['content']}"

        Respond with ONLY a single integer between 1 and 10. Do NOT include any explanations or extra characters.
        """
        try:
            resp = llm.invoke(prompt)
            text = resp.content.strip()
            match = re.search(r'\b(?:10|[1-9])\b', text)
            score = float(match.group()) if match else 1.0
            new_src = dict(src)
            new_src["rerank_score"] = score
            scored_sources.append(new_src)
        except Exception as e:
            logger.warning(f"[RERANKER] LLM fallback score failed for a chunk: {e}")
            new_src = dict(src)
            new_src["rerank_score"] = 1.0
            scored_sources.append(new_src)
            
    scored_sources.sort(key=lambda x: x.get("rerank_score", 1.0), reverse=True)
    return scored_sources[:top_k]

def rerank_documents(query: str, sources: list, reranker, llm=None, top_k=3) -> list:
    """
    Reranks a list of candidate documents based on Query-Passage scores.
    Falls back to LLM-based reranking if local Cross-Encoder is missing.
    """
    if not sources:
        return []
        
    if reranker:
        try:
            pairs = [[query, src["content"]] for src in sources]
            scores = reranker.predict(pairs)
            for src, score in zip(sources, scores):
                src["rerank_score"] = float(score)
            sources.sort(key=lambda x: x["rerank_score"], reverse=True)
            return sources[:top_k]
        except Exception as e:
            logger.warning(f"[RERANKER] Local reranking failed, attempting LLM fallback: {e}")
            
    if llm:
        return llm_rerank_fallback(query, sources, llm, top_k=top_k)
        
    return sources[:top_k]
