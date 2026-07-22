import math
import uuid
import time
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from database.qdrant import get_qdrant_client
from security.encryption import decrypt_text, encrypt_text
from rag.embedding import CachedHuggingFaceEmbeddings

logger = logging.getLogger("rag_api")

def search_qdrant(client: QdrantClient, collection_name: str, query: str, embedder, username: str, top_k=5, title_filter=None) -> List[Dict]:
    """
    Performs vector search in Qdrant with tenant isolation and optional title filtering.
    """
    cached_embedder = CachedHuggingFaceEmbeddings(embedder)
    query_vector = cached_embedder.embed_query(query)
    
    # Audit: Enforce tenant isolation on user_id payload field
    must_conditions = [
        models.Filter(
            should=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=username)),
                models.FieldCondition(key="user_id", match=models.MatchValue(value="public"))
            ]
        )
    ]
    if title_filter:
        must_conditions.append(
            models.FieldCondition(key="title", match=models.MatchValue(value=title_filter))
        )
        
    query_filter = models.Filter(must=must_conditions)
    res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k * 2,
        with_payload=True
    )
    
    processed_results = []
    for r in res.points:
        payload = r.payload
        try:
            dec_content = decrypt_text(payload["content"])
            dec_parent = decrypt_text(payload["parent_text"])
            dec_overlap = decrypt_text(payload["overlap_text"])
        except Exception:
            dec_content = payload["content"]
            dec_parent = payload.get("parent_text", payload["content"])
            dec_overlap = payload.get("overlap_text", payload["content"])
            
        processed_results.append({
            "content": dec_content,
            "parent_text": dec_parent,
            "overlap_text": dec_overlap,
            "title": payload["title"],
            "sent_index": payload["sent_index"],
            "user_id": payload["user_id"],
            "score": r.score
        })
    return processed_results

def run_bm25_on_candidates(query: str, candidates: List[Dict], top_k=5) -> List[Dict]:
    """
    Ranks candidates using BM25 query term matching on context.
    """
    if not candidates:
        return []
    corpus = [c["content"] for c in candidates]
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    query_tokens = set(query.lower().split(" "))
    scored_candidates = []
    for idx, cand in enumerate(candidates):
        cand_tokens = tokenized_corpus[idx]
        score = sum(cand_tokens.count(tok) for tok in query_tokens if tok in cand_tokens)
        scored_candidates.append((score, cand))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_candidates][:top_k]

def reciprocal_rank_fusion(vector_results: List[Dict], bm25_results: List[Dict], k=60) -> List[Dict]:
    """
    Merges dense and sparse candidates using RRF.
    """
    scores = {}
    for rank, doc in enumerate(vector_results):
        text = doc["content"]
        if text not in scores:
            scores[text] = {"doc": doc, "score": 0.0}
        scores[text]["score"] += 1.0 / (k + rank + 1)
        
    for rank, doc in enumerate(bm25_results):
        text = doc["content"]
        if text not in scores:
            scores[text] = {"doc": doc, "score": 0.0}
        scores[text]["score"] += 1.0 / (k + rank + 1)
        
    sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs]

def maximal_marginal_relevance(query: str, candidates: List[Dict], embedder, lambda_mult=0.5, top_k=3) -> List[Dict]:
    """
    Diversifies RAG context using MMR.
    """
    if not candidates or len(candidates) <= top_k:
        return candidates[:top_k]
        
    cached_embedder = CachedHuggingFaceEmbeddings(embedder)
    query_vector = cached_embedder.embed_query(query)
    texts = [c["content"] for c in candidates]
    embeddings = cached_embedder.embed_documents(texts)
    selected_indices = []
    unselected_indices = list(range(len(candidates)))
    
    def cos_sim(v1, v2):
        dot = sum(a*b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a*a for a in v1))
        norm2 = math.sqrt(sum(b*b for b in v2))
        if norm1 * norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
        
    first_idx = unselected_indices.pop(0)
    selected_indices.append(first_idx)
    
    while len(selected_indices) < top_k and unselected_indices:
        best_mmr = -100.0
        best_idx = None
        for idx in unselected_indices:
            sim_query = cos_sim(embeddings[idx], query_vector)
            sim_selected = max(cos_sim(embeddings[idx], embeddings[s_idx]) for s_idx in selected_indices)
            mmr_score = lambda_mult * sim_query - (1 - lambda_mult) * sim_selected
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        if best_idx is not None:
            unselected_indices.remove(best_idx)
            selected_indices.append(best_idx)
        else:
            break
    return [candidates[idx] for idx in selected_indices]

def compress_context_with_llm(query: str, chunks: List[Dict], llm, top_k=3) -> List[Dict]:
    """
    Contextual LLM chunk summarizer to trim irrelevant sentences.
    """
    compressed_chunks = []
    for c in chunks:
        prompt = f"""
        You are a search precision optimizer. Given the user query and a passage, extract only the sentence(s) from the passage that directly answer the query.
        Do NOT change the words or paraphrase. Only extract the exact matching sentences.
        If no sentence is relevant, respond with "NONE".

        Query: "{query}"
        Passage:
        "{c['content']}"

        Extracted sentence(s):
        """
        try:
            resp = llm.invoke(prompt)
            extracted = resp.content.strip()
            if extracted and extracted != "NONE" and len(extracted) > 10:
                new_c = dict(c)
                new_c["content"] = extracted
                compressed_chunks.append(new_c)
            else:
                compressed_chunks.append(c)
        except Exception:
            compressed_chunks.append(c)
    return compressed_chunks[:top_k]

def lost_in_the_middle_reorder(sources: List[Dict]) -> List[Dict]:
    """
    Re-orders sources so high-scoring ones sit at the beginning/end to combat LLM forgetfulness.
    """
    if len(sources) <= 2:
        return sources
        
    sorted_sources = sorted(sources, key=lambda x: x.get("similarity", 0.5), reverse=True)
    reordered = [None] * len(sorted_sources)
    left = 0
    right = len(sorted_sources) - 1
    
    for i, item in enumerate(sorted_sources):
        if i % 2 == 0:
            reordered[left] = item
            left += 1
        else:
            reordered[right] = item
            right -= 1
    return reordered

def retrieve_context(query: str, client: QdrantClient, embedder, top_k=3, vector_weight=0.5, window_size=2, metadata_filter=None, user_id=None, parent_retrieval=False) -> List[Dict]:
    """
    Consolidated RAG retrieval using hybrid search, RRF, MMR, and Lost-in-the-Middle reordering.
    """
    # 1. Fetch dense vector results
    vector_candidates = search_qdrant(
        client, "research_papers", query, embedder, username=user_id, top_k=top_k * 4, title_filter=metadata_filter
    )
    
    # 2. Fetch all matching documents for BM25 candidates
    all_chunks = []
    try:
        must_cond = [
            models.Filter(
                should=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="user_id", match=models.MatchValue(value="public"))
                ]
            )
        ]
        if metadata_filter:
            must_cond.append(models.FieldCondition(key="title", match=models.MatchValue(value=metadata_filter)))
            
        scroll_res = client.scroll(
            collection_name="research_papers",
            scroll_filter=models.Filter(must=must_cond),
            limit=5000,
            with_payload=True
        )
        if scroll_res and scroll_res[0]:
            for item in scroll_res[0]:
                payload = item.payload
                try:
                    dec_content = decrypt_text(payload["content"])
                    dec_parent = decrypt_text(payload["parent_text"])
                    dec_overlap = decrypt_text(payload["overlap_text"])
                except Exception:
                    dec_content = payload["content"]
                    dec_parent = payload.get("parent_text", payload["content"])
                    dec_overlap = payload.get("overlap_text", payload["content"])
                all_chunks.append({
                    "content": dec_content,
                    "parent_text": dec_parent,
                    "overlap_text": dec_overlap,
                    "title": payload["title"],
                    "sent_index": payload["sent_index"],
                    "user_id": payload["user_id"]
                })
    except Exception as e:
        logger.warning(f"[SEARCH] Scroll candidates error: {e}")
        
    # 3. Perform sparse BM25
    bm25_candidates = run_bm25_on_candidates(query, all_chunks, top_k=top_k * 4)
    
    # 4. RRF ranking merge
    rrf_candidates = reciprocal_rank_fusion(vector_candidates, bm25_candidates, k=60)
    
    # 5. MMR diversification
    mmr_candidates = maximal_marginal_relevance(query, rrf_candidates, embedder, lambda_mult=0.5, top_k=top_k * 2)
    
    # Format candidates
    sources = []
    for c in mmr_candidates:
        retrieved_text = c["parent_text"] if parent_retrieval else c["content"]
        score = c.get("score", 0.5)
        sources.append({
            "title": c["title"],
            "content": retrieved_text,
            "similarity": score,
            "page": c.get("sent_index", 0) // 5 + 1
        })
        
    # Apply Lost-in-the-Middle reordering to retrieved sources
    reordered_sources = lost_in_the_middle_reorder(sources[:top_k])
    return reordered_sources

# -------------------------
# Semantic Query Caching Methods
# -------------------------
def check_semantic_cache(client: QdrantClient, query: str, embedder, score_threshold=0.90, ttl_days=7) -> tuple:
    cached_embedder = CachedHuggingFaceEmbeddings(embedder)
    query_vector = cached_embedder.embed_query(query)
    try:
        res = client.query_points(
            collection_name="query_cache",
            query=query_vector,
            limit=1
        )
        if res.points:
            pt = res.points[0]
            similarity = pt.score
            if similarity >= score_threshold:
                payload = pt.payload
                entry_time = payload.get("timestamp", 0)
                if time.time() - entry_time > ttl_days * 24 * 3600:
                    client.delete(collection_name="query_cache", points_selector=[pt.id])
                    return None, 0.0
                return payload["response"], similarity
    except Exception as e:
        logger.warning(f"[CACHE] Check error: {e}")
    return None, 0.0

def save_to_semantic_cache(client: QdrantClient, query: str, response: str, source_files: List[str], embedder):
    cached_embedder = CachedHuggingFaceEmbeddings(embedder)
    vector = cached_embedder.embed_query(query)
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"cache_{query}"))
    try:
        client.upsert(
            collection_name="query_cache",
            points=[models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "query": query,
                    "response": response,
                    "source_files": ",".join(source_files),
                    "timestamp": time.time()
                }
            )]
        )
    except Exception as e:
        logger.warning(f"[CACHE] Save error: {e}")

def invalidate_semantic_cache_by_file(client: QdrantClient, filename: str):
    try:
        results = client.scroll(
            collection_name="query_cache",
            limit=1000,
            with_payload=True
        )
        if results and results[0]:
            points_to_delete = []
            for item in results[0]:
                source_files = item.payload.get("source_files", "").split(",")
                if filename in source_files:
                    points_to_delete.append(item.id)
            if points_to_delete:
                client.delete(collection_name="query_cache", points_selector=points_to_delete)
    except Exception as e:
        logger.warning(f"[CACHE] Invalidation failed: {e}")

def generate_metadata_filter(query: str, llm, unique_files: List[str]) -> Optional[str]:
    """
    Analyzes the query and checks if the user is asking about a specific document from a list of unique file names.
    """
    if not unique_files:
        return None
        
    files_list_str = ", ".join([f"'{f}'" for f in unique_files])
    prompt = f"""
    You are a database helper. 
    Analyze the user's query and decide if they are asking about a specific document or file from this list: [{files_list_str}].

    User Query: "{query}"

    If they are asking about a specific file, respond with ONLY the exact name of the file from the list.
    If they are NOT asking about a specific file, or if the file they mention is not in the list, respond with 'NONE'.
    Do NOT include any punctuation, quotes, or extra text.
    """
    try:
        response = llm.invoke(prompt)
        selected_file = response.content.strip()
        selected_file = selected_file.replace("'", "").replace('"', "")
        if selected_file in unique_files:
            return selected_file
    except Exception as e:
        logger.warning(f"[SEARCH] Error generating metadata filter: {e}")
    return None

def spell_correct_query(query: str) -> str:
    """
    Corrects spelling errors in the query using pyspellchecker before retrieval.
    """
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        domain_terms = ['rrf', 'hyde', 'smote', 'adasyn', 'mcc', 'auc-roc', 'auc-pr', 'g-mean', 'bleu', 'rouge', 'f1', 'f1-score']
        spell.word_frequency.load_words(domain_terms)
        
        words = query.split()
        corrected = []
        for w in words:
            stripped = w.strip(".,?!;:()\"'")
            if stripped.isupper() or "-" in stripped or stripped.lower() in domain_terms:
                corrected.append(w)
            else:
                corr = spell.correction(stripped)
                if corr:
                    left_punc = w[:len(w) - len(w.lstrip(".,?!;:()\"'"))]
                    right_punc = w[len(w.rstrip(".,?!;:()\"'")):]
                    corrected.append(left_punc + corr + right_punc)
                else:
                    corrected.append(w)
                    
        corrected_query = " ".join(corrected)
        if corrected_query.lower() != query.lower():
            logger.info(f"[SPELL CORRECT] '{query}' -> '{corrected_query}'")
        return corrected_query
    except Exception:
        return query

def generate_hyde_response(query: str, llm) -> str:
    """
    Generates a hypothetical answer paragraph using LLM (HyDE).
    """
    prompt = f"""
    You are a helpful research assistant. 
    Write a short hypothetical paragraph that directly answers the following question. 
    Do not worry about being perfectly accurate; just write a plausible answer.

    Question: {query}
    Hypothetical Answer:
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        if text:
            return text
    except Exception as e:
        logger.warning(f"[SEARCH] HyDE response generation failed: {e}")
    return query

