import logging
from typing import List
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from database.sqlite import check_embedding_cache, save_embedding_cache

logger = logging.getLogger("rag_api")

def make_embedder() -> FastEmbedEmbeddings:
    """
    Creates a sentence transformer model to convert text into embeddings.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

class CachedHuggingFaceEmbeddings:
    """
    Wrapper around embedder that leverages SQLite caching for text embeddings.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        for idx, text in enumerate(texts):
            cached = check_embedding_cache(text)
            if cached:
                results.append(cached)
            else:
                results.append(None)
                texts_to_embed.append(text)
                indices_to_embed.append(idx)
                
        if texts_to_embed:
            embedded = self.embedder.embed_documents(texts_to_embed)
            for idx, vector in zip(indices_to_embed, embedded):
                save_embedding_cache(texts[idx], vector)
                results[idx] = vector
                
        return results

    def embed_query(self, query: str) -> List[float]:
        cached = check_embedding_cache(query)
        if cached:
            return cached
        vector = self.embedder.embed_query(query)
        save_embedding_cache(query, vector)
        return vector
