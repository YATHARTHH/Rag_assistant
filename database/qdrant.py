import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("rag_api")

def get_qdrant_client() -> QdrantClient:
    """
    Returns an instance of the local Qdrant vector client.
    """
    return QdrantClient(path="./qdrant_db")

def init_qdrant_collections(client: QdrantClient):
    """
    Initializes Qdrant collections with INT8 scalar quantization and keyword payload indices.
    """
    collections = ["research_papers", "query_cache"]
    for cname in collections:
        try:
            client.get_collection(collection_name=cname)
        except Exception:
            logger.info(f"[QDRANT] Creating collection: {cname}")
            client.create_collection(
                collection_name=cname,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True
                    )
                )
            )
            # Create payload indexes (supported in server mode, ignored gracefully in local mode)
            try:
                client.create_payload_index(
                    collection_name=cname,
                    field_name="user_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                client.create_payload_index(
                    collection_name=cname,
                    field_name="title",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass

def delete_file_from_qdrant(client: QdrantClient, filename: str, username: str):
    """
    Deletes all chunks associated with a file and user from research_papers collection.
    """
    try:
        client.delete(
            collection_name="research_papers",
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(key="title", match=models.MatchValue(value=filename)),
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=username))
                ]
            )
        )
    except Exception as e:
        logger.warning(f"[QDRANT] Delete file {filename} failed: {e}")

def add_chunks_to_qdrant(client: QdrantClient, chunks: list, username: str, embedder, doc_metadata: dict = None):
    """
    Embeds, encrypts, and uploads document chunks to Qdrant collection.
    """
    import uuid
    from security.encryption import encrypt_text
    from rag.embedding import CachedHuggingFaceEmbeddings
    
    cached_embedder = CachedHuggingFaceEmbeddings(embedder)
    texts = [c["content"] for c in chunks]
    embeddings = cached_embedder.embed_documents(texts)
    points = []
    meta = doc_metadata or {}
    for idx, c in enumerate(chunks):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{username}_{c['title']}_sent_{c['sent_index']}"))
        enc_content = encrypt_text(c["content"])
        enc_parent = encrypt_text(c["parent_text"])
        enc_overlap = encrypt_text(c["overlap_text"])
        points.append(models.PointStruct(
            id=point_id,
            vector=embeddings[idx],
            payload={
                "content": enc_content,
                "parent_text": enc_parent,
                "overlap_text": enc_overlap,
                "title": c["title"],
                "sent_index": c["sent_index"],
                "user_id": username,
                "author": meta.get("author", "Unknown"),
                "creation_date": meta.get("creation_date", ""),
                "file_size_kb": meta.get("file_size_kb", 0.0),
                "page_count": meta.get("page_count", 0)
            }
        ))
    try:
        client.upsert(
            collection_name="research_papers",
            points=points
        )
    except Exception as e:
        logger.error(f"[QDRANT] Upsert chunks failed: {e}")

