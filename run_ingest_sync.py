import os

import tasks
from database.qdrant import (
    add_chunks_to_qdrant,
    delete_file_from_qdrant,
    get_qdrant_client,
    init_qdrant_collections,
)
from rag.chunking import parent_child_chunking
from rag.embedding import make_embedder
from rag.search import invalidate_semantic_cache_by_file

print("Loading embedder...")
embedder = make_embedder()
print("Connecting to Qdrant...")
client = get_qdrant_client()
init_qdrant_collections(client)

docs_dir = "./docs"
if os.path.exists(docs_dir):
    print(f"Ingesting local files from {docs_dir}...")
    for fname in os.listdir(docs_dir):
        fpath = os.path.join(docs_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.txt', '.pdf', '.md', '.docx', '.csv', '.html', '.json')):
            print(f"Indexing {fname}...")
            res = tasks.ingest_file_task(fpath, fname, "public")
            if res and res.get("status") == "completed":
                content = res.get("content", "")
                chunks = parent_child_chunking(content, fname, embedder)
                if chunks:
                    delete_file_from_qdrant(client, fname, "public")
                    add_chunks_to_qdrant(client, chunks, "public", embedder)
                    invalidate_semantic_cache_by_file(client, fname)
            print(f"Finished indexing {fname}.")
    print("[SUCCESS] Local documents ingestion complete.")

