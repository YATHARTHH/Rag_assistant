import os
import sys
import rag_demo

print("Loading embedder...")
embedder = rag_demo.make_embedder()
print("Connecting to Qdrant...")
client = rag_demo.get_qdrant_client()
rag_demo.init_qdrant_collections(client)

import tasks
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
                chunks = rag_demo.parent_child_chunking(content, fname, embedder)
                if chunks:
                    rag_demo.delete_file_from_qdrant(client, fname, "public")
                    rag_demo.add_chunks_to_qdrant(client, chunks, "public", embedder)
                    rag_demo.invalidate_semantic_cache_by_file(client, fname)
            print(f"Finished indexing {fname}.")
    print("[SUCCESS] Local documents ingestion complete.")
