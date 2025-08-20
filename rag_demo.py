import os
import chromadb
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ---------- 0) Load environment ----------
load_dotenv()  # Reads .env file if present

# ---------- 1) Loading ----------
def load_research_publications(documents_path: str) -> list[str]:
    docs = []
    for fname in os.listdir(documents_path):
        if fname.endswith(".txt"):
            path = os.path.join(documents_path, fname)
            try:
                loaded = TextLoader(path).load()
                docs.extend(loaded)
                print(f"Loaded: {fname}")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    print(f"Total docs (chunks by loader): {len(docs)}")
    return [d.page_content for d in docs]

# ---------- 2) Chunking ----------
def chunk_research_paper(paper_content: str, title: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    pieces = splitter.split_text(paper_content)
    return [
        {"content": c, "title": title, "chunk_id": f"{title}_{i}"}
        for i, c in enumerate(pieces)
    ]

# ---------- 3) Embeddings ----------
def make_embedder():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

def embed_documents(embedder, texts: list[str]) -> list[list[float]]:
    return embedder.embed_documents(texts)

# ---------- 4) Vector DB (Chroma) ----------
def get_collection(db_path="./research_db", name="ml_publications"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def insert_publications(collection, publications: list[str], embedder):
    next_id = collection.count()
    for idx, publication in enumerate(publications):
        title = f"pub_{next_id}_{idx}"
        chunks = chunk_research_paper(publication, title)
        texts = [c["content"] for c in chunks]
        vecs = embed_documents(embedder, texts)
        ids = [f"document_{i}" for i in range(next_id, next_id + len(chunks))]
        collection.add(
            embeddings=vecs,
            ids=ids,
            documents=texts,
            metadatas=[{"title": c["title"], "chunk_id": c["chunk_id"]} for c in chunks],
        )
        next_id += len(chunks)
    print(f"Ingestion complete. Collection count: {collection.count()}")

# ---------- 5) Retrieval ----------
def search_research_db(query: str, collection, embedder, top_k=5):
    qv = embedder.embed_query(query)
    res = collection.query(
        query_embeddings=[qv],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    if not res["documents"] or len(res["documents"][0]) == 0:
        return []
    chunks = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i]
        # Convert distance to a bounded similarity in (0,1]
        similarity = 1.0 / (1.0 + dist)
        chunks.append({"content": doc, "title": meta.get("title", "unknown"), "similarity": similarity})
    return chunks

# ---------- 6) Generation ----------
def answer_research_question(query: str, collection, embedder, llm, top_k=3):
    hits = search_research_db(query, collection, embedder, top_k=top_k)
    if not hits:
        return "No relevant research found for this query.", []

    # Trim each chunk a bit to keep prompt lean
    context = "\n\n".join([f"From {h['title']} (sim={h['similarity']:.3f}):\n{h['content'][:1200]}" for h in hits])

    prompt_tmpl = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an assistant answering strictly from the provided research context.\n\n"
            "Research Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer: Provide a precise, well-structured answer. "
            "If something isn't supported by the context, say so. "
            "End with a short bullet list of key citations like [title]."
        )
    )
    prompt = prompt_tmpl.format(context=context, question=query)
    # Either .invoke() or .predict(); .predict returns str directly
    try:
        answer = llm.invoke(prompt)
    except Exception:
        answer = llm.invoke(prompt).content
    return answer, hits

# ---------- 7) Main demo ----------
if __name__ == "__main__":
    # 0) Load Groq API key from env
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file!")

    # 1) Init
    collection = get_collection("./research_db", "ml_publications")
    embedder = make_embedder()
    llm = ChatGroq(model="llama3-8b-8192", api_key=groq_key)

    # 2) Ingest sample docs (expects ./docs/*.txt)
    docs_dir = "./docs"
    if os.path.isdir(docs_dir):
        pubs = load_research_publications(docs_dir)
        if pubs:
            insert_publications(collection, pubs, embedder)
    else:
        print("No ./docs folder detected; skipping ingestion (collection may already have data).")

    # 3) Ask a question
    question = "What are effective techniques for handling class imbalance?"
    answer, sources = answer_research_question(question, collection, embedder, llm, top_k=3)

    # 4) Print results
    print("\n=== AI Answer ===\n")
    print(answer)
    print("\n=== Sources Used ===")
    for s in sources:
        print(f"- {s['title']} (sim={s['similarity']:.3f})")
