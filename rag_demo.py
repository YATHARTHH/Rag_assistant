"""
RAG Pipeline Script
-------------------
Implements a simple Retrieval-Augmented Generation (RAG) system.

Steps:
1. Load documents from ./docs
2. Split into overlapping chunks
3. Embed with HuggingFace model
4. Store in Chroma vector DB
5. Query -> Retrieve -> Generate answer using Groq LLM

This script focuses on:
- Chunk overlap (for semantic continuity across chunks)
- Similarity retrieval (cosine similarity via Chroma)
- Grounding LLM answers in retrieved sources
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -------------------------
# Embedding Model
# -------------------------
def make_embedder():
    """
    Creates a sentence transformer model to convert text into embeddings.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -------------------------
# Chunking Function
# -------------------------
def chunk_research_paper(paper_content: str, title: str):
    """
    Splits the document into overlapping chunks.
    - chunk_size = 1000 characters
    - chunk_overlap = 200 characters
    Overlap ensures continuity of meaning across chunks (so that cut sentences
    still retain context in adjacent chunks).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(paper_content)
    return [{"title": title, "content": chunk} for chunk in chunks]


# -------------------------
# Database Preparation
# -------------------------
def get_collection(embedder, persist_dir="./research_db"):
    """
    Returns a Chroma vector store collection.
    Stores embeddings and metadata in ./research_db for persistence.
    """
    return Chroma(
        collection_name="research_papers",
        embedding_function=embedder,
        persist_directory=persist_dir
    )


# -------------------------
# Retrieval
# -------------------------
def search_research_db(query: str, collection, embedder, top_k=5):
    """
    Retrieves top-k most relevant chunks using cosine similarity (via ChromaDB).
    Similarity scores are in range (0,1], with higher = more relevant.
    """
    results = collection.similarity_search_with_score(query, k=top_k)
    return [
        {"title": doc.metadata.get("title", "Unknown"), "content": doc.page_content, "similarity": 1 - score}
        for doc, score in results
    ]


# -------------------------
# Answer Generation
# -------------------------
def answer_research_question(query: str, collection, embedder, llm, top_k=3):
    """
    Retrieves context and uses the LLM to generate an answer.
    Retrieval acts as 'memory' by grounding the model's reasoning
    in relevant past documents.
    """
    sources = search_research_db(query, collection, embedder, top_k=top_k)

    # Concatenate retrieved chunks for context
    context = "\n\n".join([src["content"] for src in sources])

    prompt = f"""
    You are an AI research assistant.
    Use the following retrieved context to answer the question.

    Question: {query}

    Context:
    {context}

    Answer:
    """

    response = llm.invoke(prompt)
    return response.content, sources


# -------------------------
# Main Ingestion Step
# -------------------------
if __name__ == "__main__":
    embedder = make_embedder()
    collection = get_collection(embedder)

    docs_dir = "./docs"
    if not os.path.exists(docs_dir):
        print("⚠️ No docs directory found. Please add some .txt files inside ./docs")
        exit()

    for fname in os.listdir(docs_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
                content = f.read()
                chunks = chunk_research_paper(content, fname)
                collection.add_texts(
                    texts=[c["content"] for c in chunks],
                    metadatas=[{"title": c["title"]} for c in chunks]
                )
    collection.persist()
    print("✅ Documents ingested and stored in ./research_db")
