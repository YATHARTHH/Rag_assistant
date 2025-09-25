import streamlit as st
from rag_demo import make_embedder, get_collection, answer_research_question
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="RAG-based AI Assistant", layout="centered")
st.title("📘 RAG-based AI Assistant")
st.markdown("Ask questions about your documents and get context-aware answers.")

# Supported Groq Models
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

def get_working_llm(api_key):
    """Try supported models and return the first one that works."""
    for model_name in SUPPORTED_MODELS:
        try:
            llm = ChatGroq(temperature=0.2, groq_api_key=api_key, model=model_name)
            # Optionally run a quick test query
            llm.invoke("Hello")  
            st.info(f"Using model: {model_name}")
            return llm
        except Exception as e:
            st.warning(f"Model {model_name} not available: {e}")
    st.error("No working Groq model found. Check your API key or model list.")
    return None

# User Input
query = st.text_input("Enter your question:")

if query.strip():
    with st.spinner("Setting up backend..."):
        embedder = make_embedder()
        collection = get_collection(embedder)
        llm = get_working_llm(GROQ_API_KEY)

    if llm:
        if st.button("Get Answer"):
            with st.spinner("🔎 Retrieving and thinking..."):
                answer, sources = answer_research_question(query, collection, embedder, llm, top_k=3)
            st.subheader("💡 Answer")
            st.write(answer)

            if sources:
                st.subheader("📂 Top Retrieved Documents")
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"📄 {i}. {doc['title']} (similarity={doc['similarity']:.3f})"):
                        st.write(doc["content"])
