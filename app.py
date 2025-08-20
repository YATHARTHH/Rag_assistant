import streamlit as st
import os
from rag_demo import answer_research_question, get_collection, make_embedder
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# ---------- Load environment ----------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG-based AI Assistant")
st.write("Ask me anything! I will retrieve and generate answers using my knowledge base.")

# Initialize Chroma collection and embeddings
collection = get_collection("./research_db", "ml_publications")
embedder = make_embedder()
llm = ChatGroq(model="llama3-8b-8192", api_key=groq_key)

# User Input
query = st.text_input("Enter your question:", "")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        answer, sources = answer_research_question(query, collection, embedder, llm, top_k=3)
    
    st.subheader("Answer:")
    st.write(answer)

    # Show retrieved documents
    if sources:
        st.subheader("Top Retrieved Documents:")
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**{i}.** {doc['title']} (similarity={doc['similarity']:.3f})")
            st.markdown(f"{doc['content'][:500]}...")  # show first 500 chars
