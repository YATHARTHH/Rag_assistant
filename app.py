import streamlit as st
import requests
import pandas as pd
import time

# API Server URL
API_URL = "http://127.0.0.1:8000"

# -------------------------
# Page Configurations & Setup
# -------------------------
st.set_page_config(
    page_title="RAG AI Research Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .reportview-container {
        background: #f5f7fb;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stChatInputContainer {
        border-radius: 10px;
    }
    .st-emotion-cache-1c7n2ka {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 1.5rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.2em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #007bff;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .source-card {
        border-left: 4px solid #007bff;
        background-color: #f8f9fa;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 0 4px 4px 0;
    }
    .routing-tag {
        font-size: 0.85em;
        font-weight: 500;
        color: #6c757d;
        background: #e9ecef;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 8px;
        margin-right: 5px;
    }
    .filter-tag {
        font-size: 0.85em;
        font-weight: 500;
        color: #fff;
        background: #28a745;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar: Auth & Configurations
# -------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=70)
    st.title("RAG Control Panel")
    st.markdown("---")
    
    # 1. Authentication Manager
    if "token" not in st.session_state:
        st.subheader("Login / Sign Up")
        auth_mode = st.radio("Choose Mode", ["Login", "Sign Up"])
        username_in = st.text_input("Username", key="username_in")
        password_in = st.text_input("Password", type="password", key="password_in")
        
        if auth_mode == "Login":
            if st.button("Log In", use_container_width=True):
                try:
                    resp = requests.post(f"{API_URL}/auth/login", json={"username": username_in, "password": password_in})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.token = data["token"]
                        st.session_state.username = data["username"]
                        st.session_state.messages = []
                        st.session_state.ingest_tasks = {}
                        st.toast("✅ Logged in successfully!", icon="🎉")
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Login failed."))
                except Exception as e:
                    st.error(f"Backend offline: {e}")
        else:
            if st.button("Register Account", use_container_width=True):
                try:
                    resp = requests.post(f"{API_URL}/auth/signup", json={"username": username_in, "password": password_in})
                    if resp.status_code == 200:
                        st.success("✅ Account created! Switch to Login mode.")
                    else:
                        st.error(resp.json().get("detail", "Signup failed."))
                except Exception as e:
                    st.error(f"Backend offline: {e}")
                    
        # Stop render if unauthorized
        st.stop()
        
    else:
        st.success(f"🔑 Logged in as: **{st.session_state.username}**")
        if st.button("Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    st.markdown("---")
    
    # 2. LLM Model Settings
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Language Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    # 3. Retrieval & Reranking Settings
    st.subheader("Retrieval & Reranking")
    top_k = st.slider("Final Output Chunks", min_value=1, max_value=10, value=3)
    vector_weight = st.slider(
        "Retrieval Balance (Hybrid Search)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="1.0 = Dense Vector search only. 0.0 = Sparse BM25 Keyword search only. 0.5 = Balanced hybrid."
    )
    
    window_size = st.slider(
        "Sentence Window Context",
        min_value=0,
        max_value=4,
        value=2,
        help="Number of sibling sentences retrieved before and after the matching sentence to give the LLM broader context."
    )
    
    enable_reranking = st.toggle("Enable Cross-Encoder Reranker", value=False)
    rerank_pool = st.slider(
        "Candidate Retrieval Pool size", 
        min_value=top_k + 2, 
        max_value=20, 
        value=max(top_k * 3, 10),
        disabled=not enable_reranking
    )

    # 4. Asynchronous Document Ingestion
    st.markdown("---")
    st.subheader("📤 Upload Documents (Async)")
    uploaded_files = st.file_uploader(
        "Upload files to your DB", 
        type=["txt", "pdf", "md"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            state_key = f"upload_triggered_{uploaded_file.name}_{uploaded_file.size}"
            if state_key not in st.session_state:
                with st.spinner(f"Queuing {uploaded_file.name}..."):
                    try:
                        # Streamlit file upload reading
                        file_bytes = uploaded_file.read()
                        payload = {
                            "file_name": uploaded_file.name,
                            "file_bytes_hex": file_bytes.hex(),
                            "username": st.session_state.username
                        }
                        resp = requests.post(f"{API_URL}/ingest", json=payload)
                        if resp.status_code == 200:
                            task_id = resp.json()["task_id"]
                            st.session_state.ingest_tasks[task_id] = {
                                "file_name": uploaded_file.name,
                                "status": "processing"
                            }
                            st.session_state[state_key] = True
                            st.toast(f"📥 Queued {uploaded_file.name} successfully!", icon="🚀")
                        else:
                            st.error(f"Upload failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Uploader error: {e}")
                        
    # 5. Ingestion Progress & Polling Tracker
    if "ingest_tasks" in st.session_state and st.session_state.ingest_tasks:
        st.markdown("### 🔄 Indexing Tasks")
        completed_tasks = []
        for task_id, task in st.session_state.ingest_tasks.items():
            if task["status"] == "processing":
                # Poll Backend for status
                try:
                    resp = requests.get(f"{API_URL}/ingest/status/{task_id}")
                    if resp.status_code == 200:
                        backend_status = resp.json()["status"]
                        task["status"] = backend_status
                except Exception:
                    pass
            
            # Show visual indicators
            if task["status"] == "completed":
                st.caption(f"✅ `{task['file_name']}`: Indexing Completed")
                completed_tasks.append(task_id)
            elif task["status"].startswith("error"):
                st.error(f"❌ `{task['file_name']}`: {task['status']}")
            else:
                st.caption(f"🔄 `{task['file_name']}`: Indexing In Progress...")
        
        # Keep completed tasks in list or clear them as needed
        # We rerun if any status changes to keep the logs active
        time.sleep(0.5)

# -------------------------
# Main Page Layout: Tabs
# -------------------------
# Construct tabs
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

# -------------------------
# TAB 1: Chat Interface
# -------------------------
with tab_chat:
    # Display conversation history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("routing"):
                st.markdown(f"<div class='routing-tag'>⚡ Route: {message['routing']}</div>", unsafe_allow_html=True)
            if message.get("filter") and message["filter"] != "None" and message["filter"] != "N/A":
                st.markdown(f"<div class='filter-tag'>📂 File Filter: {message['filter']}</div>", unsafe_allow_html=True)
                
            st.markdown(message["content"])
            
            # Show grounding evaluation if available
            if message["role"] == "assistant" and message.get("eval"):
                scores = message["eval"]
                st.caption(
                    f"📊 **Evaluation scores:** Groundedness/Faithfulness: **{scores['faithfulness']:.2f}** | "
                    f"Answer Relevance: **{scores['relevance']:.2f}** | "
                    f"Context Precision: **{scores['precision']:.2f}**"
                )
            
            # If assistant has sources, show them in an expander
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View Grounding Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        score_label = f"Rerank Score: {src['rerank_score']:.2f}" if "rerank_score" in src else f"Similarity: {src['similarity']:.2f}"
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}: {src['title']}</strong> ({score_label})
                            <p style="margin-top: 5px; font-size: 0.9em; color: #555;">{src['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # User Query Input
    if query := st.chat_input("Ask a question about your documents..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

# Handle query logic on reload
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query = st.session_state.messages[-1]["content"]
    
    # Generate assistant response
    with tab_chat:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            status_placeholder.markdown("*🔎 Sending query to local backend API...*")
            
            # Format payload
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                chat_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "sources": msg.get("sources"),
                    "routing": msg.get("routing"),
                    "filter": msg.get("filter"),
                    "eval": msg.get("eval")
                })
                
            payload = {
                "query": query,
                "history": chat_history,
                "username": st.session_state.username,
                "model_name": model_name,
                "temperature": temperature,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "window_size": window_size,
                "enable_reranking": enable_reranking,
                "rerank_pool": rerank_pool
            }
            
            try:
                # Query backend server and enable chunked streaming
                response = requests.post(f"{API_URL}/chat", json=payload, stream=True)
                
                if response.status_code != 200:
                    status_placeholder.empty()
                    error_msg = f"API Error: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    full_response = ""
                    routing_info = "Document Retrieval (RAG)"
                    filter_info = "None"
                    sources = []
                    eval_scores = None
                    # Read token chunks stream with buffer parser
                    buffer = ""
                    for chunk in response.iter_content(decode_unicode=True):
                        if chunk:
                            buffer += chunk
                            
                            # Parse metadata envelope if present
                            if "__METADATA_START__" in buffer and "__METADATA_END__" in buffer:
                                try:
                                    start_idx = buffer.find("__METADATA_START__")
                                    end_idx = buffer.find("__METADATA_END__")
                                    meta_str = buffer[start_idx + len("__METADATA_START__"):end_idx]
                                    meta_data = eval(meta_str)
                                    routing_info = meta_data.get("routing", routing_info)
                                    filter_info = meta_data.get("filter", filter_info)
                                    sources = meta_data.get("sources", [])
                                    
                                    # Update UI header indicator tag
                                    tag_html = f"<div class='routing-tag'>⚡ Route: {routing_info}</div>"
                                    if filter_info != "None":
                                        tag_html += f"<div class='filter-tag'>📂 File Filter: {filter_info}</div>"
                                    status_placeholder.markdown(tag_html, unsafe_allow_html=True)
                                    
                                    # Strip metadata envelope from stream buffer
                                    buffer = buffer[:start_idx] + buffer[end_idx + len("__METADATA_END__"):]
                                except Exception:
                                    pass
                                    
                            # Parse evaluation metric envelope if present
                            if "__EVAL_START__" in buffer and "__EVAL_END__" in buffer:
                                try:
                                    start_idx = buffer.find("__EVAL_START__")
                                    end_idx = buffer.find("__EVAL_END__")
                                    eval_str = buffer[start_idx + len("__EVAL_START__"):end_idx]
                                    eval_scores = eval(eval_str)
                                    
                                    # Strip evaluation envelope from stream buffer
                                    buffer = buffer[:start_idx] + buffer[end_idx + len("__EVAL_END__"):]
                                except Exception:
                                    pass
                                    
                            # Construct display output hiding partial protocol tags
                            clean_display = buffer
                            for tag in ["__METADATA_START__", "__METADATA_END__", "__EVAL_START__", "__EVAL_END__"]:
                                if tag in clean_display:
                                    clean_display = clean_display.replace(tag, "")
                                    
                            # Hide partial starts of protocol tags if they fall on chunk boundaries
                            for tag in ["__METADATA_START__", "__METADATA_END__", "__EVAL_START__", "__EVAL_END__"]:
                                for i in range(1, len(tag)):
                                    if clean_display.endswith(tag[:i]):
                                        clean_display = clean_display[:-i]
                                        break
                                        
                            response_placeholder.markdown(clean_display + "▌")
                            full_response = clean_display
                            
                    # Remove streaming cursor
                    response_placeholder.markdown(full_response)
                    
                    # Log grounding eval scores below answer bubble
                    if eval_scores:
                        st.caption(
                            f"📊 **Evaluation scores:** Groundedness/Faithfulness: **{eval_scores['faithfulness']:.2f}** | "
                            f"Answer Relevance: **{eval_scores['relevance']:.2f}** | "
                            f"Context Precision: **{eval_scores['precision']:.2f}**"
                        )
                        
                    # Render sources if RAG was run
                    if sources:
                        with st.expander("📄 View Grounding Sources"):
                            for i, src in enumerate(sources, 1):
                                score_label = f"Rerank Score: {src['rerank_score']:.2f}" if "rerank_score" in src else f"Similarity: {src['similarity']:.2f}"
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {i}: {src['title']}</strong> ({score_label})
                                    <p style="margin-top: 5px; font-size: 0.9em; color: #555;">{src['content']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    # Save assistant message to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources if sources else None,
                        "routing": routing_info,
                        "filter": filter_info,
                        "eval": eval_scores
                    })
                    st.rerun()
                    
            except Exception as e:
                status_placeholder.empty()
                error_msg = f"Failed to connect to backend server: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()

# -------------------------
# TAB 2: Evaluation Dashboard
# -------------------------
with tab_eval:
    st.header("📊 Automated RAG Evaluation Dashboard")
    st.markdown("Provides continuous monitoring of RAG retrieval quality and answer groundedness using LLM-as-a-Judge evaluators.")
    st.markdown("---")
    
    rag_messages = [
        msg for msg in st.session_state.messages 
        if msg["role"] == "assistant" and msg.get("eval") is not None
    ]
    
    if not rag_messages:
        st.info("💡 **No RAG-based queries have been logged yet.** Run a retrieval question in the Chat tab to view evaluations.")
    else:
        avg_faithfulness = sum(m["eval"]["faithfulness"] for m in rag_messages) / len(rag_messages)
        avg_relevance = sum(m["eval"]["relevance"] for m in rag_messages) / len(rag_messages)
        avg_precision = sum(m["eval"]["precision"] for m in rag_messages) / len(rag_messages)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Groundedness / Faithfulness", f"{avg_faithfulness:.2%}")
        m_col2.metric("Answer Relevance", f"{avg_relevance:.2%}")
        m_col3.metric("Context Precision", f"{avg_precision:.2%}")
        
        st.markdown("### 📈 Metric Performance Over Time")
        
        chart_data = pd.DataFrame([
            {
                "Turn": idx + 1,
                "Faithfulness": msg["eval"]["faithfulness"],
                "Relevance": msg["eval"]["relevance"],
                "Context Precision": msg["eval"]["precision"]
            }
            for idx, msg in enumerate(rag_messages)
        ])
        
        st.line_chart(chart_data.set_index("Turn"))
        
        st.markdown("### 📋 Evaluation Logs")
        
        log_records = []
        for msg in rag_messages:
            msg_idx = st.session_state.messages.index(msg)
            user_query = st.session_state.messages[msg_idx - 1]["content"] if msg_idx > 0 else "Unknown"
            
            log_records.append({
                "User Query": user_query,
                "Active Filter": msg.get("filter", "None"),
                "Faithfulness": f"{msg['eval']['faithfulness']:.2f}",
                "Relevance": f"{msg['eval']['relevance']:.2f}",
                "Context Precision": f"{msg['eval']['precision']:.2f}"
            })
            
        st.dataframe(pd.DataFrame(log_records), use_container_width=True)
