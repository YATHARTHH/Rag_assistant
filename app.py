import streamlit as st
import requests
import pandas as pd
import json
import time
import io

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
    .hallucination-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar: Auth, Settings & File Manager
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
                        st.session_state.role = data["role"]
                        st.session_state.messages = []
                        st.session_state.ingest_tasks = {}
                        st.session_state.current_session_id = None
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
                    
        st.stop()
        
    else:
        st.success(f"🔑 User: **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    st.markdown("---")
    
    # 2. LLM Model Settings
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Language Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    # 3. Retrieval & Reranking Settings
    st.subheader("RAG Retrieval Logic")
    top_k = st.slider("Final Output Chunks", min_value=1, max_value=10, value=3)
    vector_weight = st.slider(
        "Retrieval Balance (Hybrid RRF)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.0 = Keyword (BM25) only, 1.0 = Vector search only, 0.5 = RRF Fusion."
    )
    window_size = st.slider("Sentence Window Context", min_value=0, max_value=4, value=2)
    
    enable_reranking = st.toggle("Enable Cross-Encoder Reranker", value=False)
    rerank_pool = st.slider(
        "Rerank Candidate Pool", 
        min_value=top_k + 2, 
        max_value=20, 
        value=max(top_k * 3, 10),
        disabled=not enable_reranking
    )
    
    # Upgrades: Parent Document, HyDE, Prompt style
    parent_retrieval = st.toggle("Enable Parent-Document Retrieval", value=True)
    hyde = st.toggle("Enable HyDE Query Expansion", value=False)
    
    prompt_style = st.selectbox(
        "Assistant Prompt Style",
        options=["Strict Fact-Only", "General Assistant", "Technical Summary"],
        index=0
    )
    
    # 4. Asynchronous Document Ingestion
    st.markdown("---")
    st.subheader("📤 Upload Documents (Celery + Qdrant)")
    uploaded_files = st.file_uploader(
        "Add files to vector index", 
        type=["txt", "pdf", "md", "docx", "csv", "json"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            state_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            if state_key not in st.session_state:
                with st.spinner(f"Queuing {uploaded_file.name}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        payload = {
                            "file_name": uploaded_file.name,
                            "file_bytes_hex": file_bytes.hex()
                        }
                        resp = requests.post(f"{API_URL}/ingest", json=payload, headers=headers)
                        if resp.status_code == 200:
                            task_id = resp.json()["task_id"]
                            st.session_state.ingest_tasks[task_id] = {
                                "file_name": uploaded_file.name,
                                "status": "processing"
                            }
                            st.session_state[state_key] = True
                            st.toast(f"📥 Queued in Celery worker: {uploaded_file.name}", icon="🚀")
                        else:
                            st.error(f"Upload failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Uploader error: {e}")
                        
    # 5. Ingestion Polling Tracker
    if "ingest_tasks" in st.session_state and st.session_state.ingest_tasks:
        completed_tasks = []
        for task_id, task in list(st.session_state.ingest_tasks.items()):
            if task["status"] == "processing":
                try:
                    resp = requests.get(f"{API_URL}/ingest/status/{task_id}", headers=headers)
                    if resp.status_code == 200:
                        backend_status = resp.json()["status"]
                        task["status"] = backend_status
                except Exception:
                    pass
            
            if task["status"] == "completed":
                st.caption(f"✅ `{task['file_name']}`: Indexed successfully")
                completed_tasks.append(task_id)
            elif task["status"].startswith("error"):
                st.error(f"❌ `{task['file_name']}`: {task['status']}")
            else:
                st.caption(f"🔄 `{task['file_name']}`: Parsing & Chunking...")
        time.sleep(0.5)

    # 6. Local Document File Manager
    st.markdown("---")
    st.subheader("📂 Document File Manager")
    try:
        stats_resp = requests.get(f"{API_URL}/db/stats", headers=headers)
        if stats_resp.status_code == 200:
            db_stats = stats_resp.json()
            st.caption(f"Total Database Chunks: {db_stats['total_chunks']}")
            for fname in db_stats["unique_files"]:
                col_name, col_del = st.columns([0.8, 0.2])
                col_name.caption(f"📄 {fname}")
                if col_del.button("🗑️", key=f"del_{fname}"):
                    del_resp = requests.delete(f"{API_URL}/db/files/{fname}", headers=headers)
                    if del_resp.status_code == 200:
                        st.toast(f"Deleted {fname}", icon="🗑️")
                        st.rerun()
    except Exception:
        pass

    # 7. Telemetry & Observability Links
    st.markdown("---")
    st.subheader("📊 Developer Telemetry")
    st.markdown("[🔍 Open Arize Phoenix Tracing (Port 6006)](http://localhost:6006)", unsafe_allow_html=True)
    st.markdown("[📈 Open Prometheus Metrics](/metrics)", unsafe_allow_html=True)

# -------------------------
# Main Page Layout: Tabs
# -------------------------
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

# Load Chat history / Sessions from SQLite database
if "sessions_loaded" not in st.session_state:
    try:
        sess_resp = requests.get(f"{API_URL}/chat/sessions", headers=headers)
        if sess_resp.status_code == 200:
            st.session_state.sessions_list = sess_resp.json()
            st.session_state.sessions_loaded = True
    except Exception:
        st.session_state.sessions_list = []

# -------------------------
# TAB 1: Chat Interface
# -------------------------
with tab_chat:
    # Historic Sessions Selector
    if "sessions_list" in st.session_state and st.session_state.sessions_list:
        sess_options = {s["session_id"]: f"Session: {s['session_id'][:8]} ({s['created_at'][:16]})" for s in st.session_state.sessions_list}
        selected_sess = st.selectbox("Restore Previous Chat Session", options=["New Chat"] + list(sess_options.keys()), format_func=lambda x: sess_options.get(x, "New Chat"))
        
        if selected_sess != "New Chat" and st.session_state.current_session_id != selected_sess:
            hist_resp = requests.get(f"{API_URL}/chat/history/{selected_sess}", headers=headers)
            if hist_resp.status_code == 200:
                st.session_state.messages = hist_resp.json()
                st.session_state.current_session_id = selected_sess
                st.rerun()

    # Display conversation history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("routing"):
                st.markdown(f"<div class='routing-tag'>⚡ Route: {message['routing']}</div>", unsafe_allow_html=True)
            if message.get("filter") and message["filter"] != "None":
                st.markdown(f"<div class='filter-tag'>📂 File Filter: {message['filter']}</div>", unsafe_allow_html=True)
                
            st.markdown(message["content"])
            
            # Hallucination warning banner
            if message.get("eval") and message["eval"].get("faithfulness", 1.0) < 0.70:
                st.markdown("<div class='hallucination-warning'>⚠️ Warning: Answer has high risk of hallucination (groundedness score < 0.70)</div>", unsafe_allow_html=True)
                
            # Render sources if present
            if message.get("sources"):
                with st.expander("📄 View Grounding Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}: {src['title']}</strong> (Page {src.get('page', 1)} | Relevance: {src['similarity']:.2f})
                            <p style="margin-top: 5px; font-size: 0.9em; color: #555;">{src['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            # Feedback loop triggers
            if message["role"] == "assistant" and "msg_id" in message:
                f_col1, f_col2 = st.columns([0.05, 0.95])
                if f_col1.button("👍", key=f"up_{idx}"):
                    requests.post(f"{API_URL}/chat/feedback", json={"message_id": message["msg_id"], "rating": 1}, headers=headers)
                    st.toast("Submitted positive feedback!")
                if f_col2.button("👎", key=f"down_{idx}"):
                    requests.post(f"{API_URL}/chat/feedback", json={"message_id": message["msg_id"], "rating": -1}, headers=headers)
                    st.toast("Submitted negative feedback!")

    # Chat Input Box
    if query := st.chat_input("Ask a question about the documents..."):
        # Display user question
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Display assistant generator stream
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown("🔍 Generating response stream...")
            response_placeholder = st.empty()
            
            payload = {
                "query": query,
                "history": st.session_state.messages[:-1],
                "username": st.session_state.username,
                "model_name": model_name,
                "temperature": temperature,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "window_size": window_size,
                "enable_reranking": enable_reranking,
                "rerank_pool": rerank_pool,
                "prompt_style": prompt_style,
                "parent_retrieval": parent_retrieval,
                "hyde": hyde
            }
            
            try:
                # Initiate SSE stream
                resp = requests.post(f"{API_URL}/chat", json=payload, headers=headers, stream=True)
                if resp.status_code == 200:
                    status_placeholder.empty()
                    full_response = ""
                    routing_info = "Direct Chat"
                    filter_info = "None"
                    sources = []
                    eval_scores = None
                    msg_id = str(uuid.uuid4().hex[:8]) if "uuid" in globals() else str(time.time())
                    
                    # Read stream lines (SSE data: prefix formats)
                    for line in resp.iter_lines():
                        if line:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                token = line_str[6:]
                                
                                # Intercept meta boundary data envelopes
                                if token.startswith("__METADATA_START__") and token.endswith("__METADATA_END__"):
                                    meta_data = json.loads(token[18:-16])
                                    routing_info = meta_data["routing"]
                                    filter_info = meta_data["filter"]
                                    sources = meta_data["sources"]
                                    
                                    # Show route details dynamically
                                    st.markdown(f"<div class='routing-tag'>⚡ Route: {routing_info}</div>", unsafe_allow_html=True)
                                    if filter_info != "None":
                                        st.markdown(f"<div class='filter-tag'>📂 File Filter: {filter_info}</div>", unsafe_allow_html=True)
                                        
                                # Intercept evaluation results boundary envelope
                                elif token.startswith("__EVAL_START__") and token.endswith("__EVAL_END__"):
                                    eval_scores = json.loads(token[14:-12])
                                    st.caption(f"📊 Evaluation scores: Groundedness/Faithfulness: {eval_scores['faithfulness']:.2f} | Answer Relevance: {eval_scores['relevance']:.2f} | Context Precision: {eval_scores['precision']:.2f}")
                                    
                                    if eval_scores['faithfulness'] < 0.70:
                                        st.markdown("<div class='hallucination-warning'>⚠️ Warning: Answer has high risk of hallucination (groundedness score < 0.70)</div>", unsafe_allow_html=True)
                                else:
                                    full_response += token
                                    response_placeholder.markdown(full_response + "▌")
                                    
                    response_placeholder.markdown(full_response)
                    
                    # Render sources dropdown
                    if sources:
                        with st.expander("📄 View Grounding Sources"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {i}: {src['title']}</strong> (Page {src.get('page', 1)} | Relevance: {src['similarity']:.2f})
                                    <p style="margin-top: 5px; font-size: 0.9em; color: #555;">{src['content']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    # Save response back to memory state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources if sources else None,
                        "routing": routing_info,
                        "filter": filter_info,
                        "eval": eval_scores,
                        "msg_id": msg_id
                    })
                    st.rerun()
                else:
                    status_placeholder.empty()
                    st.error(f"Error calling backend: {resp.status_code} - {resp.text}")
            except Exception as e:
                status_placeholder.empty()
                st.error(f"Request connection error: {e}")

    # Export Chat History Button
    if st.session_state.messages:
        chat_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Export Chat History (JSON)",
            data=chat_json,
            file_name=f"chat_history_{st.session_state.username}.json",
            mime="application/json"
        )

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
