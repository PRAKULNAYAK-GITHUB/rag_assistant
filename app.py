import streamlit as st

from rag_core.config import settings
from rag_core.ingestion import delete_document, index_saved_upload, index_upload, list_unindexed_uploads, reindex_document
from rag_core.llm import generate_answer_stream
from rag_core.prompts import build_grounded_prompt
from rag_core.retrieval import build_citations, retrieve
from rag_core.store import (
    add_message,
    create_session,
    get_messages,
    get_or_create_default_session,
    list_documents,
    list_sessions,
    set_active_session,
)


st.set_page_config(page_title="LangChain RAG Explorer", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #f7f6f2; }
    .block-container { padding-top: 1.4rem; max-width: 1280px; }
    [data-testid="stSidebar"] { background: #fbfaf7; border-right: 1px solid #ded8cc; }
    .stButton > button { border-radius: 8px; border: 1px solid #cfc8bc; }
    .stTextInput > div > div > input { border-radius: 8px; }
    .chat-shell { border: 1px solid #ded8cc; border-radius: 8px; padding: 16px; background: #fffdf9; }
    .source-box { border: 1px solid #ded8cc; border-radius: 8px; padding: 12px; background: #fffdf9; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = get_or_create_default_session()

with st.sidebar:
    st.title("RAG Explorer")
    st.caption("LangChain + FastEmbed + Qdrant + Ollama")

    if st.button("New chat", type="primary", use_container_width=True):
        st.session_state.session_id = create_session()
        st.rerun()

    st.divider()
    st.subheader("Chats")
    for session in list_sessions():
        label = session["title"] or "New chat"
        if st.button(label, key=f"session-{session['id']}", use_container_width=True):
            set_active_session(session["id"])
            st.session_state.session_id = session["id"]
            st.rerun()

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    if uploads:
        st.info(f"{len(uploads)} file(s) selected. Click below to index them into Qdrant.")
    if uploads and st.button("Index selected file(s)", type="primary", use_container_width=True):
        with st.spinner("Extracting, chunking, embedding, and storing in Qdrant..."):
            for upload in uploads:
                index_upload(upload)
        st.success("Indexed uploads.")
        st.rerun()

    unindexed_uploads = list_unindexed_uploads()
    if unindexed_uploads:
        st.warning(f"{len(unindexed_uploads)} uploaded file(s) are saved but not indexed.")
        for path in unindexed_uploads:
            if st.button(f"Index saved: {path.name[37:] if len(path.name) > 37 else path.name}", key=f"saved-{path.name}", use_container_width=True):
                with st.spinner("Indexing saved upload into Qdrant..."):
                    index_saved_upload(path)
                st.rerun()

    selected_document_ids = []
    documents = list_documents()
    for doc in documents:
        selected = st.checkbox(
            f"{doc['filename']} ({doc['chunk_count']} chunks)",
            value=True,
            key=f"select-doc-{doc['id']}",
        )
        if selected:
            selected_document_ids.append(doc["id"])

        col_a, col_b = st.columns(2)
        if col_a.button("Re-index", key=f"reindex-{doc['id']}", use_container_width=True):
            with st.spinner("Re-indexing document..."):
                reindex_document(doc["id"])
            st.rerun()
        if col_b.button("Delete", key=f"delete-{doc['id']}", use_container_width=True):
            delete_document(doc["id"])
            st.rerun()

st.title("LangChain RAG Explorer")
st.caption(
    f"Embeddings: {settings.fastembed_model} | Vector DB: Qdrant | LLM: local Ollama {settings.ollama_chat_model}"
)

documents = list_documents()
messages = get_messages(st.session_state.session_id)

status_a, status_b, status_c = st.columns(3)
status_a.metric("Indexed documents", len(documents))
status_b.metric("Selected for retrieval", len(selected_document_ids))
status_c.metric("Chat messages", len(messages))

with st.expander("Document inventory", expanded=False):
    if documents:
        import pandas as pd

        st.dataframe(
            pd.DataFrame(documents)[["filename", "chunk_count", "created_at", "updated_at"]],
            use_container_width=True,
        )
    else:
        st.info("No indexed documents yet. Upload files in the sidebar, then click 'Index selected file(s)'.")

st.subheader("Chat")
if not documents:
    st.info("Index at least one document to start asking grounded questions.")
elif not selected_document_ids:
    st.warning("Select at least one indexed document in the sidebar before asking a question.")

with st.form("ask-form", clear_on_submit=True):
    question = st.text_area(
        "Ask the AI about your indexed documents",
        placeholder="Example: Summarize the main points and cite the source pages.",
        height=90,
    )
    submitted = st.form_submit_button(
        "Ask",
        type="primary",
        disabled=not documents or not selected_document_ids,
    )

if submitted and question.strip():
    add_message(st.session_state.session_id, "user", question)
    try:
        with st.status("Working on your answer...", expanded=True) as status:
            st.write("Retrieving relevant chunks from Qdrant...")
            hits = retrieve(question, selected_document_ids)
            st.write(f"Retrieved {len(hits)} chunk(s).")
            st.write("Building grounded prompt...")
            prompt = build_grounded_prompt(question, hits, get_messages(st.session_state.session_id))
            st.write(f"Streaming answer from local Ollama model `{settings.ollama_chat_model}`...")
            with st.chat_message("assistant"):
                answer = st.write_stream(generate_answer_stream(prompt))
            citations = build_citations(hits)
            add_message(st.session_state.session_id, "assistant", answer, citations)
            status.update(label="Answer ready", state="complete")
    except Exception as exc:
        error_message = (
            "I could not finish the answer. "
            f"Reason: {exc}"
        )
        add_message(st.session_state.session_id, "assistant", error_message, [])
        st.error(error_message)
    st.rerun()

messages = get_messages(st.session_state.session_id)
if messages:
    for row in messages:
        with st.chat_message(row["role"]):
            st.markdown(row["content"])
else:
    st.markdown(
        """
        <div class="chat-shell">
          Your conversation will appear here after you ask the first question.
        </div>
        """,
        unsafe_allow_html=True,
    )

last_answer = next((row for row in reversed(messages) if row["role"] == "assistant"), None)
if last_answer and last_answer.get("citations"):
    st.subheader("Sources")
    for citation in last_answer["citations"]:
        page = f", page {citation['page']}" if citation.get("page") else ""
        st.markdown(
            f"""
            <div class="source-box">
              <strong>{citation['filename']}{page}</strong>
              <p>{citation['snippet']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
