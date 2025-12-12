import os
import time
import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    GeminiConfig, init_gemini,
    VectorIndex, build_or_update_index, answer_with_rag
)

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Chat with PDF/DOCX (Gemini RAG)", page_icon="📄", layout="wide")
st.title("📄 Chat with PDF/DOCX — Gemini RAG (Token Optimized)")

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

UPLOAD_DIR = "data/uploads"
INDEX_DIR = "data/index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

if not API_KEY:
    st.error("Missing GEMINI_API_KEY in .env")
    st.stop()

cfg = GeminiConfig(api_key=API_KEY, chat_model=CHAT_MODEL, embed_model=EMBED_MODEL)
init_gemini(cfg)

# -----------------------------
# Session state
# -----------------------------
if "vindex" not in st.session_state:
    st.session_state.vindex = VectorIndex()
    st.session_state.loaded = st.session_state.vindex.load(INDEX_DIR)

if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.caption("Chunking")
    chunk_size = st.slider("Chunk size (words)", 300, 2000, CHUNK_SIZE, 50)
    overlap = st.slider("Chunk overlap (words)", 0, 500, CHUNK_OVERLAP, 10)

    st.caption("Retrieval")
    top_k = st.slider("Top-K chunks", 1, 12, TOP_K, 1)

    st.caption("Context")
    max_context_chars = st.slider("Max context chars", 2000, 30000, MAX_CONTEXT_CHARS, 500)

    st.divider()

    if st.button("🧹 Clear index + chat", use_container_width=True):
        # Reset memory
        st.session_state.vindex = VectorIndex()
        st.session_state.chat = []

        # Delete persisted index
        for fn in ["faiss.index", "meta.json", "hash_to_id.json"]:
            p = os.path.join(INDEX_DIR, fn)
            if os.path.exists(p):
                os.remove(p)

        st.success("Cleared.")
        st.rerun()

# -----------------------------
# Upload + indexing
# -----------------------------
st.subheader("1) Upload documents")
uploaded = st.file_uploader(
    "Upload PDFs or DOCX (multiple supported)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

colA, colB = st.columns([1, 1])

with colA:
    st.info(f"Index status: **{len(st.session_state.vindex.meta)} chunks**"
            + (" (loaded from disk)" if st.session_state.loaded else ""))

with colB:
    if uploaded and st.button("📌 Process & Index", use_container_width=True):
        saved_files = []
        for f in uploaded:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            saved_files.append((f.name, save_path))

        with st.spinner("Extracting, chunking, embedding, indexing..."):
            t0 = time.time()
            added = build_or_update_index(
                cfg=cfg,
                vindex=st.session_state.vindex,
                files=saved_files,
                chunk_size=chunk_size,
                overlap=overlap
            )
            st.session_state.vindex.save(INDEX_DIR)
            dt = time.time() - t0

        st.success(f"Done. Added **{added}** new chunks in {dt:.1f}s.")
        st.rerun()

st.divider()

# -----------------------------
# Chat
# -----------------------------
st.subheader("2) Ask questions")

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    if len(st.session_state.vindex.meta) == 0:
        with st.chat_message("assistant"):
            st.warning("Index is empty. Upload documents and click **Process & Index** first.")
        st.session_state.chat.append(("assistant", "Index is empty. Please upload and index documents first."))
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating answer..."):
            answer, hits, csv_context = answer_with_rag(
                cfg=cfg,
                vindex=st.session_state.vindex,
                question=question,
                top_k=top_k,
                max_context_chars=max_context_chars
            )

        st.write(answer)

        with st.expander("🔎 Retrieved chunks (sources)"):
            for i, h in enumerate(hits, start=1):
                st.markdown(f"**{i}. {h['source']} — chunk {h['chunk_id']}**")
                st.write(h["content"][:800] + ("..." if len(h["content"]) > 800 else ""))

        with st.expander("🧾 CSV context sent to Gemini (token-optimized)"):
            st.code(csv_context, language="text")

    st.session_state.chat.append(("assistant", answer))
