import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


import numpy as np
import pandas as pd
import faiss
import pdfplumber
import docx
import google.generativeai as genai

_LOCAL_MODEL = None

# -----------------------------
# Helpers
# -----------------------------
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_csv_text(s: str) -> str:
    # Token-friendly: remove commas/newlines that bloat CSV tokens
    s = s.replace(",", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Extraction
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                out.append(txt)
    return _clean_text(" ".join(out))

def extract_text_from_docx(path: str) -> str:
    d = docx.Document(path)
    parts = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    return _clean_text(" ".join(parts))

def extract_text(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return extract_text_from_pdf(path)
    if p.endswith(".docx"):
        return extract_text_from_docx(path)
    raise ValueError("Unsupported file type (only .pdf, .docx)")


# -----------------------------
# Chunking (token-aware-ish)
# -----------------------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple word-based chunking that is stable and cheap.
    chunk_size and overlap are in approx words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunk = _clean_text(chunk)
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)

    return chunks


# -----------------------------
# CSV compression (token saver)
# -----------------------------
def chunks_to_csv_rows(chunks: List[Dict]) -> str:
    """
    chunks: [{"source":..., "chunk_id":..., "content":...}, ...]
    Return token-efficient CSV string (single text block).
    """
    df = pd.DataFrame(chunks)
    # keep it compact
    csv_text = df.to_csv(index=False)
    csv_text = csv_text.replace("\n", " ")
    csv_text = re.sub(r"\s+", " ", csv_text).strip()
    return csv_text


# -----------------------------
# Gemini
# -----------------------------
@dataclass
class GeminiConfig:
    api_key: str
    chat_model: str = "gemini-2.0-flash"
    embed_model: str = "models/embedding-001"

def init_gemini(cfg: GeminiConfig):
    genai.configure(api_key=cfg.api_key)


def embed_texts(cfg, texts, task_type=None):
    """
    Local embeddings (Gemini-free)
    """
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        model_name = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        _LOCAL_MODEL = SentenceTransformer(model_name)

    vectors = _LOCAL_MODEL.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10
    )
    return np.array(vectors, dtype="float32")

def generate_answer(cfg: GeminiConfig, question: str, csv_context: str) -> str:
    model = genai.GenerativeModel(cfg.chat_model)

    prompt = f"""
You are a question-answering assistant.
Use ONLY the CSV_DATA to answer.
If the answer is not in the CSV_DATA, say: "I don't know based on the provided documents."
Keep it concise (max 3-5 sentences).

QUESTION:
{question}

CSV_DATA:
{csv_context}
""".strip()

    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip() or "Sorry, I couldn't generate a response."


# -----------------------------
# FAISS Index (with persistence)
# -----------------------------
class VectorIndex:
    """
    Stores:
    - chunks metadata (jsonl)
    - faiss index
    - dedupe map: chunk_hash -> idx
    """
    def __init__(self):
        self.index = None
        self.meta: List[Dict] = []
        self.hash_to_id: Dict[str, int] = {}

    def build(self, embeddings: np.ndarray, meta: List[Dict]):
        if len(embeddings) == 0:
            raise ValueError("No embeddings to build index.")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.meta = meta

        self.hash_to_id = {}
        for i, m in enumerate(meta):
            self.hash_to_id[m["chunk_hash"]] = i

    def add_new(self, embeddings: np.ndarray, meta: List[Dict]):
        if self.index is None:
            self.build(embeddings, meta)
            return

        self.index.add(embeddings)
        base = len(self.meta)
        self.meta.extend(meta)
        for i, m in enumerate(meta):
            self.hash_to_id[m["chunk_hash"]] = base + i

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None or len(self.meta) == 0:
            return []
        D, I = self.index.search(np.array([query_vec], dtype="float32"), top_k)
        hits = []
        for idx in I[0]:
            if 0 <= idx < len(self.meta):
                hits.append(self.meta[idx])
        return hits

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)
        with open(os.path.join(folder, "hash_to_id.json"), "w", encoding="utf-8") as f:
            json.dump(self.hash_to_id, f)

    def load(self, folder: str):
        idx_path = os.path.join(folder, "faiss.index")
        meta_path = os.path.join(folder, "meta.json")
        map_path = os.path.join(folder, "hash_to_id.json")

        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            return False

        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                self.hash_to_id = json.load(f)
        else:
            self.hash_to_id = {m["chunk_hash"]: i for i, m in enumerate(self.meta)}
        return True


# -----------------------------
# Build / Update pipeline
# -----------------------------
def build_or_update_index(
    cfg: GeminiConfig,
    vindex: VectorIndex,
    files: List[Tuple[str, str]],
    chunk_size: int,
    overlap: int,
):
    """
    files: [(filename, filepath), ...]
    Dedupe chunks by hash so re-upload doesn't re-embed.
    """
    new_meta = []
    new_texts = []

    for filename, path in files:
        text = extract_text(path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for j, c in enumerate(chunks):
            c_clean = _clean_text(c)
            c_hash = _sha1(c_clean)

            # skip if already indexed
            if c_hash in vindex.hash_to_id:
                continue

            new_texts.append(c_clean)
            new_meta.append({
                "source": filename,
                "chunk_id": j,
                "content": c_clean,
                "content_csv": _safe_csv_text(c_clean),
                "chunk_hash": c_hash,
            })

    if not new_texts:
        return 0  # nothing new

    # Embed new chunks
    embs = embed_texts(cfg, new_texts, task_type="retrieval_document")
    vindex.add_new(embs, new_meta)
    return len(new_texts)

def answer_with_rag(
    cfg: GeminiConfig,
    vindex: VectorIndex,
    question: str,
    top_k: int,
    max_context_chars: int = 12000,
) -> Tuple[str, List[Dict], str]:
    """
    Returns: (answer, hits, csv_context)
    """
    qvec = embed_texts(cfg, [question], task_type="retrieval_query")[0]
    hits = vindex.search(qvec, top_k=top_k)

    # Build token-friendly CSV context
    rows = []
    for h in hits:
        rows.append({
            "source": h["source"],
            "chunk_id": h["chunk_id"],
            "content": h["content_csv"],
        })

    csv_context = chunks_to_csv_rows(rows)
    csv_context = csv_context[:max_context_chars]

    answer = generate_answer(cfg, question, csv_context)
    return answer, hits, csv_context
