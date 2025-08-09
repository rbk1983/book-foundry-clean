import os, sys, time, math, hashlib
import streamlit as st
from typing import List, Dict, Any

# ---------------- Diagnostics (render first) ----------------
st.set_page_config(page_title="Book Foundry — No-DB (Py3.13 ready)", layout="wide")
st.sidebar.markdown("### 🔎 Diagnostics")
st.sidebar.write("Python:", sys.version)

try:
    import openai as _oa
    st.sidebar.write("openai pkg:", getattr(_oa, "__version__", "unknown"))
except Exception as e:
    st.sidebar.write("openai import error:", e)

try:
    import httpx as _hx
    st.sidebar.write("httpx:", _hx.__version__)
except Exception as e:
    st.sidebar.write("httpx import error:", e)

# ---------------- Safe OpenAI init ----------------
from openai import OpenAI

def _load_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            if "OPENAI_API_KEY" in st.secrets:
                key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
st.sidebar.write("Has OPENAI_API_KEY:", bool(OPENAI_KEY))
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is missing. Add it in App → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_KEY)

# ---------------- Lightweight loaders & chunker ----------------
from pypdf import PdfReader
from docx import Document as DocxDocument
from markdown import markdown
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)
    elif ext == ".docx":
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in [".md", ".markdown"]:
        html = markdown(open(path, "r", encoding="utf-8", errors="ignore").read())
        return BeautifulSoup(html, "html.parser").get_text("\n")
    else:
        return open(path, "r", encoding="utf-8", errors="ignore").read()

def chunk_text(text: str, chunk_size=1200, overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n","\n",". ","? ","! "," ",""]
    )
    return splitter.split_text(text)

# ---------------- Embeddings (no numpy; pure Python cosine) ----------------
def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def cosine_sim(a: List[float], b: List[float]) -> float:
    # Pure Python cosine to avoid numpy dependency
    dot = 0.0; na = 0.0; nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# ---------------- Session state ----------------
if "plan" not in st.session_state:
    st.session_state.plan = {
        "title": "My Third Book",
        "thesis": "",
        "style": "",
        "chapters": {}  # "1": {"title": "...", "synopsis": "...", "draft": "..."}
    }
if "corpus" not in st.session_state:
    # corpus = list of dicts: {id, text, tags, source, emb}
    st.session_state.corpus = []

plan = st.session_state.plan
corpus: List[Dict[str, Any]] = st.session_state.corpus

# ---------------- UI ----------------
st.title("📚 Book Foundry — No-DB (clean start)")
st.caption("Uploads → in-memory embeddings → outline → chapter drafts. No Chroma, so it works on Python 3.13.")

with st.sidebar:
    st.header("Settings")
    plan["title"] = st.text_input("Project name", value=plan["title"]).strip()
    chat_model = st.text_input("Chat model", value="gpt-4o")
    embed_model = st.text_input("Embedding model", value="text-embedding-3-large")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.7, 0.1)
    top_k = st.slider("Top-K retrieved chunks", 1, 20, 8, 1)
    chunk_size = st.slider("Chunk size", 600, 2400, 1200, 100)
    overlap = st.slider("Chunk overlap", 50, 400, 200, 10)
    tag_filter = st.text_input("Tag filter (comma-sep, optional)", value="")

tab_sources, tab_outline, tab_draft, tab_export = st.tabs(
    ["📥 Sources", "🧭 Outline", "✍️ Draft", "📤 Export"]
)

# ---------------- SOURCES ----------------
with tab_sources:
    st.subheader("Upload sources (PDF/DOCX/MD/TXT)")
    files = st.file_uploader("Select files", type=["pdf","docx","md","markdown","txt"], accept_multiple_files=True)
    tags_str = st.text_input("Tags for these files (e.g., Book1, Research)")

    if st.button("Ingest files", type="primary"):
        if not files:
            st.warning("No files selected.")
        else:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] or ["General"]
            added_chunks = 0
            os.makedirs("data", exist_ok=True)
            for f in files:
                tmp = os.path.join("data", f"{int(time.time())}_{f.name}")
                with open(tmp, "wb") as out:
                    out.write(f.read())
                raw = load_text_from_file(tmp)
                pieces = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
                embs = embed_texts(pieces, model=embed_model)
                for i, (text, emb) in enumerate(zip(pieces, embs)):
                    corpus.append({
                        "id": f"{sha16(f.name)}:{i}",
                        "text": text,
                        "tags": tags,
                        "source": f.name,
                        "emb": emb,
                    })
                    added_chunks += 1
            st.success(f"Ingested {len(files)} file(s) → {added_chunks} chunks.")
    if corpus:
        st.write(f"Corpus size: **{len(corpus)}** chunks across your uploads.")
    else:
        st.info("No chunks yet. Upload and ingest above.")

# ---------------- OUTLINE ----------------
with tab_outline:
    st.subheader("Thesis & Style")
    c1, c2 = st.columns(2)
    with c1:
        plan["thesis"] = st.text_area("Book thesis or goal", value=plan["thesis"], height=120)
    with c2:
        plan["style"] = st.text_area("Style sheet (voice, tense, pacing, terminology)", value=plan["style"], height=120)

    if st.button("Generate outline", type="primary"):
        msgs = [
            {"role":"system","content":"You are a meticulous long-form book-writing assistant. Reply in clean Markdown."},
            {"role":"user","content":f"Plan a new book.\n\nThesis:\n{plan['thesis']}\n\nConstraints: 12–18 chapters, coherent arc.\nProduce title options, detailed TOC, 2–4 sentence synopsis per chapter, and a brief style sheet."}
        ]
        resp = client.chat.completions.create(model=chat_model, temperature=temperature, messages=msgs)
        st.markdown(resp.choices[0].message.content)
        st.info("Copy/paste chapter titles & synopses into the Draft tab.")

# ---------------- RETRIEVAL (in-memory) ----------------
def retrieve(query: str, k: int, tag_filter_text: str) -> List[Dict[str, Any]]:
    if not corpus:
        return []
    filters = {t.strip().lower() for t in tag_filter_text.split(",") if t.strip()}
    qvec = embed_texts([query], model=embed_model)[0]
    scored = []
    for rec in corpus:
        if filters and not (set(map(str.lower, rec["tags"])) & filters):
            continue
        score = cosine_sim(qvec, rec["emb"])
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

# ---------------- DRAFT ----------------
with tab_draft:
    st.subheader("Draft a chapter")
    # very simple chapter editor
    ch_keys = list(st.session_state.plan["chapters"].keys()) or ["1"]
    default_key = ch_keys[0]
    ch_num = st.text_input("Chapter number", value=default_key)
    ch_state = st.session_state.plan["chapters"].setdefault(ch_num, {"title":"", "synopsis":"", "draft":"_
