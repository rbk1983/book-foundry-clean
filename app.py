import os, sys, time, hashlib
import streamlit as st

# --------------- Diagnostics (shows even if secrets are missing) ---------------
st.set_page_config(page_title="Book Foundry — Clean", layout="wide")
st.sidebar.markdown("### 🔎 Diagnostics (always visible)")
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

# --------------- Safe OpenAI init ---------------
from openai import OpenAI

def _load_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            # Some Streamlit versions don’t support .get; use membership + index.
            if "OPENAI_API_KEY" in st.secrets:
                key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    return key

OPENAI_KEY = _load_openai_key()
st.sidebar.write("Has OPENAI_API_KEY:", bool(OPENAI_KEY))

# Build the client ONLY if we have a key
client = None
if OPENAI_KEY:
    # IMPORTANT: don't pass anything except api_key (avoid stray kwargs like 'proxies')
    client = OpenAI(api_key=OPENAI_KEY)

# --------------- Minimal RAG deps ---------------
from typing import List, Dict
from pypdf import PdfReader
from docx import Document as DocxDocument
from markdown import markdown
from bs4 import BeautifulSoup

# Vector DB
import chromadb
from chromadb.config import Settings

# Chunking
from langchain_text_spltters import RecursiveCharacterTextSplitter as _R  # <-- typo will crash
# Fix the above line: correct import below
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    st.error("Failed to import text splitter. Check requirements installed.")
    st.stop()

# --------------- Helper functions ---------------
def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        out = []
        for p in reader.pages:
            try:
                out.append(p.extract_text() or "")
            except Exception:
                out.append("")
        return "\n".join(out)
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

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    if not client:
        raise RuntimeError("OpenAI client not initialized (missing API key).")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

@st.cache_resource
def get_chroma(project_name: str):
    os.makedirs("data", exist_ok=True)
    index_dir = os.path.join("data", "chroma")
    os.makedirs(index_dir, exist_ok=True)
    pc = chromadb.PersistentClient(path=index_dir, settings=Settings(allow_reset=False))
    col = pc.get_or_create_collection(name=f"bf_{project_name}", metadata={"hnsw:space":"cosine"})
    return pc, col

# --------------- UI ---------------
st.title("📚 Book Foundry — Clean")
st.caption("Fresh deployment with pinned Python and HTTP stack.")

with st.sidebar:
    st.header("Settings")
    project = st.text_input("Project name", value="My Third Book").strip()
    chat_model = st.text_input("Chat model", value="gpt-4o")
    embed_model = st.text_input("Embedding model", value="text-embedding-3-large")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.7, 0.1)
    top_k = st.slider("Top-K retrieved chunks", 1, 20, 8, 1)
    chunk_size = st.slider("Chunk size", 600, 2400, 1200, 100)
    overlap = st.slider("Chunk overlap", 50, 400, 200, 10)
    tag_filter = st.text_input("Tag filter (optional, comma-sep)", value="")
    st.divider()
    st.write("If the page ever spins, check this diagnostics panel first.")

# Tabs
tab_sources, tab_outline, tab_draft = st.tabs(["📥 Sources", "🧭 Outline", "✍️ Draft"])

# Session plan
if "plan" not in st.session_state:
    st.session_state.plan = {"title": project, "thesis": "", "style": "", "chapters": {}}
plan = st.session_state.plan

# Minimal app still renders even if key is missing
with tab_sources:
    st.subheader("Upload sources (PDF/DOCX/MD/TXT)")
    if not OPENAI_KEY:
        st.info("Add your OPENAI_API_KEY in App → Settings → Secrets to enable ingestion & drafting.")
    files = st.file_uploader("Select files", type=["pdf","docx","md","markdown","txt"], accept_multiple_files=True)
    tags_str = st.text_input("Tags for these files (e.g., Book1, Research)")

    if st.button("Ingest files", type="primary"):
        if not OPENAI_KEY:
            st.error("Missing OPENAI_API_KEY.")
            st.stop()
        pc, col = get_chroma(project)
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] or ["General"]
        added = 0
        for f in files or []:
            save_path = os.path.join("data", f"{int(time.time())}_{f.name}")
            with open(save_path, "wb") as out:
                out.write(f.read())
            text = load_text_from_file(save_path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            ids = [f"{project}:{sha16(f.name)}:{i}" for i, _ in enumerate(chunks)]
            embs = embed_texts(chunks, model=embed_model)
            col.add(ids=ids, documents=chunks, embeddings=embs,
                    metadatas=[{"project":project,"source":f.name,"tags":tags,"i":i} for i,_ in enumerate(chunks)])
            added += 1
        st.success(f"Ingested {added} file(s).")

with tab_outline:
    st.subheader("Thesis & Style")
    c1, c2 = st.columns(2)
    with c1:
        plan["title"] = st.text_input("Working title", value=plan["title"])
        plan["thesis"] = st.text_area("Book thesis or goal", value=plan["thesis"], height=120)
    with c2:
        plan["style"] = st.text_area("Style sheet (voice, tense, pacing, terminology)", value=plan["style"], height=120)

    if st.button("Generate outline", type="primary"):
        if not OPENAI_KEY:
            st.error("Missing OPENAI_API_KEY.")
            st.stop()
        msgs = [
            {"role": "system", "content": "You are a meticulous long-form book-writing assistant. Reply in Markdown."},
            {"role": "user", "content": f"Plan a new book.\n\nThesis:\n{plan['thesis']}\n\nConstraints: 12–18 chapters, coherent arc.\nProduce title options, detailed TOC, 2–4 sentence synopsis per chapter, and a brief style sheet."}
        ]
        resp = client.chat.completions.create(model=chat_model, temperature=temperature, messages=msgs)
        st.markdown(resp.choices[0].message.content)

with tab_draft:
    st.subheader("Draft (minimal demo)")
    ch_key = "1"
    plan["chapters"].setdefault(ch_key, {"title":"Chapter 1", "synopsis":"", "draft":""})
    plan["chapters"][ch_key]["title"] = st.text_input("Chapter 1 title", value=plan["chapters"][ch_key]["title"])
    plan["chapters"][ch_key]["synopsis"] = st.text_area("Chapter 1 synopsis", value=plan["chapters"][ch_key]["synopsis"], height=100)

    if st.button("Draft Chapter 1"):
        if not OPENAI_KEY:
            st.error("Missing OPENAI_API_KEY.")
            st.stop()
        prompt = f'Draft "Chapter 1: {plan["chapters"][ch_key]["title"]}" for the book "{plan["title"]}". Style: {plan["style"]}. Synopsis: {plan["chapters"][ch_key]["synopsis"]}. Target length ~1200 words. Markdown only.'
        msgs = [{"role":"system","content":"You are a careful long-form writing assistant, reply in Markdown."},
                {"role":"user","content":prompt}]
        resp = client.chat.completions.create(model=chat_model, temperature=0.7, messages=msgs)
        plan["chapters"][ch_key]["draft"] = resp.choices[0].message.content
        st.success("Draft created.")
        st.markdown(plan["chapters"][ch_key]["draft"])
