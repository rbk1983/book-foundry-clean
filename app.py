# app.py — Book Auto-Generator (robust autosave + resume)

import os, sys, time, math, hashlib, base64, json, re, io, random, datetime as _dt
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Book Auto-Generator", layout="wide")

# ---------- Diagnostics ----------
with st.sidebar:
    st.markdown("### 🔍 Diagnostics")
    st.write("Python:", sys.version.split()[0])
    try:
        import openai as _oa
        st.write("openai pkg:", getattr(_oa, "__version__", "unknown"))
    except Exception as e:
        st.write("openai import error:", e)
    try:
        import httpx as _hx
        st.write("httpx:", getattr(_hx, "__version__", "unknown"))
    except Exception as e:
        st.write("httpx import error:", e)
    has_tavily = False
    try:
        has_tavily = bool(st.secrets.get("TAVILY_API_KEY", ""))
    except Exception:
        has_tavily = bool(os.getenv("TAVILY_API_KEY", ""))
    st.write("Tavily key:", "✅" if has_tavily else "—")

os.makedirs("data", exist_ok=True)

# ---------- Secrets helper ----------
def _sec(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

# ---------- OpenAI client + robust retry ----------
from openai import OpenAI
OPENAI_KEY = _sec("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)

def _backoff_sleep(attempt: int):
    # jittered exponential backoff: 0.5, 1, 2, 4, 8, ... plus small jitter
    base = min(8, 0.5 * (2 ** attempt))
    time.sleep(base + random.uniform(0, 0.5))

def chat_with_retry(messages: List[Dict[str,str]], model: str, temperature: float, max_tokens: int, tries: int = 5) -> str:
    last_err = None
    for attempt in range(tries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=messages, max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            _backoff_sleep(attempt)
    raise RuntimeError(f"OpenAI chat failed after {tries} attempts: {last_err}")

def embed_with_retry(texts: List[str], model: str, tries: int = 5) -> List[List[float]]:
    last_err = None
    for attempt in range(tries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            last_err = e
            _backoff_sleep(attempt)
    raise RuntimeError(f"OpenAI embeddings failed after {tries} attempts: {last_err}")

# ---------- GitHub helpers ----------
import httpx

def gh_headers():
    tok = _sec("GITHUB_TOKEN")
    return {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def gh_repo_info():
    owner  = _sec("GH_REPO_OWNER")
    repo   = _sec("GH_REPO_NAME")
    branch = _sec("GH_BRANCH")
    return owner, repo, branch

def gh_put_file(path_rel: str, content_bytes: bytes, message: str) -> dict:
    owner, repo, branch = gh_repo_info()
    if not (owner and repo):
        raise RuntimeError("GH_REPO_OWNER or GH_REPO_NAME missing in secrets.")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=60.0) as c:
        r0 = c.get(url, params=params, headers=gh_headers())
        sha = r0.json().get("sha") if r0.status_code == 200 else None
        if r0.status_code not in (200, 404):
            r0.raise_for_status()
        payload = {
            "message": message,
            "content": base64.b64encode(content_bytes).decode("utf-8"),
        }
        if branch:
            payload["branch"] = branch
        if sha:
            payload["sha"] = sha
        r = c.put(url, headers=gh_headers(), data=json.dumps(payload))
        r.raise_for_status()
        return r.json()

def gh_list_any(path_rel: str) -> List[dict]:
    owner, repo, branch = gh_repo_info()
    if not (owner and repo):
        return []
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=30.0) as c:
        r = c.get(url, params=params, headers=gh_headers())
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

def gh_get_file(path_rel: str) -> bytes:
    owner, repo, branch = gh_repo_info()
    if not (owner and repo):
        raise RuntimeError("Missing GH repo settings")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=60.0) as c:
        r = c.get(url, params=params, headers=gh_headers())
        r.raise_for_status()
        data = r.json()
        if data.get("encoding") == "base64" and "content" in data:
            return base64.b64decode(data["content"])
        raise RuntimeError("Unexpected GitHub get-file response format")

# ---------- Text loaders & chunking ----------
from pypdf import PdfReader
from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from markdown import markdown
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            parts = []
            for p in reader.pages:
                try:
                    parts.append(p.extract_text() or "")
                except Exception:
                    parts.append("")
            return "\n".join(parts).strip()
        elif ext == ".docx":
            doc = DocxDocument(path)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        elif ext in (".md",".markdown"):
            html = markdown(open(path, "r", encoding="utf-8", errors="ignore").read())
            return BeautifulSoup(html, "html.parser").get_text("\n").strip()
        else:
            return open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read {os.path.basename(path)}: {e}")

def chunk_text(text: str, chunk_size=1000, overlap=150) -> List[str]:
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n","\n",". ","? ","! "," ",""]
    )
    pieces = [t.strip() for t in splitter.split_text(text)]
    return [p for p in pieces if p]

# ---------- Embeddings & retrieval ----------
def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    return embed_with_retry(texts, model=model)

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = 0.0; na = 0.0; nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x*x
        nb += y*y
    if na == 0 or nb == 0: return 0.0
    return dot / (math.sqrt(na)*math.sqrt(nb))

def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# ---------- Tavily search (with retry) ----------
def tavily_search(query: str,
                  max_results: int = 6,
                  include_domains: Optional[List[str]] = None,
                  exclude_domains: Optional[List[str]] = None,
                  search_depth: str = "advanced",
                  days: Optional[int] = None,
                  tries: int = 4) -> List[Dict[str, Any]]:
    api_key = _sec("TAVILY_API_KEY")
    if not api_key:
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": False,
        "include_raw_content": False,
        "source": "book-autogen"
    }
    if include_domains: payload["include_domains"] = include_domains
    if exclude_domains: payload["exclude_domains"] = exclude_domains
    if days is not None: payload["days"] = days

    last_err = None
    for attempt in range(tries):
        try:
            with httpx.Client(timeout=45.0) as c:
                r = c.post("https://api.tavily.com/search", json=payload)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                out = []
                for rr in results:
                    out.append({
                        "url": rr.get("url"),
                        "title": rr.get("title"),
                        "content": rr.get("content", ""),
                        "score": rr.get("score", 0.0)
                    })
                return out
        except Exception as e:
            last_err = e
            _backoff_sleep(attempt)
    # if Tavily fails, just return []
    return []

# ---------- State ----------
if "records" not in st.session_state:
    st.session_state.records = []  # [{id, text, tags, source, emb}]
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"
if "emb_model" not in st.session_state:
    st.session_state.emb_model = "text-embedding-3-large"
if "busy" not in st.session_state:
    st.session_state.busy = False

records: List[Dict[str,Any]] = st.session_state.records

# ---------- Sidebar Settings ----------
with st.sidebar:
    st.header("Settings")
    st.text("OpenAI: ✅")
    st.write("Has GITHUB_TOKEN:", bool(_sec("GITHUB_TOKEN")))
    owner, repo, branch = gh_repo_info()
    st.write("Repo:", f"{owner}/{repo}" if owner and repo else "❌")
    st.write("Branch:", branch or "(default)")
    st.divider()
    st.markdown("**Models**")
    st.session_state.model = st.text_input("Chat model", value=st.session_state.model, disabled=st.session_state.busy)
    st.session_state.emb_model = st.text_input("Embedding model", value=st.session_state.emb_model, disabled=st.session_state.busy)
    temperature = st.slider("Creativity", 0.0, 1.2, 0.6, 0.05, disabled=st.session_state.busy)
    top_k = st.slider("Top-K from books", 3, 30, 10, 1, disabled=st.session_state.busy)
    chunk_size = st.slider("Chunk size", 600, 2400, 1000, 100, disabled=st.session_state.busy)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 150, 10, disabled=st.session_state.busy)
    st.divider()
    with st.expander("🧪 GitHub quick tools", expanded=False):
        if st.button("List /uploads", disabled=st.session_state.busy):
            try:
                items = gh_list_any("uploads")
                files = [i for i in items if i.get("type") == "file"]
                if not files: st.info("No files in /uploads yet.")
                else:
                    for f in files: st.write("•", f.get("name"))
            except Exception as e:
                st.error(f"GitHub test failed: {e}")
        if st.button("Restore & ingest all files from GitHub /uploads", disabled=st.session_state.busy):
            try:
                items = gh_list_any("uploads")
                files = [i for i in items if i.get("type") == "file"]
                if not files:
                    st.info("No files found in /uploads.")
                else:
                    added = 0
                    for fmeta in files:
                        name = fmeta["name"]
                        data = gh_get_file(f"uploads/{name}")
                        tmp = os.path.join("data", f"restored_{name}")
                        with open(tmp, "wb") as out:
                            out.write(data)
                        try:
                            raw = load_text_from_file(tmp)
                            pieces = chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)
                            embs = embed_texts(pieces, model=st.session_state.emb_model)
                            for i,(txt,emb) in enumerate(zip(pieces,embs)):
                                records.append({
                                    "id": f"{sha16(name)}:{i}:{len(records)}",
                                    "text": txt,
                                    "tags": ["Books"],
                                    "source": name,
                                    "emb": emb,
                                })
                            added += len(embs)
                        except Exception as e:
                            st.error(f"Failed ingest {name}: {e}")
                    st.success(f"Restored and ingested {len(files)} file(s) → {added} chunks.")
            except Exception as e:
                st.error(f"Restore failed: {e}")

# ---------- Ingest (optional) ----------
st.subheader("1) (Optional) Upload source books or materials")
files = st.file_uploader("PDF/DOCX/MD/TXT — your books or any key sources",
                         type=["pdf","docx","md","markdown","txt"], accept_multiple_files=True, disabled=st.session_state.busy)
tags_str = st.text_input("Tags for these files (comma-separated)", value="Books", disabled=st.session_state.busy)
persist = st.checkbox("Save originals to GitHub (/uploads)", value=True, disabled=st.session_state.busy)

if st.button("Ingest selected files", disabled=st.session_state.busy):
    if not files:
        st.warning("No files selected.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] or ["Books"]
        added = 0
        for f in files:
            b = f.read()
            tmp = os.path.join("data", f"{int(time.time())}_{f.name}")
            with open(tmp, "wb") as out:
                out.write(b)
            if persist:
                try:
                    gh_put_file(f"uploads/{int(time.time())}_{f.name}", b, f"Add source {f.name}")
                except Exception as e:
                    st.error(f"GitHub upload failed for {f.name}: {e}")
            try:
                raw = load_text_from_file(tmp)
                pieces = chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)
                embs = embed_texts(pieces, model=st.session_state.emb_model)
                for i,(txt,emb) in enumerate(zip(pieces,embs)):
                    records.append({
                        "id": f"{sha16(f.name)}:{i}:{len(records)}",
                        "text": txt,
                        "tags": tags,
                        "source": f.name,
                        "emb": emb,
                    })
                added += len(embs)
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")
        st.success(f"Ingested: {len(files)} file(s) → {added} chunks.")
st.caption(f"Corpus size: {len(records)} chunks. (Zero is okay — model + web URLs only mode)")

# ---------- Retrieve ----------
def retrieve(query: str, k: int, tags: Optional[List[str]]=None) -> List[Dict[str,Any]]:
    if not records or not query.strip():
        return []
    qvecs = embed_texts([query], model=st.session_state.emb_model)
    if not qvecs:
        return []
    qvec = qvecs[0]
    scored = []
    tags_lower = set([t.lower() for t in (tags or [])])
    for r in records:
        if tags_lower and not (set([t.lower() for t in r["tags"]]) & tags_lower):
            continue
        s = cosine_sim(qvec, r["emb"])
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _,r in scored[:k]]

def make_context(hits: List[Dict[str,Any]]) -> str:
    out = []
    for i,h in enumerate(hits):
        out.append(f"[S{i+1} | {','.join(h.get('tags',[]))} | {h.get('source','')}] {h['text'][:1200]}")
    return "\n\n".join(out) if out else "(no retrieved context)"

# ---------- Outline controls ----------
st.subheader("2) Define the project")
col1, col2 = st.columns(2)
with col1:
    book_title = st.text_input("Book title", value="Working Title: The Essence of Modern Luxury", disabled=st.session_state.busy)
    thesis = st.text_area("Book thesis / goal (one paragraph)", height=120, disabled=st.session_state.busy)
with col2:
    style = st.text_area("Voice & Style Sheet (tense, pacing, terminology, audience)", height=120, disabled=st.session_state.busy)

st.subheader("3) Outline & generation settings")
outline_mode = st.radio("Outline source", ["Auto-generate", "Paste outline", "Upload outline file"],
                        index=0, horizontal=True, disabled=st.session_state.busy)
num_chapters = st.number_input("Total number of chapters (Intro is #1, Final Conclusion is last)",
                               8, 24, 12, 1, disabled=st.session_state.busy)
words_per_chapter = st.number_input("Target words per chapter", 1200, 10000, 3500, 100, disabled=st.session_state.busy)

st.markdown("**Web research** (optional)")
use_web = st.checkbox("Enable web research with manual URLs (below)", disabled=st.session_state.busy)
ref_urls = st.text_area("Manual Reference URLs (one per line)",
                        placeholder="https://example.com/report\nhttps://another.com/profile",
                        height=100, disabled=st.session_state.busy)
urls_list = [u.strip() for u in ref_urls.splitlines() if u.strip()] if use_web else []

st.markdown("**Web research guidance (Tavily)**")
research_guidance = st.text_area(
    "Tell the system how to search and what to prioritize (e.g., peer-reviewed studies since 2019; include hyperlinks to journals/DOIs; favor meta-analyses).",
    height=120, placeholder="Prioritize recent peer-reviewed research (2019+), systematic reviews, and meta-analyses. Add inline links to journals/DOIs. Include major industry reports if relevant.",
    disabled=st.session_state.busy
)
use_tavily = st.checkbox("Use Tavily web research automatically", value=True, disabled=st.session_state.busy)
max_sources_per_chapter = st.slider("Max Tavily sources per chapter", 2, 20, 6, 1, disabled=st.session_state.busy)
prefer_academic = st.checkbox("Prefer academic/journal sources", value=True, disabled=st.session_state.busy)
recent_year_cutoff = st.number_input("Prefer sources published ≥ this year (0 to ignore)", min_value=0, max_value=2100, value=2019, step=1, disabled=st.session_state.busy)
include_domains_txt = st.text_input("Include only these domains (comma-separated, optional)", value="", disabled=st.session_state.busy)
exclude_domains_txt = st.text_input("Exclude these domains (comma-separated, optional)", value="", disabled=st.session_state.busy)

# Paste outline
if outline_mode == "Paste outline":
    pasted_outline = st.text_area("Paste your chapter list (one per line; optional synopsis after a dash). You may include bullets under each chapter for section titles.",
                                  height=220, placeholder="Chapter 1: Introduction — why this book now\n- Why this topic now\n- Who this is for\nChapter 2: ...\nChapter 12: Final Conclusion — the path forward",
                                  disabled=st.session_state.busy)
else:
    pasted_outline = ""

# Upload outline file
if outline_mode == "Upload outline file":
    outline_file = st.file_uploader("Upload outline (TXT/MD/DOCX/PDF)", type=["txt","md","markdown","docx","pdf"],
                                    accept_multiple_files=False, disabled=st.session_state.busy)
else:
    outline_file = None

st.divider()
colA, colB = st.columns([1,1])
with colA:
    generate = st.button("🚀 One-Click: Generate Entire Book", type="primary", disabled=st.session_state.busy)
with colB:
    resume = st.button("▶️ Resume last run (from GitHub drafts)", disabled=st.session_state.busy)

# ---------- Helper prompts ----------
def plan_outline(context: str) -> str:
    prompt = f"""
You are a senior acquisitions editor. Propose a complete outline (chapters + brief synopses) for a 250–300 page book.
Title: {book_title}
Thesis: {thesis}
Voice & Style: {style}
Desired total chapters: {num_chapters}

RULES
- The outline MUST include an **Introduction** as Chapter 1 and a **Final Conclusion** as the last chapter.
- Total chapters MUST equal {num_chapters}.
- Each chapter gets a short, actionable synopsis (2–4 sentences).
- If context is minimal or absent, rely on general knowledge; keep a coherent arc.
- Output format:
Chapter 1: Introduction — 2–4 sentence synopsis
Chapter 2: Title — synopsis
...
Chapter {num_chapters}: Final Conclusion — synopsis

Context excerpts (may be empty):
{context}
""".strip()
    msgs = [
        {"role":"system","content":"You are an expert editor who creates commercially strong nonfiction outlines."},
        {"role":"user","content":prompt}
    ]
    return chat_with_retry(msgs, model=st.session_state.model, temperature=0.4, max_tokens=2000)

def parse_outline_lines(text: str) -> List[Dict[str,Any]]:
    chapters = []
    lines = text.replace("\r","\n").split("\n")
    current = None
    for line in lines:
        m = re.match(r"^\s*Chapter\s+(\d+)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
        mdh = re.match(r"^\s*#\s+(.+)$", line.strip())
        if m or mdh:
            if m:
                num = m.group(1).strip()
                tail = m.group(2).strip()
                title = tail
                synopsis = ""
                if "—" in tail:
                    parts = tail.split("—",1)
                    title = parts[0].strip()
                    synopsis = parts[1].strip()
                elif "-" in tail:
                    parts = tail.split("-",1)
                    title = parts[0].strip()
                    synopsis = parts[1].strip()
            else:
                num = str(len(chapters)+1)
                title = mdh.group(1).strip()
                synopsis = ""
            current = {"num":num, "title":title, "synopsis":synopsis, "sections":[]}
            chapters.append(current)
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        subh   = re.match(r"^\s*###\s+(.+)$", line)
        enum   = re.match(r"^\s*\d+[\.)]\s+(.+)$", line)
        if current and (bullet or subh or enum):
            sec_title = (bullet or subh or enum).group(1).strip()
            if sec_title:
                current["sections"].append(sec_title)
    return chapters

def chapter_section_plan(ch_num: str, ch_title: str, ch_synopsis: str, target_words: int, sections: int, context: str, forced_sections: Optional[List[str]]=None) -> str:
    if forced_sections:
        plan_lines = []
        fs = forced_sections[:]
        if not fs or fs[-1].strip().lower() != "conclusion":
            fs.append("Conclusion")
        for i, t in enumerate(fs, start=1):
            plan_lines.append(f"{i}. {t}")
        return "\n".join(plan_lines)

    prompt = f"""
Draft a section plan for Chapter {ch_num}: "{ch_title}" (~{target_words} words).
Synopsis: {ch_synopsis}
Create {sections} sections with short, specific titles (1 line each). Reserve the last section for "Conclusion" (exactly that word).
No prose, just the numbered plan.

Context (may be empty):
{context}
""".strip()
    msgs = [
        {"role":"system","content":"You are an expert nonfiction book outliner."},
        {"role":"user","content":prompt}
    ]
    return chat_with_retry(msgs, model=st.session_state.model, temperature=0.4, max_tokens=600)

def draft_section(si: int, sections_total: int, ch_num: str, ch_title: str, ch_synopsis: str, section_words: int, section_plan: str, context: str, urls: List[str], guidance: str, recent_cutoff: int) -> str:
    urls_block = "\n".join(f"- {u}" for u in urls) if urls else "(none)"
    guidance_text = guidance.strip() if guidance else ""
    recency_line = ""
    if isinstance(recent_cutoff, int) and recent_cutoff > 0:
        recency_line = f"Prefer sources published in {recent_cutoff} or later when citing web material."

    prompt = f"""
Write Section {si} (~{section_words} words) for Chapter {ch_num}: "{ch_title}".
Follow this approved section plan:
{section_plan}

Synopsis: {ch_synopsis}

Context from my books (may be empty; paraphrase freely; you may quote verbatim only with attribution):
{context}

Web references (cite ONLY if you actually use a fact; add inline Markdown links): 
{urls_block}

Additional research guidance (apply if web links are provided):
{guidance_text}
{recency_line}

MANDATORY RULES
- Write ONLY Section {si}. Do not draft other sections.
- Keep a single "## Conclusion" for the final section ONLY. If this is not the last section, do NOT write any conclusion section.
- If you include a verbatim quote from my books, attribute inline as:
  “quote” — Full Name, Role/Title at Hotel/Restaurant (Country)
- Paraphrased book content: no citation.
- Web facts/figures: add a Markdown hyperlink right where used.
- Use crisp headings (## ... ; ### for subheads).
- Maintain voice; avoid repetition; end with a segue to the next section.
""".strip()
    msgs = [
        {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
        {"role":"user","content":prompt}
    ]
    return chat_with_retry(msgs, model=st.session_state.model, temperature=0.6, max_tokens=1500)

def stitch_chapter(ch_num: str, ch_title: str, target_words: int, sections: List[str]) -> str:
    prompt = f"""
Combine the sections below into a cohesive chapter (~{target_words} words).
Smooth transitions, remove duplicates, normalize headings.

CRITICAL RULES
- Exactly ONE "## Conclusion" at the end. If earlier sections added any conclusion-like headings, merge their content into regular sections and remove the extra heading.
- Preserve all inline attributions for verbatim quotes from my books:
  “quote” — Full Name, Role/Title at Hotel/Restaurant (Country)
- Keep web hyperlinks where used; do not invent links; do not add citations to paraphrased book content. Preserve the exact URL text when merging.

Sections:
{"\n\n---\n\n".join(sections)}
""".strip()
    msgs = [
        {"role":"system","content":"You are a meticulous editor. Reply in Markdown."},
        {"role":"user","content":prompt}
    ]
    return chat_with_retry(msgs, model=st.session_state.model, temperature=0.3, max_tokens=3200)

def dedup_boundary(old_text: str, new_text: str, lookback: int=160) -> str:
    old = (old_text or "").strip()
    newt = (new_text or "").strip()
    if not old: return newt
    old_end = old[-lookback:].lower()
    new_start = newt[:lookback].lower()
    if old_end in new_start or new_start in old_end:
        for i in range(min(lookback, len(newt))):
            if old.endswith(newt[:i]):
                return newt[i:]
    return newt

# ---------- Markdown → Docx (Title page & TOC) ----------
from docx import Document as DocxDocument
def add_toc(paragraph):
    fld = OxmlElement('w:fldSimple')
    fld.set(qn('w:instr'), 'TOC \\o "1-3" \\h \\z \\u')
    r = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = "Table of Contents (open in Word > References > Update Table)"
    r.append(t)
    fld.append(r)
    paragraph._p.addnext(fld)

def md_to_docx(md_text: str, title: str, thesis: str, author: Optional[str] = None) -> bytes:
    doc = DocxDocument()
    h = doc.add_heading(title, level=0); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if author:
        a = doc.add_paragraph(author); a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_paragraph(f"Thesis: {thesis}"); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()
    p_toc = doc.add_paragraph(); add_toc(p_toc); doc.add_page_break()
    lines = md_text.replace("\r","\n").split("\n")
    for line in lines:
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        else:
            para = doc.add_paragraph()
            last = 0
            for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", line):
                before = line[last:m.start()]
                if before: para.add_run(before)
                text = m.group(1); url = m.group(2)
                run = para.add_run(text); run.font.underline = True
                para.add_run(f" ({url})")
                last = m.end()
            tail = line[last:]
            if tail: para.add_run(tail)
    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()

# ---------- Autosave / Resume ----------
def safe_title_slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", (title or "Book").strip()) or "Book"

def draft_dir(title: str, ts: int) -> str:
    return f"outputs/drafts/{safe_title_slug(title)}_{ts}"

def save_manifest(title: str, ts: int, manifest: Dict[str,Any]):
    path = f"{draft_dir(title, ts)}/manifest.json"
    gh_put_file(path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"), "Update manifest")

def save_chapter_draft(title: str, ts: int, ch_num: str, ch_title: str, text: str):
    name = f"ch_{int(ch_num):02d}_{safe_title_slug(ch_title)}.md"
    path = f"{draft_dir(title, ts)}/{name}"
    gh_put_file(path, text.encode("utf-8"), f"Add {name}")

def find_latest_manifest() -> Optional[Tuple[str,int,Dict[str,Any]]]:
    # Returns (dir_path, ts, manifest_json) for latest
    base = "outputs/drafts"
    items = gh_list_any(base)
    dirs = [i for i in items if i.get("type") == "dir"]
    latest = None; latest_ts = -1; latest_path = None
    for d in dirs:
        name = d["name"]  # e.g., Title_1700000000
        m = re.search(r"_(\d+)$", name)
        if not m: continue
        ts = int(m.group(1))
        if ts > latest_ts:
            latest_ts = ts
            latest_path = d["path"]
    if latest_path:
        try:
            bytes_man = gh_get_file(f"{latest_path}/manifest.json")
            j = json.loads(bytes_man.decode("utf-8"))
            return latest_path, latest_ts, j
        except Exception:
            return None
    return None

# ---------- Generation / Resume ----------
def build_outline_and_chapters(thesis_ctx: str, outline_mode: str, pasted_outline: str, outline_file) -> List[Dict[str,Any]]:
    # Returns chapters list
    if outline_mode == "Auto-generate":
        st.info("Generating outline…")
        outline_text = plan_outline(thesis_ctx)
        st.markdown("#### Proposed Outline")
        st.code(outline_text)
        parsed_chapters = parse_outline_lines(outline_text)
    elif outline_mode == "Paste outline":
        outline_text = pasted_outline or ""
        st.markdown("#### Using your pasted outline")
        st.code(outline_text)
        parsed_chapters = parse_outline_lines(outline_text)
    elif outline_mode == "Upload outline file":
        if not outline_file:
            st.error("Please upload an outline file or switch modes.")
            st.stop()
        bytes_io = outline_file.read()
        tmp = os.path.join("data", f"outline_{int(time.time())}_{outline_file.name}")
        with open(tmp, "wb") as f:
            f.write(bytes_io)
        try:
            text = load_text_from_file(tmp)
        except Exception as e:
            st.error(f"Could not read outline: {e}")
            st.stop()
        outline_text = text
        st.markdown("#### Parsed outline source (raw)")
        st.code(outline_text[:4000] + ("..." if len(outline_text) > 4000 else ""))
        parsed_chapters = parse_outline_lines(outline_text)
    else:
        parsed_chapters = []

    def coerce_intro_conclusion(chaps: List[Dict[str,Any]], total: int) -> List[Dict[str,Any]]:
        arr = [c for c in chaps if c.get("title")]
        if not arr or "intro" not in arr[0]["title"].lower():
            arr.insert(0, {"num":"1","title":"Introduction","synopsis": arr[0]["synopsis"] if arr else "", "sections": []})
        if "conclusion" not in arr[-1]["title"].lower():
            arr.append({"num":str(len(arr)+1),"title":"Final Conclusion","synopsis":"","sections":[]})
        if len(arr) > total:
            middle_keep = total - 2
            arr = [arr[0]] + arr[1:1+middle_keep] + [arr[-1]]
        elif len(arr) < total:
            to_add = total - len(arr)
            for i in range(to_add):
                arr.insert(-1, {"num":"0","title":f"Chapter Placeholder {i+1}","synopsis":"","sections":[]})
        for i, c in enumerate(arr, start=1):
            c["num"] = str(i)
        return arr

    chapters = coerce_intro_conclusion(parsed_chapters, int(num_chapters))
    st.markdown("#### Outline Preview")
    for c in chapters:
        st.markdown(f"- **Chapter {c['num']}: {c['title']}** — {c.get('synopsis','')}")
        secs = c.get("sections") or []
        if secs:
            for s in secs:
                st.markdown(f"  - {s}")
    if not chapters:
        st.error("No chapters parsed; please adjust your outline or settings.")
        st.stop()
    return chapters

def wc(t: str) -> int:
    return len((t or "").split())

def generate_book(run_mode: str = "fresh", resume_payload: Optional[Dict[str,Any]] = None):
    st.session_state.busy = True
    try:
        # Prepare context
        thesis_hits = retrieve(thesis or book_title, k=top_k, tags=None) if records else []
        thesis_ctx = make_context(thesis_hits)

        # Outline
        if run_mode == "fresh":
            chapters = build_outline_and_chapters(thesis_ctx, outline_mode, pasted_outline, outline_file)
            ts = int(time.time())
            author = _sec("BOOK_AUTHOR") or ""
            manifest = {
                "title": book_title,
                "author": author,
                "thesis": thesis,
                "style": style,
                "num_chapters": int(num_chapters),
                "words_per_chapter": int(words_per_chapter),
                "chapters": chapters,
                "completed": [],
                "next_index": 0,
                "urls_mode": "manual+auto" if (use_tavily or urls_list) else "none",
                "use_tavily": bool(use_tavily),
                "research_guidance": research_guidance,
                "recent_year_cutoff": int(recent_year_cutoff),
                "prefer_academic": bool(prefer_academic),
                "include_domains": [d.strip() for d in include_domains_txt.split(",") if d.strip()],
                "exclude_domains": [d.strip() for d in exclude_domains_txt.split(",") if d.strip()],
                "top_k": int(top_k),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "temperature": float(temperature),
                "timestamp": ts
            }
            # Create draft dir by saving initial manifest
            gh_put_file(f"{draft_dir(book_title, ts)}/_header.txt", f"Draft start: {book_title} @ {ts}".encode("utf-8"), "Init draft folder")
            save_manifest(book_title, ts, manifest)
        else:
            # resume mode
            if not resume_payload:
                st.error("No resume payload provided.")
                return
            manifest = resume_payload["manifest"]
            ts = resume_payload["ts"]
            chapters = manifest["chapters"]
            author = manifest.get("author","")

        # UI progress
        progress = st.progress(0.0, text="Starting…")
        manuscript_parts = [f"# {book_title}\n\n*Author:* {manifest.get('author','')}\n\n*Thesis:* {manifest.get('thesis','')}\n"]
        prior_text = ""

        # Domain filters
        include_domains = manifest.get("include_domains") or [d.strip() for d in include_domains_txt.split(",") if d.strip()]
        exclude_domains = manifest.get("exclude_domains") or [d.strip() for d in exclude_domains_txt.split(",") if d.strip()]

        start_idx = manifest.get("next_index", 0)
        total_ch = len(chapters)

        for idx in range(start_idx, total_ch):
            c = chapters[idx]
            ch_num = c["num"]; ch_title = c["title"]; ch_syn = c.get("synopsis","")
            progress.progress(idx/total_ch, text=f"Drafting Chapter {ch_num}: {ch_title}")

            query = f"{book_title} | {ch_title} | {ch_syn} | continuity: {prior_text[-600:] if prior_text else ''}"
            hits = retrieve(query, k=manifest.get("top_k", top_k), tags=None) if records else []
            ctx = make_context(hits)

            manual_urls = urls_list[:] if urls_list else []
            tavily_urls = []
            if manifest.get("use_tavily") and _sec("TAVILY_API_KEY"):
                rq_parts = [f"Topic: {ch_title}", f"Synopsis: {ch_syn}", f"Book thesis: {manifest.get('thesis','')}", f"Style: {manifest.get('style','')}"]
                if (manifest.get("research_guidance") or "").strip():
                    rq_parts.append(f"Guidance: {manifest['research_guidance'].strip()}")
                research_query = " | ".join(rq_parts)
                days_window = None
                ryc = manifest.get("recent_year_cutoff", 0)
                if isinstance(ryc, int) and ryc > 0:
                    start = _dt.datetime(ryc, 1, 1)
                    days_window = max(1, (_dt.datetime.utcnow() - start).days)
                raw = tavily_search(
                    query=research_query,
                    max_results=max_sources_per_chapter,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    search_depth="advanced",
                    days=days_window
                )
                if manifest.get("prefer_academic") and raw:
                    def is_academic(u: str) -> bool:
                        u = (u or "").lower()
                        return any(x in u for x in [".edu", ".ac.", "nature.com", "science.org", "sciencedirect.com", "jstor.org", "springer", "wiley.com", "tandfonline.com", "cell.com", "nejm.org", "thelancet.com", "doi.org"])
                    raw_sorted = sorted(raw, key=lambda r: (not is_academic(r.get("url","")), -float(r.get("score",0.0))))
                else:
                    raw_sorted = sorted(raw, key=lambda r: -float(r.get("score",0.0)))
                tavily_urls = [r["url"] for r in raw_sorted if r.get("url")]

            effective_urls = []
            seen = set()
            for u in (manual_urls + tavily_urls):
                if u not in seen:
                    effective_urls.append(u); seen.add(u)

            forced_secs = c.get("sections") if isinstance(c.get("sections"), list) and c.get("sections") else None
            sections_target = max(5, min(8, int(round(manifest.get("words_per_chapter", words_per_chapter)/600))))
            plan_text = chapter_section_plan(ch_num, ch_title, ch_syn, manifest.get("words_per_chapter", words_per_chapter), sections_target, ctx, forced_sections=forced_secs)

            sec_texts = []
            per_sec = max(450, min(900, int(manifest.get("words_per_chapter", words_per_chapter)/sections_target)))
            total_sections_to_write = len(forced_secs) if forced_secs else sections_target
            for si in range(1, total_sections_to_write+1):
                s_txt = draft_section(si, total_sections_to_write, ch_num, ch_title, ch_syn, per_sec, plan_text, ctx, effective_urls, manifest.get("research_guidance",""), manifest.get("recent_year_cutoff",0))
                if sec_texts:
                    s_txt = dedup_boundary(sec_texts[-1], s_txt, lookback=160)
                sec_texts.append(s_txt)

            ch_text = stitch_chapter(ch_num, ch_title, manifest.get("words_per_chapter", words_per_chapter), sec_texts)

            passes=0
            while wc(ch_text) < int(manifest.get("words_per_chapter", words_per_chapter)*0.95) and passes < 2:
                cont_prompt = f"""Continue Chapter {ch_num}: "{ch_title}" seamlessly from where it stops.
Do NOT repeat prior text. Keep headings; add 1–2 subsections if appropriate.
Aim for ~{per_sec}–{per_sec+200} words. Current ending:
{ch_text[-900:]}"""
                msgs = [
                    {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
                    {"role":"user","content":cont_prompt}
                ]
                cont = chat_with_retry(msgs, model=st.session_state.model, temperature=0.6, max_tokens=1500)
                ch_text += "\n\n" + dedup_boundary(ch_text, cont, lookback=180)
                passes += 1

            # Autosave chapter draft
            save_chapter_draft(manifest["title"], manifest["timestamp"], ch_num, ch_title, ch_text)

            # Update manifest
            manifest["completed"] = sorted(list(set(manifest.get("completed",[]) + [int(ch_num)])))
            manifest["next_index"] = idx + 1
            save_manifest(manifest["title"], manifest["timestamp"], manifest)

            manuscript_parts.append(f"\n\n# Chapter {ch_num}: {ch_title}\n\n{ch_text}\n")
            prior_text += f"\n\n[END CH {ch_num}]\n{ch_text}\n"

        progress.progress(1.0, text="Combining…")
        manuscript = "\n".join(manuscript_parts).strip()

        # Final save & downloads
        ts_final = int(time.time())
        safe_title = safe_title_slug(manifest["title"])
        md_name = f"{safe_title}_{ts_final}.md"
        docx_name = f"{safe_title}_{ts_final}.docx"

        st.download_button("⬇️ Download manuscript (Markdown)", manuscript, file_name=md_name)

        try:
            docx_bytes = md_to_docx(manuscript, manifest["title"], manifest.get("thesis",""), author=manifest.get("author",""))
            st.download_button("⬇️ Download manuscript (Word .docx)", docx_bytes, file_name=docx_name)
        except Exception as e:
            st.warning(f"Could not generate .docx: {e}")
            docx_bytes = None

        try:
            res = gh_put_file(f"outputs/{md_name}", manuscript.encode("utf-8"), f"Add manuscript {md_name}")
            owner, repo, branch = gh_repo_info()
            link_branch = branch or "main"
            gh_link_md = f"https://github.com/{owner}/{repo}/blob/{link_branch}/{res.get('content',{}).get('path', f'outputs/{md_name}')}"
            st.success(f"Saved Markdown to GitHub: {gh_link_md}")
        except Exception as e:
            st.warning(f"Could not save Markdown to GitHub automatically: {e}")

        try:
            if docx_bytes:
                res2 = gh_put_file(f"outputs/{docx_name}", docx_bytes, f"Add manuscript {docx_name}")
                owner, repo, branch = gh_repo_info()
                link_branch = branch or "main"
                gh_link_docx = f"https://github.com/{owner}/{repo}/blob/{link_branch}/{res2.get('content',{}).get('path', f'outputs/{docx_name}')}"
                st.success(f"Saved Word to GitHub: {gh_link_docx}")
        except Exception as e:
            st.warning(f"Could not save Word to GitHub automatically: {e}")

        st.success("✅ Book generation complete.")
    except Exception as e:
        st.error(f"Generation stopped with error: {e}")
    finally:
        st.session_state.busy = False

# ---------- Run buttons ----------
if generate:
    generate_book(run_mode="fresh", resume_payload=None)

if resume:
    try:
        found = find_latest_manifest()
        if not found:
            st.warning("No prior manifest found in outputs/drafts/. Start a fresh run first.")
        else:
            path, ts, manifest = found
            st.info(f"Resuming: {manifest.get('title','(unknown)')} — next chapter index {manifest.get('next_index',0)}")
            generate_book(run_mode="resume", resume_payload={"ts": ts, "manifest": manifest})
    except Exception as e:
        st.error(f"Resume failed: {e}")
