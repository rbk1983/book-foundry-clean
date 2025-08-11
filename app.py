
import os, sys, time, math, hashlib, base64, json, re, datetime as dt
from typing import List, Dict, Any, Optional

import streamlit as st

# ------------------ Page ------------------
st.set_page_config(page_title="Book Auto-Generator (v3)", layout="wide")

# ------------------ Diagnostics ------------------
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

os.makedirs("data", exist_ok=True)

# ------------------ Secrets helper ------------------
def _sec(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

# ------------------ OpenAI client ------------------
from openai import OpenAI
OPENAI_KEY = _sec("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)

# ------------------ GitHub helpers ------------------
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

def gh_list_dir(path_rel: str) -> List[dict]:
    owner, repo, branch = gh_repo_info()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=30.0) as c:
        r = c.get(url, params=params, headers=gh_headers())
        if r.status_code == 404:
            return []
        r.raise_for_status()
        items = r.json()
        return [i for i in items if i.get("type") == "file"]

# ------------------ Tavily web search helper ------------------
def tavily_search(query: str,
                  max_results: int = 6,
                  include_domains: Optional[List[str]] = None,
                  exclude_domains: Optional[List[str]] = None,
                  search_depth: str = "advanced",
                  days: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Calls Tavily /search API. Returns list of dicts with keys: url, title, content, score (when available).
    """
    api_key = _sec("TAVILY_API_KEY")
    if not api_key:
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,  # "basic" or "advanced"
        "include_answer": False,
        "include_raw_content": False,
        "source": "book-autogen"
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    if days is not None:
        payload["days"] = days

    try:
        with httpx.Client(timeout=40.0) as c:
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
    except Exception:
        return []

# ------------------ Text loaders & chunking ------------------
from pypdf import PdfReader
from docx import Document as DocxDocument
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
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

# ------------------ Embeddings & retrieval ------------------
def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    resp = client.embeddings.create(model=st.session_state.emb_model, input=texts)
    return [d.embedding for d in resp.data]

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

# ------------------ State ------------------
if "records" not in st.session_state:
    st.session_state.records = []  # [{id, text, tags, source, emb}]
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"
if "emb_model" not in st.session_state:
    st.session_state.emb_model = "text-embedding-3-large"

records: List[Dict[str,Any]] = st.session_state.records

# ------------------ Sidebar Settings ------------------
with st.sidebar:
    st.header("Settings")
    st.text("OpenAI: ✅")
    st.write("Has GITHUB_TOKEN:", bool(_sec("GITHUB_TOKEN")))
    owner, repo, branch = gh_repo_info()
    st.write("Repo:", f"{owner}/{repo}" if owner and repo else "❌")
    st.write("Branch:", branch or "(default)")
    st.write("Tavily key:", "✅" if _sec("TAVILY_API_KEY") else "—")
    st.divider()
    st.markdown("**Models**")
    st.session_state.model = st.text_input("Chat model", value=st.session_state.model)
    st.session_state.emb_model = st.text_input("Embedding model", value=st.session_state.emb_model)
    temperature = st.slider("Creativity", 0.0, 1.2, 0.5, 0.05)
    top_k = st.slider("Top-K from books", 3, 30, 10, 1)
    chunk_size = st.slider("Chunk size", 600, 2400, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 150, 10)
    st.divider()
    with st.expander("🧪 GitHub quick test", expanded=False):
        if st.button("List /uploads"):
            try:
                files = gh_list_dir("uploads")
                if not files: st.info("No files in /uploads yet.")
                else:
                    for f in files: st.write("•", f.get("name"))
            except Exception as e:
                st.error(f"GitHub test failed: {e}")

# ------------------ Ingest (optional) ------------------
st.subheader("1) (Optional) Upload source books or materials")
files = st.file_uploader("PDF/DOCX/MD/TXT — your books or any key sources", type=["pdf","docx","md","markdown","txt"], accept_multiple_files=True)
tags_str = st.text_input("Tags for these files (comma-separated)", value="Books")
persist = st.checkbox("Save originals to GitHub (/uploads)", value=True)

if st.button("Ingest selected files"):
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

# ------------------ Retrieve ------------------
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

# ------------------ Outline controls ------------------
st.subheader("2) Define the project")
col1, col2 = st.columns(2)
with col1:
    book_title = st.text_input("Book title", value="Working Title: The Essence of Modern Luxury")
    thesis = st.text_area("Book thesis / goal (one paragraph)", height=120)
with col2:
    style = st.text_area("Voice & Style Sheet (tense, pacing, terminology, audience)", height=120)

st.subheader("3) Outline & generation settings")
outline_mode = st.radio("Outline source", ["Auto-generate", "I will paste an outline"], index=0, horizontal=True)
num_chapters = st.number_input("Number of chapters (auto mode)", 8, 20, 12, 1)
words_per_chapter = st.number_input("Target words per chapter", 1200, 8000, 3500, 100)

st.markdown("**Web research** (optional)")
use_web = st.checkbox("Enable manual web references (paste URLs below)", value=False)
ref_urls = st.text_area("Reference URLs (one per line)", placeholder="https://example.com/report\nhttps://another.com/profile", height=100)
urls_list = [u.strip() for u in ref_urls.splitlines() if u.strip()] if use_web else []

st.markdown("**Web research guidance (Tavily auto-search, optional)**")
research_guidance = st.text_area(
    "Tell the system how to search and what to prioritize (e.g., peer‑reviewed since 2019, meta-analyses, add journal/DOI links, include leading industry reports).",
    height=120,
    placeholder="Prioritize recent peer‑reviewed research (2019+), systematic reviews, and meta‑analyses. Add inline links to journals or DOIs. Include major industry reports if relevant."
)
use_tavily = st.checkbox("Use Tavily web research automatically", value=True)
max_sources_per_chapter = st.slider("Max Tavily sources per chapter", 2, 12, 6, 1)
prefer_academic = st.checkbox("Prefer academic/journal sources", value=True)
recent_year_cutoff = st.number_input("Prefer sources published ≥ this year (0 to ignore)", min_value=0, max_value=2100, value=2019, step=1)
include_domains_txt = st.text_input("Include only these domains (comma‑separated, optional)", value="")
exclude_domains_txt = st.text_input("Exclude these domains (comma‑separated, optional)", value="")

if outline_mode == "I will paste an outline":
    pasted_outline = st.text_area("Paste your chapter list (one per line; optional synopsis after a dash):", height=160, placeholder="Chapter 1: Title — optional synopsis\nChapter 2: ...")
else:
    pasted_outline = ""

st.divider()
generate = st.button("🚀 One-Click: Generate Entire Book", type="primary")

# ------------------ Helper prompts ------------------
def plan_outline(context: str) -> str:
    prompt = f"""
You are a senior acquisitions editor. Propose a complete outline (chapters + brief synopses) for a 250–300 page book.
Title: {book_title}
Thesis: {thesis}
Voice & Style: {style}
Desired chapter count: {num_chapters}

RULES
- Use 10–14 chapters, coherent arc from premise → development → application → conclusion.
- Each chapter gets a short, actionable synopsis (2–4 sentences).
- If context from my books is minimal or absent, rely on your general knowledge; fill gaps with sensible structure.
- Output format:
Chapter 1: Title — 2–4 sentence synopsis
Chapter 2: Title — synopsis
...
Chapter N: Title — synopsis

Context excerpts (may be empty):
{context}
""".strip()
    msgs = [
        {"role":"system","content":"You are an expert editor who creates commercially strong nonfiction outlines."},
        {"role":"user","content":prompt}
    ]
    resp = client.chat.completions.create(model=st.session_state.model, temperature=0.4, messages=msgs, max_tokens=1800)
    return resp.choices[0].message.content

def parse_outline(text: str) -> List[Dict[str,str]]:
    chapters = []
    for line in text.replace("\r","\n").split("\n"):
        m = re.match(r"^\s*Chapter\s+(\d+)\s*:\s*(.+)$", line.strip(), re.IGNORECASE)
        if not m: 
            continue
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
        chapters.append({"num":num, "title":title, "synopsis":synopsis})
    return chapters

def chapter_section_plan(ch_num: str, ch_title: str, ch_synopsis: str, target_words: int, sections: int, context: str) -> str:
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
    resp = client.chat.completions.create(model=st.session_state.model, temperature=0.4, messages=msgs, max_tokens=600)
    return resp.choices[0].message.content

def draft_section(si: int, sections_total: int, ch_num: str, ch_title: str, ch_synopsis: str, section_words: int, section_plan: str, context: str, urls: List[str], guidance: str, year_cutoff: int) -> str:
    urls_block = "\n".join(f"- {u}" for u in urls) if urls else "(none)"
    guidance_text = guidance.strip() if guidance else ""
    recency_line = ""
    if isinstance(year_cutoff, int) and year_cutoff > 0:
        recency_line = f"Prefer sources published in {year_cutoff} or later when citing web material."

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
    resp = client.chat.completions.create(model=st.session_state.model, temperature=0.6, messages=msgs, max_tokens=1500)
    return resp.choices[0].message.content

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
    resp = client.chat.completions.create(model=st.session_state.model, temperature=0.3, messages=msgs, max_tokens=3200)
    return resp.choices[0].message.content

def dedup_boundary(old_text: str, new_text: str, lookback: int=120) -> str:
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

# ----------- Markdown → Docx (basic) -----------
def add_hyperlink_text(para, text, url):
    run = para.add_run(text)
    run.font.underline = True
    para.add_run(f" ({url})")

def md_to_docx(md_text: str, title: str, thesis: str) -> bytes:
    doc = DocxDocument()
    # Title page
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("\n")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_paragraph(f"Thesis: {thesis}")
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

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
            pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
            last = 0
            for m in pattern.finditer(line):
                before = line[last:m.start()]
                if before:
                    para.add_run(before)
                text = m.group(1)
                url = m.group(2)
                add_hyperlink_text(para, text, url)
                last = m.end()
            tail = line[last:]
            if tail:
                para.add_run(tail)
    tmp = f"/mnt/data/_tmp_{int(time.time())}.docx"
    doc.save(tmp)
    with open(tmp, "rb") as f:
        return f.read()

# ------------------ Generation ------------------
if generate:
    # Prepare overall context from top-k chunks relevant to thesis (if any)
    thesis_hits = retrieve(thesis or book_title, k=top_k, tags=None) if records else []
    thesis_ctx = make_context(thesis_hits)

    # 1) Outline
    if outline_mode == "Auto-generate":
        st.info("Generating outline…")
        outline_text = plan_outline(thesis_ctx)
        st.markdown("#### Proposed Outline")
        st.code(outline_text)
    else:
        outline_text = pasted_outline
        st.markdown("#### Using your pasted outline")
        st.code(outline_text)

    # Parse outline
    chapters = parse_outline(outline_text)
    if not chapters:
        st.error("Could not parse any chapters from the outline. Make sure lines start with 'Chapter N:'")
        st.stop()

    # 2) Draft chapters sequentially
    progress = st.progress(0.0, text="Starting…")
    manuscript_parts = [f"# {book_title}\n\n*Thesis:* {thesis}\n"]
    prior_text = ""

    for idx, ch in enumerate(chapters):
        ch_num = ch["num"]; ch_title = ch["title"]; ch_syn = ch.get("synopsis","")
        progress.progress(idx/len(chapters), text=f"Drafting Chapter {ch_num}: {ch_title}")

        # Retrieval for this chapter if we have records
        query = f"{book_title} | {ch_title} | {ch_syn} | continuity: {prior_text[-600:] if prior_text else ''}"
        hits = retrieve(query, k=top_k, tags=None) if records else []
        ctx = make_context(hits)

        # ----- Chapter-specific web sources -----
        manual_urls = urls_list[:] if urls_list else []

        tavily_urls = []
        if use_tavily and _sec("TAVILY_API_KEY"):
            rq_parts = [
                f"Topic: {ch_title}",
                f"Synopsis: {ch_syn}",
                f"Book thesis: {thesis}",
            ]
            if research_guidance.strip():
                rq_parts.append(f"Guidance: {research_guidance.strip()}")
            research_query = " | ".join(rq_parts)

            include_domains = [d.strip() for d in include_domains_txt.split(",") if d.strip()] if include_domains_txt else None
            exclude_domains = [d.strip() for d in exclude_domains_txt.split(",") if d.strip()] if exclude_domains_txt else None

            days_window = None
            if isinstance(recent_year_cutoff, int) and recent_year_cutoff > 0:
                start = dt.datetime(recent_year_cutoff, 1, 1)
                days_window = max(1, (dt.datetime.utcnow() - start).days)

            raw = tavily_search(
                query=research_query,
                max_results=max_sources_per_chapter,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                search_depth="advanced",
                days=days_window
            )

            if prefer_academic and raw:
                def is_academic(u: str) -> bool:
                    u = (u or "").lower()
                    return any(x in u for x in [".edu", ".ac.", "nature.com", "science.org", "sciencedirect.com", "jstor.org", "springer", "wiley.com", "tandfonline.com", "cell.com", "nejm.org", "thelancet.com", "doi.org"])
                raw_sorted = sorted(raw, key=lambda r: (not is_academic(r.get("url","")), -float(r.get("score",0.0))))
            else:
                raw_sorted = sorted(raw, key=lambda r: -float(r.get("score",0.0)))

            tavily_urls = [r["url"] for r in raw_sorted if r.get("url")]

        # Merge and dedupe URLs
        effective_urls = []
        seen = set()
        for u in (manual_urls + tavily_urls):
            if u not in seen:
                effective_urls.append(u)
                seen.add(u)

        # Section plan
        sections_target = max(5, min(8, int(round(words_per_chapter/600))))
        plan_text = chapter_section_plan(ch_num, ch_title, ch_syn, words_per_chapter, sections_target, ctx)

        # Draft each section
        sec_texts = []
        per_sec = max(450, min(900, int(words_per_chapter/sections_target)))
        for si in range(1, sections_target+1):
            s_txt = draft_section(si, sections_target, ch_num, ch_title, ch_syn, per_sec, plan_text, ctx, effective_urls, research_guidance, recent_year_cutoff)
            if sec_texts:
                s_txt = dedup_boundary(sec_texts[-1], s_txt, lookback=160)
            sec_texts.append(s_txt)

        # Stitch
        ch_text = stitch_chapter(ch_num, ch_title, words_per_chapter, sec_texts)

        # Auto-continue pass if short (<95% target), up to 2 times
        def wc(t): return len((t or "").split())
        passes=0
        while wc(ch_text) < int(words_per_chapter*0.95) and passes < 2:
            cont_prompt = f"""Continue Chapter {ch_num}: "{ch_title}" seamlessly from where it stops.
Do NOT repeat prior text. Keep headings; add 1–2 subsections if appropriate.
Aim for ~{per_sec}–{per_sec+200} words. Current ending:
{ch_text[-900:]}"""
            msgs = [
                {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
                {"role":"user","content":cont_prompt}
            ]
            cont = client.chat.completions.create(model=st.session_state.model, temperature=0.6, messages=msgs, max_tokens=1500).choices[0].message.content
            ch_text += "\n\n" + dedup_boundary(ch_text, cont, lookback=180)
            passes += 1

        manuscript_parts.append(f"\n\n# Chapter {ch_num}: {ch_title}\n\n{ch_text}\n")
        prior_text += f"\n\n[END CH {ch_num}]\n{ch_text}\n"

    progress.progress(1.0, text="Combining…")
    manuscript = "\n".join(manuscript_parts).strip()

    # 3) Save to GitHub and offer downloads
    ts = int(time.time())
    safe_title = re.sub(r"[^A-Za-z0-9_\-]+","_", book_title.strip()) or "Book"
    md_name = f"{safe_title}_{ts}.md"
    docx_name = f"{safe_title}_{ts}.docx"

    # Download buttons
    st.download_button("⬇️ Download manuscript (Markdown)", manuscript, file_name=md_name)

    # Build a .docx version
    try:
        docx_bytes = md_to_docx(manuscript, book_title, thesis)
        st.download_button("⬇️ Download manuscript (Word .docx)", docx_bytes, file_name=docx_name)
    except Exception as e:
        st.warning(f"Could not generate .docx: {e}")
        docx_bytes = None

    # GitHub save
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
