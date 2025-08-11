
import os, sys, time, math, hashlib, base64, json, re
from typing import List, Dict, Any

import streamlit as st

# ==================== Page + Diagnostics ====================
st.set_page_config(page_title="Rahim's AI Book Studio", layout="wide")

st.sidebar.markdown("### 🔎 Diagnostics")
st.sidebar.write("Python:", sys.version)

try:
    import openai as _oa
    st.sidebar.write("openai pkg:", getattr(_oa, "__version__", "unknown"))
except Exception as e:
    st.sidebar.write("openai import error:", e)

try:
    import httpx as _hx
    st.sidebar.write("httpx:", getattr(_hx, "__version__", "unknown"))
except Exception as e:
    st.sidebar.write("httpx import error:", e)

# Ensure working data directory exists
os.makedirs("data", exist_ok=True)

# ==================== Secrets helper ====================
def _sec(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

# ==================== OpenAI client ====================
from openai import OpenAI

OPENAI_KEY = _sec("OPENAI_API_KEY")
st.sidebar.write("Has OPENAI_API_KEY:", bool(OPENAI_KEY))
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is missing. Add it in App → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_KEY)

# ==================== GitHub helpers ====================
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
        raise RuntimeError("GH_REPO_OWNER or GH_REPO_NAME missing in Secrets.")
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
            r_repo = c.get(f"https://api.github.com/repos/{owner}/{repo}", headers=gh_headers())
            if r_repo.status_code != 200:
                raise RuntimeError(f"Repo {owner}/{repo} not accessible (status {r_repo.status_code}).")
            return []
        r.raise_for_status()
        items = r.json()
        return [i for i in items if i.get("type") == "file"]

def gh_get_file(path_rel: str) -> bytes:
    owner, repo, branch = gh_repo_info()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=60.0) as c:
        r = c.get(url, params=params, headers=gh_headers())
        r.raise_for_status()
        data = r.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"])
        if "download_url" in data:
            r2 = c.get(data["download_url"])
            r2.raise_for_status()
            return r2.content
        return b""

# ==================== Loaders, chunking, embeddings ====================
from pypdf import PdfReader
from docx import Document as DocxDocument
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
        elif ext in [".md", ".markdown"]:
            html = markdown(open(path, "r", encoding="utf-8", errors="ignore").read())
            return BeautifulSoup(html, "html.parser").get_text("\n").strip()
        else:
            return open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read {os.path.basename(path)}: {e}")

def chunk_text(text: str, chunk_size=1200, overlap=200) -> List[str]:
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    pieces = [t.strip() for t in splitter.split_text(text)]
    return [p for p in pieces if p]

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def cosine_sim(a: List[float], b: List[float]) -> float:
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

# ==================== Mapping parser (for prefill helper) ====================
def load_mapping_files_from_data():
    files = []
    if os.path.exists("data"):
        for fname in os.listdir("data"):
            if fname.lower().startswith("mapping_") and fname.lower().endswith(".md"):
                files.append((fname, os.path.join("data", fname)))
    return sorted(files)

def parse_mapping_markdown(md_text: str):
    chapters = []
    current_ch = None
    current_sc = None

    lines = md_text.replace("\r\n","\n").replace("\r","\n").split("\n")
    h1 = re.compile(r"^#\s*Chapter\s+(\d+)\s*:\s*(.+)\s*$", re.IGNORECASE)
    h2 = re.compile(r"^##\s*Subchapter\s+([\d\.]+)\s*:\s*(.+)\s*$", re.IGNORECASE)

    for raw in lines:
        line = raw.strip()

        m = h1.match(line)
        if m:
            current_ch = {
                "num": m.group(1),
                "title": m.group(2).strip(),
                "subchapters": [],
                "bullets": []
            }
            chapters.append(current_ch)
            current_sc = None
            continue

        m = h2.match(line)
        if m and current_ch:
            current_sc = {
                "num": m.group(1),
                "title": m.group(2).strip(),
                "bullets": []
            }
            current_ch["subchapters"].append(current_sc)
            continue

        if line.startswith("-") and (current_ch is not None):
            m1 = re.match(r"^\s*-\s*\[Source:\s*([^\]]+)\]\s*(.*)$", line)
            src=None; excerpt=None; why=None
            if m1:
                src = m1.group(1).strip()
                tail = m1.group(2).strip()
                m2 = re.search(r"“([^”]+)”", tail) or re.search(r"\"([^\"]+)\"", tail)
                if m2:
                    excerpt = m2.group(1).strip()
                m3 = re.search(r"—\s*(.+)$", tail) or re.search(r"-\s*(.+)$", tail)
                if m3:
                    why = m3.group(1).strip()
            entry = {"source": src, "excerpt": excerpt, "why": why}
            if current_sc is not None:
                current_sc["bullets"].append(entry)
            else:
                current_ch["bullets"].append(entry)

    return {"chapters": chapters}

# ==================== Session state ====================
if "plan" not in st.session_state:
    st.session_state.plan = {
        "title": "My Third Book",
        "thesis": "",
        "style": "",
        "chapters": {}
    }

if "corpus" not in st.session_state:
    st.session_state.corpus = []

plan = st.session_state.plan
corpus: List[Dict[str, Any]] = st.session_state.corpus

# ==================== Sidebar settings ====================
st.title("Rahim's AI Book Studio")
st.caption("Use your previous books + uploads to outline and draft a new 250–300 page book.")

with st.sidebar:
    st.header("Settings")
    plan["title"] = st.text_input("Project name", value=plan["title"]).strip()
    chat_model = st.text_input("Chat model", value="gpt-4o")
    embed_model = st.text_input("Embedding model", value="text-embedding-3-large")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.5, 0.1)
    top_k = st.slider("Top-K retrieved chunks", 1, 30, 8, 1)
    chunk_size = st.slider("Chunk size", 600, 2400, 1000, 100)
    overlap = st.slider("Chunk overlap", 50, 400, 150, 10)
    tag_filter = st.text_input("Tag filter (comma-sep, optional)", value="")
    persist_to_github = st.checkbox("Persist uploads to GitHub", value=True)
    st.caption("Files ≤100 MB saved under /uploads/ in your storage repo.")

    st.divider()
    st.markdown("### 🧪 GitHub connection test")
    try:
        owner, repo, branch = gh_repo_info()
        token_present = bool(_sec("GITHUB_TOKEN"))
        st.write("Owner:", owner or "❌")
        st.write("Repo:", repo or "❌")
        st.write("Branch:", branch or "(repo default)")
        st.write("Has token:", token_present)
        if st.button("List /uploads in repo"):
            files = gh_list_dir("uploads")
            if not files:
                st.info("No files in /uploads (yet).")
            else:
                for f in files:
                    st.write("•", f.get("name"))
    except Exception as e:
        st.error(f"GitHub test failed: {e}")

    st.markdown("#### Token sanity checks")
    if st.button("Test /user and repo access"):
        try:
            with httpx.Client(timeout=30.0, headers=gh_headers()) as c:
                u = c.get("https://api.github.com/user")
                st.write("/user status:", u.status_code)
                if u.status_code == 200:
                    st.write("Authenticated as:", u.json().get("login"))
                else:
                    st.error(f"/user error: {u.text[:200]}")
                r = c.get(f"https://api.github.com/repos/{owner}/{repo}")
                st.write("/repos status:", r.status_code)
                if r.status_code == 200:
                    st.success("Repo is accessible ✅")
                else:
                    st.error(f"/repos error: {r.text[:200]}")
        except Exception as e:
            st.error(f"Token test failed: {e}")

    st.markdown("#### Write test file")
    if st.button("Create test file in /uploads"):
        try:
            ts = int(time.time())
            path_rel = f"uploads/test_{ts}.txt"
            res = gh_put_file(path_rel, f"hello from streamlit at {ts}\n".encode("utf-8"),
                              f"Add test file {ts}")
            link_branch = branch or "main"
            link = f"https://github.com/{owner}/{repo}/blob/{link_branch}/{res.get('content',{}).get('path', path_rel)}"
            st.success("Wrote: " + link)
        except httpx.HTTPStatusError as e:
            st.error(f"Write failed: {e}\n\n{e.response.text[:500]}")
        except Exception as e:
            st.error(f"Write failed (generic): {e}")

    st.divider()
    st.markdown("### ✍️ Longform drafting")
    longform = st.checkbox("Longform mode (section-by-section)", value=True)
    sections_target = st.slider("Sections to draft", 3, 10, 6, 1)
    section_words = st.slider("Words per section (~)", 400, 900, 600, 50)

# ==================== Tabs ====================
tab_sources, tab_outline, tab_draft, tab_export = st.tabs(
    ["📥 Sources", "🧭 Outline", "✍️ Draft", "📤 Export"]
)

# ==================== Utility to add records ====================
def add_records(texts: List[str], tags: List[str], source: str, model: str, added_counter: Dict[str, int] = None):
    clean_texts = [t for t in texts if t and t.strip()]
    if not clean_texts:
        return 0
    embs = embed_texts(clean_texts, model=model)
    count = 0
    for i, (text, emb) in enumerate(zip(clean_texts, embs)):
        corpus.append({
            "id": f"{sha16(source)}:{i}:{len(corpus)}",
            "text": text,
            "tags": tags,
            "source": source,
            "emb": emb,
        })
        count += 1
    if added_counter is not None:
        added_counter["n"] = added_counter.get("n", 0) + count
    return count

# ==================== SOURCES ====================
with tab_sources:
    st.subheader("Upload sources (PDF/DOCX/MD/TXT)")
    files = st.file_uploader("Select files", type=["pdf","docx","md","markdown","txt"], accept_multiple_files=True)
    tags_str = st.text_input("Tags for these files (e.g., ChefsBook, HoteliersBook, Outline)")

    if st.button("Ingest files", type="primary"):
        if not files:
            st.warning("No files selected.")
        else:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] or ["General"]
            added = {"n": 0}
            uploaded_to_gh = []

            for f in files:
                raw_bytes = f.read()
                size_mb = len(raw_bytes) / (1024 * 1024)

                if persist_to_github:
                    if size_mb > 95:
                        st.error(f"‘{f.name}’ is {size_mb:.1f} MB (>100 MB GitHub limit). Use S3 for this one.")
                        continue
                    ts = int(time.time())
                    gh_rel = f"uploads/{ts}_{f.name}"
                    try:
                        res = gh_put_file(gh_rel, raw_bytes, f"Add upload {f.name}")
                        uploaded_to_gh.append(res.get("content", {}).get("path", gh_rel))
                    except httpx.HTTPStatusError as e:
                        st.error(f"GitHub upload failed for {f.name}: {e}\n\n{e.response.text[:500]}")
                        continue
                    except Exception as e:
                        st.error(f"GitHub upload failed for {f.name}: {e}")
                        continue

                tmp = os.path.join("data", f"{int(time.time())}_{f.name}")
                with open(tmp, "wb") as out:
                    out.write(raw_bytes)

                try:
                    raw = load_text_from_file(tmp)
                    pieces = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
                    added_count = add_records(pieces, tags, f.name, embed_model, added_counter=added)
                    if added_count == 0:
                        st.warning(f"‘{f.name}’ contained no extractable text; skipped.")
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")

            st.success(f"Ingested {len(files)} file(s) → {added['n']} chunks.")
            if uploaded_to_gh:
                owner, repo, branch = gh_repo_info()
                link_branch = branch or "main"
                st.info("Saved to GitHub:\n" + "\n".join(
                    [f"- https://github.com/{owner}/{repo}/blob/{link_branch}/{p}" for p in uploaded_to_gh]
                ))

    if corpus:
        st.write(f"Corpus size: **{len(corpus)}** chunks across your uploads.")
    else:
        st.info("No chunks yet. Upload and ingest above.")

    st.markdown("#### Restore from GitHub storage")
    if st.button("Load all files from GitHub"):
        try:
            os.makedirs("data", exist_ok=True)
            items = gh_list_dir("uploads")
            if not items:
                st.info("No files in GitHub /uploads yet.")
            else:
                reingested = 0
                for it in sorted(items, key=lambda x: x.get("name", "")):
                    name = it["name"]
                    b = gh_get_file(f"uploads/{name}")
                    tmp = os.path.join("data", name)
                    with open(tmp, "wb") as out:
                        out.write(b)
                    try:
                        raw = load_text_from_file(tmp)
                        pieces = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
                        reingested += add_records(pieces, ["GitHub"], name, embed_model)
                    except Exception as e:
                        st.error(f"Failed to reprocess {name}: {e}")
                st.success(f"Restored {len(items)} file(s) → {reingested} chunks.")
        except httpx.HTTPStatusError as e:
            st.error(f"GitHub restore failed: {e}\n\n{e.response.text[:500]}")
        except Exception as e:
            st.error(f"GitHub restore failed: {e}")

# ==================== OUTLINE ====================
with tab_outline:
    st.subheader("Thesis & Style")
    c1, c2 = st.columns(2)
    with c1:
        plan["thesis"] = st.text_area("Book thesis or goal", value=plan["thesis"], height=140)
    with c2:
        plan["style"] = st.text_area("Style sheet (voice, tense, pacing, terminology)", value=plan["style"], height=140)

    # Auto-detect master outline file in local /data
    outline_file = None
    if os.path.exists("data"):
        for fname in os.listdir("data"):
            if "outline" in fname.lower() and "master" in fname.lower():
                outline_file = fname
                break

    if outline_file:
        st.session_state["outline_file_path"] = os.path.join("data", outline_file)
        st.caption(f"📄 Using master outline: **{outline_file}**")
    else:
        st.caption("⚠️ No master outline detected. Name it with both ‘outline’ and ‘master’ (e.g., `BookOutline_MASTER.docx`), then **Load all files from GitHub** in the Sources tab.")

    if st.button("Generate outline", type="primary"):
        msgs = [
            {"role":"system","content":"You are a meticulous long-form book-writing assistant. Reply in clean Markdown."},
            {"role":"user","content":f"Plan a new 250–300 page book.\nTitle: {plan['title']}\n\nThesis:\n{plan['thesis']}\n\nUse only my uploaded corpus as background.\nConstraints: 10–14 chapters, coherent arc.\nProduce: 3 title options, detailed TOC, 3–6 subtopics per chapter, 2–4 sentence synopsis per chapter, and a brief style sheet."}
        ]
        try:
            resp = client.chat.completions.create(model=chat_model, temperature=0.5, messages=msgs, max_tokens=1800)
            st.markdown(resp.choices[0].message.content)
            st.info("Copy/paste chapter titles & synopses into the Draft tab.")
        except Exception as e:
            st.error(f"Outline generation failed: {e}")

    # --- Strict source mapping that mirrors the master outline exactly ---
    mapping_container = st.container()
    with mapping_container:
        colA, colB = st.columns([1,1])
        with colA:
            do_save_mapping = st.checkbox("Also save mapping to GitHub (/uploads)", value=True)
        with colB:
            mapping_filename = st.text_input("Mapping filename (no path)", value=f"mapping_{int(time.time())}.md")

    if st.button("Generate source mapping (lock to master outline)"):
        if not st.session_state.get("outline_file_path"):
            st.error("No master outline detected. Load your outline into /data first (Sources → Load all files from GitHub).")
        else:
            thesis_mission = plan["thesis"] or "Create a unified, updated narrative based on my two books; maintain my voice and structure."
            outline_path = st.session_state["outline_file_path"]
            try:
                ext = os.path.splitext(outline_path)[1].lower()
                if ext == ".docx":
                    _doc = DocxDocument(outline_path)
                    master_outline_text = "\n".join(p.text for p in _doc.paragraphs)
                else:
                    master_outline_text = open(outline_path, "r", encoding="utf-8", errors="ignore").read()
            except Exception as e:
                st.error(f"Could not read master outline: {e}")
                master_outline_text = ""

            strict_instructions = """
Use only the chapter and subchapter structure from the uploaded master outline below.
Rules:
1) Do NOT create, remove, rename, paraphrase, or reorder any chapters or subchapters.
2) Copy all section titles exactly as written in the outline.
3) For each chapter/subchapter, list the most relevant excerpts from my ingested sources (two books + any uploads).
4) If no material exists for a section, write: GAP — needs new research or original writing.
5) Keep structure identical to the outline. Use this format:

# Chapter X: [Exact title]
## Subchapter X.X: [Exact title]
- [Source: filename] "short excerpt / anchor phrase" — why it fits
- [Source: filename] "short excerpt / anchor phrase" — why it fits
""".strip()

            content_for_llm = f"""
MASTER OUTLINE (do not alter):
{master_outline_text}

BOOK THESIS / MISSION:
{thesis_mission}

STYLE / VOICE GUIDANCE:
{plan['style']}
""".strip()

            msgs = [
                {"role":"system","content":"You are a meticulous book development editor. Follow instructions exactly and mirror the provided outline structure."},
                {"role":"user","content":strict_instructions},
                {"role":"user","content":content_for_llm},
            ]

            try:
                resp = client.chat.completions.create(
                    model=chat_model,
                    temperature=0.2,
                    messages=msgs,
                    max_tokens=2800
                )
                mapping_md = resp.choices[0].message.content
                st.success("Source mapping created (mirrors master outline).")
                st.markdown(mapping_md)

                if do_save_mapping:
                    try:
                        fname = mapping_filename.strip() or f"mapping_{int(time.time())}.md"
                        if "/" in fname or "\\" in fname:
                            st.warning("Mapping filename should not include a path; saving under /uploads automatically.")
                            fname = os.path.basename(fname)
                        rel = f"uploads/{fname}"
                        gh_put_file(rel, mapping_md.encode("utf-8"), f"Add mapping {fname}")
                        owner, repo, branch = gh_repo_info()
                        link_branch = (branch or "main")
                        st.info(f"Saved: https://github.com/{owner}/{repo}/blob/{link_branch}/{rel}")
                    except Exception as e:
                        st.error(f"Could not save mapping to GitHub: {e}")
            except Exception as e:
                st.error(f"Source mapping failed: {e}")

# ==================== Retrieval ====================
def retrieve(query: str, k: int, tag_filter_text: str) -> List[Dict[str, Any]]:
    if not corpus:
        return []
    filters = {t.strip().lower() for t in tag_filter_text.split(",") if t.strip()}
    q = (query or "").strip()
    if not q:
        return []
    qvecs = embed_texts([q], model=embed_model)
    if not qvecs:
        return []
    qvec = qvecs[0]
    scored = []
    for rec in corpus:
        if filters and not (set(map(str.lower, rec["tags"])) & filters):
            continue
        score = cosine_sim(qvec, rec["emb"])
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

def make_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(hits):
        src = h.get("source","")
        tags = ",".join(h.get("tags",[]))
        snippet = h["text"][:1200]
        blocks.append(f"[S{i+1} | {tags} | {src}] {snippet}")
    return "\n\n".join(blocks) if blocks else "(no retrieved context)"

# ==================== DRAFT ====================
with tab_draft:
    st.subheader("Draft a chapter")

    # -------- Mapping → Prefill helper --------
    with st.expander("📚 Use mapping to prefill chapter fields", expanded=False):
        st.caption("Pick a saved mapping_*.md from /data, then choose a chapter (and optional subchapter). We’ll prefill chapter #, title, and a retrieval-ready query hint.")

        mapping_files = load_mapping_files_from_data()
        if not mapping_files:
            st.info("No mapping files found in /data. In the Outline tab, generate a source mapping (and keep 'save mapping to GitHub' checked), then go to Sources → Load all files from GitHub.")
        else:
            mlabel = [f"{i+1}. {fname}" for i,(fname,_) in enumerate(mapping_files)]
            idx = st.selectbox("Choose a mapping file", list(range(len(mapping_files))), format_func=lambda i: mlabel[i])
            chosen_name, chosen_path = mapping_files[idx]
            try:
                mtext = open(chosen_path, "r", encoding="utf-8", errors="ignore").read()
                parsed = parse_mapping_markdown(mtext)
                chapters_list = parsed["chapters"]
                if not chapters_list:
                    st.warning("Could not find any '# Chapter N: Title' headings in this mapping.")
                else:
                    ch_labels = [f"{c['num']}: {c['title']}" for c in chapters_list]
                    ch_idx = st.selectbox("Chapter", list(range(len(chapters_list))), format_func=lambda i: ch_labels[i])
                    selected_ch = chapters_list[ch_idx]

                    sc_labels = ["(None — use chapter level)"] + [f"{sc['num']}: {sc['title']}" for sc in selected_ch.get("subchapters", [])]
                    sc_idx = st.selectbox("Subchapter (optional)", list(range(len(sc_labels))), format_func=lambda i: sc_labels[i])
                    selected_sc = None if sc_idx == 0 else selected_ch["subchapters"][sc_idx-1]

                    bullets = (selected_sc["bullets"] if selected_sc else selected_ch.get("bullets", [])) or []
                    anchors = []
                    src_set = set()
                    for b in bullets[:10]:
                        if b.get("source"):
                            src_set.add(b["source"])
                        if b.get("excerpt"):
                            anchors.append(b["excerpt"])
                    src_hint = ", ".join(sorted(src_set))
                    ex_hint = "; ".join(anchors)

                    suggested_query = f"Chapter {selected_ch['num']}: {selected_ch['title']}"
                    if selected_sc:
                        suggested_query += f" — Subchapter {selected_sc['num']}: {selected_sc['title']}"
                    if src_hint:
                        suggested_query += f" | Sources: {src_hint}"
                    if ex_hint:
                        suggested_query += f" | Anchors: {ex_hint}"

                    st.text_area("Preview of retrieval hint", value=suggested_query, height=100, key="mapping_preview_hint")

                    if st.button("Prefill Draft fields from mapping"):
                        st.session_state.setdefault("plan", {}).setdefault("chapters", {})
                        ch_number = selected_ch["num"]
                        st.session_state["plan"]["chapters"].setdefault(ch_number, {"title":"", "synopsis":"", "draft":"", "status":"draft"})
                        ch_state_mut = st.session_state["plan"]["chapters"][ch_number]

                        prefill_title = selected_sc["title"] if selected_sc else selected_ch["title"]
                        ch_state_mut["title"] = prefill_title
                        if anchors:
                            ch_state_mut["synopsis"] = (ch_state_mut.get("synopsis") or f"This section builds on: {', '.join(anchors[:5])}.")
                        else:
                            ch_state_mut["synopsis"] = ch_state_mut.get("synopsis") or ""

                        st.session_state["__prefill_query_hint__"] = st.session_state.get("mapping_preview_hint","")
                        st.session_state["__prefill_ch_number__"] = ch_number

                        st.success(f"Prefilled Chapter {ch_number} — {prefill_title}. Scroll to the editor; fields will be set.")
                        st.rerun()
            except Exception as e:
                st.error(f"Could not read or parse mapping: {e}")

    # -------- Chapter tracker UI --------
    st.markdown("**Chapter tracker**")
    if plan["chapters"]:
        for k in sorted(plan["chapters"].keys(), key=lambda x: int(x) if x.isdigit() else x):
            item = plan["chapters"][k]
            st.write(f"• {k}: {item.get('title','')} — status: {item.get('status','(none)')} — words: {len(item.get('draft','').split()) if item.get('draft') else 0}")
    else:
        st.info("No chapters yet — add one below.")

    # -------- Editor --------
    if "__prefill_ch_number__" in st.session_state:
        default_ch = st.session_state["__prefill_ch_number__"]
        del st.session_state["__prefill_ch_number__"]
    else:
        current_keys = list(plan["chapters"].keys())
        default_ch = current_keys[0] if current_keys else "1"

    ch_num = st.text_input("Chapter number", value=default_ch)
    ch_state = plan["chapters"].setdefault(ch_num, {"title":"", "synopsis":"", "draft":"", "status":"draft"})
    ch_state["title"] = st.text_input("Chapter title", value=ch_state["title"])
    ch_state["synopsis"] = st.text_area("Chapter synopsis", value=ch_state["synopsis"], height=140)
    target_words = st.number_input("Target words", 800, 8000, 3500, 100)

    if "__prefill_query_hint__" in st.session_state:
        prefill_q = st.session_state["__prefill_query_hint__"]
        del st.session_state["__prefill_query_hint__"]
        query_hint = st.text_input("Optional retrieval hint (keywords)", value=prefill_q)
    else:
        query_hint = st.text_input("Optional retrieval hint (keywords)")

    # -------- Draft button --------
    if st.button("Retrieve & Draft", type="primary"):
        q = query_hint or f"{plan['title']} - {ch_state['title']} - {ch_state['synopsis']}"
        hits = retrieve(q, top_k, tag_filter)
        context = make_context(hits)

        if longform:
            # Plan sections
            plan_prompt = f"""
You will draft a long chapter in {sections_target} sections to reach ~{target_words} words.
Topic: Chapter {ch_num}: "{ch_state['title']}"
Book: "{plan['title']}"
Style: {plan['style']}
Synopsis: {ch_state['synopsis']}
Thesis: {plan['thesis']}

First, produce ONLY a numbered section plan with {sections_target} short, specific section titles (1–2 lines each) covering the full scope. Avoid overlap. Do not write the prose yet.
Context notes (may quote/attribute as [S1], [S2], etc. later):
{context}
""".strip()

            msgs = [
                {"role":"system","content":"You are an expert long-form writing assistant."},
                {"role":"user","content":plan_prompt}
            ]
            try:
                plan_resp = client.chat.completions.create(
                    model=chat_model, temperature=0.4, messages=msgs, max_tokens=1200
                )
                section_plan = plan_resp.choices[0].message.content
                st.markdown("### Section plan")
                st.markdown(section_plan)

                # Loop sections
                assembled = []
                for si in range(1, sections_target+1):
                    sec_prompt = f"""
Draft Section {si} (~{section_words} words) for Chapter {ch_num}: "{ch_state['title']}".
Follow this approved section plan:
{section_plan}

Constraints:
- Write ONLY Section {si}. Do not draft other sections.
- Maintain voice and pacing. Avoid repetition with prior sections.
- Use context notes below; paraphrase and attribute facts as [S1], [S2], etc. where appropriate.
- End Section {si} with a line that naturally sets up the next section.

Context notes:
{context}
""".strip()
                    msgs = [
                        {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
                        {"role":"user","content":sec_prompt}
                    ]
                    sec_resp = client.chat.completions.create(
                        model=chat_model, temperature=0.6, messages=msgs, max_tokens=1200
                    )
                    assembled.append(sec_resp.choices[0].message.content)

                # Stitch + polish
                stitch_prompt = f"""
Combine the sections below into a cohesive chapter (~{target_words} words).
Keep the existing text; smooth transitions; remove duplicates; consistent headings (## / ###).
Open with a strong hook and close with a forward-looking ending to tee up the next chapter.

Sections:
{"\n\n---\n\n".join(assembled)}
""".strip()
                msgs = [
                    {"role":"system","content":"You are a meticulous editor. Reply in Markdown."},
                    {"role":"user","content":stitch_prompt}
                ]
                final_resp = client.chat.completions.create(
                    model=chat_model, temperature=0.3, messages=msgs, max_tokens=1800
                )
                ch_state["draft"] = final_resp.choices[0].message.content
                ch_state["status"] = "draft"
                st.success("Draft created (longform).")
                st.markdown(ch_state["draft"])
            except Exception as e:
                st.error(f"Longform drafting failed: {e}")

        else:
            prompt = f"""
Draft Chapter {ch_num}: "{ch_state['title']}" for the book "{plan['title']}".
Target length ~{target_words} words.
Style sheet: {plan['style']}
Chapter synopsis: {ch_state['synopsis']}
Book thesis: {plan['thesis']}

Use these retrieved notes (paraphrase; attribute inline as [S1], [S2], etc. when you borrow specific facts or phrasing). Keep voice consistent.
{context}

Write a cohesive chapter in Markdown with:
- opening hook
- 3–6 sections (## / ### headings)
- smooth transitions
- ending that tees up the next chapter.
""".strip()
            msgs = [{"role":"system","content":"You are a careful long-form writing assistant, reply in Markdown."},
                    {"role":"user","content":prompt}]
            try:
                resp = client.chat.completions.create(
                    model=chat_model,
                    temperature=0.5,
                    messages=msgs,
                    max_tokens=4096
                )
                ch_state["draft"] = resp.choices[0].message.content
                ch_state["status"] = "draft"
                st.success("Draft created.")
                st.markdown(ch_state["draft"])
            except Exception as e:
                st.error(f"Drafting failed: {e}")

    # -------- Revise --------
    if ch_state.get("draft"):
        goals = st.text_input("Revision goals (e.g., tighten intro, add example)")
        if st.button("Revise chapter"):
            msgs = [
                {"role":"system","content":"You are a meticulous editor. Reply in Markdown, preserving headings."},
                {"role":"user","content":f"Revise the chapter to meet these goals: {goals or 'Improve clarity and flow; preserve voice and length.'}\n\nChapter:\n{ch_state['draft']}"}
            ]
            try:
                resp = client.chat.completions.create(model=chat_model, temperature=0.5, messages=msgs, max_tokens=1800)
                ch_state["draft"] = resp.choices[0].message.content
                ch_state["status"] = "revised"
                st.success("Revised.")
                st.markdown(ch_state["draft"])
            except Exception as e:
                st.error(f"Revision failed: {e}")

# ==================== EXPORT ====================
with tab_export:
    st.subheader("Export manuscript (Markdown)")
    ordered = sorted([(k, v) for k, v in plan["chapters"].items() if v.get("draft")], key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    if not ordered:
        st.info("Draft at least one chapter first.")
    else:
        manuscript = f"# {plan['title']}\n\n*Thesis:* {plan['thesis']}\n\n"
        for k, v in ordered:
            manuscript += f"\n\n# Chapter {k}: {v.get('title','')}\n\n{v.get('draft','')}\n"
        st.download_button("Download Markdown", manuscript, file_name=f"{plan['title'].replace(' ','_')}.md")
