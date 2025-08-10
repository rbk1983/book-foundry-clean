import os, sys, time, math, hashlib, base64, json
from typing import List, Dict, Any

import streamlit as st

# ==================== Page + Diagnostics ====================
st.set_page_config(page_title="Book Foundry — GitHub-persist (Py3.13 ready)", layout="wide")

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

# ==================== Secrets helper ====================
def _sec(name: str):
    """Safe access for Streamlit secrets/env (no .get on st.secrets)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

# ==================== OpenAI client (safe init) ====================
from openai import OpenAI

OPENAI_KEY = _sec("OPENAI_API_KEY")
st.sidebar.write("Has OPENAI_API_KEY:", bool(OPENAI_KEY))
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is missing. Add it in App → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_KEY)

# ==================== GitHub helpers (persistence) ====================
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
    branch = _sec("GH_BRANCH")  # may be None/empty -> use repo default
    return owner, repo, branch

def gh_put_file(path_rel: str, content_bytes: bytes, message: str) -> dict:
    """Create/update a file in repo at path_rel. Returns GitHub response JSON."""
    owner, repo, branch = gh_repo_info()
    if not (owner and repo):
        raise RuntimeError("GH_REPO_OWNER or GH_REPO_NAME missing in Secrets.")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=60.0) as c:
        # Check if exists to include SHA
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
    """List files under path_rel. Returns list of items (dicts)."""
    owner, repo, branch = gh_repo_info()
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_rel}"
    params = {"ref": branch} if branch else None
    with httpx.Client(timeout=30.0) as c:
        r = c.get(url, params=params, headers=gh_headers())
        if r.status_code == 404:
            # Distinguish between "repo not accessible" and "folder not found"
            r_repo = c.get(f"https://api.github.com/repos/{owner}/{repo}", headers=gh_headers())
            if r_repo.status_code != 200:
                raise RuntimeError(f"Repo {owner}/{repo} not accessible (status {r_repo.status_code}).")
            # Repo exists -> just no /uploads yet
            return []
        r.raise_for_status()
        items = r.json()
        return [i for i in items if i.get("type") == "file"]

def gh_get_file(path_rel: str) -> bytes:
    """Fetch file content bytes from repo/path."""
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
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    return splitter.split_text(text)

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
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

# ==================== Session state ====================
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

# ==================== Sidebar settings ====================
st.title("📚 Book Foundry — GitHub-persist (clean start)")
st.caption("Uploads → in-memory embeddings → outline/draft. Files persisted to your private GitHub repo.")

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
        except Exception as e:
            st.error(f"Write failed: {e}")

# ==================== Tabs ====================
tab_sources, tab_outline, tab_draft, tab_export = st.tabs(
    ["📥 Sources", "🧭 Outline", "✍️ Draft", "📤 Export"]
)

# ==================== SOURCES ====================
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
            uploaded_to_gh = []
            os.makedirs("data", exist_ok=True)

            for f in files:
                raw_bytes = f.read()
                size_mb = len(raw_bytes) / (1024 * 1024)

                gh_path = None
                if persist_to_github:
                    if size_mb > 95:
                        st.error(f"‘{f.name}’ is {size_mb:.1f} MB (>100 MB GitHub limit). Use S3 for this one.")
                        continue
                    ts = int(time.time())
                    gh_rel = f"uploads/{ts}_{f.name}"
                    try:
                        res = gh_put_file(gh_rel, raw_bytes, f"Add upload {f.name}")
                        gh_path = res.get("content", {}).get("path", gh_rel)
                        uploaded_to_gh.append(gh_path)
                    except Exception as e:
                        st.error(f"GitHub upload failed for {f.name}: {e}")
                        continue  # skip ingest if persistence failed

                # also save to temp so we can parse now
                tmp = os.path.join("data", f"{int(time.time())}_{f.name}")
                with open(tmp, "wb") as out:
                    out.write(raw_bytes)

                # parse + chunk + embed
                try:
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
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")

            st.success(f"Ingested {len(files)} file(s) → {added_chunks} chunks.")
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
            items = gh_list_dir("uploads")
            if not items:
                st.info("No files in GitHub /uploads yet.")
            else:
                reingested = 0
                for it in sorted(items, key=lambda x: x.get("name", "")):
                    name = it["name"]
                    b = gh_get_file(f"uploads/{name}")
                    # write to temp, then parse
                    tmp = os.path.join("data", name)
                    with open(tmp, "wb") as out:
                        out.write(b)
                    raw = load_text_from_file(tmp)
                    pieces = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
                    embs = embed_texts(pieces, model=embed_model)
                    for i, (text, emb) in enumerate(zip(pieces, embs)):
                        corpus.append({
                            "id": f"{sha16(name)}:{i}",
                            "text": text,
                            "tags": ["GitHub"],
                            "source": name,
                            "emb": emb,
                        })
                        reingested += 1
                st.success(f"Restored {len(items)} file(s) → {reingested} chunks.")
        except Exception as e:
            st.error(f"GitHub restore failed: {e}")

# ==================== OUTLINE ====================
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
        resp = client.chat.completions.create(model=st.session_state.get("chat_model", "gpt-4o") or "gpt-4o",
                                              temperature=temperature, messages=msgs)
        st.markdown(resp.choices[0].message.content)
        st.info("Copy/paste chapter titles & synopses into the Draft tab.")

# ==================== Retrieval (in-memory) ====================
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

# ==================== DRAFT ====================
with tab_draft:
    st.subheader("Draft a chapter")
    current_keys = list(plan["chapters"].keys())
    default_ch = current_keys[0] if current_keys else "1"
    ch_num = st.text_input("Chapter number", value=default_ch)
    ch_state = plan["chapters"].setdefault(ch_num, {"title":"", "synopsis":"", "draft":""})
    ch_state["title"] = st.text_input("Chapter title", value=ch_state["title"])
    ch_state["synopsis"] = st.text_area("Chapter synopsis", value=ch_state["synopsis"], height=100)
    target_words = st.number_input("Target words", 800, 8000, 3500, 100)
    query_hint = st.text_input("Optional retrieval hint (keywords)")

    if st.button("Retrieve & Draft", type="primary"):
        q = query_hint or f"{plan['title']} - {ch_state['title']} - {ch_state['synopsis']}"
        hits = retrieve(q, top_k, tag_filter)
        blocks = [f"[S{i+1}] {h['text'][:1200]}" for i, h in enumerate(hits)]
        context = "\n\n".join(blocks) if blocks else "(no retrieved context)"
        prompt = f"""
Draft Chapter {ch_num}: "{ch_state['title']}" for the book "{plan['title']}".
Target length ~{target_words} words.
Style sheet: {plan['style']}
Chapter synopsis: {ch_state['synopsis']}
Book thesis: {plan['thesis']}

Use these retrieved notes (paraphrase; cite as [S1], [S2] if used):
{context}

Write a cohesive chapter in Markdown with:
- opening hook
- 3–6 sections (## / ### headings)
- smooth transitions
- ending that tees up the next chapter.
""".strip()
        msgs = [{"role":"system","content":"You are a careful long-form writing assistant, reply in Markdown."},
                {"role":"user","content":prompt}]
        resp = client.chat.completions.create(model=chat_model, temperature=temperature, messages=msgs)
        ch_state["draft"] = resp.choices[0].message.content
        st.success("Draft created.")
        st.markdown(ch_state["draft"])

    if ch_state.get("draft"):
        goals = st.text_input("Revision goals (e.g., tighten intro, add example)")
        if st.button("Revise chapter"):
            msgs = [
                {"role":"system","content":"You are a meticulous editor. Reply in Markdown."},
                {"role":"user","content":f"Revise the chapter to meet these goals: {goals or 'Improve clarity and flow; preserve voice and length.'}\n\nChapter:\n{ch_state['draft']}"}
            ]
            resp = client.chat.completions.create(model=chat_model, temperature=0.5, messages=msgs)
            ch_state["draft"] = resp.choices[0].message.content
            st.success("Revised.")
            st.markdown(ch_state["draft"])

# ==================== EXPORT ====================
with tab_export:
    st.subheader("Export manuscript (Markdown)")
    ordered = sorted([(k, v) for k, v in plan["chapters"].items() if v.get("draft")], key=lambda x: int(x[0]))
    if not ordered:
        st.info("Draft at least one chapter first.")
    else:
        manuscript = f"# {plan['title']}\n\n*Thesis:* {plan['thesis']}\n\n"
        for k, v in ordered:
            manuscript += f"\n\n# Chapter {k}: {v.get('title','')}\n\n{v.get('draft','')}\n"
        st.download_button("Download Markdown", manuscript, file_name=f"{plan['title'].replace(' ','_')}.md")
