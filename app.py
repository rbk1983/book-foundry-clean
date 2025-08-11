
import os, sys, time, json, math, hashlib, base64, re
from typing import List, Dict, Any
import streamlit as st

# ============== Page ==============
st.set_page_config(page_title="Rahim's AI Book Studio", layout="wide")
st.sidebar.markdown("### 🔎 Diagnostics")
st.sidebar.write("Python:", sys.version)
try:
    import openai as _oa
    st.sidebar.write("openai pkg:", getattr(_oa, "__version__", "unknown"))
except Exception as e:
    st.sidebar.write("openai import error:", e)

from openai import OpenAI
import numpy as np
from io import BytesIO

# ============== Secrets ==============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "index_v1.jsonl")

# ============== Helpers ==============
def read_pdf_bytes(b: bytes) -> str:
    from pypdf import PdfReader
    pdf = PdfReader(BytesIO(b))
    texts = []
    for page in pdf.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n\n".join(texts)

def read_docx_bytes(b: bytes) -> str:
    from docx import Document
    f = BytesIO(b)
    doc = Document(f)
    return "\n\n".join([p.text for p in doc.paragraphs])

def chunk_text(txt: str, size: int = 1200, overlap: int = 100) -> List[str]:
    words = txt.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return [c for c in chunks if c.strip()]

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    res = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / denom) if denom != 0 else 0.0

def load_index() -> List[Dict[str, Any]]:
    if not os.path.exists(INDEX_PATH):
        return []
    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_index(rows: List[Dict[str, Any]]):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def index_stats(rows: List[Dict[str, Any]]):
    return len([r for r in rows if r.get("kind") == "chunk"])

def retrieve(query: str, rows: List[Dict[str, Any]], model: str, top_k: int = 12):
    # Build query embedding
    q_emb = embed_texts([query], model=model)[0]
    q_vec = np.array(q_emb, dtype=np.float32)
    scored = []
    for r in rows:
        if r.get("kind") != "chunk":
            continue
        v = np.array(r["embedding"], dtype=np.float32)
        s = cosine_sim(q_vec, v)
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]

def make_context(hits: List[Dict[str, Any]]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {})
        label = meta.get("label", "")
        out.append(f"[S{i}] from {label}\n{h['text']}")
    return "\n\n".join(out)

# ============== Sidebar settings ==============
st.sidebar.markdown("### ⚙️ Settings")
chat_model = st.sidebar.text_input("Chat model", value=st.secrets.get("CHAT_MODEL", "gpt-4o-mini"))
embed_model = st.sidebar.text_input("Embedding model", value=st.secrets.get("EMBED_MODEL", "text-embedding-3-large"))
temp = st.sidebar.slider("Creativity (temperature)", 0.0, 1.2, 0.6, 0.1)
top_k = st.sidebar.slider("Top-k retrieved chunks", 4, 50, 14, 1)
chunk_size = st.sidebar.slider("Chunk size (words)", 300, 2000, 1000, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap (words)", 0, 400, 120, 10)

st.sidebar.markdown("---")
st.sidebar.checkbox("Longform mode", value=True, key="longform_mode")
sections_default = st.sidebar.number_input("Sections to draft (longform)", 3, 12, 6, 1)
words_per_section = st.sidebar.number_input("Words per section (guide)", 200, 1200, 600, 50)

# ============== Session state ==============
if "plan" not in st.session_state:
    st.session_state.plan = {"title": "", "style": "", "thesis": ""}
if "chapters" not in st.session_state:
    st.session_state.chapters = {}

plan = st.session_state.plan
chapters = st.session_state.chapters

# ============== Tabs ==============
tabs = st.tabs(["📚 Library", "🧭 Outline", "✍️ Draft"])

# ============== Library Tab ==============
with tabs[0]:
    st.subheader("Library")
    rows = load_index()
    st.write(f"Indexed chunks: {index_stats(rows)}")

    upl = st.file_uploader("Upload PDFs or DOCX (stored in ./data)", type=["pdf","docx"], accept_multiple_files=True)
    label = st.text_input("Label for these files (e.g., Book 1, Book 2)", value="")
    if upl:
        new_rows = rows[:]
        total_chunks = 0
        for f in upl:
            try:
                b = f.read()
                if f.name.lower().endswith(".pdf"):
                    text = read_pdf_bytes(b)
                else:
                    text = read_docx_bytes(b)
                chunks = chunk_text(text, size=int(chunk_size), overlap=int(chunk_overlap))
                emb = embed_texts(chunks, model=embed_model)
                base_id = hashlib.sha1(f.name.encode("utf-8")).hexdigest()[:8]
                for i, (c, e) in enumerate(zip(chunks, emb), 1):
                    new_rows.append({
                        "kind":"chunk",
                        "id": f"{base_id}-{i}",
                        "text": c,
                        "embedding": e,
                        "meta": {"label": label or f.name}
                    })
                total_chunks += len(chunks)
            except Exception as e:
                st.error(f"Failed to ingest {f.name}: {e}")
        save_index(new_rows)
        st.success(f"Ingested {len(upl)} file(s) → {total_chunks} chunks.")
        rows = new_rows

    st.markdown("**Reload index** if you edited files on disk.")
    if st.button("Reload index"):
        rows = load_index()
        st.success(f"Loaded. Indexed chunks: {index_stats(rows)}")

# ============== Outline Tab ==============
with tabs[1]:
    st.subheader("Master Outline")
    plan["title"] = st.text_input("Book title", value=plan.get("title",""))
    plan["thesis"] = st.text_area("Book thesis or goal", height=120, value=plan.get("thesis",""))
    plan["style"] = st.text_area("Style sheet (voice, tense, pacing, terminology)", height=160, value=plan.get("style",""))

    st.markdown("---")
    ch_num = st.number_input("Chapter #", 1, 100, 1)
    if ch_num not in chapters:
        chapters[int(ch_num)] = {"title":"", "synopsis":"", "draft":""}
    ch_state = chapters[int(ch_num)]
    ch_state["title"] = st.text_input("Chapter title", value=ch_state.get("title",""))
    ch_state["synopsis"] = st.text_area("Chapter synopsis (2-5 sentences)", value=ch_state.get("synopsis",""))

    st.info("Tip: Keep chapter synopsis tight; we use it to guide retrieval and the section plan.")

# ============== Draft Tab ==============
with tabs[2]:
    st.subheader("Draft Chapter")
    ch_num = st.number_input("Chapter # to draft", 1, 100, 1, key="draft_ch_num")
    ch_state = chapters.get(int(ch_num), {"title":"","synopsis":"","draft":""})
    st.write(f"**Chapter:** {ch_state.get('title','(untitled)')}")

    target_words = st.number_input("Target words", 800, 20000, 3500, 100)
    sections_target = st.number_input("Sections (longform)", 3, 12, int(sections_default), 1)
    section_words = max(150, int(words_per_section))

    prefill_q = f"{plan['title']} - {ch_state['title']} - {ch_state['synopsis']}".strip(" -")
    query_hint = st.text_input("Optional retrieval hint (keywords)", value=prefill_q) if prefill_q else st.text_input("Optional retrieval hint (keywords)")
    ref_urls = st.text_area(
        "Reference URLs for web facts (optional - one per line)",
        placeholder="https://example.com/industry-report\nhttps://another.com/profile",
        height=100
    )

    # -------- Draft button --------
    if st.button("Retrieve & Draft", type="primary"):
        q = query_hint or f"{plan['title']} - {ch_state['title']} - {ch_state['synopsis']}"
        rows = load_index()
        hits = retrieve(q, rows, model=embed_model, top_k=int(top_k))
        context = make_context(hits)

        urls_list = [u.strip() for u in (ref_urls or "").splitlines() if u.strip()]
        urls_block = "\n".join(f"- {u}" for u in urls_list) if urls_list else "(none provided)"

        if st.session_state.get("longform_mode", True):
            # ---- Section planning ----
            plan_prompt = f"""
You will draft a long chapter in {sections_target} sections to reach ~{target_words} words.
Topic: Chapter {ch_num}: "{ch_state['title']}"
Book: "{plan['title']}"
Style: {plan['style']}
Synopsis: {ch_state['synopsis']}
Thesis: {plan['thesis']}

Context notes (from my uploaded books; paraphrase as needed):
{context}

Reference URLs for any *web-sourced facts* (only cite if you use them later):
{urls_block}

Produce ONLY a numbered section plan with {sections_target} short, specific section titles (1-2 lines each) covering the full scope.
Rules:
- Reserve the final section title for "Conclusion" (exactly that word).
- No other section should be titled Conclusion.
- No prose yet - just the plan.
""".strip()

            msgs = [
                {"role":"system","content":"You are an expert long-form writing assistant."},
                {"role":"user","content":plan_prompt}
            ]
            try:
                plan_resp = client.chat.completions.create(
                    model=chat_model, temperature=0.3, messages=msgs, max_tokens=800
                )
                section_plan = plan_resp.choices[0].message.content
            except Exception as e:
                st.error(f"Planning failed: {e}")
                section_plan = ""

            st.markdown("**Section Plan**")
            st.code(section_plan or "(no plan)")

            # ---- Draft sections ----
            assembled = []
            for si in range(1, int(sections_target)+1):
                sec_prompt = f"""
Draft Section {si} (~{section_words} words) for Chapter {ch_num}: "{ch_state['title']}".
Follow this approved section plan:
{section_plan}

Context notes (from my uploaded books; paraphrase as needed):
{context}

Reference URLs for any *web-sourced facts* (only cite if you use them here):
{urls_block}

MANDATORY RULES
- Write ONLY Section {si}. Do not draft other sections.
- Keep a single "## Conclusion" for the final section ONLY. If this is not the last section, do NOT write any conclusion section.
- If you use a verbatim quote from my books, put it in quotes and attribute inline as:
  - [Full Name], [Role/Title] at [Hotel/Restaurant] ([Country])
- Paraphrased book content: no citation.
- Web-sourced facts/figures: add a Markdown hyperlink to the source URL right where it appears.
- Maintain voice, no repetition, smooth handoff line to the next section.
""".strip()
                msgs = [
                    {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
                    {"role":"user","content":sec_prompt}
                ]
                try:
                    sec_resp = client.chat.completions.create(model=chat_model, temperature=temp, messages=msgs, max_tokens=1500)
                    assembled.append(sec_resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"Section {si} failed: {e}")

            # ---- Stitch ----
            sep = "\n\n---\n\n"
            stitch_prompt = f"""
Combine the sections below into a cohesive chapter (~{target_words} words).
Keep the existing text; smooth transitions; remove duplicates; consistent headings (## / ###).

CRITICAL STRUCTURE RULES
- There must be exactly ONE "## Conclusion" section, and it must be the final section in the chapter.
- If earlier sections introduced any conclusion-like heading, merge their content into regular sections and remove the extra conclusion headings.

ATTRIBUTION RULES
- Preserve all inline attributions for verbatim quotes from my books:
  "quote text" - [Full Name], [Role/Title] at [Hotel/Restaurant] ([Country])
- Keep web hyperlinks where used; do not invent links; do not add citations to paraphrased book content.

Sections:
{sep.join(assembled)}
""".strip()
            msgs = [
                {"role":"system","content":"You are a meticulous editor. Reply in Markdown."},
                {"role":"user","content":stitch_prompt}
            ]
            try:
                final_resp = client.chat.completions.create(model=chat_model, temperature=0.3, messages=msgs, max_tokens=3200)
                ch_state["draft"] = final_resp.choices[0].message.content
                ch_state["status"] = "drafted"
            except Exception as e:
                st.error(f"Stitching failed: {e}")

            # ---- Auto-continue if short ----
            def _wc(t): return len((t or "").split())
            passes = 0
            while _wc(ch_state["draft"]) < int(target_words * 0.95) and passes < 2:
                cont_prompt = f"""Continue the chapter seamlessly from where it stops.
Do NOT repeat previous text. Keep voice and structure; add substance.
Aim for ~{int(target_words/sections_target)}–{int(target_words/sections_target)+200} words.
Current chapter ends with:
{ch_state['draft'][-800:]}"""
                msgs = [
                    {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
                    {"role":"user","content":cont_prompt}
                ]
                try:
                    c_resp = client.chat.completions.create(model=chat_model, temperature=0.6, messages=msgs, max_tokens=1600)
                    ch_state["draft"] += "\n\n" + c_resp.choices[0].message.content
                except Exception:
                    break
                passes += 1

            st.markdown(ch_state["draft"])
            st.info("If the draft ends abruptly or is short, click 'Continue chapter' below.")
            if st.button("Continue chapter"):
                cont_prompt2 = f"""Continue the chapter seamlessly from where it stops.
Do NOT repeat prior sentences. Keep voice and structure; end with a clear closing that tees up the next chapter.
Current chapter ends with:
{ch_state['draft'][-1000:]}"""
                msgs = [
                    {"role":"system","content":"You are an expert long-form writing assistant. Reply in Markdown."},
                    {"role":"user","content":cont_prompt2}
                ]
                try:
                    c2 = client.chat.completions.create(model=chat_model, temperature=0.6, messages=msgs, max_tokens=1600)
                    ch_state["draft"] += "\n\n" + c2.choices[0].message.content
                    st.success("Extended.")
                    st.markdown(ch_state["draft"])
                except Exception as e:
                    st.error(f"Continuation failed: {e}")

        else:
            # ---- Single-pass chapter ----
            prompt = f"""
Draft Chapter {ch_num}: "{ch_state['title']}" for the book "{plan['title']}".
Target length ~{target_words} words.
Style sheet: {plan['style']}
Chapter synopsis: {ch_state['synopsis']}
Book thesis: {plan['thesis']}

Context notes (from my uploaded books; paraphrase as needed):
{context}

Reference URLs for any *web-sourced facts* (only cite if you use them):
{urls_block}

STRUCTURE RULES (MANDATORY)
- Use clear Markdown headings (## for major sections, ### for subsections).
- Include exactly ONE final section titled "## Conclusion" - place it at the very end.
- Do NOT include any other sections titled Conclusion earlier in the chapter.

ATTRIBUTION RULES (MANDATORY)
- If you include an exact, verbatim quote from my two books, put it in quotes and immediately attribute inline in this form:
  - [Full Name], [Role/Title] at [Hotel/Restaurant] ([Country])
  Example: "quote text" - Eric Ripert, Chef at Le Bernardin (USA)
- If you paraphrase or synthesize ideas from my books, do NOT add citations.
- If you use any *web-sourced facts or figures*, provide a Markdown hyperlink right where it appears (e.g., [source](https://...)). Only cite if you actually used a URL from the list.

WRITE THE CHAPTER
- Open with a strong hook.
- Include 3-6 sections with crisp, informative headings.
- Use specifics, examples, and short anecdotes; avoid fluff.
- Smooth transitions between sections.
- End with the single, final "## Conclusion" that tees up the next chapter.
""".strip()
            msgs = [{"role":"system","content":"You are an expert long-form writing assistant."},
                    {"role":"user","content":prompt}]
            try:
                resp = client.chat.completions.create(model=chat_model, temperature=temp, messages=msgs, max_tokens=4096)
                ch_state["draft"] = resp.choices[0].message.content
                ch_state["status"] = "drafted"
                st.markdown(ch_state["draft"])
            except Exception as e:
                st.error(f"Drafting failed: {e}")

    # -------- Revise --------
    if ch_state.get("draft"):
        goals = st.text_input("Revision goals (e.g., tighten intro, add example)")
        if st.button("Revise chapter"):
            msgs = [
                {"role":"system","content":"You are a meticulous editor. Reply in Markdown, preserving headings."},
                {"role":"user","content":f"""Revise the chapter with these mandatory rules:

STRUCTURE
- Ensure exactly one "## Conclusion" at the end; remove or merge any other conclusion headings.
- Normalize headings (## major, ### sub).
- Fix any duplicated or dangling transitions.

ATTRIBUTION
- Verbatim quotes from my books: keep quotes and add/verify inline attribution as:
  - [Full Name], [Role/Title] at [Hotel/Restaurant] ([Country])
- Paraphrased book content: no citation.
- Web-sourced facts/figures: must carry a valid Markdown hyperlink right where they appear.

GOALS (optional from user): {goals or 'Improve clarity, tighten flow, add concrete examples, maintain voice and length.'}

Chapter:
{ch_state['draft']}
"""}]
            try:
                resp = client.chat.completions.create(model=chat_model, temperature=0.5, messages=msgs, max_tokens=1800)
                ch_state["draft"] = resp.choices[0].message.content
                ch_state["status"] = "revised"
                st.success("Revised.")
                st.markdown(ch_state["draft"])
            except Exception as e:
                st.error(f"Revision failed: {e}")
