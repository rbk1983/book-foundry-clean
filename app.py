# app.py — Book Auto-Generator (tabs + autosave/resume + post-process)
import os, sys, time, math, hashlib, base64, json, re, io, random, datetime as _dt
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# ========== Page & watchdog ==========
st.set_page_config(page_title="Book Auto-Generator", layout="wide")

# ---- Emergency unlock support ----
if "busy" not in st.session_state:
    st.session_state.busy = False
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()
now = time.time()
if st.session_state.busy and (now - st.session_state.last_tick) > 180:
    st.warning("Previous run appears stalled; unlocking UI.")
    st.session_state.busy = False

# ========== Diagnostics (sidebar) ==========
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
        has_tavily = bool(os.getenv("TAVILY_A
