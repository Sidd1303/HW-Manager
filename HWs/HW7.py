import os
import hashlib
from collections import deque
from datetime import datetime

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

# --- Chroma & embeddings ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions

# --- LLM SDKs (no streaming) ---
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai


# =========================
# Constants & Mappings
# =========================

DATASET_PATH = "./news_data/news_dataset.csv"
VECTOR_DIR   = "./vectorstore/news"

PROVIDERS = ["OpenAI", "Anthropic", "Google Gemini"]
MODEL_MAP = {
    "OpenAI": {
        "Advanced": "gpt-4.1",
        "Lite":     "gpt-5-chat-latest",
    },
    "Anthropic": {
        "Advanced": "claude-opus-4-1",
        "Lite":     "claude-3-5-haiku-latest",
    },
    "Google Gemini": {
        "Advanced": "gemini-2.5-pro",
        "Lite":     "gemini-2.5-flash-lite",
    },
}


# =========================
# Utilities
# =========================

def _read_csv_or_fail(path: str) -> pd.DataFrame:
    """Load CSV robustly and normalize NaNs/whitespace."""
    if not os.path.exists(path):
        st.error(f"❌ Dataset not found at `{path}`. Put your CSV there and rerun.")
        st.stop()
    try:
        # keep_default_na=False to keep empty strings as empty (not NaN),
        # na_filter=False to avoid auto NaN parsing that can hide content.
        df = pd.read_csv(path, keep_default_na=False, na_filter=False)
        if df.empty:
            st.error("❌ CSV is empty.")
            st.stop()

        # Normalize: strip whitespace in string-like cells
        def _strip_cell(x):
            try:
                s = str(x)
                return " ".join(s.split())
            except Exception:
                return x

        for c in df.columns:
            df[c] = df[c].apply(_strip_cell)

        return df
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()


def _hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(df.columns.astype(str)).encode("utf-8"))
    for row in df.itertuples(index=False, name=None):
        h.update(("␟".join([str(x) for x in row])).encode("utf-8"))
    return h.hexdigest()[:12]


def _clean_html_if_any(text: str) -> str:
    """If a cell contains HTML, strip it; otherwise just return text."""
    try:
        soup = BeautifulSoup(text, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        return " ".join(soup.get_text().split())
    except Exception:
        return text


def _row_to_doc(row: pd.Series) -> str:
    """
    Build a retrieval doc from ANY non-empty columns in the row.
    We don't assume column names; we include everything with content.
    """
    parts = []
    for col, val in row.items():
        if val is None:
            continue
        sval = str(val).strip()
        if not sval:
            continue
        # strip HTML if present
        sval = _clean_html_if_any(sval)
        if sval:
            parts.append(f"{col.upper()}: {sval}")

    doc = "\n".join(parts).strip()

    # Fallback: if still empty, use entire row as string
    if not doc:
        doc = _clean_html_if_any(" ".join([f"{c}: {row[c]}" for c in row.index])).strip()

    return doc


def _ensure_vector_db(df: pd.DataFrame, openai_key: str):
    """
    Build or reuse a persisted Chroma collection for this CSV (by hash).
    Uses OpenAI text-embedding-3-small.
    """
    os.makedirs(VECTOR_DIR, exist_ok=True)
    c = chromadb.PersistentClient(path=VECTOR_DIR)

    col_name = f"news_{_hash_df(df)}"
    collection = c.get_or_create_collection(
        name=col_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        ),
    )

    # If already embedded, skip
    try:
        count = collection.count()
    except Exception:
        count = len(collection.get().get("ids", []))

    if count and count > 0:
        st.sidebar.info(f"🔁 Vector DB ready: `{col_name}` ({count} docs).")
        return collection, col_name

    # --- DEBUG PANEL: show columns and sample row
    with st.expander("🔎 CSV Debug (before embedding)"):
        st.write("**Columns detected:**", list(df.columns))
        st.write("**First row preview:**")
        st.json(df.iloc[0].to_dict() if len(df) > 0 else {})

    # Build all documents (1 doc per row)
    ids, docs, metas = [], [], []
    for i, row in df.reset_index(drop=True).iterrows():
        doc = _row_to_doc(row)
        if not doc.strip():
            continue
        ids.append(f"row_{i}")
        docs.append(doc)
        metas.append({
            "row_index": int(i),
            "title": str(row.get("title", "")),
            "source": str(row.get("source", "")),
            "url": str(row.get("url", "")),
            "date": str(row.get("date", "")),
        })

    if not docs:
        st.error("❌ No usable text found in the CSV rows **even after normalization**. "
                 "Check that the CSV actually contains text columns.")
        st.stop()

    collection.add(ids=ids, documents=docs, metadatas=metas)
    st.sidebar.success(f"✅ Embedded {len(docs)} rows into `{col_name}`.")
    return collection, col_name


def _pretty_sources(metas: list[dict]) -> list[str]:
    shown = []
    for m in metas:
        title = (m.get("title") or "").strip() or "<untitled>"
        src   = (m.get("source") or "").strip()
        url   = (m.get("url") or "").strip()
        date  = (m.get("date") or "").strip()
        piece = f"- **{title}** — {src or 'unknown'} — {date or 'n/a'}"
        if url:
            piece += f" — {url}"
        shown.append(piece)
    return shown


# =========================
# LLM Helpers (no streaming)
# =========================

def _run_openai(model: str, api_key: str, prompt: str) -> str:
    client = _OpenAI(api_key=api_key)
    resp = client.chat_completions.create(  # newer SDKs alias; fallback:
        model=model,
        messages=[{"role": "user", "content": prompt}],
    ) if hasattr(_OpenAI(), "chat_completions") else client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    # Handle both response styles
    choice = resp.choices[0]
    content = getattr(getattr(choice, "message", None), "content", None) or getattr(choice, "text", None)
    return content or ""


def _run_anthropic(model: str, api_key: str, prompt: str) -> str:
    client = _Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return resp.content[0].text or ""
    except Exception:
        return ""


def _run_gemini(model: str, api_key: str, prompt: str) -> str:
    _genai.configure(api_key=api_key)
    g = _genai.GenerativeModel(model)
    r = g.generate_content(prompt)
    return getattr(r, "text", "") or ""


def _call_llm(provider: str, model: str, keys: dict, prompt: str) -> str:
    if provider == "OpenAI":
        return _run_openai(model, keys["OPENAI_API_KEY"], prompt)
    if provider == "Anthropic":
        return _run_anthropic(model, keys["ANTHROPIC_API_KEY"], prompt)
    return _run_gemini(model, keys["GEMINI_API_KEY"], prompt)


# =========================
# App (auto-loads CSV)
# =========================

def render():
    st.title("🗞️ HW7 – News RAG Assistant (Auto CSV)")
    st.caption("Auto-loads the local CSV, persists the vector DB, and supports Advanced vs Lite per provider.")

    # --- API keys from secrets ---
    KEYS = {
        "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": st.secrets.get("ANTHROPIC_API_KEY"),
        "GEMINI_API_KEY": st.secrets.get("GEMINI_API_KEY"),
    }
    if not KEYS["OPENAI_API_KEY"]:
        st.error("❌ Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        st.stop()

    # --- Sidebar: provider + tier ---
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("Provider", options=PROVIDERS, index=0)
    tier = st.sidebar.radio("Model Tier", options=["Advanced", "Lite"], horizontal=True)
    model_name = MODEL_MAP[provider][tier]
    st.sidebar.write(f"**Model:** {model_name}")

    # Reset chat when provider/tier changes
    key_combo = f"{provider}::{tier}"
    if "hw7_combo" not in st.session_state or st.session_state.hw7_combo != key_combo:
        st.session_state.hw7_combo = key_combo
        st.session_state.hw7_chat = []
        st.sidebar.info("🔄 Chat was reset due to model change.")

    # --- Load CSV (auto) ---
    df = _read_csv_or_fail(DATASET_PATH)

    # --- Build / reuse vector DB for this CSV hash ---
    collection, col_name = _ensure_vector_db(df, KEYS["OPENAI_API_KEY"])

    # --- Tabs: Chatbot | Most Interesting | Topic Search | Evaluation ---
    tabs = st.tabs(["💬 Chatbot", "⭐ Most Interesting", "🎯 Topic Search", "📊 Evaluation"])

    # ====== Chatbot ======
    with tabs[0]:
        st.write("Ask news questions. Answers must come **only** from this CSV. If not found, the bot will say so.")
        # show history
        for m in st.session_state.hw7_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # memory (last 5 Q/A → 10 turns)
        def _memory_text():
            dq = deque(st.session_state.hw7_chat, maxlen=10)
            txt = ""
            for msg in dq:
                who = "User" if msg["role"] == "user" else "Assistant"
                txt += f"{who}: {msg['content']}\n"
            return txt

        user_q = st.chat_input("Ask about the news dataset…")
        if user_q:
            st.session_state.hw7_chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            res = collection.query(query_texts=[user_q], n_results=4)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            context = "\n\n".join(d for d in docs if d and d.strip())

            prompt = (
                "You are a news assistant grounded ONLY in the provided context. "
                'If the answer is not present, reply exactly: "I could not find this in the news dataset." '
                f"\n\nConversation (recent):\n{_memory_text()}"
                f"\n\nContext:\n{context}\n\nQuestion: {user_q}\nAnswer:"
            )
            try:
                ans = _call_llm(provider, model_name, KEYS, prompt).strip()
                if not ans:
                    ans = "I could not find this in the news dataset."
            except Exception as e:
                ans = f"⚠️ {provider} failed: {e}"

            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state.hw7_chat.append({"role": "assistant", "content": ans})
            st.session_state.hw7_chat = st.session_state.hw7_chat[-10:]

            if metas:
                with st.expander("📂 Sources used (this turn)"):
                    for line in _pretty_sources(metas):
                        st.markdown(line)

    # ====== Most Interesting ======
    with tabs[1]:
        st.write("Ranks potentially **legally interesting** items using a simple heuristic.")
        top_n = st.slider("Top N", 3, 15, 5)

        def score_row(r):
            score = 0.0
            text = " ".join([str(r.get(c, "")) for c in df.columns]).lower()
            for kw in ["lawsuit", "complaint", "regulator", "ftc", "doj", "settlement", "antitrust", "compliance"]:
                if kw in text:
                    score += 1.0
            try:
                d = pd.to_datetime(r.get("date", ""), errors="coerce")
                if pd.notna(d):
                    age_days = max(1, (datetime.now() - d.to_pydatetime()).days)
                    score += 3.0 / age_days
            except Exception:
                pass
            return score

        scored = df.copy()
        scored["_score"] = scored.apply(score_row, axis=1)
        scored = scored.sort_values("_score", ascending=False).head(top_n)

        for _, row in scored.iterrows():
            st.markdown(f"**{row.get('title','<untitled>')}**  \n"
                        f"{row.get('source','unknown')} — {row.get('date','n/a')}  \n"
                        f"{row.get('summary', '')[:400]}{'…' if str(row.get('summary','')).__len__()>400 else ''}")
            if str(row.get("url","")).strip():
                st.write(row["url"])
            st.divider()

    # ====== Topic Search ======
    with tabs[2]:
        topic = st.text_input("Enter a topic (e.g., antitrust in tech)")
        if st.button("Search Topic") and topic:
            res = collection.query(query_texts=[topic], n_results=6)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            context = "\n\n".join(d for d in docs if d and d.strip())
            q = f"Summarize the key points on this topic based only on the context.\n\nContext:\n{context}\n\nTopic: {topic}\nAnswer:"
            try:
                ans = _call_llm(provider, model_name, KEYS, q).strip() or "I could not find this in the news dataset."
            except Exception as e:
                ans = f"⚠️ {provider} failed: {e}"
            st.markdown(ans)
            if metas:
                with st.expander("📂 Sources used"):
                    for line in _pretty_sources(metas):
                        st.markdown(line)

    # ====== Evaluation ======
    with tabs[3]:
        st.write("Compare **Advanced vs Lite** for the same provider on one question.")
        eval_q = st.text_input("Enter a single evaluation question")
        if st.button("Run Evaluation") and eval_q:
            adv_model = MODEL_MAP[provider]["Advanced"]
            lite_model = MODEL_MAP[provider]["Lite"]
            res = collection.query(query_texts=[eval_q], n_results=6)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            context = "\n\n".join(d for d in docs if d and d.strip())

            base_prompt = (
                "Answer ONLY from the provided context. "
                'If not present, reply exactly: "I could not find this in the news dataset."\n\n'
                f"Context:\n{context}\n\nQuestion: {eval_q}\nAnswer:"
            )
            try:
                adv_ans = _call_llm(provider, adv_model, KEYS, base_prompt).strip()
            except Exception as e:
                adv_ans = f"⚠️ {provider} {adv_model} failed: {e}"

            try:
                lite_ans = _call_llm(provider, lite_model, KEYS, base_prompt).strip()
            except Exception as e:
                lite_ans = f"⚠️ {provider} {lite_model} failed: {e}"

            cols = st.columns(2)
            with cols[0]:
                st.subheader(f"{provider} — Advanced")
                st.markdown(adv_ans or "_(no text)_")
            with cols[1]:
                st.subheader(f"{provider} — Lite")
                st.markdown(lite_ans or "_(no text)_")

            if metas:
                with st.expander("📂 Sources used"):
                    for line in _pretty_sources(metas):
                        st.markdown(line)
