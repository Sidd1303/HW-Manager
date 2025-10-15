# HWs/HW7.py

import os
import csv
from collections import deque
from typing import List, Dict, Tuple, Optional, Union

import streamlit as st

# ---- Optional sqlite shim for Chroma on some Linux images ----
try:
    __import__("pysqlite3")  # noqa: F401
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# Vector DB
import chromadb
from chromadb.utils import embedding_functions

# LLM SDKs (no streaming, no temp/max_tokens unless required)
from openai import OpenAI as _OpenAI
from anthropic import Anthropic as _Anthropic
import google.generativeai as _genai

# Robust datetime parsing
from datetime import datetime, timezone


# =========================
# Globals / Constants
# =========================
CSV_PATH = "news_data/news_dataset.csv"
PERSIST_DIR = "vectorstore/news"
COLLECTION_NAME = "news_collection"

# Provider → (Advanced, Lite) model names (non-reasoning)
MODEL_MAP = {
    "OpenAI": {
        "Advanced": "gpt-4.1",
        "Lite": "gpt-5-chat-latest",
    },
    "Anthropic": {
        "Advanced": "claude-opus-4-1",
        "Lite": "claude-3-5-haiku-latest",
    },
    "Google Gemini": {
        "Advanced": "gemini-2.5-pro",
        "Lite": "gemini-2.5-flash-lite",
    },
}


# =========================
# CSV Loader & Preprocessing
# =========================

def load_news_csv(path: str) -> List[Dict]:
    """
    Load news CSV and normalize keys.
    We try multiple possible column names so your file is flexible.
    Returns list of dict rows with:
      - title (fallbacks tried)
      - text (fallbacks tried)
      - date/published (if present)
      - topic/section/category (if present)
      - url/source (if present)
      - full_text = title + " - " + text
    """
    if not os.path.exists(path):
        st.error(f"❌ CSV not found at {path}. Place your file at this path.")
        return []

    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}

            # Title candidates
            title = (
                r.get("title") or
                r.get("headline") or
                r.get("article_title") or
                r.get("Title") or
                ""
            )

            # Text/content candidates
            text = (
                r.get("text") or
                r.get("content") or
                r.get("body") or
                r.get("article") or
                r.get("story") or
                r.get("summary") or
                r.get("Text") or
                ""
            )

            # Date candidates
            date_val = r.get("date") or r.get("published") or r.get("publish_date") or r.get("Date")

            # Topic/Category candidates
            topic = r.get("topic") or r.get("category") or r.get("section")

            # Optional helpful info
            url = r.get("url") or r.get("link")
            source = r.get("source") or r.get("publisher")

            full_text = f"{title} - {text}".strip(" -")

            # Keep if we have some usable text
            if title or text:
                rows.append(
                    {
                        "title": title,
                        "text": text,
                        "date": date_val,
                        "topic": topic,
                        "url": url,
                        "source": source,
                        "full_text": full_text,
                    }
                )

    if not rows:
        st.error("❌ Could not load any rows from the CSV (no usable title/text).")
    return rows


# =========================
# Date Parsing & Scoring
# =========================

def _parse_datetime(val: Union[str, datetime, None]) -> Optional[datetime]:
    """Best-effort parsing of a date/time value into a timezone-aware UTC datetime."""
    if val is None:
        return None

    if isinstance(val, datetime):
        dt = val
    else:
        s = str(val).strip()
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                dt = None
        if dt is None:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def recency_score(dt_val: Union[str, datetime, None]) -> float:
    """
    Higher for more recent dates.
    Returns 0.3 baseline if no usable date (so recency contributes modestly).
    """
    dt = _parse_datetime(dt_val)
    if dt is None:
        return 0.3

    now = datetime.now(timezone.utc)
    if dt > now:
        dt = now

    days = max(1, (now - dt).days)
    return 1.0 / (1.0 + (days / 7.0))  # smooth weekly decay


# =========================
# Interestingness Ranking
# =========================

def interestingness(row: Dict) -> float:
    """
    Heuristic ranking for 'most interesting' news for a global law firm.
    Blend of:
      - recency (50%)
      - legal/financial cues in title (30%)
      - length/coverage proxy (20%)
    """
    title = (row.get("title") or "").lower()
    text = row.get("text") or ""
    s_rec = recency_score(row.get("date"))
    legal_cues = (
        "lawsuit", "sues", "sued", "settlement", "regulator", "compliance",
        "sanction", "fine", "acquires", "acquisition", "merger", "m&a",
        "investigation", "antitrust", "fraud", "subpoena", "deal", "IPO",
        "GDPR", "privacy", "patent", "trademark"
    )
    s_title = 1.0 if any(k in title for k in legal_cues) else 0.6
    s_len = min(len(text) / 1500.0, 1.0)

    score = 0.5 * s_rec + 0.3 * s_title + 0.2 * s_len
    return float(score)


# =========================
# Chroma Vector Store
# =========================

def build_or_load_collection(
    items: List[Dict],
    persist_dir: str,
    openai_key: str,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.api.models.Collection.Collection:
    """
    Create/load persistent Chroma collection from CSV rows.
    Each row becomes one document with metadata we can show as provenance.
    """
    os.makedirs(persist_dir, exist_ok=True)

    # write check (avoid "readonly database")
    try:
        testfile = os.path.join(persist_dir, ".writecheck")
        with open(testfile, "w") as w:
            w.write("ok")
        os.remove(testfile)
    except Exception as e:
        st.warning(f"⚠️ Persist dir not writable ({persist_dir}): {e}. Using in-memory DB.")
        client = chromadb.Client()
    else:
        client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small",
        ),
    )

    # Already embedded?
    try:
        existing = collection.count()
    except Exception:
        existing = len(collection.get().get("ids", []))

    if existing and existing > 0:
        return collection

    # Build from items
    docs, ids, metas = [], [], []
    for i, row in enumerate(items):
        text = row.get("full_text", "")
        if not text:
            continue
        docs.append(text)
        ids.append(f"row_{i}")
        metas.append(
            {
                "title": row.get("title"),
                "date": row.get("date"),
                "topic": row.get("topic"),
                "url": row.get("url"),
                "source": row.get("source"),
                "row_index": i,
            }
        )

    if not docs:
        st.error("❌ No usable text found in the CSV rows.")
        return collection

    collection.add(documents=docs, ids=ids, metadatas=metas)
    st.sidebar.success(f"✅ Embedded {len(docs)} documents from CSV into Chroma.")
    return collection


def retrieve_similar(
    collection: chromadb.api.models.Collection.Collection,
    query: str,
    k: int = 5,
) -> Tuple[List[Dict], List[int]]:
    """
    Query Chroma and return matched rows and row indices.
    """
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    out_rows, row_indices = [], []
    for m in metas:
        # Reconstruct a row-ish dict from metadata (best effort)
        r = {
            "title": m.get("title"),
            "date": m.get("date"),
            "topic": m.get("topic"),
            "url": m.get("url"),
            "source": m.get("source"),
        }
        out_rows.append(r)
        row_indices.append(m.get("row_index"))
    return out_rows, row_indices


# =========================
# LLM Runners (no streaming)
# =========================

def run_openai(model: str, api_key: str, prompt: str) -> str:
    client = _OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def run_anthropic(model: str, api_key: str, prompt: str) -> str:
    client = _Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return (resp.content[0].text or "").strip()
    except Exception:
        return ""


def run_gemini(model: str, api_key: str, prompt: str) -> str:
    _genai.configure(api_key=api_key)
    g = _genai.GenerativeModel(model)
    r = g.generate_content(prompt)
    return (getattr(r, "text", "") or "").strip()


# =========================
# Prompt Builders
# =========================

def serialize_memory(history: List[Dict], max_pairs: int = 5) -> str:
    dq = deque(history, maxlen=max_pairs * 2)
    lines = []
    for msg in dq:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_rag_prompt(memory_txt: str, context: str, question: str) -> str:
    return (
        "You are a News Assistant for a global law firm. Use ONLY the provided 'Context' "
        "(retrieved from the local news dataset). "
        "If the answer is not in the context, respond exactly with: "
        "\"I could not find this in the news dataset.\"\n\n"
        f"Conversation (last 5 Q/A):\n{memory_txt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# =========================
# Page UI
# =========================

def render():
    st.header("HW7 – News RAG Bot (ChromaDB, CSV-embedded)")

    # Secrets
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
    ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")

    if not OPENAI_KEY:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        return
    if not ANTHROPIC_KEY:
        st.warning("ANTHROPIC_API_KEY not set; Anthropic option will fail.")
    if not GEMINI_KEY:
        st.warning("GEMINI_API_KEY not set; Gemini option will fail.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic", "Google Gemini"])
    mode = st.sidebar.radio("Mode", ["Most Interesting", "Topic Search", "Chatbot", "Evaluation"], horizontal=False)
    tier_choice = st.sidebar.radio("Model Tier", ["Advanced", "Lite"], horizontal=True)
    force_rebuild = st.sidebar.checkbox("Force rebuild vector DB", value=False)

    adv_model = MODEL_MAP[provider]["Advanced"]
    lite_model = MODEL_MAP[provider]["Lite"]
    st.sidebar.caption(f"Advanced: `{adv_model}` · Lite: `{lite_model}`")

    # Load CSV
    rows = load_news_csv(CSV_PATH)
    if not rows:
        return

    # Build / Load Chroma
    if force_rebuild and os.path.exists(PERSIST_DIR):
        try:
            import shutil
            shutil.rmtree(PERSIST_DIR)
            st.sidebar.info("🧹 Deleted existing vector DB; rebuilding…")
        except Exception as e:
            st.sidebar.warning(f"Could not delete vector DB: {e}")

    collection = build_or_load_collection(items=rows, persist_dir=PERSIST_DIR, openai_key=OPENAI_KEY)
    if not collection:
        return

    # Debug panel (helpful for provenance & sanity)
    with st.expander("🔍 Debug / Provenance"):
        try:
            cnt = collection.count()
        except Exception:
            cnt = "?"
        st.write(f"Embeddings in collection: **{cnt}**")
        st.write({"csv_path": CSV_PATH, "persist_dir": PERSIST_DIR, "collection": COLLECTION_NAME})

    # Helper for model run
    def _call_llm(model_name: str, prompt: str) -> str:
        if provider == "OpenAI":
            return run_openai(model_name, OPENAI_KEY, prompt)
        elif provider == "Anthropic":
            return run_anthropic(model_name, ANTHROPIC_KEY, prompt)
        else:
            return run_gemini(model_name, GEMINI_KEY, prompt)

    # ======== Mode: Most Interesting ========
    if mode == "Most Interesting":
        st.subheader("📈 Most Interesting News (Ranked for a global law firm)")
        n_show = st.number_input("How many items?", min_value=5, max_value=50, value=10, step=1)

        # Rank all rows
        scored = []
        for r in rows:
            s = interestingness(r)
            scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n_show]

        for idx, (score, r) in enumerate(top, 1):
            title = r.get("title") or "(no title)"
            date_val = r.get("date") or "(no date)"
            source = r.get("source") or "(no source)"
            url = r.get("url")
            st.markdown(f"**{idx}. {title}**  \nScore: `{score:.3f}` · Date: `{date_val}` · Source: `{source}`")
            if url:
                st.write(url)
            st.divider()

        st.caption("Heuristic = 50% recency, 30% legal/finance cues in title, 20% article length proxy.")

    # ======== Mode: Topic Search ========
    elif mode == "Topic Search":
        st.subheader("🔎 Find news about a specific topic")
        topic_q = st.text_input("Your topic (e.g., 'JP Morgan opening smaller branches')")
        k = st.slider("Top-k results", min_value=3, max_value=10, value=5)
        if topic_q:
            hits, idxs = retrieve_similar(collection, topic_q, k=k)
            if not hits:
                st.write("I could not find this in the news dataset.")
            else:
                for i, h in enumerate(hits, 1):
                    title = h.get("title") or "(no title)"
                    date_val = h.get("date") or "(no date)"
                    source = h.get("source") or "(no source)"
                    url = h.get("url")
                    st.markdown(f"**{i}. {title}**  \nDate: `{date_val}` · Source: `{source}`")
                    if url:
                        st.write(url)
                    st.divider()

    # ======== Mode: Chatbot ========
    elif mode == "Chatbot":
        st.subheader("💬 News Chatbot (RAG)")
        model_name = adv_model if tier_choice == "Advanced" else lite_model

        chat_key = f"{provider}:{tier_choice}"
        if "hw7_chat_key" not in st.session_state or st.session_state.hw7_chat_key != chat_key:
            st.session_state.hw7_chat_key = chat_key
            st.session_state.hw7_chat = []
            st.sidebar.info(f"🔄 Chat reset for {chat_key}")

        # Show history
        for m in st.session_state.hw7_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask about the news in the dataset…")
        if user_q:
            st.session_state.hw7_chat.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Retrieve context from Chroma
            res = collection.query(query_texts=[user_q], n_results=5)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            context = "\n\n".join(d for d in docs if d and d.strip())
            sources = [
                {"title": m.get("title"), "date": m.get("date"), "source": m.get("source"), "url": m.get("url")}
                for m in metas
            ]

            memory_txt = serialize_memory(st.session_state.hw7_chat, max_pairs=5)
            prompt = build_rag_prompt(memory_txt, context, user_q)

            try:
                ans = _call_llm(model_name, prompt)
                if not ans:
                    ans = "I could not find this in the news dataset."
            except Exception as e:
                ans = f"⚠️ {provider} / {model_name} failed: {e}"

            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state.hw7_chat.append({"role": "assistant", "content": ans})

            # Sources this turn
            if sources:
                with st.expander("📂 Sources used (this turn)"):
                    for s in sources:
                        title = s.get("title") or "(no title)"
                        date_v = s.get("date") or "(no date)"
                        source = s.get("source") or "(no source)"
                        url = s.get("url")
                        st.write(f"- {title} · {date_v} · {source}")
                        if url:
                            st.write(f"  {url}")

            # Keep last 5 Q/A pairs (10 messages)
            st.session_state.hw7_chat = st.session_state.hw7_chat[-10:]

    # ======== Mode: Evaluation ========
    else:
        st.subheader("🧪 Evaluation: Advanced vs Lite (same provider)")
        QUESTIONS = [
            "Find the most interesting regulatory enforcement item.",
            "Show recent M&A or acquisition-related stories.",
            "Any lawsuits or investigations against a major bank?",
            "Summarize a significant data privacy or GDPR story.",
            "Are there notable fines or sanctions in the dataset?",
        ]
        st.write("This will query RAG context for each question and compare outputs.")
        if st.button("Run Evaluation"):
            for i, q in enumerate(QUESTIONS, 1):
                st.markdown(f"### Q{i}. {q}")

                # retrieve once, ask both models with same context
                res = collection.query(query_texts=[q], n_results=5)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                context = "\n\n".join(d for d in docs if d and d.strip())
                sources = [
                    {"title": m.get("title"), "date": m.get("date"), "source": m.get("source"), "url": m.get("url")}
                    for m in metas
                ]
                prompt = build_rag_prompt("", context, q)

                try:
                    adv_out = _call_llm(MODEL_MAP[provider]["Advanced"], prompt) or "I could not find this in the news dataset."
                except Exception as e:
                    adv_out = f"⚠️ Advanced failed: {e}"

                try:
                    lite_out = _call_llm(MODEL_MAP[provider]["Lite"], prompt) or "I could not find this in the news dataset."
                except Exception as e:
                    lite_out = f"⚠️ Lite failed: {e}"

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Advanced – {MODEL_MAP[provider]['Advanced']}**")
                    st.write(adv_out)
                with c2:
                    st.markdown(f"**Lite – {MODEL_MAP[provider]['Lite']}**")
                    st.write(lite_out)

                with st.expander("📂 Sources"):
                    for s in sources:
                        title = s.get("title") or "(no title)"
                        date_v = s.get("date") or "(no date)"
                        source = s.get("source") or "(no source)"
                        url = s.get("url")
                        st.write(f"- {title} · {date_v} · {source}")
                        if url:
                            st.write(f"  {url}")
                st.divider()
